# PR Plan: Chunked Speaker-Bank Training From Segments

## Summary

Observed issue: Matching group-audio diarization labels to the existing speaker bank fails (all unknown) despite correct wiring. Analysis shows a large domain gap between bank embeddings (single near-field Discord samples) and group-audio embeddings (far-field room mic). The bank currently contains one embedding per speaker, which cannot represent channel and persona variation.

Goal: Rework the speaker-bank training pipeline to ingest many short, labeled segments per speaker (10–30s), generating a diverse embedding set per speaker. Cluster these embeddings to form multiple centroids (per persona/channel). Then use these clusters for matching against group-audio diarization.

## Scope

- Add a training pathway that consumes pre-computed segments (from our JSON/JSONL outputs) and/or VAD-chunked single-speaker stems to produce many embeddings per speaker.
- Persist these embeddings in the existing `SpeakerBank`, letting DBSCAN form sub-clusters per speaker.
- Provide CLI controls and config to drive this path and evaluate with PCA plots.

Non-goals: Changing the ASR/align models; replacing diarization; altering default matching logic beyond config additions (we can consider margin-based fallback in a follow-up).

## Design Overview

### Inputs

- Group session outputs: our `.jsonl` includes per-segment records with `start`, `end`, and `speaker`. We will use these to derive per-speaker segments for training.
- Single-speaker stems (Discord bot recordings): optionally chunk via VAD into 10–30s segments per speaker when segment JSON is unavailable.

Segment sourcing priority (auto):
- For a given audio basename `<base>`, resolve segments in this order:
  1) `.outputs/<base>/<base>.jsonl` (preferred, produced by ASR+align)
  2) `.outputs/<base>/<base>.json` (fallback diarization JSON)
  3) `--segments-json` override (if a file path is provided, it wins; if a directory is provided, match by basename)
- Basename rules: strip standard audio suffixes; case-insensitive; error on ambiguous multiples with guidance.

### Embedding Extraction for Segments

- Add a helper to produce speaker embeddings for arbitrary time ranges of an audio file.
  - Implement in `whisperx_backend.py` as `extract_embeddings_for_segments(audio_path, segments, device, hf_token, quiet)`.
  - Strategy:
    1) Try to access the whisperx/pyannote pipeline’s embedder (wespeaker) and run it on cropped waveforms for each segment.
    2) If direct embedder access is unavailable across versions, fall back to computing an embedding per aggregated diar label as last resort (logged and flagged, since it reduces training granularity).
  - Segments are tuples of `(start_sec, end_sec, speaker_label)`.
  - Normalize all resulting vectors to unit length (cosine space), consistent with current normalization.
  - Audio alignment tolerance: apply small padding around segments (default `pre_pad=0.15s`, `post_pad=0.15s`), clamped to file bounds, to absorb minor drift between alignment and embedding windows.

### Training Data Builders

1) Train-from-segments (from JSON/JSONL)
   - Parse our session `.jsonl` (or `.json`) to collect segments grouped by `speaker`.
   - Optional filters:
     - `min_segment_dur` (default 6s) and `max_segment_dur` (default 30s).
     - Downsample long segments by sliding windows (e.g., 15s window, 7.5s stride).
     - Limit per-speaker segment count (e.g., cap at 300 embeddings to bound runtime).
   - For each segment window, extract an embedding and add it to the bank with metadata:
     `{file, start, end, speaker, mode: 'train_from_segments', session: <name>}`.
   - Concurrency & batching: process segments per file with configurable `embed_batch_size` and CPU `embed_workers` for cropping; batch GPU forwards when supported, otherwise pipeline CPU I/O concurrently and queue GPU work.

2) Train-from-stems (VAD chunking)
   - For single-speaker sources (Discord bot WAV/OGG), apply VAD and segment into 10–30s windows.
   - Label is provided by `speaker_mapping` (existing mechanism) or CLI `--single-file-speaker`.
   - Extract embeddings per window and append to the bank with metadata:
     `{file, chunk_idx, start, end, mode: 'train_vad_chunks'}`.

### Speaker Bank

- Reuse existing `SpeakerBank` persistence and clustering.
- Expect many embeddings per speaker; DBSCAN will produce multiple clusters (personas / channels).
- PCA rendering already supported; expose artifact path in summary.

### Matching (unchanged default)

- Continue matching diar label -> best bank cluster by cosine with threshold and variance radius.
- Optional enhancement (follow-up): margin-based acceptance (top1 - top2 >= margin) and/or per-segment matching mode (compute embeddings per diarized segment and match individually) behind config flags.

## CLI & Config Changes

### Config (YAML)

Add under `speaker_bank:`

```
speaker_bank:
  # existing keys ...
  train:
    from_segments: true            # consume precomputed segments for training
    segment_source: auto           # auto | session_jsonl | diarization_json
    min_segment_dur: 6.0           # seconds
    max_segment_dur: 30.0          # seconds
    window_size: 15.0              # seconds (for long segments)
    window_stride: 7.5             # seconds
    max_embeddings_per_speaker: 300
    vad_chunk_stems: false         # when training from stems without segments
    pre_pad: 0.15                  # seconds before segment start
    post_pad: 0.15                 # seconds after segment end
    embed_workers: 4               # CPU workers for cropping/IO
    embed_batch_size: 16           # segments per GPU forward (if supported)
```

### CLI arguments

- `--speaker-bank-train-only PATH` (exists): extend to accept directories containing audio and optionally matching segment JSON/JSONL files.
- New:
  - `--segments-json PATH` (file or directory): session `.jsonl` or `.json` describing segments. If a directory, auto-pair files by basename with audio.
  - `--train-segment-source {auto,session_jsonl,diarization_json}`
  - `--min-segment-dur SECONDS`, `--max-segment-dur SECONDS`
  - `--window-size SECONDS`, `--window-stride SECONDS`
  - `--max-embeddings-per-speaker N`
  - `--speaker-bank-train-chunk-stems` to turn on VAD chunking of stems when segments JSON is absent.
  - `--pre-pad SECONDS`, `--post-pad SECONDS`
  - `--embed-workers N`, `--embed-batch-size N`

## Control Flow

### A) Train From Segments

1. Resolve bank root/profile (existing).
2. Discover training inputs:
   - If `--segments-json` given, load that (or walk a directory and pair with audio basenames).
   - Else, when `train.from_segments` is true, try to infer segments from `.outputs/<base>/<base>.jsonl`.
3. Parse segments and group by `speaker`; filter by duration; window long segments.
4. Call `extract_embeddings_for_segments` with `(start, end, speaker)` for each audio file, using `pre_pad`/`post_pad`, `embed_batch_size`, and `embed_workers`.
5. Append normalized embeddings to the bank with rich metadata (file, start, end, label, mode).
6. Save bank, which triggers clustering; write training summary and PCA artifact path.

### B) Train From Stems (VAD Chunks)

1. Resolve speaker label via `speaker_mapping` or `--single-file-speaker`.
2. Apply VAD to split into chunks (10–30s, merge small gaps).
3. Extract embeddings per chunk; add to bank.
4. Save and render PCA.

## Implementation Plan

1) Backend: segment embedding helper
   - File: `src/transcriber/whisperx_backend.py`
   - Add `extract_embeddings_for_segments(audio_path, segments, hf_token, device, quiet)`.
     - Internals: acquire diarization pipeline (current `_get_diar_pipeline`), try to access/embed on waveform crops. Provide version-robust try/except shims; log how many succeeded; return `{(start,end,speaker)->vec}`.

2) Segment parsing utilities
   - File: `src/transcriber/audio.py` or new `src/transcriber/segments.py`
   - Add `load_segments_from_jsonl(path)` and `pair_audio_with_segments(input_dir)` helpers.
   - Normalize records into `List[Segment(start,end,speaker,file)]`.

3) Extend training path
   - File: `src/transcriber/cli.py`
   - Extend `run_speaker_bank_training`:
     - Accept `segments_json`, `train_segment_source`, `min/max_segment_dur`, `window_size`, `window_stride`, `vad_chunk_stems`, `max_embeddings_per_speaker`.
     - Branch: if segments available, run A); else if stems and `vad_chunk_stems`, run B); else fallback to current single-embedding behavior (and warn).
     - Collect and log counts per speaker; write `default.training_summary.json` with `embeddings_added`, clustering summary, and PCA artifact path.

4) Config plumbing
   - Add fields to `_apply_config_defaults`, `_resolve_speaker_bank_settings`, and CLI args mapping.

5) Documentation
   - Update `readme.md` with examples:
     - Train from JSONL segments.
     - Train from stems with VAD chunking.
   - Describe PCA inspection.

## Testing Plan

- Unit tests (pytest)
  - `tests/speaker_bank/test_segments_parsing.py`: parse JSONL into segments.
  - `tests/speaker_bank/test_segment_embeddings.py`: mock embedder; verify per-segment embedding extraction and bank extension.
  - `tests/speaker_bank/test_chunked_training.py`: simulate 2 speakers with synthetic vectors; verify multiple clusters are created and centroids are reasonable.
  - `tests/speaker_bank/test_alignment_tolerance.py`: verify padding/clamping.
  - `tests/speaker_bank/test_concurrency_flags.py`: ensure batching/workers flow to backend helper (mocked).

- Integration smoke test
  - Command: `uv run transcribe --speaker-bank-train-only audio/Session_32.zip --segments-json .outputs/Session_32/Session_32.jsonl --no-quiet --log-level INFO`
  - Verify summary: `embeddings_added >> 5` and clusters per speaker ≥ 2 when personas exist.
  - Verify PCA image at `<profile>/pca.png`.

## Validation & Acceptance

- After training, run: `uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO`.
- Expect: `.speaker_bank.json` shows `attempted == # diar labels` and non-zero `matched` with reasonable scores.
- Visual check: PCA shows distinct clusters per speaker; clusters span both near-field stems and room-mic segments.
 - Artifact paths: training summary `default.training_summary.json` and PCA `pca.png` are written under `<speaker_bank_root>/<profile>/`.

## Risks & Mitigations

- WhisperX/pyannote API variance: guard embedder access with multiple code paths and explicit logs; if only label-level embeddings are possible, warn and skip per-segment training for that file.
- Performance: extracting hundreds of embeddings can be slow; batch by file and show progress; cap per-speaker embeddings.
- Gated models: honor `HF_TOKEN`; respect `local_files_only`.
 - HF token guidance: document `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env vars and surface clear warnings when gated models are required.

## Rollout Plan

1. Land helpers + CLI flags behind config.
2. Run training on a single ZIP; verify PCA and match rates.
3. Iterate thresholds and windowing; then train full bank.
4. Optionally add margin-based matching in a follow-up PR if needed.

## Example Commands

- Train from session segments JSONL:
```
uv run transcribe \
  --speaker-bank-train-only audio/Session_32.zip \
  --segments-json .outputs/Session_32/Session_32.jsonl \
  --no-quiet --log-level INFO
```

- Train from stems with VAD chunking:
```
uv run transcribe \
  --speaker-bank-train-only /path/to/stems \
  --speaker-bank-train-chunk-stems \
  --speaker-mapping config/speaker_mapping.yaml \
  --no-quiet --log-level INFO
```

- Transcribe and apply bank:
```
uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO
```

## Follow-ups / TODOs (after chunked training lands)

- Score calibration and normalization (optional, incremental):
  - AS-Norm/cohort normalization: compute score normalization using an impostor cohort (e.g., other-speaker centroids) to reduce channel effects. Expose as `speaker_bank.scoring.as_norm: {enabled, cohort_size}`.
  - CORAL whitening: estimate covariance from bank embeddings and apply a whitening transform to session embeddings before cosine scoring. Expose a `speaker_bank.scoring.whiten: true` toggle and cache the transform with the profile.
  - Margin-based acceptance: if `top1 < threshold` but `(top1 - top2) >= margin`, accept the match. Config: `speaker_bank.scoring.margin: 0.10`.

- Cluster labeling and curation:
  - Add a small tool to list clusters per speaker with counts/variance and (optionally) sample audio snippets per cluster for quick listening and manual labeling of personas.
  - Persist human-friendly cluster aliases in `bank.json` (e.g., `cluster_alias: "DM (gravelly)"`).

- Quality controls:
  - Filter training segments by SNR/energy and discard low-speech segments (e.g., < 30% voiced frames) to avoid noisy embeddings.
  - Cap per-session contribution to avoid over-representing a single recording.

- Evaluation harness:
  - Add a small script to compute match rate on a labeled validation set (JSONL with ground-truth speakers) and report precision/recall vs. threshold and margin.
  - Track metrics in the training summary and optionally serialize ROC-like curves.

- Performance & ops:
  - Batch embedding extraction across segments per file where possible; consider CPU diarization (`pyannote_on_cpu: true`) for training on machines without GPUs.
  - Add resume support to skip already-embedded segments based on metadata hashes.

- Documentation:
  - Expand README with a “Speaker Bank Best Practices” section covering domain mismatch, chunk sizing, and recommended thresholds by device/model.
