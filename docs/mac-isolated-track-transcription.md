# Qwen transcription of isolated speaker tracks on Mac

Use this path when each file contains one known speaker, including multitrack ZIP
archives. It runs Qwen3-ASR-1.7B on Apple Silicon through MLX, supplies a canonical
vocabulary during recognition, and aligns words with Qwen's forced aligner. It
uses file identity for attribution and preserves concurrent turns between tracks.
For a complete mixed recording use [the mixed recording workflow](mac-single-file-transcription.md).

## Install

Keep the MLX runtime separate from the main application environment. From the
repository root, with `uv` and `ffmpeg` installed:

```sh
uv sync
uv venv --python 3.12 .venv-qwen
uv pip install --python .venv-qwen/bin/python -r requirements/qwen-mac.lock.txt
```

The backend pins Qwen ASR and aligner model revisions. The first online run can
download them to the selected Hugging Face cache; later runs can use
`--local-files-only`. All recognition happens on the Mac. Metal access is required;
sandboxed or headless environments that cannot access the GPU cannot run this
backend. There is no cloud transcription fallback.

## Vocabulary

`--vocabulary` accepts the D&D summary repository's `campaign.json`, a JSON list of
strings, or a custom bank:

```json
{"terms": [{"canonical": "Example Character", "spoken_forms": ["Example"]}]}
```

The campaign reader imports canonical player characters from all configured parties,
characters, items, locations, confirmed terms, and table people's names. It includes
an alias only when it is an exact word span of the canonical name. Other aliases
and speaker handles may be transcription errors, so they are not automatically
boosted. Add confirmed nicknames through `spoken_forms` in a custom bank. For a
smaller session-specific vocabulary, supply an explicit JSON list.

The source is read on every run. Both the source-file hash and the resolved hint
list are included in the cache provenance; a changed bank receives fresh decodes.
The source campaign file is never edited. Vocabulary hints do not force a word
into the transcript, nor do they rewrite output afterward. A spoken short form is
retained instead of automatically expanded to the character's full name.

## Run

An individual known-speaker recording:

```sh
uv run transcribe /path/to/speaker.wav \
  --backend qwen --model qwen --device auto --cache-mode repo \
  --asr-python .venv-qwen/bin/python \
  --vocabulary /path/to/campaign.json \
  --single-file-speaker "Player name" --output-dir outputs/qwen
```

A ZIP of isolated tracks:

```sh
uv run transcribe /path/to/Session_32.zip \
  --backend qwen --model qwen --device auto --cache-mode repo \
  --vocabulary /path/to/campaign.json \
  --speaker-mapping /path/to/speaker_mapping.yaml \
  --output-dir outputs/qwen
```

ZIP filenames provide fallback labels when no mapping is supplied. A directory of
tracks requires a mapping covering every file. This backend rejects diarization
speaker-count options and enabled speaker-bank matching. Track start times must
already share the same session origin; the backend does not infer missing offsets.

Copy [the example config](../config/transcriber.qwen.example.yaml) to
`config/transcriber.qwen.local.yaml` and adjust the source/interpreter paths. The
example enables offline mode, so use it after downloading the pinned models, or
set `local_files_only: false` for the first run. The local config is ignored by Git.
It explicitly disables downstream postprocessing so the run stays local.

```sh
uv run transcribe /path/to/Session_32.zip \
  --config config/transcriber.qwen.local.yaml \
  --speaker-mapping /path/to/speaker_mapping.yaml
```

On Apple Silicon, `backend: auto` now selects Qwen for multitrack ZIPs, mapped
directories, and recordings with `--single-file-speaker`. An explicit `qwen` or
this config also selects it. Unmapped directories and mixed recordings retain their
other backends. Qwen processes chunks sequentially; the ordinary `--batch-size`
setting does not change its decoding.

## NVIDIA GPU (CUDA)

The same chunking, vocabulary, alignment and export run on Linux with an NVIDIA GPU
through the official `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`
checkpoints (bfloat16, pinned revisions) and the `qwen-asr` package. The decoder
prompt is built exactly as mlx-audio builds it, so both runtimes receive the same
request. CUDA is never selected automatically: pass `--backend qwen --device cuda`.
The completion record names the runtime (`asr_runtime: cuda`), model, revision and
aligner, and the chunk cache provenance records torch, CUDA and `qwen-asr` versions.

```sh
uv sync
uv venv --python 3.12 .venv-qwen-cuda
uv pip sync --python .venv-qwen-cuda/bin/python \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match requirements/qwen-cuda.lock.txt
```

Copy [the CUDA example config](../config/transcriber.qwen.cuda.example.yaml) to
`config/transcriber.qwen.cuda.local.yaml`, set the absolute cache directory, and
place the Silero model at `<cache>/vad/silero-vad-6.2.2.onnx` (or run once online).
The model and aligner need about 7 GB of GPU memory.

The two runtimes are different weights and kernels (the MLX aligner is 8-bit), so
their transcripts are close but not byte-identical; the decoded 16 kHz audio can
also differ in float rounding between ffmpeg versions, which the cache's
`audio_sha256` records. Treat a CUDA transcript as its own method in provenance
gates rather than as the MLX model id.

## Results, resume, and quality

Exports use the normal per-input output directory and include TXT, SRT, and JSONL
with word timestamps and track speaker labels. Private raw decodes and aligned
chunk records live under `outputs/qwen/.qwen/` by default. Repeating the same run
reuses completed chunks with matching provenance. Changes to audio, vocabulary,
model snapshot, MLX version, or decoding settings produce a separate cache.

Silero VAD 6.2.2 runs locally before recognition. It uses a 0.35 speech threshold,
0.20 release threshold, a half-second silence boundary and 250 ms of padding on
both sides. Responses as short as 64 ms can be retained. The pinned ONNX model is
downloaded and checksum-verified on the first online run. Detected regions retain
their original track offsets; audio files are not destructively shortened or
concatenated. Long gaps are excluded from recognition, preserving synchronization
between tracks. These conservative settings still need listening checks for
quiet speech, noise and vocalizations. See the
[Silero interface](https://github.com/snakers4/silero-vad/blob/v6.2.2/src/silero_vad/utils_vad.py).

Each speech region is split into cores of at most 30 seconds, decoded with up to
three seconds of surrounding real audio within that region.
Words belong to the core containing their aligned midpoint, avoiding duplicate
publication of context words. Each chunk gets a fresh token budget. Truncated
decodes are retried once without hints. If the retry still exhausts its budget,
or alignment changes lexical tokens or produces invalid timestamps, the output
contains `[Transcription needs review]` over that original-time interval, with
`review_required` and a reason in JSONL. Raw recognition is retained in the cache;
it is not published as reliable dialogue. End-of-clip rounding within the aligner's
80 ms timestamp grid is clamped to the audio boundary and counted. Zero-duration
words are counted in the chunk audit. Noise-only regions and chunk joins still
need listening review.

The two-session pilot exposed a quiet-input failure where Qwen copied the complete
name bank into its output. When the output contains the complete bank of at least four
terms in order (ignoring punctuation and case), the backend retains a `hint_echo.json`
record and retries that chunk without hints. This narrowly handles the observed
failure; it does not detect partial hint copying or all hallucinations. Chunk
records include retry flags and processing time. Worker responses include model
setup, per-track, and total timing; resumed runs measure cache reuse, so benchmark
fresh cache identities when reporting inference speed.

## Listen and correct

The current two-session listening workspace is in `outputs/listening-review/`.
Serve it with byte-range support so playback can seek directly to any phrase:

```sh
uv run python -m transcriber.review_server outputs/listening-review --port 8769
```

Open the printed localhost URL. Choose a recording and speaker, then play. Words
highlight as they are spoken. Click a phrase to replay it, or pause with
**Correct current phrase**. Enter what you hear and optionally classify the issue
or add notes. **Next flagged passage** moves through detected recognition or
alignment failures across all speakers; use **Replay selection** to listen. This
queue does not identify every transcription error. **Missing speech here** records
an interval even when there is no
transcribed text. Turn **Skip detected silence** off to audit potentially missed
speech. Original audio remains available in the player.

Edits save in that browser, origin and transcript version. **Export feedback**
downloads JSON containing audio hashes, original timestamps/text, corrections,
issue categories and notes. Keep/export this file before clearing browser data or
changing the local server port. **Import feedback** restores a matching review.
No audio or feedback is uploaded. Failed recognition clips are explicitly marked
for listening, with no invented word timings.

To build another review from saved worker request/response JSON:

```sh
uv run python -m transcriber.review_player \
  --session "Session label" /path/to/request.json /path/to/response.json \
  --mapping /path/to/speaker-handles.json --output outputs/listening-review
```

Repeat `--session` for multiple recordings. The source audio paths in the request
must still exist. The builder writes local AAC copies for browser playback, plus
a manifest with the original audio hashes and word timing. `--feedback exported.json`
can carry corrections to new transcripts only when the original audio hashes
match; it preserves the old text and time intervals instead of guessing new word
or segment indexes. Such carried feedback retains its source dataset identifier.

Use a new output directory for comparisons or archive revisions. The backend is a
working candidate, not a demonstrated WER improvement. The human-reference
[evaluation plan](specs/isolated_track_asr_20261012_decision_report.md) remains the
gate for replacing archive transcripts. Existing review excerpts can verify local
execution and vocabulary effects; they are not an independent accuracy benchmark.
