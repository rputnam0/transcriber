# Named transcription of a mixed recording on Apple Silicon

This workflow transcribes a complete mono/stereo mix, identifies enrolled speakers,
and exports timestamped TXT, SRT, JSON and a searchable HTML reader. It uses
Qwen3-ASR-1.7B, Qwen forced alignment, and a privately fine-tuned MOSS checkpoint.
During overlapping exchanges it retains MOSS's separate turns and exposes Qwen's
wording as an alternate. It does not infer names from conversation content.

This is a separate workflow from `uv run transcribe`. The normal CLI remains useful
for isolated speaker recordings and multitrack archives. Scripts named `sweep`,
`train`, `evaluate` and `score` preserve experiments; they are not all release defaults.

## Install

Use an Apple Silicon Mac, Python 3.12, `uv`, and `ffmpeg`/`ffprobe` on PATH. The tested
machine had 48 GB unified memory. Keep large audio and checkpoints outside Git.

```sh
uv sync
uv venv --python 3.12 "$HOME/.cache/transcriber/asr-env"
uv pip install --python "$HOME/.cache/transcriber/asr-env/bin/python" \
  -r requirements/mac-asr.lock.txt
```

The application environment runs enrollment/export; the independent MLX environment
runs ASR/diarization inference. Pyannote enrollment needs access to its gated
`pyannote/speaker-diarization-community-1` weights. Authenticate with Hugging Face
and accept the model's access conditions if not already cached; never commit tokens.

## Restore private voices

The public repository contains code, tests and dependency versions, **not** voice
weights, embeddings, session rosters, recordings, or human-review snapshots. Download
the private backup's `backup_manifest.json`, `private-assets.zip`, and **every**
`model-partNNN.zip` into one folder. The manifest records exact SHA-256
values; the restore tool reads the parts directly from their ZIP files.

```sh
uv run python scripts/restore_mac_voice_assets.py \
  --downloads /path/to/downloaded-backup \
  --output "$HOME/.cache/transcriber/private-voices"
```

The restore command verifies every part and the assembled checkpoint before making
it available. It also restores `speaker_model.npz`, `turn_refinement.json`, and
`rosters.json`. The roster maps session numbers to allowed speaker handles and handles
to display names. Add a future session's known attendees before running it. Preserve
the checkpoint's stable speaker ordering; roster restriction must not renumber IDs.
New voices require new enrollment/training, not merely another display name.

## Run or resume a complete recording

```sh
uv run python scripts/transcribe_mac_recording.py \
  --audio /path/to/complete-recording.m4a --session 70 \
  --assets "$HOME/.cache/transcriber/private-voices" \
  --asr-python "$HOME/.cache/transcriber/asr-env/bin/python" \
  --work "$HOME/.cache/transcriber/session70-run1" \
  --output transcripts/session70-run1
```

Use a **new work/output directory** for different audio, models, or rosters. Repeating
the same command resumes verified caches. `--dry-run` prints subprocess arguments
without running inference. `--models /path/to/models.json` reuses an existing resolved
model manifest; otherwise the pinned public ASR/alignment revisions download to the
Hugging Face cache. `--batch-size 1` lowers peak inference memory.

The pipeline decodes all source audio into hashed 30-second chunks. Qwen also receives
three seconds of real context on each side; aligned words are assigned to one core
interval. Decode failures are retried with provenance retained. Local utterance-level
voice evidence can correct an individual speaker name without changing every turn
sharing an ID. Low-confidence names stay explicitly unconfirmed.

```sh
uv run python scripts/serve_transcript_reader.py \
  --root transcripts/session70-run1 --port 8768
```

Open `http://127.0.0.1:8768/`. Click timestamps to listen, or search/filter the text.
The listening AAC copy is separate from the original inference input. The server is
local-only. Drive does not host this audio player; upload TXT/SRT or standalone text
HTML for reading there. Keep the `.turns.json` and `release_audit.json` privately for
future re-export and audit.

## Quality and validation

Speaker-label training is not proof of better word recognition. Re-running the
diarizer can also change previously correct names. In one frozen review comparison,
the latest run fixed four labels and regressed five among 254 comparable passages;
35 others were ambiguous/unmatched. This user-selected sample is not a whole-session
accuracy estimate. Comparisons of ASR against automatic stem transcripts are proxy
measurements, not human word-error-rate ground truth. Overlap and fantasy names
remain difficult. Preserve prior transcripts and explicit human annotations.

The release audit verifies continuous source coverage, audio hashes, a consistent
checkpoint fingerprint, allowed speaker names, valid timestamp bounds, and accounting
for all ASR words. Some forced-alignment words have zero duration and are review flagged.
The audit establishes processing completeness, not correctness of every word.

Run `make lint` and `make test` before publishing code changes. Real-model inference is
an additional check; unit tests do not establish recognition quality.

Primary model references: [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
[MOSS Transcribe Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize),
and [MLX Audio](https://github.com/Blaizzy/mlx-audio).
