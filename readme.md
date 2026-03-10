# Transcriber

GPU‑accelerated speech transcription powered by Whisper/WhisperX with optional alignment and diarization. Works on single audio files, directories, or multi‑track ZIPs.

## Features
- WhisperX backend (ASR + alignment; diarization when available)
- Faster‑Whisper backend (fast ASR only)
- Parakeet backend (Apple Silicon local ASR with timestamps; no diarization)
- Config‑driven runs with CLI override, clean progress bars, auto‑batching
- Outputs: TXT (Speaker HH:MM:SS Text), SRT, JSONL; optional diarization JSON

## Requirements
- Python 3.11, FFmpeg
- NVIDIA GPU recommended (CUDA 12.8 wheels configured). CPU works with Faster‑Whisper.

## Install & Run
- Setup: `make setup`
- Config‑driven: `cp config/transcriber.example.yaml config/transcriber.yaml && uv run transcribe`
- Direct CLI: `uv run transcribe audio/Session_32.zip --model large-v3 --speaker-mapping config/speaker_mapping.example.yaml`
- Apple Silicon local run: `uv run transcribe audio/sample.wav --backend parakeet --model parakeet`
- Docker (GPU): `make docker-build` then `HF_TOKEN=... make docker-run INPUT=audio/Session_32.zip MODEL=large-v3`

## Device Selection
- `--device cuda` forces GPU; if CUDA/cuDNN fails, the run errors (no CPU fallback).
- `--device auto` picks GPU when available, otherwise CPU (fallback allowed).
- `--device cpu` forces CPU. See CPU workarounds: `--vad-on-cpu`, `--pyannote-on-cpu`.

## cuDNN/cuBLAS Split‑Libs (CUDA)
If you hit: “Unable to load any of {libcudnn_cnn.so.9…}”, the loader can’t see split cuDNN libs from pip (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`). The CLI auto‑adds these paths and preloads cuDNN. For custom entry points, export (replace `3.X`):

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.X/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.X/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

Verify: `uv run python -c "import torch;print('cuda',torch.cuda.is_available(),'cudnn',getattr(torch.backends.cudnn,'is_available',lambda:False)())"`

## CLI (Synopsis)
- `uv run transcribe INPUT [--backend whisperx|faster|parakeet] [options]`
- Common options: `--model large-v3` `--compute-type float16|int8_float16` `--device auto|cuda|cpu` `--batch-size N` `--auto-batch/--no-auto-batch` `--speaker-mapping PATH` `--min-speakers/--max-speakers` `--output-dir outputs` `--cache-root PATH|--cache-mode home|repo|env` `--local-files-only` `--log-level INFO|WARNING` `--config PATH`

## Configuration
Auto‑discovery: `--config`, `TRANSCRIBER_CONFIG`, `./transcriber.yaml|.json`, `./config/…`, `~/.config/transcriber/config.yaml|.json`. See `config/transcriber.example.yaml`.
Local runtime configs copied from `config/*.example.yaml` are intentionally gitignored.

## Watch Mode
- Keep two configs: `config/transcriber.yaml` for interactive CLI runs, and `config/transcriber.watch.yaml` (copy from `config/transcriber.watch.example.yaml`) for the background watcher.
- Launch the watcher interactively with `uv run transcribe --watch --config config/transcriber.watch.yaml`. Use `watch_input` (config) or `--watch-input` (CLI) when the directory you monitor differs from the one-shot `input`.
- Config sticks after startup; restart the process (Ctrl+C and rerun). If you installed it with `scripts/install_watch_service.sh`, restart via `systemctl --user restart transcriber-watch.service`.
- The bundled `scripts/watch_service_runner.sh` now passes the watch config automatically; override with `WATCH_CONFIG=/path/to/custom.yaml`.
- Logs go to stdout by default. Manually: pipe to a file and tail it (`uv run … --watch | tee -a watch.log`, then `tail -f watch.log`). For the systemd user service, follow logs with `journalctl --user -u transcriber-watch.service -f -o cat`.

## Speaker Mapping
Map track IDs or diarization labels to display names. Example (`config/speaker_mapping.example.yaml`):

autbot_80hd55561_0: "B. Ver"
bfschmity_0: "Kaladen"
travisaurus6985_0: "Cyrus Schwert"

Matching is case-insensitive; numeric prefixes like `2-foo_0.ogg` are ignored; for `x-foo_0`, `foo_0` is preferred.

## Apple Silicon Parakeet
- Use `--backend parakeet` to run the MLX Parakeet backend on Apple Silicon Macs.
- Parakeet supports timestamps, TXT/SRT/JSONL output, and single-speaker label overrides.
- Parakeet does not currently provide diarization, so `--min-speakers` and `--max-speakers` are ignored on that backend.

## Speaker Bank
- Enable the speaker embedding bank to tag recurring players across sessions: `--speaker-bank` (default when configured) or `speaker_bank.enabled` in the config.
- Profiles live under the cache root (`<cache>/speaker_bank/<profile>/` by default). Override with `--speaker-bank-path` or `speaker_bank.path`.
- Store bank data separately from Hugging Face caches by combining `--speaker-bank-root` with `--hf-cache-root`; useful when models are shared globally but embeddings stay project-local.
- Train incrementally from single-speaker stems with `--speaker-bank-train-stems`; updates are persisted automatically and a PCA snapshot is stored alongside debug summaries.
- Match mixed files against the bank during transcription; unmatched segments are labelled `unknown` and retain their raw diarization label in `speaker_raw` for inspection.
- Run offline training on processed audio with `--speaker-bank-train-only PATH` to ingest new voices without triggering transcription.

## Inputs & Outputs
- Inputs: single audio (.wav, .mp3, .ogg, .flac, .m4a), directory (recursive), or ZIP (multi-track safe extract)
- Outputs in `outputs/<INPUT_BASE>/`: `.txt`, `.srt`, `.jsonl`, optional `.diarization.json`
- Generated transcripts and evaluation artifacts are local-only and should live under `.outputs/` or `transcripts/`, both of which are gitignored.

## Caching
Default root `~/hf_cache`: HF hub (`$HF_HOME/hub`), models (`$HF_HOME/models`), align (`$HF_HOME/align`). Configure with `--cache-root` or `--cache-mode home|repo|env`. Use `--hf-cache-root` to pin model caches independently, and `--speaker-bank-root` to relocate bank profiles (e.g., keep embeddings inside the repo while sharing models globally). Offline: `--local-files-only` (set `HF_TOKEN` for gated models).

## Troubleshooting
- cuDNN error (`libcudnn_cnn.so.9…`): use the CLI (auto‑fix) or export split‑lib paths as above.
- No diarization pipeline: ASR+alignment still run; set `HF_TOKEN` or use `--pyannote-on-cpu`.
- Need strict GPU: use `--device cuda`. For fallback behavior, use `--device auto`.

## Development
- Lint: `make lint`  |  Tests: `make test`  |  Local run: `make run INPUT=audio/sample.wav MODEL=medium`
