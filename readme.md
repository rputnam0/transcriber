# Transcriber

GPU-accelerated speech transcription powered by faster-whisper plus direct pyannote diarization. Works on single audio files, directories, or multi-track ZIPs.

## Features
- Faster-whisper backend with native word timestamps
- Direct pyannote `speaker-diarization-community-1` diarization with exclusive speaker turns
- Speaker-bank labeling for recurring speakers in multi-speaker sessions
- Parakeet backend (Apple Silicon local ASR with timestamps; no diarization)
- Config‑driven runs with CLI override, clean progress bars, auto‑batching
- Outputs: TXT (Speaker HH:MM:SS Text), SRT, JSONL, `.diarization.json`, `.exclusive_diarization.json`

## Requirements
- Python 3.11, FFmpeg
- `uv sync` is system-aware: Linux installs the default `faster` GPU stack plus the Parakeet CPU runtime, while Apple Silicon gets the CPU-safe stack plus `parakeet-mlx`.
- `HF_TOKEN` is required for gated pyannote models such as `pyannote/speaker-diarization-community-1`.

## Install & Run
- Setup: `make setup`
- Minimal setup: `uv sync`
- Config‑driven: `cp config/transcriber.example.yaml config/transcriber.yaml && uv run transcribe`
- Direct CLI: `uv run transcribe audio/Session_32.zip --model large-v3 --speaker-mapping config/speaker_mapping.example.yaml`
- Apple Silicon local run: `uv run transcribe audio/sample.wav --backend parakeet --model parakeet`
- Docker (GPU): `make docker-build` then `HF_TOKEN=... make docker-run INPUT=audio/Session_32.zip MODEL=large-v3`

## Device Selection
- `--device cuda` forces GPU; if CUDA/cuDNN fails, the run errors (no CPU fallback).
- `--device auto` picks GPU when available, otherwise CPU (fallback allowed).
- `--device cpu` forces CPU. Use `--pyannote-on-cpu` when you want diarization and speaker embeddings to stay on CPU as well.

## Platform Notes
- `backend: auto` is the default. It resolves to `faster` for CUDA and speaker-aware runs, and to `parakeet` for multi-track ZIP stems or explicit single-speaker CPU runs on Apple Silicon and Linux.
- Windows stays on `faster` under `auto`; the current Parakeet runtime support is Linux plus Apple Silicon.
- Linux x86_64 / WSL2: `uv sync` installs the GPU-enabled Torch and ONNX Runtime stack used by the `faster` backend, plus the NeMo Parakeet runtime for CPU fallback.
- Apple Silicon: `uv sync` installs CPU-safe dependencies plus `parakeet-mlx`.
- The CLI still auto-preloads pip-installed cuDNN/cuBLAS split libraries on Linux, so most users do not need extra `LD_LIBRARY_PATH` setup.

## CLI (Synopsis)
- `uv run transcribe INPUT [--backend auto|faster|parakeet] [options]`
- Common options: `--model large-v3` `--compute-type float16|int8_float16` `--device auto|cuda|cpu` `--batch-size N` `--auto-batch/--no-auto-batch` `--speaker-mapping PATH` `--min-speakers/--max-speakers` `--output-dir outputs` `--cache-root PATH|--cache-mode home|repo|env` `--local-files-only` `--log-level INFO|WARNING` `--config PATH`
- `auto` is the default runtime backend.

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
- Chunked training from labelled segments (`--segments-json` or `speaker_bank.train.from_segments`) generates dozens of embeddings per speaker. Segment artifacts are auto-discovered from `.outputs/<session>/<session>.jsonl` or `.diarization.json`; control with `speaker_bank.train.segment_source`.
- Configure chunk sizing via `speaker_bank.train.window_size`/`window_stride`, limit samples per speaker with `speaker_bank.train.max_embeddings_per_speaker`, and pad windows using `speaker_bank.train.pre_pad`/`post_pad` to absorb alignment drift.
- Use `--speaker-bank-train-only PATH --segments-json DIR` to regenerate the bank from processed sessions, then run transcription normally to apply the refreshed clusters.
- For matching, enable per-segment aggregation (`speaker_bank.match_per_segment: true`), adjust acceptance with `speaker_bank.scoring_margin`, and tune prototype density with `speaker_bank.prototypes_per_cluster` to increase recall on cross-domain audio while preserving precision.
- Match mixed files against the bank during transcription; unmatched segments are labelled `unknown` and retain their raw diarization label in `speaker_raw` for inspection.
- Run offline training on processed audio with `--speaker-bank-train-only PATH` to ingest new voices without triggering transcription.

## Inputs & Outputs
- Inputs: single audio (.wav, .mp3, .ogg, .flac, .m4a), directory (recursive), or ZIP (multi-track safe extract)
- Outputs in `outputs/<INPUT_BASE>/`: `.txt`, `.srt`, `.jsonl`, `.diarization.json`, `.exclusive_diarization.json`
- Generated transcripts and evaluation artifacts are local-only and should live under `.outputs/` or `transcripts/`, both of which are gitignored.

## Caching
Default root `~/hf_cache`: HF hub (`$HF_HOME/hub`), models (`$HF_HOME/models`), align (`$HF_HOME/align`). Configure with `--cache-root` or `--cache-mode home|repo|env`. Use `--hf-cache-root` to pin model caches independently, and `--speaker-bank-root` to relocate bank profiles (e.g., keep embeddings inside the repo while sharing models globally). Offline: `--local-files-only` (set `HF_TOKEN` for gated models).

## Troubleshooting
- cuDNN error (`libcudnn_cnn.so.9…`): run via the CLI first. It auto-preloads the packaged cuDNN/cuBLAS libraries.
- No diarization pipeline: ASR still runs; set `HF_TOKEN`, make sure the pyannote gated terms are accepted, or use `--pyannote-on-cpu`.
- Need strict GPU: use `--device cuda`. For fallback behavior, use `--device auto`.

## Development
- Lint: `make lint`  |  Tests: `make test`  |  Local run: `make run INPUT=audio/sample.wav MODEL=medium`
