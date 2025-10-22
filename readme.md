# Transcriber

GPU‑accelerated speech transcription powered by Whisper/WhisperX with optional alignment
and diarization. Works on single audio files, directories, or multi‑track ZIPs.


## Features

- WhisperX backend (ASR + word alignment + diarization when available)
- Faster‑Whisper backend (fast ASR only)
- Config‑driven runs with auto‑discovery (`transcriber.yaml`) and CLI override
- Quiet‑by‑default output with clean progress bars and ETA
- VRAM‑aware auto‑batching (upper‑bounded by `--batch-size`) and OOM fallback
- Per‑track speaker labeling with flexible mapping
- Text output: `Speaker HH:MM:SS Text`
- Combined SRT and JSONL exports; optional diarization JSON
- Safe ZIP extraction (prevents path traversal)
- Reusable caches under `~/hf_cache` (HF hub, models, aligners)


## Quick Start

Prerequisites
- Python 3.11
- FFmpeg installed on your system
- NVIDIA GPU recommended for WhisperX; CPU works with the faster‑whisper backend

Setup
- `make setup`  # creates venv with uv and installs dependencies

Basic usage (config‑driven)
- `cp config/transcriber.example.yaml config/transcriber.yaml`
- edit values (optionally set `input:`)
- `uv run transcribe`

One‑liner (WhisperX on GPU)
- `make run INPUT=audio/sample.wav MODEL=medium`

Direct CLI
- `uv run transcribe audio/Session_32.zip --model large-v3 --speaker-mapping config/speaker_mapping.example.yaml`

Docker (GPU)
- `make docker-build`
- `HF_TOKEN=... make docker-run INPUT=audio/Session_32.zip MODEL=large-v3`


## CLI Usage

The console entry point is `transcribe` (installed from `src/transcriber/cli.py`).

Synopsis
- `uv run transcribe INPUT [--backend whisperx|faster] [options]`
- `uv run transcribe` (auto‑loads `transcriber.yaml` if present)
- `uv run transcribe --config config/transcriber.yaml` (explicit config)

Key options
- `--backend` WhisperX (`whisperx`, default) or faster‑whisper (`faster`)
- `--model` Whisper/WhisperX model (e.g., `large-v3`, `medium.en`)
- `--compute-type` Precision (e.g., `float16`, `int8_float16`, CPU uses `int8` by default)
- `--batch-size` Transcription batch size (default: 32)
  - With `--auto-batch` (default), your provided `--batch-size` is treated as an upper bound and a conservative size is chosen for your GPU/model.
- `--speaker-mapping` YAML/JSON mapping of track IDs -> friendly names (see below)
- `--min-speakers/--max-speaker` Target diarization cluster counts (WhisperX only)
- `--output-dir` Output folder (default: `outputs`)
- `--no-srt` / `--no-jsonl` Skip writing SRT / JSONL
- `--cache-root` Base directory for caches (HF hub + models + align)
- `--local-files-only` Fail fast if models are not already cached (offline)
- `--single-file-speaker` Force a label when transcribing a single file without diarization
- `--log-level` `DEBUG|INFO|WARNING|ERROR` (default: `ERROR`)
- `--vad-on-cpu` Force WhisperX VAD (voice activity detection) to run on CPU. Use this if you hit GPU/cuDNN issues during the VAD phase.
- `--pyannote-on-cpu` Force all pyannote stages (VAD + diarization) to run on CPU. This is useful when debugging GPU issues specifically related to pyannote.
- `--config` Load defaults from a YAML/JSON config (see below). CLI flags still override config values.
 - `--auto-batch` / `--no-auto-batch` Enable/disable batch-size recommendation.

Examples
- Single file with a fixed label (ASR only):
  - `uv run transcribe audio/sample.wav --backend faster --model medium.en --single-file-speaker "Group Discussion"`
- Multi‑track ZIP with mapping and diarization:
  - `uv run transcribe audio/Session_32.zip --model large-v3 --speaker-mapping config/speaker_mapping.example.yaml --min-speakers 6 --max-speakers 6`
- Offline run with a shared cache:
  - `HF_TOKEN=... uv run transcribe audio/Session_32.zip --local-files-only --cache-root /home/$USER/hf_cache`
- Config‑driven run (no flags):
  - `cp config/transcriber.example.yaml config/transcriber.yaml`
  - Edit values as needed (optionally set `input:`), then:
  - `uv run transcribe --config config/transcriber.yaml`
- Watch a folder for new audio/ZIPs and auto‑transcribe:
  - `uv run transcribe audio/ --watch` (INPUT should be a directory)
  - If you accidentally pass a file/ZIP path, watch mode will watch its parent directory
  - New files are processed when their corresponding `<base>.txt` is not found and the file is stable for a few seconds (configurable via `--watch-stability`).

## Configuration

You can reduce CLI flags by providing a config file. The CLI looks for it in this order:
- `--config /path/to/transcriber.yaml`
- `TRANSCRIBER_CONFIG` environment variable
- `./transcriber.yaml` or `./transcriber.json`
- `./config/transcriber.yaml` or `./config/transcriber.json`
- `~/.config/transcriber/config.yaml` or `config.json`

Example `config/transcriber.example.yaml` is included. CLI flags always override config values. You can also set `input:` in the config to avoid the positional argument entirely. When `auto_batch: true`, the recommended batch size considers GPU VRAM and model size; your `batch_size` acts as a cap, so setting it too high won’t degrade performance.


## Speaker Mapping

Provide a YAML or JSON file that maps track IDs (stems) to display names.

YAML example (`config/speaker_mapping.example.yaml`)
```
autbot_80hd55561_0: "B. Ver"
bfschmity_0: "Kaladen"
travisaurus6985_0: "Cyrus Schwert"
```

Matching rules (robust across sessions)
- File extensions are ignored and matching is case‑insensitive.
- If a filename has a leading numeric prefix like `2-foo_0.ogg`, the `2-` is ignored.
- If a filename contains hyphens, the portion after the last hyphen is preferred (e.g., `x-foo_0` -> `foo_0`).
- Exact or substring matches select the display name.

Multi‑track behavior
- For multi‑file inputs (e.g., a ZIP), each file’s segments are labeled with the per‑file speaker from the mapping, regardless of diarization cluster names.
- For single‑file inputs, labels come from diarization when available; otherwise `--single-file-speaker` can set a fixed label.


## Inputs

- Single audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`)
- Directory: all supported audio files are discovered recursively
- ZIP archive: extracted to a temporary folder, supports multi‑track recording exports


## Outputs

By default, outputs are nested under a folder named after your input base. For example, `Session 32.zip` writes to `outputs/Session 32/`:
- `outputs/Session 32/Session 32.txt` — lines of `Speaker HH:MM:SS Text`
- `outputs/Session 32/Session 32.srt` — combined subtitle file
- `outputs/Session 32/Session 32.jsonl` — per‑segment records: `{file, start, end, speaker, text}`
- `outputs/Session 32/Session 32.diarization.json` — diarization segments per file (when available)


## Caching

The tool coordinates three caches (default root: `~/hf_cache`):
- HF Hub artifacts: `HF_HOME` (and `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`)
- Faster‑Whisper model downloads: `$HF_HOME/models`
- WhisperX align models: `$HF_HOME/align`

Ways to configure
- Default (cache_mode: home): `~/hf_cache` (the CLI expands and resolves the path and sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`, and `TRANSFORMERS_CACHE` under it)
- Per‑run: `--cache-root ~/hf_cache`
- Config: `cache_mode: home|repo|env`
  - `home`: use `~/hf_cache`
  - `repo`: use `./.hf_cache` inside the project (portable per‑repo cache)
  - `env`: do not override, rely on `HF_*` environment variables
- Environment: `export HF_HOME=~/hf_cache` (optionally `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`)
- Project var: `TRANSCRIBER_CACHE_ROOT=~/hf_cache`

Offline mode
- Use `--local-files-only` to refuse network access and only use cached models.
- Set `HF_TOKEN` if you rely on gated models (e.g., diarization). The CLI logs will indicate if a token is missing.
- The CLI disables `hf_transfer` by default; if you override it (`HF_HUB_ENABLE_HF_TRANSFER=1`), install `hf_transfer`.


## GPU / CPU Behavior

- WhisperX prefers CUDA; otherwise CPU.
- Faster‑Whisper auto‑detects device; CPU falls back to `int8`.
- Diarization (pyannote) uses PyTorch; ensure a matching cuDNN is available for GPU.

Performance tips
- Auto‑batch picks a conservative batch size for your GPU and model; your `--batch-size` is treated as a cap
- Very large batches can be slower (and cause OOM). The pipeline reduces batch size automatically if OOM is detected
- Free VRAM (e.g., move pyannote to CPU with `--pyannote-on-cpu`) to enable larger batches when beneficial

Troubleshooting diarization on GPU
- Error like `Unable to load any of {libcudnn_cnn.so.9...}` means split cuDNN libs are missing/not visible.
- Fix: install cuDNN 9 for your CUDA 12.x runtime, or run via the provided Docker image.
- Workaround: use `--backend faster` (ASR only) or let diarization run on CPU (slower).
- If VAD itself is the issue, try `--vad-on-cpu` so only VAD runs on CPU while keeping ASR/align on GPU.
- If pyannote broadly is the culprit, use `--pyannote-on-cpu` to force both VAD and diarization to CPU.


## Development

- Lint: `make lint`  (ruff + black)
- Tests: `make test`  (pytest)
  - Targeted: `uv run pytest tests/test_speaker_mapping.py -k strips_numeric_prefix`
- Local run: `make run INPUT=audio/sample.wav MODEL=medium`


## Docker

Build and run (GPU)
- `make docker-build`
- `docker images | grep transcriber`  # verify
- One‑shot: `HF_TOKEN=... make docker-run INPUT=audio/Session_32.zip MODEL=large-v3`
- Watch: `HF_TOKEN=... make docker-run-watch`

## Watcher Logs & Debugging

Inspect the systemd user service
- Status: `systemctl --user status transcriber-watch.service`
- Tail logs live: `journalctl --user -u transcriber-watch.service -f -o cat`
- Jump to end: `journalctl --user -u transcriber-watch.service -e -o cat`
- Last N lines: `journalctl --user -u transcriber-watch.service -n 200 -o cat`
- Recent window: `journalctl --user -u transcriber-watch.service -S -1h -o cat`

Ensure only one watcher is running
- Check processes: `pgrep -a -u "$USER" -f "uv run transcribe --watch"`
- Restart cleanly:
  - `systemctl --user daemon-reload`
  - `systemctl --user restart transcriber-watch.service`

Explicit config
- The service runs with `--config /home/$USER/projects/transcriber/config/transcriber.yaml` to ensure consistent behavior across shells.

Quarantine behavior
- Corrupted inputs (e.g., bad ZIP CRC) are moved to `audio/quarantine/` to avoid infinite retries. Check there if a file doesn’t reappear in logs.

The image pins a CUDA channel and installs compatible PyTorch wheels to avoid host CUDA/cuDNN mismatches.


## Environment Parity & Deployment Notes

To avoid CUDA/cuDNN mismatches across shells, services, and hosts, keep these rules:

- Single‑strategy rule (critical)
  - If your project uses pip `nvidia‑cu12` split libraries (as this repo does), keep `LD_LIBRARY_PATH` restricted to the venv’s `site-packages/nvidia/*/lib` and `site-packages/torch/lib`. Do not append `/usr/local/cuda-*/lib64` to `LD_LIBRARY_PATH` in the same process.
  - If you prefer system CUDA/cuDNN, do not install pip `nvidia‑cu12` split packages in that venv.

- Watch service env
  - The systemd unit consumes `~/.config/transcriber/watch.env`. Regenerate it per machine:
    - `bash transcriber/scripts/install_watch_service.sh` (or `--env-file ~/.config/transcriber/watch.env`)
    - This script auto‑discovers the venv’s NVIDIA/Torch lib paths and writes a compatible unit file.
  - Keep `LD_LIBRARY_PATH` to venv libs only. `PATH` may include `/usr/local/cuda-12.x/bin` safely.
  - Apply changes: `systemctl --user daemon-reload && systemctl --user restart transcriber-watch.service`.

- New machine checklist
  - `make setup` (creates venv and installs deps via `uv`)
  - `bash transcriber/scripts/install_watch_service.sh`
  - Drop an audio ZIP into the watched folder; verify logs show `Using device=cuda` and track times in ~30–60s.

- Common pitfalls
  - Mixing system CUDA libraries with pip `nvidia‑cu12` in `LD_LIBRARY_PATH` can cause native crashes (e.g., `free(): double free detected`).
  - Ensure Torch CUDA wheels match the CUDA channel you expect (this repo pins cu128 in `pyproject.toml`).

### Docker (Deterministic GPU Runtime)

Use the provided `Dockerfile` to run on GPU without host CUDA/cuDNN variance. The image pins Torch to CUDA 12.8 + cuDNN 9, matching this repo’s configuration.

Build
- `docker build -t transcriber-gpu -f Dockerfile .`

Run (one‑shot)
- `docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN -v "$PWD:/app" -w /app transcriber-gpu \
   uv run transcribe audio/Session_29.zip --model large-v3 --log-level WARNING`

Run (watch mode)
- `docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN -v "$PWD:/app" -w /app transcriber-gpu \
   uv run transcribe audio/ --watch`

Notes
- Mount your project/workdir and caches as needed; e.g., add `-v "$HOME/hf_cache:/root/hf_cache" -e HF_HOME=/root/hf_cache` for persistent model caches.
- Ensure the host NVIDIA container runtime is configured so `--gpus all` exposes the GPU.


## Notes

- Quiet mode suppresses library logs/warnings but keeps progress bars with ETA
- Some diarization models on Hugging Face are gated—export `HF_TOKEN` if required
- If you prefer SRT cues to include speaker names, open an issue—we can add an option
