# Repository Guidelines

## Project Structure & Module Organization
Primary code lives in `src/transcriber/`; `cli.py` exposes the entry point, `asr.py` wraps faster-whisper, `diarization.py` handles pyannote diarization and embeddings, and `transcript_pipeline.py` assembles the end-to-end runtime. Helpers such as `audio.py`, `consolidate.py`, and `srt.py` manage ingest and exports. The `audio/` folder holds small fixtures for manual checks—store anything larger outside the repo. Runtime configuration is auto-discovered from the root or `config/`, so share team presets as `config/transcriber.yaml` and keep environment-specific values out of source control.

## Build, Test, and Development Commands
Use `make setup` after cloning to install the uv-managed Python 3.11 interpreter, sync dependencies, and enable pre-commit hooks. `uv run transcribe audio/sample.wav --model medium` is the fastest way to verify the pipeline; `make run INPUT=audio/Session_32.zip MODEL=large-v3` exercises the GPU faster-whisper plus direct pyannote path. Keep linting clean with `make lint` (Ruff + Black) and execute `make test` before every PR. Docker flows are available via `make docker-build` and `make docker-run`, remembering to mount a Hugging Face cache through `HF_CACHE` for repeatable performance.

## Coding Style & Naming Conventions
Target CPython 3.11 while preserving compatibility with 3.10–3.12 per `pyproject.toml`. Ruff and Black enforce a 100-character line limit, four-space indentation, and prefer explicit imports; rely on `uv run ruff . --fix` for quick tidy-ups. Name modules and functions with lowercase underscores and keep identifiers descriptive of Whisper or diarization behaviour.

## Testing Guidelines
Pytest is the testing framework; place new tests under `tests/` mirroring the source tree (`tests/test_cli.py`, `tests/test_transcript_pipeline.py`, `tests/test_diarization.py`). Reuse the clip stubs in `audio/` or temporary files for fixtures so runs stay fast. Aim to cover success paths, model/cache selection, and diarization toggles, and report any intentionally skipped cases in the PR.

## Commit & Pull Request Guidelines
Write imperative, single-focus commits such as `Add diarization cache fallback`, and squash noisy WIP commits before review. Document validation steps in the PR description after running `make lint`, `make test`, and a representative transcription command. Link tracking issues, attach relevant CLI snippets, and highlight required environment variables like `HF_TOKEN` or custom cache roots.

## Security & Configuration Tips
Store secrets in the environment or manager tooling—never in git. Keep quarantine directories and cache roots inside trusted paths, and delete extracted ZIP contents after inspection. When sharing logs, strip sensitive audio names, tokens, and local absolute paths.
