#!/usr/bin/env bash
set -euo pipefail

# Wrapper invoked by the systemd watcher service. It mirrors the interactive
# development environment by launching the CLI from the repo's virtualenv and
# ensures split CUDA libraries packaged with nvidia-* wheels remain discoverable.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
BIN_DIR="$VENV_DIR/bin"
TRANSCRIBE_BIN="$BIN_DIR/transcribe"
PYTHON_BIN="$BIN_DIR/python"

if [[ ! -x "$TRANSCRIBE_BIN" ]]; then
  echo "transcribe entrypoint missing at $TRANSCRIBE_BIN" >&2
  exit 1
fi

# Export the same virtualenv variables that `source .venv/bin/activate` would.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$BIN_DIR:${PATH:-}"
export PYTHONUNBUFFERED=1

# Prepend CUDA split-lib directories from the venv, mirroring CLI auto-detection.
if [[ -x "$PYTHON_BIN" ]]; then
  CUDA_LIBS="$("$PYTHON_BIN" <<'PY'
import sys
from pathlib import Path

dirs = []
for root in map(Path, sys.path):
    for sub in (root / "nvidia" / "cudnn" / "lib", root / "nvidia" / "cublas" / "lib"):
        if sub.is_dir():
            dirs.append(str(sub))
print(":".join(dirs))
PY
  )"
  if [[ -n "$CUDA_LIBS" ]]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
  fi
fi
WATCH_CONFIG_DEFAULT="$REPO_DIR/config/transcriber.watch.yaml"
CONFIG_ARGS=()
if [[ -n "${WATCH_CONFIG:-}" ]]; then
  CONFIG_ARGS+=(--config "$WATCH_CONFIG")
elif [[ -f "$WATCH_CONFIG_DEFAULT" ]]; then
  CONFIG_ARGS+=(--config "$WATCH_CONFIG_DEFAULT")
fi

exec "$TRANSCRIBE_BIN" "${CONFIG_ARGS[@]}" audio/ --watch "$@"
