#!/usr/bin/env bash
set -euo pipefail

# Install and start a user-level systemd service that watches ./audio using the
# project's virtualenv (mirroring the interactive environment).
#
# Usage: scripts/install_watch_service.sh [--device cuda|cpu|auto] [--no-start]
#        [--no-linger] [extra transcribe args]

DEVICE="auto"
START=1
LINGER=1
EXTRA_ARGS=()

while (("$#")); do
  case "$1" in
    --device)
      DEVICE="${2:-auto}"
      shift 2
      ;;
    --no-start)
      START=0
      shift
      ;;
    --no-linger)
      LINGER=0
      shift
      ;;
    --linger)
      LINGER=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
TRANSCRIBE_BIN="$VENV_DIR/bin/transcribe"
RUNNER="$REPO_DIR/scripts/watch_service_runner.sh"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_NAME="transcriber-watch.service"
mkdir -p "$UNIT_DIR"

if [[ ! -x "$RUNNER" ]]; then
  echo "Watcher runner $RUNNER is missing or not executable." >&2
  exit 1
fi

if [[ ! -x "$TRANSCRIBE_BIN" ]]; then
  cat >&2 <<'ERR'
transcribe CLI not found in .venv.
Run `make setup` (or `uv sync`) from the project root to build the virtualenv,
then re-run this installer.
ERR
  exit 1
fi

CMD=("$RUNNER" "--device" "$DEVICE")
if ((${#EXTRA_ARGS[@]} > 0)); then
  CMD+=("${EXTRA_ARGS[@]}")
fi
CMD_STR="$(printf "%q " "${CMD[@]}")"
CMD_STR="${CMD_STR% }"

cat >"$UNIT_DIR/$UNIT_NAME" <<UNIT
[Unit]
Description=Transcriber Watcher (audio directory)
After=default.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
Environment=VIRTUAL_ENV=$VENV_DIR
Environment=PATH=$VENV_DIR/bin:/usr/bin:/bin
ExecStart=$CMD_STR
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"

if (( LINGER )); then
  if command -v loginctl >/dev/null 2>&1; then
    if ! loginctl show-user "$USER" -p Linger 2>/dev/null | grep -qi 'Linger=yes'; then
      if loginctl enable-linger "$USER"; then
        echo "Enabled lingering for $USER (service survives logout)."
      else
        echo "Unable to enable lingering automatically. Run: sudo loginctl enable-linger $USER" >&2
      fi
    fi
  else
    echo "loginctl not found; skipping linger enablement." >&2
  fi
fi

if (( START )); then
  systemctl --user restart "$UNIT_NAME"
  echo "Watcher service started. Follow logs via: journalctl --user -u $UNIT_NAME -f -o cat"
else
  echo "Watcher service installed. Start with: systemctl --user start $UNIT_NAME"
fi
