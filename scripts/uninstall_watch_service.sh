#!/usr/bin/env bash
set -euo pipefail

# Remove the user-level systemd watcher service and optionally disable lingering.
#
# Usage: scripts/uninstall_watch_service.sh [--disable-linger]

DISABLE_LINGER=0

while (("$#")); do
  case "$1" in
    --disable-linger)
      DISABLE_LINGER=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

UNIT_NAME="transcriber-watch.service"
UNIT_PATH="$HOME/.config/systemd/user/$UNIT_NAME"

if command -v systemctl >/dev/null 2>&1; then
  if systemctl --user --quiet is-active "$UNIT_NAME"; then
    systemctl --user stop "$UNIT_NAME"
  fi
  if systemctl --user --quiet is-enabled "$UNIT_NAME"; then
    systemctl --user disable "$UNIT_NAME"
  fi
else
  echo "systemctl not found; skipping stop/disable actions." >&2
fi

if [[ -f "$UNIT_PATH" ]]; then
  rm -f "$UNIT_PATH"
  echo "Removed $UNIT_PATH"
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload
fi

if (( DISABLE_LINGER )); then
  if command -v loginctl >/dev/null 2>&1; then
    if loginctl show-user "$USER" -p Linger 2>/dev/null | grep -qi 'Linger=yes'; then
      if loginctl disable-linger "$USER"; then
        echo "Disabled lingering for $USER."
      else
        echo "Unable to disable lingering automatically. Run: sudo loginctl disable-linger $USER" >&2
      fi
    fi
  else
    echo "loginctl not found; skipping linger disablement." >&2
  fi
fi

echo "Watcher service uninstalled."

