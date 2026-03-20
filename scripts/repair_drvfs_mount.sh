#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: repair_drvfs_mount.sh --probe-path PATH --drive LETTER: [options]

Options:
  --mount-point PATH   Linux mount point (default: /mnt/<drive-letter>)
  --distro NAME        WSL distro name for the Windows-side remount helper (default: Ubuntu)
  --quiet              Suppress informational logs unless a repair is attempted
EOF
  exit 2
}

probe_path=""
drive_letter=""
mount_point=""
distro_name="${WATCH_MOUNT_REPAIR_DISTRO:-Ubuntu}"
quiet=0

while (($#)); do
  case "$1" in
    --probe-path)
      probe_path="${2:-}"
      shift 2
      ;;
    --drive)
      drive_letter="${2:-}"
      shift 2
      ;;
    --mount-point)
      mount_point="${2:-}"
      shift 2
      ;;
    --distro)
      distro_name="${2:-}"
      shift 2
      ;;
    --quiet)
      quiet=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "$probe_path" || -z "$drive_letter" ]]; then
  usage
fi

if [[ -z "$mount_point" ]]; then
  drive_trimmed="${drive_letter%:}"
  mount_point="/mnt/${drive_trimmed,,}"
fi

if [[ ! "$drive_letter" =~ ^[A-Za-z]:$ ]]; then
  echo "repair_drvfs_mount.sh: drive must look like G:" >&2
  exit 2
fi

if ls -d "$probe_path" >/dev/null 2>&1; then
  if ((quiet == 0)); then
    echo "repair_drvfs_mount.sh: mount healthy at $probe_path" >&2
  fi
  exit 0
fi

wsl_exe="/mnt/c/Windows/System32/wsl.exe"
if [[ ! -x "$wsl_exe" ]]; then
  echo "repair_drvfs_mount.sh: cannot find $wsl_exe" >&2
  exit 1
fi

mount_options="metadata,uid=$(id -u),gid=$(id -g)"
if ((quiet == 0)); then
  echo "repair_drvfs_mount.sh: repairing $drive_letter on $mount_point via distro $distro_name" >&2
fi

printf -v repair_cmd \
  'set -euo pipefail; nsenter -t 1 -m -- umount -l %q 2>/dev/null || true; nsenter -t 1 -m -- mkdir -p %q; nsenter -t 1 -m -- mount -t drvfs %q %q -o %q' \
  "$mount_point" \
  "$mount_point" \
  "$drive_letter" \
  "$mount_point" \
  "$mount_options"

"$wsl_exe" -d "$distro_name" -u root -- bash -lc "$repair_cmd"

for _ in 1 2 3 4 5; do
  if ls -d "$probe_path" >/dev/null 2>&1; then
    if ((quiet == 0)); then
      echo "repair_drvfs_mount.sh: mount restored at $probe_path" >&2
    fi
    exit 0
  fi
  sleep 1
done

echo "repair_drvfs_mount.sh: remount attempted but $probe_path is still unavailable" >&2
exit 1
