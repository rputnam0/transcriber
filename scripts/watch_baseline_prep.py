from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_bytes(value: object) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tail stage_metrics.jsonl and report stalls, RSS growth, GPU detaches, and stage transitions."
    )
    parser.add_argument("metrics_log", type=Path, help="Path to stage_metrics.jsonl.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval.")
    parser.add_argument("--stall-seconds", type=float, default=900.0, help="Warn after this much inactivity.")
    parser.add_argument("--rss-gib-threshold", type=float, default=20.0, help="Warn above this RSS threshold.")
    parser.add_argument(
        "--gpu-detach-grace",
        type=int,
        default=3,
        help="Warn after this many consecutive records report no GPU memory.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    metrics_log = Path(args.metrics_log).expanduser().resolve()
    last_count = 0
    last_timestamp = time.time()
    last_rss_bytes = 0.0
    gpu_detached_count = 0

    while True:
        records = _load_records(metrics_log)
        new_records = records[last_count:]
        if new_records:
            last_timestamp = time.time()
        for record in new_records:
            stage = str(record.get("stage") or "unknown")
            status = str(record.get("status") or "unknown")
            rss_bytes = float(record.get("rss_bytes") or 0.0)
            gpu_memory_bytes = record.get("gpu_memory_bytes")
            cache_hit = record.get("cache_hit")
            session = record.get("session")
            variant = record.get("variant")
            parts = [f"[{stage}]", status]
            if session:
                parts.append(f"session={session}")
            if variant:
                parts.append(f"variant={variant}")
            if cache_hit is not None:
                parts.append(f"cache_hit={cache_hit}")
            parts.append(f"rss={_format_bytes(rss_bytes)}")
            parts.append(f"gpu={_format_bytes(gpu_memory_bytes)}")
            print(" ".join(parts), flush=True)

            if rss_bytes >= float(args.rss_gib_threshold) * (1024**3) and rss_bytes >= last_rss_bytes:
                print(
                    f"WARNING: RSS is high and non-decreasing: {_format_bytes(rss_bytes)}",
                    flush=True,
                )
            last_rss_bytes = max(last_rss_bytes, rss_bytes)
            if gpu_memory_bytes is None:
                gpu_detached_count += 1
                if gpu_detached_count >= int(args.gpu_detach_grace):
                    print("WARNING: GPU memory has been absent across multiple records.", flush=True)
            else:
                gpu_detached_count = 0

        last_count = len(records)
        idle_seconds = time.time() - last_timestamp
        if idle_seconds >= float(args.stall_seconds):
            print(
                f"WARNING: no new metrics records for {idle_seconds:.0f}s at {metrics_log}",
                flush=True,
            )
            last_timestamp = time.time()
        time.sleep(max(float(args.poll_seconds), 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
