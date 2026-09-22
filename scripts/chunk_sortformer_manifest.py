from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_rttm_speakers(path: Path) -> list[dict]:
    intervals = []
    if not path.exists():
        return intervals
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                continue
            if duration <= 0.0:
                continue
            intervals.append(
                {
                    "start": start,
                    "end": start + duration,
                    "speaker": parts[7],
                }
            )
    return intervals


def _active_speakers(
    intervals: Sequence[Mapping[str, object]],
    *,
    start: float,
    end: float,
) -> set[str]:
    return {
        str(item["speaker"])
        for item in intervals
        if float(item.get("end") or 0.0) > start and float(item.get("start") or 0.0) < end
    }


def chunk_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    chunk_seconds: float,
    hop_seconds: float | None = None,
    min_active_speakers: int = 1,
    min_chunk_seconds: float = 1.0,
) -> tuple[list[dict], dict]:
    if chunk_seconds <= 0.0:
        raise ValueError("chunk_seconds must be positive")
    hop_seconds = chunk_seconds if hop_seconds is None else hop_seconds
    if hop_seconds <= 0.0:
        raise ValueError("hop_seconds must be positive")

    chunked = []
    skipped = Counter()
    speaker_count_distribution: Counter[int] = Counter()
    for row in rows:
        duration = float(row.get("duration") or 0.0)
        base_offset = float(row.get("offset") or 0.0)
        if duration <= 0.0:
            skipped["empty_duration"] += 1
            continue
        intervals = _parse_rttm_speakers(Path(str(row.get("rttm_filepath") or "")))
        max_start = max(duration - min_chunk_seconds, 0.0)
        chunk_index = 0
        relative_start = 0.0
        while relative_start <= max_start + 1e-6:
            relative_end = min(relative_start + chunk_seconds, duration)
            actual_duration = relative_end - relative_start
            if actual_duration < min_chunk_seconds:
                skipped["too_short"] += 1
                break
            absolute_start = base_offset + relative_start
            absolute_end = base_offset + relative_end
            speakers = _active_speakers(intervals, start=absolute_start, end=absolute_end)
            if len(speakers) < min_active_speakers:
                skipped["too_few_active_speakers"] += 1
            else:
                item = dict(row)
                item["offset"] = round(absolute_start, 3)
                item["duration"] = round(actual_duration, 3)
                item["num_speakers"] = len(speakers)
                item["chunk_index"] = chunk_index
                item["chunk_source_offset"] = float(row.get("offset") or 0.0)
                item["chunk_source_duration"] = duration
                chunked.append(item)
                speaker_count_distribution[len(speakers)] += 1
            chunk_index += 1
            relative_start += hop_seconds
            if math.isclose(relative_start, duration):
                break

    summary = {
        "rows": len(chunked),
        "chunk_seconds": float(chunk_seconds),
        "hop_seconds": float(hop_seconds),
        "min_active_speakers": int(min_active_speakers),
        "skipped": dict(sorted(skipped.items())),
        "speaker_count_distribution": {
            str(key): value for key, value in sorted(speaker_count_distribution.items())
        },
    }
    return chunked, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a NeMo Sortformer manifest into shorter offset/duration rows."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--chunk-seconds", type=float, required=True)
    parser.add_argument("--hop-seconds", type=float)
    parser.add_argument("--min-active-speakers", type=int, default=1)
    parser.add_argument("--min-chunk-seconds", type=float, default=1.0)
    args = parser.parse_args()

    chunked, summary = chunk_rows(
        _read_jsonl(args.manifest),
        chunk_seconds=float(args.chunk_seconds),
        hop_seconds=args.hop_seconds,
        min_active_speakers=int(args.min_active_speakers),
        min_chunk_seconds=float(args.min_chunk_seconds),
    )
    summary.update(
        {
            "input_manifest": str(args.manifest),
            "output_manifest": str(args.output_manifest),
        }
    )
    _write_jsonl(args.output_manifest, chunked)
    summary_path = args.summary_json or args.output_manifest.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
