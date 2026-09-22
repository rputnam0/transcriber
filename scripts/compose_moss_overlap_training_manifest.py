from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Mapping


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compose_records(
    overlap_rows: Iterable[Mapping[str, object]],
    ordinary_rows: Iterable[Mapping[str, object]],
    *,
    ordinary_examples: int,
    seed: int,
    maximum_ordinary_brief_overlap_fraction: float = 0.0,
) -> tuple[list[dict], dict]:
    overlap = [dict(row) for row in overlap_rows]
    ordinary = []
    seen = set()
    for raw_row in ordinary_rows:
        row = dict(raw_row)
        metadata = dict(row.get("metadata") or {})
        cut_id = str(metadata.get("cut_id") or "")
        brief_overlap_fraction = float(metadata.get("brief_overlap_word_fraction") or 0.0)
        is_ordinary = metadata.get("has_overlap") is False
        is_low_overlap = (
            maximum_ordinary_brief_overlap_fraction > 0.0
            and brief_overlap_fraction <= maximum_ordinary_brief_overlap_fraction
        )
        if not (is_ordinary or is_low_overlap) or not cut_id or cut_id in seen:
            continue
        ordinary.append(row)
        seen.add(cut_id)
    random.Random(seed).shuffle(ordinary)
    ordinary = ordinary[:ordinary_examples]
    records = overlap + ordinary
    random.Random(seed).shuffle(records)
    brief_words = sum(
        int(dict(row.get("metadata") or {}).get("brief_overlap_word_count") or 0) for row in overlap
    )
    all_words = sum(int(dict(row.get("metadata") or {}).get("word_count") or 0) for row in records)
    return records, {
        "overlap_examples": len(overlap),
        "ordinary_examples": len(ordinary),
        "records": len(records),
        "brief_overlap_words": brief_words,
        "all_target_words": all_words,
        "brief_overlap_word_fraction": brief_words / max(1, all_words),
        "maximum_ordinary_brief_overlap_fraction": (maximum_ordinary_brief_overlap_fraction),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix overlap-centered MOSS crops with a small ordinary-speech guardrail set."
    )
    parser.add_argument(
        "--overlap-manifest",
        type=Path,
        action="append",
        required=True,
        help="Overlap manifest; repeat to combine real and synthetic crops.",
    )
    parser.add_argument("--ordinary-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ordinary-examples", type=int, default=30)
    parser.add_argument(
        "--maximum-ordinary-brief-overlap-fraction",
        type=float,
        default=0.0,
        help="Also admit guardrail examples at or below this brief-overlap word fraction.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    records, summary = compose_records(
        (row for manifest in args.overlap_manifest for row in _read_jsonl(manifest)),
        _read_jsonl(args.ordinary_manifest),
        ordinary_examples=args.ordinary_examples,
        seed=args.seed,
        maximum_ordinary_brief_overlap_fraction=(args.maximum_ordinary_brief_overlap_fraction),
    )
    if not records:
        raise ValueError("No records selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = {
        "overlap_manifests": [str(path) for path in args.overlap_manifest],
        "ordinary_manifest": str(args.ordinary_manifest),
        "output": str(args.output),
        "mono_input_only": True,
        **summary,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
