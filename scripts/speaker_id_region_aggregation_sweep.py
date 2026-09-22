from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_clean_bank,
    _load_rows,
    _load_titanet_word_npz,
    evaluate_mask_embeddings,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402


def _rows_by_window(rows: Sequence[MaskRow]) -> Dict[str, List[MaskRow]]:
    by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        by_window[row.window].append(row)
    return by_window


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _region_ids(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    max_gap: float,
    max_duration: float,
    max_words: int,
) -> List[List[int]]:
    regions: List[List[int]] = []
    current: List[int] = []
    current_start = 0.0
    previous_end = 0.0
    for index, (start, end) in enumerate(zip(starts, ends)):
        start_f = float(start)
        end_f = float(end)
        should_break = False
        if current:
            gap = start_f - previous_end
            duration = end_f - current_start
            should_break = (
                gap > max_gap
                or (max_duration > 0 and duration > max_duration)
                or (max_words > 0 and len(current) >= max_words)
            )
        if should_break:
            regions.append(current)
            current = []
        if not current:
            current_start = start_f
        current.append(index)
        previous_end = end_f
    if current:
        regions.append(current)
    return regions


def _aggregated_embeddings_for_rows(
    rows: Sequence[MaskRow],
    *,
    titanet_cache_root: Path,
    window_seconds: float,
    max_gap: float,
    max_duration: float,
    max_words: int,
    include_center_weight: float,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for window_name, window_rows in sorted(_rows_by_window(rows).items()):
        payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        embeddings = _normalize_rows(np.asarray(payload["embeddings"], dtype=np.float32))
        starts = np.asarray(payload["word_starts"], dtype=np.float32)
        ends = np.asarray(payload["word_ends"], dtype=np.float32)
        regions = _region_ids(
            starts,
            ends,
            max_gap=max_gap,
            max_duration=max_duration,
            max_words=max_words,
        )
        region_by_word: Dict[int, List[int]] = {}
        for region in regions:
            for index in region:
                region_by_word[index] = region
        for row in window_rows:
            region = region_by_word.get(row.index, [row.index])
            region_matrix = embeddings[region]
            if include_center_weight > 0:
                center = embeddings[row.index : row.index + 1]
                center_rows = np.repeat(center, int(include_center_weight), axis=0)
                region_matrix = np.vstack([region_matrix, center_rows])
            vectors.append(region_matrix.mean(axis=0))
        print(
            f"aggregated_regions {window_name}: rows={len(window_rows)} regions={len(regions)}",
            flush=True,
        )
    return _normalize_rows(np.vstack(vectors).astype(np.float32))


def _score_config(
    rows: Sequence[MaskRow],
    *,
    clean_bank,
    training_items: Mapping[str, object],
    mixed_embeddings: np.ndarray,
    titanet_cache_root: Path,
    window_seconds: float,
    max_gap: float,
    max_duration: float,
    max_words: int,
    center_weight: float,
) -> Dict[str, object]:
    aggregated = _aggregated_embeddings_for_rows(
        rows,
        titanet_cache_root=titanet_cache_root,
        window_seconds=window_seconds,
        max_gap=max_gap,
        max_duration=max_duration,
        max_words=max_words,
        include_center_weight=center_weight,
    )
    result = evaluate_mask_embeddings(
        aggregated,
        rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
    )
    result.pop("predictions", None)
    mixed = evaluate_mask_embeddings(
        mixed_embeddings,
        rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
    )
    mixed.pop("predictions", None)
    return {
        "max_gap": max_gap,
        "max_duration": max_duration,
        "max_words": max_words,
        "center_weight": center_weight,
        "region_aggregation": result,
        "mixed_same_rows": mixed,
    }


def _parse_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pause-bounded local region aggregation of word Titanet embeddings."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_region_aggregation_sweep.json"),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--gaps", default="0.05,0.10,0.20,0.35,0.50")
    parser.add_argument("--durations", default="1.5,2.5,4.0,6.0")
    parser.add_argument("--max-words", default="3,5,8,12")
    parser.add_argument("--center-weights", default="0,1,3")
    args = parser.parse_args()

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    rows_by_window = _rows_by_window(rows)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    mixed_embeddings = _mixed_same_rows(
        rows_by_window,
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )

    results: List[Dict[str, object]] = []
    for max_gap in _parse_floats(str(args.gaps)):
        for max_duration in _parse_floats(str(args.durations)):
            for max_words in _parse_ints(str(args.max_words)):
                for center_weight in _parse_floats(str(args.center_weights)):
                    print(
                        "region_config "
                        f"gap={max_gap} duration={max_duration} "
                        f"max_words={max_words} center_weight={center_weight}",
                        flush=True,
                    )
                    results.append(
                        _score_config(
                            rows,
                            clean_bank=clean_bank,
                            training_items=training_items,
                            mixed_embeddings=mixed_embeddings,
                            titanet_cache_root=args.titanet_cache_root.expanduser(),
                            window_seconds=float(args.window_seconds),
                            max_gap=max_gap,
                            max_duration=max_duration,
                            max_words=max_words,
                            center_weight=center_weight,
                        )
                    )

    best = max(
        results,
        key=lambda item: float(item["region_aggregation"]["direct"]["accuracy"]),
    )
    payload = {
        "model": "region_aggregation_titanet",
        "selected_rows": len(rows),
        "selected_speakers": dict(Counter(row.truth for row in rows)),
        "window_seconds": float(args.window_seconds),
        "results": results,
        "best": best,
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy,baseline_accuracy", flush=True)
    for result in sorted(
        results,
        key=lambda item: float(item["region_aggregation"]["direct"]["accuracy"]),
        reverse=True,
    )[:12]:
        direct = result["region_aggregation"]["direct"]
        baseline = result["mixed_same_rows"]["direct"]
        print(
            ",".join(
                [
                    (
                        f"gap={result['max_gap']}/dur={result['max_duration']}/"
                        f"words={result['max_words']}/center={result['center_weight']}"
                    ),
                    str(direct["examples"]),
                    f"{float(direct['accuracy']):.4f}",
                    f"{float(baseline['accuracy']):.4f}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
