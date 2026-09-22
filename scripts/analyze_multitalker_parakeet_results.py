from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evaluate_multitalker_parakeet_cutset import reference_words_from_supervisions
from score_multitalker_parakeet import (
    _best_one_to_one_mapping,
    _prediction_streams,
    _reference_streams,
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _counter_intersection_size(left: Sequence[str], right: Sequence[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def lexical_decomposition(
    reference_words: Sequence[Mapping[str, object]],
    predicted_segments: Sequence[Mapping[str, object]],
) -> dict:
    reference, _ = _reference_streams(reference_words)
    predicted = _prediction_streams(predicted_segments)
    reference_bag = [token for tokens in reference.values() for token in tokens]
    predicted_bag = [token for tokens in predicted.values() for token in tokens]
    lexical_matches = _counter_intersection_size(reference_bag, predicted_bag)

    pair_matches = {
        (predicted_speaker, reference_speaker): _counter_intersection_size(
            predicted_tokens, reference_tokens
        )
        for predicted_speaker, predicted_tokens in predicted.items()
        for reference_speaker, reference_tokens in reference.items()
    }
    mapping = _best_one_to_one_mapping(
        pair_matches,
        sorted(predicted),
        sorted(reference),
    )
    attributed_matches = sum(
        pair_matches[(predicted_speaker, reference_speaker)]
        for predicted_speaker, reference_speaker in mapping.items()
    )
    many_to_one_matches = sum(
        max(
            (
                pair_matches[(predicted_speaker, reference_speaker)]
                for reference_speaker in reference
            ),
            default=0,
        )
        for predicted_speaker in predicted
    )
    return {
        "reference_words": len(reference_bag),
        "predicted_words": len(predicted_bag),
        "bag_lexical_matched_words": lexical_matches,
        "bag_one_to_one_attributed_matched_words": attributed_matches,
        "bag_many_to_one_stream_matched_words_proxy": many_to_one_matches,
    }


def summarize(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    reference_words = sum(int(row["reference_words"]) for row in rows)
    predicted_words = sum(int(row["predicted_words"]) for row in rows)
    lexical_matches = sum(int(row["bag_lexical_matched_words"]) for row in rows)
    attributed_matches = sum(int(row["bag_one_to_one_attributed_matched_words"]) for row in rows)
    many_to_one_matches = sum(
        int(row["bag_many_to_one_stream_matched_words_proxy"]) for row in rows
    )
    return {
        "cuts": len(rows),
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "bag_lexical_matched_words": lexical_matches,
        "bag_lexical_recall_upper_bound": (
            lexical_matches / reference_words if reference_words else 0.0
        ),
        "bag_lexical_prediction_precision_upper_bound": (
            lexical_matches / predicted_words if predicted_words else 0.0
        ),
        "bag_one_to_one_attributed_matched_words": attributed_matches,
        "bag_one_to_one_attributed_recall_upper_bound": (
            attributed_matches / reference_words if reference_words else 0.0
        ),
        "bag_owner_fraction_given_lexical_match": (
            attributed_matches / lexical_matches if lexical_matches else 0.0
        ),
        "bag_many_to_one_stream_matched_words_proxy": many_to_one_matches,
        "bag_many_to_one_stream_recall_proxy": (
            many_to_one_matches / reference_words if reference_words else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose multitalker Parakeet results into lexical and ownership ceilings."
    )
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from lhotse import CutSet

    result_rows = {str(row["cut_id"]): row for row in _read_jsonl(args.results)}
    decomposed = []
    for cut in CutSet.from_file(args.cuts):
        result = result_rows.get(cut.id)
        if result is None:
            continue
        row = lexical_decomposition(
            reference_words_from_supervisions(cut.supervisions),
            result.get("segments") or [],
        )
        row.update({"cut_id": cut.id, "has_overlap": bool(result.get("has_overlap"))})
        decomposed.append(row)

    report = {
        "metric_note": (
            "Bag metrics ignore timing and word order and are diagnostic upper bounds. "
            "Many-to-one can double-count a reference word across fragmented streams."
        ),
        "all": summarize(decomposed),
        "overlap_clips": summarize(row for row in decomposed if row["has_overlap"]),
        "non_overlap_clips": summarize(row for row in decomposed if not row["has_overlap"]),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
