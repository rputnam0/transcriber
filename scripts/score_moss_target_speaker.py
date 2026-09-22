from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _tokens(value: object) -> list[str]:
    return TOKEN_RE.findall(str(value or "").lower())


def _lcs_reference_indices(reference: Sequence[str], prediction: Sequence[str]) -> set[int]:
    rows = len(reference)
    columns = len(prediction)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            if reference[row] == prediction[column]:
                lengths[row][column] = lengths[row + 1][column + 1] + 1
            else:
                lengths[row][column] = max(
                    lengths[row + 1][column],
                    lengths[row][column + 1],
                )
    matched = set()
    row = 0
    column = 0
    while row < rows and column < columns:
        if reference[row] == prediction[column]:
            matched.add(row)
            row += 1
            column += 1
        elif lengths[row + 1][column] >= lengths[row][column + 1]:
            row += 1
        else:
            column += 1
    return matched


def _prediction_tokens(record: Mapping[str, object]) -> list[str]:
    segments = sorted(
        list(record.get("segments") or []),
        key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)),
    )
    return [token for segment in segments for token in _tokens(segment.get("text"))]


def score_target_outputs(
    manifest_rows: Iterable[Mapping[str, object]],
    output_records: Iterable[Mapping[str, object]],
) -> dict:
    references = {
        str(dict(row.get("metadata") or {}).get("cut_id") or ""): dict(row.get("metadata") or {})
        for row in manifest_rows
    }
    outputs = {str(record.get("cut_id") or ""): dict(record) for record in output_records}
    missing = sorted(set(references) - set(outputs))
    if missing:
        raise ValueError(
            f"Missing model outputs for {len(missing)} records, including {missing[0]}"
        )

    totals = Counter()
    records = []
    for cut_id, metadata in references.items():
        output = outputs[cut_id]
        predicted = _prediction_tokens(output)
        reference_items = []
        for segment in list(metadata.get("reference_segments") or []):
            for token in _tokens(segment.get("text")):
                reference_items.append(
                    {"token": token, "brief_overlap": bool(segment.get("brief_overlap"))}
                )
        reference = [str(item["token"]) for item in reference_items]
        matched_indices = _lcs_reference_indices(reference, predicted)
        bag_reference = Counter(reference)
        bag_prediction = Counter(predicted)
        bag_matches = sum((bag_reference & bag_prediction).values())
        matched = len(matched_indices)
        brief_indices = {
            index for index, item in enumerate(reference_items) if item["brief_overlap"]
        }
        brief_matches = len(matched_indices & brief_indices)
        negative = bool(metadata.get("negative"))
        totals["records"] += 1
        totals["negative_records"] += int(negative)
        totals["negative_records_with_words"] += int(negative and bool(predicted))
        totals["negative_false_positive_words"] += len(predicted) if negative else 0
        totals["reference_words"] += len(reference)
        totals["predicted_words"] += len(predicted)
        totals["bag_matches"] += bag_matches
        totals["sequence_matches"] += matched
        totals["brief_overlap_reference_words"] += len(brief_indices)
        totals["brief_overlap_sequence_matches"] += brief_matches
        records.append(
            {
                "cut_id": cut_id,
                "session": metadata.get("session"),
                "target_speaker": metadata.get("target_speaker"),
                "negative": negative,
                "reference_words": len(reference),
                "predicted_words": len(predicted),
                "bag_matches": bag_matches,
                "sequence_matches": matched,
                "brief_overlap_reference_words": len(brief_indices),
                "brief_overlap_sequence_matches": brief_matches,
            }
        )
    return {
        **dict(totals),
        "target_bag_recall": totals["bag_matches"] / max(1, totals["reference_words"]),
        "target_bag_precision": totals["bag_matches"] / max(1, totals["predicted_words"]),
        "target_sequence_recall": totals["sequence_matches"] / max(1, totals["reference_words"]),
        "target_sequence_precision": totals["sequence_matches"] / max(1, totals["predicted_words"]),
        "brief_overlap_sequence_recall": totals["brief_overlap_sequence_matches"]
        / max(1, totals["brief_overlap_reference_words"]),
        "negative_record_false_positive_rate": totals["negative_records_with_words"]
        / max(1, totals["negative_records"]),
        "records_detail": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score enrollment-conditioned MOSS target-speaker recovery."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_output = json.loads(args.model_output.read_text(encoding="utf-8"))
    score = score_target_outputs(
        _read_jsonl(args.manifest),
        list(model_output.get("records") or []),
    )
    payload = {
        "manifest": str(args.manifest),
        "model_output": str(args.model_output),
        **score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records_detail"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
