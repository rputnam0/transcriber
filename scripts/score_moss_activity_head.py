from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def reference_overlap_frames(
    activity: Sequence[Mapping[str, object]],
    *,
    frame_count: int,
    frame_hz: float,
) -> list[bool]:
    output = []
    for frame in range(frame_count):
        center = (frame + 0.5) / frame_hz
        active = {
            int(interval.get("speaker_index") or 0)
            for interval in activity
            if float(interval.get("start") or 0.0) <= center < float(interval.get("end") or 0.0)
        }
        output.append(len(active) >= 2)
    return output


def threshold_metrics(
    probabilities: Sequence[float],
    truth: Sequence[bool],
    *,
    threshold: float,
) -> dict:
    predicted = [value >= threshold for value in probabilities]
    true_positive = sum(expected and actual for expected, actual in zip(truth, predicted))
    false_positive = sum(not expected and actual for expected, actual in zip(truth, predicted))
    false_negative = sum(expected and not actual for expected, actual in zip(truth, predicted))
    true_negative = sum(not expected and not actual for expected, actual in zip(truth, predicted))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def score_activity(
    manifest_rows: Iterable[Mapping[str, object]],
    output_records: Iterable[Mapping[str, object]],
    *,
    target_recall: float,
) -> dict:
    metadata_by_cut = {
        str(dict(row.get("metadata") or {}).get("cut_id") or ""): dict(row.get("metadata") or {})
        for row in manifest_rows
    }
    probabilities = []
    truth = []
    for record in output_records:
        cut_id = str(record.get("cut_id") or "")
        values = [float(value) for value in record.get("activity_overlap_probabilities") or []]
        frame_hz = float(record.get("activity_frame_hz") or 0.0)
        if not values or frame_hz <= 0:
            raise ValueError(f"No activity probabilities for {cut_id}")
        metadata = metadata_by_cut.get(cut_id)
        if metadata is None:
            raise ValueError(f"No activity reference for {cut_id}")
        probabilities.extend(values)
        truth.extend(
            reference_overlap_frames(
                list(metadata.get("activity") or []),
                frame_count=len(values),
                frame_hz=frame_hz,
            )
        )
    thresholds = sorted(set([index / 1000 for index in range(0, 1001)] + probabilities))
    metrics = [
        threshold_metrics(probabilities, truth, threshold=threshold) for threshold in thresholds
    ]
    best_f1 = max(metrics, key=lambda item: (item["f1"], item["recall"], item["threshold"]))
    recall_candidates = [item for item in metrics if item["recall"] >= target_recall]
    recall_operating_point = max(
        recall_candidates,
        key=lambda item: (item["precision"], item["threshold"]),
    )
    return {
        "frames": len(truth),
        "overlap_frames": sum(truth),
        "overlap_frame_fraction": sum(truth) / max(1, len(truth)),
        "probability_min": min(probabilities),
        "probability_max": max(probabilities),
        "target_recall": target_recall,
        "best_f1": best_f1,
        "target_recall_operating_point": recall_operating_point,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the mono MOSS auxiliary overlap head on held-out activity labels."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()
    model_output = json.loads(args.model_output.read_text(encoding="utf-8"))
    score = score_activity(
        _read_jsonl(args.manifest),
        list(model_output.get("records") or []),
        target_recall=args.target_recall,
    )
    payload = {
        "manifest": str(args.manifest),
        "model_output": str(args.model_output),
        "uses_reference_activity_for_calibration_only": True,
        **score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
