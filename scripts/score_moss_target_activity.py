from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

from score_moss_activity_head import threshold_metrics


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def target_activity_frames(
    activity: Sequence[Mapping[str, object]],
    *,
    frame_count: int,
    frame_hz: float,
    valid_start: float,
    valid_end: float,
) -> tuple[list[float], list[bool]]:
    indices = [
        frame for frame in range(frame_count) if valid_start <= (frame + 0.5) / frame_hz < valid_end
    ]
    truth = []
    for frame in indices:
        center = (frame + 0.5) / frame_hz
        truth.append(
            any(
                float(interval.get("start") or 0.0) <= center < float(interval.get("end") or 0.0)
                for interval in activity
            )
        )
    return indices, truth


def _operating_points(
    probabilities: Sequence[float],
    truth: Sequence[bool],
    *,
    target_recall: float,
) -> dict:
    thresholds = sorted(set([index / 1000 for index in range(1001)] + list(probabilities)))
    metrics = [
        threshold_metrics(probabilities, truth, threshold=threshold) for threshold in thresholds
    ]
    best_f1 = max(metrics, key=lambda item: (item["f1"], item["recall"], item["threshold"]))
    recall_candidates = [item for item in metrics if item["recall"] >= target_recall]
    target_point = max(
        recall_candidates,
        key=lambda item: (item["precision"], item["threshold"]),
    )
    return {"best_f1": best_f1, "target_recall_operating_point": target_point}


def score_target_activity(
    manifest_rows: Iterable[Mapping[str, object]],
    output_records: Iterable[Mapping[str, object]],
    *,
    target_recall: float,
    presence_top_frames: int,
) -> dict:
    metadata_by_cut = {
        str(dict(row.get("metadata") or {}).get("cut_id") or ""): dict(row.get("metadata") or {})
        for row in manifest_rows
    }
    frame_probabilities = []
    frame_truth = []
    presence_probabilities = []
    presence_truth = []
    records = []
    for record in output_records:
        cut_id = str(record.get("cut_id") or "")
        metadata = metadata_by_cut.get(cut_id)
        if metadata is None:
            raise ValueError(f"No target activity reference for {cut_id}")
        probabilities = [
            float(value) for value in record.get("target_activity_probabilities") or []
        ]
        frame_hz = float(record.get("target_activity_frame_hz") or 0.0)
        if not probabilities or frame_hz <= 0:
            raise ValueError(f"No target activity probabilities for {cut_id}")
        valid_start = float(
            metadata.get("activity_valid_start") or metadata.get("timestamp_shift") or 0.0
        )
        valid_end = float(metadata.get("activity_valid_end") or 0.0)
        if valid_end <= valid_start:
            valid_end = len(probabilities) / frame_hz
        indices, expected = target_activity_frames(
            list(metadata.get("activity") or []),
            frame_count=len(probabilities),
            frame_hz=frame_hz,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        valid_probabilities = [probabilities[index] for index in indices]
        if not valid_probabilities:
            raise ValueError(f"No valid mono frames for {cut_id}")
        top_count = min(max(1, presence_top_frames), len(valid_probabilities))
        presence_score = mean(sorted(valid_probabilities, reverse=True)[:top_count])
        target_present = bool(metadata.get("activity"))
        frame_probabilities.extend(valid_probabilities)
        frame_truth.extend(expected)
        presence_probabilities.append(presence_score)
        presence_truth.append(target_present)
        records.append(
            {
                "cut_id": cut_id,
                "negative": not target_present,
                "valid_frames": len(valid_probabilities),
                "active_frames": sum(expected),
                "presence_score": presence_score,
                "probability_mean": mean(valid_probabilities),
                "probability_max": max(valid_probabilities),
            }
        )
    positive_scores = [
        score for score, expected in zip(presence_probabilities, presence_truth) if expected
    ]
    negative_scores = [
        score for score, expected in zip(presence_probabilities, presence_truth) if not expected
    ]
    return {
        "records": len(records),
        "positive_records": sum(presence_truth),
        "negative_records": len(presence_truth) - sum(presence_truth),
        "valid_mono_frames": len(frame_truth),
        "target_active_frames": sum(frame_truth),
        "presence_top_frames": presence_top_frames,
        "positive_presence_score_mean": mean(positive_scores) if positive_scores else 0.0,
        "negative_presence_score_mean": mean(negative_scores) if negative_scores else 0.0,
        "frame_activity": _operating_points(
            frame_probabilities,
            frame_truth,
            target_recall=target_recall,
        ),
        "target_presence": _operating_points(
            presence_probabilities,
            presence_truth,
            target_recall=target_recall,
        ),
        "records_detail": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate enrollment-conditioned target activity on mono-only frames."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--presence-top-frames", type=int, default=8)
    args = parser.parse_args()
    model_output = json.loads(args.model_output.read_text(encoding="utf-8"))
    score = score_target_activity(
        _read_jsonl(args.manifest),
        list(model_output.get("records") or []),
        target_recall=args.target_recall,
        presence_top_frames=args.presence_top_frames,
    )
    payload = {
        "manifest": str(args.manifest),
        "model_output": str(args.model_output),
        "uses_reference_activity_for_calibration_only": True,
        "excludes_enrollment_prefix": True,
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
