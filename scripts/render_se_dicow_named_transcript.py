from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from evaluate_se_dicow_oracle_cut import normalized_words


TIMESTAMP_PATTERN = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>")


def parse_timestamped_segments(text: str) -> list[dict]:
    matches = list(TIMESTAMP_PATTERN.finditer(text))
    segments = []
    for current, following in zip(matches, matches[1:]):
        content = text[current.end() : following.start()].strip()
        start = float(current.group(1))
        end = float(following.group(1))
        if content and end >= start:
            segments.append({"start": start, "end": end, "text": content})
    return segments


def clip_segment_to_duration(segment: Mapping[str, object], max_duration: float) -> dict | None:
    start = float(segment["start"])
    end = min(float(segment["end"]), max_duration)
    if start >= max_duration or end <= start:
        return None
    return {**segment, "end": end}


def _cut_start(cut_id: object) -> float:
    match = re.match(r"session_\d+_w(\d+)_c(\d+)", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot infer absolute cut time from {cut_id!r}")
    return int(match.group(1)) + int(match.group(2)) / 1000.0


def assigned_binding_evidence(
    binding_record: Mapping[str, object],
    *,
    speaker: str,
    mode: str,
) -> dict:
    mapping = dict(binding_record.get(f"{mode}_mapping") or {})
    slot = next((slot for slot, assigned in mapping.items() if assigned == speaker), None)
    if slot is None:
        return {
            "slot": None,
            "assigned_score": None,
            "assigned_margin": None,
            "binding_confidence": "review",
            "binding_reason": "unmapped-speaker",
        }
    evidence = dict(dict(binding_record.get("binding_evidence") or {}).get(slot) or {})
    scores = {str(name): float(score) for name, score in dict(evidence.get("scores") or {}).items()}
    assigned_score = scores.get(speaker)
    alternatives = [score for name, score in scores.items() if name != speaker]
    assigned_margin = (
        assigned_score - max(alternatives) if assigned_score is not None and alternatives else None
    )
    return {
        "slot": slot,
        "assigned_score": assigned_score,
        "assigned_margin": assigned_margin,
        "binding_confidence": "pending",
        "binding_reason": None,
    }


def activity_evidence(
    probabilities: np.ndarray,
    *,
    slot: str | None,
    start: float,
    end: float,
    duration: float = 30.0,
) -> dict:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if slot is None or values.ndim != 2 or values.shape[0] == 0:
        return {
            "mean_target_probability": 0.0,
            "mean_competing_probability": 0.0,
            "mean_overlap_probability": 0.0,
        }
    slot_index = int(slot.rsplit("_", 1)[-1])
    first = max(0, min(values.shape[0] - 1, int(start / duration * values.shape[0])))
    last = max(first + 1, min(values.shape[0], int(np.ceil(end / duration * values.shape[0]))))
    region = np.clip(values[first:last], 0.0, 1.0)
    target = region[:, slot_index]
    other_slots = [index for index in range(region.shape[1]) if index != slot_index]
    competing = np.max(region[:, other_slots], axis=1) if other_slots else np.zeros_like(target)
    return {
        "mean_target_probability": float(target.mean()),
        "mean_competing_probability": float(competing.mean()),
        "mean_overlap_probability": float((target * competing).mean()),
    }


def _text_similarity(left: str, right: str) -> float:
    left_words = set(normalized_words(left))
    right_words = set(normalized_words(right))
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 0.0


def mark_duplicate_hypotheses(segments: list[dict], *, threshold: float = 0.7) -> None:
    for index, left in enumerate(segments):
        for right in segments[index + 1 :]:
            if left["speaker"] == right["speaker"]:
                continue
            if min(left["end"], right["end"]) <= max(left["start"], right["start"]):
                continue
            if _text_similarity(str(left["text"]), str(right["text"])) < threshold:
                continue
            for segment in (left, right):
                reasons = segment.setdefault("review_reasons", [])
                if "duplicate-overlapping-hypothesis" not in reasons:
                    reasons.append("duplicate-overlapping-hypothesis")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render chronological named SE-DiCoW output with explicit review evidence."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--binding-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument(
        "--binding-mode", choices=("independent", "one_to_one"), default="one_to_one"
    )
    parser.add_argument("--binding-margin-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-review-threshold", type=float, default=0.25)
    parser.add_argument("--max-duration", type=float)
    args = parser.parse_args()

    binding_payload = json.loads(args.binding_json.read_text(encoding="utf-8"))
    bindings = {
        str(record.get("cut_id") or ""): record
        for record in list(binding_payload.get("records") or [])
    }
    segments = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.name == "session_aggregate_score.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "cut_id" not in result or "records" not in result:
            continue
        cut_id = str(result["cut_id"])
        cut_start = _cut_start(cut_id)
        binding = bindings[cut_id]
        probabilities = np.load(str(binding["probability_path"]), allow_pickle=False)
        for record in list(result.get("records") or []):
            speaker = str(record.get("speaker") or "")
            binding_evidence = assigned_binding_evidence(
                binding,
                speaker=speaker,
                mode=args.binding_mode,
            )
            for parsed in parse_timestamped_segments(str(record.get("raw_prediction") or "")):
                evidence = activity_evidence(
                    probabilities,
                    slot=binding_evidence["slot"],
                    start=float(parsed["start"]),
                    end=float(parsed["end"]),
                )
                reasons = []
                margin = binding_evidence["assigned_margin"]
                if margin is None:
                    reasons.append("unmapped-speaker")
                elif margin < args.binding_margin_threshold:
                    reasons.append("weak-speaker-binding")
                if evidence["mean_overlap_probability"] >= args.overlap_review_threshold:
                    reasons.append("crosstalk")
                segments.append(
                    {
                        "session": result["session"],
                        "cut_id": cut_id,
                        "start": cut_start + float(parsed["start"]),
                        "end": cut_start + float(parsed["end"]),
                        "speaker": speaker,
                        "text": parsed["text"],
                        "review_reasons": reasons,
                        **binding_evidence,
                        **evidence,
                    }
                )
    if args.max_duration is not None:
        segments = [
            clipped
            for segment in segments
            if (clipped := clip_segment_to_duration(segment, args.max_duration)) is not None
        ]
    segments.sort(key=lambda item: (item["session"], item["start"], item["end"], item["speaker"]))
    mark_duplicate_hypotheses(segments)
    for segment in segments:
        segment["requires_review"] = bool(segment["review_reasons"])
        segment["binding_confidence"] = (
            "review"
            if {"weak-speaker-binding", "unmapped-speaker"} & set(segment["review_reasons"])
            else "high"
        )
    summary = {
        "segment_count": len(segments),
        "review_segment_count": sum(segment["requires_review"] for segment in segments),
        "binding_margin_threshold": args.binding_margin_threshold,
        "overlap_review_threshold": args.overlap_review_threshold,
        "segments": segments,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = []
    for segment in segments:
        marker = (
            f" [REVIEW: {', '.join(segment['review_reasons'])}]"
            if segment["requires_review"]
            else ""
        )
        lines.append(
            f"[{segment['start']:.2f}-{segment['end']:.2f}] "
            f"{segment['speaker']}{marker}: {segment['text']}"
        )
    args.output_text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "segments"}, indent=2))


if __name__ == "__main__":
    main()
