from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping, Sequence


CUT_ID_RE = re.compile(r"session_\d+_w\d+_c(?P<milliseconds>\d+)")


def cut_offset_seconds(cut_id: str) -> float:
    match = CUT_ID_RE.match(cut_id)
    if not match:
        raise ValueError(f"Cannot parse cut offset from {cut_id!r}")
    return int(match.group("milliseconds")) / 1000.0


def assigned_binding_evidence(
    binding: Mapping[str, object],
    *,
    stream: str,
    mode: str,
) -> dict:
    mapping = dict(binding.get(f"{mode}_mapping") or {})
    speaker = str(mapping.get(stream) or "").strip()
    identity = dict(dict(binding.get("identity_evidence") or {}).get(stream) or {})
    if identity.get("source") == "long-transcript-lexical-anchor":
        return {
            "speaker": speaker or None,
            "assigned_score": float(identity.get("lexical_coverage") or 0.0),
            "assigned_margin": float(identity.get("lexical_margin") or 0.0),
            "scores": {},
            "identity_source": "long-transcript-lexical-anchor",
        }
    evidence = dict(dict(binding.get("binding_evidence") or {}).get(stream) or {})
    scores = {str(key): float(value) for key, value in dict(evidence.get("scores") or {}).items()}
    assigned_score = scores.get(speaker)
    alternatives = [value for name, value in scores.items() if name != speaker]
    assigned_margin = (
        assigned_score - max(alternatives) if assigned_score is not None and alternatives else None
    )
    return {
        "speaker": speaker or None,
        "assigned_score": assigned_score,
        "assigned_margin": assigned_margin,
        "scores": scores,
        "identity_source": "historical-enrollment",
    }


def segment_overlap_probability(
    record: Mapping[str, object],
    *,
    start: float,
    end: float,
) -> float | None:
    probabilities = [float(value) for value in record.get("activity_overlap_probabilities") or []]
    frame_hz = float(record.get("activity_frame_hz") or 0.0)
    if not probabilities or frame_hz <= 0:
        return None
    first = max(0, int(start * frame_hz))
    last = min(len(probabilities), max(first + 1, int(end * frame_hz + 0.999)))
    values = probabilities[first:last]
    return max(values) if values else None


def stream_is_overlap_only(binding: Mapping[str, object], stream: str) -> bool:
    metadata = next(
        (
            dict(item)
            for item in list(binding.get("stream_metadata") or [])
            if str(item.get("speaker") or "") == stream
        ),
        {},
    )
    intervals = list(metadata.get("selected_intervals") or [])
    return bool(intervals) and all(bool(interval.get("overlapped")) for interval in intervals)


def render_named_segments(
    records: Sequence[Mapping[str, object]],
    bindings: Sequence[Mapping[str, object]],
    *,
    mode: str,
    binding_margin_threshold: float,
    binding_score_threshold: float,
    overlap_review_threshold: float,
) -> list[dict]:
    bindings_by_cut = {str(binding.get("cut_id") or ""): binding for binding in bindings}
    output = []
    for record in records:
        cut_id = str(record.get("cut_id") or "")
        binding = bindings_by_cut.get(cut_id, {})
        offset = cut_offset_seconds(cut_id)
        for raw_segment in list(record.get("segments") or []):
            segment = dict(raw_segment)
            stream = str(segment.get("speaker") or "")
            local_start = float(segment.get("start") or 0.0)
            local_end = float(segment.get("end") or local_start)
            evidence = assigned_binding_evidence(binding, stream=stream, mode=mode)
            overlap_probability = segment_overlap_probability(
                record,
                start=local_start,
                end=local_end,
            )
            review_reasons = []
            if not evidence["speaker"]:
                review_reasons.append("unresolved-speaker")
            if (
                evidence["assigned_score"] is None
                or evidence["assigned_score"] < binding_score_threshold
            ):
                review_reasons.append("low-speaker-score")
            if (
                evidence["assigned_margin"] is None
                or evidence["assigned_margin"] < binding_margin_threshold
            ):
                review_reasons.append("low-speaker-margin")
            overlap_only = stream_is_overlap_only(binding, stream)
            if overlap_only and evidence["identity_source"] != "long-transcript-lexical-anchor":
                review_reasons.append("overlap-only-identity-evidence")
            if (
                overlap_probability is not None
                and overlap_probability >= overlap_review_threshold
                and review_reasons
            ):
                review_reasons.append("ambiguous-crosstalk")
            output.append(
                {
                    "cut_id": cut_id,
                    "start": offset + local_start,
                    "end": offset + local_end,
                    "speaker": evidence["speaker"] or "unknown",
                    "speaker_raw": stream,
                    "text": str(segment.get("text") or "").strip(),
                    "binding_score": evidence["assigned_score"],
                    "binding_margin": evidence["assigned_margin"],
                    "overlap_probability": overlap_probability,
                    "overlap_only_identity_evidence": overlap_only,
                    "identity_source": evidence["identity_source"],
                    "confidence": "low" if review_reasons else "high",
                    "needs_review": bool(review_reasons),
                    "review_reasons": review_reasons,
                }
            )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


def _clock(seconds: float) -> str:
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a globally timestamped, review-aware named MOSS transcript."
    )
    parser.add_argument("--moss-output", type=Path, required=True)
    parser.add_argument("--binding-output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument(
        "--binding-mode", choices=("independent", "one_to_one"), default="one_to_one"
    )
    parser.add_argument("--binding-margin-threshold", type=float, default=0.25)
    parser.add_argument("--binding-score-threshold", type=float, default=0.4)
    parser.add_argument("--overlap-review-threshold", type=float, default=0.665)
    args = parser.parse_args()

    moss = json.loads(args.moss_output.read_text(encoding="utf-8"))
    binding = json.loads(args.binding_output.read_text(encoding="utf-8"))
    if bool(binding.get("inference_uses_isolated_target_audio")):
        raise ValueError("Binding output used isolated evaluation audio")
    provenance = dict(binding.get("enrollment_provenance") or {})
    if bool(provenance.get("uses_evaluation_session_audio", True)):
        raise ValueError("Binding output used evaluation-session enrollment")
    segments = render_named_segments(
        list(moss.get("records") or []),
        list(binding.get("record_bindings") or []),
        mode=args.binding_mode,
        binding_margin_threshold=args.binding_margin_threshold,
        binding_score_threshold=args.binding_score_threshold,
        overlap_review_threshold=args.overlap_review_threshold,
    )
    payload = {
        "moss_output": str(args.moss_output),
        "binding_output": str(args.binding_output),
        "mono_only_inference": True,
        "historical_cross_session_enrollment_only": True,
        "segment_count": len(segments),
        "review_segment_count": sum(segment["needs_review"] for segment in segments),
        "segments": segments,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = []
    for segment in segments:
        review = " [REVIEW]" if segment["needs_review"] else ""
        lines.append(
            f"{segment['speaker']} {_clock(segment['start'])}-{_clock(segment['end'])}"
            f"{review}: {segment['text']}"
        )
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "segment_count": payload["segment_count"],
                "review_segment_count": payload["review_segment_count"],
                "output_json": str(args.output_json),
                "output_text": str(args.output_text),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
