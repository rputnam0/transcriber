from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


MASK_PROFILES = {
    "mild": {
        "boundary_jitter": 0.24,
        "drop_probability": 0.04,
        "short_drop_probability": 0.10,
        "leak_probability": 0.05,
        "max_leak_seconds": 0.65,
    },
    "strong": {
        "boundary_jitter": 0.55,
        "drop_probability": 0.10,
        "short_drop_probability": 0.22,
        "leak_probability": 0.12,
        "max_leak_seconds": 1.10,
    },
}


def _as_span(supervision: object) -> dict:
    if isinstance(supervision, Mapping):
        start = float(supervision["start"])
        duration = float(
            supervision.get(
                "duration",
                float(supervision.get("end", start)) - start,
            )
        )
        return {
            "speaker": str(supervision["speaker"]),
            "start": start,
            "end": start + duration,
            "text": str(supervision.get("text") or "").strip(),
        }
    return {
        "speaker": str(supervision.speaker),
        "start": float(supervision.start),
        "end": float(supervision.end),
        "text": str(supervision.text or "").strip(),
    }


def _merge_regions(
    regions: Sequence[tuple[float, float]], *, max_gap: float
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(regions):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def corrupt_activity_supervisions(
    supervisions: Sequence[object],
    *,
    duration: float,
    profile: str,
    rng: random.Random,
) -> list[dict]:
    """Create imperfect activity masks while retaining each speaker's complete transcript."""
    if profile not in MASK_PROFILES:
        raise ValueError(f"Unknown mask corruption profile: {profile}")
    config = MASK_PROFILES[profile]
    spans = [_as_span(supervision) for supervision in supervisions]
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        by_speaker[span["speaker"]].append(span)

    corrupted: dict[str, list[tuple[float, float]]] = {}
    for speaker, speaker_spans in by_speaker.items():
        regions = []
        for span in sorted(speaker_spans, key=lambda item: (item["start"], item["end"])):
            span_duration = span["end"] - span["start"]
            drop_probability = float(config["drop_probability"])
            if span_duration < 0.8:
                drop_probability = float(config["short_drop_probability"])
            if rng.random() < drop_probability:
                continue
            start = span["start"] + rng.uniform(
                -float(config["boundary_jitter"]), float(config["boundary_jitter"])
            )
            end = span["end"] + rng.uniform(
                -float(config["boundary_jitter"]), float(config["boundary_jitter"])
            )
            start = min(duration, max(0.0, start))
            end = min(duration, max(start + 0.08, end))
            if end > start:
                regions.append((start, end))

        # A target-speaker ASR pass requires at least a small enrollment/activity cue.
        if not regions:
            anchor = max(speaker_spans, key=lambda item: item["end"] - item["start"])
            midpoint = (anchor["start"] + anchor["end"]) / 2.0
            regions = [(max(0.0, midpoint - 0.08), min(duration, midpoint + 0.08))]
        corrupted[speaker] = regions

    speakers = sorted(by_speaker)
    for speaker in speakers:
        other_regions = [
            (span["start"], span["end"])
            for other_speaker, other_spans in by_speaker.items()
            if other_speaker != speaker
            for span in other_spans
        ]
        if other_regions and rng.random() < float(config["leak_probability"]):
            start, end = rng.choice(other_regions)
            leak_duration = min(end - start, float(config["max_leak_seconds"]))
            if leak_duration > 0.08:
                leak_start = rng.uniform(start, max(start, end - leak_duration))
                corrupted[speaker].append((leak_start, min(duration, leak_start + leak_duration)))

    output = []
    for speaker in speakers:
        source_spans = sorted(by_speaker[speaker], key=lambda item: (item["start"], item["end"]))
        transcript = " ".join(span["text"] for span in source_spans if span["text"])
        regions = _merge_regions(corrupted[speaker], max_gap=0.10)
        for index, (start, end) in enumerate(regions):
            output.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": transcript if index == 0 else "",
                }
            )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


def activity_iou(
    reference: Sequence[object],
    candidate: Sequence[object],
    *,
    duration: float,
    frame_seconds: float = 0.08,
) -> float:
    frame_count = max(1, round(duration / frame_seconds))

    def frame_set(items: Sequence[object]) -> set[tuple[str, int]]:
        active = set()
        for item in items:
            span = _as_span(item)
            start = max(0, int(span["start"] / frame_seconds))
            end = min(frame_count, max(start + 1, int(-(-span["end"] // frame_seconds))))
            active.update((span["speaker"], frame) for frame in range(start, end))
        return active

    reference_frames = frame_set(reference)
    candidate_frames = frame_set(candidate)
    union = reference_frames | candidate_frames
    return len(reference_frames & candidate_frames) / len(union) if union else 1.0


def inherit_transcript_spans(
    custom: Mapping[str, object] | None,
    transcript_spans: Sequence[Mapping[str, object]],
) -> dict:
    output = dict(custom or {})
    if not output.get("transcript_spans") and transcript_spans:
        output["transcript_spans"] = [dict(span) for span in transcript_spans]
    return output


def augment_cutset(
    *,
    input_cuts: Path,
    output_cuts: Path,
    variants: Sequence[str],
    seed: int,
    append_cuts: Sequence[Path] = (),
) -> dict:
    from lhotse import CutSet, SupervisionSegment
    from lhotse.utils import fastcopy

    cuts = CutSet.from_file(input_cuts)
    augmented = []
    ious = defaultdict(list)
    for cut_index, cut in enumerate(cuts):
        source = [_as_span(supervision) for supervision in cut.supervisions]
        for variant_index, variant in enumerate(variants):
            if variant == "clean":
                spans = source
            else:
                spans = corrupt_activity_supervisions(
                    source,
                    duration=cut.duration,
                    profile=variant,
                    rng=random.Random(seed + cut_index * 1009 + variant_index * 9176),
                )
            variant_id = f"{cut.id}-mask-{variant}"
            new_supervisions = [
                SupervisionSegment(
                    id=f"{variant_id}-sup{index:04d}",
                    recording_id=cut.recording_id,
                    start=float(span["start"]),
                    duration=max(0.01, float(span["end"]) - float(span["start"])),
                    channel=cut.channel,
                    text=str(span["text"]),
                    speaker=str(span["speaker"]),
                    language="en",
                )
                for index, span in enumerate(spans)
            ]
            custom = dict(cut.custom or {})
            custom["activity_mask_source"] = (
                "isolated-track-teacher" if variant == "clean" else f"synthetic-{variant}"
            )
            augmented.append(
                fastcopy(cut, id=variant_id, supervisions=new_supervisions, custom=custom)
            )
            ious[variant].append(activity_iou(source, spans, duration=cut.duration))

    appended_counts = {}
    transcript_spans_by_recording = {
        cut.recording_id: list(dict(cut.custom or {}).get("transcript_spans") or []) for cut in cuts
    }
    for path in append_cuts:
        extra = CutSet.from_file(path)
        extra_count = len(extra)
        for cut in extra:
            custom = inherit_transcript_spans(
                cut.custom,
                transcript_spans_by_recording.get(cut.recording_id, []),
            )
            augmented.append(fastcopy(cut, custom=custom))
        appended_counts[str(path)] = extra_count

    output_cuts.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(augmented).to_file(output_cuts)
    summary = {
        "input_cuts": str(input_cuts),
        "output_cuts": str(output_cuts),
        "source_cuts": len(cuts),
        "augmented_cuts": len(augmented),
        "variants": list(variants),
        "appended_cuts": appended_counts,
        "mean_activity_iou": {
            variant: sum(values) / len(values) for variant, values in sorted(ious.items())
        },
        "seed": seed,
    }
    output_cuts.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment target-speaker ASR cuts with mono-realistic activity-mask errors."
    )
    parser.add_argument("--input-cuts", type=Path, required=True)
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument("--variants", default="clean,mild,strong")
    parser.add_argument("--append-cuts", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unknown = sorted(set(variants) - {"clean", *MASK_PROFILES})
    if unknown:
        parser.error(f"unknown mask variants: {', '.join(unknown)}")
    augment_cutset(
        input_cuts=args.input_cuts,
        output_cuts=args.output_cuts,
        variants=variants,
        seed=args.seed,
        append_cuts=args.append_cuts,
    )


if __name__ == "__main__":
    main()
