from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from build_moss_diarization_dataset import (
    DEFAULT_PROMPT,
    _resolve_audio,
    activity_metadata,
    group_words_into_segments,
    segments_overlap_in_audio,
)


CUT_ID_RE = re.compile(r"session_(\d+)_w(\d+)_c(\d+)")


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_session_speaker_ids(
    reference_rows: Iterable[Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    first_onsets: dict[str, dict[str, float]] = defaultdict(dict)
    for row in reference_rows:
        session = str(row.get("session") or "")
        window_start = float(row.get("window_start") or 0.0)
        for word in list(row.get("words") or []):
            speaker = str(word.get("speaker") or "").strip()
            onset = window_start + float(word.get("start") or 0.0)
            if speaker:
                first_onsets[session][speaker] = min(
                    onset,
                    first_onsets[session].get(speaker, onset),
                )
    return {
        session: {
            speaker: f"S{index + 1:02d}"
            for index, (speaker, _onset) in enumerate(
                sorted(onsets.items(), key=lambda item: (item[1], item[0]))
            )
        }
        for session, onsets in first_onsets.items()
    }


def segment_is_impossible(
    segment: Mapping[str, object],
    *,
    maximum_words_per_second: float,
    minimum_words_for_rate_check: int,
) -> bool:
    word_count = len(list(segment.get("words") or []))
    duration = float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)
    return (
        word_count >= minimum_words_for_rate_check
        and word_count / max(0.01, duration) > maximum_words_per_second
    )


def segment_overlap_flags(segments: Sequence[Mapping[str, object]]) -> list[dict]:
    output = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "")
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        overlap = any(
            str(other.get("speaker") or "") != speaker and segments_overlap_in_audio(segment, other)
            for other in segments
        )
        output.append({**dict(segment), "overlap": overlap, "brief": end - start <= 2.0})
    return output


def render_weighted_target(
    segments: Sequence[Mapping[str, object]],
    speaker_ids: Mapping[str, str],
) -> tuple[str, list[dict], list[dict]]:
    parts: list[str] = []
    loss_spans = []
    activity = []
    position = 0

    def append(value: str, *, kind: str, weight: float) -> None:
        nonlocal position
        start = position
        parts.append(value)
        position += len(value)
        loss_spans.append({"start": start, "end": position, "kind": kind, "weight": weight})

    flagged = segment_overlap_flags(segments)
    for segment in sorted(
        flagged,
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
            speaker_ids[str(item["speaker"])],
        ),
    ):
        speaker_id = speaker_ids[str(segment["speaker"])]
        start = float(segment["start"])
        end = float(segment["end"])
        append(f"[{start:.2f}]", kind="timestamp", weight=2.0)
        append(f"[{speaker_id}]", kind="speaker_tag", weight=3.0)
        parts.append(" ")
        position += 1
        text_kind = "brief_overlap_word" if segment["brief"] and segment["overlap"] else "word"
        append(
            " ".join(str(word) for word in list(segment.get("words") or [])),
            kind=text_kind,
            weight=4.0 if text_kind == "brief_overlap_word" else 1.0,
        )
        append(f"[{end:.2f}]", kind="timestamp", weight=2.0)
    activity = activity_metadata(flagged, speaker_ids=speaker_ids)
    return "".join(parts), loss_spans, activity


def _cut_coordinates(cut_id: object) -> tuple[str, float, float]:
    match = CUT_ID_RE.match(str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot parse cut id {cut_id!r}")
    session, window_seconds, cut_milliseconds = (int(value) for value in match.groups())
    return f"Session {session}", float(window_seconds), cut_milliseconds / 1000.0


def build_overlap_crops(
    *,
    cutset: Path,
    forced_word_reference: Path,
    output_dir: Path,
    output_jsonl: Path,
    summary_path: Path,
    crop_seconds: float,
    maximum_words_per_second: float,
    minimum_words_for_rate_check: int,
    maximum_examples: int = 0,
) -> dict:
    reference_rows = list(_read_jsonl(forced_word_reference))
    reference_index = {
        (str(row.get("session") or ""), float(row.get("window_start") or 0.0)): row
        for row in reference_rows
    }
    speaker_ids_by_session = stable_session_speaker_ids(reference_rows)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_crops = set()
    rejected_impossible_events = 0
    total_events = 0
    brief_overlap_words = 0
    all_words = 0
    for cut in _read_jsonl(cutset):
        session, window_start, cut_start = _cut_coordinates(cut.get("id"))
        reference = reference_index.get((session, window_start))
        if reference is None:
            continue
        duration = float(cut.get("duration") or 0.0)
        words = []
        for raw_word in list(reference.get("words") or []):
            midpoint = (float(raw_word.get("start") or 0.0) + float(raw_word.get("end") or 0.0)) / 2
            if cut_start <= midpoint < cut_start + duration:
                word = dict(raw_word)
                word["start"] = max(0.0, float(word.get("start") or 0.0) - cut_start)
                word["end"] = min(duration, float(word.get("end") or 0.0) - cut_start)
                words.append(word)
        segments = group_words_into_segments(
            words,
            turn_gap_seconds=0.75,
            maximum_segment_seconds=15.0,
        )
        flagged = segment_overlap_flags(segments)
        events = [segment for segment in flagged if segment["brief"] and segment["overlap"]]
        if not events:
            continue
        audio_path = _resolve_audio(cut, cutset=cutset)
        wave, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = np.asarray(wave.mean(axis=1), dtype=np.float32)
        for event_index, event in enumerate(events):
            total_events += 1
            if segment_is_impossible(
                event,
                maximum_words_per_second=maximum_words_per_second,
                minimum_words_for_rate_check=minimum_words_for_rate_check,
            ):
                rejected_impossible_events += 1
                continue
            center = (float(event["start"]) + float(event["end"])) / 2
            crop_start = min(max(0.0, center - crop_seconds / 2), max(0.0, duration - crop_seconds))
            crop_end = min(duration, crop_start + crop_seconds)
            crop_key = (str(cut.get("id") or ""), round(crop_start, 2))
            if crop_key in seen_crops:
                continue
            crop_words = []
            for raw_word in words:
                midpoint = (float(raw_word["start"]) + float(raw_word["end"])) / 2
                if crop_start <= midpoint < crop_end:
                    word = dict(raw_word)
                    word["start"] = max(0.0, float(word["start"]) - crop_start)
                    word["end"] = min(crop_end - crop_start, float(word["end"]) - crop_start)
                    crop_words.append(word)
            crop_segments = group_words_into_segments(
                crop_words,
                turn_gap_seconds=0.75,
                maximum_segment_seconds=15.0,
            )
            if any(
                segment_is_impossible(
                    segment,
                    maximum_words_per_second=maximum_words_per_second,
                    minimum_words_for_rate_check=minimum_words_for_rate_check,
                )
                for segment in crop_segments
            ):
                rejected_impossible_events += 1
                continue
            crop_flagged = segment_overlap_flags(crop_segments)
            event_words = sum(
                len(list(segment.get("words") or []))
                for segment in crop_flagged
                if segment["brief"] and segment["overlap"]
            )
            if event_words == 0:
                continue
            target, loss_spans, activity = render_weighted_target(
                crop_segments,
                speaker_ids_by_session[session],
            )
            first = int(round(crop_start * sample_rate))
            last = min(len(mono), int(round(crop_end * sample_rate)))
            crop_wave = mono[first:last]
            output_audio = audio_dir / (
                f"{str(cut.get('id') or '')}_bo{event_index:03d}_{int(round(crop_start * 1000)):05d}.wav"
            )
            sf.write(output_audio, crop_wave, sample_rate, subtype="PCM_16")
            records.append(
                {
                    "conversation": [
                        {"role": "user", "message_type": "text", "content": DEFAULT_PROMPT},
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": str(output_audio.resolve()),
                        },
                        {"role": "assistant", "message_type": "text", "content": target},
                    ],
                    "metadata": {
                        "cut_id": f"{cut.get('id')}-brief-overlap-{event_index:03d}",
                        "source_cut_id": str(cut.get("id") or ""),
                        "session": session,
                        "crop_start": crop_start,
                        "crop_duration": len(crop_wave) / sample_rate,
                        "word_count": len(crop_words),
                        "brief_overlap_word_count": event_words,
                        "brief_overlap_word_fraction": event_words / max(1, len(crop_words)),
                        "speaker_count": len({word["speaker"] for word in crop_words}),
                        "mono_input_only": True,
                        "target_source": "quality-filtered-forced-word-reference",
                        "loss_spans": loss_spans,
                        "activity": activity,
                        "stable_session_speaker_ids": speaker_ids_by_session[session],
                    },
                }
            )
            seen_crops.add(crop_key)
            brief_overlap_words += event_words
            all_words += len(crop_words)
            if maximum_examples > 0 and len(records) >= maximum_examples:
                break
        if maximum_examples > 0 and len(records) >= maximum_examples:
            break
    if not records:
        raise ValueError("No quality-filtered brief-overlap crops were built")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "cutset": str(cutset),
        "forced_word_reference": str(forced_word_reference),
        "output_jsonl": str(output_jsonl),
        "crop_seconds": crop_seconds,
        "total_brief_overlap_events": total_events,
        "rejected_impossible_events_or_crops": rejected_impossible_events,
        "examples": len(records),
        "brief_overlap_words": brief_overlap_words,
        "all_target_words": all_words,
        "brief_overlap_word_fraction": brief_overlap_words / max(1, all_words),
        "maximum_words_per_second": maximum_words_per_second,
        "minimum_words_for_rate_check": minimum_words_for_rate_check,
        "mono_input_only": True,
        "clean_stem_role": "offline ownership/timing labels only",
        "serialization": "onset order, deterministic speaker-ID tie break",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build quality-filtered mono crops centered on real brief interruptions."
    )
    parser.add_argument("--cutset", type=Path, required=True)
    parser.add_argument("--forced-word-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--maximum-words-per-second", type=float, default=6.0)
    parser.add_argument("--minimum-words-for-rate-check", type=int, default=4)
    parser.add_argument("--maximum-examples", type=int, default=0)
    args = parser.parse_args()
    build_overlap_crops(
        cutset=args.cutset,
        forced_word_reference=args.forced_word_reference,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
        crop_seconds=args.crop_seconds,
        maximum_words_per_second=args.maximum_words_per_second,
        minimum_words_for_rate_check=args.minimum_words_for_rate_check,
        maximum_examples=args.maximum_examples,
    )


if __name__ == "__main__":
    main()
