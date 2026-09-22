from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CUT_ID_RE = re.compile(r"session_(\d+)_w(\d+)_c(\d+)")
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _cut_coordinates(cut: Mapping[str, object]) -> tuple[str, float, float]:
    match = CUT_ID_RE.match(str(cut.get("id") or ""))
    if not match:
        raise ValueError(f"Cannot parse cut id {cut.get('id')!r}")
    session, window_seconds, cut_milliseconds = (int(value) for value in match.groups())
    return f"Session {session}", float(window_seconds), cut_milliseconds / 1000.0


def _resolve_audio(cut: Mapping[str, object], *, cutset: Path) -> Path:
    recording = dict(cut.get("recording") or {})
    sources = list(recording.get("sources") or [])
    if not sources:
        raise ValueError(f"Cut {cut.get('id')} has no recording source")
    path = Path(str(sources[0].get("source") or ""))
    for candidate in (path, Path.cwd() / path, cutset.resolve().parent / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def group_words_into_segments(
    words: Sequence[Mapping[str, object]],
    *,
    turn_gap_seconds: float,
    maximum_segment_seconds: float,
) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for raw_word in words:
        word = dict(raw_word)
        speaker = str(word.get("speaker") or "").strip()
        text = str(word.get("text") or "").strip()
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        if speaker and text and end > start:
            by_speaker[speaker].append(
                {"speaker": speaker, "start": start, "end": end, "text": text}
            )

    segments = []
    for speaker, speaker_words in sorted(by_speaker.items()):
        current = None
        for word in sorted(speaker_words, key=lambda item: (item["start"], item["end"])):
            starts_new = (
                current is None
                or word["start"] - current["end"] > turn_gap_seconds
                or word["end"] - current["start"] > maximum_segment_seconds
            )
            if starts_new:
                if current is not None:
                    segments.append(current)
                current = {
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word["text"]],
                    "word_spans": [word],
                }
            else:
                current["end"] = max(current["end"], word["end"])
                current["words"].append(word["text"])
                current["word_spans"].append(word)
        if current is not None:
            segments.append(current)

    return sorted(segments, key=lambda item: (item["start"], item["end"], item["speaker"]))


def speech_intervals(segment: Mapping[str, object]) -> list[tuple[float, float]]:
    detailed = list(segment.get("word_spans") or segment.get("word_intervals") or [])
    intervals = [
        (float(item.get("start") or 0.0), float(item.get("end") or 0.0))
        for item in detailed
        if float(item.get("end") or 0.0) > float(item.get("start") or 0.0)
    ]
    if intervals:
        return intervals
    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or start)
    return [(start, end)] if end > start else []


def segments_overlap_in_audio(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    return any(
        first_start < second_end and second_start < first_end
        for first_start, first_end in speech_intervals(first)
        for second_start, second_end in speech_intervals(second)
    )


def anonymous_target(segments: Sequence[Mapping[str, object]], *, duration: float) -> str:
    target, _loss_spans, _brief_overlap_words = anonymous_target_with_loss_spans(
        segments,
        duration=duration,
    )
    return target


def anonymous_target_with_loss_spans(
    segments: Sequence[Mapping[str, object]],
    *,
    duration: float,
    brief_turn_seconds: float = 2.0,
) -> tuple[str, list[dict], int]:
    speaker_ids = anonymous_speaker_ids(segments)
    parts = []
    loss_spans = []
    brief_overlap_words = 0
    for index, segment in enumerate(segments):
        speaker = str(segment["speaker"])
        start = min(duration, max(0.0, float(segment["start"])))
        end = min(duration, max(start, float(segment["end"])))
        text = " ".join(str(word) for word in segment["words"]).strip()
        if end <= start or not text:
            continue
        overlaps = any(
            other_index != index
            and str(other["speaker"]) != speaker
            and segments_overlap_in_audio(segment, other)
            for other_index, other in enumerate(segments)
        )
        brief_overlap = end - start <= brief_turn_seconds and overlaps
        start_tag = f"[{start:.2f}]"
        speaker_tag = f"[{speaker_ids[speaker]}]"
        end_tag = f"[{end:.2f}]"
        serialized = (
            (start_tag, "timestamp", 2.0),
            (speaker_tag, "speaker_tag", 3.0),
            (" ", None, 0.0),
            (
                text,
                "brief_overlap_word" if brief_overlap else "word",
                4.0 if brief_overlap else 1.0,
            ),
            (end_tag, "timestamp", 2.0),
        )
        for value, kind, weight in serialized:
            span_start = sum(len(part) for part in parts)
            parts.append(value)
            if kind is not None:
                loss_spans.append(
                    {
                        "start": span_start,
                        "end": span_start + len(value),
                        "kind": kind,
                        "weight": weight,
                    }
                )
        if brief_overlap:
            brief_overlap_words += len(segment["words"])
    return "".join(parts), loss_spans, brief_overlap_words


def anonymous_speaker_ids(
    segments: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    output = {}
    for segment in sorted(
        segments,
        key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])),
    ):
        speaker = str(segment["speaker"])
        if speaker not in output:
            output[speaker] = f"S{len(output) + 1:02d}"
    return output


def activity_metadata(
    segments: Sequence[Mapping[str, object]],
    *,
    speaker_ids: Mapping[str, str] | None = None,
    maximum_merge_gap_seconds: float = 0.08,
) -> list[dict]:
    resolved_ids = dict(speaker_ids or anonymous_speaker_ids(segments))
    intervals_by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        intervals_by_speaker[str(segment["speaker"])].extend(speech_intervals(segment))

    output = []
    for speaker, raw_intervals in intervals_by_speaker.items():
        merged: list[list[float]] = []
        for start, end in sorted(raw_intervals):
            if merged and start - merged[-1][1] <= maximum_merge_gap_seconds:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        speaker_id = resolved_ids[speaker]
        output.extend(
            {
                "speaker": speaker_id,
                "speaker_index": int(speaker_id[1:]) - 1,
                "start": start,
                "end": end,
            }
            for start, end in merged
        )
    return sorted(
        output,
        key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])),
    )


def reference_segment_metadata(segments: Sequence[Mapping[str, object]]) -> list[dict]:
    speaker_ids = anonymous_speaker_ids(segments)
    output = []
    for segment in segments:
        speaker = str(segment["speaker"])
        brief = float(segment["end"]) - float(segment["start"]) <= 2.0
        word_spans = []
        for word in list(segment.get("word_spans") or []):
            start = float(word.get("start") or 0.0)
            end = float(word.get("end") or start)
            word_spans.append(
                {
                    "text": str(word.get("text") or ""),
                    "start": start,
                    "end": end,
                    "overlap": any(
                        str(other["speaker"]) != speaker
                        and any(
                            start < other_end and other_start < end
                            for other_start, other_end in speech_intervals(other)
                        )
                        for other in segments
                    ),
                    "brief": brief,
                }
            )
        output.append(
            {
                "speaker": speaker_ids[speaker],
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "overlap": (
                    any(word["overlap"] for word in word_spans)
                    if word_spans
                    else any(
                        str(other["speaker"]) != speaker
                        and segments_overlap_in_audio(segment, other)
                        for other in segments
                    )
                ),
                "brief": brief,
                "word_spans": word_spans,
            }
        )
    return output


def _has_overlap(segments: Sequence[Mapping[str, object]]) -> bool:
    return any(
        first["speaker"] != second["speaker"] and segments_overlap_in_audio(first, second)
        for index, first in enumerate(segments)
        for second in segments[index + 1 :]
    )


def build_dataset(
    *,
    cutset: Path,
    forced_word_reference: Path,
    output_jsonl: Path,
    summary_path: Path,
    prompt: str,
    excluded_sessions: set[str],
    minimum_words: int,
    turn_gap_seconds: float,
    maximum_segment_seconds: float,
    overlap_repeat: int,
) -> dict:
    reference_index = {
        (str(row.get("session") or ""), float(row.get("window_start") or 0.0)): row
        for row in _read_jsonl(forced_word_reference)
    }
    records = []
    skipped_missing_reference = 0
    skipped_few_words = 0
    source_examples = 0
    overlap_examples = 0
    sessions = set()
    for cut in _read_jsonl(cutset):
        session, window_start, cut_start = _cut_coordinates(cut)
        if session in excluded_sessions:
            continue
        reference = reference_index.get((session, window_start))
        if reference is None:
            skipped_missing_reference += 1
            continue
        duration = float(cut.get("duration") or 0.0)
        cut_end = cut_start + duration
        words = []
        for raw_word in list(reference.get("words") or []):
            midpoint = (float(raw_word.get("start") or 0.0) + float(raw_word.get("end") or 0.0)) / 2
            if not cut_start <= midpoint < cut_end:
                continue
            word = dict(raw_word)
            word["start"] = max(0.0, float(word.get("start") or 0.0) - cut_start)
            word["end"] = min(duration, float(word.get("end") or 0.0) - cut_start)
            words.append(word)
        if len(words) < minimum_words:
            skipped_few_words += 1
            continue
        segments = group_words_into_segments(
            words,
            turn_gap_seconds=turn_gap_seconds,
            maximum_segment_seconds=maximum_segment_seconds,
        )
        target, loss_spans, brief_overlap_word_count = anonymous_target_with_loss_spans(
            segments,
            duration=duration,
        )
        if not target:
            skipped_few_words += 1
            continue
        audio = _resolve_audio(cut, cutset=cutset)
        conversation = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio)},
            {"role": "assistant", "message_type": "text", "content": target},
        ]
        has_overlap = _has_overlap(segments)
        stable_session_speaker_ids = anonymous_speaker_ids(segments)
        repeats = overlap_repeat if has_overlap else 1
        for repeat in range(repeats):
            records.append(
                {
                    "conversation": conversation,
                    "metadata": {
                        "cut_id": str(cut.get("id") or ""),
                        "session": session,
                        "word_count": len(words),
                        "speaker_count": len({word["speaker"] for word in words}),
                        "has_overlap": has_overlap,
                        "brief_overlap_word_count": brief_overlap_word_count,
                        "brief_overlap_word_fraction": brief_overlap_word_count
                        / max(1, len(words)),
                        "overlap_repeat_index": repeat,
                        "mono_input_only": True,
                        "target_source": "forced-word-reference-from-labeled-stems",
                        "loss_spans": loss_spans,
                        "activity": activity_metadata(segments),
                        "activity_supervised": True,
                        "activity_valid_start": 0.0,
                        "activity_valid_end": duration,
                        "reference_segments": reference_segment_metadata(segments),
                        "stable_session_speaker_ids": stable_session_speaker_ids,
                    },
                }
            )
        source_examples += 1
        overlap_examples += int(has_overlap)
        sessions.add(session)

    if not records:
        raise ValueError("No MOSS training examples were built")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "cutset": str(cutset),
        "forced_word_reference": str(forced_word_reference),
        "output_jsonl": str(output_jsonl),
        "prompt": prompt,
        "sessions": sorted(sessions),
        "excluded_sessions": sorted(excluded_sessions),
        "source_examples": source_examples,
        "training_records": len(records),
        "source_overlap_examples": overlap_examples,
        "source_overlap_fraction": overlap_examples / max(1, source_examples),
        "overlap_repeat": overlap_repeat,
        "skipped_missing_reference": skipped_missing_reference,
        "skipped_few_words": skipped_few_words,
        "mono_input_only": True,
        "clean_stem_role": "offline target authoring only",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build joint MOSS ASR/diarization data from mono cuts and word-level labels."
    )
    parser.add_argument("--cutset", type=Path, required=True)
    parser.add_argument("--forced-word-reference", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--exclude-sessions", default="Session 34")
    parser.add_argument("--minimum-words", type=int, default=4)
    parser.add_argument("--turn-gap-seconds", type=float, default=0.75)
    parser.add_argument("--maximum-segment-seconds", type=float, default=15.0)
    parser.add_argument("--overlap-repeat", type=int, default=2)
    args = parser.parse_args()
    if args.overlap_repeat < 1:
        parser.error("--overlap-repeat must be positive")
    build_dataset(
        cutset=args.cutset,
        forced_word_reference=args.forced_word_reference,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
        prompt=args.prompt,
        excluded_sessions={
            value.strip() for value in args.exclude_sessions.split(",") if value.strip()
        },
        minimum_words=args.minimum_words,
        turn_gap_seconds=args.turn_gap_seconds,
        maximum_segment_seconds=args.maximum_segment_seconds,
        overlap_repeat=args.overlap_repeat,
    )


if __name__ == "__main__":
    main()
