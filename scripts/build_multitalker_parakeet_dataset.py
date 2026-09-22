from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _word_text(word: Mapping[str, object]) -> str:
    return str(word.get("text") or word.get("normalized") or "").strip()


def resolve_audio_path(
    raw_path: object,
    *,
    dataset_jsonl: Path,
    audio_root: Path | None,
) -> Path:
    path = Path(str(raw_path or ""))
    candidates = [path]
    if not path.is_absolute():
        if audio_root is not None:
            candidates.append(audio_root / path)
        candidates.extend(parent / path for parent in dataset_jsonl.resolve().parents)
        candidates.append(dataset_jsonl.resolve().parent / "audio" / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def group_speaker_words(
    words: Sequence[Mapping[str, object]],
    *,
    max_gap_seconds: float,
    collar_seconds: float,
    clip_duration: float,
    clip_offset: float = 0.0,
) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    by_source_span: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for word in words:
        speaker = str(word.get("speaker") or "").strip()
        text = _word_text(word)
        if not speaker or not text:
            continue
        item = {
            "speaker": speaker,
            "start": float(word.get("start") or 0.0),
            "end": float(word.get("end") or word.get("start") or 0.0),
            "text": text,
        }
        source_start = word.get("source_span_start")
        source_end = word.get("source_span_end")
        if source_start is not None and source_end is not None:
            by_source_span[(speaker, float(source_start), float(source_end))].append(item)
        else:
            by_speaker[speaker].append(item)

    spans = []
    for (speaker, source_start, source_end), span_words in sorted(by_source_span.items()):
        span_words.sort(key=lambda item: (item["start"], item["end"]))
        relative_start = source_start - clip_offset
        relative_end = source_end - clip_offset
        if relative_end <= 0.0 or relative_start >= clip_duration:
            relative_start = min(word["start"] for word in span_words)
            relative_end = max(word["end"] for word in span_words)
        spans.append(
            {
                "speaker": speaker,
                "start": relative_start,
                "end": relative_end,
                "words": [word["text"] for word in span_words],
            }
        )

    for speaker, speaker_words in sorted(by_speaker.items()):
        speaker_words.sort(key=lambda item: (item["start"], item["end"]))
        current = None
        for word in speaker_words:
            if current is None or word["start"] > current["end"] + max_gap_seconds:
                if current is not None:
                    spans.append(current)
                current = {
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word["text"]],
                }
            else:
                current["end"] = max(current["end"], word["end"])
                current["words"].append(word["text"])
        if current is not None:
            spans.append(current)

    for span in spans:
        span["start"] = max(0.0, float(span["start"]) - collar_seconds)
        span["end"] = min(clip_duration, float(span["end"]) + collar_seconds)
        span["text"] = " ".join(span.pop("words"))
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["speaker"]))


def build_cutset(
    *,
    dataset_jsonl: Path,
    output_path: Path,
    max_speakers: int,
    max_gap_seconds: float,
    collar_seconds: float,
    max_clips: int,
    audio_root: Path | None,
) -> dict:
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

    cuts = []
    skipped = 0
    missing_audio = []
    supervision_count = 0
    for row in _read_jsonl(dataset_jsonl):
        audio_path = resolve_audio_path(
            row.get("audio_path"), dataset_jsonl=dataset_jsonl, audio_root=audio_root
        )
        speakers = sorted({str(word.get("speaker") or "") for word in row.get("words") or []})
        if not audio_path.exists():
            if len(missing_audio) < 5:
                missing_audio.append(str(row.get("audio_path") or ""))
            skipped += 1
            continue
        if not speakers or len(speakers) > max_speakers:
            skipped += 1
            continue
        clip_id = str(row.get("clip_id") or audio_path.stem)
        recording = Recording.from_file(audio_path, recording_id=clip_id)
        duration = min(float(row.get("duration") or recording.duration), recording.duration)
        spans = group_speaker_words(
            row.get("words") or [],
            max_gap_seconds=max_gap_seconds,
            collar_seconds=collar_seconds,
            clip_duration=duration,
            clip_offset=float(row.get("relative_start") or 0.0),
        )
        supervisions = [
            SupervisionSegment(
                id=f"{clip_id}-sup{index:04d}",
                recording_id=clip_id,
                start=float(span["start"]),
                duration=max(0.01, float(span["end"]) - float(span["start"])),
                channel=0,
                text=str(span["text"]),
                speaker=str(span["speaker"]),
                language="en",
            )
            for index, span in enumerate(spans)
            if span["text"] and float(span["end"]) > float(span["start"])
        ]
        if not supervisions:
            skipped += 1
            continue
        cuts.append(
            MonoCut(
                id=clip_id,
                start=0.0,
                duration=duration,
                channel=0,
                recording=recording,
                supervisions=supervisions,
            )
        )
        supervision_count += len(supervisions)
        if max_clips > 0 and len(cuts) >= max_clips:
            break

    if not cuts:
        raise RuntimeError(
            f"No usable cuts found in {dataset_jsonl}; sample missing audio paths: {missing_audio}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(cuts).to_file(output_path)
    summary = {
        "dataset_jsonl": str(dataset_jsonl),
        "output_path": str(output_path),
        "cuts": len(cuts),
        "supervisions": supervision_count,
        "skipped": skipped,
        "max_speakers": max_speakers,
        "max_gap_seconds": max_gap_seconds,
        "collar_seconds": collar_seconds,
        "audio_root": str(audio_root) if audio_root else None,
        "missing_audio_examples": missing_audio,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert forced-word mono clips into NeMo multitalker Lhotse cuts."
    )
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--collar-seconds", type=float, default=0.16)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--audio-root", type=Path)
    args = parser.parse_args()
    build_cutset(
        dataset_jsonl=args.dataset_jsonl,
        output_path=args.output,
        max_speakers=args.max_speakers,
        max_gap_seconds=args.max_gap_seconds,
        collar_seconds=args.collar_seconds,
        max_clips=args.max_clips,
        audio_root=args.audio_root,
    )


if __name__ == "__main__":
    main()
