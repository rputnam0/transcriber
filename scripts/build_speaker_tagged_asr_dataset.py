from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import soundfile as sf


CHUNK_RE = re.compile(r"session_(\d+)_(\d+)_(\d+)")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _load_reference_groups(path: Path) -> dict[tuple[str, float, float], list[dict]]:
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in _read_jsonl(path):
        groups[_group_key(row)] = [dict(word) for word in row.get("words") or []]
    return groups


def _parse_audio_key(path: Path) -> tuple[str, float, float]:
    match = CHUNK_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse session/window from {path}")
    session = f"Session {int(match.group(1))}"
    start = int(match.group(2)) / 1000.0
    end = int(match.group(3)) / 1000.0
    return session, round(start, 3), round(end, 3)


def _clip_words(
    words: Sequence[Mapping[str, object]],
    *,
    start: float,
    end: float,
) -> list[dict]:
    clipped = []
    for word in words:
        word_start = float(word.get("start") or 0.0)
        word_end = float(word.get("end") or word_start)
        if word_start < end and word_end > start:
            item = dict(word)
            item["start"] = max(0.0, word_start - start)
            item["end"] = min(end - start, word_end - start)
            clipped.append(item)
    clipped.sort(key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)))
    return clipped


def _word_text(word: Mapping[str, object]) -> str:
    return str(word.get("text") or word.get("normalized") or "").strip()


def tagged_transcript(
    words: Sequence[Mapping[str, object]], *, max_speakers: int
) -> tuple[str, dict[str, str]]:
    speaker_to_tag: dict[str, str] = {}
    parts: list[str] = []
    last_tag: str | None = None
    for word in words:
        speaker = str(word.get("speaker") or "").strip()
        text = _word_text(word)
        if not speaker or not text:
            continue
        if speaker not in speaker_to_tag:
            if len(speaker_to_tag) >= max_speakers:
                return "", speaker_to_tag
            speaker_to_tag[speaker] = f"[S{len(speaker_to_tag)}]"
        tag = speaker_to_tag[speaker]
        if tag != last_tag:
            parts.append(tag)
            last_tag = tag
        parts.append(text)
    return " ".join(parts), speaker_to_tag


def _write_clip(
    source_audio: Path,
    output_audio: Path,
    *,
    offset: float,
    duration: float,
) -> None:
    info = sf.info(source_audio)
    sample_rate = int(info.samplerate)
    start = max(0, int(round(offset * sample_rate)))
    frames = max(1, int(round(duration * sample_rate)))
    audio, _ = sf.read(source_audio, start=start, frames=frames, dtype="float32", always_2d=False)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio, audio, sample_rate)


def build_dataset(
    *,
    chunk_manifest: Path,
    reference_jsonl: Path,
    output_dir: Path,
    max_speakers: int,
    min_words: int,
    max_clips: int,
) -> list[dict]:
    references = _load_reference_groups(reference_jsonl)
    rows = []
    for chunk in _read_jsonl(chunk_manifest):
        audio_path = Path(str(chunk.get("audio_filepath") or ""))
        if not audio_path.exists():
            continue
        session, window_start, window_end = _parse_audio_key(audio_path)
        reference_words = references.get((session, window_start, window_end))
        if not reference_words:
            continue
        offset = float(chunk.get("offset") or 0.0)
        duration = float(chunk.get("duration") or 0.0)
        words = _clip_words(reference_words, start=offset, end=offset + duration)
        speakers = sorted({str(word.get("speaker") or "") for word in words if word.get("speaker")})
        if len(words) < min_words or len(speakers) > max_speakers:
            continue
        transcript, mapping = tagged_transcript(words, max_speakers=max_speakers)
        if not transcript:
            continue
        clip_id = (
            f"{audio_path.stem}_o{int(round(offset * 1000)):09d}_d{int(round(duration * 1000)):06d}"
        )
        output_audio = output_dir / "audio" / f"{clip_id}.wav"
        _write_clip(audio_path, output_audio, offset=offset, duration=duration)
        rows.append(
            {
                "clip_id": clip_id,
                "session": session,
                "source_audio": str(audio_path),
                "audio_path": str(output_audio),
                "window_start": window_start,
                "window_end": window_end,
                "relative_start": offset,
                "relative_end": offset + duration,
                "duration": duration,
                "word_count": len(words),
                "speaker_count": len(speakers),
                "speakers": speakers,
                "speaker_to_tag": mapping,
                "target_text": transcript,
                "words": words,
            }
        )
        if max_clips > 0 and len(rows) >= max_clips:
            break
    _write_jsonl(output_dir / "speaker_tagged_asr_dataset.jsonl", rows)
    summary = {
        "chunk_manifest": str(chunk_manifest),
        "reference_jsonl": str(reference_jsonl),
        "output_dir": str(output_dir),
        "clips": len(rows),
        "words": sum(int(row["word_count"]) for row in rows),
        "speaker_count_distribution": {
            str(count): sum(1 for row in rows if int(row["speaker_count"]) == count)
            for count in sorted({int(row["speaker_count"]) for row in rows})
        },
    }
    (output_dir / "speaker_tagged_asr_dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build speaker-tagged ASR clips from forced words."
    )
    parser.add_argument("--chunk-manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--min-words", type=int, default=12)
    parser.add_argument("--max-clips", type=int, default=0)
    args = parser.parse_args()

    build_dataset(
        chunk_manifest=args.chunk_manifest,
        reference_jsonl=args.reference_jsonl,
        output_dir=args.output_dir,
        max_speakers=args.max_speakers,
        min_words=args.min_words,
        max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
