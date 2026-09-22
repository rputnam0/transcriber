from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from build_multitalker_parakeet_drive_dataset import (
    _extract_audio_members,
    _load_audio_chunk,
    _mix_audio_chunk,
    _read_jsonl,
    _safe_id,
    align_text_to_vad,
    energy_vad_regions,
    select_chunk_supervisions,
)


def target_speaker_spans(row: Mapping[str, object]) -> list[dict]:
    speaker = str(row.get("speaker_id") or "")
    return [
        dict(span)
        for span in row.get("word_spans") or []
        if str(span.get("speaker") or "") == speaker and str(span.get("text") or "").strip()
    ]


def round_robin_rows(rows: Iterable[Mapping[str, object]], *, split: str) -> list[dict]:
    by_speaker: dict[str, list[dict]] = {}
    for row in rows:
        if str(row.get("split_id") or "") != split:
            continue
        speaker = str(row.get("speaker_id") or "")
        if speaker:
            by_speaker.setdefault(speaker, []).append(dict(row))
    ordered = []
    max_rows = max((len(items) for items in by_speaker.values()), default=0)
    for index in range(max_rows):
        for speaker in sorted(by_speaker):
            if index < len(by_speaker[speaker]):
                ordered.append(by_speaker[speaker][index])
    return ordered


def build_clean_dataset(
    *,
    manifest_path: Path,
    output_dir: Path,
    split: str,
    chunk_seconds: float,
    chunk_hop_seconds: float,
    min_words: int,
    collar_seconds: float,
    max_clips_per_speaker: int,
    max_clips: int,
    stem_cache_dir: Path | None,
) -> dict:
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

    rows = round_robin_rows(_read_jsonl(manifest_path), split=split)
    cache_root = stem_cache_dir or (output_dir / "_stems")
    cuts = []
    clips_by_speaker: Counter[str] = Counter()
    words_by_speaker: Counter[str] = Counter()
    sessions = set()
    skipped_words = 0
    for row in rows:
        speaker = str(row["speaker_id"])
        if max_clips_per_speaker > 0 and clips_by_speaker[speaker] >= max_clips_per_speaker:
            continue
        session = str(row["session"])
        sessions.add(session)
        members = _extract_audio_members(
            Path(str(row["source_zip"])), cache_root / _safe_id(session)
        )
        target_path = members.get(str(row["target_member"]))
        if target_path is None:
            continue
        source_spans = target_speaker_spans(row)
        window_duration = float(row.get("duration") or 0.0)
        offset = 0.0
        while offset < window_duration - 1e-6:
            if max_clips_per_speaker > 0 and clips_by_speaker[speaker] >= max_clips_per_speaker:
                break
            duration = min(chunk_seconds, window_duration - offset)
            text_spans, chunk_summary = select_chunk_supervisions(
                source_spans,
                chunk_start=offset,
                chunk_duration=duration,
                collar_seconds=0.0,
            )
            if int(chunk_summary["word_count"]) < min_words:
                skipped_words += 1
                offset += chunk_hop_seconds
                continue
            clip_id = (
                f"clean_{_safe_id(session)}_{_safe_id(speaker)}_"
                f"w{int(round(float(row['window_start']))):06d}_"
                f"c{int(round(offset * 1000)):06d}"
            )
            audio_path = output_dir / "audio" / f"{clip_id}.wav"
            _mix_audio_chunk(
                [target_path],
                audio_path,
                start=float(row["window_start"]) + offset,
                duration=duration,
            )
            waveform, sample_rate = _load_audio_chunk(
                target_path,
                start=float(row["window_start"]) + offset,
                duration=duration,
            )
            spans = align_text_to_vad(
                text_spans,
                {speaker: energy_vad_regions(waveform, sample_rate=sample_rate)},
                duration=duration,
                collar_seconds=collar_seconds,
            )
            recording = Recording.from_file(audio_path, recording_id=clip_id)
            supervisions = [
                SupervisionSegment(
                    id=f"{clip_id}-sup{index:04d}",
                    recording_id=clip_id,
                    start=float(span["start"]),
                    duration=max(0.01, float(span["end"]) - float(span["start"])),
                    channel=0,
                    text=str(span["text"]),
                    speaker=speaker,
                    language="en",
                )
                for index, span in enumerate(spans)
            ]
            cuts.append(
                MonoCut(
                    id=clip_id,
                    start=0.0,
                    duration=min(duration, recording.duration),
                    channel=0,
                    recording=recording,
                    supervisions=supervisions,
                    custom={"training_domain": "clean-isolated-speaker"},
                )
            )
            clips_by_speaker[speaker] += 1
            words_by_speaker[speaker] += int(chunk_summary["word_count"])
            if max_clips > 0 and len(cuts) >= max_clips:
                break
            offset += chunk_hop_seconds
        if max_clips > 0 and len(cuts) >= max_clips:
            break

    if not cuts:
        raise RuntimeError(f"No usable clean {split} cuts were produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    cuts_path = output_dir / f"{split}_cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(cuts_path)
    summary = {
        "manifest_path": str(manifest_path),
        "cuts_path": str(cuts_path),
        "split": split,
        "cuts": len(cuts),
        "hours": sum(cut.duration for cut in cuts) / 3600.0,
        "sessions": sorted(sessions),
        "clips_by_speaker": dict(sorted(clips_by_speaker.items())),
        "words_by_speaker": dict(sorted(words_by_speaker.items())),
        "total_words": sum(words_by_speaker.values()),
        "min_words": min_words,
        "chunk_seconds": chunk_seconds,
        "chunk_hop_seconds": chunk_hop_seconds,
        "max_clips_per_speaker": max_clips_per_speaker,
        "skipped_too_few_words": skipped_words,
        "activity_supervision": "isolated-track-energy-vad-training-only",
    }
    (output_dir / f"{split}_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean isolated-speaker cuts for domain ASR adaptation."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--chunk-hop-seconds", type=float, default=30.0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--collar-seconds", type=float, default=0.12)
    parser.add_argument("--max-clips-per-speaker", type=int, default=800)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--stem-cache-dir", type=Path)
    args = parser.parse_args()
    build_clean_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split=args.split,
        chunk_seconds=args.chunk_seconds,
        chunk_hop_seconds=args.chunk_hop_seconds,
        min_words=args.min_words,
        collar_seconds=args.collar_seconds,
        max_clips_per_speaker=args.max_clips_per_speaker,
        max_clips=args.max_clips,
        stem_cache_dir=args.stem_cache_dir,
    )


if __name__ == "__main__":
    main()
