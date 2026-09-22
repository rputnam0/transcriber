from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np


AUDIO_SUFFIXES = {".ogg", ".wav", ".flac", ".mp3", ".m4a"}


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "unknown"


def deduplicate_windows(rows: Iterable[Mapping[str, object]], *, split: str) -> list[dict]:
    windows = {}
    for row in rows:
        if str(row.get("split_id") or "") != split:
            continue
        key = (
            str(row.get("session") or ""),
            round(float(row.get("window_start") or 0.0), 3),
            round(float(row.get("window_end") or 0.0), 3),
        )
        window = windows.setdefault(key, dict(row))
        speaker_members = window.setdefault("speaker_members", {})
        speaker = str(row.get("speaker_id") or "")
        member = str(row.get("target_member") or "")
        if speaker and member:
            speaker_members[speaker] = member
    return [windows[key] for key in sorted(windows)]


def select_chunk_supervisions(
    spans: Sequence[Mapping[str, object]],
    *,
    chunk_start: float,
    chunk_duration: float,
    collar_seconds: float,
) -> tuple[list[dict], dict]:
    chunk_end = chunk_start + chunk_duration
    selected = []
    boundary_clipped = 0
    for span in spans:
        start = float(span.get("start") or 0.0)
        end = float(span.get("end") or start)
        midpoint = (start + end) / 2.0
        text = str(span.get("text") or "").strip()
        speaker = str(span.get("speaker") or "").strip()
        if not text or not speaker or midpoint < chunk_start or midpoint >= chunk_end:
            continue
        clipped_start = max(start, chunk_start)
        clipped_end = min(end, chunk_end)
        if clipped_end <= clipped_start:
            continue
        if clipped_start > start or clipped_end < end:
            boundary_clipped += 1
        selected.append(
            {
                "speaker": speaker,
                "start": max(0.0, clipped_start - chunk_start - collar_seconds),
                "end": min(chunk_duration, clipped_end - chunk_start + collar_seconds),
                "text": text,
            }
        )
    selected.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return selected, {
        "boundary_clipped_spans": boundary_clipped,
        "word_count": sum(len(span["text"].split()) for span in selected),
        "speakers": sorted({span["speaker"] for span in selected}),
    }


def activity_statistics(spans: Sequence[Mapping[str, object]], *, duration: float) -> dict:
    frame_seconds = 0.08
    frame_count = max(1, int(math.ceil(duration / frame_seconds)))
    active = [set() for _ in range(frame_count)]
    for span in spans:
        start_frame = max(0, int(math.floor(float(span["start"]) / frame_seconds)))
        end_frame = min(
            frame_count,
            max(start_frame + 1, int(math.ceil(float(span["end"]) / frame_seconds))),
        )
        for frame in range(start_frame, end_frame):
            active[frame].add(str(span["speaker"]))
    speech_frames = sum(bool(speakers) for speakers in active)
    overlap_frames = sum(len(speakers) > 1 for speakers in active)
    return {
        "speech_seconds": speech_frames * frame_seconds,
        "overlap_seconds": overlap_frames * frame_seconds,
        "overlap_speech_fraction": overlap_frames / speech_frames if speech_frames else 0.0,
    }


def energy_vad_regions(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    frame_seconds: float = 0.02,
    min_speech_seconds: float = 0.08,
    bridge_gap_seconds: float = 0.14,
    pad_seconds: float = 0.08,
) -> list[tuple[float, float]]:
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = len(samples) // frame_size
    if not frame_count:
        return []
    framed = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    threshold = max(
        5e-4,
        min(3e-3, float(np.quantile(rms, 0.95)) * 0.01),
        float(np.quantile(rms, 0.20)) * 4.0,
    )
    active = rms >= threshold

    bridge_frames = max(0, int(round(bridge_gap_seconds / frame_seconds)))
    index = 0
    while index < frame_count:
        if active[index]:
            index += 1
            continue
        gap_start = index
        while index < frame_count and not active[index]:
            index += 1
        if gap_start > 0 and index < frame_count and index - gap_start <= bridge_frames:
            active[gap_start:index] = True

    min_frames = max(1, int(round(min_speech_seconds / frame_seconds)))
    regions = []
    index = 0
    duration = len(samples) / sample_rate
    while index < frame_count:
        if not active[index]:
            index += 1
            continue
        region_start = index
        while index < frame_count and active[index]:
            index += 1
        if index - region_start < min_frames:
            continue
        regions.append(
            (
                max(0.0, region_start * frame_seconds - pad_seconds),
                min(duration, index * frame_seconds + pad_seconds),
            )
        )
    return regions


def align_text_to_vad(
    text_spans: Sequence[Mapping[str, object]],
    regions_by_speaker: Mapping[str, Sequence[tuple[float, float]]],
    *,
    duration: float,
    collar_seconds: float,
    match_pad_seconds: float = 0.6,
) -> list[dict]:
    aligned: dict[tuple[str, float, float], list[str]] = {}
    for span in sorted(
        text_spans,
        key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])),
    ):
        speaker = str(span["speaker"])
        span_start = float(span["start"])
        span_end = float(span["end"])
        matching = [
            region
            for region in regions_by_speaker.get(speaker, [])
            if region[1] > span_start - match_pad_seconds
            and region[0] < span_end + match_pad_seconds
        ]
        if not matching:
            matching = [(span_start, span_end)]
        for region in matching:
            aligned.setdefault((speaker, float(region[0]), float(region[1])), [])
        aligned[(speaker, float(matching[0][0]), float(matching[0][1]))].append(str(span["text"]))

    supervisions = [
        {
            "speaker": speaker,
            "start": max(0.0, start - collar_seconds),
            "end": min(duration, end + collar_seconds),
            "text": " ".join(aligned[(speaker, start, end)]),
        }
        for speaker, start, end in sorted(aligned, key=lambda item: (item[1], item[2], item[0]))
    ]
    return [span for span in supervisions if span["end"] > span["start"]]


def _extract_audio_members(zip_path: Path, cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            output_path = cache_dir / Path(member.filename).name
            if not output_path.exists() or output_path.stat().st_size != member.file_size:
                with archive.open(member) as source, output_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            extracted[member.filename] = output_path
            extracted[Path(member.filename).name] = output_path
    return extracted


def _mix_audio_chunk(
    inputs: Sequence[Path],
    output_path: Path,
    *,
    start: float,
    duration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command.extend(["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            f"amix=inputs={len(inputs)}:normalize=0:dropout_transition=0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)


def _load_audio_chunk(path: Path, *, start: float, duration: float) -> tuple[np.ndarray, int]:
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(path)
    waveform, sample_rate = sf.read(
        path,
        start=max(0, int(round(start * info.samplerate))),
        frames=max(1, int(round(duration * info.samplerate))),
        dtype="float32",
        always_2d=True,
    )
    mono = waveform.mean(axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // divisor, int(sample_rate) // divisor)
        sample_rate = 16000
    expected = max(1, int(round(duration * sample_rate)))
    if len(mono) < expected:
        mono = np.pad(mono, (0, expected - len(mono)))
    return np.asarray(mono[:expected], dtype=np.float32), int(sample_rate)


def build_dataset(
    *,
    manifest_path: Path,
    output_dir: Path,
    split: str,
    chunk_seconds: float,
    chunk_hop_seconds: float,
    max_speakers: int,
    min_words: int,
    collar_seconds: float,
    max_clips: int,
    activity_source: str,
) -> dict:
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

    windows = deduplicate_windows(_read_jsonl(manifest_path), split=split)
    cache_root = output_dir / "_stems"
    cuts = []
    skipped_speakers = 0
    skipped_words = 0
    boundary_clipped = 0
    total_speech = 0.0
    total_overlap = 0.0
    overlap_clips = 0
    sessions = set()
    for window in windows:
        session = str(window["session"])
        sessions.add(session)
        members = _extract_audio_members(
            Path(str(window["source_zip"])), cache_root / _safe_id(session)
        )
        mixture_inputs = [members[str(member)] for member in window.get("mixture_members") or []]
        if not mixture_inputs:
            continue
        window_duration = float(window.get("duration") or 0.0)
        offset = 0.0
        while offset < window_duration - 1e-6:
            duration = min(chunk_seconds, window_duration - offset)
            text_spans, chunk_summary = select_chunk_supervisions(
                window.get("word_spans") or [],
                chunk_start=offset,
                chunk_duration=duration,
                collar_seconds=0.0 if activity_source == "energy-vad" else collar_seconds,
            )
            speaker_count = len(chunk_summary["speakers"])
            if not speaker_count or speaker_count > max_speakers:
                skipped_speakers += 1
                offset += chunk_hop_seconds
                continue
            if int(chunk_summary["word_count"]) < min_words:
                skipped_words += 1
                offset += chunk_hop_seconds
                continue
            clip_id = (
                f"{_safe_id(session)}_w{int(round(float(window['window_start']))):06d}_"
                f"c{int(round(offset * 1000)):06d}"
            )
            audio_path = output_dir / "audio" / f"{clip_id}.wav"
            _mix_audio_chunk(
                mixture_inputs,
                audio_path,
                start=float(window["window_start"]) + offset,
                duration=duration,
            )
            spans = text_spans
            if activity_source == "energy-vad":
                regions_by_speaker = {}
                speaker_members = dict(window.get("speaker_members") or {})
                for speaker in chunk_summary["speakers"]:
                    member = speaker_members.get(speaker)
                    if not member:
                        continue
                    waveform, sample_rate = _load_audio_chunk(
                        members[str(member)],
                        start=float(window["window_start"]) + offset,
                        duration=duration,
                    )
                    regions_by_speaker[speaker] = energy_vad_regions(
                        waveform, sample_rate=sample_rate
                    )
                spans = align_text_to_vad(
                    text_spans,
                    regions_by_speaker,
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
                    speaker=str(span["speaker"]),
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
                    custom={
                        # Keep transcript timing independent from the activity-mask source.
                        # Energy VAD is privileged stem-side supervision and must not rewrite
                        # the target-ASR labels used by downstream training and evaluation.
                        "transcript_spans": text_spans,
                        "activity_supervision_source": activity_source,
                    },
                )
            )
            activity = activity_statistics(spans, duration=duration)
            total_speech += float(activity["speech_seconds"])
            total_overlap += float(activity["overlap_seconds"])
            overlap_clips += int(float(activity["overlap_seconds"]) > 0.0)
            boundary_clipped += int(chunk_summary["boundary_clipped_spans"])
            if max_clips > 0 and len(cuts) >= max_clips:
                break
            offset += chunk_hop_seconds
        if max_clips > 0 and len(cuts) >= max_clips:
            break

    if not cuts:
        raise RuntimeError(f"No usable {split} cuts were produced from {manifest_path}")
    cuts_path = output_dir / f"{split}_cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(cuts_path)
    summary = {
        "manifest_path": str(manifest_path),
        "cuts_path": str(cuts_path),
        "split": split,
        "sessions": sorted(sessions),
        "session_count": len(sessions),
        "window_count": len(windows),
        "cuts": len(cuts),
        "hours": sum(cut.duration for cut in cuts) / 3600.0,
        "supervisions": sum(len(cut.supervisions) for cut in cuts),
        "overlap_clips": overlap_clips,
        "overlap_clip_fraction": overlap_clips / len(cuts),
        "labeled_speech_hours": total_speech / 3600.0,
        "labeled_overlap_hours": total_overlap / 3600.0,
        "overlap_speech_fraction": total_overlap / total_speech if total_speech else 0.0,
        "boundary_clipped_spans": boundary_clipped,
        "skipped_too_many_speakers": skipped_speakers,
        "skipped_too_few_words": skipped_words,
        "max_speakers": max_speakers,
        "min_words": min_words,
        "chunk_seconds": chunk_seconds,
        "chunk_hop_seconds": chunk_hop_seconds,
        "activity_source": activity_source,
    }
    (output_dir / f"{split}_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mono multitalker Parakeet cuts from synchronized Drive speaker tracks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--chunk-hop-seconds", type=float, default=30.0)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--collar-seconds", type=float, default=0.16)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--activity-source",
        choices=("energy-vad", "transcript"),
        default="energy-vad",
    )
    args = parser.parse_args()
    build_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split=args.split,
        chunk_seconds=args.chunk_seconds,
        chunk_hop_seconds=args.chunk_hop_seconds,
        max_speakers=args.max_speakers,
        min_words=args.min_words,
        collar_seconds=args.collar_seconds,
        max_clips=args.max_clips,
        activity_source=args.activity_source,
    )


if __name__ == "__main__":
    main()
