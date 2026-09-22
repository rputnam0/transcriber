from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import soxr

from build_moss_brief_overlap_crops import render_weighted_target
from build_moss_diarization_dataset import DEFAULT_PROMPT


SAMPLE_RATE = 16_000


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def group_high_quality_utterances(
    words: Sequence[Mapping[str, object]],
    *,
    minimum_score: float,
    maximum_word_seconds: float,
    maximum_gap_seconds: float,
    maximum_utterance_seconds: float,
    maximum_words_per_second: float,
) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for raw_word in words:
        word = dict(raw_word)
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or 0.0)
        score = float(word.get("score") or 0.0)
        speaker = str(word.get("speaker") or "").strip()
        if (
            speaker
            and word.get("alignment_source") == "mms"
            and score >= minimum_score
            and 0.02 <= end - start <= maximum_word_seconds
        ):
            by_speaker[speaker].append(word)

    utterances = []
    for speaker, speaker_words in by_speaker.items():
        speaker_words.sort(key=lambda word: (float(word["start"]), float(word["end"])))
        groups: list[list[dict]] = []
        current: list[dict] = []
        for word in speaker_words:
            if current:
                gap = float(word["start"]) - float(current[-1]["end"])
                duration = float(word["end"]) - float(current[0]["start"])
                if gap > maximum_gap_seconds or duration > maximum_utterance_seconds:
                    groups.append(current)
                    current = []
            current.append(word)
        if current:
            groups.append(current)
        for group in groups:
            start = float(group[0]["start"])
            end = float(group[-1]["end"])
            duration = end - start
            if duration <= 0 or len(group) / duration > maximum_words_per_second:
                continue
            utterances.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "words": group,
                    "mean_score": sum(float(word["score"]) for word in group) / len(group),
                }
            )
    return utterances


def _safe_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _stem_index(stems_root: Path, sessions: Iterable[str]) -> dict[tuple[str, str], Path]:
    index = {}
    for session in sorted(set(sessions)):
        session_root = stems_root / _safe_id(session)
        if not session_root.is_dir():
            raise FileNotFoundError(session_root)
        for path in session_root.rglob("*"):
            if path.is_file():
                index[(session, path.name)] = path
    return index


def build_utterance_pool(
    reference_rows: Iterable[Mapping[str, object]],
    enrollment_rows: Iterable[Mapping[str, object]],
    *,
    stems_root: Path,
    enrollment_split: str,
    minimum_score: float,
    maximum_word_seconds: float,
    maximum_gap_seconds: float,
    maximum_utterance_seconds: float,
    maximum_words_per_second: float,
) -> list[dict]:
    source_rows = {}
    sessions = set()
    for raw_row in enrollment_rows:
        row = dict(raw_row)
        if str(row.get("split_id") or "") != enrollment_split:
            continue
        key = (
            str(row.get("session") or ""),
            float(row.get("window_start") or 0.0),
            str(row.get("speaker_id") or ""),
        )
        source_rows.setdefault(key, row)
        sessions.add(key[0])
    stems = _stem_index(stems_root, sessions)
    utterances = []
    for raw_reference in reference_rows:
        reference = dict(raw_reference)
        session = str(reference.get("session") or "")
        window_start = float(reference.get("window_start") or 0.0)
        grouped = group_high_quality_utterances(
            list(reference.get("words") or []),
            minimum_score=minimum_score,
            maximum_word_seconds=maximum_word_seconds,
            maximum_gap_seconds=maximum_gap_seconds,
            maximum_utterance_seconds=maximum_utterance_seconds,
            maximum_words_per_second=maximum_words_per_second,
        )
        for utterance in grouped:
            source = source_rows.get((session, window_start, utterance["speaker"]))
            if source is None:
                continue
            member = Path(str(source.get("target_member") or "")).name
            stem = stems.get((session, member))
            if stem is None:
                continue
            utterances.append(
                {
                    **utterance,
                    "session": session,
                    "window_start": window_start,
                    "stem": str(stem.resolve()),
                    "member": member,
                }
            )
    return utterances


def _read_region(path: Path, *, start: float, end: float) -> tuple[np.ndarray, float]:
    info = sf.info(path)
    clipped_start = max(0.0, start)
    first = int(round(clipped_start * info.samplerate))
    frames = max(1, int(round((end - clipped_start) * info.samplerate)))
    wave, sample_rate = sf.read(
        path,
        start=first,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    mono = np.asarray(wave.mean(axis=1), dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, SAMPLE_RATE).astype(np.float32)
    return mono, clipped_start


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64)) + 1e-10))


def _utterance_audio(
    utterance: Mapping[str, object], *, padding: float
) -> tuple[np.ndarray, float]:
    session_start = float(utterance["window_start"]) + float(utterance["start"])
    session_end = float(utterance["window_start"]) + float(utterance["end"])
    wave, read_start = _read_region(
        Path(str(utterance["stem"])),
        start=session_start - padding,
        end=session_end + padding,
    )
    pre_speech = session_start - read_start
    return wave, pre_speech


def _add_at(destination: np.ndarray, source: np.ndarray, *, start_seconds: float) -> None:
    first = int(round(start_seconds * SAMPLE_RATE))
    source_first = max(0, -first)
    destination_first = max(0, first)
    count = min(len(source) - source_first, len(destination) - destination_first)
    if count > 0:
        destination[destination_first : destination_first + count] += source[
            source_first : source_first + count
        ]


def synthesize_pair(
    primary: Mapping[str, object],
    interruption: Mapping[str, object],
    *,
    crop_seconds: float,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict], float]:
    if primary["speaker"] == interruption["speaker"]:
        raise ValueError("Synthetic overlap requires different speakers")
    padding = 0.12
    primary_wave, primary_pre = _utterance_audio(primary, padding=padding)
    interruption_wave, interruption_pre = _utterance_audio(interruption, padding=padding)
    primary_duration = float(primary["duration"])
    interruption_duration = float(interruption["duration"])
    latest_primary = max(0.25, crop_seconds - primary_duration - 0.25)
    primary_start = rng.uniform(0.25, latest_primary)
    center_low = primary_start + min(0.25, primary_duration * 0.2)
    center_high = primary_start + max(primary_duration * 0.8, primary_duration - 0.25)
    interruption_center = rng.uniform(center_low, max(center_low, center_high))
    interruption_start = min(
        max(0.08, interruption_center - interruption_duration / 2),
        max(0.08, crop_seconds - interruption_duration - 0.08),
    )

    primary_rms = _rms(primary_wave)
    interruption_rms = _rms(interruption_wave)
    primary_dbfs = rng.uniform(-24.0, -18.0)
    relative_db = rng.uniform(-12.0, 2.0)
    primary_target_rms = 10 ** (primary_dbfs / 20)
    primary_wave = primary_wave * (primary_target_rms / max(primary_rms, 1e-5))
    interruption_target_rms = primary_target_rms * 10 ** (relative_db / 20)
    interruption_wave = interruption_wave * (interruption_target_rms / max(interruption_rms, 1e-5))

    mixture = np.zeros(int(round(crop_seconds * SAMPLE_RATE)), dtype=np.float32)
    _add_at(mixture, primary_wave, start_seconds=primary_start - primary_pre)
    _add_at(
        mixture,
        interruption_wave,
        start_seconds=interruption_start - interruption_pre,
    )
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    noise = noise_rng.normal(0.0, 1.0, len(mixture)).astype(np.float32)
    noise = np.convolve(noise, np.asarray([0.2, 0.6, 0.2], dtype=np.float32), mode="same")
    noise_target = primary_target_rms * 10 ** (rng.uniform(-42.0, -32.0) / 20)
    mixture += noise * (noise_target / max(_rms(noise), 1e-5))
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.98:
        mixture *= 0.98 / peak

    def segment(utterance: Mapping[str, object], start: float) -> dict:
        source_start = float(utterance["start"])
        return {
            "speaker": str(utterance["speaker"]),
            "start": start,
            "end": start + float(utterance["duration"]),
            "words": [str(word.get("text") or "") for word in utterance["words"]],
            "word_intervals": [
                {
                    "start": start + float(word["start"]) - source_start,
                    "end": start + float(word["end"]) - source_start,
                    "text": str(word.get("text") or ""),
                }
                for word in utterance["words"]
            ],
        }

    return (
        mixture,
        [segment(primary, primary_start), segment(interruption, interruption_start)],
        relative_db,
    )


def build_synthetic_dataset(
    *,
    forced_word_reference: Path,
    enrollment_manifest: Path,
    stems_root: Path,
    output_dir: Path,
    output_jsonl: Path,
    summary_path: Path,
    examples: int,
    crop_seconds: float,
    seed: int,
    enrollment_split: str,
    minimum_score: float,
    maximum_word_seconds: float,
    maximum_words_per_second: float,
) -> dict:
    utterances = build_utterance_pool(
        _read_jsonl(forced_word_reference),
        _read_jsonl(enrollment_manifest),
        stems_root=stems_root,
        enrollment_split=enrollment_split,
        minimum_score=minimum_score,
        maximum_word_seconds=maximum_word_seconds,
        maximum_gap_seconds=0.55,
        maximum_utterance_seconds=5.5,
        maximum_words_per_second=maximum_words_per_second,
    )
    primary_by_session: dict[str, list[dict]] = defaultdict(list)
    interruption_by_session: dict[str, list[dict]] = defaultdict(list)
    for utterance in utterances:
        if (
            1.25 <= float(utterance["duration"]) <= min(5.5, crop_seconds - 0.5)
            and len(utterance["words"]) >= 3
        ):
            primary_by_session[str(utterance["session"])].append(utterance)
        if 0.12 <= float(utterance["duration"]) <= 2.0 and len(utterance["words"]) <= 8:
            interruption_by_session[str(utterance["session"])].append(utterance)
    sessions = sorted(
        session
        for session, primaries in primary_by_session.items()
        if primaries
        and any(
            primary["speaker"] != interruption["speaker"]
            for primary in primaries
            for interruption in interruption_by_session.get(session, [])
        )
    )
    if not sessions:
        raise ValueError("No same-session primary/interruption pairs survived quality filters")

    rng = random.Random(seed)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    brief_words = 0
    all_words = 0
    relative_levels = []
    attempts = 0
    while len(records) < examples and attempts < examples * 50:
        attempts += 1
        session = sessions[len(records) % len(sessions)]
        primary = rng.choice(primary_by_session[session])
        candidates = [
            utterance
            for utterance in interruption_by_session[session]
            if utterance["speaker"] != primary["speaker"]
        ]
        if not candidates:
            continue
        interruption = rng.choice(candidates)
        mixture, segments, relative_db = synthesize_pair(
            primary,
            interruption,
            crop_seconds=crop_seconds,
            rng=rng,
        )
        ordered_speakers = sorted(
            segments,
            key=lambda item: (float(item["start"]), str(item["speaker"])),
        )
        speaker_ids = {
            str(segment["speaker"]): f"S{index + 1:02d}"
            for index, segment in enumerate(ordered_speakers)
        }
        target, loss_spans, activity = render_weighted_target(segments, speaker_ids)
        example_index = len(records)
        audio_path = audio_dir / f"synthetic_overlap_{example_index:05d}.wav"
        sf.write(audio_path, mixture, SAMPLE_RATE, subtype="PCM_16")
        interruption_words = len(interruption["words"])
        total_words = len(primary["words"]) + interruption_words
        records.append(
            {
                "conversation": [
                    {"role": "user", "message_type": "text", "content": DEFAULT_PROMPT},
                    {"role": "user", "message_type": "audio", "content": str(audio_path.resolve())},
                    {"role": "assistant", "message_type": "text", "content": target},
                ],
                "metadata": {
                    "cut_id": f"synthetic-overlap-{example_index:05d}",
                    "session": session,
                    "crop_duration": crop_seconds,
                    "word_count": total_words,
                    "brief_overlap_word_count": interruption_words,
                    "brief_overlap_word_fraction": interruption_words / max(1, total_words),
                    "speaker_count": 2,
                    "mono_input_only": True,
                    "production_mixture_is_mono": True,
                    "target_source": "synthetic-mono-from-quality-filtered-isolated-stems",
                    "loss_spans": loss_spans,
                    "activity": activity,
                    "stable_session_speaker_ids": speaker_ids,
                    "relative_interruption_db": relative_db,
                    "primary_source": {
                        key: primary[key]
                        for key in ("session", "window_start", "speaker", "start", "end", "stem")
                    },
                    "interruption_source": {
                        key: interruption[key]
                        for key in ("session", "window_start", "speaker", "start", "end", "stem")
                    },
                    "clean_stems_used_for_training_synthesis_only": True,
                    "clean_stem_activity_available_at_inference": False,
                },
            }
        )
        brief_words += interruption_words
        all_words += total_words
        relative_levels.append(relative_db)
    if len(records) < examples:
        raise RuntimeError(f"Built only {len(records)} of {examples} requested examples")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "forced_word_reference": str(forced_word_reference),
        "enrollment_manifest": str(enrollment_manifest),
        "stems_root": str(stems_root),
        "output_jsonl": str(output_jsonl),
        "examples": len(records),
        "sessions": len(sessions),
        "quality_utterances": len(utterances),
        "primary_utterances": sum(len(values) for values in primary_by_session.values()),
        "interruption_utterances": sum(len(values) for values in interruption_by_session.values()),
        "brief_overlap_words": brief_words,
        "all_target_words": all_words,
        "brief_overlap_word_fraction": brief_words / max(1, all_words),
        "relative_interruption_db_min": min(relative_levels),
        "relative_interruption_db_max": max(relative_levels),
        "minimum_alignment_score": minimum_score,
        "maximum_word_seconds": maximum_word_seconds,
        "maximum_words_per_second": maximum_words_per_second,
        "seed": seed,
        "mono_input_only": True,
        "clean_stem_role": "offline training synthesis and ownership/timing labels only",
        "serialization": "onset order with deterministic speaker-ID tie break",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mono MOSS crops with quiet, quality-filtered synthetic interruptions."
    )
    parser.add_argument("--forced-word-reference", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--stems-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2_000)
    parser.add_argument("--crop-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--enrollment-split", default="train")
    parser.add_argument("--minimum-score", type=float, default=0.45)
    parser.add_argument("--maximum-word-seconds", type=float, default=1.25)
    parser.add_argument("--maximum-words-per-second", type=float, default=6.0)
    args = parser.parse_args()
    build_synthetic_dataset(
        forced_word_reference=args.forced_word_reference,
        enrollment_manifest=args.enrollment_manifest,
        stems_root=args.stems_root,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
        examples=args.examples,
        crop_seconds=args.crop_seconds,
        seed=args.seed,
        enrollment_split=args.enrollment_split,
        minimum_score=args.minimum_score,
        maximum_word_seconds=args.maximum_word_seconds,
        maximum_words_per_second=args.maximum_words_per_second,
    )


if __name__ == "__main__":
    main()
