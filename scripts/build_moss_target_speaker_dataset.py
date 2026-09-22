from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import soxr

from build_moss_diarization_dataset import DEFAULT_PROMPT
from evaluate_sortformer_enrollment_binding import trim_clean_enrollment
from run_se_dicow_target_asr import select_max_energy_window
from transcriber.multitrack_eval import extract_session_stems


SAMPLE_RATE = 16_000
SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\]\s*"
    r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\](?=\[|$)",
    re.DOTALL,
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_serialized_segments(target: str) -> list[dict]:
    segments = []
    for match in SEGMENT_RE.finditer(target):
        text = " ".join(match.group("text").split())
        if text:
            segments.append(
                {
                    "start": float(match.group("start")),
                    "end": float(match.group("end")),
                    "speaker": match.group("speaker"),
                    "text": text,
                }
            )
    return segments


def select_cross_session_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    speaker: str,
    excluded_session: str,
    max_clips: int,
) -> list[dict]:
    candidates = [
        dict(row)
        for row in rows
        if str(row.get("speaker_id") or "") == speaker
        and str(row.get("session") or "") != excluded_session
        and list(row.get("positive_enrollment_spans") or [])
        and row.get("target_member")
        and row.get("source_zip")
    ]
    candidates.sort(
        key=lambda row: (
            -sum(
                float(span.get("duration") or 0.0)
                for span in list(row.get("positive_enrollment_spans") or [])
            ),
            str(row.get("session") or ""),
            str(row.get("row_id") or ""),
        )
    )
    selected = []
    seen_sessions = set()
    for row in candidates:
        session = str(row.get("session") or "")
        if session in seen_sessions:
            continue
        selected.append(row)
        seen_sessions.add(session)
        if len(selected) >= max_clips:
            break
    return selected


class EnrollmentStemCache:
    """Extract compressed source tracks once; enrollment reads only the requested spans."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._members_by_session: dict[str, dict[str, Path]] = {}

    def stem_for_member(self, row: Mapping[str, object], member: object) -> Path:
        session = str(row.get("session") or "")
        if session not in self._members_by_session:
            session_id = re.sub(r"[^a-z0-9]+", "_", session.lower()).strip("_")
            extracted = extract_session_stems(
                Path(str(row["source_zip"])),
                self.root / session_id,
            )
            self._members_by_session[session] = {path.name: path for path in extracted}
        member_name = Path(str(member)).name
        try:
            return self._members_by_session[session][member_name]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing {member_name} in extracted {session} stems") from exc


def _read_audio_region(path: Path, *, start: float, duration: float) -> np.ndarray:
    info = sf.info(path)
    first = max(0, int(round(start * info.samplerate)))
    frames = max(1, int(round(duration * info.samplerate)))
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
    return mono


def materialize_profile(
    rows: Sequence[Mapping[str, object]],
    *,
    stem_cache: EnrollmentStemCache,
    output: Path,
    profile_seconds: float,
) -> dict:
    if not rows:
        raise ValueError("At least one cross-session enrollment row is required")
    samples_per_clip = int(round(profile_seconds * SAMPLE_RATE / len(rows)))
    clips = []
    sources = []
    for row in rows:
        stem = stem_cache.stem_for_member(row, row["target_member"])
        parts = [
            _read_audio_region(
                stem,
                start=float(span.get("start") or 0.0),
                duration=float(span.get("duration") or 0.0),
            )
            for span in list(row.get("positive_enrollment_spans") or [])
            if float(span.get("duration") or 0.0) > 0.0
        ]
        if not parts:
            continue
        packed = trim_clean_enrollment(
            np.concatenate(parts),
            sample_rate=SAMPLE_RATE,
        )
        clips.append(select_max_energy_window(packed, frames=samples_per_clip))
        sources.append(
            {
                "session": str(row.get("session") or ""),
                "row_id": str(row.get("row_id") or ""),
            }
        )
    if not clips:
        raise ValueError("Cross-session enrollment rows contained no usable audio")
    target_samples = int(round(profile_seconds * SAMPLE_RATE))
    profile = np.concatenate(clips)[:target_samples]
    profile = np.pad(profile, (0, max(0, target_samples - len(profile))))
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, profile, SAMPLE_RATE, subtype="PCM_16")
    return {"path": str(output.resolve()), "source_rows": sources}


def render_target(
    segments: Sequence[Mapping[str, object]],
    *,
    source_speaker_id: str,
    timestamp_shift: float,
) -> tuple[str, list[dict], list[dict]]:
    selected = [
        dict(segment)
        for segment in segments
        if str(segment.get("speaker") or "") == source_speaker_id
    ]
    selected.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    parts = []
    loss_spans = []
    references = []
    position = 0

    def append(value: str, *, kind: str, weight: float) -> None:
        nonlocal position
        start = position
        parts.append(value)
        position += len(value)
        loss_spans.append({"start": start, "end": position, "kind": kind, "weight": weight})

    for segment in selected:
        start = float(segment["start"])
        end = float(segment["end"])
        append(f"[{start + timestamp_shift:.2f}]", kind="timestamp", weight=2.0)
        append("[S01]", kind="speaker_tag", weight=3.0)
        parts.append(" ")
        position += 1
        kind = "brief_overlap_word" if segment.get("brief_overlap") else "word"
        append(str(segment["text"]), kind=kind, weight=4.0 if kind == "brief_overlap_word" else 1.0)
        append(f"[{end + timestamp_shift:.2f}]", kind="timestamp", weight=2.0)
        references.append(
            {
                "start": start,
                "end": end,
                "text": str(segment["text"]),
                "brief_overlap": bool(segment.get("brief_overlap")),
            }
        )
    return "".join(parts), loss_spans, references


def _load_mono(path: Path) -> np.ndarray:
    wave, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(wave.mean(axis=1), dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, SAMPLE_RATE).astype(np.float32)
    return mono


def _safe_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _source_segments(row: Mapping[str, object]) -> list[dict]:
    metadata = dict(row.get("metadata") or {})
    activity = list(metadata.get("activity") or [])
    segments = parse_serialized_segments(
        str(list(row.get("conversation") or [])[-1].get("content") or "")
    )
    for segment in segments:
        matching = [
            interval
            for interval in activity
            if str(interval.get("speaker") or "") == str(segment.get("speaker") or "")
            and abs(float(interval.get("start") or 0.0) - float(segment["start"])) <= 0.02
            and abs(float(interval.get("end") or 0.0) - float(segment["end"])) <= 0.02
        ]
        segment["brief_overlap"] = any(
            interval.get("brief") and interval.get("overlap") for interval in matching
        )
    return segments


def build_target_speaker_dataset(
    *,
    crop_manifest: Path,
    enrollment_manifest: Path,
    output_dir: Path,
    output_jsonl: Path,
    summary_path: Path,
    enrollment_split: str,
    profile_seconds: float,
    gap_seconds: float,
    max_enrollment_clips: int,
    include_negatives_every: int,
    negative_speakers_per_crop: int,
    maximum_crops: int,
    stems_cache_root: Path,
) -> dict:
    crops = list(_read_jsonl(crop_manifest))
    if maximum_crops > 0:
        crops = crops[:maximum_crops]
    enrollment_rows = [
        row
        for row in _read_jsonl(enrollment_manifest)
        if str(row.get("split_id") or "") == enrollment_split
    ]
    if not crops or not enrollment_rows:
        raise ValueError("Crop and enrollment manifests must both contain records")
    stem_cache = EnrollmentStemCache(stems_cache_root)
    profiles: dict[tuple[str, str], dict] = {}
    records = []
    positive_records = 0
    negative_records = 0
    target_words = 0
    brief_overlap_words = 0
    timestamp_shift = profile_seconds + gap_seconds

    def profile_for(session: str, speaker: str) -> dict:
        key = (session, speaker)
        if key not in profiles:
            selected = select_cross_session_rows(
                enrollment_rows,
                speaker=speaker,
                excluded_session=session,
                max_clips=max_enrollment_clips,
            )
            if not selected:
                raise ValueError(f"No cross-session enrollment for {speaker} excluding {session}")
            profiles[key] = materialize_profile(
                selected,
                stem_cache=stem_cache,
                output=output_dir / "enrollment" / _safe_id(session) / f"{_safe_id(speaker)}.wav",
                profile_seconds=profile_seconds,
            )
            profiles[key]["excluded_session"] = session
            if session in {source["session"] for source in profiles[key]["source_rows"]}:
                raise ValueError(f"Enrollment leakage for {speaker} in {session}")
        return profiles[key]

    for crop_index, crop in enumerate(crops):
        metadata = dict(crop.get("metadata") or {})
        session = str(metadata.get("session") or "")
        source_ids = dict(metadata.get("stable_session_speaker_ids") or {})
        id_to_name = {speaker_id: name for name, speaker_id in source_ids.items()}
        segments = _source_segments(crop)
        active_ids = sorted({str(segment["speaker"]) for segment in segments})
        mono_path = Path(str(list(crop.get("conversation") or [])[1].get("content") or ""))
        if not mono_path.is_absolute():
            mono_path = (crop_manifest.parent / mono_path).resolve()
        mono = _load_mono(mono_path)

        def add_record(speaker: str, source_id: str | None, *, negative: bool) -> None:
            nonlocal positive_records, negative_records, target_words, brief_overlap_words
            profile = profile_for(session, speaker)
            enrollment = _load_mono(Path(profile["path"]))
            combined = np.concatenate(
                (
                    enrollment,
                    np.zeros(int(round(gap_seconds * SAMPLE_RATE)), dtype=np.float32),
                    mono,
                )
            )
            suffix = "negative" if negative else _safe_id(speaker)
            cut_id = str(metadata.get("cut_id") or f"crop-{crop_index:05d}")
            audio_path = output_dir / "audio" / f"{_safe_id(cut_id)}__{suffix}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, combined, SAMPLE_RATE, subtype="PCM_16")
            if negative:
                target = "NO_TARGET_SPEECH"
                loss_spans = [{"start": 0, "end": len(target), "kind": "no_target", "weight": 1.0}]
                references = []
                activity = []
                sample_loss_weight = 0.25
                negative_records += 1
            else:
                target, loss_spans, references = render_target(
                    segments,
                    source_speaker_id=str(source_id),
                    timestamp_shift=timestamp_shift,
                )
                if not target:
                    return
                positive_records += 1
                target_words += sum(len(item["text"].split()) for item in references)
                brief_overlap_words += sum(
                    len(item["text"].split()) for item in references if item["brief_overlap"]
                )
                activity = [
                    {
                        "speaker": "S01",
                        "speaker_index": 0,
                        "start": float(item["start"]) + timestamp_shift,
                        "end": float(item["end"]) + timestamp_shift,
                    }
                    for item in references
                ]
                sample_loss_weight = 1.0
            prompt = (
                DEFAULT_PROMPT
                + f" The first {profile_seconds:.2f} seconds are historical enrollment for S01. "
                + f"The mono conversation starts at {timestamp_shift:.2f} seconds. "
                + "Transcribe only S01 from the conversation; ignore enrollment speech and all other voices. "
                + "If S01 is absent, output NO_TARGET_SPEECH."
            )
            records.append(
                {
                    "conversation": [
                        {"role": "user", "message_type": "text", "content": prompt},
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": str(audio_path.resolve()),
                        },
                        {"role": "assistant", "message_type": "text", "content": target},
                    ],
                    "metadata": {
                        "cut_id": f"{cut_id}::target::{_safe_id(speaker)}",
                        "source_cut_id": cut_id,
                        "session": session,
                        "target_speaker": speaker,
                        "target_source_speaker_id": source_id,
                        "negative": negative,
                        "reference_segments": references,
                        "activity": activity,
                        "activity_supervised": True,
                        "activity_valid_start": timestamp_shift,
                        "activity_valid_end": timestamp_shift + len(mono) / SAMPLE_RATE,
                        "sample_loss_weight": sample_loss_weight,
                        "timestamp_shift": timestamp_shift,
                        "profile_seconds": profile_seconds,
                        "gap_seconds": gap_seconds,
                        "loss_spans": loss_spans,
                        "mono_input_only": True,
                        "production_mixture_is_mono": True,
                        "current_session_stems_used_as_input": False,
                        "historical_enrollment": True,
                        "enrollment_uses_evaluation_session_audio": False,
                        "enrollment_source_rows": profile["source_rows"],
                        "source_mono_audio": str(mono_path),
                    },
                }
            )

        for source_id in active_ids:
            speaker = id_to_name.get(source_id)
            if speaker:
                add_record(speaker, source_id, negative=False)
        if include_negatives_every > 0 and crop_index % include_negatives_every == 0:
            active_names = {
                id_to_name[source_id] for source_id in active_ids if source_id in id_to_name
            }
            absent = sorted(set(source_ids) - active_names)
            if absent:
                offset = crop_index % len(absent)
                ordered_absent = absent[offset:] + absent[:offset]
                selected_absent = (
                    ordered_absent
                    if negative_speakers_per_crop < 0
                    else ordered_absent[:negative_speakers_per_crop]
                )
                for speaker in selected_absent:
                    add_record(speaker, None, negative=True)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "crop_manifest": str(crop_manifest),
        "enrollment_manifest": str(enrollment_manifest),
        "output_jsonl": str(output_jsonl),
        "records": len(records),
        "positive_records": positive_records,
        "negative_records": negative_records,
        "target_words": target_words,
        "brief_overlap_words": brief_overlap_words,
        "brief_overlap_word_fraction": brief_overlap_words / max(1, target_words),
        "profile_count": len(profiles),
        "profile_seconds": profile_seconds,
        "gap_seconds": gap_seconds,
        "negative_speakers_per_crop": negative_speakers_per_crop,
        "maximum_crops": maximum_crops,
        "cross_session_enrollment_required": True,
        "production_mixture_is_mono": True,
        "current_session_stems_used_as_input": False,
        "clean_stem_role": "offline historical enrollment preparation and ownership labels only",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build overlap-centered MOSS target-ASR records with cross-session enrollment."
    )
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--enrollment-split", default="train")
    parser.add_argument("--profile-seconds", type=float, default=4.0)
    parser.add_argument("--gap-seconds", type=float, default=0.5)
    parser.add_argument("--max-enrollment-clips", type=int, default=2)
    parser.add_argument("--include-negatives-every", type=int, default=4)
    parser.add_argument(
        "--negative-speakers-per-crop",
        type=int,
        default=1,
        help="Number of absent speakers per selected crop; use -1 for every absent roster speaker.",
    )
    parser.add_argument("--maximum-crops", type=int, default=0)
    parser.add_argument("--stems-cache-root", type=Path, required=True)
    args = parser.parse_args()
    build_target_speaker_dataset(
        crop_manifest=args.crop_manifest,
        enrollment_manifest=args.enrollment_manifest,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
        enrollment_split=args.enrollment_split,
        profile_seconds=args.profile_seconds,
        gap_seconds=args.gap_seconds,
        max_enrollment_clips=args.max_enrollment_clips,
        include_negatives_every=args.include_negatives_every,
        negative_speakers_per_crop=args.negative_speakers_per_crop,
        maximum_crops=args.maximum_crops,
        stems_cache_root=args.stems_cache_root,
    )


if __name__ == "__main__":
    main()
