from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from evaluate_sortformer_enrollment_binding import (
    _embed_waveforms,
    _load_titanet,
    _load_wave,
    assignment_maps,
    enrollment_leave_one_out_score,
    enrollment_paths,
    enrollment_provenance,
    trim_clean_enrollment,
)


SAMPLE_RATE = 16_000


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def _overlaps_other(
    segment: Mapping[str, object],
    segments: Sequence[Mapping[str, object]],
) -> bool:
    speaker = str(segment.get("speaker") or "")
    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or 0.0)
    return any(
        str(other.get("speaker") or "") != speaker
        and float(other.get("start") or 0.0) < end
        and float(other.get("end") or 0.0) > start
        for other in segments
    )


def stream_waveforms(
    mixture: np.ndarray,
    segments: Sequence[Mapping[str, object]],
    *,
    sample_rate: int,
    minimum_segment_seconds: float = 0.35,
    minimum_stream_seconds: float = 0.5,
    maximum_stream_seconds: float = 60.0,
    exclusive_only: bool = True,
) -> tuple[list[str], list[np.ndarray], list[dict]]:
    candidates: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
    for segment in segments:
        speaker = str(segment.get("speaker") or "").strip()
        start = max(0.0, float(segment.get("start") or 0.0))
        end = min(len(mixture) / sample_rate, float(segment.get("end") or 0.0))
        overlapped = _overlaps_other(segment, segments)
        if not speaker or end - start < minimum_segment_seconds:
            continue
        if exclusive_only and overlapped:
            continue
        candidates[speaker].append((start, end, overlapped))

    speakers = []
    waves = []
    metadata = []
    maximum_samples = max(1, int(round(maximum_stream_seconds * sample_rate)))
    for speaker, intervals in sorted(candidates.items()):
        # Long, clean turns are the most stable evidence for a stream-level identity embedding.
        ranked = sorted(intervals, key=lambda item: (-(item[1] - item[0]), item[0]))
        parts = []
        selected = []
        samples = 0
        for start, end, overlapped in ranked:
            first = int(round(start * sample_rate))
            last = min(len(mixture), int(round(end * sample_rate)))
            remaining = maximum_samples - samples
            if last <= first or remaining <= 0:
                continue
            part = mixture[first : min(last, first + remaining)]
            if not np.any(np.isfinite(part)):
                continue
            parts.append(np.nan_to_num(part, nan=0.0, posinf=0.0, neginf=0.0))
            selected.append(
                {
                    "start": start,
                    "end": min(end, start + len(part) / sample_rate),
                    "overlapped": overlapped,
                }
            )
            samples += len(part)
        if samples / sample_rate < minimum_stream_seconds:
            continue
        speakers.append(speaker)
        waves.append(np.concatenate(parts).astype(np.float32))
        metadata.append(
            {
                "speaker": speaker,
                "embedding_seconds": samples / sample_rate,
                "selected_segment_count": len(selected),
                "exclusive_only": exclusive_only,
                "selected_intervals": sorted(selected, key=lambda item: item["start"]),
            }
        )
    return speakers, waves, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Name MOSS speaker streams from mono regions and cross-session enrollment."
    )
    parser.add_argument("--moss-output", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-enrollment-clips", type=int, default=4)
    parser.add_argument("--minimum-segment-seconds", type=float, default=0.35)
    parser.add_argument("--minimum-stream-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-stream-seconds", type=float, default=60.0)
    parser.add_argument("--include-generated-overlap", action="store_true")
    args = parser.parse_args()

    moss = json.loads(args.moss_output.read_text(encoding="utf-8"))
    if bool(moss.get("uses_reference_activity")) or bool(moss.get("uses_isolated_audio")):
        raise ValueError("MOSS output is not a valid mono-only inference artifact")
    audio_path = Path(str(moss.get("audio") or "")).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    mixture = _load_wave(audio_path, sample_rate=SAMPLE_RATE)
    streams, stream_waves, stream_metadata = stream_waveforms(
        mixture,
        list(moss.get("segments") or []),
        sample_rate=SAMPLE_RATE,
        minimum_segment_seconds=args.minimum_segment_seconds,
        minimum_stream_seconds=args.minimum_stream_seconds,
        maximum_stream_seconds=args.maximum_stream_seconds,
        exclusive_only=not args.include_generated_overlap,
    )
    if not streams:
        raise ValueError("No MOSS streams had enough mono evidence for enrollment binding")

    provenance = enrollment_provenance(args.enrollment_manifest, session=args.session)
    if provenance["uses_evaluation_session_audio"]:
        raise ValueError("Enrollment manifest includes evaluation-session audio")
    paths_by_speaker = enrollment_paths(
        args.enrollment_manifest,
        session=args.session,
        max_clips_per_speaker=args.max_enrollment_clips,
    )
    if not paths_by_speaker:
        raise ValueError(f"No enrollment audio found for {args.session}")
    roster = sorted(paths_by_speaker)
    enrollment_labels = [speaker for speaker in roster for _path in paths_by_speaker[speaker]]
    enrollment_waves = [
        trim_clean_enrollment(_load_wave(path), sample_rate=SAMPLE_RATE)
        for speaker in roster
        for path in paths_by_speaker[speaker]
    ]

    model = _load_titanet(args.device)
    enrollment_vectors = _embed_waveforms(
        model,
        enrollment_waves,
        sample_rate=SAMPLE_RATE,
        batch_size=args.batch_size,
        device=args.device,
    )
    stream_vectors = _embed_waveforms(
        model,
        stream_waves,
        sample_rate=SAMPLE_RATE,
        batch_size=args.batch_size,
        device=args.device,
    )
    centroids = np.stack(
        [
            _normalize_rows(
                enrollment_vectors[[label == speaker for label in enrollment_labels]].mean(
                    axis=0, keepdims=True
                )
            )[0]
            for speaker in roster
        ]
    )
    similarity = _normalize_rows(stream_vectors) @ _normalize_rows(centroids).T
    independent, one_to_one, evidence = assignment_maps(similarity, streams, roster)
    payload = {
        "moss_output": str(args.moss_output),
        "mono_audio": str(audio_path),
        "session": args.session,
        "binding_model": "titanet_small",
        "binding_scope": "full-moss-stream",
        "activity_source": "moss-joint-mono-output",
        "inference_uses_isolated_target_audio": False,
        "enrollment_provenance": provenance,
        "enrollment_sources": {
            speaker: [str(path) for path in paths_by_speaker[speaker]] for speaker in roster
        },
        "enrollment_leave_one_out": enrollment_leave_one_out_score(
            enrollment_vectors,
            enrollment_labels,
        ),
        "streams": stream_metadata,
        "independent_mapping": independent,
        "one_to_one_mapping": one_to_one,
        "binding_evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "streams": len(streams),
                "roster_speakers": len(roster),
                "independent_mapping": independent,
                "one_to_one_mapping": one_to_one,
                "binding_evidence": evidence,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
