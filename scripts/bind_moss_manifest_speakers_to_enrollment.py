from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind_moss_speakers_to_enrollment import (
    SAMPLE_RATE,
    _normalize_rows,
    stream_waveforms,
)
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


def validate_mono_output(payload: dict) -> None:
    if bool(payload.get("uses_reference_activity")) or bool(payload.get("uses_isolated_audio")):
        raise ValueError("MOSS output is not a valid mono-only inference artifact")
    for record in list(payload.get("records") or []):
        if bool(record.get("uses_reference_activity")) or bool(record.get("uses_isolated_audio")):
            raise ValueError("A MOSS manifest record used privileged inference audio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind window-local MOSS streams to historical speaker enrollment in one batch."
    )
    parser.add_argument("--moss-output", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-enrollment-clips", type=int, default=4)
    parser.add_argument("--minimum-segment-seconds", type=float, default=0.2)
    parser.add_argument("--minimum-stream-seconds", type=float, default=0.25)
    parser.add_argument("--maximum-stream-seconds", type=float, default=20.0)
    parser.add_argument(
        "--exclusive-only",
        action="store_true",
        help="Exclude generated overlap regions from identity evidence.",
    )
    args = parser.parse_args()

    moss = json.loads(args.moss_output.read_text(encoding="utf-8"))
    validate_mono_output(moss)
    records = list(moss.get("records") or [])
    if not records:
        raise ValueError("MOSS output has no manifest records")

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

    stream_waves = []
    prepared_records = []
    for record in records:
        cut_id = str(record.get("cut_id") or "").strip()
        audio_path = Path(str(record.get("audio") or "")).expanduser().resolve()
        if not cut_id or not audio_path.is_file():
            raise ValueError(f"Invalid MOSS manifest record: {cut_id or '<missing-cut-id>'}")
        mixture = _load_wave(audio_path, sample_rate=SAMPLE_RATE)
        streams, waves, metadata = stream_waveforms(
            mixture,
            list(record.get("segments") or []),
            sample_rate=SAMPLE_RATE,
            minimum_segment_seconds=args.minimum_segment_seconds,
            minimum_stream_seconds=args.minimum_stream_seconds,
            maximum_stream_seconds=args.maximum_stream_seconds,
            exclusive_only=args.exclusive_only,
        )
        first_vector = len(stream_waves)
        stream_waves.extend(waves)
        prepared_records.append(
            {
                "cut_id": cut_id,
                "audio": str(audio_path),
                "streams": streams,
                "stream_metadata": metadata,
                "first_vector": first_vector,
                "vector_count": len(waves),
            }
        )
    if not stream_waves:
        raise ValueError("No MOSS streams had enough mono evidence for enrollment binding")

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

    bindings = []
    for prepared in prepared_records:
        first = int(prepared.pop("first_vector"))
        count = int(prepared.pop("vector_count"))
        streams = list(prepared.pop("streams"))
        vectors = stream_vectors[first : first + count]
        if count:
            similarity = _normalize_rows(vectors) @ _normalize_rows(centroids).T
            independent, one_to_one, evidence = assignment_maps(
                similarity,
                streams,
                roster,
            )
        else:
            independent, one_to_one, evidence = {}, {}, {}
        bindings.append(
            {
                **prepared,
                "independent_mapping": independent,
                "one_to_one_mapping": one_to_one,
                "binding_evidence": evidence,
            }
        )

    payload = {
        "moss_output": str(args.moss_output),
        "session": args.session,
        "binding_model": "titanet_small",
        "binding_scope": "per-window-moss-stream",
        "activity_source": "moss-joint-mono-output",
        "inference_uses_isolated_target_audio": False,
        "generated_overlap_used_for_binding": not args.exclusive_only,
        "enrollment_provenance": provenance,
        "enrollment_sources": {
            speaker: [str(path) for path in paths_by_speaker[speaker]] for speaker in roster
        },
        "enrollment_leave_one_out": enrollment_leave_one_out_score(
            enrollment_vectors,
            enrollment_labels,
        ),
        "record_bindings": bindings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "record_count": len(bindings),
                "stream_count": len(stream_waves),
                "roster_speakers": len(roster),
                "generated_overlap_used_for_binding": not args.exclusive_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
