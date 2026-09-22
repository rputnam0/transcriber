from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf

from train_se_dicow_cutset_adapter import (
    SAMPLE_RATE,
    _read_jsonl,
    _safe_id,
    enrollment_index,
    load_enrollment,
)


def _enrollment_seconds(row: Mapping[str, object]) -> float:
    return sum(
        max(0.0, float(span.get("duration") or 0.0))
        for span in list(row.get("positive_enrollment_spans") or [])
    )


def select_diverse_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    max_clips: int,
) -> list[Mapping[str, object]]:
    best_by_session: dict[str, Mapping[str, object]] = {}
    for row in rows:
        session = str(row.get("session") or "")
        current = best_by_session.get(session)
        if current is None or _enrollment_seconds(row) > _enrollment_seconds(current):
            best_by_session[session] = row
    candidates = sorted(
        best_by_session.values(),
        key=lambda row: (-_enrollment_seconds(row), str(row.get("session") or "")),
    )
    return candidates[:max_clips]


def materialize_profiles(
    *,
    enrollment_manifest: Path,
    stem_root: Path,
    output_dir: Path,
    split: str,
    evaluation_sessions: Sequence[str],
    max_clips_per_speaker: int,
    profile_seconds: float,
    speakers: set[str] | None = None,
) -> dict:
    index = enrollment_index(_read_jsonl(enrollment_manifest), split=split)
    if speakers is not None:
        missing = sorted(speakers - set(index))
        if missing:
            raise ValueError(f"No enrollment rows found for: {', '.join(missing)}")
        index = {speaker: index[speaker] for speaker in sorted(speakers)}
    if not index:
        raise ValueError(f"No enrollment rows found for split {split!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seconds_per_clip = profile_seconds / max(1, max_clips_per_speaker)
    manifest_rows = []
    profiles = {}
    for speaker, speaker_rows in index.items():
        selected = select_diverse_rows(speaker_rows, max_clips=max_clips_per_speaker)
        waves = [
            load_enrollment(
                row,
                stem_root=stem_root,
                max_seconds=seconds_per_clip,
            )
            for row in selected
        ]
        packed = np.concatenate(waves).astype(np.float32)
        target_samples = int(round(profile_seconds * SAMPLE_RATE))
        packed = np.pad(packed[:target_samples], (0, max(0, target_samples - len(packed))))
        profile_path = output_dir / f"{_safe_id(speaker)}.wav"
        sf.write(profile_path, packed, SAMPLE_RATE, subtype="PCM_16")
        source_rows = [
            {
                "session": str(row.get("session") or ""),
                "target_member": str(row.get("target_member") or ""),
                "enrollment_seconds": _enrollment_seconds(row),
            }
            for row in selected
        ]
        profiles[speaker] = {
            "path": str(profile_path.resolve()),
            "source_rows": source_rows,
        }
        for evaluation_session in evaluation_sessions:
            manifest_rows.append(
                {
                    "session": evaluation_session,
                    "speaker_id": speaker,
                    "window_start": 0.0,
                    "split_id": "cross-session-enrollment",
                    "materialized": {"positive_enrollment_paths": [str(profile_path.resolve())]},
                    "enrollment_source_split": split,
                    "enrollment_source_sessions": [row["session"] for row in source_rows],
                    "uses_evaluation_session_audio": False,
                }
            )
    output_manifest = output_dir / "cross_session_enrollment_manifest.jsonl"
    output_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "enrollment_manifest": str(enrollment_manifest),
        "stem_root": str(stem_root),
        "source_split": split,
        "evaluation_sessions": list(evaluation_sessions),
        "speaker_count": len(profiles),
        "profile_seconds": profile_seconds,
        "max_clips_per_speaker": max_clips_per_speaker,
        "uses_evaluation_session_audio": False,
        "clean_energy_trimming_scope": "isolated-enrollment-only",
        "output_manifest": str(output_manifest),
        "profiles": profiles,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean known-speaker profiles using training sessions only."
    )
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--stem-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--evaluation-sessions",
        default="Session 63,Session 66,Session 64,Session 67",
    )
    parser.add_argument("--max-clips-per-speaker", type=int, default=4)
    parser.add_argument("--profile-seconds", type=float, default=30.0)
    parser.add_argument(
        "--speakers",
        help="Optional comma-separated expected roster; defaults to every speaker in the split.",
    )
    args = parser.parse_args()
    if args.max_clips_per_speaker < 1:
        parser.error("--max-clips-per-speaker must be positive")
    if not math.isfinite(args.profile_seconds) or args.profile_seconds <= 0.0:
        parser.error("--profile-seconds must be positive")
    sessions = [value.strip() for value in args.evaluation_sessions.split(",") if value.strip()]
    speakers = (
        {value.strip() for value in args.speakers.split(",") if value.strip()}
        if args.speakers
        else None
    )
    materialize_profiles(
        enrollment_manifest=args.enrollment_manifest,
        stem_root=args.stem_root,
        output_dir=args.output_dir,
        split=args.split,
        evaluation_sessions=sessions,
        max_clips_per_speaker=args.max_clips_per_speaker,
        profile_seconds=args.profile_seconds,
        speakers=speakers,
    )


if __name__ == "__main__":
    main()
