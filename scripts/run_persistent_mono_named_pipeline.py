from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_ROOT = Path(__file__).resolve().parent


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def retarget_enrollment_manifest(
    source: Path,
    output: Path,
    *,
    session: str,
) -> dict:
    rows_by_speaker: dict[str, dict] = {}
    for raw in _read_jsonl(source):
        row = dict(raw)
        speaker = str(row.get("speaker_id") or "").strip()
        if bool(row.get("uses_evaluation_session_audio", True)):
            continue
        materialized = dict(row.get("materialized") or {})
        paths = list(materialized.get("positive_enrollment_paths") or [])
        if speaker and paths and speaker not in rows_by_speaker:
            resolved_paths = []
            for value in paths:
                path = Path(str(value)).expanduser()
                candidates = (path, Path.cwd() / path, source.resolve().parent / path)
                resolved = next(
                    (candidate.resolve() for candidate in candidates if candidate.exists()), None
                )
                if resolved is None:
                    raise FileNotFoundError(f"Enrollment audio does not exist: {value}")
                resolved_paths.append(str(resolved))
            materialized["positive_enrollment_paths"] = resolved_paths
            row["materialized"] = materialized
            row["session"] = session
            rows_by_speaker[speaker] = row
    if not rows_by_speaker:
        raise ValueError(f"No cross-session materialized enrollment profiles found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows_by_speaker.values()),
        encoding="utf-8",
    )
    return {
        "session": session,
        "speaker_count": len(rows_by_speaker),
        "speakers": sorted(rows_by_speaker),
        "uses_evaluation_session_audio": False,
        "manifest": str(output),
    }


def _run(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            check=True,
            cwd=SCRIPT_ROOT.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def cut_binding_maps(path: Path, *, mode: str) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record.get("cut_id") or ""): {
            str(slot): str(speaker)
            for slot, speaker in dict(record.get(f"{mode}_mapping") or {}).items()
        }
        for record in list(payload.get("records") or [])
    }


def write_empty_cut_result(path: Path, *, cut_id: str, session: str) -> None:
    path.write_text(
        json.dumps(
            {
                "cut_id": cut_id,
                "session": session,
                "activity_source": "mono-sortformer-enrollment-binding",
                "uses_reference_activity": False,
                "uses_reference_slot_mapping": False,
                "inference_uses_isolated_target_audio": False,
                "speakers": [],
                "records": [],
                "skipped_reason": "no-active-sortformer-slot",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce a named, review-aware transcript from one mono conversation file."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--sortformer-restore", type=Path, required=True)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        action="append",
        required=True,
        help="Adapter chain in chronological order; repeat for each stage.",
    )
    parser.add_argument("--session-number", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--binding-mode", choices=("independent", "one_to_one"), default="one_to_one"
    )
    parser.add_argument("--binding-margin-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-review-threshold", type=float, default=0.25)
    parser.add_argument("--target-mask-broadening-alpha", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = args.output_dir / "prepared"
    activity_dir = args.output_dir / "sortformer"
    result_dir = args.output_dir / "target_asr"
    result_dir.mkdir(parents=True, exist_ok=True)
    session = f"Session {args.session_number}"
    runtime_enrollment = args.output_dir / "runtime_enrollment_manifest.jsonl"
    enrollment_summary = retarget_enrollment_manifest(
        args.enrollment_manifest,
        runtime_enrollment,
        session=session,
    )

    _run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "prepare_mono_audio_cutset.py"),
            "--audio",
            str(args.audio),
            "--output-dir",
            str(prepared_dir),
            "--session-number",
            str(args.session_number),
        ],
        log_path=args.output_dir / "prepare.log",
    )
    cutset = prepared_dir / "mono_cuts.jsonl.gz"
    activity_cutset = activity_dir / "mono_cuts_sortformer.jsonl.gz"
    _run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "build_contiguous_sortformer_mask_cutset.py"),
            "--input-cuts",
            str(cutset),
            "--output-cuts",
            str(activity_cutset),
            "--restore-path",
            str(args.sortformer_restore),
            "--device",
            args.device,
            "--batch-size",
            "1",
        ],
        log_path=args.output_dir / "sortformer.log",
    )
    binding_json = activity_dir / "enrollment_binding.json"
    _run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "evaluate_sortformer_enrollment_binding.py"),
            "--activity-cutset",
            str(activity_cutset),
            "--enrollment-manifest",
            str(runtime_enrollment),
            "--output",
            str(binding_json),
            "--device",
            args.device,
        ],
        log_path=args.output_dir / "binding.log",
    )

    cut_ids = [str(cut["id"]) for cut in _read_jsonl(cutset)]
    mappings_by_cut = cut_binding_maps(binding_json, mode=args.binding_mode)
    for cut_index, cut_id in enumerate(cut_ids):
        output_path = result_dir / f"{cut_id}.json"
        if not mappings_by_cut.get(cut_id):
            write_empty_cut_result(output_path, cut_id=cut_id, session=session)
            print(f"skipped silent cut {cut_index + 1}/{len(cut_ids)}", flush=True)
            continue
        command = [
            sys.executable,
            str(SCRIPT_ROOT / "evaluate_se_dicow_oracle_cut.py"),
            "--cutset",
            str(cutset),
            "--cut-id",
            cut_id,
            "--enrollment-manifest",
            str(runtime_enrollment),
            "--output",
            str(output_path),
        ]
        for adapter_dir in args.adapter_dir[:-1]:
            command.extend(["--initial-adapter-dir", str(adapter_dir)])
        command.extend(
            [
                "--adapter-dir",
                str(args.adapter_dir[-1]),
                "--conditioning",
                "sortformer-enrollment",
                "--activity-cutset",
                str(activity_cutset),
                "--slot-mapping-json",
                str(binding_json),
                "--slot-mapping-mode",
                args.binding_mode,
                "--activity-mask-mode",
                "soft",
                "--speaker-set",
                "enrollment-roster",
                "--target-mask-broadening-alpha",
                str(args.target_mask_broadening_alpha),
                "--device",
                args.device,
            ]
        )
        _run(command, log_path=result_dir / f"{cut_id}.log")
        print(f"transcribed cut {cut_index + 1}/{len(cut_ids)}", flush=True)

    output_json = args.output_dir / "named_transcript_with_review.json"
    output_text = args.output_dir / "named_transcript_with_review.txt"
    preparation = json.loads(
        (prepared_dir / "mono_cutset_summary.json").read_text(encoding="utf-8")
    )
    _run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "render_se_dicow_named_transcript.py"),
            "--input-dir",
            str(result_dir),
            "--binding-json",
            str(binding_json),
            "--output-json",
            str(output_json),
            "--output-text",
            str(output_text),
            "--binding-mode",
            args.binding_mode,
            "--binding-margin-threshold",
            str(args.binding_margin_threshold),
            "--overlap-review-threshold",
            str(args.overlap_review_threshold),
            "--max-duration",
            str(preparation["source_duration"]),
        ],
        log_path=args.output_dir / "render.log",
    )
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    summary = {
        "audio": str(args.audio.resolve()),
        "output_text": str(output_text.resolve()),
        "output_json": str(output_json.resolve()),
        "cut_count": len(cut_ids),
        "skipped_silent_cut_count": sum(not mappings_by_cut.get(cut_id) for cut_id in cut_ids),
        "adapter_dirs": [str(path) for path in args.adapter_dir],
        "target_mask_broadening_alpha": args.target_mask_broadening_alpha,
        "segment_count": rendered["segment_count"],
        "review_segment_count": rendered["review_segment_count"],
        "enrollment": enrollment_summary,
        "mono_only_inference": True,
    }
    (args.output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
