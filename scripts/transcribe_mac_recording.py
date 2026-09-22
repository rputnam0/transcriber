"""Resume a complete mixed-recording transcript with private enrolled voices on a Mac.

Run in the application environment; ASR inference uses --asr-python. Each recording
gets its own work/output directories. Previous audio or model caches are never reused
for a different recording. See docs/mac-single-file-transcription.md.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_mac_asr_quality import save


def build_plan(audio, session, work, rosters):
    roster = json.loads(rosters.read_text())["session_rosters"].get(str(session))
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("Provide a nonempty, unique roster for this session")
    source = work / "prepared" / f"session{session}"
    return dict(
        recordings=[
            dict(
                session=session,
                audio=str(audio.resolve()),
                deployment=str(source.resolve()),
                baseline_deployment=str(source.resolve()),
                source_note="Entire supplied recording; completeness of the original must be verified separately.",
            )
        ]
    )


def commands(args, plan):
    scripts = Path(__file__).resolve().parent

    def command(python, script, *options):
        return [str(python), str(scripts / script), *map(str, options)]

    app = sys.executable
    source = args.work / "prepared" / f"session{args.session}"
    common = ["--plan", plan]
    jobs = []
    models = args.models or args.work / "models.json"
    if args.models is None:
        jobs.append(command(args.asr_python, "resolve_mac_asr_models.py", "--output", models))
    jobs += [
        command(
            app, "diarize_moss_recording.py", "prepare", "--audio", args.audio, "--output", source
        ),
        command(
            args.asr_python,
            "run_moss_mlx_recordings.py",
            *common,
            "--model",
            args.assets / "moss-checkpoint",
            "--output",
            args.work / "moss",
            "--batch-size",
            args.batch_size,
        ),
        command(
            args.asr_python,
            "run_context_asr_recordings.py",
            *common,
            "--models",
            models,
            "--output",
            args.work / "asr",
            "--batch-size",
            args.batch_size,
        ),
        command(
            args.asr_python,
            "repair_context_asr.py",
            "--root",
            args.work / "asr",
            "--models",
            models,
        ),
        command(
            app,
            "attribute_mac_recordings.py",
            *common,
            "--work",
            args.work,
            "--identity",
            args.assets / "speaker_model.npz",
            "--policy",
            args.assets / "turn_refinement.json",
            "--rosters",
            args.rosters,
        ),
        command(
            app,
            "prepare_transcript_listening_audio.py",
            *common,
            "--output",
            args.work / "listening",
        ),
        command(
            app,
            "finish_mac_asr_release.py",
            *common,
            "--work",
            args.work,
            "--output",
            args.output,
            "--rosters",
            args.rosters,
        ),
    ]
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["audio", "assets", "work", "output", "asr-python"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--rosters", type=Path)
    p.add_argument("--models", type=Path, help="Optional existing resolved model cache manifest")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.session < 1:
        p.error("Session and batch size must be positive")
    args.rosters = args.rosters or args.assets / "rosters.json"
    for path in [
        args.audio,
        args.asr_python,
        args.rosters,
        args.assets / "speaker_model.npz",
        args.assets / "turn_refinement.json",
        args.assets / "moss-checkpoint/config.json",
        args.assets / "moss-checkpoint/model.safetensors",
    ]:
        if not path.is_file():
            p.error(f"Required file is missing: {path}")
    data = build_plan(args.audio, args.session, args.work, args.rosters)
    plan = args.work / "processing_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != data:
        p.error("Work directory belongs to different inputs; choose a fresh directory")
    if not args.dry_run:
        save(plan, data)
    for job in commands(args, plan):
        print(json.dumps(job), flush=True)
        if not args.dry_run:
            subprocess.run(job, check=True)


if __name__ == "__main__":
    main()
