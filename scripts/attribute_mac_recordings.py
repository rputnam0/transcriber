"""Apply frozen enrollment/local-utterance policy to fresh MOSS recording outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument(
        "--identity",
        type=Path,
        default=Path(".outputs/early_domain_20260918/identity/speaker_model.npz"),
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=Path(".outputs/early_domain_20260918/diarization_dev/turn_refinement.json"),
    )
    p.add_argument("--rosters", type=Path, default=Path("config/early_session_rosters.json"))
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    scripts = Path(__file__).resolve().parent
    for entry in json.loads(args.plan.read_text())["recordings"]:
        session = entry["session"]
        target = args.work / "moss" / f"session{session}"
        while True:
            manifest = target / "manifest.json"
            records = json.loads(manifest.read_text()) if manifest.exists() else []
            if records and all(
                (target / "predictions" / f"{r['cut_id']}.json").exists() for r in records
            ):
                break
            if not args.wait:
                raise RuntimeError(f"Session {session} inference is incomplete")
            time.sleep(10)
        subprocess.run(
            [
                sys.executable,
                str(scripts / "diarize_moss_recording.py"),
                "attribute",
                "--output",
                str(target),
                "--identity",
                str(args.identity),
                "--policy",
                str(args.policy),
                "--method",
                "moss_hybrid_local_names",
                "--session",
                str(session),
                "--rosters",
                str(args.rosters),
                "--device",
                "mps",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(scripts / "export_moss_transcripts.py"),
                "--deployment",
                str(target),
                "--session",
                str(session),
                "--rosters",
                str(args.rosters),
                "--output",
                str(args.work / "moss_transcripts"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
