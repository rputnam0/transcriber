"""Reattribute completed recordings without retranscribing or changing review snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from diarize_moss_recording import attribute
from export_moss_transcripts import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--rosters", type=Path, default=Path("config/early_session_rosters.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    summaries = []
    for entry in plan["recordings"]:
        source = Path(entry["deployment"])
        target = args.output / f"session{entry['session']}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ["manifest.json", "source.json"]:
            destination = target / name
            if destination.exists() and destination.read_bytes() != (source / name).read_bytes():
                raise ValueError(f"Changed immutable input: {destination}")
            shutil.copy2(source / name, destination)
        link = target / "predictions"
        if not link.exists():
            link.symlink_to((source / "predictions").resolve(), target_is_directory=True)
        if link.resolve() != (source / "predictions").resolve():
            raise ValueError("Prediction cache points to another source")
        attribute(
            SimpleNamespace(
                output=target,
                identity=args.identity,
                policy=args.policy,
                method="moss_hybrid_local_names",
                session=entry["session"],
                rosters=args.rosters,
                device="mps",
                baseline=source,
            )
        )
        old = json.loads((source / "named.json").read_text())["segments"]
        new = json.loads((target / "named.json").read_text())["segments"]

        def immutable(turns):
            return [
                (t["start"], t["end"], t["text"], t["local_speaker"], t["cut_id"]) for t in turns
            ]

        if immutable(old) != immutable(new):
            raise AssertionError("Reattribution changed words, timing, or turn identity")
        export(target, args.rosters, entry["session"], args.transcripts)
        summaries.append(
            dict(
                session=entry["session"],
                turns=len(new),
                changed=sum(a["speaker"] != b["speaker"] for a, b in zip(old, new)),
            )
        )
        entry["baseline_deployment"] = str(source)
        entry["deployment"] = str(target.resolve())
        print("UPGRADED", summaries[-1], flush=True)
    (args.output / "processing_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
