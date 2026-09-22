"""Summarize automatic voice evidence without treating it as proof of attendance."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(deployment):
    transcript = json.loads((deployment / "named.json").read_text())
    grouped = defaultdict(list)
    for turn in transcript["segments"]:
        grouped[turn["speaker"]].append(turn)
    rows = []
    for speaker, turns in sorted(grouped.items(), key=lambda item: str(item[0])):
        durations = [turn["end"] - turn["start"] for turn in turns]
        rows.append(
            {
                "speaker": speaker,
                "turns": len(turns),
                "sum_turn_durations_seconds": round(sum(durations), 3),
                "longest_turn_seconds": round(max(durations), 3),
                "turns_longer_than_5_seconds": sum(duration > 5 for duration in durations),
                "review_flagged_turns": sum(turn["review_required"] for turn in turns),
                "uncertain_enrollment_turns": sum(
                    "uncertain enrollment" in turn["review_reasons"] for turn in turns
                ),
            }
        )
    assert sum(row["turns"] for row in rows) == len(transcript["segments"])
    return {"deployment": str(deployment), "speakers": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "note": "Automatic assignments and heuristic review flags are not attendance ground truth or calibrated accuracy. Durations are summed turn lengths and may overlap.",
        "recordings": [summarize(path) for path in args.deployment],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for recording in report["recordings"]:
        print(Path(recording["deployment"]).name)
        for row in recording["speakers"]:
            print(
                f"  {row['speaker']}: {row['turns']} turns, "
                f"{row['sum_turn_durations_seconds']:.1f}s, "
                f"{row['review_flagged_turns']} review flags"
            )


if __name__ == "__main__":
    main()
