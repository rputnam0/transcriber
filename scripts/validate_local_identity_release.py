"""Compare a reattributed transcript release with frozen, eligible human grades."""

import argparse
import hashlib
import json
from pathlib import Path


def validate(analysis, transcripts):
    rows = json.loads((analysis / "turns.json").read_text())
    rows = [
        r
        for r in rows
        if not r["legacy"] and r["positive_label_eligible"] and r["verdict"] != "unsure"
    ]
    files = {s: transcripts / f"Session {s}.turns.json" for s in {r["session"] for r in rows}}
    turns = {s: json.loads(p.read_text())["segments"] for s, p in files.items()}
    before = after = fixed = broken = 0
    changes = []
    for row in rows:
        new = turns[row["session"]][row["turn_id"]]
        if any(new[key] != row[key] for key in ["start", "end", "text", "cut_id", "local_speaker"]):
            raise ValueError("Review no longer identifies the same utterance")
        was_right = row["predicted"] == row["truth"]
        is_right = new["speaker"] == row["truth"]
        before += was_right
        after += is_right
        fixed += not was_right and is_right
        broken += was_right and not is_right
        if new["speaker"] != row["predicted"]:
            changes.append(
                dict(
                    session=row["session"],
                    turn_id=row["turn_id"],
                    start=row["start"],
                    before=row["predicted"],
                    after=new["speaker"],
                    truth=row["truth"],
                )
            )
    return dict(
        total=len(rows),
        before_correct=before,
        after_correct=after,
        fixed=fixed,
        broken=broken,
        passed=broken == 0 and after >= before,
        changes=changes,
        analysis_sha256=hashlib.sha256((analysis / "turns.json").read_bytes()).hexdigest(),
        transcript_sha256={s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in files.items()},
        caveat="User-selected reviewed turns only; not whole-recording accuracy. Unsure labels excluded.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = validate(args.analysis, args.transcripts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Human-review regression gate failed")


if __name__ == "__main__":
    main()
