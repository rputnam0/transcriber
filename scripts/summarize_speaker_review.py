"""Summarize human turn grades against immutable original predictions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(transcript_path, review_path, roster):
    transcript_bytes = transcript_path.read_bytes()
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    digest = hashlib.sha256(transcript_bytes).hexdigest()
    if digest != review["transcript_sha256"]:
        raise ValueError("Review and transcript hashes differ")
    segments = json.loads(transcript_bytes)["segments"]
    counts = Counter()
    overlap_counts = Counter()
    errors = []
    reviewed_end = 0
    for key, grade in review["reviews"].items():
        verdict = grade["verdict"]
        if verdict == "ungraded":
            continue
        i = int(key)
        turn = segments[i]
        counts[verdict] += 1
        reviewed_end = max(reviewed_end, turn["end"])
        overlaps = any(
            j != i and min(other["end"], turn["end"]) > max(other["start"], turn["start"])
            for j, other in enumerate(segments)
        )
        if overlaps:
            overlap_counts[verdict] += 1
        if verdict == "wrong":
            errors.append(
                {
                    "turn_id": i,
                    "start": turn["start"],
                    "end": turn["end"],
                    "predicted_speaker": turn["speaker"],
                    "predicted_speaker_absent": turn["speaker"] not in roster,
                }
            )
    denominator = counts["correct"] + counts["wrong"]
    return {
        "transcript_sha256": digest,
        "review_snapshot_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "review_updated_at": review.get("updated_at"),
        "confirmed_roster": roster,
        "graded_turns": sum(counts.values()),
        "grades": dict(counts),
        "correct_fraction_among_decisive_grades": (
            counts["correct"] / denominator if denominator else None
        ),
        "latest_reviewed_end_seconds": reviewed_end,
        "overlapping_turn_grades": dict(overlap_counts),
        "overlap_definition": "Positive intersection of original predicted intervals; not human overlap annotation.",
        "errors": errors,
        "scope": "User-reviewed initial portion of one recording; turn grades, not word accuracy or DER.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--roster", nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.transcript, args.review, args.roster)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
