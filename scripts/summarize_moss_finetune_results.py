#!/usr/bin/env python3
"""Summarize a frozen candidate against public MOSS and the original named ASR baseline."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from test_moss_mac import combine, save


def paired_interval(public_records, candidate_records, method, repeats=5000):
    """Resample paired one-minute windows within each session; keep both halves together."""
    paired = {r["cut_id"]: r for r in candidate_records}
    groups = defaultdict(lambda: defaultdict(lambda: np.zeros(3, float)))
    for row in public_records:
        other = paired[row["cut_id"]]
        if row["reference"] != other["reference"]:
            raise ValueError("Cannot compare different references")
        session, minute, _ = row["cut_id"].split("_")
        base, extra = row["scores"]["moss_named"], other["scores"][method]
        groups[session][minute] += [
            base["named_edit_errors"],
            extra["named_edit_errors"],
            base["reference_words"],
        ]
    matrices = [np.stack(list(minutes.values())) for minutes in groups.values()]
    rng = np.random.default_rng(20260919)
    values = []
    for _ in range(repeats):
        total = sum(
            matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0) for matrix in matrices
        )
        values.append((total[1] - total[0]) / max(total[2], 1))
    return dict(
        candidate_minus_public_95_percentile_interval=np.percentile(values, [2.5, 97.5]).tolist(),
        repeats=repeats,
        unit="one-minute window, stratified within each session",
        caveat="Conditional on these sessions and automatic references; does not estimate new-session or annotation uncertainty.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    public, candidate = [json.loads(path.read_text()) for path in [args.public, args.candidate]]
    if public["split"] != candidate["split"]:
        raise ValueError("Split mismatch")
    sources = [
        ("baseline", public, "baseline"),
        ("public_moss", public, "moss_named"),
        ("candidate", candidate, args.method),
    ]
    methods, by_session, by_speaker = {}, [], []
    for label, report, method in sources:
        methods[label] = report["metrics"][method]
        grouped = defaultdict(list)
        for row in report["records"]:
            grouped[row["cut_id"].split("_")[0]].append(row["scores"][method])
        for session, scores in sorted(grouped.items()):
            metric = combine(scores)
            by_session.append(
                dict(
                    session=session,
                    method=label,
                    named_word_error_rate=metric["named_word_error_rate"],
                    speaker_macro_f1=metric["macro_f1"],
                    brief_overlap_recall=metric["categories"]["brief_overlap"]["recall"],
                )
            )
        for name, metric in methods[label]["per_speaker"].items():
            by_speaker.append(
                dict(
                    speaker=name,
                    method=label,
                    reference_words=metric["reference_words"],
                    precision=metric["precision"],
                    recall=metric["recall"],
                    f1=metric["f1"],
                )
            )
    result = dict(
        split=public["split"],
        seconds=public["seconds"],
        metrics=methods,
        candidate_method=args.method,
        interval=paired_interval(public["records"], candidate["records"], args.method),
        references=public["reference_type"],
        reference_root=public.get("reference_root"),
    )
    save(args.output / "comparison.json", result)
    for filename, rows in [("by_session.csv", by_session), ("by_speaker.csv", by_speaker)]:
        with (args.output / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "methods": {k: v["named_word_error_rate"] for k, v in methods.items()},
                "interval": result["interval"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
