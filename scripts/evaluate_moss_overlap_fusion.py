#!/usr/bin/env python3
"""Test adding only confident brief overlapping turns to a frozen base transcript.

The fusion rule consumes predicted turns and enrollment evidence only. References are
used afterward for scoring. No base words are removed or renamed.
"""
import argparse
import json
from pathlib import Path

from evaluate_se_dicow_oracle_cut import normalized_words
from prepare_early_domain_corpus import NAMES
from score_moss_manifest import temporal_attributed_score
from test_moss_mac import combine, named_edit_errors, save


def fuse(base, extra, bindings):
    output, additions = list(base), []
    for turn in extra:
        duration = turn["end"] - turn["start"]
        if not 0.1 <= duration <= 2.0 or len(normalized_words(turn["text"])) > 6:
            continue
        evidence = [b for b in bindings.values() if b.get("proposed_speaker") == turn["speaker"]]
        # The caller supplies each turn's original binding where available. Never borrow
        # confidence from a different local cluster merely because it has the same name.
        binding = turn.get("enrollment") or (evidence[0] if len(evidence) == 1 else {})
        if not (
            binding.get("mean_posterior", 0) >= 0.75
            and binding.get("chunk_agreement", 0) >= 0.6
            and binding.get("cosine", 0) >= 0.35
        ):
            continue
        if any(
            t["speaker"] == turn["speaker"]
            and min(t["end"] + 0.3, turn["end"]) > max(t["start"] - 0.3, turn["start"])
            for t in output
        ):
            continue
        if not any(
            t["speaker"] != turn["speaker"]
            and min(t["end"], turn["end"]) - max(t["start"], turn["start"])
            >= max(0.1, 0.25 * duration)
            for t in base
        ):
            continue
        additions.append(turn)
        output.append(turn)
    return sorted(output, key=lambda t: (t["start"], t["end"])), additions


def metrics(reference, turns):
    names = {n: n for n in NAMES}
    score = temporal_attributed_score(
        reference, turns, names, brief_turn_seconds=2, tolerance_seconds=0.5
    )
    score["named_edit_errors"] = named_edit_errors(reference, turns, names)
    score["per_speaker"] = {
        name: temporal_attributed_score(
            [r for r in reference if r["speaker"] == name],
            [t for t in turns if t["speaker"] == name],
            names,
            brief_turn_seconds=2,
            tolerance_seconds=0.5,
        )
        for name in NAMES
    }
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--extra", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base, extra = [json.loads(path.read_text()) for path in [args.base, args.extra]]
    candidates = {r["cut_id"]: r for r in extra["records"]}
    scores, records = [], []
    for row in base["records"]:
        candidate = candidates[row["cut_id"]]
        if row["reference"] != candidate["reference"]:
            raise ValueError("Reference versions differ")
        turns, additions = fuse(
            row["named_segments"], candidate["named_segments"], candidate["bindings"]
        )
        score = metrics(row["reference"], turns)
        scores.append(score)
        records.append(
            dict(cut_id=row["cut_id"], scores=score, additions=additions, named_segments=turns)
        )
    result = dict(
        split=base["split"],
        criterion="fixed conservative overlap-addition rule",
        metrics=combine(scores),
        records=records,
    )
    save(args.output, result)
    print(
        json.dumps(
            dict(metrics=result["metrics"], added_turns=sum(len(r["additions"]) for r in records)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
