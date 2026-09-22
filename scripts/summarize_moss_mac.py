#!/usr/bin/env python3
"""Summarize the frozen MOSS comparison, including paired-window uncertainty."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(run, output):
    selection = json.loads((run / "selection.json").read_text())
    selected = selection["method"]
    reports = {
        split: json.loads((run / f"{split}_scores.json").read_text()) for split in ("dev", "test")
    }
    test = reports["test"]
    # Resample original one-minute windows, keeping adjacent halves together.
    groups = {}
    for record in test["records"]:
        key = record["cut_id"].rsplit("_", 1)[0]
        groups.setdefault(key, []).append(record)
    paired = []
    for records in groups.values():
        paired.append(
            [
                sum(r["scores"]["baseline"]["reference_words"] for r in records),
                sum(r["scores"]["baseline"]["named_edit_errors"] for r in records),
                sum(r["scores"][selected]["named_edit_errors"] for r in records),
            ]
        )
    paired = np.asarray(paired)
    rng = np.random.default_rng(20260919)
    samples = paired[rng.integers(len(paired), size=(10000, len(paired)))].sum(axis=1)
    difference = (samples[:, 2] - samples[:, 1]) / np.maximum(samples[:, 0], 1)
    interval = np.quantile(difference, [0.025, 0.975]).tolist()
    summary = dict(
        selection=selection,
        metrics={s: r["metrics"] for s, r in reports.items()},
        paired_bootstrap=dict(
            unit="original 60-second window",
            windows=len(paired),
            resamples=10000,
            seed=20260919,
            test_wer_difference_95_percentile_interval=interval,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "per_speaker.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "speaker", "reference_words", "baseline_f1", "moss_f1"])
        for split, report in reports.items():
            for name, metrics in report["metrics"][selected]["per_speaker"].items():
                writer.writerow(
                    [
                        split,
                        name,
                        metrics["reference_words"],
                        report["metrics"]["baseline"]["per_speaker"][name]["f1"],
                        metrics["f1"],
                    ]
                )
    lines = [
        "# MOSS on the Mac: frozen single-file comparison",
        "",
        "This tests the public MOSS checkpoint, not the unavailable WSL overlap-trained "
        "checkpoint. WSL SSH timed out and its selected checkpoint is absent locally.",
        "",
        "Inference uses only mono mixtures. The existing identity model uses Sessions 55/61; "
        "the conservative correction policy was previously selected on 62. "
        "No MOSS training occurred. The naming variant was chosen on Session 62 before "
        "running Session 63. Session 63 is held out from fitting, but was already used in "
        "the earlier Mac audit; this is not a fresh untouched dataset.",
        "",
        f"Selected naming: `{selected}` (minimum development named word error).",
        "",
        "| Split / method | Named word error ↓ | Timed named-word F1 ↑ | Speaker macro F1 ↑ | Brief-overlap recall ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, report in reports.items():
        for key, label in [
            ("baseline", "Current pipeline"),
            (selected, "MOSS"),
            ("moss_oracle_names", "MOSS + reference-derived names (diagnostic)"),
        ]:
            m = report["metrics"][key]
            brief = m["categories"]["brief_overlap"]
            lines.append(
                f"| {split} / {label} | {m['named_word_error_rate']:.1%} | "
                f"{m['f1']:.1%} | {m['macro_f1']:.1%} | "
                f"{brief['matched_words']}/{brief['reference_words']} ({brief['recall']:.1%}) |"
            )
    lines += [
        "",
        f"Paired bootstrap 95% interval for test named-word-error change (MOSS minus "
        f"baseline): **{interval[0]*100:+.1f} to {interval[1]*100:+.1f} percentage points**. "
        "This resamples 12 windows within one session; it does not estimate uncertainty "
        "across new sessions or errors in automatic reference labels.",
        "",
        "## Decision",
        "",
        "Retain MOSS as an experimental candidate; do not replace the production default from "
        "this evidence. Its test average improves, especially attribution for kinglizard7958 "
        "and travisaurus6985, but the word-error uncertainty interval includes regression. "
        "The large development brief-overlap gain did not repeat on test: only one additional "
        "brief-overlap word was recovered.",
        "",
        "Reference-derived naming recovers 12/35 brief-overlap test words versus 5/35 with "
        "automatic naming. That identifies remaining name-binding errors as well as speech "
        "recovery errors. This oracle uses reference text and cannot be shipped. "
        "The WSL overlap-trained checkpoint remains untested on this Mac; these results "
        "neither validate nor reject that checkpoint. No early-session human accuracy claim "
        "follows from this benchmark.",
        "",
        "## Measurement limits",
        "",
        "- Each split contains the original 12 one-minute windows, decoded as 24 independent "
        "30-second mono clips. All predicted words and all eligible reference words count; "
        "missing speech and duplicate text are penalized.",
        "- References are cached isolated-track Parakeet words, filtered by source VAD. "
        "They are not human gold. Using Parakeet for both the reference and baseline can "
        "favor the baseline's lexical conventions/errors.",
        "- Named word error is speaker-concatenated edit distance per 30-second cut divided "
        "by reference words. A wrong owner can count as both deletion and insertion. "
        "It measures transcription plus attribution, not pure diarization DER.",
        "- Timed matching requires equal normalized token and name, with spans within 0.5 s. "
        "MOSS has coarser segment timestamps, which can favor it on this metric. The named "
        "edit-distance metric is independent of timestamp precision.",
        "- Brief turns are word sequences separated by gaps over 0.3 s and lasting at most "
        "2 s. Overlap comes from intersecting word spans across speakers, not sentence spans.",
        "- Reference-derived names are a diagnostic oracle, never used in deployed naming.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "RUN=.outputs/moss_mac_20260919",
        'CORPUS="$HOME/.cache/transcriber/early-domain-20260918"',
        '.venv/bin/python scripts/test_moss_mac.py prepare --corpus "$CORPUS" --output "$RUN"',
        "# Run infer for dev, score dev, freeze selection, then infer and score test.",
        'HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 "$HOME/.cache/transcriber/moss-mac-env/bin/python" scripts/test_moss_mac.py infer --output "$RUN" --split dev',
        '.venv/bin/python scripts/test_moss_mac.py score --output "$RUN" --split dev --corpus "$CORPUS" --baseline .outputs/early_domain_20260918 --identity .outputs/early_domain_20260918/identity/speaker_model.npz',
        '.venv/bin/python scripts/test_moss_mac.py select --output "$RUN"',
        'HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 "$HOME/.cache/transcriber/moss-mac-env/bin/python" scripts/test_moss_mac.py infer --output "$RUN" --split test',
        '.venv/bin/python scripts/test_moss_mac.py score --output "$RUN" --split test --corpus "$CORPUS" --baseline .outputs/early_domain_20260918 --identity .outputs/early_domain_20260918/identity/speaker_model.npz',
        '.venv/bin/python scripts/summarize_moss_mac.py --run "$RUN" --output docs/analysis/moss_mac_20260919',
        "```",
        "",
        "Exact model revision, dependency versions, audio hashes, experiment plan, "
        "per-window predictions and scores are retained in the run directory. The existing "
        "app environment and production defaults were not replaced.",
        "",
        "Validation: 16 focused comparison, scoring, and identity tests passed. "
        "Ruff and Black pass for the new experiment files. Official MOSS inference code "
        "commit is recorded in provenance.json; dependencies are frozen in moss_requirements.txt.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["paired_bootstrap"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.output)
