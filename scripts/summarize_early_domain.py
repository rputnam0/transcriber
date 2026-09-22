#!/usr/bin/env python3
"""Rebuild the compact EDA report and plots from saved experiment results (no GPU)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from prepare_early_domain_corpus import NAMES
from train_early_speaker_identity import scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def read(path):
        return json.loads(path.read_text())

    identity = read(args.run / "identity/results.json")
    train = dict(np.load(args.run / "identity/train_features.npz"))
    test = dict(np.load(args.run / "identity/test_features.npz"))
    model = dict(np.load(args.run / "identity/speaker_model.npz"))
    clean = LogisticRegression(C=100, class_weight="balanced", max_iter=2000, random_state=20260918)
    clean.fit(train["clean"], train["labels"])
    # Comparators selected on development; test scores do not choose or revise the model.
    ablation = {
        "centroid_20": dict(
            dev=identity["dev"]["centroid_20"],
            test=scores(dict(weights=model["centroids"] * 20, bias=np.zeros(6)), test),
        ),
        "clean_C100": dict(
            dev=identity["dev"]["clean_C100"],
            test=scores(dict(weights=clean.coef_, bias=clean.intercept_), test),
        ),
        "augmented_C100": dict(dev=identity["dev"]["augmented_C100"], test=identity["test"]),
    }
    summary = dict(identity_ablation=ablation, train_counts=identity["train_counts"])
    for name in ["diarization_dev", "diarization_test"]:
        summary[name] = read(args.run / name / "turn_refinement.json")
    for name in ["words_dev", "words_test", "long_dev", "long_test"]:
        summary[name] = read(args.run / name / "results.json")
    summary["named_activity_rejected"] = read(args.run / "named_activity_extended/selection.json")
    summary["early_session_stereo"] = read(args.run / "session1/stereo_eda.json")
    manifest = read(args.corpus / "manifest.json")
    summary["early_noise"] = manifest["early_noise"]
    summary["level_eda"] = read(args.run / "level_eda_reproducible.json")
    from evaluate_early_named_diarization import activity_metrics

    refined_dev = np.load(args.run / "diarization_dev/refined_frames.npz")
    neural_dev = np.load(args.run / "named_activity_extended/dev_predictions.npz")
    summary["rejected_neural_addition_dev"] = {
        str(t): activity_metrics(
            refined_dev["reference"], refined_dev["predicted"] | (neural_dev["probability"] >= t)
        )
        for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    }
    summary["session1"] = read(args.run / "session1_windowed/summary.json")
    summary["source_speech_minutes"] = {
        s: {n: round(t["speech_seconds"] / 60, 3) for n, t in m["tracks"].items()}
        for s, m in manifest["sessions"].items()
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))
    test_comparison = summary["diarization_test"]["comparisons"]
    baseline, refined = test_comparison[0]["metrics"], test_comparison[-1]["metrics"]
    with (args.output / "per_speaker_test.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "speaker",
                "reference_seconds",
                "baseline_f1",
                "refined_f1",
                "refined_precision",
                "refined_recall",
            ]
        )
        for name in NAMES:
            a, b = baseline["per_speaker"][name], refined["per_speaker"][name]
            writer.writerow(
                [name, b["reference_seconds"], a["f1"], b["f1"], b["precision"], b["recall"]]
            )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    x = np.arange(6)
    axes[0].bar(
        x - 0.18,
        [baseline["per_speaker"][n]["f1"] * 100 for n in NAMES],
        0.36,
        label="Global cluster name",
        color="#a4b5c6",
    )
    axes[0].bar(
        x + 0.18,
        [refined["per_speaker"][n]["f1"] * 100 for n in NAMES],
        0.36,
        label="Plus local turn check",
        color="#187d87",
    )
    axes[0].set(
        xticks=x,
        xticklabels=[n.rstrip("0123456789") for n in NAMES],
        ylim=(0, 100),
        ylabel="Speaker activity F1 (%)",
        title="Held-out Session 63: each speaker matters",
    )
    axes[0].tick_params(axis="x", rotation=32)
    axes[0].legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    axes[0].set_title("Held-out Session 63: each speaker matters", pad=34)
    labels = ["Centroid", "Clean classifier", "Augmented classifier"]
    x = np.arange(3)
    axes[1].bar(
        x - 0.18,
        [a["dev"]["macro_loss"] for a in ablation.values()],
        0.36,
        label="Development",
        color="#a4b5c6",
    )
    axes[1].bar(
        x + 0.18,
        [a["test"]["macro_loss"] for a in ablation.values()],
        0.36,
        label="Test",
        color="#187d87",
    )
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Speaker-balanced cross entropy (lower is better)",
        title="Noise augmentation did not generalize",
    )
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        "Automatic source-derived references; these are not human-gold accuracy scores", fontsize=11
    )
    fig.savefig(args.output / "evidence.png", dpi=180)
    plt.close(fig)
    paths = [
        Path(__file__),
        *Path("scripts").glob("*early*py"),
        Path("scripts/attribute_cached_diarization.py"),
        Path("scripts/refine_named_turns.py"),
        Path("scripts/diarize_named_recording.py"),
        Path("scripts/train_named_activity.py"),
        Path("scripts/evaluate_named_words.py"),
        Path("scripts/evaluate_named_long_context.py"),
        Path("src/transcriber/diarization.py"),
        Path("src/transcriber/transcript_pipeline.py"),
        Path("src/transcriber/parakeet_backend.py"),
    ]
    provenance = dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        packages={
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "torchaudio",
                "pyannote.audio",
                "parakeet-mlx",
                "scikit-learn",
                "onnxruntime",
            ]
        },
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths)},
        manifest_sha256=hashlib.sha256((args.corpus / "manifest.json").read_bytes()).hexdigest(),
        identity_model_sha256=hashlib.sha256(
            (args.run / "identity/speaker_model.npz").read_bytes()
        ).hexdigest(),
        vad_model_sha256=manifest["vad_model_sha256"],
        model="pyannote/speaker-diarization-community-1",
        training_sessions=[55, 61],
        development_session=62,
        test_session=63,
        early_target="Session 1: unlabelled noise only; no identity pseudo-labels",
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("REPORT_DATA_READY", args.output)


if __name__ == "__main__":
    main()
