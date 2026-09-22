#!/usr/bin/env python3
"""Plot the frozen held-out comparison without treating word error as speaker accuracy."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    keys = ["baseline", "public_moss", "candidate"]
    labels = ["Existing pipeline", "Public MOSS", "Fine-tuned MOSS"]
    colors = ["#6B7280", "#2864AE", "#007C70"]
    metrics = [data["metrics"][k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
    errors = [m["named_word_error_rate"] * 100 for m in metrics]
    overlaps = [m["categories"]["brief_overlap"]["matched_words"] for m in metrics]
    total = metrics[0]["categories"]["brief_overlap"]["reference_words"]
    for ax, values in zip(axes, [errors, overlaps], strict=True):
        ax.barh(range(3), values, color=colors, height=0.58)
    for i, value in enumerate(errors):
        axes[0].text(value + 0.6, i, f"{value:.1f}%", va="center", fontsize=10)
    for i, value in enumerate(overlaps):
        axes[1].text(value + total * 0.015, i, f"{value}/{total}", va="center", fontsize=10)
    axes[0].set(xlim=(0, max(errors) * 1.22), xlabel="Named word error (%) · lower is better")
    axes[1].set(xlim=(0, total), xlabel="Brief-overlap words recovered · higher is better")
    fig.suptitle("Held-out mono audio: transcription plus speaker attribution", x=0.02, ha="left")
    fig.text(
        0.02,
        0.015,
        "24 minutes across two sessions. Automatic source-word references; not human-verified speaker accuracy.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
