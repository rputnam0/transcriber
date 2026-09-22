#!/usr/bin/env python3
"""Summarize reference omissions without using any model-under-test predictions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.references.glob("words_*/*/*.audit.json")):
        data = json.loads(path.read_text())
        session, window = path.parent.name.split("_")
        rows.append(
            dict(
                split=path.parent.parent.name.removeprefix("words_"),
                session=session,
                window=int(window),
                speaker=path.name.removesuffix(".audit.json"),
                words_before=data["words_before"],
                words_after=data["words_after"],
                added_words=data["words_after"] - data["words_before"],
                missing_speech_regions=len(data["crops"]),
                empty_source_recovered=int(data["words_before"] == 0 and data["words_after"] > 0),
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "source_reference_omissions.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for field in [
            "words_before",
            "words_after",
            "added_words",
            "missing_speech_regions",
            "empty_source_recovered",
        ]:
            totals[row["split"]][field] += row[field]
    (args.output / "source_reference_omissions.json").write_text(
        json.dumps(dict(totals), indent=2) + "\n"
    )
    print(json.dumps(dict(totals), indent=2))
    by_speaker = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        if row["split"] == "train":
            values = by_speaker[row["speaker"]]
            values[0] += row["words_before"]
            values[1] += row["added_words"]
            values[2] += row["empty_source_recovered"]
    with (args.output / "training_reference_by_speaker.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["speaker", "original_words", "added_words", "empty_source_windows_recovered"]
        )
        writer.writerows([name, *values] for name, values in sorted(by_speaker.items()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(by_speaker, key=lambda n: by_speaker[n][0])
    before = [by_speaker[n][0] for n in names]
    added = [by_speaker[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.barh(names, before, color="#35658a", label="Original automatic words")
    ax.barh(names, added, left=before, color="#df9b39", label="Added from uncovered speech")
    for i, (old, new) in enumerate(zip(before, added, strict=True)):
        ax.text(old + new + 150, i, f"+{new:,} ({new / old:.1%})", va="center", fontsize=9)
    ax.set_xlim(0, max(old + new for old, new in zip(before, added)) * 1.25)
    ax.set_xlabel("Training-source words across five sessions")
    ax.set_title("Automatic source transcripts omitted short utterances", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(args.output / "source_reference_omissions.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
