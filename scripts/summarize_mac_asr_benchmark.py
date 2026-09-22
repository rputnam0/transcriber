"""Paired ASR comparison against time-sorted, machine-derived stem references.

These are proxy error rates, not human word-accuracy measurements. Overlapping
voices admit multiple word orders; order-insensitive content recall is also saved.
"""

from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from run_mac_asr_quality import save


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def summarize(manifest, predictions, normalize=words):
    records = json.loads(manifest.read_text())
    rows = []
    totals = {}
    for engine in ["baseline", "qwen", "whisper", "granite", "mega", "granite5"]:
        total = Counter()
        for record in records:
            path = predictions / engine / f"{record['cut_id']}.json"
            if engine == "baseline":
                text = record["baseline_text"]
            elif path.exists():
                text = json.loads(path.read_text())["text"]
            else:
                continue
            hyp = normalize(text)
            row = dict(
                engine=engine,
                cut_id=record["cut_id"],
                category=record["category"],
                hypothesis_words=len(hyp),
            )
            if "reference" in record:
                ref = normalize(
                    " ".join(
                        t["text"]
                        for t in sorted(record["reference"], key=lambda t: (t["start"], t["end"]))
                    )
                )
                row.update(
                    reference_words=len(ref),
                    edits=edit_distance(ref, hyp),
                    matched_bag_words=sum((Counter(ref) & Counter(hyp)).values()),
                )
                total.update(
                    {
                        k: row[k]
                        for k in [
                            "reference_words",
                            "edits",
                            "matched_bag_words",
                            "hypothesis_words",
                        ]
                    }
                )
                total["clips"] += 1
            row["excessive_word_count"] = len(hyp) > 200
            rows.append(row)
        if total:
            totals[engine] = dict(
                total,
                proxy_wer=total["edits"] / total["reference_words"],
                bag_recall=total["matched_bag_words"] / total["reference_words"],
                bag_precision=total["matched_bag_words"] / total["hypothesis_words"],
            )
    return dict(
        caveat="Automatic isolated-stem references; overlap ordering ambiguous. Not human gold WER. "
        "Development only. Early recordings have no word-level reference.",
        normalization="lowercase ASCII words/apostrophes; reference sorted by start/end; no number expansion",
        totals=totals,
        records=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--whisper-normalization", action="store_true")
    args = p.parse_args()
    normalize = words
    if args.whisper_normalization:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        english = EnglishTextNormalizer({})

        def normalize(text):
            return english(text).split()

    result = summarize(args.manifest, args.predictions, normalize)
    if args.whisper_normalization:
        result["normalization"] = (
            "Whisper EnglishTextNormalizer with empty spelling map; numbers and contractions normalized"
        )
    save(args.output, result)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
