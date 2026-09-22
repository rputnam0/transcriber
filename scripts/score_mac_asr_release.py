"""Reproduce the end-to-end, speaker-attributed proxy check on held-out mixtures."""

import argparse
from collections import Counter
import json
from pathlib import Path
from score_moss_manifest import temporal_attributed_score
from summarize_mac_asr_benchmark import words, edit_distance
from run_mac_asr_quality import save


def score(root, baseline):
    records = json.loads((root / "benchmark.json").read_text())
    old = {r["cut_id"]: r for r in json.loads(baseline.read_text())["records"]}
    totals = {}
    rows = []
    for engine in ["baseline", "pilot", "qwen_fused"]:
        total = Counter()
        categories = {}
        for r in records:
            if engine == "baseline":
                segments = old[r["cut_id"]]["named_segments"]
            else:
                folder = "moss" if engine == "pilot" else "fused"
                segments = json.loads(
                    (
                        root
                        / folder
                        / f"session{r['session']}"
                        / "attribution"
                        / f"{r['cut_id']}.json"
                    ).read_text()
                )["turns"]
                segments = [
                    dict(t, start=t["start"] - r["start"], end=t["end"] - r["start"])
                    for t in segments
                ]
            names = {t["speaker"]: t["speaker"] for t in segments}
            result = temporal_attributed_score(
                r["reference"], segments, names, brief_turn_seconds=2, tolerance_seconds=1.0
            )
            for key in ["matched_words", "reference_words", "predicted_words"]:
                total[key] += result[key]
            for key, values in result["categories"].items():
                counter = categories.setdefault(key, Counter())
                for field in ["matched_words", "reference_words"]:
                    counter[field] += values[field]
            ref = words(
                " ".join(
                    t["text"] for t in sorted(r["reference"], key=lambda t: (t["start"], t["end"]))
                )
            )
            hyp = words(
                " ".join(t["text"] for t in sorted(segments, key=lambda t: (t["start"], t["end"])))
            )
            total["edits"] += edit_distance(ref, hyp)
            total["wer_words"] += len(ref)
            rows.append(dict(engine=engine, cut_id=r["cut_id"], scores=result))
        totals[engine] = dict(
            total,
            attributed_recall=total["matched_words"] / total["reference_words"],
            attributed_precision=total["matched_words"] / total["predicted_words"],
            proxy_wer=total["edits"] / total["wer_words"],
            categories={
                k: dict(v, recall=v["matched_words"] / max(1, v["reference_words"]))
                for k, v in categories.items()
            },
        )
    return dict(
        totals=totals,
        records=rows,
        caveat="Proxy stem transcripts, 16 clips from sessions held out of pilot training. "
        "Temporal speaker-attributed recall uses 1 second tolerance and coarse turn spans for all methods. "
        "Initial fusion failed overlap check; this validation influenced the short-overlap retention design. "
        "This is not an untouched final evaluation or human word-accuracy estimate.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = score(a.root, a.baseline)
    save(a.output, result)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
