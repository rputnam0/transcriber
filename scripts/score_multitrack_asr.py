"""Score timed isolated-track candidates against a machine reference on paired windows.

Run in the application or Mac ASR environment. References are automatic, not human
word-accuracy ground truth. Digital-zero windows override hallucinated reference
text. Coarse legacy text must first be independently aligned; it must not be
selected by its line start alone. Private inputs and detailed outputs stay out of Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


FIELDS = [
    "reference_words",
    "hypothesis_words",
    "deletions",
    "insertions",
    "substitutions",
    "silent_words",
    "repeated_phrase_windows",
]


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower())


def edit_counts(reference, hypothesis):
    """Minimum edit path, preferring substitution over insertion/deletion on ties."""
    previous = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, left in enumerate(reference, 1):
        current = [(i, i, 0, 0)]
        for j, right in enumerate(hypothesis, 1):
            if left == right:
                current.append(previous[j - 1])
                continue
            cost, deletions, insertions, substitutions = previous[j - 1]
            replace = (cost + 1, deletions, insertions, substitutions + 1)
            cost, deletions, insertions, substitutions = previous[j]
            delete = (cost + 1, deletions + 1, insertions, substitutions)
            cost, deletions, insertions, substitutions = current[-1]
            insert = (cost + 1, deletions, insertions + 1, substitutions)
            current.append(min([replace, delete, insert], key=lambda item: item[0]))
        previous = current
    _, deletions, insertions, substitutions = previous[-1]
    return dict(deletions=deletions, insertions=insertions, substitutions=substitutions)


def repeated_phrase(tokens):
    """Flag adjacent repeated phrases of at least three words; never delete them."""
    return any(
        tokens[i : i + n] == tokens[i + n : i + 2 * n]
        for n in range(3, min(13, len(tokens) // 2 + 1))
        for i in range(len(tokens) - 2 * n + 1)
    )


def timed_words(path):
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines()]
        result = []
        for record in records:
            if record.get("text", "").strip() and not record.get("words"):
                raise ValueError("Candidate has untimed text; align it before scoring")
            result.extend({**w, "speaker": record["speaker"]} for w in record.get("words", []))
        return result
    return json.loads(path.read_text())


def extract(records, window):
    return " ".join(
        str(w.get("word", w.get("text", "")))
        for w in records
        if w["speaker"] == window["speaker"]
        and window["start"] <= (w["start"] + w["end"]) / 2 < window["start"] + window["duration"]
    )


def score(manifest, reference, candidates):
    rows = []
    for window in manifest:
        reference_text = json.loads((reference / f"{window['cut_id']}.json").read_text())["text"]
        digital_silence = window["digital_silence"]
        ref = [] if digital_silence else words(reference_text)
        for candidate, records in candidates.items():
            text = extract(records, window)
            hyp = words(text)
            edits = edit_counts(ref, hyp)
            rows.append(
                dict(
                    cut_id=window["cut_id"],
                    speaker=window["speaker"],
                    stratum=window["stratum"],
                    candidate=candidate,
                    reference_words=len(ref),
                    hypothesis_words=len(hyp),
                    deletions=edits["deletions"],
                    insertions=edits["insertions"],
                    substitutions=edits["substitutions"],
                    silent_words=len(hyp) if digital_silence else 0,
                    repeated_phrase_windows=int(repeated_phrase(hyp)),
                    raw_reference_text=reference_text,
                    text=text,
                )
            )
    totals = {}
    for candidate in candidates:
        for speaker in sorted({r["speaker"] for r in rows}) + ["ALL"]:
            subset = [
                r
                for r in rows
                if r["candidate"] == candidate and (speaker == "ALL" or r["speaker"] == speaker)
            ]
            total = {k: sum(r[k] for r in subset) for k in FIELDS}
            total["windows"] = len(subset)
            total["proxy_wer"] = (
                sum(total[k] for k in ["deletions", "insertions", "substitutions"])
                / total["reference_words"]
                if total["reference_words"]
                else None
            )
            totals[f"{candidate} / {speaker}"] = total
    return dict(
        totals=totals,
        records=rows,
        caveat="Qwen automatic reference; silence corrected with source energy. "
        "Not human gold WER. Repeated phrases are flags, not proven hallucinations.",
        normalization="Lowercase ASCII alphanumerics and internal apostrophes; "
        "no number expansion; timed-word midpoint ownership.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=timed.json[l]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    for item in args.candidate:
        name, path = item.split("=", 1)
        candidates[name] = timed_words(Path(path))
    result = score(json.loads(args.manifest.read_text()), args.reference, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
