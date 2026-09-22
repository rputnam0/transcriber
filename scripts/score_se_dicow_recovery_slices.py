from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _cut_order(cut_id: object) -> tuple[int, int, int]:
    match = re.match(r"session_(\d+)_w(\d+)_c(\d+)", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot order cut id {cut_id!r}")
    return tuple(int(value) for value in match.groups())


def _reference_spans(cut: Mapping[str, object]) -> list[dict]:
    custom = dict(cut.get("custom") or {})
    raw_spans = list(custom.get("transcript_spans") or [])
    if raw_spans:
        return [dict(span) for span in raw_spans]
    return [
        {
            "speaker": supervision.get("speaker"),
            "start": supervision.get("start"),
            "end": float(supervision.get("start") or 0.0)
            + float(supervision.get("duration") or 0.0),
            "text": supervision.get("text"),
        }
        for supervision in list(cut.get("supervisions") or [])
    ]


def _activity_spans(cut: Mapping[str, object]) -> list[dict]:
    spans = []
    for supervision in list(cut.get("supervisions") or []):
        start = float(supervision.get("start") or 0.0)
        spans.append(
            {
                "speaker": str(supervision.get("speaker") or ""),
                "start": start,
                "end": start + float(supervision.get("duration") or 0.0),
            }
        )
    return spans or _reference_spans(cut)


def _lcs_reference_indices(reference: Sequence[str], prediction: Sequence[str]) -> set[int]:
    rows = len(reference)
    columns = len(prediction)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            if reference[row] == prediction[column]:
                lengths[row][column] = lengths[row + 1][column + 1] + 1
            else:
                lengths[row][column] = max(
                    lengths[row + 1][column],
                    lengths[row][column + 1],
                )
    matched = set()
    row = 0
    column = 0
    while row < rows and column < columns:
        if reference[row] == prediction[column]:
            matched.add(row)
            row += 1
            column += 1
        elif lengths[row + 1][column] >= lengths[row][column + 1]:
            row += 1
        else:
            column += 1
    return matched


def score_recovery_slices(
    cuts: Iterable[Mapping[str, object]],
    results: Iterable[Mapping[str, object]],
    *,
    brief_turn_seconds: float = 2.0,
) -> dict:
    predictions: dict[tuple[str, str], list[tuple[tuple[int, int, int], str]]] = defaultdict(list)
    for result in results:
        cut_id = str(result.get("cut_id") or "")
        if not cut_id or not result.get("records"):
            continue
        order = _cut_order(cut_id)
        session = f"Session {order[0]}"
        for record in list(result.get("records") or []):
            speaker = str(record.get("speaker") or "")
            predictions[(session, speaker)].append((order, str(record.get("prediction") or "")))

    references: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for cut in sorted(cuts, key=lambda item: _cut_order(item.get("id"))):
        order = _cut_order(cut.get("id"))
        session = f"Session {order[0]}"
        spans = _reference_spans(cut)
        activity_spans = _activity_spans(cut)
        for span in spans:
            speaker = str(span.get("speaker") or "")
            text_tokens = _tokens(span.get("text"))
            if not speaker or not text_tokens:
                continue
            start = float(span.get("start") or 0.0)
            end = float(span.get("end", start) or start)
            brief = end - start <= brief_turn_seconds
            duration = max(0.0, end - start)
            for index, token in enumerate(text_tokens):
                midpoint = start + duration * (index + 0.5) / len(text_tokens)
                overlap = any(
                    str(other.get("speaker") or "") != speaker
                    and float(other.get("start") or 0.0)
                    <= midpoint
                    < float(other.get("end", other.get("start") or 0.0) or 0.0)
                    for other in activity_spans
                )
                references[(session, speaker)].append(
                    {"token": token, "overlap": overlap, "brief": brief}
                )

    category_flags = {
        "all": lambda item: True,
        "ordinary_nonoverlap": lambda item: not item["overlap"],
        "overlap": lambda item: item["overlap"],
        "brief_turn": lambda item: item["brief"],
        "brief_overlap": lambda item: item["brief"] and item["overlap"],
    }
    totals = {category: {"reference_words": 0, "matched_words": 0} for category in category_flags}
    for key, reference_items in references.items():
        reference_tokens = [str(item["token"]) for item in reference_items]
        prediction_tokens = [
            token
            for _order, text in sorted(predictions.get(key, []), key=lambda item: item[0])
            for token in _tokens(text)
        ]
        matched = _lcs_reference_indices(reference_tokens, prediction_tokens)
        for category, selected in category_flags.items():
            indices = [index for index, item in enumerate(reference_items) if selected(item)]
            totals[category]["reference_words"] += len(indices)
            totals[category]["matched_words"] += sum(index in matched for index in indices)

    for values in totals.values():
        values["recall"] = values["matched_words"] / max(1, values["reference_words"])
    return {
        "brief_turn_seconds": brief_turn_seconds,
        "overlap_definition": (
            "uniform reference-word midpoint falls inside another speaker's privileged "
            "clean-track activity; scoring only"
        ),
        "categories": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score named target-ASR recovery on ordinary, brief-turn, and overlap slices."
    )
    parser.add_argument("--cutset", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brief-turn-seconds", type=float, default=2.0)
    args = parser.parse_args()

    results = []
    for path in sorted(args.results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("cut_id") and result.get("records"):
            results.append(result)
    scored = {
        "cutset": str(args.cutset),
        "results_dir": str(args.results_dir),
        **score_recovery_slices(
            _read_jsonl(args.cutset),
            results,
            brief_turn_seconds=args.brief_turn_seconds,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()
