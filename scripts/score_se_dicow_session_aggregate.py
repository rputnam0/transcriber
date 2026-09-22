from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from evaluate_se_dicow_oracle_cut import attributed_bag_metrics, attributed_sequence_metrics


def _cut_order(cut_id: object) -> tuple[int, int, int]:
    match = re.match(r"session_(\d+)_w(\d+)_c(\d+)", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot order cut id {cut_id!r}")
    return tuple(int(value) for value in match.groups())


def aggregate_records(results: Iterable[Mapping[str, object]]) -> dict:
    grouped: dict[tuple[str, str], list[tuple[tuple[int, int, int], dict]]] = defaultdict(list)
    cut_ids = set()
    for raw_result in results:
        result = dict(raw_result)
        cut_id = str(result.get("cut_id") or "")
        cut_ids.add(cut_id)
        order = _cut_order(cut_id)
        session = str(result.get("session") or "")
        for raw_record in list(result.get("records") or []):
            record = dict(raw_record)
            speaker = str(record.get("speaker") or "")
            grouped[(session, speaker)].append((order, record))

    records = []
    for (session, speaker), items in sorted(grouped.items()):
        ordered = [record for _order, record in sorted(items, key=lambda item: item[0])]
        reference = " ".join(str(record.get("reference") or "") for record in ordered).strip()
        prediction = " ".join(str(record.get("prediction") or "") for record in ordered).strip()
        records.append(
            {
                "session": session,
                "speaker": speaker,
                "reference": reference,
                "prediction": prediction,
                **attributed_bag_metrics(reference, prediction),
                **attributed_sequence_metrics(reference, prediction),
            }
        )
    totals = {
        key: sum(int(record[key]) for record in records)
        for key in (
            "reference_words",
            "predicted_words",
            "bag_matches",
            "sequence_matches",
            "sequence_prediction_matches",
        )
    }
    return {
        "cut_count": len(cut_ids),
        "session_speaker_records": records,
        **totals,
        "attributed_bag_recall": totals["bag_matches"] / max(1, totals["reference_words"]),
        "attributed_bag_precision": totals["bag_matches"] / max(1, totals["predicted_words"]),
        "attributed_sequence_recall": totals["sequence_matches"]
        / max(1, totals["reference_words"]),
        "attributed_sequence_precision": totals["sequence_prediction_matches"]
        / max(1, totals["predicted_words"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score contiguous SE-DiCoW cuts after aggregating each session and speaker."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No JSON results found in {args.input_dir}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    scored = {
        "input_dir": str(args.input_dir),
        "scoring_scope": "contiguous-session-speaker-aggregate",
        **aggregate_records(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in scored.items() if key != "session_speaker_records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
