from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def assigned_margin_rows(payloads: Iterable[Mapping[str, object]], *, mode: str) -> list[dict]:
    rows = []
    for payload in payloads:
        for record in list(payload.get("records") or []):
            expected = dict(record.get("oracle_mapping_diagnostic_only") or {})
            mapping = dict(record.get(f"{mode}_mapping") or {})
            evidence_by_slot = dict(record.get("binding_evidence") or {})
            for slot, truth in expected.items():
                assigned = mapping.get(slot)
                evidence = dict(evidence_by_slot.get(slot) or {})
                scores = {
                    str(speaker): float(score)
                    for speaker, score in dict(evidence.get("scores") or {}).items()
                }
                if assigned is None or assigned not in scores:
                    continue
                alternatives = [score for speaker, score in scores.items() if speaker != assigned]
                margin = scores[assigned] - max(alternatives) if alternatives else 0.0
                rows.append(
                    {
                        "cut_id": str(record.get("cut_id") or ""),
                        "slot": str(slot),
                        "truth": str(truth),
                        "assigned": str(assigned),
                        "assigned_margin": margin,
                        "correct": assigned == truth,
                    }
                )
    return rows


def calibrate_threshold(
    rows: Sequence[Mapping[str, object]],
    *,
    target_precision: float,
    threshold_step: float,
) -> dict:
    thresholds = []
    threshold = 0.0
    while threshold <= 1.0 + 1e-9:
        accepted = [row for row in rows if float(row["assigned_margin"]) >= threshold]
        correct = sum(bool(row["correct"]) for row in accepted)
        thresholds.append(
            {
                "threshold": round(threshold, 10),
                "accepted": len(accepted),
                "coverage": len(accepted) / max(1, len(rows)),
                "correct": correct,
                "precision": correct / max(1, len(accepted)),
            }
        )
        threshold += threshold_step
    eligible = [
        item
        for item in thresholds
        if item["accepted"] > 0 and item["precision"] >= target_precision
    ]
    selected = (
        max(eligible, key=lambda item: (item["coverage"], -item["threshold"])) if eligible else None
    )
    return {
        "target_precision": target_precision,
        "threshold_step": threshold_step,
        "reference_rows": len(rows),
        "selected": selected,
        "sweep": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a known-speaker binding review threshold on development records."
    )
    parser.add_argument("--binding-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("independent", "one_to_one"), default="one_to_one")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.binding_json]
    rows = assigned_margin_rows(payloads, mode=args.mode)
    result = {
        "binding_json": [str(path) for path in args.binding_json],
        "mode": args.mode,
        **calibrate_threshold(
            rows,
            target_precision=args.target_precision,
            threshold_step=args.threshold_step,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
