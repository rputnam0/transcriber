"""Reparse saved MOSS output to detect silently omitted timestamped dialogue.

Run with the MOSS environment. This checks parser integrity, not ASR accuracy.
It reads caches without changing predictions or speaker assignments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moss_batch_inference import (
    decode_issues,
    recover_compact_segments,
    repair_speaker_brackets,
)
from run_moss_transcribe_diarize import normalized_segments


def audit(deployment, parse_transcript):
    records = json.loads((deployment / "manifest.json").read_text())
    errors = []
    audited = 0
    missing = []
    for record in records:
        path = deployment / "predictions" / f"{record['cut_id']}.json"
        if not path.exists():
            missing.append(record["cut_id"])
            continue
        prediction = json.loads(path.read_text())
        raw = prediction["raw_text"]
        repaired = repair_speaker_brackets(raw)
        parsed, recovered = recover_compact_segments(repaired, parse_transcript(repaired))
        expected = normalized_segments(parsed, duration=record["duration"])
        issues = decode_issues(raw)
        if expected != prediction["segments"]:
            issues.append("cached_segments_differ_from_reparsed_output")
        if (repaired != raw or recovered) and not prediction.get("parser_repair"):
            issues.append("missing_parser_repair_provenance")
        if prediction["sha256"] != record["sha256"]:
            issues.append("audio_hash_mismatch")
        if issues:
            errors.append(
                {
                    "cut_id": record["cut_id"],
                    "issues": issues,
                    "cached_turns": len(prediction["segments"]),
                    "expected_turns": len(expected),
                }
            )
        audited += 1
    return {
        "deployment": str(deployment),
        "expected_chunks": len(records),
        "audited_chunks": audited,
        "missing_chunks": missing,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from moss_transcribe_diarize import parse_transcript

    reports = [audit(path, parse_transcript) for path in args.deployment]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(
            f"{Path(report['deployment']).name}: "
            f"{report['audited_chunks']}/{report['expected_chunks']} chunks; "
            f"{len(report['errors'])} errors"
        )
    if any(report["errors"] or report["missing_chunks"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
