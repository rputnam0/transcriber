"""Audit complete source coverage and build the single-file transcript library.

The plan records Drive byte sizes and historical source limitations separately.
A complete model run is never evidence that the source covers the whole session.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

from moss_batch_inference import decode_issues, repair_speaker_brackets


def check_coverage(records, duration):
    """Require ordered, contiguous coverage of the entire decoded recording."""
    if not records or duration <= 0:
        raise ValueError("Empty recording or manifest")
    cursor = 0.0
    for record in records:
        if abs(record["start"] - cursor) > 1 / 16000:
            raise ValueError(f"Gap or overlap at {cursor:.6f} seconds")
        if not 0 < record["duration"] <= 30.001:
            raise ValueError("Invalid chunk duration")
        cursor = record["start"] + record["duration"]
    if abs(cursor - duration) > 0.25:
        raise ValueError(f"Processed {cursor:.3f}s, source contains {duration:.3f}s")
    return cursor


def inspect(entry, export_dir):
    result = dict(entry)
    source = Path(entry["audio"])
    deployment = Path(entry["deployment"])
    result["status"] = "awaiting source file"
    if not source.is_file():
        return result
    result["local_bytes"] = source.stat().st_size
    if result["local_bytes"] != entry["drive_bytes"]:
        result["status"] = "incomplete download or wrong source"
        return result
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source)],
        capture_output=True,
        text=True,
        check=True,
    )
    result["source_duration_seconds"] = float(json.loads(probe.stdout)["format"]["duration"])
    result["status"] = "downloaded; awaiting preparation"
    manifest = deployment / "manifest.json"
    if not manifest.is_file():
        return result
    provenance = json.loads((deployment / "source.json").read_text())
    if Path(provenance["path"]).resolve() != source.resolve():
        raise ValueError("Deployment points to another source")
    if provenance["bytes"] != source.stat().st_size:
        raise ValueError("Source bytes changed since preparation")
    records = json.loads(manifest.read_text())
    result["prepared_seconds"] = check_coverage(records, result["source_duration_seconds"])
    result["expected_chunks"] = len(records)
    predictions = 0
    attributed = 0
    for record in records:
        prediction = deployment / "predictions" / f"{record['cut_id']}.json"
        attribution = deployment / "attribution" / f"{record['cut_id']}.json"
        if not prediction.exists():
            continue
        raw = prediction.read_bytes()
        decoded = json.loads(raw)
        if decoded["sha256"] != record["sha256"]:
            raise ValueError("Prediction audio does not match manifest")
        if decode_issues(decoded.get("raw_text", "")):
            raise ValueError(f"Unfinished or repetitive decode in chunk {record['cut_id']}")
        if repair_speaker_brackets(decoded.get("raw_text", "")) != decoded.get("raw_text", ""):
            if not decoded.get("parser_repair"):
                raise ValueError(f"Unrepaired speaker marker in chunk {record['cut_id']}")
        predictions += 1
        if attribution.exists():
            binding = json.loads(attribution.read_text())
            if binding["provenance"]["prediction_sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("Attribution does not match prediction")
            attributed += 1
    result.update(predicted_chunks=predictions, attributed_chunks=attributed)
    result["status"] = f"transcribing: {predictions}/{len(records)} chunks"
    if predictions == len(records):
        result["status"] = f"attributing speakers: {attributed}/{len(records)} chunks"
    export_audit = export_dir / f"Session {entry['session']}.audit.json"
    named = deployment / "named.json"
    if attributed == len(records) and named.exists():
        result["status"] = "processed; awaiting transcript export"
        if export_audit.exists():
            audit = json.loads(export_audit.read_text())
            if audit["source_transcript_sha256"] != hashlib.sha256(named.read_bytes()).hexdigest():
                raise ValueError("Export is stale")
            result["status"] = "complete source file processed"
            result["unresolved_turns"] = audit["unresolved_turns"]
            result["raw_turns"] = audit["raw_turns"]
            result["transcript"] = f"Session {entry['session']}.html"
    return result


def write_library(results, destination):
    rows = []
    for item in results:
        session = f"Session {item['session']}"
        if item.get("transcript"):
            session = f'<a href="{html.escape(item["transcript"], quote=True)}">{session}</a>'
        duration = item.get("source_duration_seconds")
        duration = (
            f"{int(duration // 3600)}:{int(duration % 3600 // 60):02}:{int(duration % 60):02}"
            if duration
            else "Pending download"
        )
        rows.append(
            f"<tr><td>{session}</td><td>{duration}</td><td>{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item.get('source_note', ''))}</td></tr>"
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Single-file transcripts</title>
<style>body{font:17px/1.6 -apple-system,sans-serif;background:#f7f6f0;color:#263f37;margin:0}main{max-width:1150px;margin:auto;padding:40px 24px}h1{font:44px Georgia,serif}a{color:#30644d}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:16px;border-bottom:1px solid #dcded4;vertical-align:top}th{font-size:13px}td:last-child{font-size:14px;max-width:390px}.note{color:#63746a} .scroll{overflow-x:auto}</style>
<main><h1>Single-file transcripts</h1><p>Timestamped dialogue with Dungeon Master and character names.</p>
<p class="note">Processing status below is checked against each recording's full duration. Source excerpts remain explicitly identified; completing an excerpt does not recover missing audio. These are automatic drafts, with uncertain speakers marked for review.</p>
<div class="scroll"><table><thead><tr><th>Transcript</th><th>Source duration</th><th>Processing</th><th>Source coverage</th></tr></thead><tbody>"""
    document += "\n".join(rows) + "</tbody></table></div></main></html>"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "index.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    results = []
    for entry in plan["recordings"]:
        try:
            results.append(inspect(entry, args.output))
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            results.append({**entry, "status": "audit failed", "error": str(error)})
    write_library(results, args.output)
    (args.output / "coverage.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{k: r.get(k) for k in ("session", "status", "error")} for r in results]))
    if any(r["status"] == "audit failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
