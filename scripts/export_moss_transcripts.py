"""Apply a session roster to cached MOSS names and export standard readable transcripts.

Original speech, timing, and valid name assignments are preserved. Only out-of-roster
names are reconsidered using cached voice scores. No model is retrained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from transcriber.consolidate import _format_ts, consolidate, save_outputs


def resolve_name(turn, binding, allowed):
    """Restrict existing voice evidence; never infer a voice from adjacent dialogue."""
    if turn["speaker"] in allowed:
        return turn["speaker"], "original valid-roster assignment", {}
    scores = binding.get("probabilities", {})
    eligible = sorted(((float(scores.get(name, 0)), name) for name in allowed), reverse=True)
    total = sum(p for p, _ in eligible)
    if total <= 0:
        return None, "no uncontested voice evidence for an eligible speaker", {}
    probability, name = eligible[0]
    conditional = probability / total
    cosine = float(binding.get("cosines", {}).get(name, -1))
    evidence = {
        "candidate": name,
        "original_posterior": probability,
        "roster_conditional_posterior": conditional,
        "cosine": cosine,
    }
    # These are conservative export guardrails, not calibrated probabilities of correctness.
    if conditional >= 0.65 and cosine >= 0.35:
        return name, "tentative roster-constrained voice assignment", evidence
    return None, "eligible voice evidence is ambiguous", evidence


def paragraphs(turns):
    """Merge adjacent fragments only; never merge distinct unresolved clusters."""
    output = []
    for turn in turns:
        alternative = " ".join(w["text"] for w in turn.get("alternate_asr_words", []))
        alternatives = (
            [{"start": turn["start"], "text": alternative}]
            if alternative and alternative != turn["text"]
            else []
        )
        old = output[-1] if output else None
        key = turn["speaker_handle"] or (turn["cut_id"], turn["local_speaker"])
        if (
            old
            and old["merge_key"] == key
            and 0 <= turn["start"] - old["end"] <= 1.2
            and turn["end"] - old["start"] <= 45
        ):
            old["text"] += " " + turn["text"]
            old["end"] = turn["end"]
            old["turn_ids"].append(turn["turn_id"])
            old["alternatives"].extend(alternatives)
            old["roster_review_required"] |= turn["roster_review_required"]
            old["decode_review_required"] = old.get("decode_review_required", False) or turn.get(
                "decode_review_required", False
            )
            old["local_identity_review_required"] = bool(
                old.get("local_identity_review_required") or turn.get("local_identity_correction")
            )
        else:
            output.append(
                {
                    **turn,
                    "merge_key": key,
                    "turn_ids": [turn["turn_id"]],
                    "alternatives": alternatives,
                }
            )
    return output


def reader(title, rows, summary):
    body = []
    for row in rows:
        label = (
            " <small>check speaker</small>"
            if row["roster_review_required"]
            or row.get("local_identity_correction")
            or row.get("local_identity_review_required")
            else ""
        )
        if row.get("decode_review_required"):
            label += " <small>check passage</small>"
        alternative_html = "".join(
            "<details><summary>Alternate wording · " + _format_ts(a["start"]) + "</summary>"
            "<small>Another transcription of this passage; simultaneous voices may be combined.</small>"
            "<p>" + html.escape(a["text"]) + "</p></details>"
            for a in row.get("alternatives", [])
        )
        body.append(
            f'<article id="t{row["turn_id"]}" data-speaker="{html.escape(row["speaker"], quote=True)}">'
            f'<header><b>{html.escape(row["speaker"])}</b> '
            f'<a href="#t{row["turn_id"]}">{_format_ts(row["start"])}</a>{label}</header>'
            f'<p>{html.escape(row["text"])}</p>{alternative_html}</article>'
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>"""
        + html.escape(title)
        + """</title><style>
body{font:17px/1.7 -apple-system,BlinkMacSystemFont,sans-serif;color:#263f37;background:#f7f6f0;margin:0}
main{max-width:880px;margin:auto;padding:40px 24px}h1{font:44px Georgia,serif;margin-bottom:8px}
.subtitle{color:#63746a;font-size:14px}.toolbar{position:sticky;top:0;background:#f7f6f0f5;padding:15px 0;display:flex;gap:10px;border-bottom:1px solid #dcded4}
input,select{font:14px sans-serif;padding:10px;border:1px solid #cbd3c8;border-radius:6px;background:#fff;min-width:0}input{flex:1}
article{padding:21px 0;border-bottom:1px solid #e0e2d8;scroll-margin-top:80px}header{font-size:13px;display:flex;gap:14px;align-items:center}header a{color:#64756b;font-variant-numeric:tabular-nums}a{color:#30644d}p{margin:9px 0 0}small{font-size:11px;color:#8c6b33}.empty{display:none}
@media print{.toolbar,.nav{display:none}body{background:white}main{padding:0}article{break-inside:avoid}}</style>
<main><div class="nav"><a href="index.html">← Single-file transcripts</a></div><h1>"""
        + html.escape(title)
        + """</h1>
<p class="subtitle">"""
        + html.escape(summary)
        + """</p>
<div class="toolbar"><input id="search" type="search" aria-label="Search transcript" placeholder="Find a word or phrase…"><select id="speaker" aria-label="Filter speaker"><option value="">All voices</option></select></div>
"""
        + "\n".join(body)
        + """<p id="empty" hidden>No matching passages.</p></main>
<script>
const rows=[...document.querySelectorAll('article')], select=document.querySelector('#speaker'), search=document.querySelector('#search');
[...new Set(rows.map(r=>r.dataset.speaker))].sort().forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;select.append(option)});
function filter(){let count=0;for(const row of rows){row.hidden=!!((select.value&&row.dataset.speaker!==select.value)||!row.textContent.toLowerCase().includes(search.value.toLowerCase()));if(!row.hidden)count++}document.querySelector('#empty').hidden=count>0;}
select.onchange=filter;search.oninput=filter;
</script></html>"""
    )


def export(deployment, roster_path, session, output):
    config = json.loads(roster_path.read_text())
    allowed = config["session_rosters"].get(str(session))
    if not allowed:
        raise ValueError("No roster for this session; configure it explicitly")
    names = {
        **config["display_names"],
        **config.get("session_display_names", {}).get(str(session), {}),
    }
    source_path = deployment / "named.json"
    raw = source_path.read_bytes()
    source = json.loads(raw)
    records = json.loads((deployment / "manifest.json").read_text())
    if not records:
        raise ValueError("Empty deployment manifest")
    attribution = {}
    repaired_cuts = []
    for record in records:
        key = record["cut_id"]
        prediction_path = deployment / "predictions" / f"{key}.json"
        cached_path = deployment / "attribution" / f"{key}.json"
        prediction = json.loads(prediction_path.read_text())
        if prediction.get("decode_repair") or prediction.get("parser_repair"):
            repaired_cuts.append(key)
        cached = json.loads(cached_path.read_text())
        if prediction["sha256"] != record["sha256"]:
            raise ValueError("Prediction/manifest audio mismatch")
        if (
            cached["provenance"]["prediction_sha256"]
            != hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Attribution/prediction mismatch")
        attribution[key] = cached
    cached_turns = sorted(
        [t for a in attribution.values() for t in a["turns"]], key=lambda t: (t["start"], t["end"])
    )
    if cached_turns != source["segments"]:
        raise ValueError("Named transcript differs from the verified chunk attributions")
    turns, changes = [], []
    for i, original in enumerate(source["segments"]):
        turn = copy.deepcopy(original)
        binding = attribution[turn["cut_id"]]["bindings"].get(turn["local_speaker"], {})
        handle, reason, evidence = resolve_name(turn, binding, allowed)
        changed = handle != original["speaker"]
        turn.update(
            turn_id=i,
            original_speaker=original["speaker"],
            speaker_handle=handle,
            speaker=names[handle] if handle else "Unconfirmed speaker",
            roster_review_required=changed,
            decode_review_required=(
                turn["cut_id"] in repaired_cuts
                or turn.get("asr_source", "").startswith("moss_overlap")
                or any(
                    "word timing uncertain" in reason or "ASR/diarization" in reason
                    for reason in turn.get("review_reasons", [])
                )
            ),
            roster_reason=reason,
        )
        if changed:
            turn["review_required"] = True
            turn["review_reasons"].append(reason)
            changes.append(
                {
                    "turn_id": i,
                    "start": turn["start"],
                    "end": turn["end"],
                    "before": original["speaker"],
                    "after": handle,
                    "reason": reason,
                    **evidence,
                }
            )
        turns.append(turn)
    if [(t["start"], t["end"], t["text"]) for t in turns] != [
        (t["start"], t["end"], t["text"]) for t in source["segments"]
    ]:
        raise AssertionError("Export must preserve speech and timing")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"Session {session}"
    grouped = paragraphs(turns)
    per_file = [(stem, grouped)]
    save_outputs(stem, str(output), per_file, consolidate(per_file), None)
    report = {
        "session": session,
        "method": source["provenance"]["method"],
        "pipeline_provenance": source["provenance"],
        "source_transcript_sha256": hashlib.sha256(raw).hexdigest(),
        "roster_sha256": hashlib.sha256(roster_path.read_bytes()).hexdigest(),
        "allowed_speakers": allowed,
        "display_names": {k: names[k] for k in allowed},
        "raw_turns": len(turns),
        "paragraphs": len(grouped),
        "duration_seconds": max(r["start"] + r["duration"] for r in records),
        "source_note": config.get("session_source_notes", {}).get(str(session), ""),
        "roster_changes": changes,
        "repaired_decode_chunks": repaired_cuts,
        "unresolved_turns": sum(t["speaker_handle"] is None for t in turns),
        "named_turns": dict(Counter(t["speaker"] for t in turns)),
        "human_grades_used_for_training": False,
        "note": "Original words/timing unchanged. Roster corrections are tentative, not human verified. Unresolved names are explicit.",
    }
    (output / f"{stem}.audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / f"{stem}.turns.json").write_text(
        json.dumps({"provenance": report, "segments": turns}, indent=2) + "\n"
    )
    subtitle = (
        f'{round(report["duration_seconds"] / 60)} minutes · roster of {len(allowed)} people · '
        "Dungeon Master and character names · Automatic transcript"
    )
    if report["unresolved_turns"]:
        subtitle += f' · {report["unresolved_turns"]} turns have unconfirmed speakers'
    if report["source_note"]:
        subtitle += f' · {report["source_note"]}'
    (output / f"{stem}.html").write_text(reader(stem, grouped, subtitle))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["session", "raw_turns", "paragraphs", "unresolved_turns", "named_turns"]
            }
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--rosters", type=Path, default=Path("config/early_session_rosters.json"))
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.deployment, args.rosters, args.session, args.output)


if __name__ == "__main__":
    main()
