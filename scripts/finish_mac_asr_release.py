"""Validate and publish each complete local ASR + trained-diarization session rerun.

Reads fresh ASR, diarization and enrollment outputs; never modifies review snapshots.
Use --wait only while inference/attribution workers are running in this task.
"""

from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import time
from export_moss_transcripts import export
from fuse_asr_diarization import fuse_session
from run_mac_asr_quality import save
from repair_context_asr import text_issues


def validate(moss, asr, fused, allowed):
    records = json.loads((moss / "manifest.json").read_text())
    end = 0.0
    asr_words = 0
    issues = {}
    repaired = []
    diarizer = None
    for r in records:
        if abs(r["start"] - end) > 1e-6:
            raise ValueError("Gap or overlap in source coverage")
        end = r["start"] + r["duration"]
        m = json.loads((moss / "predictions" / f"{r['cut_id']}.json").read_text())
        a = json.loads((asr / "predictions" / f"{r['cut_id']}.json").read_text())
        if m["sha256"] != r["sha256"]:
            raise ValueError("Diarization audio mismatch")
        if diarizer is None:
            diarizer = m["provenance"]
            if len(diarizer.get("checkpoint_sha256", "")) != 64:
                raise ValueError("Diarizer checkpoint fingerprint is missing")
        elif m["provenance"] != diarizer:
            raise ValueError("Diarizer changed within the recording")
        if a["core_start"] != r["start"] or a["core_end"] != end:
            raise ValueError("ASR coverage mismatch")
        if "token_budget" in a["issues"] or text_issues(a["text"]):
            raise ValueError("Incomplete ASR decode")
        if a.get("decode_repair"):
            repaired.append(dict(cut_id=r["cut_id"], **a["decode_repair"]))
        if m.get("decode_repair", {}).get("remaining_issues"):
            raise ValueError("Unrepaired diarization decode")
        asr_words += len(a["words"])
        for issue in a["issues"]:
            issues[issue] = issues.get(issue, 0) + 1
    data = json.loads((fused / "named.json").read_text())
    for t in data["segments"]:
        if not 0 <= t["start"] < t["end"] <= end + 0.002:
            raise ValueError("Invalid transcript timestamp")
        if t["speaker"] not in allowed + ["Unknown"]:
            raise ValueError("Out-of-roster speaker")
    audit = json.loads((fused / "fusion_audit.json").read_text())
    represented = sum(
        len(t.get("words", [])) + len(t.get("alternate_asr_words", [])) for t in data["segments"]
    )
    if represented + len(audit["cross_boundary_duplicates"]) != asr_words:
        raise ValueError("ASR words were lost during fusion")
    return dict(
        duration_seconds=end,
        source_chunks=len(records),
        asr_words=asr_words,
        accounted_asr_words=represented,
        issues=issues,
        asr_repaired_clips=repaired,
        supplementary_overlap_turns=audit["supplemental_overlap_turns"],
        cross_boundary_duplicate_words=len(audit["cross_boundary_duplicates"]),
        diarizer=m["provenance"],
        asr=a["provenance"],
    )


def add_audio(output, session, source):
    folder = output / "audio"
    folder.mkdir(exist_ok=True)
    link = folder / f"session{session}{source.suffix}"
    if not link.exists():
        link.symlink_to(source.resolve())
    if link.resolve() != source.resolve():
        raise ValueError("Audio link changed")
    page = output / f"Session {session}.html"
    text = page.read_text()
    player = f"""<audio id="audio" preload="metadata" src="audio/{html.escape(link.name,quote=True)}"></audio>
<div style="display:flex;gap:12px;align-items:center;margin:16px 0">
<button id="play-pause" type="button">Play audio</button>
<input id="audio-position" aria-label="Audio position" type="range" min="0" max="1" step="1" value="0" style="flex:1">
<output id="audio-time">00:00:00</output></div>
<p class="subtitle" id="audio-status">Click any timestamp to play from that passage. “Check passage” includes uncertain overlap or timing.</p>"""
    text = text.replace('<div class="toolbar">', player + '<div class="toolbar">', 1)
    script = """<script>
const audio=document.querySelector('#audio');
const toggle=document.querySelector('#play-pause'),position=document.querySelector('#audio-position');
const showError=error=>{document.querySelector('#audio-status').textContent='Playback could not start: '+error.message};
toggle.addEventListener('click',()=>{if(audio.paused){audio.play().catch(showError)}else{audio.pause()}});
audio.addEventListener('play',()=>{toggle.textContent='Pause audio'});
audio.addEventListener('pause',()=>{toggle.textContent='Play audio'});
audio.addEventListener('loadedmetadata',()=>{position.max=audio.duration});
audio.addEventListener('timeupdate',()=>{position.value=audio.currentTime;document.querySelector('#audio-time').textContent=new Date(audio.currentTime*1000).toISOString().slice(11,19)});
position.addEventListener('input',()=>{audio.currentTime=Number(position.value)});
for(const row of document.querySelectorAll('article')){const anchor=row.querySelector('header a');anchor.addEventListener('click',()=>{const parts=anchor.textContent.split(':').map(Number);audio.currentTime=parts.reduce((a,b)=>a*60+b,0);audio.play().catch(showError)})}
</script>"""
    text = text.replace("</html>", script + "</html>")
    page.write_text(text)


def index(output, plan, reports):
    rows = []
    for entry in plan["recordings"]:
        session = entry["session"]
        report = next((r for r in reports if r["session"] == session), None)
        if report:
            duration = report["duration_seconds"]
            hours = int(duration // 3600)
            minutes = int(duration % 3600 // 60)
            label = f"{hours}h {minutes}m"
            stem = f"Session {session}"
            note = (
                " · excerpt only; full recording pending"
                if session == 1 and duration < 1000
                else ""
            )
            rows.append(
                f'<li><a href="{stem}.html">{stem}</a> · {label}{note} · <a href="{stem}.txt">Text</a> · <a href="{stem}.srt">SRT</a></li>'
            )
        else:
            rows.append(f"<li>Session {session} · rerun in progress</li>")
    content = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reprocessed session transcripts</title><style>body{font:18px/1.7 system-ui;background:#f7f6f0;color:#263f37;max-width:850px;margin:60px auto;padding:24px}h1{font:40px Georgia}a{color:#30644d}li{padding:10px}p{color:#566b60}</style><h1>Reprocessed session transcripts</h1><p>Dedicated Qwen3 transcription with the 20-step trained MOSS diarization model. Read by Dungeon Master and character name, or click a timestamp to listen. Difficult overlap and uncertain names are marked for review.</p><ul>"""
        + "".join(rows)
        + """</ul><p>Previous transcripts and annotations are preserved. These are automatic transcripts; proper names, simultaneous speech, and word timing can still contain errors.</p></html>"""
    )
    (output / "index.html").write_text(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rosters", type=Path, default=Path("config/early_session_rosters.json"))
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    rosters = json.loads(args.rosters.read_text())
    reports = []
    args.output.mkdir(parents=True, exist_ok=True)
    index(args.output, plan, reports)
    for e in plan["recordings"]:
        s = e["session"]
        moss = args.work / "moss" / f"session{s}"
        asr = args.work / "asr" / f"session{s}"
        exported = args.work / "moss_transcripts" / f"Session {s}.turns.json"
        while True:
            records = (
                json.loads((moss / "manifest.json").read_text())
                if (moss / "manifest.json").exists()
                else []
            )
            if (
                records
                and exported.exists()
                and (args.work / "listening" / f"session{s}.m4a").exists()
                and (args.work / "listening" / f"session{s}.source.json").exists()
                and all((asr / "predictions" / f"{r['cut_id']}.json").exists() for r in records)
            ):
                break
            if not args.wait:
                raise RuntimeError(f"Session {s} is not ready")
            time.sleep(10)
        fused = args.work / "fused" / f"session{s}"
        fuse_session(moss, asr, exported, fused)
        report = validate(moss, asr, fused, rosters["session_rosters"][str(s)])
        report.update(session=s, source_note=e.get("source_note", ""))
        export(fused, args.rosters, s, args.output)
        add_audio(args.output, s, args.work / "listening" / f"session{s}.m4a")
        reports.append(report)
        save(args.output / "release_audit.json", reports)
        index(args.output, plan, reports)
        print("PUBLISHED", s, "hours", report["duration_seconds"] / 3600, flush=True)


if __name__ == "__main__":
    main()
