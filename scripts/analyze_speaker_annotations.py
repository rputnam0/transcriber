"""Snapshot human grades, audit joins, and diagnose named-diarization failures.

Never treats unreviewed turns as correct or missing corrections as positive labels.
Predicted-interval overlap is a proxy, not independent human overlap ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def union_seconds(intervals):
    cursor = None
    total = 0.0
    for start, end in sorted(intervals):
        if cursor is None or start > cursor:
            total += end - start
        elif end > cursor:
            total += end - cursor
        cursor = max(cursor or end, end)
    return total


def rate(rows):
    counts = Counter(r["verdict"] for r in rows)
    decisive = counts["correct"] + counts["wrong"]
    return dict(
        turns=len(rows),
        grades=dict(counts),
        decisive=decisive,
        correct_fraction=counts["correct"] / decisive if decisive else None,
    )


def load_review(review_path, transcript_path, session, names, output, legacy=False):
    review_raw, transcript_raw = review_path.read_bytes(), transcript_path.read_bytes()
    review = json.loads(review_raw)
    if digest(transcript_raw) != review["transcript_sha256"]:
        raise ValueError(f"Session {session}: transcript hash mismatch")
    snapshot = output / "snapshots" / f"session{session}"
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename, raw in [("review.json", review_raw), ("transcript.json", transcript_raw)]:
        target = snapshot / filename
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Frozen snapshot differs; use a new output directory")
        target.write_bytes(raw)
    turns = json.loads(transcript_raw)["segments"]
    rows = []
    for key, grade in review["reviews"].items():
        if str(int(key)) != key or not 0 <= int(key) < len(turns):
            raise ValueError(f"Invalid turn key {key}")
        verdict = grade["verdict"]
        if verdict not in {"correct", "wrong", "unsure", "ungraded"}:
            raise ValueError("Invalid verdict")
        t = turns[int(key)]
        predicted = names.get(t["speaker"], t["speaker"])
        slot = grade.get("speaker_slot")
        corrected = None
        if slot is not None:
            if type(slot) is not int or not 0 <= slot < len(review["roster"]):
                raise ValueError("Invalid speaker slot")
            corrected = review["roster"][slot]
            if corrected not in names.values():
                raise ValueError("Unmapped review roster label")
        truth = predicted if verdict == "correct" else corrected if verdict == "wrong" else None
        if verdict == "correct" and corrected and corrected != predicted:
            raise ValueError("Correct grade conflicts with correction")
        if verdict == "wrong" and corrected == predicted:
            raise ValueError("Wrong grade agrees with original")
        overlap = []
        for j, other in enumerate(turns):
            if j == int(key):
                continue
            a, b = max(t["start"], other["start"]), min(t["end"], other["end"])
            if b > a:
                overlap.append((a, b))
        duration = t["end"] - t["start"]
        row = dict(
            session=session,
            turn_id=int(key),
            legacy=legacy,
            verdict=verdict,
            predicted=predicted,
            corrected=corrected,
            truth=truth,
            start=t["start"],
            end=t["end"],
            duration=duration,
            text=t["text"],
            text_error=bool(grade.get("text_error")),
            note=grade.get("note", ""),
            cut_id=t["cut_id"],
            local_speaker=t["local_speaker"],
            overlap_fraction=union_seconds(overlap) / duration if duration else 0,
            review_required=bool(
                t.get("review_required")
                or t.get("roster_review_required")
                or t.get("decode_review_required")
            ),
            review_reasons=t.get("review_reasons", []),
            transcript_sha256=digest(transcript_raw),
            review_sha256=digest(review_raw),
            positive_label_eligible=bool(truth and not grade.get("text_error")),
        )
        rows.append(row)
    return rows, dict(
        session=session,
        legacy=legacy,
        review_source=str(review_path),
        transcript_source=str(transcript_path),
        transcript_sha256=digest(transcript_raw),
        review_sha256=digest(review_raw),
        updated_at=review.get("updated_at"),
        total_model_turns=len(turns),
        **rate([r for r in rows if r["verdict"] != "ungraded"]),
    )


def summarize(rows, sources):
    new = [r for r in rows if not r["legacy"] and r["verdict"] != "ungraded"]
    clusters = defaultdict(list)
    for r in new:
        if r["positive_label_eligible"]:
            clusters[(r["session"], r["cut_id"], r["local_speaker"])].append(r)
    collisions = []
    ceiling_errors = 0
    for (session, cut_id, local_speaker), group in clusters.items():
        truths = Counter(r["truth"] for r in group)
        if len(truths) > 1:
            unavoidable = len(group) - max(truths.values())
            ceiling_errors += unavoidable
            collisions.append(
                dict(
                    session=session,
                    cut_id=cut_id,
                    local_speaker=local_speaker,
                    truth_counts=dict(truths),
                    minimum_errors_with_one_name=unavoidable,
                    turn_ids=[r["turn_id"] for r in group],
                )
            )
    wrong = [r for r in new if r["verdict"] == "wrong"]
    confusions = Counter((r["predicted"], r["truth"]) for r in wrong if r["truth"])
    by_truth = {
        name: dict(
            labelled=sum(r["truth"] == name for r in new),
            missed=sum(r["truth"] == name for r in wrong),
        )
        for name in sorted({r["truth"] for r in new if r["truth"]})
    }
    return dict(
        sources=sources,
        new_grades=rate(new),
        by_session={
            str(s): rate([r for r in new if r["session"] == s])
            for s in sorted({r["session"] for r in new})
        },
        by_duration={
            "at_most_1_second": rate([r for r in new if r["duration"] <= 1]),
            "over_1_second": rate([r for r in new if r["duration"] > 1]),
        },
        by_overlap={
            "any_predicted_overlap": rate([r for r in new if r["overlap_fraction"] > 0]),
            "no_predicted_overlap": rate([r for r in new if r["overlap_fraction"] == 0]),
        },
        by_review_flag={
            str(flag): rate([r for r in new if r["review_required"] == flag])
            for flag in [False, True]
        },
        explicit_corrections=sum(bool(r["corrected"]) and r["verdict"] == "wrong" for r in new),
        positive_label_turns=sum(r["positive_label_eligible"] for r in new),
        by_true_speaker=by_truth,
        confusions=[dict(predicted=a, truth=b, count=n) for (a, b), n in confusions.most_common()],
        mixed_identity_clusters=collisions,
        minimum_cluster_constant_errors=ceiling_errors,
        graded_span={
            str(s): dict(
                first=min(r["start"] for r in new if r["session"] == s),
                last=max(r["end"] for r in new if r["session"] == s),
                distinct_chunks=len({r["cut_id"] for r in new if r["session"] == s}),
            )
            for s in {r["session"] for r in new}
        },
        errors=wrong,
        limitations=[
            "User-selected turns; these rates are not whole-recording accuracy or DER.",
            "Overlap is inferred from predicted intervals and may miss actual overlapping speakers.",
            "Legacy Session 1 was reviewed before roster corrections and is reported separately.",
            "Wrong without a corrected identity is a negative constraint, not a target-speaker label.",
            "Train/test splits must separate recordings or at least contiguous time blocks, not neighboring turns.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reviews", type=Path, default=Path.home() / ".cache/transcriber/speaker-review"
    )
    p.add_argument(
        "--deployments",
        type=Path,
        default=Path.home() / ".cache/transcriber/moss-training-20260919/deployment",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = json.loads(Path("config/early_session_rosters.json").read_text())["display_names"]
    rows = []
    sources = []
    for review in sorted((args.reviews / "library").rglob("review.json")):
        session = int(json.loads(review.read_text())["session"])
        r, s = load_review(review, review.with_name("transcript.json"), session, names, args.output)
        rows += r
        sources.append(s)
    legacy = args.reviews / "session1/review.json"
    if legacy.exists():
        r, s = load_review(
            legacy,
            args.deployments / "session1_selected_20260919/named.json",
            1,
            names,
            args.output,
            True,
        )
        rows += r
        sources.append(s)
    keys = [(r["session"], r["turn_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Multiple reviewed versions; resolve explicitly")
    report = summarize(rows, sources)
    (args.output / "turns.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    candidates = [r for r in rows if not r["legacy"] and r["positive_label_eligible"]]
    for row in candidates:
        prediction = json.loads(
            (
                args.deployments
                / f"session{row['session']}_selected_20260919"
                / "predictions"
                / f"{row['cut_id']}.json"
            ).read_text()
        )
        row["clip_start"] = row["start"] - prediction["start"]
        row["clip_end"] = row["end"] - prediction["start"]
        if not 0 <= row["clip_start"] < row["clip_end"] <= prediction["duration"] + 1e-6:
            raise ValueError("Supervision interval is outside the audio clip")
        row["audio_sha256"] = prediction["sha256"]
    (args.output / "supervised_turns.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    **r,
                    "split_group": f"session{r['session']}",
                    "audio": str(
                        args.deployments
                        / f"session{r['session']}_selected_20260919"
                        / "audio"
                        / f"{r['cut_id']}.wav"
                    ),
                    "usage": "speaker-identity supervision only; words and timing are not human-verified",
                }
            )
            + "\n"
            for r in candidates
        )
    )
    print(json.dumps({k: v for k, v in report.items() if k not in {"sources", "errors"}}, indent=2))


if __name__ == "__main__":
    main()
