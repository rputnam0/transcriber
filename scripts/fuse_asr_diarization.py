"""Attach new ASR words to fresh named MOSS turns using lexical and time evidence.

Never rewrite recognized words. Keep independently decoded secondary overlap as
explicitly flagged supplemental turns, rather than silently losing the second voice.
"""

from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import shutil
from run_mac_asr_quality import save


def tokens(text):
    return re.findall(r"[\w]+(?:['’][\w]+)*", text.lower())


def norm(text):
    return "".join(c.lower() for c in text if c.isalnum())


def outside(mid, turn):
    return max(turn["start"] - mid, mid - turn["end"], 0.0)


def word_owners(words, turns):
    """Local lexical matching disambiguates simultaneous turns; time covers ASR edits."""
    anchors = [[] for _ in words]
    mids = [(w["start"] + w["end"]) / 2 for w in words]
    normalized = [norm(w["text"]) for w in words]
    for ti, turn in enumerate(turns):
        selected = [i for i, mid in enumerate(mids) if outside(mid, turn) <= 1.25]
        if not selected:
            continue
        # Use the same whitespace tokenization as the forced aligner.
        reference = [norm(t) for t in turn["text"].split() if norm(t)]
        matcher = SequenceMatcher(
            None, reference, [normalized[i] for i in selected], autojunk=False
        )
        for block in matcher.get_matching_blocks():
            for j in range(block.size):
                wi = selected[block.b + j]
                score = 3.0 + min(block.size, 8) * 0.5 - outside(mids[wi], turn) * 2
                anchors[wi].append((score, ti, block.size))
    owners = [None] * len(words)
    methods = ["unresolved"] * len(words)
    for wi, candidates in enumerate(anchors):
        if candidates:
            ranked = sorted(candidates, reverse=True)
            best = ranked[0]
            if (
                len(ranked) == 1
                or best[0] - ranked[1][0] >= 0.4
                or turns[best[1]]["speaker_handle"] == turns[ranked[1][1]]["speaker_handle"]
            ):
                owners[wi] = best[1]
                methods[wi] = "lexical_and_time"
    fixed = list(owners)
    for wi, owner in enumerate(fixed):
        if owner is not None:
            continue
        left = next(
            (fixed[j] for j in range(wi - 1, max(-1, wi - 4), -1) if fixed[j] is not None), None
        )
        right = next(
            (fixed[j] for j in range(wi + 1, min(len(words), wi + 4)) if fixed[j] is not None), None
        )
        if left is not None and left == right and outside(mids[wi], turns[left]) <= 1.25:
            owners[wi], methods[wi] = left, "between_lexical_anchors"
            continue
        candidates = []
        for ti, turn in enumerate(turns):
            distance = outside(mids[wi], turn)
            if distance <= 0.35:
                overlap = max(
                    0.0, min(words[wi]["end"], turn["end"]) - max(words[wi]["start"], turn["start"])
                )
                candidates.append((overlap - distance, ti))
        if not candidates:
            # Forced word alignment can trail a coarse MOSS endpoint. Only
            # bridge a small gap if all nearby acoustic turns name the same voice.
            nearby = [
                (outside(mids[wi], turn), ti)
                for ti, turn in enumerate(turns)
                if outside(mids[wi], turn) <= 1.25
            ]
            if nearby and len({turns[ti]["speaker_handle"] for _, ti in nearby}) == 1:
                owners[wi] = min(nearby)[1]
                methods[wi] = "nearby_single_voice"
                continue
        if candidates:
            candidates.sort(reverse=True)
            winner = candidates[0][1]
            owners[wi] = winner
            distinct = {turns[ti]["speaker_handle"] for _, ti in candidates}
            methods[wi] = "time_only" if len(distinct) == 1 else "ambiguous_overlap"
    return owners, methods


def fuse_words(words, turns, core_cut_id=None):
    owners, methods = word_owners(words, turns)
    output = []
    counts = Counter()
    for w, owner, method in zip(words, owners, methods, strict=True):
        counts[owner] += 1
        source = turns[owner] if owner is not None else None
        reasons = []
        if method in {"unresolved", "ambiguous_overlap", "nearby_single_voice"}:
            reasons.append("speaker ambiguous in ASR/diarization alignment")
        if w["end"] == w["start"]:
            reasons.append("word timing uncertain")
        if source:
            reasons += source.get("review_reasons", [])
            handle = source.get("speaker_handle")
        else:
            handle = None
        item = dict(
            start=w["start"],
            end=w["end"],
            speaker=handle or "Unknown",
            text=w["text"],
            local_speaker=source["local_speaker"] if source else "UNALIGNED",
            cut_id=w["cut_id"],
            review_required=bool(reasons),
            review_reasons=sorted(set(reasons)),
            asr_source="qwen3_asr",
            attribution_method=method,
            source_moss_turn_id=source["turn_id"] if source else None,
            words=[dict(w, attribution_method=method)],
        )
        previous = output[-1] if output else None
        if (
            previous
            and previous["source_moss_turn_id"] == item["source_moss_turn_id"]
            and previous["speaker"] == item["speaker"]
            and previous["cut_id"] == item["cut_id"]
            and item["start"] - previous["end"] < 1.5
        ):
            previous["text"] += " " + item["text"]
            previous["end"] = max(previous["end"], item["end"])
            previous["words"].extend(item["words"])
            previous["review_reasons"] = sorted(set(previous["review_reasons"] + reasons))
            previous["review_required"] = bool(previous["review_reasons"])
        else:
            output.append(item)
    # Mono forced alignment cannot represent two independent simultaneous streams.
    # Route the complete overlapping MOSS turns together, not a mixture of short
    # MOSS interjections plus Qwen's flattened version of the same exchange.
    core_cut_id = core_cut_id or (words[0]["cut_id"] if words else None)
    routed = {
        ti
        for ti, turn in enumerate(turns)
        if turn["cut_id"] == core_cut_id
        and turn.get("speaker_handle")
        and any(
            other.get("speaker_handle")
            and other["speaker_handle"] != turn["speaker_handle"]
            and min(turn["end"], other["end"]) - max(turn["start"], other["start"]) >= 0.15
            for other in turns
        )
    }
    routed_ids = {turns[ti]["turn_id"] for ti in routed}
    alternative_words = {ti: [] for ti in routed}
    retained = []
    for item in output:
        current = None
        for word in item["words"]:
            mid = (word["start"] + word["end"]) / 2
            candidates = [
                ti
                for ti in routed
                if outside(mid, turns[ti]) == 0
                or item["source_moss_turn_id"] == turns[ti]["turn_id"]
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda ti: (
                        turns[ti]["turn_id"] != item["source_moss_turn_id"],
                        outside(mid, turns[ti]),
                        ti,
                    ),
                )
                alternative_words[chosen].append(word)
                current = None
                continue
            if current is None:
                current = dict(
                    item, start=word["start"], end=word["end"], text=word["text"], words=[word]
                )
                retained.append(current)
            else:
                current["text"] += " " + word["text"]
                current["end"] = max(current["end"], word["end"])
                current["words"].append(word)
    for ti in sorted(routed):
        turn = turns[ti]
        retained.append(
            dict(
                start=turn["start"],
                end=turn["end"],
                speaker=turn["speaker_handle"],
                text=turn["text"],
                local_speaker=turn["local_speaker"],
                cut_id=turn["cut_id"],
                review_required=True,
                review_reasons=["overlapping exchange retained from MOSS; verify wording"],
                asr_source="moss_overlap_priority",
                attribution_method="moss_overlap",
                source_moss_turn_id=turn["turn_id"],
                words=[],
                alternate_asr_words=alternative_words[ti],
            )
        )
    return sorted(retained, key=lambda t: (t["start"], t["end"])), dict(
        qwen_words=len(words),
        unresolved_words=methods.count("unresolved"),
        ambiguous_overlap_words=methods.count("ambiguous_overlap"),
        attribution_methods=dict(Counter(methods)),
        supplemental_overlap_turns=0,
        overlap_turns_retained=len(routed_ids),
        qwen_words_in_overlap_alternative=sum(len(ws) for ws in alternative_words.values()),
    )


def fuse_session(moss, asr, moss_export, output):
    records = json.loads((moss / "manifest.json").read_text())
    turns = json.loads(moss_export.read_text())["segments"]
    output.mkdir(parents=True, exist_ok=True)
    for filename in ["manifest.json", "source.json"]:
        shutil.copy2(moss / filename, output / filename)
    all_turns = []
    reports = []
    previous = None
    duplicate_words = []
    for r in records:
        asr_path = asr / "predictions" / f"{r['cut_id']}.json"
        raw = json.loads(asr_path.read_text())
        if raw["core_start"] != r["start"] or raw["core_end"] != r["start"] + r["duration"]:
            raise ValueError("ASR/diarization clock mismatch")
        words = []
        for word in raw["words"]:
            word = dict(word, cut_id=r["cut_id"])
            if (
                previous
                and previous["cut_id"] != word["cut_id"]
                and norm(previous["text"]) == norm(word["text"])
                and min(previous["end"], word["end"]) > max(previous["start"], word["start"])
            ):
                duplicate_words.append(word)
                continue
            words.append(word)
            previous = word
        nearby = [
            t
            for t in turns
            if t["end"] >= r["start"] - 2 and t["start"] <= r["start"] + r["duration"] + 2
        ]
        fused, report = fuse_words(words, nearby, r["cut_id"])
        # Supplements belong to their original core, and are emitted exactly once.
        fused = [
            t
            for t in fused
            if t["asr_source"] != "moss_overlap_supplement" or t["cut_id"] == r["cut_id"]
        ]
        for t in fused:
            t["cut_id"] = r["cut_id"]
            if raw["issues"] and any(w["end"] == w["start"] for w in t.get("words", [])):
                t["review_required"] = True
            t["end"] = max(t["start"] + 0.001, t["end"])
        report["supplemental_overlap_turns"] = sum(
            t["asr_source"] == "moss_overlap_supplement" for t in fused
        )
        report.update(cut_id=r["cut_id"], asr_issues=raw["issues"])
        provenance = dict(
            method="qwen_asr_moss_pilot_local_names",
            fusion_version=4,
            asr_sha256=hashlib.sha256(asr_path.read_bytes()).hexdigest(),
            moss_named_sha256=hashlib.sha256(moss_export.read_bytes()).hexdigest(),
            asr_provenance=raw["provenance"],
        )
        predpath = output / "predictions" / f"{r['cut_id']}.json"
        save(predpath, dict(**r, segments=fused, raw_text=raw["text"], provenance=provenance))
        old = json.loads((moss / "attribution" / f"{r['cut_id']}.json").read_text())
        save(
            output / "attribution" / f"{r['cut_id']}.json",
            dict(
                provenance=dict(
                    provenance, prediction_sha256=hashlib.sha256(predpath.read_bytes()).hexdigest()
                ),
                bindings=old["bindings"],
                turns=fused,
            ),
        )
        all_turns += fused
        reports.append(report)
    all_turns.sort(key=lambda t: (t["start"], t["end"]))
    save(
        output / "named.json",
        dict(
            source=json.loads((moss / "source.json").read_text()),
            provenance=provenance,
            segments=all_turns,
            reference_used=False,
            review_note="New dedicated ASR; trained MOSS identity model; uncertain overlap flagged.",
        ),
    )
    save(
        output / "fusion_audit.json",
        dict(
            records=reports,
            cross_boundary_duplicates=duplicate_words,
            qwen_words=sum(r["qwen_words"] for r in reports),
            supplemental_overlap_turns=sum(r["supplemental_overlap_turns"] for r in reports),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["moss", "asr", "moss-export", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    fuse_session(args.moss, args.asr, args.moss_export, args.output)


if __name__ == "__main__":
    main()
