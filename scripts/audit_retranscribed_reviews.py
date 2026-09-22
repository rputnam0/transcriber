"""Compare retranscribed passages with eligible frozen speaker reviews.

Turn IDs change on retranscription. Match words inside the reviewed time interval;
report unmatched/ambiguous passages separately. This is not whole-session accuracy.
"""

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
from summarize_mac_asr_benchmark import words
from run_mac_asr_quality import save


def audit(annotations, transcripts):
    reviewed = json.loads(annotations.read_text())
    results = []
    sessions = {}
    missing = set()
    for r in reviewed:
        if r["legacy"] or not r["positive_label_eligible"] or r["verdict"] == "unsure":
            continue
        session = r["session"]
        if session not in sessions:
            path = transcripts / f"Session {session}.turns.json"
            if not path.exists():
                missing.add(session)
                continue
            sessions[session] = json.loads(path.read_text())["segments"]
        candidate = []
        for t in sessions[session]:
            if t["end"] < r["start"] - 0.5 or t["start"] > r["end"] + 0.5:
                continue
            spans = t.get("words") or [t]
            for span in spans:
                if span["end"] < r["start"] - 0.5 or span["start"] > r["end"] + 0.5:
                    continue
                for word in words(span["text"]):
                    candidate.append((word, t["speaker"], span["start"], span["end"]))
        reference = words(r["text"])
        candidate.sort(key=lambda c: (c[2], c[3]))
        matches = SequenceMatcher(
            None, reference, [w[0] for w in candidate], autojunk=False
        ).get_matching_blocks()
        votes = Counter(candidate[b.b + i][1] for b in matches for i in range(b.size))
        count = sum(votes.values())
        coverage = count / max(1, len(reference))
        winner, win_count = votes.most_common(1)[0] if votes else (None, 0)
        # The same phrase may occur on two simultaneous voices. A single
        # SequenceMatcher path arbitrarily picks one; that is not evidence
        # against the other human-confirmed turn.
        speaker_coverage = {}
        for name in {c[1] for c in candidate}:
            blocks = SequenceMatcher(
                None, reference, [c[0] for c in candidate if c[1] == name], autojunk=False
            ).get_matching_blocks()
            speaker_coverage[name] = sum(b.size for b in blocks) / max(1, len(reference))
        multiple_matches = sum(v >= 0.6 for v in speaker_coverage.values()) > 1
        scoreable = coverage >= 0.6 and win_count / max(1, count) >= 0.75 and not multiple_matches
        results.append(
            dict(
                session=session,
                old_turn_id=r["turn_id"],
                start=r["start"],
                text=r["text"],
                truth=r["truth"],
                before=r["predicted"],
                after=winner,
                matched_reference_fraction=coverage,
                matched_word_votes=dict(votes),
                per_speaker_word_coverage=speaker_coverage,
                competing_speaker_matches=multiple_matches,
                comparable=scoreable,
                correct=(winner == r["truth"]) if scoreable else None,
            )
        )
    compared = [r for r in results if r["comparable"]]
    return dict(
        caveat="User-selected passages. Exact normalized words matched within old time span ±0.5s; "
        "requires 60% word coverage and 75% name agreement, without another voice matching 60% "
        "of the same phrase. Unmatched/ambiguous passages excluded and reported. "
        "Unsure, text-error and legacy grades excluded. No labels used as new training targets.",
        eligible_available=len(results),
        comparable=len(compared),
        unmatched_or_ambiguous=len(results) - len(compared),
        before_correct=sum(r["before"] == r["truth"] for r in compared),
        after_correct=sum(r["correct"] for r in compared),
        fixed=sum(r["before"] != r["truth"] and r["correct"] for r in compared),
        broken=sum(r["before"] == r["truth"] and not r["correct"] for r in compared),
        missing_sessions=sorted(missing),
        records=results,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.annotations, a.transcripts)
    save(a.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
