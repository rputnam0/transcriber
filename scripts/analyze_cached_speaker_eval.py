from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_words(rows: List[dict]) -> List[dict]:
    words: List[dict] = []
    for row in rows:
        segment_speaker = row.get("speaker")
        for word in row.get("words") or []:
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            text = str(word.get("word") or word.get("text") or "").strip()
            speaker = word.get("speaker") or segment_speaker
            if not speaker or not text:
                continue
            start_value = float(start)
            end_value = float(end)
            words.append(
                {
                    "speaker": str(speaker),
                    "start": start_value,
                    "end": end_value,
                    "midpoint": (start_value + end_value) / 2.0,
                }
            )
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def _dominant_reference_speaker(segment: dict, reference_words: List[dict], tolerance: float) -> tuple[Optional[str], int]:
    counts: Counter[str] = Counter()
    midpoint = (float(segment["start"]) + float(segment["end"])) / 2.0
    for word in reference_words:
        if float(word["end"]) < float(segment["start"]) - tolerance:
            continue
        if float(word["start"]) > float(segment["end"]) + tolerance:
            break
        overlaps = max(float(segment["start"]), float(word["start"])) < min(
            float(segment["end"]), float(word["end"])
        )
        if overlaps or abs(float(word["midpoint"]) - midpoint) <= tolerance:
            counts[str(word["speaker"])] += 1
    if not counts:
        return None, 0
    speaker, count = counts.most_common(1)[0]
    return speaker, int(count)


def _guess_reference_path(
    predicted_window_dir: Path,
    *,
    session_name: str,
    session22_reference_root: Optional[Path],
    session61_reference_root: Optional[Path],
) -> Optional[Path]:
    if session_name.lower() == "session22" and session22_reference_root is not None:
        return session22_reference_root / predicted_window_dir.name / "clips" / "clips.jsonl"
    if session_name.lower() == "session61" and session61_reference_root is not None:
        return session61_reference_root / predicted_window_dir.name / "reference" / "clips" / "clips.jsonl"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cached speaker-eval errors for flip/smoothing patterns.")
    parser.add_argument("--eval-root", required=True, help="Cached eval root containing Session22/Session61 folders.")
    parser.add_argument("--session22-reference-root", help="Reference cache root for Session22 windows.")
    parser.add_argument("--session61-reference-root", help="Reference cache root for Session61 windows.")
    parser.add_argument("--tolerance", type=float, default=0.35)
    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    session22_reference_root = (
        Path(args.session22_reference_root).expanduser().resolve()
        if args.session22_reference_root
        else None
    )
    session61_reference_root = (
        Path(args.session61_reference_root).expanduser().resolve()
        if args.session61_reference_root
        else None
    )

    overall_correct_durations: List[float] = []
    overall_wrong_durations: List[float] = []
    sandwich_patterns: Counter[str] = Counter()
    session_results: Dict[str, object] = {}

    for session_dir in sorted(eval_root.iterdir()):
        if not session_dir.is_dir():
            continue
        predicted_root = session_dir / "predicted"
        if not predicted_root.exists():
            continue
        session_payload: Dict[str, object] = {"windows": {}}
        for predicted_window_dir in sorted(predicted_root.glob("window_*")):
            predicted_jsonl = predicted_window_dir / "mixed" / "mixed.jsonl"
            if not predicted_jsonl.exists():
                continue
            reference_jsonl = _guess_reference_path(
                predicted_window_dir,
                session_name=session_dir.name,
                session22_reference_root=session22_reference_root,
                session61_reference_root=session61_reference_root,
            )
            if reference_jsonl is None or not reference_jsonl.exists():
                continue
            predicted_rows = _load_jsonl(predicted_jsonl)
            reference_rows = _load_jsonl(reference_jsonl)
            reference_words = _extract_words(reference_rows)
            segments = []
            for row in predicted_rows:
                start = float(row.get("start") or 0.0)
                end = float(row.get("end") or 0.0)
                if end <= start:
                    continue
                predicted_speaker = str(row.get("speaker") or "unknown")
                reference_speaker, supporting_words = _dominant_reference_speaker(
                    {"start": start, "end": end},
                    reference_words,
                    tolerance=float(args.tolerance),
                )
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "predicted_speaker": predicted_speaker,
                        "reference_speaker": reference_speaker,
                        "supporting_words": supporting_words,
                        "speaker_match_source": row.get("speaker_match_source"),
                        "speaker_match_score": row.get("speaker_match_score"),
                    }
                )

            total_segments = sum(1 for segment in segments if segment["reference_speaker"])
            wrong_segments = sum(
                1
                for segment in segments
                if segment["reference_speaker"]
                and segment["predicted_speaker"] != segment["reference_speaker"]
            )
            for index, segment in enumerate(segments):
                reference_speaker = segment["reference_speaker"]
                if not reference_speaker:
                    continue
                duration = float(segment["duration"])
                if segment["predicted_speaker"] == reference_speaker:
                    overall_correct_durations.append(duration)
                else:
                    overall_wrong_durations.append(duration)
                if 0 < index < len(segments) - 1:
                    previous_speaker = str(segments[index - 1]["predicted_speaker"])
                    next_speaker = str(segments[index + 1]["predicted_speaker"])
                    if previous_speaker == next_speaker and previous_speaker != str(segment["predicted_speaker"]):
                        sandwich_patterns["sandwich_total"] += 1
                        if previous_speaker == reference_speaker and segment["predicted_speaker"] != reference_speaker:
                            sandwich_patterns["sandwich_fixable"] += 1
                        if segment["predicted_speaker"] == reference_speaker:
                            sandwich_patterns["sandwich_already_correct"] += 1

            session_payload["windows"][predicted_window_dir.name] = {
                "total_segments": total_segments,
                "wrong_segments": wrong_segments,
                "wrong_rate": (wrong_segments / total_segments) if total_segments else 0.0,
            }
        session_results[session_dir.name] = session_payload

    summary = {
        "eval_root": str(eval_root),
        "sessions": session_results,
        "correct_duration_median": median(overall_correct_durations) if overall_correct_durations else 0.0,
        "wrong_duration_median": median(overall_wrong_durations) if overall_wrong_durations else 0.0,
        "wrong_duration_pct_le_2s": (
            sum(1 for value in overall_wrong_durations if value <= 2.0) / len(overall_wrong_durations)
            if overall_wrong_durations
            else 0.0
        ),
        "correct_duration_pct_le_2s": (
            sum(1 for value in overall_correct_durations if value <= 2.0) / len(overall_correct_durations)
            if overall_correct_durations
            else 0.0
        ),
        "sandwich_patterns": dict(sandwich_patterns),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
