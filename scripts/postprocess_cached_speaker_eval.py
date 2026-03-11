from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

from transcriber.multitrack_eval import WordSpan, extract_words_from_jsonl, score_word_speaker_alignment
from transcriber.segment_postprocess import smooth_short_speaker_flips


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="DOE for cached speaker-label postprocessing.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--session22-reference-root", required=True)
    parser.add_argument("--session61-reference-root", required=True)
    parser.add_argument("--durations", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--max-scores", default="0.35,0.40,0.45,0.50,0.60,none")
    parser.add_argument("--max-run-segments", default="1,2")
    parser.add_argument("--max-gap", type=float, default=0.75)
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    session22_reference_root = Path(args.session22_reference_root).expanduser().resolve()
    session61_reference_root = Path(args.session61_reference_root).expanduser().resolve()
    durations = [float(item) for item in args.durations.split(",") if item.strip()]
    max_scores: List[float | None] = []
    for item in args.max_scores.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value == "none":
            max_scores.append(None)
        else:
            max_scores.append(float(value))
    max_run_segments = [int(item) for item in args.max_run_segments.split(",") if item.strip()]

    session_windows: Dict[str, List[tuple[Path, Path]]] = {}
    for session_dir in sorted(eval_root.iterdir()):
        if not session_dir.is_dir():
            continue
        predicted_root = session_dir / "predicted"
        if not predicted_root.exists():
            continue
        pairs: List[tuple[Path, Path]] = []
        for predicted_window_dir in sorted(predicted_root.glob("window_*")):
            predicted_jsonl = predicted_window_dir / "mixed" / "mixed.jsonl"
            reference_jsonl = _guess_reference_path(
                predicted_window_dir,
                session_name=session_dir.name,
                session22_reference_root=session22_reference_root,
                session61_reference_root=session61_reference_root,
            )
            if predicted_jsonl.exists() and reference_jsonl is not None and reference_jsonl.exists():
                pairs.append((predicted_jsonl, reference_jsonl))
        if pairs:
            session_windows[session_dir.name] = pairs

    results: List[dict] = []
    baseline = {
        "config": {
            "max_total_duration": 0.0,
            "max_score": None,
            "max_run_segments": 0,
            "max_gap": float(args.max_gap),
        },
        "mean_accuracy": 0.0,
        "mean_coverage": 0.0,
        "sessions": {},
    }
    for session_name, pairs in session_windows.items():
        accuracies: List[float] = []
        coverages: List[float] = []
        for predicted_jsonl, reference_jsonl in pairs:
            metrics = score_word_speaker_alignment(
                extract_words_from_jsonl(reference_jsonl),
                extract_words_from_jsonl(predicted_jsonl),
                tolerance_seconds=float(args.tolerance),
            )
            accuracies.append(float(metrics.get("accuracy") or 0.0))
            coverages.append(float(metrics.get("coverage") or 0.0))
        baseline["sessions"][session_name] = {
            "mean_accuracy": _mean(accuracies),
            "mean_coverage": _mean(coverages),
        }
    baseline["mean_accuracy"] = _mean(
        [payload["mean_accuracy"] for payload in baseline["sessions"].values()]
    )
    baseline["mean_coverage"] = _mean(
        [payload["mean_coverage"] for payload in baseline["sessions"].values()]
    )
    results.append(baseline)

    for max_total_duration, max_score, run_segments in product(durations, max_scores, max_run_segments):
        result = {
            "config": {
                "max_total_duration": float(max_total_duration),
                "max_score": max_score,
                "max_run_segments": int(run_segments),
                "max_gap": float(args.max_gap),
            },
            "mean_accuracy": 0.0,
            "mean_coverage": 0.0,
            "sessions": {},
        }
        for session_name, pairs in session_windows.items():
            accuracies: List[float] = []
            coverages: List[float] = []
            for predicted_jsonl, reference_jsonl in pairs:
                smoothed_rows = smooth_short_speaker_flips(
                    _load_jsonl(predicted_jsonl),
                    max_total_duration=float(max_total_duration),
                    max_gap=float(args.max_gap),
                    max_score=max_score,
                    max_run_segments=int(run_segments),
                )
                reference_words = extract_words_from_jsonl(reference_jsonl)
                predicted_words: List[WordSpan] = []
                for row in smoothed_rows:
                    speaker = row.get("speaker")
                    for word in row.get("words") or []:
                        start = word.get("start")
                        end = word.get("end")
                        if start is None or end is None:
                            continue
                        text = str(word.get("word") or word.get("text") or "").strip()
                        word_speaker = word.get("speaker") or speaker
                        if not word_speaker or not text:
                            continue
                        predicted_words.append(
                            WordSpan(
                                speaker=str(word_speaker),
                                start=float(start),
                                end=float(end),
                                text=text,
                            )
                        )
                predicted_words.sort(key=lambda item: (item.start, item.end, item.text))
                metrics = score_word_speaker_alignment(
                    reference_words,
                    predicted_words,
                    tolerance_seconds=float(args.tolerance),
                )
                accuracies.append(float(metrics.get("accuracy") or 0.0))
                coverages.append(float(metrics.get("coverage") or 0.0))
            result["sessions"][session_name] = {
                "mean_accuracy": _mean(accuracies),
                "mean_coverage": _mean(coverages),
            }
        result["mean_accuracy"] = _mean(
            [payload["mean_accuracy"] for payload in result["sessions"].values()]
        )
        result["mean_coverage"] = _mean(
            [payload["mean_coverage"] for payload in result["sessions"].values()]
        )
        results.append(result)

    results.sort(key=lambda item: (float(item["mean_accuracy"]), float(item["mean_coverage"])), reverse=True)
    payload = {
        "eval_root": str(eval_root),
        "results": results,
    }
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
