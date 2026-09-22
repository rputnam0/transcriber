from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from transcriber.asr import transcribe_with_faster_whisper  # noqa: E402
from transcriber.multitrack_eval import WordSpan, score_word_speaker_alignment  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _safe_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _load_reference_groups(path: Path | None) -> dict[tuple[str, float, float], list[WordSpan]]:
    if path is None:
        return {}
    groups: dict[tuple[str, float, float], list[WordSpan]] = {}
    for row in _read_jsonl(path):
        key = (
            str(row.get("session") or ""),
            round(float(row.get("window_start") or 0.0), 3),
            round(float(row.get("window_end") or 0.0), 3),
        )
        words = []
        for item in row.get("words") or []:
            word = dict(item)
            speaker = str(word.get("speaker") or "")
            text = str(word.get("text") or "").strip()
            start = word.get("start")
            end = word.get("end")
            if not speaker or not text or start is None or end is None:
                continue
            words.append(
                WordSpan(
                    speaker=speaker,
                    start=float(start),
                    end=float(end),
                    text=text,
                )
            )
        groups[key] = sorted(words, key=lambda item: (item.start, item.end, item.speaker))
    return groups


def _estimate_path(row: Mapping[str, object], estimates_dir: Path | None) -> Path | None:
    materialized = dict(row.get("materialized") or {})
    if estimates_dir is None:
        target_path = materialized.get("target_source_path")
        return Path(str(target_path)) if target_path else None
    row_id = str(row.get("row_id") or "")
    for candidate in (
        estimates_dir / f"{row_id}.wav",
        estimates_dir / row_id / "estimate.wav",
        estimates_dir / row_id / "target.wav",
    ):
        if candidate.exists():
            return candidate
    return None


def _select_groups(
    rows: Sequence[dict],
    *,
    session: str | None,
    window_start: float | None,
    max_groups: int | None,
) -> list[tuple[tuple[str, float, float], list[dict]]]:
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("materialized"):
            continue
        key = _group_key(row)
        if session and key[0] != session:
            continue
        if window_start is not None and abs(key[1] - window_start) > 1e-3:
            continue
        grouped[key].append(row)
    items = sorted(grouped.items())
    return items[:max_groups] if max_groups is not None else items


def _reference_words(
    rows: Sequence[Mapping[str, object]],
    *,
    key: tuple[str, float, float],
    reference_groups: Mapping[tuple[str, float, float], Sequence[WordSpan]],
) -> list[WordSpan]:
    if key in reference_groups:
        return list(reference_groups[key])
    words: list[WordSpan] = []
    seen = set()
    for raw_span in rows[0].get("word_spans") or []:
        span = dict(raw_span)
        speaker = str(span.get("speaker") or "")
        tokens = str(span.get("text") or "").split()
        if not speaker or not tokens:
            continue
        start = float(span.get("start") or 0.0)
        end = float(span.get("end") or start)
        if end <= start:
            end = start + 0.2
        duration = max(end - start, 0.05)
        step = duration / len(tokens)
        for index, token in enumerate(tokens):
            word_start = start + index * step
            word_end = start + (index + 1) * step
            key = (speaker, round(word_start, 3), round(word_end, 3), token)
            if key in seen:
                continue
            seen.add(key)
            words.append(WordSpan(speaker=speaker, start=word_start, end=word_end, text=token))
    words.sort(key=lambda item: (item.start, item.end, item.speaker))
    return words


def _cache_identity(
    row: Mapping[str, object],
    audio_path: Path,
    *,
    model_name: str,
    compute_type: str,
    device: str,
    batch_size: int,
) -> dict:
    stat = audio_path.stat()
    try:
        faster_whisper_version = importlib.metadata.version("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        faster_whisper_version = "unknown"
    return {
        "row_id": row.get("row_id"),
        "audio_path": str(audio_path),
        "audio_mtime_ns": stat.st_mtime_ns,
        "audio_size": stat.st_size,
        "asr_model": model_name,
        "asr_device": device,
        "compute_type": compute_type,
        "batch_size": int(batch_size),
        "decode_options": {
            "vad_filter": True,
            "word_timestamps": True,
        },
        "faster_whisper_version": faster_whisper_version,
    }


def _cache_path(cache_dir: Path, identity: Mapping[str, object]) -> Path:
    payload = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"


def _transcribe_track(
    *,
    row: Mapping[str, object],
    audio_path: Path,
    cache_dir: Path,
    model_name: str,
    compute_type: str,
    device: str,
    batch_size: int,
) -> list[WordSpan]:
    cache_identity = _cache_identity(
        row,
        audio_path,
        model_name=model_name,
        compute_type=compute_type,
        device=device,
        batch_size=batch_size,
    )
    cache_file = _cache_path(cache_dir, cache_identity)
    speaker = str(row.get("speaker_id") or "")
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        result = transcribe_with_faster_whisper(
            str(audio_path),
            model_name=model_name,
            compute_type=compute_type,
            device=device,
            batch_size=batch_size,
        )
        payload = {
            "audio_path": str(audio_path),
            "speaker_id": speaker,
            "segments": [segment.to_dict() for segment in result.segments],
            "metadata": result.metadata,
            "cache_identity": cache_identity,
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    spans: list[WordSpan] = []
    for segment in payload.get("segments") or []:
        for word in dict(segment).get("words") or []:
            item = dict(word)
            text = str(item.get("word") or item.get("text") or "").strip()
            if not text:
                continue
            start = item.get("start")
            end = item.get("end")
            if start is None or end is None:
                continue
            spans.append(
                WordSpan(
                    speaker=speaker,
                    start=float(start),
                    end=float(end),
                    text=text,
                )
            )
    return spans


def _score_group(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    *,
    estimates_dir: Path | None,
    cache_dir: Path,
    model_name: str,
    compute_type: str,
    device: str,
    batch_size: int,
    tolerance_seconds: float,
    reference_groups: Mapping[tuple[str, float, float], Sequence[WordSpan]],
) -> dict:
    reference = _reference_words(rows, key=key, reference_groups=reference_groups)
    predicted: list[WordSpan] = []
    track_records = []
    for row in rows:
        audio_path = _estimate_path(row, estimates_dir)
        if not audio_path or not audio_path.exists():
            track_records.append({"row_id": row.get("row_id"), "error": "missing_track"})
            continue
        track_words = _transcribe_track(
            row=row,
            audio_path=audio_path,
            cache_dir=cache_dir,
            model_name=model_name,
            compute_type=compute_type,
            device=device,
            batch_size=batch_size,
        )
        predicted.extend(track_words)
        track_records.append(
            {
                "row_id": row.get("row_id"),
                "speaker_id": row.get("speaker_id"),
                "audio_path": str(audio_path),
                "predicted_words": len(track_words),
            }
        )
    metrics = score_word_speaker_alignment(
        reference,
        predicted,
        tolerance_seconds=tolerance_seconds,
    )
    diagnostics = _track_coverage_diagnostics(
        reference,
        predicted,
        tolerance_seconds=tolerance_seconds,
    )
    return {
        "session": key[0],
        "window_start": key[1],
        "window_end": key[2],
        "reference_words": metrics["reference_words"],
        "predicted_words": metrics["predicted_words"],
        "matched_words": metrics["matched_words"],
        "correct_words": metrics["correct_words"],
        "timed_speaker_hit_words": metrics["timed_speaker_hit_words"],
        "coverage": metrics["coverage"],
        "accuracy": metrics["accuracy"],
        "accuracy_metric": metrics.get("accuracy_metric", "timed_speaker_hit_rate"),
        "timed_speaker_hit_rate": metrics["timed_speaker_hit_rate"],
        "matched_accuracy": metrics["matched_accuracy"],
        "lexical_matched_words": metrics["lexical_matched_words"],
        "lexical_correct_words": metrics["lexical_correct_words"],
        "lexical_coverage": metrics["lexical_coverage"],
        "lexical_accuracy": metrics["lexical_accuracy"],
        "lexical_matched_accuracy": metrics["lexical_matched_accuracy"],
        "speaker_attributed_lexical_accuracy": metrics["speaker_attributed_lexical_accuracy"],
        "speaker_attributed_lexical_matched_accuracy": metrics[
            "speaker_attributed_lexical_matched_accuracy"
        ],
        "track_diagnostics": diagnostics,
        "confusion": metrics["confusion"],
        "tracks": track_records,
    }


def _track_coverage_diagnostics(
    reference: Sequence[WordSpan],
    predicted: Sequence[WordSpan],
    *,
    tolerance_seconds: float,
) -> dict:
    temporal_candidates = 0
    same_speaker_candidates = 0
    nearest_same_speaker = 0
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    predicted_sorted = sorted(predicted, key=lambda item: (item.start, item.end))
    for ref in reference:
        by_speaker[ref.speaker]["reference"] += 1
        midpoint = ref.midpoint
        candidates = [
            pred
            for pred in predicted_sorted
            if (
                abs(pred.midpoint - midpoint) <= tolerance_seconds
                or pred.start <= midpoint <= pred.end
            )
        ]
        if candidates:
            temporal_candidates += 1
            by_speaker[ref.speaker]["temporal_candidate"] += 1
            nearest = min(candidates, key=lambda item: abs(item.midpoint - midpoint))
            if nearest.speaker == ref.speaker:
                nearest_same_speaker += 1
                by_speaker[ref.speaker]["nearest_same_speaker"] += 1
        same_speaker = [pred for pred in candidates if pred.speaker == ref.speaker]
        if same_speaker:
            same_speaker_candidates += 1
            by_speaker[ref.speaker]["same_speaker_candidate"] += 1

    reference_words = len(reference)
    predicted_words = len(predicted)
    return {
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "temporal_candidate_words": temporal_candidates,
        "same_speaker_candidate_words": same_speaker_candidates,
        "nearest_same_speaker_words": nearest_same_speaker,
        "temporal_candidate_coverage": (
            temporal_candidates / reference_words if reference_words else 0.0
        ),
        "same_speaker_candidate_coverage": (
            same_speaker_candidates / reference_words if reference_words else 0.0
        ),
        "nearest_same_speaker_accuracy": (
            nearest_same_speaker / reference_words if reference_words else 0.0
        ),
        "same_speaker_covered_words_per_prediction": (
            same_speaker_candidates / predicted_words if predicted_words else 0.0
        ),
        "prediction_density": predicted_words / reference_words if reference_words else 0.0,
        "by_speaker": {
            speaker: {
                "reference": counts["reference"],
                "temporal_candidate": counts["temporal_candidate"],
                "same_speaker_candidate": counts["same_speaker_candidate"],
                "nearest_same_speaker": counts["nearest_same_speaker"],
                "same_speaker_candidate_coverage": (
                    counts["same_speaker_candidate"] / counts["reference"]
                    if counts["reference"]
                    else 0.0
                ),
            }
            for speaker, counts in sorted(by_speaker.items())
        },
    }


def _sum_confusion(groups: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for reference, counts in dict(group.get("confusion") or {}).items():
            for predicted, count in dict(counts).items():
                confusion[str(reference)][str(predicted)] += int(count)
    return {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}


def _sum_track_diagnostics(groups: Sequence[Mapping[str, object]]) -> dict:
    totals: Counter[str] = Counter()
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        diagnostics = dict(group.get("track_diagnostics") or {})
        for key in (
            "reference_words",
            "predicted_words",
            "temporal_candidate_words",
            "same_speaker_candidate_words",
            "nearest_same_speaker_words",
        ):
            totals[key] += int(diagnostics.get(key) or 0)
        for speaker, counts in dict(diagnostics.get("by_speaker") or {}).items():
            for key in (
                "reference",
                "temporal_candidate",
                "same_speaker_candidate",
                "nearest_same_speaker",
            ):
                by_speaker[str(speaker)][key] += int(dict(counts).get(key) or 0)

    reference_words = totals["reference_words"]
    predicted_words = totals["predicted_words"]
    return {
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "temporal_candidate_words": totals["temporal_candidate_words"],
        "same_speaker_candidate_words": totals["same_speaker_candidate_words"],
        "nearest_same_speaker_words": totals["nearest_same_speaker_words"],
        "temporal_candidate_coverage": (
            totals["temporal_candidate_words"] / reference_words if reference_words else 0.0
        ),
        "same_speaker_candidate_coverage": (
            totals["same_speaker_candidate_words"] / reference_words if reference_words else 0.0
        ),
        "nearest_same_speaker_accuracy": (
            totals["nearest_same_speaker_words"] / reference_words if reference_words else 0.0
        ),
        "same_speaker_covered_words_per_prediction": (
            totals["same_speaker_candidate_words"] / predicted_words if predicted_words else 0.0
        ),
        "prediction_density": predicted_words / reference_words if reference_words else 0.0,
        "by_speaker": {
            speaker: {
                "reference": counts["reference"],
                "temporal_candidate": counts["temporal_candidate"],
                "same_speaker_candidate": counts["same_speaker_candidate"],
                "nearest_same_speaker": counts["nearest_same_speaker"],
                "same_speaker_candidate_coverage": (
                    counts["same_speaker_candidate"] / counts["reference"]
                    if counts["reference"]
                    else 0.0
                ),
            }
            for speaker, counts in sorted(by_speaker.items())
        },
    }


def _summarize(groups: Sequence[Mapping[str, object]]) -> dict:
    reference_words = sum(int(group.get("reference_words") or 0) for group in groups)
    predicted_words = sum(int(group.get("predicted_words") or 0) for group in groups)
    matched_words = sum(int(group.get("matched_words") or 0) for group in groups)
    correct_words = sum(int(group.get("correct_words") or 0) for group in groups)
    lexical_matched_words = sum(int(group.get("lexical_matched_words") or 0) for group in groups)
    lexical_correct_words = sum(int(group.get("lexical_correct_words") or 0) for group in groups)
    lexical_accuracy = lexical_correct_words / reference_words if reference_words else 0.0
    lexical_matched_accuracy = (
        lexical_correct_words / lexical_matched_words if lexical_matched_words else 0.0
    )
    return {
        "group_count": len(groups),
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "matched_words": matched_words,
        "correct_words": correct_words,
        "timed_speaker_hit_words": correct_words,
        "coverage": matched_words / reference_words if reference_words else 0.0,
        "accuracy": correct_words / reference_words if reference_words else 0.0,
        "accuracy_metric": "timed_speaker_hit_rate",
        "primary_metric": "speaker_attributed_lexical_accuracy",
        "timed_speaker_hit_rate": correct_words / reference_words if reference_words else 0.0,
        "matched_accuracy": correct_words / matched_words if matched_words else 0.0,
        "lexical_matched_words": lexical_matched_words,
        "lexical_correct_words": lexical_correct_words,
        "lexical_coverage": lexical_matched_words / reference_words if reference_words else 0.0,
        "lexical_accuracy": lexical_accuracy,
        "lexical_matched_accuracy": lexical_matched_accuracy,
        "speaker_attributed_lexical_accuracy": lexical_accuracy,
        "speaker_attributed_lexical_matched_accuracy": lexical_matched_accuracy,
        "prediction_precision_proxy": correct_words / predicted_words if predicted_words else 0.0,
        "lexical_prediction_precision_proxy": (
            lexical_correct_words / predicted_words if predicted_words else 0.0
        ),
        "track_diagnostics": _sum_track_diagnostics(groups),
        "confusion": _sum_confusion(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score speaker-attributed ASR by transcribing target-speaker extraction tracks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimates-dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional ASR cache directory. Defaults to OUTPUT_DIR/asr_cache.",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        help="Optional forced-aligned reference groups JSONL from build_tse_forced_word_reference.py.",
    )
    parser.add_argument("--session")
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerance-seconds", type=float, default=0.75)
    args = parser.parse_args()

    rows = list(_read_jsonl(args.manifest))
    groups = _select_groups(
        rows,
        session=args.session,
        window_start=args.window_start,
        max_groups=args.max_groups,
    )
    reference_groups = _load_reference_groups(args.reference_jsonl)
    cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "asr_cache"
    group_results = [
        _score_group(
            key,
            group_rows,
            estimates_dir=args.estimates_dir,
            cache_dir=cache_dir,
            model_name=args.model,
            compute_type=args.compute_type,
            device=args.device,
            batch_size=int(args.batch_size),
            tolerance_seconds=float(args.tolerance_seconds),
            reference_groups=reference_groups,
        )
        for key, group_rows in groups
    ]
    summary = _summarize(group_results)
    summary["manifest"] = str(args.manifest)
    summary["cache_dir"] = str(cache_dir)
    if args.reference_jsonl:
        summary["reference_jsonl"] = str(args.reference_jsonl)
    if args.estimates_dir:
        summary["estimates_dir"] = str(args.estimates_dir)
    summary["model"] = args.model
    summary["tolerance_seconds"] = args.tolerance_seconds
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "speaker_attributed_asr_groups.jsonl", group_results)
    (args.output_dir / "speaker_attributed_asr_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
