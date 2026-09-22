from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from score_tse_word_ownership import (
    _db_ratio,
    _estimate_path,
    _group_key,
    _load_mono,
    _load_reference_groups,
    _read_jsonl,
    _span_energy,
    _write_jsonl,
)


def _split_values(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _word_overlap_speakers(
    word: Mapping[str, object],
    words: Sequence[Mapping[str, object]],
    *,
    min_overlap_seconds: float,
) -> set[str]:
    start = float(word.get("start") or 0.0)
    end = float(word.get("end") or start)
    active = set()
    for other in words:
        speaker = str(other.get("speaker") or "")
        if not speaker:
            continue
        other_start = float(other.get("start") or 0.0)
        other_end = float(other.get("end") or other_start)
        if min(end, other_end) - max(start, other_start) >= min_overlap_seconds:
            active.add(speaker)
    return active


def _rank_reference(energies: Mapping[str, float], reference: str) -> int | None:
    if reference not in energies:
        return None
    ordered = sorted(energies.items(), key=lambda item: item[1], reverse=True)
    for index, (speaker, _energy) in enumerate(ordered, start=1):
        if speaker == reference:
            return index
    return None


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    if scores.size == 0:
        return {
            "threshold": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    candidates = np.unique(scores)
    if candidates.size > 512:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 512))
    candidates = np.unique(
        np.concatenate([candidates, [scores.min() - 1e-12, scores.max() + 1e-12]])
    )
    best = {
        "threshold": float(candidates[0]),
        "accuracy": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
    }
    positives = labels.astype(bool)
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.logical_and(predicted, positives).sum())
        fp = int(np.logical_and(predicted, ~positives).sum())
        tn = int(np.logical_and(~predicted, ~positives).sum())
        fn = int(np.logical_and(~predicted, positives).sum())
        accuracy = (tp + tn) / max(labels.size, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if (f1, accuracy) > (best["f1"], best["accuracy"]):
            best = {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels.astype(bool)]
    negatives = scores[~labels.astype(bool)]
    if positives.size == 0 or negatives.size == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[labels.astype(bool)].sum())
    auc = (positive_rank_sum - positives.size * (positives.size + 1) / 2.0) / (
        positives.size * negatives.size
    )
    return float(auc)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels.astype(bool)
    positive_count = int(positives.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = positives[order]
    tp_cumulative = np.cumsum(sorted_labels)
    precision = tp_cumulative / (np.arange(sorted_labels.size) + 1)
    return float((precision * sorted_labels).sum() / positive_count)


def _threshold_predictions(
    pair_records: Sequence[Mapping[str, object]],
    thresholds: Mapping[tuple[str, str, float], float],
) -> dict[tuple[int, str], bool]:
    predicted = {}
    for record in pair_records:
        key = (
            str(record.get("session") or ""),
            str(record.get("candidate") or ""),
            round(float(record.get("window_start") or 0.0), 3),
        )
        threshold = thresholds.get(key)
        if threshold is None:
            threshold = thresholds.get(("", str(record.get("candidate") or ""), 0.0), 0.0)
        predicted[(int(record["global_word_index"]), str(record["candidate"]))] = (
            float(record.get("energy") or 0.0) >= threshold
        )
    return predicted


def _score_activity_oracle(pair_records: Sequence[Mapping[str, object]]) -> dict:
    if not pair_records:
        return {}
    scores = np.asarray([float(record.get("energy") or 0.0) for record in pair_records])
    labels = np.asarray(
        [1 if record.get("active") else 0 for record in pair_records], dtype=np.int32
    )
    global_threshold = _best_threshold(scores, labels)

    thresholds = {}
    per_speaker = {}
    grouped: dict[tuple[str, str, float], list[Mapping[str, object]]] = defaultdict(list)
    for record in pair_records:
        key = (
            str(record.get("session") or ""),
            str(record.get("candidate") or ""),
            round(float(record.get("window_start") or 0.0), 3),
        )
        grouped[key].append(record)
    for key, records in grouped.items():
        local_scores = np.asarray([float(record.get("energy") or 0.0) for record in records])
        local_labels = np.asarray(
            [1 if record.get("active") else 0 for record in records], dtype=np.int32
        )
        local = _best_threshold(local_scores, local_labels)
        thresholds[key] = float(local["threshold"])
        per_speaker[f"{key[0]}|{key[1]}|{key[2]:.3f}"] = {
            **local,
            "pairs": len(records),
            "active_pairs": int(local_labels.sum()),
        }

    predictions = _threshold_predictions(pair_records, thresholds)
    tp = fp = tn = fn = 0
    reference_active_hits = 0
    reference_active_total = 0
    reference_word_multiactive = 0
    by_overlap: dict[str, Counter[str]] = defaultdict(Counter)
    for record in pair_records:
        active = bool(record.get("active"))
        pred = predictions[(int(record["global_word_index"]), str(record["candidate"]))]
        if active and pred:
            tp += 1
        elif active:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
        bucket = "overlap" if record.get("word_overlap") else "non_overlap"
        by_overlap[bucket]["pairs"] += 1
        by_overlap[bucket]["active_pairs"] += int(active)
        by_overlap[bucket]["predicted_pairs"] += int(pred)
        by_overlap[bucket]["true_positive_pairs"] += int(active and pred)

        if record.get("is_reference_candidate"):
            reference_active_total += 1
            reference_active_hits += int(pred)
            reference_word_multiactive += int(record.get("word_overlap"))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pair_count": int(labels.size),
        "active_pair_count": int(labels.sum()),
        "global_threshold": global_threshold,
        "oracle_group_speaker_thresholds": {
            "accuracy": (tp + tn) / max(labels.size, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "reference_active_recall": reference_active_hits / max(reference_active_total, 1),
            "reference_active_hits": reference_active_hits,
            "reference_active_total": reference_active_total,
            "reference_words_with_overlap": reference_word_multiactive,
        },
        "roc_auc": _auc(scores, labels),
        "average_precision": _average_precision(scores, labels),
        "by_overlap": {
            bucket: {
                **dict(counts),
                "active_recall": counts["true_positive_pairs"] / max(counts["active_pairs"], 1),
                "predicted_active_rate": counts["predicted_pairs"] / max(counts["pairs"], 1),
            }
            for bucket, counts in sorted(by_overlap.items())
        },
        "thresholds": per_speaker,
    }


def _load_group_signals(
    rows: Sequence[Mapping[str, object]],
    *,
    estimates_dir: Path | None,
) -> tuple[dict[str, tuple[np.ndarray, int]], list[dict]]:
    signals = {}
    errors = []
    for row in rows:
        speaker = str(row.get("speaker_id") or "")
        path = _estimate_path(row, estimates_dir)
        if not path or not path.exists():
            errors.append({"row_id": row.get("row_id"), "error": "missing_estimate"})
            continue
        signals[speaker] = _load_mono(path)
    return signals, errors


def _score_group(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    *,
    words: Sequence[Mapping[str, object]],
    estimates_dir: Path | None,
    min_span_seconds: float,
    min_overlap_seconds: float,
    global_word_offset: int,
) -> tuple[dict, list[dict], list[dict]]:
    signals, errors = _load_group_signals(rows, estimates_dir=estimates_dir)
    if not signals:
        return (
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "error": "no_signals",
                "errors": errors,
            },
            [],
            [],
        )
    sample_rates = {sample_rate for _, sample_rate in signals.values()}
    if len(sample_rates) != 1:
        return (
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "error": "sample_rate_mismatch",
                "sample_rates": sorted(sample_rates),
            },
            [],
            [],
        )
    sample_rate = sample_rates.pop()
    speakers = sorted(signals)
    word_records = []
    pair_records = []
    ranks = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = overlap_correct = non_overlap_correct = 0
    overlap_words = non_overlap_words = 0
    top2_correct = 0
    margins = []
    missing_reference = 0

    for local_index, raw_word in enumerate(words):
        word = dict(raw_word)
        reference = str(word.get("speaker") or "")
        if not reference:
            continue
        global_word_index = global_word_offset + local_index
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        active_speakers = _word_overlap_speakers(
            word,
            words,
            min_overlap_seconds=min_overlap_seconds,
        )
        has_overlap = len(active_speakers - {reference}) > 0
        overlap_words += int(has_overlap)
        non_overlap_words += int(not has_overlap)
        if reference not in signals:
            missing_reference += 1
            continue
        energies = {
            speaker: _span_energy(
                wave,
                sample_rate=sample_rate,
                start=start,
                end=end,
                min_seconds=min_span_seconds,
            )
            for speaker, (wave, _) in signals.items()
        }
        ordered = sorted(energies.items(), key=lambda item: item[1], reverse=True)
        predicted = ordered[0][0]
        rank = _rank_reference(energies, reference)
        if rank is not None:
            ranks[str(rank)] += 1
            top2_correct += int(rank <= 2)
        best_non_reference = max(
            (energy for speaker, energy in energies.items() if speaker != reference),
            default=0.0,
        )
        margin = _db_ratio(energies.get(reference, 0.0), best_non_reference)
        margins.append(margin)
        is_correct = predicted == reference
        correct += int(is_correct)
        overlap_correct += int(is_correct and has_overlap)
        non_overlap_correct += int(is_correct and not has_overlap)
        confusion[reference][predicted] += 1
        for speaker in speakers:
            pair_records.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "global_word_index": global_word_index,
                    "local_word_index": local_index,
                    "candidate": speaker,
                    "reference": reference,
                    "energy": energies[speaker],
                    "active": speaker in active_speakers,
                    "is_reference_candidate": speaker == reference,
                    "word_overlap": has_overlap,
                }
            )
        word_records.append(
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "global_word_index": global_word_index,
                "local_word_index": local_index,
                "speaker": reference,
                "predicted": predicted,
                "correct": is_correct,
                "reference_rank": rank,
                "reference_margin_db": margin,
                "reference_energy": energies.get(reference, 0.0),
                "predicted_energy": energies[predicted],
                "best_non_reference_energy": best_non_reference,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "text": word.get("text"),
                "score": word.get("score"),
                "source_span_start": word.get("source_span_start"),
                "source_span_end": word.get("source_span_end"),
                "overlap": has_overlap,
                "active_speakers": sorted(active_speakers),
                "energies": energies,
            }
        )

    reference_words = len([word for word in words if word.get("speaker")])
    return (
        {
            "session": key[0],
            "window_start": key[1],
            "window_end": key[2],
            "speakers": speakers,
            "reference_words": reference_words,
            "missing_reference_words": missing_reference,
            "correct_words": correct,
            "accuracy": correct / reference_words if reference_words else 0.0,
            "overlap_words": overlap_words,
            "overlap_correct_words": overlap_correct,
            "overlap_accuracy": overlap_correct / overlap_words if overlap_words else None,
            "non_overlap_words": non_overlap_words,
            "non_overlap_correct_words": non_overlap_correct,
            "non_overlap_accuracy": (
                non_overlap_correct / non_overlap_words if non_overlap_words else None
            ),
            "top2_correct_words": top2_correct,
            "top2_accuracy": top2_correct / reference_words if reference_words else 0.0,
            "mean_reference_margin_db": float(np.mean(margins)) if margins else None,
            "median_reference_margin_db": float(np.median(margins)) if margins else None,
            "reference_rank_counts": dict(sorted(ranks.items(), key=lambda item: int(item[0]))),
            "confusion": {speaker: dict(counts) for speaker, counts in sorted(confusion.items())},
            "errors": errors,
        },
        word_records,
        pair_records,
    )


def _summarize_groups(
    groups: Sequence[Mapping[str, object]],
    word_records: Sequence[Mapping[str, object]],
    activity: Mapping[str, object],
) -> dict:
    valid = [group for group in groups if not group.get("error")]
    reference_words = sum(int(group.get("reference_words") or 0) for group in valid)
    correct_words = sum(int(group.get("correct_words") or 0) for group in valid)
    overlap_words = sum(int(group.get("overlap_words") or 0) for group in valid)
    overlap_correct_words = sum(int(group.get("overlap_correct_words") or 0) for group in valid)
    non_overlap_words = sum(int(group.get("non_overlap_words") or 0) for group in valid)
    non_overlap_correct_words = sum(
        int(group.get("non_overlap_correct_words") or 0) for group in valid
    )
    top2_correct_words = sum(int(group.get("top2_correct_words") or 0) for group in valid)
    margin_values = [
        float(record["reference_margin_db"])
        for record in word_records
        if record.get("reference_margin_db") is not None
    ]
    wrong = [record for record in word_records if not record.get("correct")]
    wrong_overlap = sum(1 for record in wrong if record.get("overlap"))
    wrong_top2 = sum(1 for record in wrong if int(record.get("reference_rank") or 99) <= 2)
    wrong_low_score = sum(1 for record in wrong if float(record.get("score") or 0.0) < 0.05)
    wrong_short = sum(1 for record in wrong if float(record.get("duration") or 0.0) < 0.08)
    by_session: dict[str, Counter[str]] = defaultdict(Counter)
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    for record in word_records:
        for bucket in (
            by_session[str(record.get("session") or "")],
            by_speaker[str(record.get("speaker") or "")],
        ):
            bucket["words"] += 1
            bucket["correct"] += int(record.get("correct"))
            bucket["overlap"] += int(record.get("overlap"))
            bucket["top2"] += int(int(record.get("reference_rank") or 99) <= 2)

    def _quality_slice(name: str, selected: Sequence[Mapping[str, object]]) -> tuple[str, dict]:
        words = len(selected)
        correct = sum(1 for record in selected if record.get("correct"))
        top2 = sum(1 for record in selected if int(record.get("reference_rank") or 99) <= 2)
        overlap = sum(1 for record in selected if record.get("overlap"))
        return name, {
            "words": words,
            "energy_winner_accuracy": correct / words if words else 0.0,
            "top2_accuracy": top2 / words if words else 0.0,
            "overlap_words": overlap,
            "word_share": words / len(word_records) if word_records else 0.0,
        }

    quality_filters = dict(
        [
            _quality_slice("all", list(word_records)),
            _quality_slice(
                "score_ge_0_05",
                [record for record in word_records if float(record.get("score") or 0.0) >= 0.05],
            ),
            _quality_slice(
                "duration_ge_80ms",
                [record for record in word_records if float(record.get("duration") or 0.0) >= 0.08],
            ),
            _quality_slice(
                "score_ge_0_05_and_duration_ge_80ms",
                [
                    record
                    for record in word_records
                    if float(record.get("score") or 0.0) >= 0.05
                    and float(record.get("duration") or 0.0) >= 0.08
                ],
            ),
            _quality_slice(
                "non_overlap",
                [record for record in word_records if not record.get("overlap")],
            ),
            _quality_slice(
                "non_overlap_score_ge_0_05",
                [
                    record
                    for record in word_records
                    if not record.get("overlap") and float(record.get("score") or 0.0) >= 0.05
                ],
            ),
            _quality_slice(
                "non_overlap_score_ge_0_05_duration_ge_80ms",
                [
                    record
                    for record in word_records
                    if not record.get("overlap")
                    and float(record.get("score") or 0.0) >= 0.05
                    and float(record.get("duration") or 0.0) >= 0.08
                ],
            ),
        ]
    )
    return {
        "group_count": len(groups),
        "valid_groups": len(valid),
        "reference_words": reference_words,
        "correct_words": correct_words,
        "energy_winner_accuracy": correct_words / reference_words if reference_words else 0.0,
        "overlap_words": overlap_words,
        "overlap_correct_words": overlap_correct_words,
        "overlap_energy_winner_accuracy": (
            overlap_correct_words / overlap_words if overlap_words else None
        ),
        "non_overlap_words": non_overlap_words,
        "non_overlap_correct_words": non_overlap_correct_words,
        "non_overlap_energy_winner_accuracy": (
            non_overlap_correct_words / non_overlap_words if non_overlap_words else None
        ),
        "top2_correct_words": top2_correct_words,
        "top2_accuracy": top2_correct_words / reference_words if reference_words else 0.0,
        "mean_reference_margin_db": float(np.mean(margin_values)) if margin_values else None,
        "median_reference_margin_db": float(np.median(margin_values)) if margin_values else None,
        "wrong_word_count": len(wrong),
        "wrong_overlap_words": wrong_overlap,
        "wrong_overlap_share": wrong_overlap / len(wrong) if wrong else 0.0,
        "wrong_reference_rank_top2": wrong_top2,
        "wrong_reference_rank_top2_share": wrong_top2 / len(wrong) if wrong else 0.0,
        "wrong_low_forced_score_lt_0_05": wrong_low_score,
        "wrong_short_lt_80ms": wrong_short,
        "quality_filters": quality_filters,
        "activity_oracle": dict(activity),
        "by_session": {
            key: {
                "words": int(counts["words"]),
                "correct": int(counts["correct"]),
                "energy_winner_accuracy": (
                    counts["correct"] / counts["words"] if counts["words"] else 0.0
                ),
                "overlap_words": int(counts["overlap"]),
                "top2_accuracy": counts["top2"] / counts["words"] if counts["words"] else 0.0,
            }
            for key, counts in sorted(by_session.items())
        },
        "by_speaker": {
            key: {
                "words": int(counts["words"]),
                "correct": int(counts["correct"]),
                "energy_winner_accuracy": (
                    counts["correct"] / counts["words"] if counts["words"] else 0.0
                ),
                "overlap_words": int(counts["overlap"]),
                "top2_accuracy": counts["top2"] / counts["words"] if counts["words"] else 0.0,
            }
            for key, counts in sorted(by_speaker.items())
        },
    }


def _filtered_group_rows(
    manifest: Path,
    *,
    eval_splits: set[str],
    eval_sessions: set[str],
) -> dict[tuple[str, float, float], list[dict]]:
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in _read_jsonl(manifest):
        if not row.get("materialized"):
            continue
        if eval_splits and str(row.get("split_id") or "") not in eval_splits:
            continue
        if eval_sessions and str(row.get("session") or "") not in eval_sessions:
            continue
        grouped[_group_key(row)].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit clean-source forced-word ownership as single-winner and multi-label activity."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimates-dir", type=Path)
    parser.add_argument("--eval-splits", default="test")
    parser.add_argument("--eval-sessions")
    parser.add_argument("--min-span-seconds", type=float, default=0.12)
    parser.add_argument("--min-overlap-seconds", type=float, default=0.02)
    parser.add_argument("--save-pairs", action="store_true")
    args = parser.parse_args()

    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped = _filtered_group_rows(
        args.manifest,
        eval_splits=_split_values(args.eval_splits),
        eval_sessions=_split_values(args.eval_sessions),
    )
    group_results = []
    word_records = []
    pair_records = []
    global_word_offset = 0
    for key, rows in sorted(grouped.items()):
        if key not in reference_groups:
            continue
        group, words, pairs = _score_group(
            key,
            rows,
            words=reference_groups[key],
            estimates_dir=args.estimates_dir,
            min_span_seconds=float(args.min_span_seconds),
            min_overlap_seconds=float(args.min_overlap_seconds),
            global_word_offset=global_word_offset,
        )
        group_results.append(group)
        word_records.extend(words)
        pair_records.extend(pairs)
        global_word_offset += len(reference_groups[key])

    activity = _score_activity_oracle(pair_records)
    summary = _summarize_groups(group_results, word_records, activity)
    summary.update(
        {
            "manifest": str(args.manifest),
            "reference_jsonl": str(args.reference_jsonl),
            "estimates_dir": str(args.estimates_dir) if args.estimates_dir else None,
            "eval_splits": sorted(_split_values(args.eval_splits)),
            "eval_sessions": sorted(_split_values(args.eval_sessions)),
            "min_span_seconds": float(args.min_span_seconds),
            "min_overlap_seconds": float(args.min_overlap_seconds),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "clean_source_word_oracle_groups.jsonl", group_results)
    _write_jsonl(args.output_dir / "clean_source_word_oracle_words.jsonl", word_records)
    if args.save_pairs:
        _write_jsonl(args.output_dir / "clean_source_word_oracle_pairs.jsonl", pair_records)
    (args.output_dir / "clean_source_word_oracle_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
