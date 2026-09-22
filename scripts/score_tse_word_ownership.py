from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0), int(sample_rate)


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


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _load_reference_groups(path: Path | None) -> dict[tuple[str, float, float], list[dict]]:
    if path is None:
        return {}
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in _read_jsonl(path):
        key = _group_key(row)
        groups[key] = [dict(word) for word in row.get("words") or []]
    return groups


def _reference_items(
    rows: Sequence[Mapping[str, object]],
    *,
    key: tuple[str, float, float],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
) -> list[dict]:
    if key in reference_groups:
        return [
            {
                "speaker": str(word.get("speaker") or ""),
                "start": float(word.get("start") or 0.0),
                "end": float(word.get("end") or 0.0),
                "word_count": 1,
                "text": str(word.get("text") or ""),
                "score": word.get("score"),
                "reference_source": "forced",
            }
            for word in reference_groups[key]
            if word.get("speaker") and word.get("text")
        ]
    return [
        {
            **dict(span),
            "word_count": max(1, int(dict(span).get("word_count") or 1)),
            "reference_source": "manifest_span",
        }
        for span in rows[0].get("word_spans") or []
    ]


def _span_energy(
    wave: np.ndarray,
    *,
    sample_rate: int,
    start: float,
    end: float,
    min_seconds: float,
) -> float:
    midpoint = (start + end) / 2.0
    half = max((end - start) / 2.0, min_seconds / 2.0)
    start_sample = max(0, int(round((midpoint - half) * sample_rate)))
    end_sample = min(wave.shape[0], int(round((midpoint + half) * sample_rate)))
    if end_sample <= start_sample:
        return 0.0
    chunk = wave[start_sample:end_sample].astype(np.float64)
    return float(np.mean(chunk * chunk)) if chunk.size else 0.0


def _db_ratio(numerator: float, denominator: float, *, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(numerator), eps) / max(float(denominator), eps))


def _track_energy_scale(
    wave: np.ndarray,
    *,
    sample_rate: int,
    normalization: str,
    frame_seconds: float = 0.1,
) -> float:
    if normalization == "none":
        return 1.0
    if normalization != "track-p95":
        raise ValueError(f"Unknown energy normalization: {normalization}")
    frame_samples = max(1, int(round(frame_seconds * sample_rate)))
    frame_count = math.ceil(wave.shape[0] / frame_samples)
    padded = np.pad(wave.astype(np.float64), (0, frame_count * frame_samples - wave.shape[0]))
    frame_energy = np.mean(padded.reshape(frame_count, frame_samples) ** 2, axis=1)
    return max(float(np.quantile(frame_energy, 0.95)), 1e-12)


def _score_group(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    *,
    estimates_dir: Path | None,
    min_span_seconds: float,
    energy_normalization: str,
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
) -> dict:
    signals: dict[str, tuple[np.ndarray, int]] = {}
    errors = []
    for row in rows:
        speaker = str(row.get("speaker_id") or "")
        estimate_path = _estimate_path(row, estimates_dir)
        if not estimate_path or not estimate_path.exists():
            errors.append({"row_id": row.get("row_id"), "error": "missing_estimate"})
            continue
        signals[speaker] = _load_mono(estimate_path)
    if not signals:
        return {"error": "no_signals", "errors": errors}

    sample_rates = {sample_rate for _, sample_rate in signals.values()}
    if len(sample_rates) != 1:
        return {"error": "sample_rate_mismatch", "sample_rates": sorted(sample_rates)}
    sample_rate = sample_rates.pop()
    energy_scales = {
        speaker: _track_energy_scale(
            wave,
            sample_rate=sample_rate,
            normalization=energy_normalization,
        )
        for speaker, (wave, _) in signals.items()
    }
    spans = _reference_items(rows, key=key, reference_groups=reference_groups)
    reference_words = 0
    scored_words = 0
    correct_words = 0
    skipped_missing_speaker_words = 0
    margins = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for raw_span in spans:
        span = dict(raw_span)
        reference = str(span.get("speaker") or "")
        word_count = max(1, int(span.get("word_count") or 1))
        reference_words += word_count
        if reference not in signals:
            skipped_missing_speaker_words += word_count
            continue
        start = float(span.get("start") or 0.0)
        end = float(span.get("end") or start)
        energies = {
            speaker: (
                _span_energy(
                    wave,
                    sample_rate=sample_rate,
                    start=start,
                    end=end,
                    min_seconds=min_span_seconds,
                )
                / energy_scales[speaker]
            )
            for speaker, (wave, _) in signals.items()
        }
        predicted = max(energies.items(), key=lambda item: item[1])[0]
        best_non_reference = max(
            (energy for speaker, energy in energies.items() if speaker != reference),
            default=0.0,
        )
        margins.append(_db_ratio(energies.get(reference, 0.0), best_non_reference))
        scored_words += word_count
        confusion[reference][predicted] += word_count
        if predicted == reference:
            correct_words += word_count

    return {
        "speakers": sorted(signals),
        "reference_words": reference_words,
        "scored_words": scored_words,
        "skipped_missing_speaker_words": skipped_missing_speaker_words,
        "correct_words": correct_words,
        "coverage": scored_words / reference_words if reference_words else 0.0,
        "accuracy": correct_words / reference_words if reference_words else 0.0,
        "scored_accuracy": correct_words / scored_words if scored_words else 0.0,
        "mean_reference_energy_margin_db": float(np.mean(margins)) if margins else None,
        "energy_normalization": energy_normalization,
        "reference_source": spans[0].get("reference_source") if spans else "unknown",
        "confusion": {speaker: dict(counts) for speaker, counts in sorted(confusion.items())},
        "errors": errors,
    }


def _sum_confusion(groups: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for reference, counts in dict(group.get("confusion") or {}).items():
            for predicted, count in dict(counts).items():
                confusion[str(reference)][str(predicted)] += int(count)
    return {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}


def _summarize(groups: Sequence[Mapping[str, object]]) -> dict:
    valid = [group for group in groups if not group.get("error")]
    reference_words = sum(int(group.get("reference_words") or 0) for group in valid)
    scored_words = sum(int(group.get("scored_words") or 0) for group in valid)
    correct_words = sum(int(group.get("correct_words") or 0) for group in valid)
    skipped = sum(int(group.get("skipped_missing_speaker_words") or 0) for group in valid)
    margins = [
        float(group["mean_reference_energy_margin_db"])
        for group in valid
        if group.get("mean_reference_energy_margin_db") is not None
    ]
    return {
        "group_count": len(groups),
        "valid_groups": len(valid),
        "reference_words": reference_words,
        "scored_words": scored_words,
        "skipped_missing_speaker_words": skipped,
        "correct_words": correct_words,
        "coverage": scored_words / reference_words if reference_words else 0.0,
        "accuracy": correct_words / reference_words if reference_words else 0.0,
        "scored_accuracy": correct_words / scored_words if scored_words else 0.0,
        "mean_reference_energy_margin_db": float(np.mean(margins)) if margins else None,
        "confusion": _sum_confusion(valid),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score word ownership from target-speaker extraction energy tracks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimates-dir", type=Path)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--min-span-seconds", type=float, default=0.12)
    parser.add_argument(
        "--energy-normalization",
        choices=("none", "track-p95"),
        default="none",
        help="Calibrate candidate energies to remove model/enrollment output-scale differences.",
    )
    args = parser.parse_args()

    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in _read_jsonl(args.manifest):
        if row.get("materialized"):
            grouped[_group_key(row)].append(row)

    group_results = []
    for key, rows in sorted(grouped.items()):
        result = _score_group(
            key,
            rows,
            estimates_dir=args.estimates_dir,
            min_span_seconds=float(args.min_span_seconds),
            energy_normalization=str(args.energy_normalization),
            reference_groups=reference_groups,
        )
        result.update({"session": key[0], "window_start": key[1], "window_end": key[2]})
        group_results.append(result)

    summary = _summarize(group_results)
    summary["manifest"] = str(args.manifest)
    summary["energy_normalization"] = str(args.energy_normalization)
    if args.reference_jsonl:
        summary["reference_jsonl"] = str(args.reference_jsonl)
    if args.estimates_dir:
        summary["estimates_dir"] = str(args.estimates_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "tse_word_ownership_groups.jsonl", group_results)
    (args.output_dir / "tse_word_ownership_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
