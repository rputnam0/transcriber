from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0), int(sample_rate)


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _load_reference_groups(path: Path) -> dict[tuple[str, float, float], list[dict]]:
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in _read_jsonl(path):
        key = _group_key(row)
        groups[key] = [dict(word) for word in row.get("words") or []]
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


def _paint_mask(
    words: Sequence[Mapping[str, object]],
    *,
    sample_rate: int,
    samples: int,
) -> np.ndarray:
    mask = np.zeros(samples, dtype=bool)
    for word in words:
        start = max(0, int(math.floor(float(word.get("start") or 0.0) * sample_rate)))
        end = min(samples, int(math.ceil(float(word.get("end") or 0.0) * sample_rate)))
        if end > start:
            mask[start:end] = True
    return mask


def _mean_energy(wave: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    if not mask.any():
        return 0.0, 0
    chunk = wave[mask].astype(np.float64)
    return float(np.mean(chunk * chunk)) if chunk.size else 0.0, int(chunk.size)


def _db_ratio(numerator: float, denominator: float, *, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(numerator, eps) / max(denominator, eps))


def _score_row(
    row: Mapping[str, object],
    *,
    reference_words: Sequence[Mapping[str, object]],
    estimates_dir: Path | None,
) -> dict:
    speaker = str(row.get("speaker_id") or "")
    estimate_path = _estimate_path(row, estimates_dir)
    if not estimate_path or not estimate_path.exists():
        return {
            "row_id": row.get("row_id"),
            "speaker_id": speaker,
            "error": "missing_estimate",
        }
    wave, sample_rate = _load_mono(estimate_path)
    target_words = [word for word in reference_words if str(word.get("speaker") or "") == speaker]
    non_owner_words = [
        word for word in reference_words if str(word.get("speaker") or "") != speaker
    ]
    target_mask = _paint_mask(target_words, sample_rate=sample_rate, samples=wave.shape[0])
    non_owner_mask = _paint_mask(non_owner_words, sample_rate=sample_rate, samples=wave.shape[0])
    speech_mask = target_mask | non_owner_mask
    silence_mask = ~speech_mask
    target_energy, target_samples = _mean_energy(wave, target_mask)
    non_owner_energy, non_owner_samples = _mean_energy(wave, non_owner_mask)
    silence_energy, silence_samples = _mean_energy(wave, silence_mask)
    return {
        "row_id": row.get("row_id"),
        "speaker_id": speaker,
        "session": row.get("session"),
        "window_start": row.get("window_start"),
        "window_end": row.get("window_end"),
        "target_words": len(target_words),
        "non_owner_words": len(non_owner_words),
        "target_samples": target_samples,
        "non_owner_samples": non_owner_samples,
        "silence_samples": silence_samples,
        "target_energy": target_energy,
        "non_owner_energy": non_owner_energy,
        "silence_energy": silence_energy,
        "target_to_non_owner_db": _db_ratio(target_energy, non_owner_energy),
        "target_to_silence_db": _db_ratio(target_energy, silence_energy),
        "estimate_path": str(estimate_path),
    }


def _weighted_energy(rows: Sequence[Mapping[str, object]], key: str, sample_key: str) -> float:
    numerator = 0.0
    denominator = 0
    for row in rows:
        samples = int(row.get(sample_key) or 0)
        numerator += float(row.get(key) or 0.0) * samples
        denominator += samples
    return numerator / denominator if denominator else 0.0


def _summarize(rows: Sequence[Mapping[str, object]]) -> dict:
    valid = [row for row in rows if not row.get("error")]
    target_energy = _weighted_energy(valid, "target_energy", "target_samples")
    non_owner_energy = _weighted_energy(valid, "non_owner_energy", "non_owner_samples")
    silence_energy = _weighted_energy(valid, "silence_energy", "silence_samples")
    by_speaker: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in valid:
        by_speaker[str(row.get("speaker_id") or "")].append(row)
    return {
        "row_count": len(rows),
        "valid_rows": len(valid),
        "target_energy": target_energy,
        "non_owner_energy": non_owner_energy,
        "silence_energy": silence_energy,
        "target_to_non_owner_db": _db_ratio(target_energy, non_owner_energy),
        "target_to_silence_db": _db_ratio(target_energy, silence_energy),
        "mean_row_target_to_non_owner_db": (
            float(np.mean([float(row.get("target_to_non_owner_db") or 0.0) for row in valid]))
            if valid
            else 0.0
        ),
        "by_speaker": {
            speaker: _summarize_speaker(speaker_rows)
            for speaker, speaker_rows in sorted(by_speaker.items())
        },
    }


def _summarize_speaker(rows: Sequence[Mapping[str, object]]) -> dict:
    target_energy = _weighted_energy(rows, "target_energy", "target_samples")
    non_owner_energy = _weighted_energy(rows, "non_owner_energy", "non_owner_samples")
    silence_energy = _weighted_energy(rows, "silence_energy", "silence_samples")
    return {
        "rows": len(rows),
        "target_to_non_owner_db": _db_ratio(target_energy, non_owner_energy),
        "target_to_silence_db": _db_ratio(target_energy, silence_energy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score target/non-owner energy leakage using forced word references."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimates-dir", type=Path)
    args = parser.parse_args()

    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in _read_jsonl(args.manifest):
        if row.get("materialized"):
            grouped[_group_key(row)].append(row)

    rows = []
    for key, group_rows in sorted(grouped.items()):
        reference_words = reference_groups.get(key) or []
        for row in group_rows:
            rows.append(
                _score_row(
                    row,
                    reference_words=reference_words,
                    estimates_dir=args.estimates_dir,
                )
            )

    summary = _summarize(rows)
    summary["manifest"] = str(args.manifest)
    summary["reference_jsonl"] = str(args.reference_jsonl)
    if args.estimates_dir:
        summary["estimates_dir"] = str(args.estimates_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "forced_mask_energy_rows.jsonl", rows)
    (args.output_dir / "forced_mask_energy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
