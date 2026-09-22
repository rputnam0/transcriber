from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

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
        wave = np.mean(wave, axis=1, dtype=np.float32)
    return np.nan_to_num(wave.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), int(sample_rate)


def _match_length(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = max(reference.shape[0], estimate.shape[0])
    if size <= 0:
        return reference[:0], estimate[:0]
    if reference.shape[0] < size:
        reference = np.pad(reference, (0, size - reference.shape[0]))
    if estimate.shape[0] < size:
        estimate = np.pad(estimate, (0, size - estimate.shape[0]))
    return reference[:size], estimate[:size]


def _rms(wave: np.ndarray) -> float:
    if wave.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))


def _si_sdr(reference: np.ndarray, estimate: np.ndarray, *, eps: float = 1e-8) -> float:
    reference, estimate = _match_length(reference, estimate)
    if reference.size == 0 or estimate.size == 0:
        return float("-inf")
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    ref_energy = float(np.sum(reference * reference)) + eps
    projection = reference * (float(np.sum(estimate * reference)) / ref_energy)
    noise = estimate - projection
    ratio = (float(np.sum(projection * projection)) + eps) / (float(np.sum(noise * noise)) + eps)
    return 10.0 * math.log10(max(ratio, eps))


def _safe_float(value: float) -> float:
    if math.isfinite(value):
        return round(float(value), 6)
    return float(value)


def _estimate_path(row: Mapping[str, object], estimates_dir: Path | None) -> Path | None:
    if estimates_dir is None:
        return None
    row_id = str(row.get("row_id") or "")
    candidates = [
        estimates_dir / f"{row_id}.wav",
        estimates_dir / row_id / "estimate.wav",
        estimates_dir / row_id / "target.wav",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _score_row(row: Mapping[str, object], *, estimates_dir: Path | None) -> dict | None:
    materialized = dict(row.get("materialized") or {})
    mixture_path = materialized.get("mixture_path")
    target_path = materialized.get("target_source_path")
    if not mixture_path or not target_path:
        return None
    mixture_path = Path(str(mixture_path))
    target_path = Path(str(target_path))
    if not mixture_path.exists() or not target_path.exists():
        return {
            "row_id": row.get("row_id"),
            "split_id": row.get("split_id"),
            "speaker_id": row.get("speaker_id"),
            "error": "missing_materialized_audio",
            "mixture_path": str(mixture_path),
            "target_source_path": str(target_path),
        }
    target, target_sr = _load_mono(target_path)
    mixture, mixture_sr = _load_mono(mixture_path)
    if target_sr != mixture_sr:
        return {
            "row_id": row.get("row_id"),
            "split_id": row.get("split_id"),
            "speaker_id": row.get("speaker_id"),
            "error": "sample_rate_mismatch",
            "mixture_sample_rate": mixture_sr,
            "target_sample_rate": target_sr,
        }
    mixture_sisdr = _si_sdr(target, mixture)
    result = {
        "row_id": row.get("row_id"),
        "split_id": row.get("split_id"),
        "session": row.get("session"),
        "speaker_id": row.get("speaker_id"),
        "target_word_count": int(row.get("target_word_count") or 0),
        "total_word_count": int(row.get("total_word_count") or 0),
        "target_share": float(row.get("target_share") or 0.0),
        "overlap_bucket": row.get("overlap_bucket"),
        "sample_rate": target_sr,
        "duration_seconds": round(target.shape[0] / float(target_sr), 3) if target_sr else 0.0,
        "target_rms": _safe_float(_rms(target)),
        "mixture_rms": _safe_float(_rms(mixture)),
        "mixture_si_sdr": _safe_float(mixture_sisdr),
        "target_self_si_sdr": _safe_float(_si_sdr(target, target)),
        "estimate_path": None,
        "estimate_rms": None,
        "estimate_si_sdr": None,
        "estimate_si_sdri": None,
        "estimate_length_ratio": None,
    }
    estimate_path = _estimate_path(row, estimates_dir)
    if estimate_path is not None:
        estimate, estimate_sr = _load_mono(estimate_path)
        if estimate_sr == target_sr:
            estimate_sisdr = _si_sdr(target, estimate)
            length_ratio = estimate.shape[0] / target.shape[0] if target.shape[0] else 0.0
            result.update(
                {
                    "estimate_path": str(estimate_path),
                    "estimate_rms": _safe_float(_rms(estimate)),
                    "estimate_si_sdr": _safe_float(estimate_sisdr),
                    "estimate_si_sdri": _safe_float(estimate_sisdr - mixture_sisdr),
                    "estimate_length_ratio": _safe_float(length_ratio),
                }
            )
        else:
            result["estimate_error"] = "sample_rate_mismatch"
            result["estimate_sample_rate"] = estimate_sr
    elif estimates_dir is not None:
        result["estimate_error"] = "missing_estimate"
    return result


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _summarize(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    usable = [row for row in rows if not row.get("error")]
    by_split: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    by_speaker: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    by_bucket: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in usable:
        by_split[str(row.get("split_id") or "unknown")].append(row)
        by_speaker[str(row.get("speaker_id") or "unknown")].append(row)
        by_bucket[str(row.get("overlap_bucket") or "unknown")].append(row)

    def aggregate(group: Sequence[Mapping[str, object]]) -> dict:
        estimate_scores = [
            float(row["estimate_si_sdr"]) for row in group if row.get("estimate_si_sdr") is not None
        ]
        estimate_improvements = [
            float(row["estimate_si_sdri"])
            for row in group
            if row.get("estimate_si_sdri") is not None
        ]
        estimate_length_ratios = [
            float(row["estimate_length_ratio"])
            for row in group
            if row.get("estimate_length_ratio") is not None
        ]
        return {
            "rows": len(group),
            "target_words": int(sum(int(row.get("target_word_count") or 0) for row in group)),
            "estimate_count": len(estimate_scores),
            "missing_estimate_count": int(
                sum(1 for row in group if row.get("estimate_error") == "missing_estimate")
            ),
            "estimate_error_count": int(sum(1 for row in group if row.get("estimate_error"))),
            "mean_target_share": _safe_float(
                _mean([float(row.get("target_share") or 0.0) for row in group])
            ),
            "mean_mixture_si_sdr": _safe_float(
                _mean([float(row.get("mixture_si_sdr") or 0.0) for row in group])
            ),
            "mean_estimate_si_sdr": (
                _safe_float(_mean(estimate_scores)) if estimate_scores else None
            ),
            "mean_estimate_si_sdri": (
                _safe_float(_mean(estimate_improvements)) if estimate_improvements else None
            ),
            "mean_estimate_length_ratio": (
                _safe_float(_mean(estimate_length_ratios)) if estimate_length_ratios else None
            ),
        }

    errors = Counter(str(row.get("error") or "unknown") for row in rows if row.get("error"))
    estimate_errors = Counter(
        str(row.get("estimate_error") or "unknown") for row in usable if row.get("estimate_error")
    )
    return {
        "row_count": len(rows),
        "usable_rows": len(usable),
        "errors": dict(sorted(errors.items())),
        "estimate_errors": dict(sorted(estimate_errors.items())),
        "overall": aggregate(usable),
        "by_split": {key: aggregate(value) for key, value in sorted(by_split.items())},
        "by_speaker": {key: aggregate(value) for key, value in sorted(by_speaker.items())},
        "by_overlap_bucket": {key: aggregate(value) for key, value in sorted(by_bucket.items())},
    }


def _write_markdown(summary: Mapping[str, object], path: Path) -> None:
    overall = dict(summary.get("overall") or {})
    lines = [
        "# Speaker TSE Manifest Score",
        "",
        f"- Rows: {summary.get('row_count', 0)}",
        f"- Usable rows: {summary.get('usable_rows', 0)}",
        f"- Mean mixture SI-SDR: {overall.get('mean_mixture_si_sdr')}",
        f"- Mean estimate SI-SDR: {overall.get('mean_estimate_si_sdr')}",
        f"- Mean estimate SI-SDRi: {overall.get('mean_estimate_si_sdri')}",
        "",
        "## By Split",
        "",
        "| split | rows | target words | target share | mixture SI-SDR | estimate SI-SDRi |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in dict(summary.get("by_split") or {}).items():
        item = dict(metrics)
        lines.append(
            "| {split} | {rows} | {words} | {share} | {mix} | {est} |".format(
                split=split,
                rows=item.get("rows", 0),
                words=item.get("target_words", 0),
                share=item.get("mean_target_share"),
                mix=item.get("mean_mixture_si_sdr"),
                est=item.get("mean_estimate_si_sdri"),
            )
        )
    lines.extend(
        [
            "",
            "## By Speaker",
            "",
            "| speaker | rows | target words | mixture SI-SDR | estimate SI-SDRi |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for speaker, metrics in dict(summary.get("by_speaker") or {}).items():
        item = dict(metrics)
        lines.append(
            "| {speaker} | {rows} | {words} | {mix} | {est} |".format(
                speaker=speaker,
                rows=item.get("rows", 0),
                words=item.get("target_words", 0),
                mix=item.get("mean_mixture_si_sdr"),
                est=item.get("mean_estimate_si_sdri"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score materialized target-speaker extraction manifest rows."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimates-dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        result
        for row in _read_jsonl(args.manifest)
        for result in [_score_row(row, estimates_dir=args.estimates_dir)]
        if result is not None
    ]
    summary = _summarize(rows)
    summary["manifest"] = str(args.manifest)
    if args.estimates_dir:
        summary["estimates_dir"] = str(args.estimates_dir)
    _write_jsonl(args.output_dir / "speaker_tse_scores.jsonl", rows)
    (args.output_dir / "speaker_tse_score_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, args.output_dir / "speaker_tse_score_summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
