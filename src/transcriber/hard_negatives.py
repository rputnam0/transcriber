from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .prep_artifacts import load_candidate_pool
from .segment_classifier import ClassifierDataset


def normalize_pair(left: str, right: str) -> Tuple[str, str]:
    ordered = sorted([str(left).strip(), str(right).strip()])
    return ordered[0], ordered[1]


def discover_confusion_pairs(
    eval_summaries: Sequence[Mapping[str, object]],
    *,
    seed_pairs: Optional[Sequence[Sequence[str]]] = None,
    top_k: int = 5,
) -> List[Tuple[str, str]]:
    counts: Counter[Tuple[str, str]] = Counter()
    discovered: List[Tuple[str, str]] = []

    for pair in seed_pairs or []:
        if len(pair) != 2:
            continue
        normalized = normalize_pair(str(pair[0]), str(pair[1]))
        if normalized[0] and normalized[1] and normalized not in discovered:
            discovered.append(normalized)

    for summary in eval_summaries:
        for result in list(summary.get("results") or []):
            confusion = dict(result.get("metrics", {}).get("confusion") or {})
            for actual, predicted_counts in confusion.items():
                for predicted, value in dict(predicted_counts).items():
                    if actual == predicted or predicted == "<unmatched>":
                        continue
                    counts[normalize_pair(str(actual), str(predicted))] += int(value or 0)

    for pair, _count in counts.most_common(max(int(top_k), 0)):
        if pair not in discovered:
            discovered.append(pair)
    return discovered


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_eval_candidates(
    *,
    eval_summaries: Sequence[Mapping[str, object]],
    tracked_pairs: Sequence[Tuple[str, str]],
    max_margin: float,
) -> List[Tuple[Dict[str, object], np.ndarray]]:
    tracked = {normalize_pair(left, right) for left, right in tracked_pairs}
    selected: List[Tuple[Dict[str, object], np.ndarray]] = []

    for summary in eval_summaries:
        session_name = Path(str(summary.get("session_zip") or "")).stem or "unknown"
        for window_index, result in enumerate(list(summary.get("results") or []), start=1):
            predicted_segments = _load_jsonl(Path(str(result["predicted_jsonl"])))
            purity = json.loads(Path(str(result["diarization_purity_path"])).read_text(encoding="utf-8"))
            purity_records = list(purity.get("records") or [])
            payload = np.load(Path(str(result["segment_embedding_path"])), allow_pickle=False)
            segment_indices = np.asarray(payload["segment_indices"], dtype=np.int32)
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)

            for row_index, segment_index in enumerate(segment_indices.tolist()):
                if row_index >= embeddings.shape[0] or segment_index >= len(predicted_segments):
                    continue
                predicted = predicted_segments[int(segment_index)]
                if int(segment_index) >= len(purity_records):
                    continue
                purity_record = dict(purity_records[int(segment_index)])
                top_candidates = list(predicted.get("speaker_match_candidates") or [])
                if len(top_candidates) < 2:
                    continue
                first = str(top_candidates[0].get("speaker") or "").strip()
                second = str(top_candidates[1].get("speaker") or "").strip()
                if not first or not second:
                    continue
                pair = normalize_pair(first, second)
                if pair not in tracked:
                    continue
                score_margin = predicted.get("speaker_match", {}).get("margin")
                if score_margin is None:
                    score_margin = float(top_candidates[0].get("score") or 0.0) - float(
                        top_candidates[1].get("score") or 0.0
                    )
                score_margin = float(score_margin)
                if score_margin > float(max_margin):
                    continue
                actual_speaker = str(purity_record.get("speaker") or "").strip()
                actual_second = str(purity_record.get("second_speaker") or "").strip()
                if not actual_speaker or normalize_pair(actual_speaker, actual_second) != pair:
                    continue
                confusion_partner = pair[0] if actual_speaker == pair[1] else pair[1]
                selected.append(
                    (
                        {
                            "speaker": actual_speaker,
                            "confusion_partner": confusion_partner,
                            "source_session": str(purity_record.get("session") or session_name),
                            "window_index": window_index,
                            "dominant_share": float(purity_record.get("dominant_share") or 0.0),
                            "active_speakers": int(purity_record.get("active_speakers") or 0),
                            "top1_power": float(purity_record.get("top1_power") or 0.0),
                            "top2_power": float(purity_record.get("top2_power") or 0.0),
                            "score_margin": score_margin,
                            "selection_reason": "eval_top2_margin",
                            "source_kind": "eval",
                            "start": float(purity_record.get("start") or 0.0),
                            "end": float(purity_record.get("end") or 0.0),
                        },
                        np.asarray(embeddings[row_index], dtype=np.float32),
                    )
                )
    return selected


def _load_mixed_candidates(
    *,
    candidate_pool_dirs: Sequence[Path],
    tracked_pairs: Sequence[Tuple[str, str]],
    min_dominant_share: float,
) -> List[Tuple[Dict[str, object], np.ndarray]]:
    tracked = {normalize_pair(left, right) for left, right in tracked_pairs}
    selected: List[Tuple[Dict[str, object], np.ndarray]] = []

    for candidate_dir in candidate_pool_dirs:
        records, embeddings = load_candidate_pool(candidate_dir)
        for record in records:
            raw_index = record.get("embedding_index")
            index = int(raw_index) if raw_index is not None else -1
            if index < 0 or index >= embeddings.shape[0]:
                continue
            speaker = str(record.get("speaker") or "").strip()
            second_speaker = str(record.get("second_speaker") or "").strip()
            rejection = str(record.get("rejection") or "").strip()
            if not speaker or not second_speaker:
                continue
            pair = normalize_pair(speaker, second_speaker)
            if pair not in tracked:
                continue
            if rejection not in {"low_share", "not_dominant"}:
                continue
            dominant_share = float(record.get("dominant_share") or 0.0)
            if dominant_share < float(min_dominant_share):
                continue
            confusion_partner = pair[0] if speaker == pair[1] else pair[1]
            selected.append(
                (
                    {
                        "speaker": speaker,
                        "confusion_partner": confusion_partner,
                        "source_session": str(record.get("session") or "unknown"),
                        "window_index": int(record.get("window_index") or 0),
                        "dominant_share": dominant_share,
                        "active_speakers": int(record.get("active_speakers") or 0),
                        "top1_power": float(record.get("top1_power") or 0.0),
                        "top2_power": float(record.get("top2_power") or 0.0),
                        "score_margin": None,
                        "selection_reason": f"mixed_rejection_{rejection}",
                        "source_kind": "mixed_candidate_pool",
                        "start": float(record.get("start") or 0.0),
                        "end": float(record.get("end") or 0.0),
                    },
                    np.asarray(embeddings[index], dtype=np.float32),
                )
            )
    return selected


def _candidate_sort_key(item: Tuple[Dict[str, object], np.ndarray]) -> Tuple[float, float, float]:
    metadata = item[0]
    score_margin = metadata.get("score_margin")
    margin_value = float(score_margin) if score_margin is not None else 1.0
    dominant_share = float(metadata.get("dominant_share") or 0.0)
    top2_power = float(metadata.get("top2_power") or 0.0)
    return (margin_value, dominant_share, -top2_power)


def build_hard_negative_dataset(
    *,
    eval_summaries: Sequence[Mapping[str, object]],
    candidate_pool_dirs: Sequence[Path],
    base_dataset_samples: int,
    seed_pairs: Optional[Sequence[Sequence[str]]] = None,
    top_confusion_pairs: int = 5,
    max_eval_margin: float = 0.12,
    min_mixed_dominant_share: float = 0.55,
    per_pair_cap: int = 75,
    max_fraction: float = 0.20,
) -> Tuple[Optional[ClassifierDataset], List[Dict[str, object]], Dict[str, object]]:
    tracked_pairs = discover_confusion_pairs(
        eval_summaries,
        seed_pairs=seed_pairs,
        top_k=top_confusion_pairs,
    )
    raw_candidates = _load_eval_candidates(
        eval_summaries=eval_summaries,
        tracked_pairs=tracked_pairs,
        max_margin=max_eval_margin,
    )
    raw_candidates.extend(
        _load_mixed_candidates(
            candidate_pool_dirs=candidate_pool_dirs,
            tracked_pairs=tracked_pairs,
            min_dominant_share=min_mixed_dominant_share,
        )
    )

    grouped: Dict[Tuple[str, str, str], List[Tuple[Dict[str, object], np.ndarray]]] = defaultdict(list)
    for item in raw_candidates:
        metadata = item[0]
        pair = normalize_pair(str(metadata["speaker"]), str(metadata["confusion_partner"]))
        grouped[(str(metadata["speaker"]), pair[0], pair[1])].append(item)

    capped: List[Tuple[Dict[str, object], np.ndarray]] = []
    for key in sorted(grouped):
        capped.extend(sorted(grouped[key], key=_candidate_sort_key)[: max(int(per_pair_cap), 0)])

    global_cap = int(max(float(base_dataset_samples) * float(max_fraction), 0.0))
    if global_cap > 0 and len(capped) > global_cap:
        capped = sorted(capped, key=_candidate_sort_key)[:global_cap]

    if not capped:
        summary = {
            "tracked_pairs": tracked_pairs,
            "selected": 0,
            "rejections": {"empty": True},
            "limits": {
                "per_pair_cap": int(per_pair_cap),
                "max_fraction": float(max_fraction),
                "global_cap": global_cap,
            },
        }
        return None, [], summary

    metadata_records: List[Dict[str, object]] = []
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    sessions: List[str] = []
    durations: List[float] = []
    dominant_shares: List[float] = []
    top1_powers: List[float] = []
    top2_powers: List[float] = []
    active_speakers: List[int] = []
    by_pair: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    for metadata, embedding in capped:
        metadata_records.append(dict(metadata))
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        labels.append(str(metadata["speaker"]))
        sessions.append(str(metadata["source_session"]))
        durations.append(max(float(metadata["end"]) - float(metadata["start"]), 0.0))
        dominant_shares.append(float(metadata["dominant_share"]))
        top1_powers.append(float(metadata["top1_power"]))
        top2_powers.append(float(metadata["top2_power"]))
        active_speakers.append(int(metadata["active_speakers"]))
        by_pair[f"{metadata['speaker']}::{metadata['confusion_partner']}"] += 1
        by_source[str(metadata["source_kind"])] += 1

    dataset = ClassifierDataset(
        embeddings=np.vstack(embeddings).astype(np.float32),
        labels=labels,
        domains=["hard_negative"] * len(labels),
        sources=["hard_negative"] * len(labels),
        sessions=sessions,
        durations=np.asarray(durations, dtype=np.float32),
        dominant_shares=np.asarray(dominant_shares, dtype=np.float32),
        top1_powers=np.asarray(top1_powers, dtype=np.float32),
        top2_powers=np.asarray(top2_powers, dtype=np.float32),
        active_speakers=np.asarray(active_speakers, dtype=np.int32),
    )
    summary = {
        "tracked_pairs": tracked_pairs,
        "selected": dataset.samples,
        "by_pair": dict(by_pair),
        "by_source": dict(by_source),
        "limits": {
            "per_pair_cap": int(per_pair_cap),
            "max_fraction": float(max_fraction),
            "global_cap": global_cap,
        },
    }
    return dataset, metadata_records, summary
