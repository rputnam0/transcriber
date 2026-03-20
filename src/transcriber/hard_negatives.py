from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .prep_artifacts import load_candidate_pool
from .segment_classifier import ClassifierDataset


NON_SPEAKER_LABELS = {"", "unknown", "<unmatched>"}


def normalize_pair(left: str, right: str) -> Tuple[str, str]:
    ordered = sorted([str(left).strip(), str(right).strip()])
    return ordered[0], ordered[1]


DEFAULT_PAIR_WEIGHTS = {
    normalize_pair("Cyrus Schwert", "Cletus Cobbington"): 1.0,
    normalize_pair("Cyrus Schwert", "Leopold Magnus"): 0.9,
    normalize_pair("Kaladen Shash", "Leopold Magnus"): 0.8,
    normalize_pair("Kaladen Shash", "Dungeon Master"): 0.8,
    normalize_pair("David Tanglethorn", "Cletus Cobbington"): 0.7,
    normalize_pair("David Tanglethorn", "Leopold Magnus"): 0.7,
}


def _is_valid_speaker_label(label: object) -> bool:
    normalized = str(label or "").strip()
    return normalized.lower() not in NON_SPEAKER_LABELS


def _collect_eval_pair_statistics(
    eval_summaries: Sequence[Mapping[str, object]],
) -> Tuple[Counter[Tuple[str, str]], Counter[str]]:
    pair_counts: Counter[Tuple[str, str]] = Counter()
    poor_speaker_weights: Counter[str] = Counter()

    for summary in eval_summaries:
        for result in list(summary.get("results") or []):
            metrics = dict(result.get("metrics") or {})
            confusion = dict(metrics.get("confusion") or {})
            for actual, predicted_counts in confusion.items():
                if not _is_valid_speaker_label(actual):
                    continue
                for predicted, value in dict(predicted_counts).items():
                    if not _is_valid_speaker_label(predicted):
                        continue
                    if str(actual).strip() == str(predicted).strip():
                        continue
                    pair_counts[normalize_pair(str(actual), str(predicted))] += int(value or 0)

            for speaker, stats in dict(metrics.get("per_speaker_accuracy") or {}).items():
                if not _is_valid_speaker_label(speaker):
                    continue
                total = float(dict(stats).get("total") or 0.0)
                accuracy = float(dict(stats).get("accuracy") or 0.0)
                if total <= 0.0:
                    continue
                poor_speaker_weights[str(speaker).strip()] += max(0.0, 1.0 - accuracy) * total

    return pair_counts, poor_speaker_weights


def discover_confusion_pairs(
    eval_summaries: Sequence[Mapping[str, object]],
    *,
    seed_pairs: Optional[Sequence[Sequence[str]]] = None,
    top_k: int = 5,
) -> List[Tuple[str, str]]:
    discovered: List[Tuple[str, str]] = []

    for pair in seed_pairs or []:
        if len(pair) != 2:
            continue
        if not _is_valid_speaker_label(pair[0]) or not _is_valid_speaker_label(pair[1]):
            continue
        normalized = normalize_pair(str(pair[0]), str(pair[1]))
        if normalized[0] and normalized[1] and normalized not in discovered:
            discovered.append(normalized)

    counts, _poor_speaker_weights = _collect_eval_pair_statistics(eval_summaries)

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
            if not _is_valid_speaker_label(speaker) or not _is_valid_speaker_label(second_speaker):
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
                        "tracked_pair": list(pair),
                        "source_session": str(record.get("session") or "unknown"),
                        "window_index": int(record.get("window_index") or 0),
                        "dominant_share": dominant_share,
                        "active_speakers": int(record.get("active_speakers") or 0),
                        "top1_power": float(record.get("top1_power") or 0.0),
                        "top2_power": float(record.get("top2_power") or 0.0),
                        "score_margin": None,
                        "selection_reason": f"mixed_rejection_{rejection}",
                        "source_kind": "mixed_candidate_pool",
                        "style_profile": str(record.get("style_profile") or "generic"),
                        "style_score": float(record.get("style_score") or 0.0),
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


def _is_style_candidate(
    metadata: Mapping[str, object],
    *,
    style_profile_name: str,
    style_score_threshold: float,
) -> bool:
    profile = str(metadata.get("style_profile") or "")
    score = float(metadata.get("style_score") or 0.0)
    if profile == style_profile_name:
        return True
    if style_profile_name == "session61_like":
        return score >= float(style_score_threshold)
    return False


def _source_priority(
    metadata: Mapping[str, object],
    *,
    style_profile_name: str,
    style_score_threshold: float,
) -> int:
    source_kind = str(metadata.get("source_kind") or "")
    if _is_style_candidate(
        metadata,
        style_profile_name=style_profile_name,
        style_score_threshold=style_score_threshold,
    ):
        return 0
    if source_kind == "eval":
        return 1
    return 2


def _style_threshold_bucket(
    metadata: Mapping[str, object],
    *,
    style_profile_name: str,
    style_score_threshold: float,
) -> str:
    if _is_style_candidate(
        metadata,
        style_profile_name=style_profile_name,
        style_score_threshold=style_score_threshold,
    ):
        return "at_or_above_threshold"
    return "below_threshold"


def _weighted_candidate_sort_key(
    item: Tuple[Dict[str, object], np.ndarray],
    *,
    pair_weights: Mapping[Tuple[str, str], float],
    style_profile_name: str,
    style_score_threshold: float,
) -> Tuple[int, float, float, float, float]:
    metadata = item[0]
    pair = normalize_pair(str(metadata["speaker"]), str(metadata["confusion_partner"]))
    style_score = float(metadata.get("style_score") or 0.0)
    pair_weight = float(pair_weights.get(pair, 0.5))
    margin_value, dominant_share, top2_power = _candidate_sort_key(item)
    return (
        _source_priority(
            metadata,
            style_profile_name=style_profile_name,
            style_score_threshold=style_score_threshold,
        ),
        -pair_weight,
        -style_score,
        margin_value,
        dominant_share + top2_power,
    )


def _derive_eval_pair_weights(
    eval_summaries: Sequence[Mapping[str, object]],
    *,
    seed_pairs: Sequence[Tuple[str, str]],
    pair_weights: Optional[Mapping[Tuple[str, str], float]] = None,
) -> Dict[Tuple[str, str], float]:
    counts, poor_speaker_weights = _collect_eval_pair_statistics(eval_summaries)
    combined = {
        **DEFAULT_PAIR_WEIGHTS,
        **{
            normalize_pair(left, right): float(weight)
            for (left, right), weight in dict(pair_weights or {}).items()
        },
    }

    if not counts and not poor_speaker_weights:
        for pair in seed_pairs:
            combined.setdefault(normalize_pair(pair[0], pair[1]), 0.5)
        return combined

    max_count = max(counts.values(), default=0)
    max_poor_weight = max(poor_speaker_weights.values(), default=0.0)
    for pair in {normalize_pair(left, right) for left, right in seed_pairs} | set(counts):
        left, right = pair
        confusion_weight = (
            0.5 + 0.5 * (float(counts[pair]) / float(max_count)) if max_count > 0 else 0.5
        )
        poor_weight = 0.0
        if max_poor_weight > 0.0:
            poor_weight = 0.25 * (
                (float(poor_speaker_weights[left]) + float(poor_speaker_weights[right]))
                / (2.0 * float(max_poor_weight))
            )
        combined[pair] = max(float(combined.get(pair, 0.5)), confusion_weight + poor_weight)

    return combined


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
    pair_caps: Optional[Mapping[Tuple[str, str], int]] = None,
    per_speaker_cap: Optional[int] = None,
    max_fraction: float = 0.20,
    pair_weights: Optional[Mapping[Tuple[str, str], float]] = None,
    style_profile_name: str = "session61_like",
    style_score_threshold: float = 0.70,
    min_style_samples_per_pair: int = 40,
) -> Tuple[Optional[ClassifierDataset], List[Dict[str, object]], Dict[str, object]]:
    explicit_pairs: set[Tuple[str, str]] = set()
    for pair in seed_pairs or []:
        if len(pair) != 2:
            continue
        explicit_pairs.add(normalize_pair(str(pair[0]), str(pair[1])))
    for left, right in dict(pair_caps or {}).keys():
        explicit_pairs.add(normalize_pair(str(left), str(right)))
    for left, right in dict(pair_weights or {}).keys():
        explicit_pairs.add(normalize_pair(str(left), str(right)))

    tracked_pairs = discover_confusion_pairs(
        eval_summaries,
        seed_pairs=sorted(explicit_pairs),
        top_k=top_confusion_pairs,
    )
    raw_candidates = _load_mixed_candidates(
        candidate_pool_dirs=candidate_pool_dirs,
        tracked_pairs=tracked_pairs,
        min_dominant_share=min_mixed_dominant_share,
    )

    effective_pair_weights = _derive_eval_pair_weights(
        eval_summaries,
        seed_pairs=tracked_pairs,
        pair_weights=pair_weights,
    )
    effective_pair_caps = {
        normalize_pair(left, right): max(int(cap), 0)
        for (left, right), cap in dict(pair_caps or {}).items()
    }
    grouped: Dict[
        Tuple[str, str], Dict[str, List[Tuple[Dict[str, object], np.ndarray]]]
    ] = defaultdict(lambda: defaultdict(list))
    for item in raw_candidates:
        metadata = item[0]
        pair = normalize_pair(str(metadata["speaker"]), str(metadata["confusion_partner"]))
        grouped[pair][str(metadata["speaker"])].append(item)

    capped: List[Tuple[Dict[str, object], np.ndarray]] = []
    style_target = max(int(min_style_samples_per_pair), 0)
    pair_cap_value = max(int(per_pair_cap), 0)
    for key in sorted(
        grouped,
        key=lambda item: (-float(effective_pair_weights.get(item, 0.5)), item[0], item[1]),
    ):
        current_pair_cap = int(effective_pair_caps.get(key, pair_cap_value))
        pair_groups = grouped[key]
        for speaker in sorted(pair_groups):
            ordered = sorted(
                pair_groups[speaker],
                key=lambda item: _weighted_candidate_sort_key(
                    item,
                    pair_weights=effective_pair_weights,
                    style_profile_name=style_profile_name,
                    style_score_threshold=style_score_threshold,
                ),
            )
            style_selected = [
                item
                for item in ordered
                if _is_style_candidate(
                    item[0],
                    style_profile_name=style_profile_name,
                    style_score_threshold=style_score_threshold,
                )
            ][:style_target]
            selected_ids = {id(item[0]) for item in style_selected}
            capped.extend(style_selected)
            remaining_slots = max(current_pair_cap - len(style_selected), 0)
            if remaining_slots <= 0:
                continue
            filler = [item for item in ordered if id(item[0]) not in selected_ids]
            capped.extend(filler[:remaining_slots])

    speaker_cap_value = (
        max(int(per_speaker_cap), 0) if per_speaker_cap is not None else None
    )
    if speaker_cap_value:
        by_speaker: Dict[str, List[Tuple[Dict[str, object], np.ndarray]]] = defaultdict(list)
        for item in capped:
            by_speaker[str(item[0]["speaker"])].append(item)
        speaker_capped: List[Tuple[Dict[str, object], np.ndarray]] = []
        for speaker in sorted(by_speaker):
            speaker_capped.extend(
                sorted(
                    by_speaker[speaker],
                    key=lambda item: _weighted_candidate_sort_key(
                        item,
                        pair_weights=effective_pair_weights,
                        style_profile_name=style_profile_name,
                        style_score_threshold=style_score_threshold,
                    ),
                )[:speaker_cap_value]
            )
        capped = speaker_capped

    global_cap = int(max(float(base_dataset_samples) * float(max_fraction), 0.0))
    if global_cap > 0 and len(capped) > global_cap:
        capped = sorted(
            capped,
            key=lambda item: _weighted_candidate_sort_key(
                item,
                pair_weights=effective_pair_weights,
                style_profile_name=style_profile_name,
                style_score_threshold=style_score_threshold,
            ),
        )[:global_cap]

    if not capped:
        summary = {
            "tracked_pairs": tracked_pairs,
            "selected": 0,
            "by_style_threshold": {},
            "by_pair_style_threshold": {},
            "heuristics": {
                "uses_eval_confusions": bool(eval_summaries),
                "uses_eval_examples": False,
            },
            "rejections": {"empty": True},
            "limits": {
                "per_pair_cap": int(per_pair_cap),
                "pair_caps": {f"{left}::{right}": cap for (left, right), cap in effective_pair_caps.items()},
                "per_speaker_cap": speaker_cap_value,
                "max_fraction": float(max_fraction),
                "global_cap": global_cap,
                "style_profile_name": str(style_profile_name),
                "style_score_threshold": float(style_score_threshold),
                "min_style_samples_per_pair": style_target,
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
    by_style_profile: Counter[str] = Counter()
    by_style_threshold: Counter[str] = Counter()
    by_pair_style_threshold: Dict[str, Counter[str]] = defaultdict(Counter)

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
        by_style_profile[str(metadata.get("style_profile") or "generic")] += 1
        bucket = _style_threshold_bucket(
            metadata,
            style_profile_name=style_profile_name,
            style_score_threshold=style_score_threshold,
        )
        by_style_threshold[bucket] += 1
        pair_key = f"{metadata['speaker']}::{metadata['confusion_partner']}"
        by_pair_style_threshold[pair_key][bucket] += 1

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
        "by_style_profile": dict(by_style_profile),
        "by_style_threshold": dict(by_style_threshold),
        "by_pair_style_threshold": {
            pair_key: dict(counter)
            for pair_key, counter in sorted(by_pair_style_threshold.items(), key=lambda item: item[0])
        },
        "heuristics": {
            "uses_eval_confusions": bool(eval_summaries),
            "uses_eval_examples": False,
        },
        "limits": {
            "per_pair_cap": int(per_pair_cap),
            "pair_caps": {f"{left}::{right}": cap for (left, right), cap in effective_pair_caps.items()},
            "per_speaker_cap": speaker_cap_value,
            "max_fraction": float(max_fraction),
            "global_cap": global_cap,
            "style_profile_name": str(style_profile_name),
            "style_score_threshold": float(style_score_threshold),
            "min_style_samples_per_pair": style_target,
        },
    }
    return dataset, metadata_records, summary
