from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SessionGraphConfig:
    enabled: bool = False
    candidate_top_k: int = 3
    candidate_floor: float = 0.20
    knn: int = 8
    min_similarity: float = 0.25
    anchor_weight: float = 1.0
    temporal_weight: float = 0.35
    temporal_tau_seconds: float = 0.75
    temporal_max_gap_seconds: float = 2.0
    same_raw_label_weight: float = 0.25
    same_top1_weight: float = 0.10
    alpha: float = 0.85
    max_iters: int = 30
    tolerance: float = 1e-4
    strong_seed_score: float = 0.60
    strong_seed_margin: float = 0.12
    override_min_confidence: float = 0.55
    override_min_margin: float = 0.12
    override_min_delta: float = 0.03
    pair_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DiarizationRepairConfig:
    enabled: bool = False
    merge_same_raw_gap_seconds: float = 0.20
    snap_boundary_seconds: float = 0.20
    max_overlap_trim_seconds: float = 0.30
    split_on_word_gap_seconds: float = 0.35
    max_seed_overlap_seconds: float = 0.15
    min_segment_duration_seconds: float = 0.35


@dataclass(frozen=True)
class SessionSegmentEvidence:
    segment_index: int
    start: float
    end: float
    duration: float
    speaker_raw: str
    overlap_heavy: bool
    was_split: bool
    local_source: Optional[str]
    local_speaker: Optional[str]
    local_score: Optional[float]
    local_margin: Optional[float]
    local_candidates: List[Dict[str, object]]
    local_top2_pair: Optional[str]
    seed_eligible: bool


@dataclass(frozen=True)
class SessionSegmentEmbedding:
    segment_index: int
    raw_label: str
    start: float
    end: float
    embedding: np.ndarray


def _normalized_pair_key(left: object, right: object) -> Optional[str]:
    left_name = str(left or "").strip()
    right_name = str(right or "").strip()
    if not left_name or not right_name or left_name == right_name:
        return None
    ordered = sorted([left_name, right_name])
    return f"{ordered[0]}::{ordered[1]}"


def _normalize_pair_overrides(
    raw_overrides: Mapping[str, object] | None,
) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for raw_key, raw_values in dict(raw_overrides or {}).items():
        if not isinstance(raw_values, Mapping):
            continue
        parts = [part.strip() for part in str(raw_key).split("::", 1)]
        pair_key = (
            _normalized_pair_key(parts[0], parts[1])
            if len(parts) == 2
            else str(raw_key).strip() or None
        )
        if not pair_key:
            continue
        normalized_values: Dict[str, float] = {}
        for name, value in dict(raw_values).items():
            try:
                normalized_values[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        normalized[pair_key] = normalized_values
    return normalized


def _pair_override_value(
    config: SessionGraphConfig,
    pair_key: Optional[str],
    field_name: str,
    default: float,
) -> float:
    if not pair_key:
        return float(default)
    pair_config = dict(config.pair_overrides.get(pair_key) or {})
    value = pair_config.get(field_name)
    if value is None:
        return float(default)
    return float(value)


def _edge_override_value(
    config: SessionGraphConfig,
    pair_keys: Sequence[Optional[str]],
    field_name: str,
    default: float,
) -> float:
    value = float(default)
    for pair_key in pair_keys:
        if not pair_key:
            continue
        pair_config = dict(config.pair_overrides.get(pair_key) or {})
        override = pair_config.get(field_name)
        if override is None:
            continue
        value = max(value, float(override))
    return value


def segment_classifier_thresholds(
    *,
    duration: float,
    base_confidence: float,
    base_margin: float,
) -> Tuple[float, float]:
    min_confidence = float(base_confidence)
    min_margin = float(base_margin)
    if duration < 0.75:
        min_confidence = max(min_confidence, 0.60)
        min_margin = max(min_margin, 0.20)
    elif duration < 1.25:
        min_confidence = max(min_confidence, 0.50)
        min_margin = max(min_margin, 0.14)
    elif duration < 2.0:
        min_confidence = max(min_confidence, 0.35)
        min_margin = max(min_margin, 0.08)
    return min_confidence, min_margin


def label_classifier_thresholds(
    *,
    segment_count: int,
    base_confidence: float,
    base_margin: float,
) -> Tuple[float, float]:
    min_confidence = float(base_confidence)
    min_margin = float(base_margin)
    if segment_count <= 1:
        min_confidence = max(min_confidence, 0.60)
        min_margin = max(min_margin, 0.18)
    elif segment_count <= 2:
        min_confidence = max(min_confidence, 0.45)
        min_margin = max(min_margin, 0.10)
    return min_confidence, min_margin


def set_segment_speaker_label(segment: Dict[str, object], speaker: str | None) -> None:
    segment["speaker"] = speaker
    words = segment.get("words")
    if not isinstance(words, list):
        return
    for word in words:
        if not isinstance(word, dict):
            continue
        if "speaker_raw" not in word and word.get("speaker") is not None:
            word["speaker_raw"] = word.get("speaker")
        word["speaker"] = speaker


def aggregate_segment_label_candidates(
    segment_indices: List[int],
    seg_matches: Dict[int, Dict[str, object]],
    *,
    aggregation: str,
    threshold: float,
    margin_required: float,
    min_segments_per_label: int,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    stats: Dict[str, object] = {
        "segments_embedded": 0,
        "segments_matched": 0,
        "means": {},
        "means_supported": {},
        "vote_counts": {},
        "best_ratio": None,
        "second_ratio": None,
        "margin": None,
        "second_metric": None,
        "score_metric": None,
        "selection": None,
    }

    embedded_total = 0
    candidate_totals: Dict[str, float] = defaultdict(float)
    candidate_support: Dict[str, int] = defaultdict(int)
    vote_counts: Dict[str, int] = defaultdict(int)
    best_candidate_by_speaker: Dict[str, Dict[str, object]] = {}

    for seg_idx in segment_indices:
        seg_info = seg_matches.get(seg_idx) or {}
        candidates = seg_info.get("candidates") or []
        if not candidates:
            continue
        embedded_total += 1
        top_candidate = candidates[0]
        top_speaker = str(top_candidate.get("speaker") or "")
        if top_speaker:
            vote_counts[top_speaker] += 1
        for candidate in candidates:
            speaker = str(candidate.get("speaker") or "")
            if not speaker:
                continue
            score = float(candidate.get("score") or 0.0)
            candidate_totals[speaker] += score
            candidate_support[speaker] += 1
            existing = best_candidate_by_speaker.get(speaker)
            if existing is None or score > float(existing.get("score") or 0.0):
                best_candidate_by_speaker[speaker] = {
                    "segment_index": seg_idx,
                    **candidate,
                }

    stats["segments_embedded"] = embedded_total
    stats["segments_matched"] = sum(
        1 for seg_idx in segment_indices if (seg_matches.get(seg_idx) or {}).get("accepted")
    )

    if embedded_total:
        stats["means"] = {
            speaker: total / embedded_total
            for speaker, total in sorted(
                candidate_totals.items(), key=lambda item: item[1], reverse=True
            )
        }
    stats["means_supported"] = {
        speaker: candidate_totals[speaker] / candidate_support[speaker]
        for speaker in sorted(candidate_support)
        if candidate_support[speaker]
    }
    stats["vote_counts"] = dict(sorted(vote_counts.items(), key=lambda item: item[1], reverse=True))

    if embedded_total < max(1, min_segments_per_label) or not candidate_totals:
        return None, stats

    selection: Optional[Dict[str, object]] = None
    aggregation_name = (aggregation or "mean").lower()
    if aggregation_name == "vote":
        ordered_counts = sorted(
            vote_counts.items(),
            key=lambda item: (item[1], stats["means"].get(item[0], 0.0)),
            reverse=True,
        )
        if ordered_counts:
            best_speaker, best_count = ordered_counts[0]
            second_count = ordered_counts[1][1] if len(ordered_counts) > 1 else 0
            ratio_best = best_count / embedded_total if embedded_total else 0.0
            ratio_second = second_count / embedded_total if embedded_total else 0.0
            margin_value = ratio_best - ratio_second
            stats["best_ratio"] = ratio_best
            stats["second_ratio"] = ratio_second
            stats["margin"] = margin_value
            stats["second_metric"] = ratio_second
            score_metric = float((stats["means"] or {}).get(best_speaker, 0.0))
            stats["score_metric"] = score_metric
            if score_metric >= threshold and margin_value >= margin_required:
                best_candidate = best_candidate_by_speaker.get(best_speaker)
                if best_candidate:
                    selection = {
                        "speaker": best_speaker,
                        "cluster_id": best_candidate.get("cluster_id"),
                        "score": score_metric,
                        "score_max": best_candidate.get("score"),
                        "distance": best_candidate.get("distance"),
                        "margin": margin_value,
                        "second_best": ratio_second,
                        "source": "segment_vote",
                        "segments_count": best_count,
                    }
    else:
        ordered_means = sorted(
            ((speaker, float(score)) for speaker, score in (stats["means"] or {}).items()),
            key=lambda item: item[1],
            reverse=True,
        )
        if ordered_means:
            best_speaker, best_mean = ordered_means[0]
            second_mean = ordered_means[1][1] if len(ordered_means) > 1 else 0.0
            margin_value = best_mean - second_mean
            stats["margin"] = margin_value
            stats["second_metric"] = second_mean
            stats["score_metric"] = best_mean
            if best_mean >= threshold and margin_value >= margin_required:
                best_candidate = best_candidate_by_speaker.get(best_speaker)
                if best_candidate:
                    selection = {
                        "speaker": best_speaker,
                        "cluster_id": best_candidate.get("cluster_id"),
                        "score": best_mean,
                        "score_max": best_candidate.get("score"),
                        "distance": best_candidate.get("distance"),
                        "margin": margin_value,
                        "second_best": second_mean,
                        "source": "segment_mean",
                        "segments_count": candidate_support.get(best_speaker, 0),
                    }

    stats["selection"] = selection
    return selection, stats


def fuse_candidate_scores(
    classifier_candidates: Sequence[Dict[str, object]],
    bank_candidates: Sequence[Dict[str, object]],
    *,
    classifier_weight: float,
    bank_weight: float,
) -> List[Dict[str, object]]:
    fused: Dict[str, Dict[str, object]] = {}
    for candidate in classifier_candidates:
        speaker = str(candidate.get("speaker") or "").strip()
        if not speaker:
            continue
        fused[speaker] = {
            "speaker": speaker,
            "score": float(candidate.get("score") or 0.0) * float(classifier_weight),
            "classifier_score": float(candidate.get("score") or 0.0),
            "bank_score": 0.0,
            "cluster_id": candidate.get("cluster_id"),
            "distance": candidate.get("distance"),
            "source": "fusion",
        }
    for candidate in bank_candidates:
        speaker = str(candidate.get("speaker") or "").strip()
        if not speaker:
            continue
        entry = fused.setdefault(
            speaker,
            {
                "speaker": speaker,
                "score": 0.0,
                "classifier_score": 0.0,
                "bank_score": 0.0,
                "cluster_id": candidate.get("cluster_id"),
                "distance": candidate.get("distance"),
                "source": "fusion",
            },
        )
        bank_score = float(candidate.get("score") or 0.0)
        entry["score"] = float(entry["score"]) + (bank_score * float(bank_weight))
        entry["bank_score"] = bank_score
        if entry.get("cluster_id") is None and candidate.get("cluster_id") is not None:
            entry["cluster_id"] = candidate.get("cluster_id")
        if entry.get("distance") is None and candidate.get("distance") is not None:
            entry["distance"] = candidate.get("distance")
    return sorted(
        fused.values(),
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("classifier_score") or 0.0),
            float(item.get("bank_score") or 0.0),
        ),
        reverse=True,
    )


def _segment_words(segment: Mapping[str, object]) -> List[Dict[str, object]]:
    raw_words = segment.get("words")
    if not isinstance(raw_words, list):
        return []
    words: List[Dict[str, object]] = []
    for word in raw_words:
        if not isinstance(word, dict):
            continue
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            continue
        speaker_raw = word.get("speaker_raw")
        if speaker_raw is None:
            speaker_raw = word.get("speaker")
        text = str(word.get("word") or word.get("text") or "").strip()
        words.append(
            {
                **word,
                "start": float(start),
                "end": float(end),
                "speaker_raw": str(speaker_raw or "").strip() or None,
                "text": text,
            }
        )
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return words


def _segment_text(words: Sequence[Mapping[str, object]]) -> str:
    return " ".join(str(word.get("text") or "").strip() for word in words if str(word.get("text") or "").strip()).strip()


def _merge_repaired_segments(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    gap: float,
) -> Dict[str, object]:
    left_words = list(left.get("words") or [])
    right_words = list(right.get("words") or [])
    merged_words = [*left_words, *right_words]
    merged = {
        **dict(left),
        "start": float(left.get("start") or 0.0),
        "end": float(right.get("end") or left.get("end") or 0.0),
        "text": _segment_text(merged_words),
        "words": merged_words,
        "speaker_raw": left.get("speaker_raw") or right.get("speaker_raw"),
        "speaker_repair_merged": True,
        "speaker_repair_merge_gap_seconds": float(gap),
        "speaker_repair_was_split": bool(left.get("speaker_repair_was_split"))
        or bool(right.get("speaker_repair_was_split")),
        "speaker_repair_overlap_heavy": bool(left.get("speaker_repair_overlap_heavy"))
        or bool(right.get("speaker_repair_overlap_heavy")),
    }
    return merged


def repair_diarization_segments(
    segments: Sequence[dict],
    *,
    config: DiarizationRepairConfig,
) -> Tuple[List[dict], Dict[str, object]]:
    relabeled = copy.deepcopy(list(segments))
    if not config.enabled or not relabeled:
        return relabeled, {
            "enabled": bool(config.enabled),
            "segments_in": len(relabeled),
            "segments_out": len(relabeled),
            "split_segments": 0,
            "merged_segments": 0,
            "trimmed_overlaps": 0,
            "snapped_boundaries": 0,
            "overlap_heavy": 0,
        }

    repaired: List[dict] = []
    split_segments = 0
    snapped_boundaries = 0
    overlap_heavy = 0

    for segment in relabeled:
        words = _segment_words(segment)
        if not words:
            segment["speaker_raw"] = segment.get("speaker_raw") or segment.get("speaker")
            segment["speaker_repair_was_split"] = False
            segment["speaker_repair_overlap_heavy"] = False
            repaired.append(segment)
            continue
        chunks: List[List[Dict[str, object]]] = []
        current: List[Dict[str, object]] = [words[0]]
        internal_overlap_heavy = False
        for word in words[1:]:
            prev = current[-1]
            gap = float(word["start"]) - float(prev["end"])
            raw_changed = str(word.get("speaker_raw") or "") != str(prev.get("speaker_raw") or "")
            if raw_changed or gap >= float(config.split_on_word_gap_seconds):
                if raw_changed or gap > float(config.max_seed_overlap_seconds):
                    internal_overlap_heavy = True
                chunks.append(current)
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(current)
        if len(chunks) > 1:
            split_segments += 1
        for chunk in chunks:
            start = float(chunk[0]["start"])
            end = float(chunk[-1]["end"])
            raw_label = str(chunk[0].get("speaker_raw") or "").strip()
            chunk_segment = copy.deepcopy(segment)
            if abs(float(segment.get("start") or start) - start) <= float(config.snap_boundary_seconds):
                snapped_boundaries += 1
            if abs(float(segment.get("end") or end) - end) <= float(config.snap_boundary_seconds):
                snapped_boundaries += 1
            chunk_segment["start"] = start
            chunk_segment["end"] = end
            chunk_segment["speaker_raw"] = raw_label or segment.get("speaker_raw") or segment.get("speaker")
            chunk_segment["text"] = _segment_text(chunk)
            chunk_segment["words"] = chunk
            chunk_segment["speaker_repair_was_split"] = len(chunks) > 1
            chunk_segment["speaker_repair_overlap_heavy"] = bool(internal_overlap_heavy)
            chunk_segment["speaker_repair_origin"] = "repair_diarization_segments"
            if internal_overlap_heavy:
                overlap_heavy += 1
            repaired.append(chunk_segment)

    repaired.sort(key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)))

    trimmed_overlaps = 0
    for index in range(len(repaired) - 1):
        current = repaired[index]
        following = repaired[index + 1]
        current_end = float(current.get("end") or 0.0)
        next_start = float(following.get("start") or 0.0)
        overlap = current_end - next_start
        if overlap <= 0.0 or overlap > float(config.max_overlap_trim_seconds):
            continue
        midpoint = next_start + (overlap / 2.0)
        current["end"] = midpoint
        following["start"] = midpoint
        trimmed_overlaps += 1
        if overlap > float(config.max_seed_overlap_seconds):
            current["speaker_repair_overlap_heavy"] = True
            following["speaker_repair_overlap_heavy"] = True

    merged: List[dict] = []
    merged_segments = 0
    for segment in repaired:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        previous_raw = str(previous.get("speaker_raw") or "").strip()
        current_raw = str(segment.get("speaker_raw") or "").strip()
        gap = float(segment.get("start") or 0.0) - float(previous.get("end") or 0.0)
        if (
            previous_raw
            and previous_raw == current_raw
            and gap >= 0.0
            and gap <= float(config.merge_same_raw_gap_seconds)
        ):
            merged[-1] = _merge_repaired_segments(previous, segment, gap=gap)
            merged_segments += 1
            continue
        merged.append(segment)

    kept: List[dict] = []
    for segment in merged:
        duration = float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)
        if duration < float(config.min_segment_duration_seconds) and kept:
            previous = kept[-1]
            previous_raw = str(previous.get("speaker_raw") or "").strip()
            current_raw = str(segment.get("speaker_raw") or "").strip()
            if previous_raw and previous_raw == current_raw:
                kept[-1] = _merge_repaired_segments(previous, segment, gap=0.0)
                merged_segments += 1
                continue
        kept.append(segment)

    summary = {
        "enabled": True,
        "segments_in": len(relabeled),
        "segments_out": len(kept),
        "split_segments": int(split_segments),
        "merged_segments": int(merged_segments),
        "trimmed_overlaps": int(trimmed_overlaps),
        "snapped_boundaries": int(snapped_boundaries),
        "overlap_heavy": int(
            sum(1 for item in kept if bool(item.get("speaker_repair_overlap_heavy")))
        ),
    }
    return kept, summary


def _graph_config_from_speaker_bank_config(config: object) -> SessionGraphConfig:
    return SessionGraphConfig(
        enabled=bool(getattr(config, "session_graph_enabled", False)),
        candidate_top_k=max(int(getattr(config, "session_graph_candidate_top_k", 3)), 1),
        candidate_floor=float(getattr(config, "session_graph_candidate_floor", 0.20)),
        knn=max(int(getattr(config, "session_graph_knn", 8)), 1),
        min_similarity=float(getattr(config, "session_graph_min_similarity", 0.25)),
        anchor_weight=float(getattr(config, "session_graph_anchor_weight", 1.0)),
        temporal_weight=float(getattr(config, "session_graph_temporal_weight", 0.35)),
        temporal_tau_seconds=float(getattr(config, "session_graph_temporal_tau_seconds", 0.75)),
        temporal_max_gap_seconds=float(
            getattr(config, "session_graph_temporal_max_gap_seconds", 2.0)
        ),
        same_raw_label_weight=float(
            getattr(config, "session_graph_same_raw_label_weight", 0.25)
        ),
        same_top1_weight=float(getattr(config, "session_graph_same_top1_weight", 0.10)),
        alpha=float(getattr(config, "session_graph_alpha", 0.85)),
        max_iters=max(int(getattr(config, "session_graph_max_iters", 30)), 1),
        tolerance=float(getattr(config, "session_graph_tolerance", 1e-4)),
        strong_seed_score=float(getattr(config, "session_graph_strong_seed_score", 0.60)),
        strong_seed_margin=float(getattr(config, "session_graph_strong_seed_margin", 0.12)),
        override_min_confidence=float(
            getattr(config, "session_graph_override_min_confidence", 0.55)
        ),
        override_min_margin=float(getattr(config, "session_graph_override_min_margin", 0.12)),
        override_min_delta=float(getattr(config, "session_graph_override_min_delta", 0.03)),
        pair_overrides=_normalize_pair_overrides(
            getattr(config, "session_graph_pair_overrides", {}) or {}
        ),
    )


def _repair_config_from_speaker_bank_config(config: object) -> DiarizationRepairConfig:
    return DiarizationRepairConfig(
        enabled=bool(getattr(config, "repair_enabled", False)),
        merge_same_raw_gap_seconds=float(
            getattr(config, "repair_merge_same_raw_gap_seconds", 0.20)
        ),
        snap_boundary_seconds=float(getattr(config, "repair_snap_boundary_seconds", 0.20)),
        max_overlap_trim_seconds=float(
            getattr(config, "repair_max_overlap_trim_seconds", 0.30)
        ),
        split_on_word_gap_seconds=float(
            getattr(config, "repair_split_on_word_gap_seconds", 0.35)
        ),
        max_seed_overlap_seconds=float(
            getattr(config, "repair_max_seed_overlap_seconds", 0.15)
        ),
        min_segment_duration_seconds=float(
            getattr(config, "repair_min_segment_duration_seconds", 0.35)
        ),
    )


def _build_segment_embedding_cache(
    audio_path: str,
    raw_segments: Sequence[dict],
    *,
    extract_embeddings_for_segments_fn: Any,
    hf_token: Optional[str],
    diarization_model_name: Optional[str],
    force_device: Optional[str],
    quiet: bool,
    pre_pad: float,
    post_pad: float,
    batch_size: int,
    workers: int,
) -> List[SessionSegmentEmbedding]:
    payload_by_label: Dict[str, List[Tuple[int, float, float, str]]] = defaultdict(list)
    for index, segment in enumerate(raw_segments):
        raw_label = str(segment.get("speaker_raw") or segment.get("speaker") or "").strip()
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        if not raw_label or end <= start:
            continue
        payload_by_label[raw_label].append((index, start, end, raw_label))

    cached: List[SessionSegmentEmbedding] = []
    for raw_label, payload_items in payload_by_label.items():
        payload = [(start, end, label) for _, start, end, label in payload_items]
        index_map = [item[0] for item in payload_items]
        embed_results, _ = extract_embeddings_for_segments_fn(
            str(audio_path),
            payload,
            hf_token=hf_token,
            diarization_model_name=diarization_model_name,
            force_device=force_device,
            quiet=quiet,
            pre_pad=pre_pad,
            post_pad=post_pad,
            batch_size=batch_size,
            workers=workers,
        )
        for result in embed_results:
            if result.index >= len(index_map):
                continue
            segment_index = index_map[result.index]
            start, end, label = payload[result.index]
            cached.append(
                SessionSegmentEmbedding(
                    segment_index=segment_index,
                    raw_label=label,
                    start=float(start),
                    end=float(end),
                    embedding=np.asarray(result.embedding, dtype=np.float32),
                )
            )
    cached.sort(key=lambda item: (item.segment_index, item.start, item.end))
    return cached


def _candidate_score_map(
    candidates: Sequence[Mapping[str, object]],
    *,
    top_k: int,
    floor: float,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for candidate in list(candidates)[: max(int(top_k), 0)]:
        speaker = str(candidate.get("speaker") or "").strip()
        score = float(candidate.get("score") or 0.0)
        if not speaker or score <= floor:
            continue
        normalized = (score - floor) / max(1.0 - floor, 1e-6)
        scores[speaker] = max(scores.get(speaker, 0.0), min(normalized, 1.0))
    return scores


def _build_session_evidence(
    segments: Sequence[dict],
    seg_matches: Dict[int, Dict[str, object]],
    *,
    threshold: float,
    classifier_min_margin: float,
    graph_config: SessionGraphConfig,
) -> List[SessionSegmentEvidence]:
    evidence: List[SessionSegmentEvidence] = []
    for index, segment in enumerate(segments):
        seg_info = seg_matches.get(index) or {}
        local_match = seg_info.get("match") if seg_info.get("accepted") else None
        candidates = [dict(candidate) for candidate in list(seg_info.get("candidates") or [])[:3]]
        local_score = None
        local_margin = None
        local_speaker = None
        local_source = None
        if local_match:
            local_speaker = str(local_match.get("speaker") or "")
            local_score = float(local_match.get("score") or 0.0)
            local_margin = float(local_match.get("margin") or 0.0)
            local_source = str(local_match.get("source") or "")
        local_top2_pair = None
        if len(candidates) >= 2:
            local_top2_pair = _normalized_pair_key(
                candidates[0].get("speaker"),
                candidates[1].get("speaker"),
            )
        duration = max(
            float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0),
            0.0,
        )
        overlap_heavy = bool(segment.get("speaker_repair_overlap_heavy"))
        was_split = bool(segment.get("speaker_repair_was_split"))
        seed_eligible = bool(
            local_match
            and duration >= 0.75
            and float(local_score or 0.0)
            >= max(float(threshold) + 0.05, float(graph_config.strong_seed_score))
            and float(local_margin or 0.0)
            >= max(float(classifier_min_margin), float(graph_config.strong_seed_margin))
            and not overlap_heavy
            and not was_split
        )
        evidence.append(
            SessionSegmentEvidence(
                segment_index=index,
                start=float(segment.get("start") or 0.0),
                end=float(segment.get("end") or 0.0),
                duration=duration,
                speaker_raw=str(segment.get("speaker_raw") or segment.get("speaker") or ""),
                overlap_heavy=overlap_heavy,
                was_split=was_split,
                local_source=local_source or None,
                local_speaker=local_speaker or None,
                local_score=local_score,
                local_margin=local_margin,
                local_candidates=candidates,
                local_top2_pair=local_top2_pair,
                seed_eligible=seed_eligible,
            )
        )
    return evidence


def _run_session_graph(
    evidence: Sequence[SessionSegmentEvidence],
    embeddings: Mapping[int, np.ndarray],
    *,
    config: SessionGraphConfig,
) -> Dict[int, Dict[str, object]]:
    if not config.enabled or not evidence:
        return {}

    candidate_universe: List[str] = []
    for item in evidence:
        for candidate in item.local_candidates[: config.candidate_top_k]:
            speaker = str(candidate.get("speaker") or "").strip()
            if speaker and speaker not in candidate_universe:
                candidate_universe.append(speaker)
    if not candidate_universe:
        return {}

    segment_count = len(evidence)
    speaker_count = len(candidate_universe)
    total_nodes = segment_count + speaker_count
    speaker_index = {speaker: index for index, speaker in enumerate(candidate_universe)}
    W = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    normalized_embeddings: Dict[int, np.ndarray] = {}

    for item in evidence:
        embedding = embeddings.get(item.segment_index)
        if embedding is None:
            continue
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            normalized_embeddings[item.segment_index] = vector / norm

    for item in evidence:
        row_index = item.segment_index
        candidate_map = _candidate_score_map(
            item.local_candidates,
            top_k=config.candidate_top_k,
            floor=config.candidate_floor,
        )
        for speaker, score in candidate_map.items():
            anchor_index = segment_count + speaker_index[speaker]
            weight = float(score) * float(config.anchor_weight)
            if weight <= 0.0:
                continue
            W[row_index, anchor_index] = max(W[row_index, anchor_index], weight)
            W[anchor_index, row_index] = max(W[anchor_index, row_index], weight)

    matrix_rows = sorted(normalized_embeddings)
    if matrix_rows:
        matrix = np.vstack([normalized_embeddings[index] for index in matrix_rows])
        similarity = np.matmul(matrix, matrix.T)
        for row_offset, seg_index in enumerate(matrix_rows):
            row = similarity[row_offset]
            order = np.argsort(row)[::-1]
            chosen = 0
            for candidate_offset in order:
                other_index = matrix_rows[int(candidate_offset)]
                if other_index == seg_index:
                    continue
                sim = float(row[int(candidate_offset)])
                if sim < float(config.min_similarity):
                    continue
                W[seg_index, other_index] = max(W[seg_index, other_index], sim)
                W[other_index, seg_index] = max(W[other_index, seg_index], sim)
                chosen += 1
                if chosen >= int(config.knn):
                    break

    for left_index, left in enumerate(evidence):
        for right in evidence[left_index + 1 : left_index + 2]:
            gap = max(right.start - left.end, left.start - right.end, 0.0)
            if gap <= float(config.temporal_max_gap_seconds):
                weight = math.exp(-gap / max(float(config.temporal_tau_seconds), 1e-6))
                weight *= float(config.temporal_weight)
                W[left.segment_index, right.segment_index] = max(
                    W[left.segment_index, right.segment_index], weight
                )
                W[right.segment_index, left.segment_index] = max(
                    W[right.segment_index, left.segment_index], weight
                )
            if left.speaker_raw and left.speaker_raw == right.speaker_raw:
                bonus = _edge_override_value(
                    config,
                    [left.local_top2_pair, right.local_top2_pair],
                    "same_raw_label_weight",
                    float(config.same_raw_label_weight),
                )
                W[left.segment_index, right.segment_index] = max(
                    W[left.segment_index, right.segment_index], bonus
                )
                W[right.segment_index, left.segment_index] = max(
                    W[right.segment_index, left.segment_index], bonus
                )
            if (
                not left.seed_eligible
                and not right.seed_eligible
                and left.local_speaker
                and left.local_speaker == right.local_speaker
            ):
                bonus = _edge_override_value(
                    config,
                    [left.local_top2_pair, right.local_top2_pair],
                    "same_top1_weight",
                    float(config.same_top1_weight),
                )
                W[left.segment_index, right.segment_index] = max(
                    W[left.segment_index, right.segment_index], bonus
                )
                W[right.segment_index, left.segment_index] = max(
                    W[right.segment_index, left.segment_index], bonus
                )

    row_sums = np.sum(W, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    S = W / row_sums

    Y = np.zeros((total_nodes, speaker_count), dtype=np.float32)
    fixed_mask = np.zeros((total_nodes,), dtype=bool)
    for speaker, index in speaker_index.items():
        anchor_row = segment_count + index
        Y[anchor_row, index] = 1.0
        fixed_mask[anchor_row] = True
    for item in evidence:
        if not item.seed_eligible or not item.local_speaker:
            continue
        label_index = speaker_index.get(item.local_speaker)
        if label_index is None:
            continue
        Y[item.segment_index, label_index] = 1.0
        fixed_mask[item.segment_index] = True

    F = np.copy(Y)
    for _ in range(max(int(config.max_iters), 1)):
        updated = float(config.alpha) * (S @ F) + ((1.0 - float(config.alpha)) * Y)
        updated[fixed_mask] = Y[fixed_mask]
        if float(np.max(np.abs(updated - F))) <= float(config.tolerance):
            F = updated
            break
        F = updated

    results: Dict[int, Dict[str, object]] = {}
    for item in evidence:
        row = F[item.segment_index]
        if row.size == 0:
            continue
        order = np.argsort(row)[::-1]
        top_index = int(order[0])
        second_value = float(row[int(order[1])]) if len(order) > 1 else 0.0
        top_value = float(row[top_index])
        neighbors: List[Dict[str, object]] = []
        edge_weights = W[item.segment_index, :segment_count]
        if edge_weights.size:
            neighbor_order = np.argsort(edge_weights)[::-1]
            for neighbor_index in neighbor_order[:3]:
                weight = float(edge_weights[int(neighbor_index)])
                if weight <= 0.0:
                    continue
                neighbor_evidence = evidence[int(neighbor_index)]
                neighbors.append(
                    {
                        "segment_index": int(neighbor_index),
                        "speaker_raw": neighbor_evidence.speaker_raw,
                        "local_speaker": neighbor_evidence.local_speaker,
                        "weight": weight,
                    }
                )
        results[item.segment_index] = {
            "speaker": candidate_universe[top_index],
            "score": top_value,
            "margin": top_value - second_value,
            "second_best": second_value,
            "candidates": [
                {"speaker": candidate_universe[int(index)], "score": float(row[int(index)])}
                for index in order[: min(len(order), max(int(config.candidate_top_k), 1))]
            ],
            "neighbors": neighbors,
            "seed": bool(item.seed_eligible),
        }
    return results


def apply_profile_to_segments(
    *,
    audio_path: str,
    segments: Sequence[dict],
    label_embeddings: Mapping[str, np.ndarray],
    speaker_bank: object | None,
    speaker_bank_config: object,
    segment_classifier: object | None = None,
    extract_embeddings_for_segments_fn: Any,
    hf_token: Optional[str] = None,
    diarization_model_name: Optional[str] = None,
    force_device: Optional[str] = None,
    quiet: bool = True,
    precomputed_segment_embeddings: Optional[Sequence[SessionSegmentEmbedding]] = None,
) -> Tuple[List[dict], Dict[str, object], List[SessionSegmentEmbedding]]:
    relabeled = copy.deepcopy(list(segments))
    summary: Dict[str, object] = {
        "attempted": 0,
        "matched": 0,
        "matches": {},
        "segment_counts": {"matched": 0, "unknown": 0},
    }
    if speaker_bank is None or speaker_bank_config is None:
        return relabeled, summary, []
    if not getattr(speaker_bank_config, "use_existing", True):
        return relabeled, summary, []

    repair_config = _repair_config_from_speaker_bank_config(speaker_bank_config)
    graph_config = _graph_config_from_speaker_bank_config(speaker_bank_config)
    threshold = float(getattr(speaker_bank_config, "threshold", 0.0))
    margin_required = max(float(getattr(speaker_bank_config, "scoring_margin", 0.0)), 0.0)
    radius_factor = float(getattr(speaker_bank_config, "radius_factor", 0.0))
    fusion_mode = str(
        getattr(speaker_bank_config, "classifier_fusion_mode", "fallback") or "fallback"
    ).lower()
    classifier_weight = float(getattr(speaker_bank_config, "classifier_fusion_weight", 0.70))
    bank_weight = float(getattr(speaker_bank_config, "classifier_bank_weight", 0.30))

    relabeled, repair_summary = repair_diarization_segments(relabeled, config=repair_config)
    summary["repair"] = repair_summary

    segment_embeddings = list(precomputed_segment_embeddings or [])
    if repair_config.enabled or not segment_embeddings:
        segment_embeddings = _build_segment_embedding_cache(
            audio_path,
            relabeled,
            extract_embeddings_for_segments_fn=extract_embeddings_for_segments_fn,
            hf_token=hf_token,
            diarization_model_name=diarization_model_name,
            force_device=force_device,
            quiet=quiet,
            pre_pad=float(getattr(speaker_bank_config, "pre_pad", 0.15)),
            post_pad=float(getattr(speaker_bank_config, "post_pad", 0.15)),
            batch_size=int(getattr(speaker_bank_config, "embed_batch_size", 16)),
            workers=int(getattr(speaker_bank_config, "embed_workers", 4)),
        )

    label_to_segments: Dict[str, List[int]] = defaultdict(list)
    segment_labels: Dict[int, Optional[str]] = {}
    for index, segment in enumerate(relabeled):
        raw_label = segment.get("speaker_raw") or segment.get("speaker")
        segment["speaker_raw"] = raw_label
        segment_labels[index] = str(raw_label) if raw_label else None
        if raw_label:
            label_to_segments[str(raw_label)].append(index)

    label_stats: Dict[str, Dict[str, object]] = {}
    for label, indices in label_to_segments.items():
        label_stats[label] = {
            "segments_total": len(indices),
            "segments_indices": list(indices),
            "aggregation": str(getattr(speaker_bank_config, "match_aggregation", "mean")).lower(),
        }

    segment_matches: Dict[int, Dict[str, object]] = {
        index: {
            "accepted": False,
            "match": None,
            "top_score": None,
            "second_best": None,
            "margin": None,
            "candidates": [],
        }
        for index in range(len(relabeled))
    }

    for item in segment_embeddings:
        classifier_prediction = None
        classifier_candidates: List[Dict[str, object]] = []
        bank_candidates: List[Dict[str, object]] = []
        if segment_classifier is not None:
            classifier_candidates = segment_classifier.score_candidates(item.embedding)
            classifier_min_confidence, classifier_min_margin = segment_classifier_thresholds(
                duration=max(float(item.end) - float(item.start), 0.0),
                base_confidence=float(
                    getattr(speaker_bank_config, "classifier_min_confidence", 0.0)
                ),
                base_margin=float(getattr(speaker_bank_config, "classifier_min_margin", 0.0)),
            )
            classifier_prediction = segment_classifier.predict(
                item.embedding,
                min_confidence=classifier_min_confidence,
                min_margin=classifier_min_margin,
            )
        if classifier_prediction is not None and fusion_mode == "fallback":
            segment_matches[item.segment_index] = {
                "accepted": True,
                "match": {
                    "speaker": classifier_prediction.speaker,
                    "cluster_id": None,
                    "score": classifier_prediction.score,
                    "distance": None,
                    "margin": classifier_prediction.margin,
                    "second_best": classifier_prediction.second_best,
                    "source": "segment_classifier",
                },
                "top_score": classifier_prediction.score,
                "second_best": classifier_prediction.second_best,
                "margin": classifier_prediction.margin,
                "candidates": classifier_prediction.candidates,
            }
            continue

        bank_candidates = speaker_bank.score_candidates(
            item.embedding,
            radius_factor=radius_factor,
            as_norm_enabled=bool(getattr(speaker_bank_config, "scoring_as_norm_enabled", False)),
            as_norm_cohort_size=int(getattr(speaker_bank_config, "scoring_as_norm_cohort_size", 0)),
        )
        if fusion_mode == "score_sum" and classifier_candidates:
            candidates = fuse_candidate_scores(
                classifier_candidates,
                bank_candidates,
                classifier_weight=classifier_weight,
                bank_weight=bank_weight,
            )
        else:
            candidates = bank_candidates
        top1 = candidates[0] if candidates else None
        top2_score = candidates[1]["score"] if len(candidates) > 1 else None
        margin_value = (
            (top1["score"] - top2_score)
            if top1 and top2_score is not None
            else (top1["score"] if top1 else None)
        )
        accepted = bool(
            top1
            and top1["score"] >= threshold
            and (margin_value if margin_value is not None else 0.0) >= margin_required
        )
        match_payload = None
        if accepted and top1:
            match_payload = {
                "speaker": top1["speaker"],
                "cluster_id": top1.get("cluster_id"),
                "score": top1["score"],
                "distance": top1.get("distance"),
                "margin": margin_value,
                "second_best": top2_score,
                "source": (
                    "segment_fusion"
                    if fusion_mode == "score_sum" and classifier_candidates
                    else f"segment_{top1.get('source', 'centroid')}"
                ),
            }
        segment_matches[item.segment_index] = {
            "accepted": accepted,
            "match": match_payload,
            "top_score": top1["score"] if top1 else None,
            "second_best": top2_score,
            "margin": margin_value,
            "candidates": candidates,
        }

    label_matches: Dict[str, Optional[Dict[str, object]]] = {}
    for label, stats in label_stats.items():
        selection, aggregate_stats = aggregate_segment_label_candidates(
            stats.get("segments_indices", []),
            segment_matches,
            aggregation=str(stats.get("aggregation") or "mean"),
            threshold=threshold,
            margin_required=margin_required,
            min_segments_per_label=int(getattr(speaker_bank_config, "min_segments_per_label", 1)),
        )
        stats.update(aggregate_stats)
        stats["selection"] = selection
        if selection:
            label_matches[label] = {
                "speaker": selection.get("speaker"),
                "cluster_id": selection.get("cluster_id"),
                "score": selection.get("score"),
                "score_max": selection.get("score_max"),
                "distance": selection.get("distance"),
                "margin": selection.get("margin"),
                "second_best": selection.get("second_best"),
                "source": selection.get("source"),
            }
        else:
            label_matches[label] = None

    for label, vector in label_embeddings.items():
        if label in label_matches and label_matches[label]:
            continue
        stats_for_label = label_stats.get(label) or {}
        match = speaker_bank.match(
            vector,
            threshold=threshold,
            radius_factor=radius_factor,
            margin=margin_required,
            as_norm_enabled=bool(getattr(speaker_bank_config, "scoring_as_norm_enabled", False)),
            as_norm_cohort_size=int(getattr(speaker_bank_config, "scoring_as_norm_cohort_size", 0)),
        )
        if match:
            match["source"] = "label_vector"
            label_matches[label] = match
            continue
        if segment_classifier is not None:
            classifier_min_confidence, classifier_min_margin = label_classifier_thresholds(
                segment_count=int(stats_for_label.get("segments_total") or 0),
                base_confidence=float(
                    getattr(speaker_bank_config, "classifier_min_confidence", 0.0)
                ),
                base_margin=float(getattr(speaker_bank_config, "classifier_min_margin", 0.0)),
            )
            classifier_prediction = segment_classifier.predict(
                vector,
                min_confidence=classifier_min_confidence,
                min_margin=classifier_min_margin,
            )
            if classifier_prediction is not None:
                label_matches[label] = {
                    "speaker": classifier_prediction.speaker,
                    "cluster_id": None,
                    "score": classifier_prediction.score,
                    "score_max": classifier_prediction.score,
                    "distance": None,
                    "margin": classifier_prediction.margin,
                    "second_best": classifier_prediction.second_best,
                    "source": "label_classifier",
                }
                continue
        label_matches[label] = match

    summary["attempted"] = len(label_matches)
    summary["matched"] = sum(1 for match in label_matches.values() if match)

    matched_segments = 0
    unknown_segments = 0
    for label, match in label_matches.items():
        summary["matches"][label] = match
        summary.setdefault("label_stats", {})[label] = {
            "segments_total": (label_stats.get(label) or {}).get("segments_total"),
            "segments_matched": (label_stats.get(label) or {}).get("segments_matched"),
            "aggregation": (label_stats.get(label) or {}).get("aggregation"),
            "margin": (label_stats.get(label) or {}).get("margin"),
        }
        for index in label_to_segments.get(label, []):
            segment = relabeled[index]
            seg_match_info = segment_matches.get(index) or {}
            top_candidates = [
                {
                    "speaker": candidate.get("speaker"),
                    "score": candidate.get("score"),
                    "source": candidate.get("source"),
                }
                for candidate in list(seg_match_info.get("candidates") or [])[:3]
            ]
            segment_level_match = None
            if seg_match_info.get("accepted") and seg_match_info.get("match"):
                segment_level_match = dict(seg_match_info["match"])
                segment_level_match["label"] = label

            effective_match = segment_level_match
            if not effective_match and match:
                effective_match = {
                    "speaker": match["speaker"],
                    "score": match.get("score"),
                    "cluster_id": match.get("cluster_id"),
                    "distance": match.get("distance"),
                    "margin": match.get("margin"),
                    "second_best": match.get("second_best"),
                    "source": match.get("source") or "speaker_bank",
                    "label": label,
                }

            segment["speaker_match_candidates"] = top_candidates
            segment["speaker_local_match"] = effective_match
            segment["speaker_local_match_source"] = (
                effective_match.get("source") if effective_match else "speaker_bank"
            )
            segment["speaker_local_match_score"] = (
                effective_match.get("score") if effective_match else None
            )
            segment["speaker_local_match_distance"] = (
                effective_match.get("distance") if effective_match else None
            )
            segment["speaker_local_match_cluster"] = (
                effective_match.get("cluster_id") if effective_match else None
            )
            if effective_match:
                set_segment_speaker_label(segment, effective_match["speaker"])
                segment["speaker_match"] = dict(effective_match)
                segment["speaker_match_source"] = effective_match.get("source", "speaker_bank")
                segment["speaker_match_score"] = segment["speaker_match"].get("score")
                segment["speaker_match_distance"] = segment["speaker_match"].get("distance")
                segment["speaker_match_cluster"] = segment["speaker_match"].get("cluster_id")
                matched_segments += 1
            else:
                set_segment_speaker_label(segment, "unknown")
                segment["speaker_match"] = {
                    "speaker": None,
                    "score": None,
                    "cluster_id": None,
                    "distance": None,
                    "source": "speaker_bank",
                    "label": label,
                }
                segment["speaker_match_score"] = None
                segment["speaker_match_distance"] = None
                segment["speaker_match_cluster"] = None
                segment["speaker_match_source"] = "speaker_bank"
                unknown_segments += 1

    for index, segment in enumerate(relabeled):
        if segment_labels.get(index) is None:
            if not segment.get("speaker"):
                set_segment_speaker_label(segment, "unknown")
                unknown_segments += 1
            segment.setdefault(
                "speaker_match",
                {
                    "speaker": None,
                    "score": None,
                    "cluster_id": None,
                    "distance": None,
                    "source": "speaker_bank",
                    "label": None,
                },
            )
            segment.setdefault("speaker_match_source", "speaker_bank")
            segment.setdefault("speaker_match_score", None)
            segment.setdefault("speaker_match_distance", None)
            segment.setdefault("speaker_match_cluster", None)
            segment.setdefault("speaker_match_candidates", [])
            segment.setdefault("speaker_local_match", None)
            segment.setdefault("speaker_local_match_source", "speaker_bank")

    summary["segment_counts"]["matched"] = matched_segments
    summary["segment_counts"]["unknown"] = unknown_segments
    summary["pre_graph_segment_counts"] = dict(summary["segment_counts"])

    evidence = _build_session_evidence(
        relabeled,
        segment_matches,
        threshold=threshold,
        classifier_min_margin=float(getattr(speaker_bank_config, "classifier_min_margin", 0.0)),
        graph_config=graph_config,
    )
    embedding_map = {item.segment_index: np.asarray(item.embedding, dtype=np.float32) for item in segment_embeddings}
    graph_matches = _run_session_graph(evidence, embedding_map, config=graph_config)
    graph_summary = {
        "enabled": bool(graph_config.enabled),
        "seed_segments": 0,
        "overrides": 0,
        "rescued_unknowns": 0,
        "unchanged": 0,
        "graph_unresolved": 0,
        "candidate_universe": sorted(
            {
                str(candidate.get("speaker") or "")
                for item in evidence
                for candidate in item.local_candidates[: graph_config.candidate_top_k]
                if str(candidate.get("speaker") or "").strip()
            }
        ),
        "pairs": {},
    }
    graph_pair_summary: Dict[str, Dict[str, int]] = {}
    for item in evidence:
        if item.seed_eligible:
            graph_summary["seed_segments"] = int(graph_summary["seed_segments"]) + 1
        graph_match = graph_matches.get(item.segment_index)
        segment = relabeled[item.segment_index]
        pair_key = item.local_top2_pair
        segment["speaker_graph_pair"] = pair_key
        segment["speaker_graph_seed"] = bool(item.seed_eligible)
        segment["speaker_graph_source"] = "disabled"
        segment["speaker_graph_override"] = False
        segment["speaker_graph_match"] = None
        segment["speaker_graph_score"] = None
        segment["speaker_graph_margin"] = None
        segment["speaker_graph_neighbors"] = []
        if graph_match is None:
            if str(segment.get("speaker") or "").strip().lower() == "unknown":
                graph_summary["graph_unresolved"] = int(graph_summary["graph_unresolved"]) + 1
            continue
        if pair_key:
            graph_pair_summary.setdefault(
                pair_key,
                {
                    "overrides_attempted": 0,
                    "overrides_accepted": 0,
                    "rescued_unknowns": 0,
                    "unchanged": 0,
                },
            )
        segment["speaker_graph_match"] = {
            "speaker": graph_match.get("speaker"),
            "score": graph_match.get("score"),
            "margin": graph_match.get("margin"),
            "second_best": graph_match.get("second_best"),
            "candidates": list(graph_match.get("candidates") or []),
        }
        segment["speaker_graph_score"] = graph_match.get("score")
        segment["speaker_graph_margin"] = graph_match.get("margin")
        segment["speaker_graph_neighbors"] = list(graph_match.get("neighbors") or [])
        segment["speaker_graph_source"] = "session_graph"
        if item.seed_eligible:
            graph_summary["unchanged"] = int(graph_summary["unchanged"]) + 1
            if pair_key:
                graph_pair_summary[pair_key]["unchanged"] = (
                    int(graph_pair_summary[pair_key]["unchanged"]) + 1
                )
            continue
        if pair_key:
            graph_pair_summary[pair_key]["overrides_attempted"] = (
                int(graph_pair_summary[pair_key]["overrides_attempted"]) + 1
            )
        graph_speaker = str(graph_match.get("speaker") or "").strip()
        graph_score = float(graph_match.get("score") or 0.0)
        graph_margin = float(graph_match.get("margin") or 0.0)
        local_speaker = str(segment.get("speaker") or "").strip()
        local_score = (
            float(segment.get("speaker_local_match_score") or 0.0)
            if segment.get("speaker_local_match_score") is not None
            else None
        )
        override_min_confidence = _pair_override_value(
            graph_config,
            pair_key,
            "override_min_confidence",
            float(graph_config.override_min_confidence),
        )
        override_min_margin = _pair_override_value(
            graph_config,
            pair_key,
            "override_min_margin",
            float(graph_config.override_min_margin),
        )
        override_min_delta = _pair_override_value(
            graph_config,
            pair_key,
            "override_min_delta",
            float(graph_config.override_min_delta),
        )
        should_override = bool(
            graph_speaker
            and graph_score >= override_min_confidence
            and graph_margin >= override_min_margin
        )
        if should_override and local_speaker and local_speaker.lower() != "unknown":
            should_override = graph_score >= float(local_score or 0.0) + override_min_delta
        if should_override:
            previous_speaker = local_speaker or "unknown"
            set_segment_speaker_label(segment, graph_speaker)
            segment["speaker_graph_override"] = True
            segment["speaker_match"] = {
                "speaker": graph_speaker,
                "score": graph_score,
                "cluster_id": None,
                "distance": None,
                "source": "session_graph",
                "label": segment.get("speaker_raw"),
                "margin": graph_margin,
                "second_best": graph_match.get("second_best"),
            }
            segment["speaker_match_score"] = graph_score
            segment["speaker_match_distance"] = None
            segment["speaker_match_cluster"] = None
            segment["speaker_match_source"] = "session_graph"
            graph_summary["overrides"] = int(graph_summary["overrides"]) + 1
            if pair_key:
                graph_pair_summary[pair_key]["overrides_accepted"] = (
                    int(graph_pair_summary[pair_key]["overrides_accepted"]) + 1
                )
            if previous_speaker.lower() == "unknown":
                graph_summary["rescued_unknowns"] = int(graph_summary["rescued_unknowns"]) + 1
                if pair_key:
                    graph_pair_summary[pair_key]["rescued_unknowns"] = (
                        int(graph_pair_summary[pair_key]["rescued_unknowns"]) + 1
                    )
        else:
            if local_speaker.lower() == "unknown":
                graph_summary["graph_unresolved"] = int(graph_summary["graph_unresolved"]) + 1
            else:
                graph_summary["unchanged"] = int(graph_summary["unchanged"]) + 1
                if pair_key:
                    graph_pair_summary[pair_key]["unchanged"] = (
                        int(graph_pair_summary[pair_key]["unchanged"]) + 1
                    )

    graph_summary["pairs"] = {
        pair_key: graph_pair_summary[pair_key]
        for pair_key in sorted(graph_pair_summary)
    }
    summary["graph"] = graph_summary
    summary["segment_counts"]["matched"] = sum(
        1 for segment in relabeled if str(segment.get("speaker") or "").strip().lower() != "unknown"
    )
    summary["segment_counts"]["unknown"] = len(relabeled) - int(summary["segment_counts"]["matched"])
    return relabeled, summary, segment_embeddings
