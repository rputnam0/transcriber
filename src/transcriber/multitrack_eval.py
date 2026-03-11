from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np

from .cli import (
    _aggregate_segment_label_candidates,
    _label_classifier_thresholds,
    _load_yaml_or_json,
    _resolve_speaker_bank_paths,
    _resolve_speaker_bank_settings,
    _segment_classifier_thresholds,
    _set_segment_speaker_label,
    run_transcribe,
)
from .consolidate import choose_speaker, consolidate, save_outputs
from .segment_classifier import load_labeled_records, load_segment_classifier
from .speaker_bank import SpeakerBank
from .diarization import (
    extract_embeddings_for_segments,
    extract_speaker_embeddings,
    release_runtime_caches,
)


@dataclass
class WordSpan:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class CachedSegmentEmbedding:
    segment_index: int
    raw_label: str
    start: float
    end: float
    embedding: np.ndarray


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_words_from_jsonl(path: Path) -> List[WordSpan]:
    words: List[WordSpan] = []
    for record in load_jsonl_records(path):
        segment_speaker = record.get("speaker")
        raw_words = record.get("words") or []
        if not isinstance(raw_words, list):
            continue
        for word in raw_words:
            if not isinstance(word, dict):
                continue
            start = word.get("start")
            end = word.get("end")
            text = str(word.get("word") or word.get("text") or "").strip()
            speaker = word.get("speaker") or segment_speaker
            if start is None or end is None or not text or not speaker:
                continue
            words.append(
                WordSpan(
                    speaker=str(speaker),
                    start=float(start),
                    end=float(end),
                    text=text,
                )
            )
    words.sort(key=lambda item: (item.start, item.end, item.text))
    return words


def _window_records(records: Sequence[dict], start: float, end: float) -> List[dict]:
    return [
        record
        for record in records
        if float(record.get("end") or 0.0) > start and float(record.get("start") or 0.0) < end
    ]


def select_candidate_windows(
    records: Sequence[dict],
    *,
    window_seconds: float = 300.0,
    hop_seconds: float = 60.0,
    top_k: int = 3,
    min_speakers: int = 3,
) -> List[dict]:
    if not records:
        return []

    max_end = max(float(record.get("end") or 0.0) for record in records)
    candidates: List[dict] = []
    start = 0.0
    while start + window_seconds <= max_end + 1e-6:
        end = start + window_seconds
        window_rows = sorted(
            _window_records(records, start, end), key=lambda item: item.get("start") or 0.0
        )
        speaker_counts: Counter[str] = Counter()
        total_words = 0
        turns = 0
        previous_speaker: Optional[str] = None
        for row in window_rows:
            speaker = row.get("speaker")
            if not speaker:
                continue
            speaker = str(speaker)
            word_count = len(str(row.get("text") or "").split())
            speaker_counts[speaker] += word_count
            total_words += word_count
            if previous_speaker is not None and speaker != previous_speaker:
                turns += 1
            previous_speaker = speaker
        unique_speakers = len(speaker_counts)
        if unique_speakers >= min_speakers and total_words:
            dominant_words = max(speaker_counts.values())
            balance = 1.0 - (dominant_words / total_words)
            score = (unique_speakers * 1000.0) + (turns * 10.0) + total_words + (balance * 100.0)
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "score": score,
                    "speaker_count": unique_speakers,
                    "speaker_turns": turns,
                    "total_words": total_words,
                    "speakers": sorted(speaker_counts),
                }
            )
        start += hop_seconds

    candidates.sort(
        key=lambda item: (item["score"], item["speaker_count"], item["speaker_turns"]), reverse=True
    )
    selected: List[dict] = []
    for candidate in candidates:
        overlaps = False
        for chosen in selected:
            if not (candidate["end"] <= chosen["start"] or candidate["start"] >= chosen["end"]):
                overlaps = True
                break
        if overlaps:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def score_word_speaker_alignment(
    reference_words: Sequence[WordSpan],
    predicted_words: Sequence[WordSpan],
    *,
    tolerance_seconds: float = 0.35,
) -> Dict[str, object]:
    predicted = sorted(predicted_words, key=lambda item: (item.start, item.end))
    correct = 0
    matched = 0
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    per_speaker_total: Counter[str] = Counter()
    per_speaker_correct: Counter[str] = Counter()

    cursor = 0
    for ref in sorted(reference_words, key=lambda item: (item.start, item.end)):
        per_speaker_total[ref.speaker] += 1
        midpoint = ref.midpoint
        while cursor < len(predicted) and predicted[cursor].end < midpoint - tolerance_seconds:
            cursor += 1
        best_match: Optional[WordSpan] = None
        best_distance: Optional[tuple[float, float]] = None
        probe = max(cursor - 1, 0)
        while probe < len(predicted):
            pred = predicted[probe]
            if pred.start > midpoint + tolerance_seconds:
                break
            if pred.start <= midpoint <= pred.end:
                interval_distance = 0.0
            else:
                interval_distance = min(abs(midpoint - pred.start), abs(midpoint - pred.end))
            if interval_distance <= tolerance_seconds:
                candidate_distance = (interval_distance, abs(pred.midpoint - midpoint))
                if best_distance is None or candidate_distance < best_distance:
                    best_match = pred
                    best_distance = candidate_distance
            probe += 1

        if best_match is None:
            confusion[ref.speaker]["<unmatched>"] += 1
            continue

        matched += 1
        confusion[ref.speaker][best_match.speaker] += 1
        if best_match.speaker == ref.speaker:
            correct += 1
            per_speaker_correct[ref.speaker] += 1

    total = len(reference_words)
    return {
        "reference_words": total,
        "matched_words": matched,
        "correct_words": correct,
        "coverage": (matched / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "matched_accuracy": (correct / matched) if matched else 0.0,
        "per_speaker_accuracy": {
            speaker: {
                "total": per_speaker_total[speaker],
                "correct": per_speaker_correct[speaker],
                "accuracy": (
                    per_speaker_correct[speaker] / per_speaker_total[speaker]
                    if per_speaker_total[speaker]
                    else 0.0
                ),
            }
            for speaker in sorted(per_speaker_total)
        },
        "confusion": {speaker: dict(counts) for speaker, counts in sorted(confusion.items())},
    }


def compute_segment_purity(
    segments: Sequence[dict],
    stem_arrays: Sequence[Tuple[str, np.ndarray]],
    *,
    sample_rate: int = 16000,
    min_power: float = 2e-4,
    training_min_share: float = 0.80,
    diagnostic_min_share: float = 0.65,
    min_training_segment_dur: float = 0.8,
    max_training_segment_dur: float = 12.0,
) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    bucket_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    accepted = 0

    for segment in segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        duration = max(end - start, 0.0)
        start_index = max(0, int(np.floor(start * sample_rate)))
        end_index = max(start_index + 1, int(np.ceil(end * sample_rate)))

        powers: Counter[str] = Counter()
        for label, array in stem_arrays:
            clip = array[start_index : min(end_index, array.shape[0])]
            if clip.size:
                powers[label] += float(np.mean(clip * clip))

        total_power = float(sum(powers.values()))
        top_label = None
        top_power = 0.0
        second_power = 0.0
        dominant_share = 0.0
        active_speakers = sum(1 for value in powers.values() if value >= min_power)
        if total_power > 0.0 and powers:
            top_label, top_power = powers.most_common(1)[0]
            second_power = powers.most_common(2)[1][1] if len(powers) > 1 else 0.0
            dominant_share = top_power / total_power

        if dominant_share >= training_min_share:
            bucket = "high"
        elif dominant_share >= diagnostic_min_share:
            bucket = "medium"
        else:
            bucket = "low"
        bucket_counts[bucket] += 1

        rejection = None
        if duration < min_training_segment_dur:
            rejection = "too_short"
        elif duration > max_training_segment_dur:
            rejection = "too_long"
        elif total_power <= 0.0:
            rejection = "no_power"
        elif top_power < min_power:
            rejection = "low_power"
        elif dominant_share < training_min_share:
            rejection = "low_share"
        elif top_power <= second_power:
            rejection = "not_dominant"

        if rejection is None:
            accepted += 1
        else:
            rejection_counts[rejection] += 1

        records.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "raw_label": str(segment.get("speaker_raw") or segment.get("speaker") or ""),
                "speaker": top_label,
                "dominant_share": dominant_share,
                "top1_power": top_power,
                "top2_power": second_power,
                "power_gap": top_power - second_power,
                "active_speakers": active_speakers,
                "bucket": bucket,
                "accepted_training": rejection is None,
                "rejection": rejection,
            }
        )

    dominant_values = [float(item["dominant_share"]) for item in records]
    return {
        "segments": len(records),
        "accepted_training_segments": accepted,
        "bucket_counts": dict(bucket_counts),
        "rejection_counts": dict(rejection_counts),
        "dominant_share_mean": (
            (sum(dominant_values) / len(dominant_values)) if dominant_values else 0.0
        ),
        "records": records,
    }


def extract_session_stems(zip_path: Path, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_stems = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".m4a"}
    )
    if existing_stems:
        return existing_stems
    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    stems = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".m4a"}
    )
    return stems


def _load_clip_stem_arrays(
    clip_paths: Sequence[Path], clip_labels: Dict[str, str]
) -> List[Tuple[str, np.ndarray]]:
    import soundfile as sf

    stem_arrays: List[Tuple[str, np.ndarray]] = []
    for clip_path in clip_paths:
        audio_data, sample_rate = sf.read(clip_path)
        if getattr(audio_data, "ndim", 1) > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != 16000:
            raise RuntimeError(f"Expected 16 kHz clips, found {sample_rate} for {clip_path}")
        label = str(clip_labels.get(clip_path.name) or "unknown")
        stem_arrays.append((label, np.asarray(audio_data, dtype=np.float32)))
    return stem_arrays


def _run_ffmpeg(command: List[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def clip_audio(input_path: Path, output_path: Path, start: float, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def mix_audio_files(inputs: Sequence[Path], output_path: Path) -> None:
    if not inputs:
        raise ValueError("No audio inputs were provided for mixing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            f"amix=inputs={len(inputs)}:normalize=0:dropout_transition=0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    _run_ffmpeg(command)


def _resolve_run_defaults(config_path: Optional[Path]) -> Dict[str, object]:
    cfg = _load_yaml_or_json(str(config_path) if config_path else None)
    speaker_bank_config, speaker_bank_root = _resolve_speaker_bank_settings(cfg, SimpleNamespace())
    return {
        "backend": str(cfg.get("backend") or "faster"),
        "model_name": str(cfg.get("model") or "large-v3"),
        "compute_type": str(cfg.get("compute_type") or "float16"),
        "batch_size": int(cfg.get("batch_size") or 16),
        "auto_batch": bool(cfg.get("auto_batch", True)),
        "cache_mode": str(cfg.get("cache_mode") or "home"),
        "hf_cache_root": cfg.get("hf_cache_root"),
        "speaker_mapping_path": cfg.get("speaker_mapping"),
        "speaker_bank_root": cfg.get("speaker_bank_root") or speaker_bank_root,
        "speaker_bank_config": speaker_bank_config,
        "device": cfg.get("device"),
        "local_files_only": bool(cfg.get("local_files_only", False)),
    }


def _resolve_output_jsonl(input_path: Path, output_dir: Path) -> Path:
    base = input_path.stem
    nested = output_dir / base / f"{base}.jsonl"
    if nested.exists():
        return nested
    direct = output_dir / f"{base}.jsonl"
    if direct.exists():
        return direct
    matches = sorted(output_dir.rglob("*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No JSONL outputs found under {output_dir}")
    return matches[0]


def _resolve_output_dir(input_path: Path, output_dir: Path) -> Path:
    base = input_path.stem
    if output_dir.name == base:
        return output_dir
    return output_dir / base


def _speaker_hf_token() -> Optional[str]:
    return (
        os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def _save_embedding_map(path: Path, embeddings: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(embeddings)
    if labels:
        matrix = np.vstack([np.asarray(embeddings[label], dtype=np.float32) for label in labels])
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(path, labels=np.asarray(labels), embeddings=matrix)


def _load_embedding_map(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    labels = payload["labels"].tolist()
    matrix = np.asarray(payload["embeddings"], dtype=np.float32)
    if matrix.ndim == 1 and labels:
        matrix = matrix.reshape(1, -1)
    return {
        str(label): np.asarray(vector, dtype=np.float32)
        for label, vector in zip(labels, matrix, strict=False)
    }


def _save_cached_segment_embeddings(
    path: Path, embeddings: Sequence[CachedSegmentEmbedding]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings:
        matrix = np.vstack([item.embedding.astype(np.float32) for item in embeddings])
        segment_indices = np.asarray([item.segment_index for item in embeddings], dtype=np.int32)
        starts = np.asarray([item.start for item in embeddings], dtype=np.float32)
        ends = np.asarray([item.end for item in embeddings], dtype=np.float32)
        labels = np.asarray([item.raw_label for item in embeddings])
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
        segment_indices = np.zeros((0,), dtype=np.int32)
        starts = np.zeros((0,), dtype=np.float32)
        ends = np.zeros((0,), dtype=np.float32)
        labels = np.asarray([])
    np.savez_compressed(
        path,
        segment_indices=segment_indices,
        raw_labels=labels,
        starts=starts,
        ends=ends,
        embeddings=matrix,
    )


def _load_cached_segment_embeddings(path: Path) -> List[CachedSegmentEmbedding]:
    payload = np.load(path, allow_pickle=False)
    indices = np.asarray(payload["segment_indices"], dtype=np.int32)
    labels = payload["raw_labels"].tolist()
    starts = np.asarray(payload["starts"], dtype=np.float32)
    ends = np.asarray(payload["ends"], dtype=np.float32)
    matrix = np.asarray(payload["embeddings"], dtype=np.float32)
    if matrix.ndim == 1 and len(indices):
        matrix = matrix.reshape(1, -1)
    return [
        CachedSegmentEmbedding(
            segment_index=int(indices[idx]),
            raw_label=str(labels[idx]),
            start=float(starts[idx]),
            end=float(ends[idx]),
            embedding=np.asarray(matrix[idx], dtype=np.float32),
        )
        for idx in range(len(indices))
    ]


def _build_segment_embedding_cache(
    audio_path: Path,
    raw_segments: Sequence[dict],
    *,
    hf_token: Optional[str],
    force_device: Optional[str],
    quiet: bool,
) -> List[CachedSegmentEmbedding]:
    payload_by_label: Dict[str, List[Tuple[int, float, float, str]]] = defaultdict(list)
    for index, segment in enumerate(raw_segments):
        raw_label = str(segment.get("speaker_raw") or segment.get("speaker") or "").strip()
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        if not raw_label or end <= start:
            continue
        payload_by_label[raw_label].append((index, start, end, raw_label))

    cached: List[CachedSegmentEmbedding] = []
    for raw_label, payload_items in payload_by_label.items():
        payload = [(start, end, label) for _, start, end, label in payload_items]
        index_map = [item[0] for item in payload_items]
        embed_results, _ = extract_embeddings_for_segments(
            str(audio_path),
            payload,
            hf_token=hf_token,
            force_device=force_device,
            quiet=quiet,
        )
        for result in embed_results:
            if result.index >= len(index_map):
                continue
            segment_index = index_map[result.index]
            start, end, label = payload[result.index]
            cached.append(
                CachedSegmentEmbedding(
                    segment_index=segment_index,
                    raw_label=label,
                    start=float(start),
                    end=float(end),
                    embedding=np.asarray(result.embedding, dtype=np.float32),
                )
            )
    cached.sort(key=lambda item: (item.segment_index, item.start, item.end))
    return cached


def _fuse_candidate_scores(
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


def apply_profile_to_cached_segments(
    segments: Sequence[dict],
    *,
    label_embeddings: Dict[str, np.ndarray],
    segment_embeddings: Sequence[CachedSegmentEmbedding],
    speaker_bank: Optional[SpeakerBank],
    speaker_bank_config: object,
    segment_classifier: object | None = None,
) -> Tuple[List[dict], Dict[str, object]]:
    relabeled = copy.deepcopy(list(segments))
    summary: Dict[str, object] = {
        "attempted": 0,
        "matched": 0,
        "matches": {},
        "segment_counts": {"matched": 0, "unknown": 0},
    }
    if speaker_bank is None or speaker_bank_config is None:
        return relabeled, summary
    if not getattr(speaker_bank_config, "use_existing", True):
        return relabeled, summary

    threshold = float(getattr(speaker_bank_config, "threshold", 0.0))
    margin_required = max(float(getattr(speaker_bank_config, "scoring_margin", 0.0)), 0.0)
    radius_factor = float(getattr(speaker_bank_config, "radius_factor", 0.0))
    fusion_mode = str(
        getattr(speaker_bank_config, "classifier_fusion_mode", "fallback") or "fallback"
    ).lower()
    classifier_weight = float(getattr(speaker_bank_config, "classifier_fusion_weight", 0.70))
    bank_weight = float(getattr(speaker_bank_config, "classifier_bank_weight", 0.30))

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
            classifier_min_confidence, classifier_min_margin = _segment_classifier_thresholds(
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
            candidates = _fuse_candidate_scores(
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
                "cluster_id": top1["cluster_id"],
                "score": top1["score"],
                "distance": top1["distance"],
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
        selection, aggregate_stats = _aggregate_segment_label_candidates(
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
            classifier_min_confidence, classifier_min_margin = _label_classifier_thresholds(
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

            if effective_match:
                _set_segment_speaker_label(segment, effective_match["speaker"])
                segment["speaker_match"] = effective_match
                segment["speaker_match_source"] = effective_match.get("source", "speaker_bank")
                segment["speaker_match_score"] = segment["speaker_match"].get("score")
                segment["speaker_match_distance"] = segment["speaker_match"].get("distance")
                segment["speaker_match_cluster"] = segment["speaker_match"].get("cluster_id")
                matched_segments += 1
            else:
                _set_segment_speaker_label(segment, "unknown")
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
                _set_segment_speaker_label(segment, "unknown")
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

    summary["segment_counts"]["matched"] = matched_segments
    summary["segment_counts"]["unknown"] = unknown_segments
    return relabeled, summary


def _prepare_reference_outputs(
    *,
    clips_dir: Path,
    reference_out_dir: Path,
    speaker_mapping_path: Path,
    defaults: Dict[str, object],
    device_override: Optional[str],
    local_files_only_override: Optional[bool],
) -> Path:
    if list(reference_out_dir.rglob("*.jsonl")):
        return _resolve_output_jsonl(clips_dir, reference_out_dir)
    release_runtime_caches()
    run_transcribe(
        input_path=str(clips_dir),
        backend="faster",
        model_name=str(defaults["model_name"]),
        compute_type="int8",
        batch_size=1,
        output_dir=str(reference_out_dir),
        speaker_mapping_path=str(speaker_mapping_path),
        min_speakers=None,
        max_speakers=None,
        write_srt=False,
        write_jsonl=True,
        hf_cache_root=defaults["hf_cache_root"],  # type: ignore[arg-type]
        speaker_bank_root=None,
        local_files_only=(
            bool(defaults["local_files_only"])
            if local_files_only_override is None
            else local_files_only_override
        ),
        quiet=True,
        auto_batch=bool(defaults["auto_batch"]),
        cache_mode=str(defaults["cache_mode"]),
        device="cpu",
        speaker_bank_config=None,
    )
    return _resolve_output_jsonl(clips_dir, reference_out_dir)


def _prepare_predicted_cache(
    *,
    mixed_path: Path,
    raw_out_dir: Path,
    speaker_count: int,
    defaults: Dict[str, object],
    device_override: Optional[str],
    local_files_only_override: Optional[bool],
) -> Tuple[Path, Path, Path]:
    raw_jsonl_path = raw_out_dir / mixed_path.stem / f"{mixed_path.stem}.jsonl"
    label_embedding_path = raw_out_dir / "label_embeddings.npz"
    segment_embedding_path = raw_out_dir / "segment_embeddings.npz"
    if (
        raw_jsonl_path.exists()
        and label_embedding_path.exists()
        and segment_embedding_path.exists()
    ):
        return raw_jsonl_path, label_embedding_path, segment_embedding_path

    release_runtime_caches()
    run_transcribe(
        input_path=str(mixed_path),
        backend="faster",
        model_name=str(defaults["model_name"]),
        compute_type=str(defaults["compute_type"]),
        batch_size=int(defaults["batch_size"]),
        output_dir=str(raw_out_dir),
        speaker_mapping_path=str(defaults["speaker_mapping_path"] or ""),
        min_speakers=int(speaker_count),
        max_speakers=int(speaker_count),
        write_srt=False,
        write_jsonl=True,
        hf_cache_root=defaults["hf_cache_root"],  # type: ignore[arg-type]
        speaker_bank_root=None,
        local_files_only=(
            bool(defaults["local_files_only"])
            if local_files_only_override is None
            else local_files_only_override
        ),
        quiet=True,
        auto_batch=bool(defaults["auto_batch"]),
        cache_mode=str(defaults["cache_mode"]),
        device=device_override or defaults["device"],  # type: ignore[arg-type]
        speaker_bank_config=None,
    )
    raw_jsonl_path = _resolve_output_jsonl(mixed_path, raw_out_dir)
    raw_segments = load_jsonl_records(raw_jsonl_path)
    embeddings, _ = extract_speaker_embeddings(
        str(mixed_path),
        _speaker_hf_token(),
        min_speakers=int(speaker_count),
        max_speakers=int(speaker_count),
        force_device=device_override or defaults["device"],  # type: ignore[arg-type]
        quiet=True,
    )
    _save_embedding_map(label_embedding_path, embeddings)
    cached_segments = _build_segment_embedding_cache(
        mixed_path,
        raw_segments,
        hf_token=_speaker_hf_token(),
        force_device=device_override or defaults["device"],  # type: ignore[arg-type]
        quiet=True,
    )
    _save_cached_segment_embeddings(segment_embedding_path, cached_segments)
    return raw_jsonl_path, label_embedding_path, segment_embedding_path


def evaluate_multitrack_session(
    *,
    session_zip: Path,
    session_jsonl: Path,
    output_dir: Path,
    cache_root: Optional[Path],
    speaker_mapping_path: Path,
    config_path: Optional[Path],
    window_seconds: float,
    hop_seconds: float,
    top_k: int,
    min_speakers: int,
    device_override: Optional[str],
    local_files_only_override: Optional[bool],
) -> Dict[str, object]:
    defaults = _resolve_run_defaults(config_path)
    defaults["speaker_mapping_path"] = str(speaker_mapping_path)
    mapping = _load_yaml_or_json(str(speaker_mapping_path))
    windows = select_candidate_windows(
        load_labeled_records(
            session_jsonl,
            speaker_aliases={"zariel torgan": "David Tanglethorn"},
            speaker_mapping=mapping,
        ),
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        top_k=top_k,
        min_speakers=min_speakers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_root = (cache_root or (output_dir / "_cache")).expanduser().resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    stems_dir = prepared_root / "stems"
    stem_paths = extract_session_stems(session_zip, stems_dir)

    speaker_bank = None
    segment_classifier = None
    speaker_bank_config = defaults["speaker_bank_config"]
    if speaker_bank_config is not None:
        bank_root, bank_profile, _ = _resolve_speaker_bank_paths(
            speaker_bank_config,
            defaults["speaker_bank_root"],  # type: ignore[arg-type]
            defaults["hf_cache_root"],  # type: ignore[arg-type]
        )
        speaker_bank = SpeakerBank(
            bank_root,
            profile=bank_profile,
            cluster_method=speaker_bank_config.cluster_method,
            dbscan_eps=speaker_bank_config.cluster_eps,
            dbscan_min_samples=speaker_bank_config.cluster_min_samples,
            prototypes_enabled=speaker_bank_config.prototypes_enabled,
            prototypes_per_cluster=speaker_bank_config.prototypes_per_cluster,
            prototypes_method=speaker_bank_config.prototypes_method,
            scoring_whiten=speaker_bank_config.scoring_whiten,
        )
        _, _, bank_profile_dir = _resolve_speaker_bank_paths(
            speaker_bank_config,
            defaults["speaker_bank_root"],  # type: ignore[arg-type]
            defaults["hf_cache_root"],  # type: ignore[arg-type]
        )
        segment_classifier = load_segment_classifier(bank_profile_dir)

    results: List[dict] = []
    for index, window in enumerate(windows, start=1):
        window_name = f"window_{index:02d}_{int(window['start']):05d}_{int(window['end']):05d}"
        prepared_window_dir = prepared_root / window_name
        clips_dir = prepared_window_dir / "clips"
        mixed_path = prepared_window_dir / "mixed.wav"
        reference_out_dir = prepared_window_dir / "reference"
        raw_out_dir = prepared_window_dir / "raw_predicted"
        predicted_out_dir = output_dir / "predicted" / window_name

        clip_paths: List[Path] = []
        clip_labels: Dict[str, str] = {}
        for stem_path in stem_paths:
            clip_path = clips_dir / f"{stem_path.stem}.wav"
            if not clip_path.exists():
                clip_audio(
                    stem_path,
                    clip_path,
                    start=float(window["start"]),
                    duration=window_seconds,
                )
            clip_paths.append(clip_path)
            clip_labels[clip_path.name] = choose_speaker(clip_path.name, mapping)

        if not mixed_path.exists():
            mix_audio_files(clip_paths, mixed_path)

        reference_jsonl = _prepare_reference_outputs(
            clips_dir=clips_dir,
            reference_out_dir=reference_out_dir,
            speaker_mapping_path=speaker_mapping_path,
            defaults=defaults,
            device_override=device_override,
            local_files_only_override=local_files_only_override,
        )
        raw_jsonl, label_embedding_path, segment_embedding_path = _prepare_predicted_cache(
            mixed_path=mixed_path,
            raw_out_dir=raw_out_dir,
            speaker_count=int(window["speaker_count"]),
            defaults=defaults,
            device_override=device_override,
            local_files_only_override=local_files_only_override,
        )

        raw_segments = load_jsonl_records(raw_jsonl)
        purity_path = prepared_window_dir / "diarization_purity.json"
        if purity_path.exists():
            diarization_purity = json.loads(purity_path.read_text(encoding="utf-8"))
        else:
            diarization_purity = compute_segment_purity(
                raw_segments,
                _load_clip_stem_arrays(clip_paths, clip_labels),
            )
            purity_path.write_text(json.dumps(diarization_purity, indent=2), encoding="utf-8")
        relabeled_segments, speaker_bank_summary = apply_profile_to_cached_segments(
            raw_segments,
            label_embeddings=_load_embedding_map(label_embedding_path),
            segment_embeddings=_load_cached_segment_embeddings(segment_embedding_path),
            speaker_bank=speaker_bank,
            speaker_bank_config=speaker_bank_config,
            segment_classifier=segment_classifier,
        )
        consolidated_pairs = consolidate([(str(mixed_path), relabeled_segments)])
        final_predicted_dir = save_outputs(
            base_stem=mixed_path.stem,
            output_dir=str(_resolve_output_dir(mixed_path, predicted_out_dir)),
            per_file_segments=[(str(mixed_path), relabeled_segments)],
            consolidated_pairs=consolidated_pairs,
            diar_by_file=None,
            write_srt_file=False,
            write_jsonl_file=True,
        )
        predicted_jsonl = final_predicted_dir / f"{mixed_path.stem}.jsonl"
        reference_words = extract_words_from_jsonl(reference_jsonl)
        predicted_words = extract_words_from_jsonl(predicted_jsonl)
        metrics = score_word_speaker_alignment(reference_words, predicted_words)

        result = {
            "window": window,
            "name": window_name,
            "reference_jsonl": str(reference_jsonl),
            "predicted_jsonl": str(predicted_jsonl),
            "mixed_audio": str(mixed_path),
            "stem_labels": clip_labels,
            "reference_word_count": len(reference_words),
            "predicted_word_count": len(predicted_words),
            "metrics": metrics,
            "speaker_bank": speaker_bank_summary,
            "diarization_purity": {
                key: value for key, value in diarization_purity.items() if key != "records"
            },
            "diarization_purity_path": str(purity_path),
        }
        results.append(result)

    summary = {
        "session_zip": str(session_zip),
        "session_jsonl": str(session_jsonl),
        "speaker_mapping": str(speaker_mapping_path),
        "cache_root": str(prepared_root),
        "windows": windows,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate flattened multitrack diarization at word level."
    )
    parser.add_argument("--session-zip", required=True, help="Path to the multitrack session ZIP.")
    parser.add_argument(
        "--session-jsonl",
        required=True,
        help="Path to the existing labeled session JSONL used for chunk selection.",
    )
    parser.add_argument("--speaker-mapping", required=True, help="YAML/JSON speaker mapping file.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation artifacts.")
    parser.add_argument(
        "--config", help="Optional transcriber config file for model/cache defaults."
    )
    parser.add_argument(
        "--window-seconds", type=float, default=300.0, help="Window size in seconds."
    )
    parser.add_argument(
        "--hop-seconds", type=float, default=60.0, help="Window search hop in seconds."
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of non-overlapping windows to keep."
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=3,
        help="Minimum unique speakers required for a candidate window.",
    )
    parser.add_argument("--device", help="Override device for transcription runs.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force offline mode regardless of config.",
    )
    args = parser.parse_args()

    summary = evaluate_multitrack_session(
        session_zip=Path(args.session_zip),
        session_jsonl=Path(args.session_jsonl),
        output_dir=Path(args.output_dir),
        cache_root=None,
        speaker_mapping_path=Path(args.speaker_mapping),
        config_path=Path(args.config) if args.config else None,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        top_k=args.top_k,
        min_speakers=args.min_speakers,
        device_override=args.device,
        local_files_only_override=True if args.local_files_only else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
