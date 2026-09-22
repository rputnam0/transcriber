from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_direct_word_owner_baseline import (  # noqa: E402
    _WordOwnerMlp,
    _candidate_features,
    _fit_model,
    _interval_feature,
    _predict_scores,
    _score_direct_word_owners,
)
from train_tsvad_word_owner_baseline import (  # noqa: E402
    StemCache,
    _group_manifest_rows,
    _load_audio_window,
    _load_mono,
    _load_reference_groups,
    _materialized_path,
    _mean,
    _positive_enrollment_paths,
    _read_jsonl,
    _reference_words_for_training,
    _resample_if_needed,
    _row_has_audio,
    _select_device,
    _split_values,
    _write_jsonl,
)


def _safe_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _hash_payload(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:20]


def _projection_matrix(
    *,
    input_dim: int,
    output_dim: int,
    seed: int,
) -> np.ndarray | None:
    if output_dim <= 0 or output_dim >= input_dim:
        return None
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(output_dim), size=(input_dim, output_dim))
    return projection.astype(np.float32, copy=False)


def _apply_projection(features: np.ndarray, projection: np.ndarray | None) -> np.ndarray:
    if projection is None:
        return features.astype(np.float32, copy=False)
    return (features.astype(np.float32, copy=False) @ projection).astype(np.float32, copy=False)


def _frame_times(start_seconds: float, duration_seconds: float, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    step = float(duration_seconds) / max(frame_count, 1)
    return (float(start_seconds) + (np.arange(frame_count, dtype=np.float32) + 0.5) * step).astype(
        np.float32,
        copy=False,
    )


class WavlmEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        layer: int,
        device: torch.device,
        sample_rate: int,
        chunk_seconds: float,
        projection_dim: int,
        projection_seed: int,
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = str(model_name)
        self.layer = int(layer)
        self.device = device
        self.sample_rate = int(sample_rate)
        self.chunk_seconds = float(chunk_seconds)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(f"Could not determine hidden size for {self.model_name}")
        self.projection = _projection_matrix(
            input_dim=hidden_size,
            output_dim=int(projection_dim),
            seed=int(projection_seed),
        )
        self.output_dim = (
            int(self.projection.shape[1]) if self.projection is not None else hidden_size
        )

    @property
    def identity(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "layer": self.layer,
            "sample_rate": self.sample_rate,
            "chunk_seconds": self.chunk_seconds,
            "projection_dim": self.output_dim,
        }

    def encode_wave(self, wave: np.ndarray, *, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        wave = _resample_if_needed(wave, source_rate=int(sample_rate), target_rate=self.sample_rate)
        wave = np.nan_to_num(wave.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        chunk_samples = max(1, int(round(self.chunk_seconds * self.sample_rate)))
        feature_chunks: list[np.ndarray] = []
        time_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, max(wave.shape[0], 1), chunk_samples):
                chunk = wave[start : start + chunk_samples]
                if chunk.size == 0:
                    continue
                duration = chunk.shape[0] / float(self.sample_rate)
                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=False,
                )
                input_values = inputs["input_values"].to(self.device)
                outputs = self.model(input_values, output_hidden_states=self.layer != -1)
                if self.layer == -1:
                    hidden = outputs.last_hidden_state
                else:
                    hidden_states = outputs.hidden_states
                    layer = self.layer if self.layer >= 0 else len(hidden_states) + self.layer
                    hidden = hidden_states[layer]
                features = hidden.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                features = _apply_projection(features, self.projection)
                feature_chunks.append(features)
                time_chunks.append(
                    _frame_times(
                        start / float(self.sample_rate),
                        duration,
                        features.shape[0],
                    )
                )
        if not feature_chunks:
            return np.zeros((0, self.output_dim), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return (
            np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False),
            np.concatenate(time_chunks, axis=0).astype(np.float32, copy=False),
        )


def _load_group_mixture_wave(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    target_rate: int,
) -> tuple[np.ndarray, int]:
    path = _materialized_path(rows[0], "mixture_path", manifest_dir=manifest_dir)
    if path is not None and path.exists():
        return _load_mono(path)
    if stem_cache is None:
        raise ValueError("Group has no materialized mixture path and no stem cache")
    row = rows[0]
    member_waves = []
    for member in row.get("mixture_members") or []:
        member_path = stem_cache.path_for_member(row, member)
        wave_part, sample_rate_part = _load_audio_window(
            member_path,
            start_seconds=float(row.get("window_start") or 0.0),
            duration_seconds=float(row.get("duration") or 0.0),
        )
        member_waves.append(
            _resample_if_needed(
                wave_part,
                source_rate=sample_rate_part,
                target_rate=target_rate,
            )
        )
    if not member_waves:
        raise ValueError(f"Group {row.get('row_id')} has no mixture members")
    target_len = max(wave.shape[0] for wave in member_waves)
    padded = [
        np.pad(wave, (0, max(0, target_len - wave.shape[0])))[:target_len] for wave in member_waves
    ]
    return np.sum(np.stack(padded, axis=0), axis=0).astype(np.float32, copy=False), target_rate


def _cache_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    payload = np.load(path)
    return payload["features"].astype(np.float32), payload["frame_centers"].astype(np.float32)


def _load_or_encode_group_features(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    encoder: WavlmEncoder,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "kind": "mixture",
        "key": key,
        **encoder.identity,
    }
    cache_path = cache_dir / f"mixture_{_safe_id(key[0])}_{_hash_payload(cache_payload)}.npz"
    cached = _cache_npz(cache_path)
    if cached is not None:
        return cached
    wave, sample_rate = _load_group_mixture_wave(
        rows,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        target_rate=encoder.sample_rate,
    )
    features, frame_centers = encoder.encode_wave(wave, sample_rate=sample_rate)
    np.savez_compressed(cache_path, features=features, frame_centers=frame_centers)
    return features, frame_centers


def _profile_from_feature_chunks(chunks: Sequence[np.ndarray], *, output_dim: int) -> np.ndarray:
    usable = [chunk for chunk in chunks if chunk.size]
    if not usable:
        raise ValueError("No enrollment features available")
    merged = np.concatenate(usable, axis=0).astype(np.float32, copy=False)
    return np.concatenate([merged.mean(axis=0), merged.std(axis=0) + 1e-4]).astype(
        np.float32,
        copy=False,
    )[: output_dim * 2]


def _load_or_encode_profile(
    row: Mapping[str, object],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    encoder: WavlmEncoder,
    cache_dir: Path,
    max_seconds: float,
) -> np.ndarray:
    speaker = str(row.get("speaker_id") or "")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "kind": "profile",
        "row_id": row.get("row_id"),
        "speaker": speaker,
        "max_seconds": max_seconds,
        **encoder.identity,
    }
    cache_path = cache_dir / f"profile_{_safe_id(speaker)}_{_hash_payload(cache_payload)}.npz"
    if cache_path.exists():
        return np.load(cache_path)["profile"].astype(np.float32)

    paths = _positive_enrollment_paths(row, manifest_dir=manifest_dir)
    chunks: list[tuple[np.ndarray, int]] = []
    if paths:
        chunks = [_load_mono(path) for path in paths]
    else:
        if stem_cache is None or not row.get("target_member"):
            raise ValueError(f"No positive enrollment audio available for {row.get('row_id')}")
        member_path = stem_cache.path_for_member(row, row.get("target_member"))
        for span in row.get("positive_enrollment_spans") or []:
            item = dict(span)
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            if end <= start:
                continue
            chunks.append(
                _load_audio_window(
                    member_path,
                    start_seconds=start,
                    duration_seconds=end - start,
                )
            )
    feature_chunks = []
    remaining = float(max_seconds)
    for wave, sample_rate in chunks:
        if remaining > 0:
            limit = int(round(remaining * sample_rate))
            wave = wave[:limit]
            remaining -= wave.shape[0] / float(sample_rate)
        features, _ = encoder.encode_wave(wave, sample_rate=sample_rate)
        feature_chunks.append(features)
        if max_seconds > 0 and remaining <= 0:
            break
    profile = _profile_from_feature_chunks(feature_chunks, output_dim=encoder.output_dim)
    np.savez_compressed(cache_path, profile=profile)
    return profile


def _speaker_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(rows, key=lambda item: str(item.get("speaker_id") or ""))


def _group_context(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    encoder: WavlmEncoder,
    cache_dir: Path,
    max_enrollment_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    mixture_features, frame_centers = _load_or_encode_group_features(
        key,
        rows,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        encoder=encoder,
        cache_dir=cache_dir,
    )
    profiles = {}
    for row in _speaker_rows(rows):
        speaker = str(row.get("speaker_id") or "")
        profiles[speaker] = _load_or_encode_profile(
            row,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            encoder=encoder,
            cache_dir=cache_dir,
            max_seconds=max_enrollment_seconds,
        )
    return mixture_features, frame_centers, profiles, sorted(profiles)


def _reference_items(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    source: str,
) -> tuple[list[dict], str]:
    words, resolved_source = _reference_words_for_training(
        key,
        rows,
        reference_groups,
        source=source,
    )
    return [dict(word) for word in words], resolved_source


def _build_training_examples(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    include_nonmaterialized: bool,
    train_reference_source: str,
    train_splits: set[str],
    train_sessions: set[str],
    max_train_items: int,
    max_enrollment_seconds: float,
    encoder: WavlmEncoder,
    cache_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = random.Random(seed)
    feature_chunks = []
    label_chunks = []
    weight_chunks = []
    item_summaries = []
    items_seen = 0

    for key, all_rows in sorted(grouped_rows.items()):
        rows = [
            row
            for row in all_rows
            if str(row.get("split_id") or "") in train_splits
            and (not train_sessions or str(row.get("session") or "") in train_sessions)
            and _row_has_audio(row, include_nonmaterialized=include_nonmaterialized)
        ]
        if not rows:
            continue
        items, reference_source = _reference_items(
            key,
            rows,
            reference_groups,
            source=train_reference_source,
        )
        usable_items = [item for item in items if str(item.get("speaker") or "")]
        if not usable_items:
            continue
        if max_train_items > 0 and items_seen + len(usable_items) > max_train_items:
            remaining = max_train_items - items_seen
            usable_items = rng.sample(usable_items, max(0, remaining))
        if not usable_items:
            break
        mixture_features, frame_centers, profiles, speakers = _group_context(
            key,
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            encoder=encoder,
            cache_dir=cache_dir,
            max_enrollment_seconds=max_enrollment_seconds,
        )
        for item in usable_items:
            reference = str(item.get("speaker") or "")
            if reference not in profiles:
                continue
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            interval = _interval_feature(mixture_features, frame_centers, item)
            duration = abs(end - start)
            weight = max(1.0, float(item.get("word_count") or 1.0))
            for speaker in speakers:
                feature_chunks.append(
                    _candidate_features(interval, profiles[speaker], duration=duration)
                )
                label_chunks.append(1.0 if speaker == reference else 0.0)
                weight_chunks.append(weight)
            items_seen += 1
        item_summaries.append(
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "items": len(usable_items),
                "candidate_examples": len(usable_items) * len(speakers),
                "speakers": speakers,
                "reference_source": reference_source,
            }
        )
        if max_train_items > 0 and items_seen >= max_train_items:
            break

    if not feature_chunks:
        raise ValueError("No WavLM word-owner training examples were built")
    features = np.stack(feature_chunks, axis=0).astype(np.float32, copy=False)
    labels = np.asarray(label_chunks, dtype=np.float32)
    weights = np.asarray(weight_chunks, dtype=np.float32)
    return (
        features,
        labels,
        weights,
        {
            "train_splits": sorted(train_splits),
            "train_sessions": sorted(train_sessions),
            "train_reference_source": train_reference_source,
            "train_items": int(items_seen),
            "candidate_examples": int(features.shape[0]),
            "positive_examples": int(labels.sum()),
            "weighted_positive_examples": float(weights[labels > 0.5].sum()),
            "weighted_negative_examples": float(weights[labels <= 0.5].sum()),
            "item_summaries": item_summaries,
        },
    )


def _evaluate(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    model: _WordOwnerMlp,
    mean: np.ndarray,
    std: np.ndarray,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    include_nonmaterialized: bool,
    eval_splits: set[str],
    eval_sessions: set[str],
    max_enrollment_seconds: float,
    encoder: WavlmEncoder,
    cache_dir: Path,
    device: torch.device,
    save_word_records: bool,
) -> tuple[list[dict], list[dict]]:
    group_results = []
    word_records = []
    for key, all_rows in sorted(grouped_rows.items()):
        rows = [
            row
            for row in all_rows
            if str(row.get("split_id") or "") in eval_splits
            and (not eval_sessions or str(row.get("session") or "") in eval_sessions)
            and _row_has_audio(row, include_nonmaterialized=include_nonmaterialized)
        ]
        if not rows:
            continue
        if key not in reference_groups:
            group_results.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_reference_group",
                }
            )
            continue
        mixture_features, frame_centers, profiles, speakers = _group_context(
            key,
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            encoder=encoder,
            cache_dir=cache_dir,
            max_enrollment_seconds=max_enrollment_seconds,
        )
        words = [dict(word) for word in reference_groups[key]]
        scores_by_word = []
        for word in words:
            start = float(word.get("start") or 0.0)
            end = float(word.get("end") or start)
            duration = abs(end - start)
            interval = _interval_feature(mixture_features, frame_centers, word)
            candidate_matrix = np.stack(
                [
                    _candidate_features(interval, profiles[speaker], duration=duration)
                    for speaker in speakers
                ],
                axis=0,
            )
            scores = _predict_scores(model, candidate_matrix, mean=mean, std=std, device=device)
            scores_by_word.append(
                {speaker: float(score) for speaker, score in zip(speakers, scores)}
            )
        result, group_word_records = _score_direct_word_owners(words, speakers, scores_by_word)
        result.update(
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "split_id": ",".join(sorted({str(row.get("split_id") or "") for row in rows})),
                "row_count": len(rows),
            }
        )
        group_results.append(result)
        if save_word_records:
            for record in group_word_records:
                record.update(
                    {
                        "session": key[0],
                        "window_start": key[1],
                        "window_end": key[2],
                        "split_id": result["split_id"],
                    }
                )
                word_records.append(record)
    return group_results, word_records


def _sum_confusion(groups: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for reference, counts in dict(group.get("confusion") or {}).items():
            for predicted, count in dict(counts).items():
                confusion[str(reference)][str(predicted)] += int(count)
    return {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}


def _merge_by_speaker(groups: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    merged: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for speaker, counts in dict(group.get("by_speaker") or {}).items():
            merged[str(speaker)]["words"] += int(dict(counts).get("words") or 0)
            merged[str(speaker)]["correct"] += int(dict(counts).get("correct") or 0)
    return {
        speaker: {
            "words": int(counts["words"]),
            "correct": int(counts["correct"]),
            "accuracy": counts["correct"] / counts["words"] if counts["words"] else 0.0,
        }
        for speaker, counts in sorted(merged.items())
    }


def _summarize_groups(groups: Sequence[Mapping[str, object]]) -> dict:
    valid = [group for group in groups if not group.get("error")]
    reference_words = sum(int(group.get("reference_words") or 0) for group in valid)
    scored_words = sum(int(group.get("scored_words") or 0) for group in valid)
    correct_words = sum(int(group.get("correct_words") or 0) for group in valid)
    overlap_words = sum(int(group.get("overlap_words") or 0) for group in valid)
    overlap_correct_words = sum(int(group.get("overlap_correct_words") or 0) for group in valid)
    non_overlap_words = sum(int(group.get("non_overlap_words") or 0) for group in valid)
    non_overlap_correct_words = sum(
        int(group.get("non_overlap_correct_words") or 0) for group in valid
    )
    reference_margins = [
        float(group["mean_reference_margin"])
        for group in valid
        if group.get("mean_reference_margin") is not None
    ]
    top_margins = [
        float(group["mean_top_margin"])
        for group in valid
        if group.get("mean_top_margin") is not None
    ]
    by_session: dict[str, Counter[str]] = defaultdict(Counter)
    for group in valid:
        key = str(group.get("session") or "")
        by_session[key]["reference_words"] += int(group.get("reference_words") or 0)
        by_session[key]["scored_words"] += int(group.get("scored_words") or 0)
        by_session[key]["correct_words"] += int(group.get("correct_words") or 0)
    return {
        "group_count": len(groups),
        "valid_groups": len(valid),
        "reference_words": reference_words,
        "scored_words": scored_words,
        "correct_words": correct_words,
        "coverage": scored_words / reference_words if reference_words else 0.0,
        "accuracy": correct_words / reference_words if reference_words else 0.0,
        "scored_accuracy": correct_words / scored_words if scored_words else 0.0,
        "overlap_words": overlap_words,
        "overlap_correct_words": overlap_correct_words,
        "overlap_accuracy": overlap_correct_words / overlap_words if overlap_words else None,
        "non_overlap_words": non_overlap_words,
        "non_overlap_correct_words": non_overlap_correct_words,
        "non_overlap_accuracy": (
            non_overlap_correct_words / non_overlap_words if non_overlap_words else None
        ),
        "mean_reference_margin": _mean(reference_margins),
        "mean_top_margin": _mean(top_margins),
        "confusion": _sum_confusion(valid),
        "by_speaker": _merge_by_speaker(valid),
        "by_session": {
            key: {
                "reference_words": int(counts["reference_words"]),
                "scored_words": int(counts["scored_words"]),
                "correct_words": int(counts["correct_words"]),
                "accuracy": (
                    counts["correct_words"] / counts["reference_words"]
                    if counts["reference_words"]
                    else 0.0
                ),
                "scored_accuracy": (
                    counts["correct_words"] / counts["scored_words"]
                    if counts["scored_words"]
                    else 0.0
                ),
            }
            for key, counts in sorted(by_session.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen-WavLM enrollment-conditioned word-owner baseline."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-splits", default="train,dev")
    parser.add_argument("--eval-splits", default="test")
    parser.add_argument("--train-sessions")
    parser.add_argument("--eval-sessions")
    parser.add_argument(
        "--train-reference-source",
        choices=("auto", "forced", "manifest"),
        default="auto",
    )
    parser.add_argument("--include-nonmaterialized", action="store_true")
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--wavlm-layer", type=int, default=-1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--wavlm-chunk-seconds", type=float, default=20.0)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--projection-seed", type=int, default=20260601)
    parser.add_argument("--max-enrollment-seconds", type=float, default=20.0)
    parser.add_argument("--max-train-items", type=int, default=4000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_dir = args.manifest.resolve().parent
    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped_rows = _group_manifest_rows(_read_jsonl(args.manifest))
    train_splits = _split_values(args.train_splits)
    eval_splits = _split_values(args.eval_splits)
    train_sessions = _split_values(args.train_sessions or "")
    eval_sessions = _split_values(args.eval_sessions or "")
    device = _select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or args.output_dir / "wavlm_feature_cache"
    stem_cache = (
        StemCache(root=args.stems_cache_root or args.output_dir / "_stems")
        if args.include_nonmaterialized
        else None
    )
    encoder = WavlmEncoder(
        model_name=str(args.wavlm_model),
        layer=int(args.wavlm_layer),
        device=device,
        sample_rate=int(args.sample_rate),
        chunk_seconds=float(args.wavlm_chunk_seconds),
        projection_dim=int(args.projection_dim),
        projection_seed=int(args.projection_seed),
    )
    features, labels, weights, training_data_summary = _build_training_examples(
        grouped_rows,
        reference_groups,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        include_nonmaterialized=bool(args.include_nonmaterialized),
        train_reference_source=str(args.train_reference_source),
        train_splits=train_splits,
        train_sessions=train_sessions,
        max_train_items=int(args.max_train_items),
        max_enrollment_seconds=float(args.max_enrollment_seconds),
        encoder=encoder,
        cache_dir=cache_dir,
        seed=int(args.seed),
    )
    model, mean, std, history = _fit_model(
        features,
        labels,
        weights,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=device,
        seed=int(args.seed),
    )
    group_results, word_records = _evaluate(
        grouped_rows,
        reference_groups,
        model=model,
        mean=mean,
        std=std,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        include_nonmaterialized=bool(args.include_nonmaterialized),
        eval_splits=eval_splits,
        eval_sessions=eval_sessions,
        max_enrollment_seconds=float(args.max_enrollment_seconds),
        encoder=encoder,
        cache_dir=cache_dir,
        device=device,
        save_word_records=bool(args.save_word_records),
    )
    summary = _summarize_groups(group_results)
    summary.update(
        {
            "model": "wavlm_enrollment_conditioned_word_owner_mlp",
            "manifest": str(args.manifest),
            "reference_jsonl": str(args.reference_jsonl),
            "train_splits": sorted(train_splits),
            "eval_splits": sorted(eval_splits),
            "train_sessions": sorted(train_sessions),
            "eval_sessions": sorted(eval_sessions),
            "train_reference_source": str(args.train_reference_source),
            "include_nonmaterialized": bool(args.include_nonmaterialized),
            "stems_cache_root": (
                str(args.stems_cache_root or args.output_dir / "_stems")
                if args.include_nonmaterialized
                else None
            ),
            "feature_cache_dir": str(cache_dir),
            "wavlm_model": str(args.wavlm_model),
            "wavlm_layer": int(args.wavlm_layer),
            "sample_rate": int(args.sample_rate),
            "wavlm_chunk_seconds": float(args.wavlm_chunk_seconds),
            "projection_dim": encoder.output_dim,
            "projection_seed": int(args.projection_seed),
            "max_enrollment_seconds": float(args.max_enrollment_seconds),
            "max_train_items": int(args.max_train_items),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "device": str(device),
            "seed": int(args.seed),
            "elapsed_seconds": time.time() - start_time,
        }
    )
    training_summary = {
        **training_data_summary,
        "history": history,
        "feature_dim": int(features.shape[1]),
        "wavlm_output_dim": int(encoder.output_dim),
    }
    _write_jsonl(args.output_dir / "wavlm_word_owner_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "wavlm_word_owner_words.jsonl", word_records)
    (args.output_dir / "wavlm_word_owner_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "wavlm_word_owner_training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
