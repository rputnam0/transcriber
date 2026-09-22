from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_tsvad_word_owner_baseline import (
    StemCache,
    _group_manifest_rows,
    _interval_indices,
    _load_group_mixture_features,
    _load_reference_groups,
    _mean,
    _read_jsonl,
    _reference_words_for_training,
    _row_has_audio,
    _select_device,
    _speaker_profile_for_row,
    _split_values,
    _standardize_train,
    _word_has_overlap,
    _write_jsonl,
)


class _WordOwnerMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _speaker_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return sorted(rows, key=lambda item: str(item.get("speaker_id") or ""))


def _interval_feature(
    mixture_features: np.ndarray,
    frame_centers: np.ndarray,
    item: Mapping[str, object],
) -> np.ndarray:
    start = float(item.get("start") or 0.0)
    end = float(item.get("end") or start)
    indices = _interval_indices(frame_centers, start, end)
    chunk = mixture_features[indices]
    return np.concatenate(
        [
            chunk.mean(axis=0),
            chunk.std(axis=0),
            chunk.max(axis=0),
        ]
    ).astype(np.float32, copy=False)


def _candidate_features(
    interval: np.ndarray, profile: np.ndarray, *, duration: float
) -> np.ndarray:
    bins = profile.shape[0] // 2
    interval_mean = interval[:bins]
    interval_std = interval[bins : 2 * bins]
    interval_max = interval[2 * bins : 3 * bins]
    profile_mean = profile[:bins]
    profile_std = profile[bins : 2 * bins]
    scalar = np.asarray(
        [
            max(duration, 0.0),
            np.log1p(max(duration, 0.0)),
            float(np.mean(np.abs(interval_mean - profile_mean))),
            float(
                np.dot(interval_mean, profile_mean)
                / (
                    max(float(np.linalg.norm(interval_mean)), 1e-6)
                    * max(float(np.linalg.norm(profile_mean)), 1e-6)
                )
            ),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            interval_mean,
            interval_std,
            interval_max,
            profile_mean,
            profile_std,
            np.abs(interval_mean - profile_mean),
            interval_mean * profile_mean,
            scalar,
        ]
    ).astype(np.float32, copy=False)


def _group_context(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    max_enrollment_seconds: float,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    mixture_features, frame_centers = _load_group_mixture_features(
        rows,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        target_rate=target_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        feature_bins=feature_bins,
    )
    profiles = {}
    for row in _speaker_rows(rows):
        speaker = str(row.get("speaker_id") or "")
        profiles[speaker] = _speaker_profile_for_row(
            row,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            max_seconds=max_enrollment_seconds,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
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
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
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
        if not items:
            continue
        mixture_features, frame_centers, profiles, speakers = _group_context(
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            max_enrollment_seconds=max_enrollment_seconds,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
        )
        usable_items = [item for item in items if str(item.get("speaker") or "") in profiles]
        if max_train_items > 0 and items_seen + len(usable_items) > max_train_items:
            remaining = max_train_items - items_seen
            usable_items = rng.sample(usable_items, max(0, remaining))
        for item in usable_items:
            reference = str(item.get("speaker") or "")
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
        raise ValueError("No direct word-owner training examples were built")
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


def _fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[_WordOwnerMlp, np.ndarray, np.ndarray, list[dict]]:
    torch.manual_seed(seed)
    standardized, mean, std = _standardize_train(features)
    dataset = TensorDataset(
        torch.from_numpy(standardized),
        torch.from_numpy(labels.astype(np.float32, copy=False)),
        torch.from_numpy(weights.astype(np.float32, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = _WordOwnerMlp(features.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positive_weight = float(weights[labels > 0.5].sum())
    negative_weight = float(weights[labels <= 0.5].sum())
    pos_weight = torch.tensor(
        [max(1.0, negative_weight / max(positive_weight, 1.0))],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        total = 0
        correct = 0
        for batch_features, batch_labels, batch_weights in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss_values = criterion(logits, batch_labels) * batch_weights
            loss = loss_values.sum() / torch.clamp(batch_weights.sum(), min=1.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss_values.sum().item())
            total_weight += float(batch_weights.sum().item())
            total += int(batch_labels.numel())
            correct += int(((torch.sigmoid(logits) >= 0.5) == (batch_labels >= 0.5)).sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total_weight if total_weight else 0.0,
                "candidate_binary_accuracy": correct / total if total else 0.0,
            }
        )
    model.eval()
    return model, mean, std, history


def _predict_scores(
    model: _WordOwnerMlp,
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    batch = ((features - mean) / std).astype(np.float32, copy=False)
    with torch.inference_mode():
        logits = model(torch.from_numpy(batch).to(device))
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)


def _score_direct_word_owners(
    words: Sequence[Mapping[str, object]],
    speakers: Sequence[str],
    scores_by_word: Sequence[Mapping[str, float]],
) -> tuple[dict, list[dict]]:
    reference_words = 0
    scored_words = 0
    correct_words = 0
    skipped_missing_speaker_words = 0
    overlap_words = 0
    overlap_correct_words = 0
    non_overlap_words = 0
    non_overlap_correct_words = 0
    reference_margins = []
    top_margins = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    speaker_set = set(speakers)
    word_records = []

    for index, (word, scores) in enumerate(zip(words, scores_by_word, strict=False)):
        reference = str(word.get("speaker") or "")
        if not reference:
            continue
        reference_words += 1
        if reference not in speaker_set:
            skipped_missing_speaker_words += 1
            continue
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predicted = ordered[0][0]
        top_margin = ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0)
        best_non_reference = max(
            (score for speaker, score in scores.items() if speaker != reference),
            default=0.0,
        )
        reference_margin = scores.get(reference, 0.0) - best_non_reference
        has_overlap = _word_has_overlap(word, words)
        is_correct = predicted == reference

        scored_words += 1
        correct_words += int(is_correct)
        reference_margins.append(float(reference_margin))
        top_margins.append(float(top_margin))
        confusion[reference][predicted] += 1
        by_speaker[reference]["words"] += 1
        by_speaker[reference]["correct"] += int(is_correct)
        if has_overlap:
            overlap_words += 1
            overlap_correct_words += int(is_correct)
        else:
            non_overlap_words += 1
            non_overlap_correct_words += int(is_correct)

        word_records.append(
            {
                "word_index": index,
                "text": word.get("text"),
                "speaker": reference,
                "predicted": predicted,
                "correct": is_correct,
                "start": word.get("start"),
                "end": word.get("end"),
                "overlap": has_overlap,
                "reference_score": scores.get(reference),
                "predicted_score": scores[predicted],
                "reference_margin": reference_margin,
                "top_margin": top_margin,
                "scores": dict(scores),
            }
        )

    result = {
        "speakers": list(speakers),
        "reference_words": reference_words,
        "scored_words": scored_words,
        "skipped_missing_speaker_words": skipped_missing_speaker_words,
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
        "confusion": {speaker: dict(counts) for speaker, counts in sorted(confusion.items())},
        "by_speaker": {
            speaker: {
                "words": int(counts["words"]),
                "correct": int(counts["correct"]),
                "accuracy": counts["correct"] / counts["words"] if counts["words"] else 0.0,
            }
            for speaker, counts in sorted(by_speaker.items())
        },
    }
    return result, word_records


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
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
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
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            max_enrollment_seconds=max_enrollment_seconds,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
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

        result, group_word_records = _score_direct_word_owners(
            words,
            speakers,
            scores_by_word,
        )
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
        description="Train a direct enrollment-conditioned word-owner baseline."
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
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-seconds", type=float, default=0.02)
    parser.add_argument("--feature-bins", type=int, default=96)
    parser.add_argument("--max-enrollment-seconds", type=float, default=45.0)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    manifest_dir = args.manifest.resolve().parent
    hop_length = int(round(float(args.hop_seconds) * int(args.sample_rate)))
    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped_rows = _group_manifest_rows(_read_jsonl(args.manifest))
    train_splits = _split_values(args.train_splits)
    eval_splits = _split_values(args.eval_splits)
    train_sessions = _split_values(args.train_sessions or "")
    eval_sessions = _split_values(args.eval_sessions or "")
    device = _select_device(args.device)
    stem_cache = (
        StemCache(root=args.stems_cache_root or args.output_dir / "_stems")
        if args.include_nonmaterialized
        else None
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
        target_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=hop_length,
        feature_bins=int(args.feature_bins),
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
        target_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=hop_length,
        feature_bins=int(args.feature_bins),
        device=device,
        save_word_records=bool(args.save_word_records),
    )
    summary = _summarize_groups(group_results)
    summary.update(
        {
            "model": "direct_enrollment_conditioned_word_owner_mlp",
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
            "sample_rate": int(args.sample_rate),
            "n_fft": int(args.n_fft),
            "hop_seconds": float(args.hop_seconds),
            "feature_bins": int(args.feature_bins),
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
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "direct_word_owner_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "direct_word_owner_words.jsonl", word_records)
    (args.output_dir / "direct_word_owner_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "direct_word_owner_training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
