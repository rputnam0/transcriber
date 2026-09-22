from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_direct_word_owner_baseline import (  # noqa: E402
    _candidate_features,
    _group_context,
    _interval_feature,
    _reference_items,
    _score_direct_word_owners,
)
from train_tsvad_word_owner_baseline import (  # noqa: E402
    StemCache,
    _group_manifest_rows,
    _load_reference_groups,
    _mean,
    _read_jsonl,
    _row_has_audio,
    _select_device,
    _split_values,
    _write_jsonl,
)


class _CandidateScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(64, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        logits = self.net(flat).reshape(shape[:-1])
        return logits


def _word_weight(item: Mapping[str, object], *, min_score_weight: float) -> float:
    score = item.get("score")
    if score is None:
        return 1.0
    return max(float(min_score_weight), float(score))


def _pad_candidate_items(
    item_features: Sequence[np.ndarray],
    label_indices: Sequence[int],
    weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not item_features:
        raise ValueError("No listwise candidate items to pad")
    max_candidates = max(matrix.shape[0] for matrix in item_features)
    feature_dim = int(item_features[0].shape[1])
    features = np.zeros((len(item_features), max_candidates, feature_dim), dtype=np.float32)
    mask = np.zeros((len(item_features), max_candidates), dtype=bool)
    for index, matrix in enumerate(item_features):
        candidates = matrix.shape[0]
        features[index, :candidates] = matrix
        mask[index, :candidates] = True
    return (
        features,
        mask,
        np.asarray(label_indices, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def _standardize_candidate_tensor(
    features: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = features[mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    standardized = ((features - mean) / std).astype(np.float32, copy=False)
    standardized[~mask] = 0.0
    return standardized, mean, std


def _standardize_eval_matrix(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32, copy=False)


def _build_training_items(
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
    min_score_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = random.Random(seed)
    feature_items: list[np.ndarray] = []
    label_indices = []
    weights = []
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
            duration = abs(end - start)
            interval = _interval_feature(mixture_features, frame_centers, item)
            feature_items.append(
                np.stack(
                    [
                        _candidate_features(interval, profiles[speaker], duration=duration)
                        for speaker in speakers
                    ],
                    axis=0,
                )
            )
            label_indices.append(int(speakers.index(reference)))
            weights.append(_word_weight(item, min_score_weight=min_score_weight))
            items_seen += 1
        item_summaries.append(
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "items": len(usable_items),
                "speakers": speakers,
                "reference_source": reference_source,
            }
        )
        if max_train_items > 0 and items_seen >= max_train_items:
            break

    features, mask, labels, weight_array = _pad_candidate_items(
        feature_items, label_indices, weights
    )
    return (
        features,
        mask,
        labels,
        weight_array,
        {
            "train_splits": sorted(train_splits),
            "train_sessions": sorted(train_sessions),
            "train_reference_source": train_reference_source,
            "train_items": int(features.shape[0]),
            "max_candidates": int(features.shape[1]),
            "feature_dim": int(features.shape[2]),
            "mean_item_weight": float(weight_array.mean()) if weight_array.size else 0.0,
            "item_summaries": item_summaries,
        },
    )


def _fit_model(
    features: np.ndarray,
    mask: np.ndarray,
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
) -> tuple[_CandidateScorer, np.ndarray, np.ndarray, list[dict]]:
    torch.manual_seed(seed)
    standardized, mean, std = _standardize_candidate_tensor(features, mask)
    dataset = TensorDataset(
        torch.from_numpy(standardized),
        torch.from_numpy(mask),
        torch.from_numpy(labels),
        torch.from_numpy(weights),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = _CandidateScorer(features.shape[-1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        total = 0
        correct = 0
        for batch_features, batch_mask, batch_labels, batch_weights in loader:
            batch_features = batch_features.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).masked_fill(~batch_mask, -1.0e9)
            loss_values = criterion(logits, batch_labels) * batch_weights
            loss = loss_values.sum() / torch.clamp(batch_weights.sum(), min=1.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss_values.detach().sum().cpu())
            total_weight += float(batch_weights.detach().sum().cpu())
            total += int(batch_labels.numel())
            correct += int((torch.argmax(logits, dim=1) == batch_labels).sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total_weight if total_weight else 0.0,
                "listwise_training_accuracy": correct / total if total else 0.0,
            }
        )
    model.eval()
    return model, mean, std, history


def _predict_logits(
    model: _CandidateScorer,
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    standardized = _standardize_eval_matrix(features, mean=mean, std=std)
    with torch.inference_mode():
        logits = model(torch.from_numpy(standardized[None, :, :]).to(device))
    return logits.detach().cpu().numpy()[0].astype(np.float32, copy=False)


def _evaluate(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    model: _CandidateScorer,
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
            logits = _predict_logits(model, candidate_matrix, mean=mean, std=std, device=device)
            scores_by_word.append(
                {speaker: float(score) for speaker, score in zip(speakers, logits)}
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
        description="Train a listwise enrollment-conditioned word-owner baseline."
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
    parser.add_argument("--min-score-weight", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    features, mask, labels, weights, training_data_summary = _build_training_items(
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
        min_score_weight=float(args.min_score_weight),
        seed=int(args.seed),
    )
    model, mean, std, history = _fit_model(
        features,
        mask,
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
            "model": "listwise_enrollment_conditioned_word_owner_mlp",
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
            "min_score_weight": float(args.min_score_weight),
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
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "listwise_word_owner_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "listwise_word_owner_words.jsonl", word_records)
    (args.output_dir / "listwise_word_owner_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "listwise_word_owner_training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
