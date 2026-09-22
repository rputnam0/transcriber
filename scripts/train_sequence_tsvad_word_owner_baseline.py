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

from train_tsvad_word_owner_baseline import (  # noqa: E402
    StemCache,
    _conditioned_features,
    _frame_labels,
    _group_manifest_rows,
    _load_group_mixture_features,
    _load_reference_groups,
    _mean,
    _read_jsonl,
    _reference_words_for_training,
    _row_has_audio,
    _score_word_owners,
    _select_device,
    _speaker_profile_for_row,
    _split_values,
    _standardize_train,
    _write_jsonl,
)


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class SequenceTsvadModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        channels: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, channels, kernel_size=1)
        blocks = []
        for index in range(layers):
            blocks.append(_TemporalBlock(channels, dilation=2 ** (index % 6), dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, frames, input_dim]
        hidden = self.input(features.transpose(1, 2))
        hidden = self.blocks(hidden)
        return self.output(hidden).squeeze(1)


def _sample_chunk_starts(
    labels: np.ndarray,
    *,
    chunk_frames: int,
    chunks_per_sequence: int,
    positive_probability: float,
    rng: random.Random,
) -> list[int]:
    total = int(labels.shape[0])
    if total <= 0:
        return []
    max_start = max(0, total - chunk_frames)
    positive = np.flatnonzero(labels > 0.5)
    starts = []
    for _ in range(chunks_per_sequence):
        if positive.size and rng.random() < positive_probability:
            center = int(rng.choice(positive))
            start = center - rng.randrange(0, max(1, chunk_frames))
            starts.append(min(max(start, 0), max_start))
        else:
            starts.append(rng.randrange(0, max_start + 1) if max_start else 0)
    return starts


def _slice_or_pad(array: np.ndarray, *, start: int, frames: int) -> np.ndarray:
    chunk = array[start : start + frames]
    if chunk.shape[0] >= frames:
        return chunk[:frames]
    pad_width = [(0, frames - chunk.shape[0])] + [(0, 0)] * (chunk.ndim - 1)
    return np.pad(chunk, pad_width).astype(array.dtype, copy=False)


def _build_training_sequences(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    include_nonmaterialized: bool,
    train_reference_source: str,
    train_splits: set[str],
    train_sessions: set[str],
    max_sequences: int,
    chunks_per_speaker: int,
    positive_probability: float,
    chunk_frames: int,
    max_enrollment_seconds: float,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = random.Random(seed)
    feature_chunks = []
    label_chunks = []
    row_summaries = []

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
        words, reference_source = _reference_words_for_training(
            key,
            rows,
            reference_groups,
            source=train_reference_source,
        )
        if not words:
            continue
        mixture_features, frame_centers = _load_group_mixture_features(
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
        )
        for row in sorted(rows, key=lambda item: str(item.get("speaker_id") or "")):
            speaker = str(row.get("speaker_id") or "")
            labels = _frame_labels(words, speaker, frame_centers).astype(np.float32)
            if labels.size == 0:
                continue
            profile = _speaker_profile_for_row(
                row,
                manifest_dir=manifest_dir,
                stem_cache=stem_cache,
                max_seconds=max_enrollment_seconds,
                target_rate=target_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                feature_bins=feature_bins,
            )
            conditioned = _conditioned_features(mixture_features, profile)
            starts = _sample_chunk_starts(
                labels,
                chunk_frames=chunk_frames,
                chunks_per_sequence=chunks_per_speaker,
                positive_probability=positive_probability,
                rng=rng,
            )
            for start in starts:
                feature_chunks.append(_slice_or_pad(conditioned, start=start, frames=chunk_frames))
                label_chunks.append(
                    _slice_or_pad(labels[:, None], start=start, frames=chunk_frames)[:, 0]
                )
            row_summaries.append(
                {
                    "row_id": row.get("row_id"),
                    "session": row.get("session"),
                    "speaker": speaker,
                    "chunks": len(starts),
                    "positive_frames": int(labels.sum()),
                    "total_frames": int(labels.shape[0]),
                    "reference_source": reference_source,
                }
            )
            if max_sequences > 0 and len(feature_chunks) >= max_sequences:
                break
        if max_sequences > 0 and len(feature_chunks) >= max_sequences:
            break

    if not feature_chunks:
        raise ValueError("No sequence TS-VAD training examples were built")
    features = np.stack(feature_chunks, axis=0).astype(np.float32, copy=False)
    labels = np.stack(label_chunks, axis=0).astype(np.float32, copy=False)
    if max_sequences > 0 and features.shape[0] > max_sequences:
        indices = np.asarray(rng.sample(range(features.shape[0]), max_sequences), dtype=np.int64)
        features = features[indices]
        labels = labels[indices]
    return (
        features,
        labels,
        {
            "train_splits": sorted(train_splits),
            "train_sessions": sorted(train_sessions),
            "train_reference_source": train_reference_source,
            "sequences": int(features.shape[0]),
            "chunk_frames": int(chunk_frames),
            "positive_frames": int(labels.sum()),
            "negative_frames": int((labels <= 0.5).sum()),
            "row_summaries": row_summaries,
        },
    )


def _standardize_sequences(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    standardized, mean, std = _standardize_train(flat)
    return standardized.reshape(features.shape).astype(np.float32, copy=False), mean, std


def _fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    channels: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[SequenceTsvadModel, np.ndarray, np.ndarray, list[dict]]:
    torch.manual_seed(seed)
    standardized, mean, std = _standardize_sequences(features)
    dataset = TensorDataset(
        torch.from_numpy(standardized),
        torch.from_numpy(labels.astype(np.float32, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = SequenceTsvadModel(
        features.shape[-1],
        channels=channels,
        layers=layers,
        dropout=dropout,
    ).to(device)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_labels.numel())
            total += int(batch_labels.numel())
            correct += int(((torch.sigmoid(logits) >= 0.5) == (batch_labels >= 0.5)).sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total if total else 0.0,
                "frame_binary_accuracy": correct / total if total else 0.0,
            }
        )
    model.eval()
    return model, mean, std, history


def _predict_full_sequence(
    model: SequenceTsvadModel,
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    chunk_frames: int,
    hop_frames: int,
    device: torch.device,
) -> np.ndarray:
    standardized = ((features - mean) / std).astype(np.float32, copy=False)
    if standardized.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    chunk_frames = max(1, int(chunk_frames))
    hop_frames = max(1, int(hop_frames))
    output = np.zeros(standardized.shape[0], dtype=np.float32)
    weights = np.zeros(standardized.shape[0], dtype=np.float32)
    with torch.inference_mode():
        start = 0
        while start < standardized.shape[0]:
            chunk = _slice_or_pad(standardized, start=start, frames=chunk_frames)
            logits = model(torch.from_numpy(chunk[None, :, :]).to(device))
            posterior = torch.sigmoid(logits).detach().cpu().numpy()[0]
            stop = min(start + chunk_frames, standardized.shape[0])
            valid = stop - start
            output[start:stop] += posterior[:valid]
            weights[start:stop] += 1.0
            if stop >= standardized.shape[0]:
                break
            start += hop_frames
    return output / np.maximum(weights, 1e-6)


def _evaluate(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    model: SequenceTsvadModel,
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
    chunk_frames: int,
    eval_hop_frames: int,
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
        mixture_features, frame_centers = _load_group_mixture_features(
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
        )
        posteriors = {}
        speakers = []
        for row in sorted(rows, key=lambda item: str(item.get("speaker_id") or "")):
            speaker = str(row.get("speaker_id") or "")
            speakers.append(speaker)
            profile = _speaker_profile_for_row(
                row,
                manifest_dir=manifest_dir,
                stem_cache=stem_cache,
                max_seconds=max_enrollment_seconds,
                target_rate=target_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                feature_bins=feature_bins,
            )
            conditioned = _conditioned_features(mixture_features, profile)
            posteriors[speaker] = _predict_full_sequence(
                model,
                conditioned,
                mean=mean,
                std=std,
                chunk_frames=chunk_frames,
                hop_frames=eval_hop_frames,
                device=device,
            )
        words = [dict(word) for word in reference_groups[key]]
        result, group_word_records = _score_word_owners(
            words=words,
            speakers=speakers,
            frame_centers=frame_centers,
            speaker_posteriors=posteriors,
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
        description="Train a temporal enrollment-conditioned TS-VAD word-owner baseline."
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
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--eval-hop-seconds", type=float, default=2.0)
    parser.add_argument("--chunks-per-speaker", type=int, default=12)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--positive-probability", type=float, default=0.75)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_dir = args.manifest.resolve().parent
    hop_length = int(round(float(args.hop_seconds) * int(args.sample_rate)))
    chunk_frames = max(1, int(round(float(args.chunk_seconds) / float(args.hop_seconds))))
    eval_hop_frames = max(1, int(round(float(args.eval_hop_seconds) / float(args.hop_seconds))))
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
    features, labels, training_data_summary = _build_training_sequences(
        grouped_rows,
        reference_groups,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        include_nonmaterialized=bool(args.include_nonmaterialized),
        train_reference_source=str(args.train_reference_source),
        train_splits=train_splits,
        train_sessions=train_sessions,
        max_sequences=int(args.max_sequences),
        chunks_per_speaker=int(args.chunks_per_speaker),
        positive_probability=float(args.positive_probability),
        chunk_frames=chunk_frames,
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
        channels=int(args.channels),
        layers=int(args.layers),
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
        chunk_frames=chunk_frames,
        eval_hop_frames=eval_hop_frames,
        device=device,
        save_word_records=bool(args.save_word_records),
    )
    summary = _summarize_groups(group_results)
    summary.update(
        {
            "model": "sequence_tsvad_word_owner_tcn",
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
            "chunk_seconds": float(args.chunk_seconds),
            "eval_hop_seconds": float(args.eval_hop_seconds),
            "chunk_frames": int(chunk_frames),
            "eval_hop_frames": int(eval_hop_frames),
            "chunks_per_speaker": int(args.chunks_per_speaker),
            "max_sequences": int(args.max_sequences),
            "positive_probability": float(args.positive_probability),
            "channels": int(args.channels),
            "layers": int(args.layers),
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
        "feature_dim": int(features.shape[-1]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "sequence_tsvad_word_owner_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "sequence_tsvad_word_owner_words.jsonl", word_records)
    (args.output_dir / "sequence_tsvad_word_owner_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "sequence_tsvad_word_owner_training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
