from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    SPEAKER_HANDLES,
    _limit_rows_by_group,
    _load_or_create_snippets,
    _normalize_wave,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet_word_npz,
    _rms,
    _score_direct,
    _score_slices,
    _slice_wave,
    evaluate_mask_embeddings,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows, _speaker_centroids  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


class TargetActivityNet(nn.Module):
    def __init__(self, embedding_dim: int, cond_channels: int = 24, hidden: int = 56) -> None:
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, 96),
            nn.SiLU(),
            nn.Linear(96, cond_channels),
            nn.SiLU(),
        )
        channels = 1 + cond_channels
        layers: List[nn.Module] = [
            nn.Conv2d(channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(7, hidden),
            nn.SiLU(),
        ]
        for dilation in (1, 2, 4, 8, 16, 24):
            layers.extend(
                [
                    nn.Conv2d(
                        hidden,
                        hidden,
                        kernel_size=(3, 5),
                        padding=(1, 2 * dilation),
                        dilation=(1, dilation),
                    ),
                    nn.GroupNorm(7, hidden),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
                nn.GroupNorm(4, hidden // 2),
                nn.SiLU(),
                nn.Conv2d(hidden // 2, 1, kernel_size=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, logmag: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        batch, _channel, freq, frames = logmag.shape
        cond = self.cond_proj(enrollment).view(batch, -1, 1, 1).expand(-1, -1, freq, frames)
        logits = self.net(torch.cat([logmag, cond], dim=1))
        return logits.squeeze(1).mean(dim=1)


def _source_activity_targets(
    sources: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    db_below_peak: float,
) -> torch.Tensor:
    mag = torch.abs(
        torch.stft(
            sources,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
    )
    energy = mag.square().mean(dim=1)
    peak = energy.amax(dim=1, keepdim=True)
    floor = peak * (10.0 ** (-float(db_below_peak) / 10.0))
    floor = torch.maximum(floor, torch.full_like(floor, 1e-8))
    return (energy >= floor).float()


def _synth_activity_batch(
    waves: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    batch_size: int,
    speaker_order: Sequence[str],
    rng: random.Random,
    active_query_prob: float,
    max_sources: int,
    min_snr_db: float,
    max_snr_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_speaker[str(label)].append(index)

    mixtures: List[np.ndarray] = []
    query_sources: List[np.ndarray] = []
    enrollments: List[np.ndarray] = []
    query_labels: List[str] = []
    active_counts: List[int] = []
    speakers_with_data = [speaker for speaker in speaker_order if by_speaker.get(speaker)]
    if len(speakers_with_data) < 2:
        raise RuntimeError("Need at least two speakers with snippets")

    for _ in range(batch_size):
        source_count = 1
        draw = rng.random()
        if draw < 0.68:
            source_count = 2
        elif draw < 0.90:
            source_count = min(3, max_sources, len(speakers_with_data))
        source_speakers = rng.sample(speakers_with_data, k=source_count)

        source_waves: Dict[str, np.ndarray] = {}
        scaled_sources: List[np.ndarray] = []
        for source_index, speaker in enumerate(source_speakers):
            wave = _normalize_wave(waves[rng.choice(by_speaker[speaker])])
            if source_index == 0:
                gain = 1.0
            else:
                snr_db = rng.uniform(min_snr_db, max_snr_db)
                gain = 10.0 ** (-snr_db / 20.0)
            scaled = (wave * gain).astype(np.float32)
            source_waves[speaker] = scaled
            scaled_sources.append(scaled)

        mixture = np.sum(np.stack(scaled_sources), axis=0).astype(np.float32)
        peak = max(float(np.max(np.abs(mixture))), 1e-6)
        if peak > 0.95:
            scale = 0.95 / peak
            mixture = (mixture * scale).astype(np.float32)
            source_waves = {
                speaker: (wave * scale).astype(np.float32) for speaker, wave in source_waves.items()
            }

        inactive = [speaker for speaker in speakers_with_data if speaker not in source_waves]
        if rng.random() < active_query_prob or not inactive:
            query_speaker = rng.choice(source_speakers)
            query_wave = source_waves[query_speaker]
        else:
            query_speaker = rng.choice(inactive)
            query_wave = np.zeros_like(mixture, dtype=np.float32)

        mixtures.append(mixture.astype(np.float32))
        query_sources.append(query_wave.astype(np.float32))
        enrollments.append(centroids[query_speaker])
        query_labels.append(query_speaker)
        active_counts.append(source_count)

    return (
        np.stack(mixtures).astype(np.float32),
        np.stack(query_sources).astype(np.float32),
        np.stack(enrollments).astype(np.float32),
        query_labels,
        np.asarray(active_counts, dtype=np.int16),
    )


def _train_model(
    waves: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> TargetActivityNet:
    embedding_dim = next(iter(centroids.values())).shape[0]
    model = TargetActivityNet(embedding_dim=embedding_dim).to(device)
    model_path = args.model_output.expanduser()
    if model_path.exists():
        payload = torch.load(model_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4
    )
    window = torch.hann_window(int(args.n_fft), device=device)
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        mixture_np, query_np, enrollment_np, _query_labels, active_counts = _synth_activity_batch(
            waves,
            labels,
            centroids=centroids,
            batch_size=int(args.batch_size),
            speaker_order=CORE_SPEAKERS,
            rng=rng,
            active_query_prob=float(args.active_query_prob),
            max_sources=int(args.max_sources),
            min_snr_db=float(args.min_snr_db),
            max_snr_db=float(args.max_snr_db),
        )
        mixture = torch.from_numpy(mixture_np).to(device)
        query_source = torch.from_numpy(query_np).to(device)
        enrollment = torch.from_numpy(enrollment_np).to(device)
        mix_mag = torch.abs(
            torch.stft(
                mixture,
                n_fft=int(args.n_fft),
                hop_length=int(args.hop_length),
                win_length=int(args.n_fft),
                window=window,
                return_complex=True,
            )
        )
        labels_t = _source_activity_targets(
            query_source,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
            db_below_peak=float(args.activity_db_below_peak),
        )
        logits = model(torch.log1p(mix_mag).unsqueeze(1), enrollment)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = probs >= 0.5
                frame_acc = (pred == (labels_t >= 0.5)).float().mean().item()
                positive_rate = labels_t.mean().item()
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"frame_acc={frame_acc:.4f} positive_rate={positive_rate:.4f} "
                f"mean_sources={float(np.mean(active_counts)):.2f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "speakers": CORE_SPEAKERS,
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "conditioning": "clean_bank_centroid",
            "train_steps": int(args.train_steps),
        },
        model_path,
    )
    return model


def _frame_mask_for_word(
    *,
    word_start: float,
    word_end: float,
    crop_start: float,
    frames: int,
    sample_rate: int,
    hop_length: int,
    context_seconds: float,
) -> np.ndarray:
    frame_times = np.arange(frames, dtype=np.float32) * (float(hop_length) / float(sample_rate))
    rel_start = float(word_start) - float(crop_start) - float(context_seconds)
    rel_end = float(word_end) - float(crop_start) + float(context_seconds)
    mask = (frame_times >= rel_start) & (frame_times <= rel_end)
    if not np.any(mask):
        midpoint = ((float(word_start) + float(word_end)) / 2.0) - float(crop_start)
        index = int(np.argmin(np.abs(frame_times - midpoint)))
        mask[index] = True
    return mask


def _score_activity_words(
    rows: Sequence[MaskRow],
    *,
    model: TargetActivityNet,
    centroids: Mapping[str, np.ndarray],
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
    n_fft: int,
    hop_length: int,
    word_context_seconds: float,
    device: str,
) -> Dict[str, object]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    speakers = tuple(CORE_SPEAKERS)
    samples = int(round(window_seconds * sample_rate))
    window_tensor = torch.hann_window(n_fft, device=device)
    predictions: List[str] = []
    top2_hits = 0
    top3_hits = 0
    ranks: List[int] = []
    truth_scores: List[float] = []
    top_margins: List[float] = []
    score_by_key: Dict[Tuple[str, int], np.ndarray] = {}

    model.eval()
    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)

        crop_waves: List[np.ndarray] = []
        ordered_rows: List[MaskRow] = []
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            crop_start = midpoint - window_seconds / 2.0
            start_sample = int(math.floor(crop_start * sample_rate))
            crop_waves.append(_slice_wave(mixed, start_sample, start_sample + samples))
            ordered_rows.append(row)

        row_scores: List[np.ndarray] = []
        for offset in range(0, len(crop_waves), batch_size):
            batch_waves = crop_waves[offset : offset + batch_size]
            expanded_waves = np.repeat(np.stack(batch_waves), len(speakers), axis=0)
            expanded_enrollment = np.vstack(
                [centroids[speaker] for _ in batch_waves for speaker in speakers]
            ).astype(np.float32)
            mixture = torch.from_numpy(expanded_waves.astype(np.float32)).to(device)
            enrollment = torch.from_numpy(expanded_enrollment).to(device)
            with torch.inference_mode():
                stft = torch.stft(
                    mixture,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window_tensor,
                    return_complex=True,
                )
                logits = model(torch.log1p(torch.abs(stft)).unsqueeze(1), enrollment)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            frame_count = probs.shape[1]
            for local_index, row in enumerate(ordered_rows[offset : offset + len(batch_waves)]):
                midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
                crop_start = midpoint - window_seconds / 2.0
                mask = _frame_mask_for_word(
                    word_start=float(starts[row.index]),
                    word_end=float(ends[row.index]),
                    crop_start=crop_start,
                    frames=frame_count,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    context_seconds=word_context_seconds,
                )
                speaker_probs = probs[
                    local_index * len(speakers) : (local_index + 1) * len(speakers)
                ]
                row_scores.append(speaker_probs[:, mask].mean(axis=1).astype(np.float32))

        for row, scores in zip(ordered_rows, row_scores):
            score_by_key[(row.window, row.index)] = scores

        print(f"activity_scored {window_name}: {len(window_rows)} rows", flush=True)

    score_matrix = np.stack([score_by_key[(row.window, row.index)] for row in rows]).astype(
        np.float32
    )
    for row, scores in zip(rows, score_matrix):
        order = np.argsort(scores)[::-1]
        prediction = speakers[int(order[0])]
        predictions.append(prediction)
        top2 = {speakers[int(index)] for index in order[:2]}
        top3 = {speakers[int(index)] for index in order[:3]}
        top2_hits += int(row.truth in top2)
        top3_hits += int(row.truth in top3)
        rank = int(np.where(order == speakers.index(row.truth))[0][0]) + 1
        ranks.append(rank)
        truth_scores.append(float(scores[speakers.index(row.truth)]))
        top_margins.append(float(scores[order[0]] - scores[order[1]]))

    direct = _score_direct([row.truth for row in rows], predictions)
    return {
        "name": "target_activity_synthetic_pvad",
        "direct": direct,
        "top2_contains_truth": {
            "examples": len(rows),
            "correct": top2_hits,
            "accuracy": (top2_hits / len(rows)) if rows else 0.0,
        },
        "top3_contains_truth": {
            "examples": len(rows),
            "correct": top3_hits,
            "accuracy": (top3_hits / len(rows)) if rows else 0.0,
        },
        "mean_truth_rank": float(np.mean(ranks)) if ranks else 0.0,
        "mean_truth_score": float(np.mean(truth_scores)) if truth_scores else 0.0,
        "mean_top_margin": float(np.mean(top_margins)) if top_margins else 0.0,
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
        "predictions": predictions,
        "score_matrix": score_matrix,
    }


def _score_word_predictions(
    name: str,
    rows: Sequence[MaskRow],
    predictions: Sequence[str],
) -> Dict[str, object]:
    return {
        "name": name,
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
    }


def _best_router_threshold(
    values: np.ndarray,
    activity_pred: Sequence[str],
    mixed_pred: Sequence[str],
    truths: Sequence[str],
) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("inf")
    candidates = sorted({float(value) for value in values})
    thresholds = [min(candidates) - 1e-6, max(candidates) + 1e-6]
    thresholds.extend((left + right) / 2.0 for left, right in zip(candidates, candidates[1:]))
    best_threshold = thresholds[0]
    best_correct = -1
    for threshold in thresholds:
        correct = 0
        for value, activity, mixed, truth in zip(values, activity_pred, mixed_pred, truths):
            prediction = activity if float(value) >= threshold else mixed
            correct += int(prediction == truth)
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(threshold)
    return best_threshold


def _activity_feature_matrix(
    scores: np.ndarray,
    activity_predictions: Sequence[str],
    mixed_predictions: Sequence[str],
    speakers: Sequence[str],
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores, axis=1)[:, ::-1]
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    top = sorted_scores[:, 0]
    second = sorted_scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(top)
    margin = top - second
    clipped = np.clip(scores, 1e-6, 1.0)
    normalized = clipped / np.maximum(clipped.sum(axis=1, keepdims=True), 1e-6)
    entropy = -(normalized * np.log(normalized)).sum(axis=1)
    speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}
    activity_ids = np.zeros((scores.shape[0], len(speakers)), dtype=np.float32)
    mixed_ids = np.zeros((scores.shape[0], len(speakers)), dtype=np.float32)
    agree = np.zeros((scores.shape[0], 1), dtype=np.float32)
    for row_index, (activity, mixed) in enumerate(zip(activity_predictions, mixed_predictions)):
        activity_index = speaker_to_index.get(str(activity))
        mixed_index = speaker_to_index.get(str(mixed))
        if activity_index is not None:
            activity_ids[row_index, activity_index] = 1.0
        if mixed_index is not None:
            mixed_ids[row_index, mixed_index] = 1.0
        agree[row_index, 0] = float(str(activity) == str(mixed))
    return np.hstack(
        [
            scores,
            top[:, None],
            second[:, None],
            margin[:, None],
            entropy[:, None],
            agree,
            activity_ids,
            mixed_ids,
        ]
    ).astype(np.float32)


def _activity_fusion_scores(
    rows: Sequence[MaskRow],
    *,
    score_matrix: np.ndarray,
    activity_predictions: Sequence[str],
    mixed_predictions: Sequence[str],
    speakers: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    order = np.argsort(score_matrix, axis=1)[:, ::-1]
    sorted_scores = np.take_along_axis(score_matrix, order, axis=1)
    confidence = sorted_scores[:, 0]
    margin = confidence - sorted_scores[:, 1]
    clipped = np.clip(score_matrix, 1e-6, 1.0)
    normalized = clipped / np.maximum(clipped.sum(axis=1, keepdims=True), 1e-6)
    neg_entropy = (normalized * np.log(normalized)).sum(axis=1)
    groups = sorted({_window_group(row.window) for row in rows})
    by_group: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[_window_group(row.window)].append(index)

    router_values = {
        "activity_confidence_router": confidence,
        "activity_margin_router": margin,
        "activity_neg_entropy_router": neg_entropy,
    }
    predictions = {
        name: ["unknown"] * len(rows) for name in [*router_values.keys(), "activity_learned_router"]
    }
    diagnostics: List[Dict[str, object]] = []
    features = _activity_feature_matrix(
        score_matrix,
        activity_predictions,
        mixed_predictions,
        speakers,
    )
    truths = [row.truth for row in rows]

    for group in groups:
        test_indices = by_group[group]
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        train_activity = [activity_predictions[index] for index in train_indices]
        train_mixed = [mixed_predictions[index] for index in train_indices]
        train_truth = [truths[index] for index in train_indices]
        fold_diag: Dict[str, object] = {"group": group, "test_rows": len(test_indices)}

        for name, values in router_values.items():
            threshold = _best_router_threshold(
                values[train_indices],
                train_activity,
                train_mixed,
                train_truth,
            )
            fold_diag[f"{name}_threshold"] = threshold
            for index in test_indices:
                predictions[name][index] = (
                    activity_predictions[index]
                    if float(values[index]) >= threshold
                    else mixed_predictions[index]
                )

        learned_name = "activity_learned_router"
        train_y = np.asarray(
            [int(activity_predictions[index] == truths[index]) for index in train_indices],
            dtype=np.int64,
        )
        if len(np.unique(train_y)) >= 2:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=seed,
                    solver="liblinear",
                ),
            )
            model.fit(features[train_indices], train_y)
            train_prob = np.asarray(model.predict_proba(features[train_indices])[:, 1])
            threshold = _best_router_threshold(
                train_prob,
                train_activity,
                train_mixed,
                train_truth,
            )
            test_prob = np.asarray(model.predict_proba(features[test_indices])[:, 1])
            fold_diag[f"{learned_name}_threshold"] = float(threshold)
            for local_index, index in enumerate(test_indices):
                predictions[learned_name][index] = (
                    activity_predictions[index]
                    if float(test_prob[local_index]) >= threshold
                    else mixed_predictions[index]
                )
        else:
            for index in test_indices:
                predictions[learned_name][index] = predictions["activity_margin_router"][index]
            fold_diag[f"{learned_name}_threshold"] = "fallback_margin"

        diagnostics.append(fold_diag)

    return {
        "scores": {
            name: _score_word_predictions(name, rows, labels)
            for name, labels in sorted(predictions.items())
        },
        "fold_diagnostics": diagnostics,
    }


def _speaker_from_clip(path: Path) -> str | None:
    for speaker, handle in SPEAKER_HANDLES.items():
        if handle in path.name:
            return speaker
    return None


def _oracle_source_activity(
    rows: Sequence[MaskRow],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    sample_rate: int,
    context_seconds: float,
) -> Dict[str, object]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    speakers = tuple(CORE_SPEAKERS)
    predictions: List[str] = []
    top2_hits = 0
    top3_hits = 0
    ranks: List[int] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        source_by_speaker: Dict[str, np.ndarray] = {}
        for path in sorted((prepared_root / window_name / "clips").glob("*.wav")):
            speaker = _speaker_from_clip(path)
            if speaker is not None:
                source_by_speaker[speaker] = _load_audio(path, sample_rate)

        for row in window_rows:
            start_sample = int(
                math.floor((float(starts[row.index]) - context_seconds) * sample_rate)
            )
            end_sample = int(math.ceil((float(ends[row.index]) + context_seconds) * sample_rate))
            scores: List[float] = []
            for speaker in speakers:
                wave = source_by_speaker.get(speaker)
                if wave is None:
                    scores.append(float("-inf"))
                else:
                    scores.append(_rms(_slice_wave(wave, start_sample, end_sample)))
            score_array = np.asarray(scores, dtype=np.float32)
            order = np.argsort(score_array)[::-1]
            prediction = speakers[int(order[0])]
            predictions.append(prediction)
            top2_hits += int(row.truth in {speakers[int(index)] for index in order[:2]})
            top3_hits += int(row.truth in {speakers[int(index)] for index in order[:3]})
            ranks.append(int(np.where(order == speakers.index(row.truth))[0][0]) + 1)

    direct = _score_direct([row.truth for row in rows], predictions)
    return {
        "name": "oracle_source_activity_rms",
        "direct": direct,
        "top2_contains_truth": {
            "examples": len(rows),
            "correct": top2_hits,
            "accuracy": (top2_hits / len(rows)) if rows else 0.0,
        },
        "top3_contains_truth": {
            "examples": len(rows),
            "correct": top3_hits,
            "accuracy": (top3_hits / len(rows)) if rows else 0.0,
        },
        "mean_truth_rank": float(np.mean(ranks)) if ranks else 0.0,
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
        "predictions": predictions,
    }


def _prediction_union(
    rows: Sequence[MaskRow],
    left: Sequence[str],
    right: Sequence[str],
    *,
    name: str,
) -> Dict[str, object]:
    correct = [row.truth == str(a) or row.truth == str(b) for row, a, b in zip(rows, left, right)]
    return {
        "name": name,
        "examples": len(rows),
        "correct": int(sum(correct)),
        "accuracy": (float(sum(correct)) / len(rows)) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a target-speaker activity model for flat-audio speaker ID."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--quality-records",
        type=Path,
        default=Path(
            ".outputs/speaker_id_baseline_prod_graph/artifacts/bank/"
            "1e2a7a014a23be63/dataset/quality_records.jsonl"
        ),
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/prod/Audio"))
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/codex_target_activity_cache"))
    parser.add_argument(
        "--snippet-cache", type=Path, default=Path("/tmp/codex_target_activity_snippets.npz")
    )
    parser.add_argument("--model-output", type=Path, default=Path("/tmp/codex_target_activity.pt"))
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_target_activity_results.json")
    )
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument(
        "--train-sessions",
        default="Session 50,Session 51,Session 52,Session 53,Session 54,Session 55,"
        "Session 56,Session 57,Session 58,Session 59,Session 60",
    )
    parser.add_argument("--snippet-seconds", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--max-snippets-per-speaker", type=int, default=160)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--active-query-prob", type=float, default=0.68)
    parser.add_argument("--max-sources", type=int, default=3)
    parser.add_argument("--min-snr-db", type=float, default=-8.0)
    parser.add_argument("--max-snr-db", type=float, default=8.0)
    parser.add_argument("--activity-db-below-peak", type=float, default=34.0)
    parser.add_argument("--word-context-seconds", type=float, default=0.08)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    if not rows:
        raise RuntimeError("No rows selected")

    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    mixed_result = evaluate_mask_embeddings(
        _mixed_same_rows(
            rows_by_window,
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
        ),
        rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
    )
    mixed_predictions = list(mixed_result.get("predictions") or [])
    mixed_result.pop("predictions", None)

    oracle_activity = _oracle_source_activity(
        rows,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
        sample_rate=int(args.sample_rate),
        context_seconds=float(args.word_context_seconds),
    )
    oracle_predictions = list(oracle_activity.get("predictions") or [])

    activity_result: Dict[str, object] | None = None
    if not args.skip_training:
        centroids = _speaker_centroids(args.clean_bank.expanduser(), CORE_SPEAKERS)
        waves, labels = _load_or_create_snippets(args)
        print(f"loaded_snippets {dict(Counter(labels))}", flush=True)
        model = _train_model(
            waves,
            labels,
            centroids=centroids,
            args=args,
            device=device,
        )
        activity_result = _score_activity_words(
            rows,
            model=model,
            centroids=centroids,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            batch_size=int(args.eval_batch_size),
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            word_context_seconds=float(args.word_context_seconds),
            device=device,
        )

    payload: Dict[str, object] = {
        "selected_rows": len(rows),
        "selected_speakers": dict(Counter(row.truth for row in rows)),
        "row_selection": {
            "max_target_share": float(args.max_target_share),
            "eval_limit": int(args.eval_limit),
            "seed": int(args.seed),
        },
        "training": {
            "train_sessions": str(args.train_sessions),
            "snippet_seconds": float(args.snippet_seconds),
            "max_snippets_per_speaker": int(args.max_snippets_per_speaker),
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "active_query_prob": float(args.active_query_prob),
            "max_sources": int(args.max_sources),
        },
        "mixed_same_rows": mixed_result,
        "oracle_source_activity": {
            key: value for key, value in oracle_activity.items() if key != "predictions"
        },
        "oracle_source_activity_or_mixed_union": _prediction_union(
            rows,
            oracle_predictions,
            mixed_predictions,
            name="oracle_source_activity_or_mixed_union",
        ),
    }
    if activity_result is not None:
        activity_predictions = list(activity_result.get("predictions") or [])
        activity_score_matrix = np.asarray(activity_result.get("score_matrix"), dtype=np.float32)
        payload["target_activity"] = {
            key: value
            for key, value in activity_result.items()
            if key not in {"predictions", "score_matrix"}
        }
        payload["target_activity_or_mixed_union"] = _prediction_union(
            rows,
            activity_predictions,
            mixed_predictions,
            name="target_activity_or_mixed_union",
        )
        payload["target_activity_fusion"] = _activity_fusion_scores(
            rows,
            score_matrix=activity_score_matrix,
            activity_predictions=activity_predictions,
            mixed_predictions=mixed_predictions,
            speakers=CORE_SPEAKERS,
            seed=int(args.seed),
        )

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("name,examples,accuracy,top2,top3", flush=True)
    print(
        ",".join(
            [
                "mixed_same_rows/lda_shrinkage",
                str(mixed_result["direct"]["examples"]),
                f"{float(mixed_result['direct']['accuracy']):.4f}",
                "",
                "",
            ]
        ),
        flush=True,
    )
    print(
        ",".join(
            [
                "oracle_source_activity_rms",
                str(oracle_activity["direct"]["examples"]),
                f"{float(oracle_activity['direct']['accuracy']):.4f}",
                f"{float(oracle_activity['top2_contains_truth']['accuracy']):.4f}",
                f"{float(oracle_activity['top3_contains_truth']['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )
    if activity_result is not None:
        print(
            ",".join(
                [
                    "target_activity_synthetic_pvad",
                    str(activity_result["direct"]["examples"]),
                    f"{float(activity_result['direct']['accuracy']):.4f}",
                    f"{float(activity_result['top2_contains_truth']['accuracy']):.4f}",
                    f"{float(activity_result['top3_contains_truth']['accuracy']):.4f}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
