from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    ConditionedTasNetExtractor,
    _conditioning_vectors,
    _extract_batch,
    _slice_wave,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _candidate_features(
    candidate_probs: np.ndarray,
    candidate_pred: np.ndarray,
    candidates: Sequence[str],
    classes: Sequence[str],
    mixed_probs: np.ndarray,
) -> np.ndarray:
    class_index = {speaker: idx for idx, speaker in enumerate(classes)}
    num_rows, num_candidates, _num_classes = candidate_probs.shape
    candidate_arr = np.asarray(candidates, dtype=object)
    mixed_top = np.max(mixed_probs, axis=1)
    mixed_margin = _margin(mixed_probs)
    mixed_argmax = np.argmax(mixed_probs, axis=1)

    features: List[List[float]] = []
    for row_idx in range(num_rows):
        self_scores = np.zeros(num_candidates, dtype=np.float32)
        self_margins = np.zeros(num_candidates, dtype=np.float32)
        top_scores = np.max(candidate_probs[row_idx], axis=1).astype(np.float32)
        top_margins = _margin(candidate_probs[row_idx])
        for candidate_idx, candidate in enumerate(candidates):
            idx = class_index.get(candidate)
            if idx is None:
                continue
            probs = candidate_probs[row_idx, candidate_idx]
            self_scores[candidate_idx] = probs[idx]
            other = probs.copy()
            other[idx] = -1.0
            self_margins[candidate_idx] = probs[idx] - np.max(other)

        self_order = np.argsort(np.argsort(-self_scores))
        mixed_scores_for_candidates = np.asarray(
            [mixed_probs[row_idx, class_index.get(candidate, 0)] for candidate in candidates],
            dtype=np.float32,
        )
        mixed_order = np.argsort(np.argsort(-mixed_scores_for_candidates))

        for candidate_idx, candidate in enumerate(candidates):
            class_idx = class_index.get(candidate)
            mixed_score = float(mixed_probs[row_idx, class_idx]) if class_idx is not None else 0.0
            agrees = float(candidate_pred[row_idx, candidate_idx] == candidate_arr[candidate_idx])
            mixed_predicts_candidate = (
                float(classes[int(mixed_argmax[row_idx])] == candidate)
                if class_idx is not None
                else 0.0
            )
            features.append(
                [
                    float(self_scores[candidate_idx]),
                    float(self_margins[candidate_idx]),
                    float(top_scores[candidate_idx]),
                    float(top_margins[candidate_idx]),
                    agrees,
                    mixed_score,
                    float(mixed_top[row_idx]),
                    float(mixed_margin[row_idx]),
                    mixed_predicts_candidate,
                    float(self_scores[candidate_idx] - mixed_score),
                    float(self_margins[candidate_idx] - mixed_margin[row_idx]),
                    float(self_order[candidate_idx]),
                    float(mixed_order[candidate_idx]),
                ]
            )
    return np.asarray(features, dtype=np.float32)


def _selector_labels(rows: Sequence[MaskRow], candidates: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [1 if row.truth == candidate else 0 for row in rows for candidate in candidates],
        dtype=np.int64,
    )


def _fit_selector(train_x: np.ndarray, train_y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selector = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
    )
    selector.fit(train_x, train_y)
    return selector


def _best_router_threshold(
    candidate_conf: np.ndarray,
    candidate_pred: Sequence[str],
    mixed_pred: Sequence[str],
    truths: Sequence[str],
) -> float:
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19, dtype=np.float32),
                np.asarray(candidate_conf, dtype=np.float32),
            ]
        )
    )
    best_threshold = 1.01
    best_correct = -1
    for threshold in thresholds:
        routed = [
            cand if conf >= threshold else mixed
            for conf, cand, mixed in zip(candidate_conf, candidate_pred, mixed_pred)
        ]
        correct = sum(truth == pred for truth, pred in zip(truths, routed))
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(threshold)
    return best_threshold


def _load_model(
    args: argparse.Namespace, *, embedding_dim: int, device: str
) -> ConditionedTasNetExtractor:
    model_path = args.model_output.expanduser()
    payload = torch.load(model_path, map_location=device, weights_only=False)
    saved_args = dict(payload.get("args") or {})
    model = ConditionedTasNetExtractor(
        embedding_dim=embedding_dim,
        enc_feats=int(saved_args.get("enc_feats", args.enc_feats)),
        bottleneck=int(saved_args.get("bottleneck", args.bottleneck)),
        cond_dim=int(saved_args.get("cond_dim", args.cond_dim)),
        enc_kernel=int(saved_args.get("enc_kernel", args.enc_kernel)),
        layers=int(saved_args.get("layers", args.layers)),
        stacks=int(saved_args.get("stacks", args.stacks)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _ordered_rows(rows: Sequence[MaskRow]) -> Tuple[List[MaskRow], Dict[str, List[MaskRow]]]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    ordered: List[MaskRow] = []
    for window in sorted(rows_by_window):
        ordered.extend(rows_by_window[window])
    return ordered, rows_by_window


def _write_candidate_embeddings(
    rows: Sequence[MaskRow],
    *,
    model: ConditionedTasNetExtractor,
    centroids: Mapping[str, np.ndarray],
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    row_batch_size: int,
    embed_batch_size: int,
    device: str,
) -> None:
    ordered_rows, rows_by_window = _ordered_rows(rows)
    candidates = tuple(CORE_SPEAKERS)
    titanet = _load_titanet(device)
    samples = int(round(window_seconds * sample_rate))

    all_embeddings: List[np.ndarray] = []
    all_windows: List[str] = []
    all_indices: List[int] = []
    all_truths: List[str] = []
    all_shares: List[float] = []
    all_active: List[int] = []

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
        waves: List[np.ndarray] = []
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            waves.append(_slice_wave(mixed, start_sample, start_sample + samples))

        window_vectors: List[np.ndarray] = []
        for offset in range(0, len(waves), row_batch_size):
            batch_waves = waves[offset : offset + row_batch_size]
            expanded_waves = np.repeat(np.stack(batch_waves), len(candidates), axis=0)
            expanded_labels = [speaker for _wave in batch_waves for speaker in candidates]
            extracted = _extract_batch(
                model,
                expanded_waves,
                expanded_labels,
                centroids=centroids,
                device=device,
            )
            embedded = _embed_waveforms(
                titanet,
                list(extracted),
                sample_rate=sample_rate,
                batch_size=embed_batch_size,
                device=device,
            )
            window_vectors.append(
                embedded.reshape(len(batch_waves), len(candidates), embedded.shape[-1])
            )
        all_embeddings.append(np.vstack(window_vectors).astype(np.float32))
        all_windows.extend([window_name] * len(window_rows))
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        print(f"candidate_embedded {window_name}: {len(window_rows)} rows", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.concatenate(all_embeddings, axis=0).astype(np.float32),
        candidates=np.asarray(candidates),
        windows=np.asarray(all_windows),
        indices=np.asarray(all_indices, dtype=np.int32),
        truths=np.asarray(all_truths),
        target_shares=np.asarray(all_shares, dtype=np.float32),
        active_5pct=np.asarray(all_active, dtype=np.int16),
    )


def _load_candidate_embeddings(path: Path) -> Tuple[np.ndarray, Tuple[str, ...], List[MaskRow]]:
    payload = np.load(path, allow_pickle=False)
    rows = [
        MaskRow(
            window=str(window),
            index=int(index),
            truth=str(truth),
            target_file="",
            target_share=float(share),
            active_5pct=int(active),
        )
        for window, index, truth, share, active in zip(
            payload["windows"].tolist(),
            payload["indices"].tolist(),
            payload["truths"].tolist(),
            payload["target_shares"].tolist(),
            payload["active_5pct"].tolist(),
        )
    ]
    return (
        np.asarray(payload["embeddings"], dtype=np.float32),
        tuple(str(item) for item in payload["candidates"].tolist()),
        rows,
    )


def _rows_for_items(items) -> Tuple[np.ndarray, List[str]]:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for item in items:
        vectors.extend(np.asarray(item.embeddings, dtype=np.float32))
        labels.extend(item.truths)
    return np.vstack(vectors), labels


def _fit_lda(train_x: np.ndarray, train_y: Sequence[str]):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(np.asarray(train_x, dtype=np.float32), list(train_y))
    return model


def _margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] < 2:
        return np.ones(probabilities.shape[0], dtype=np.float32)
    sorted_probs = np.sort(probabilities, axis=1)
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)


def _score_predictions(
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


def _aggregate_class_evidence(
    candidate_probs: np.ndarray,
    candidate_argmax: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, tuple[List[str], np.ndarray]]:
    row_count, candidate_count, class_count = candidate_probs.shape
    class_sum = candidate_probs.sum(axis=1)
    class_noisy_or = 1.0 - np.prod(1.0 - np.clip(candidate_probs, 0.0, 1.0), axis=1)
    vote_scores = np.zeros((row_count, class_count), dtype=np.float32)
    row_ids = np.arange(row_count)
    for candidate_index in range(candidate_count):
        vote_scores[row_ids, candidate_argmax[:, candidate_index]] += 1.0
    vote_with_sum_tiebreak = vote_scores + (class_sum / max(float(candidate_count), 1.0)) * 1e-3

    evidence = {
        "class_sum_probability": class_sum,
        "class_noisy_or_probability": class_noisy_or,
        "class_vote_sum_tiebreak": vote_with_sum_tiebreak,
    }
    class_array = np.asarray(classes, dtype=object)
    outputs: Dict[str, tuple[List[str], np.ndarray]] = {}
    for name, scores in evidence.items():
        best = np.argmax(scores, axis=1)
        outputs[name] = (
            [str(item) for item in class_array[best].tolist()],
            scores[row_ids, best].astype(np.float32),
        )
    return outputs


def _evaluate_candidate_embeddings(
    candidate_embeddings: np.ndarray,
    candidates: Sequence[str],
    rows: Sequence[MaskRow],
    *,
    clean_bank,
    training_items,
    mixed_embeddings: np.ndarray,
) -> Dict[str, object]:
    groups = sorted({_window_group(row.window) for row in rows})
    predictions: Dict[str, List[str]] = {
        "mixed": ["unknown"] * len(rows),
        "candidate_self_prob": ["unknown"] * len(rows),
        "candidate_self_margin": ["unknown"] * len(rows),
        "candidate_agree_prob": ["unknown"] * len(rows),
        "class_noisy_or_probability": ["unknown"] * len(rows),
        "class_noisy_or_probability_router": ["unknown"] * len(rows),
        "class_sum_probability": ["unknown"] * len(rows),
        "class_sum_probability_router": ["unknown"] * len(rows),
        "class_vote_sum_tiebreak": ["unknown"] * len(rows),
        "class_vote_sum_tiebreak_router": ["unknown"] * len(rows),
        "max_any_probability": ["unknown"] * len(rows),
        "confidence_router": ["unknown"] * len(rows),
        "learned_candidate_selector": ["unknown"] * len(rows),
        "learned_candidate_router": ["unknown"] * len(rows),
        "true_condition_classifier": ["unknown"] * len(rows),
    }
    any_candidate_contains_truth = [False] * len(rows)
    diagnostic_rows: List[Dict[str, object]] = []
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_window_group(row.window)].append(index)

    for group in groups:
        train_items = [item for item in training_items.values() if item.window.group != group]
        train_x, train_y = _rows_for_items(train_items)
        train_x = np.vstack([clean_bank.embeddings, train_x]).astype(np.float32)
        train_y = list(clean_bank.labels) + train_y
        lda = _fit_lda(train_x, train_y)
        classes = [str(item) for item in lda.classes_.tolist()]
        class_index = {speaker: idx for idx, speaker in enumerate(classes)}
        test_indices = by_group_index[group]

        mixed_probs = lda.predict_proba(mixed_embeddings[test_indices])
        mixed_pred = [classes[int(index)] for index in np.argmax(mixed_probs, axis=1)]
        mixed_margin = _margin(mixed_probs)
        mixed_conf = np.max(mixed_probs, axis=1)

        flat = candidate_embeddings[test_indices].reshape(
            len(test_indices) * len(candidates),
            candidate_embeddings.shape[-1],
        )
        candidate_probs = lda.predict_proba(flat).reshape(
            len(test_indices),
            len(candidates),
            len(classes),
        )
        candidate_argmax = np.argmax(candidate_probs, axis=2)
        candidate_pred = np.asarray(classes, dtype=object)[candidate_argmax]

        self_scores = np.zeros((len(test_indices), len(candidates)), dtype=np.float32)
        self_margins = np.zeros_like(self_scores)
        for candidate_idx, candidate in enumerate(candidates):
            idx = class_index.get(candidate)
            if idx is None:
                continue
            probs = candidate_probs[:, candidate_idx, :]
            self_scores[:, candidate_idx] = probs[:, idx]
            other = probs.copy()
            other[:, idx] = -1.0
            self_margins[:, candidate_idx] = probs[:, idx] - np.max(other, axis=1)

        best_self = np.argmax(self_scores, axis=1)
        best_margin = np.argmax(self_margins, axis=1)
        flat_best = np.argmax(candidate_probs.reshape(len(test_indices), -1), axis=1)
        flat_class = flat_best % len(classes)
        aggregate_predictions = _aggregate_class_evidence(
            candidate_probs,
            candidate_argmax,
            classes,
        )
        agreement_scores = self_scores.copy()
        agreement_scores[candidate_pred != np.asarray(candidates, dtype=object)[None, :]] = -1.0
        best_agree = np.argmax(agreement_scores, axis=1)
        no_agree = np.max(agreement_scores, axis=1) < 0.0
        best_agree[no_agree] = best_self[no_agree]

        best_extracted_conf = self_scores[np.arange(len(test_indices)), best_self]
        best_extracted_margin = self_margins[np.arange(len(test_indices)), best_self]
        use_extracted = (best_extracted_conf > mixed_conf) & (best_extracted_margin > mixed_margin)

        test_index_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_index_set]
        learned_best = best_self
        learned_conf = best_extracted_conf
        learned_threshold = 1.01
        aggregate_thresholds: Dict[str, float] = {}
        if train_indices:
            train_flat = candidate_embeddings[train_indices].reshape(
                len(train_indices) * len(candidates),
                candidate_embeddings.shape[-1],
            )
            train_candidate_probs = lda.predict_proba(train_flat).reshape(
                len(train_indices),
                len(candidates),
                len(classes),
            )
            train_candidate_pred = np.asarray(classes, dtype=object)[
                np.argmax(train_candidate_probs, axis=2)
            ]
            train_candidate_argmax = np.argmax(train_candidate_probs, axis=2)
            train_mixed_probs = lda.predict_proba(mixed_embeddings[train_indices])
            train_mixed_pred = [
                classes[int(index)] for index in np.argmax(train_mixed_probs, axis=1)
            ]
            train_aggregate = _aggregate_class_evidence(
                train_candidate_probs,
                train_candidate_argmax,
                classes,
            )
            train_truth = [rows[index].truth for index in train_indices]
            for aggregate_name, (train_labels, train_confidence) in train_aggregate.items():
                aggregate_thresholds[aggregate_name] = _best_router_threshold(
                    train_confidence,
                    train_labels,
                    train_mixed_pred,
                    train_truth,
                )
            train_x = _candidate_features(
                train_candidate_probs,
                train_candidate_pred,
                candidates,
                classes,
                train_mixed_probs,
            )
            train_y = _selector_labels([rows[index] for index in train_indices], candidates)
            if len(np.unique(train_y)) == 2:
                selector = _fit_selector(train_x, train_y)
                train_scores = selector.predict_proba(train_x)[:, 1].reshape(
                    len(train_indices),
                    len(candidates),
                )
                train_best = np.argmax(train_scores, axis=1)
                train_conf = train_scores[np.arange(len(train_indices)), train_best]
                train_candidate_labels = [candidates[int(index)] for index in train_best]
                learned_threshold = _best_router_threshold(
                    train_conf,
                    train_candidate_labels,
                    train_mixed_pred,
                    [rows[index].truth for index in train_indices],
                )
                test_x = _candidate_features(
                    candidate_probs,
                    candidate_pred,
                    candidates,
                    classes,
                    mixed_probs,
                )
                test_scores = selector.predict_proba(test_x)[:, 1].reshape(
                    len(test_indices),
                    len(candidates),
                )
                learned_best = np.argmax(test_scores, axis=1)
                learned_conf = test_scores[np.arange(len(test_indices)), learned_best]

        candidate_to_position = {speaker: index for index, speaker in enumerate(candidates)}
        for local_idx, global_idx in enumerate(test_indices):
            truth_position = candidate_to_position.get(rows[global_idx].truth)
            true_condition_pred = (
                str(candidate_pred[local_idx, truth_position])
                if truth_position is not None
                else "unknown"
            )
            predictions["mixed"][global_idx] = mixed_pred[local_idx]
            predictions["candidate_self_prob"][global_idx] = candidates[int(best_self[local_idx])]
            predictions["candidate_self_margin"][global_idx] = candidates[
                int(best_margin[local_idx])
            ]
            predictions["candidate_agree_prob"][global_idx] = candidates[int(best_agree[local_idx])]
            for aggregate_name, (aggregate_labels, aggregate_conf) in aggregate_predictions.items():
                aggregate_label = aggregate_labels[local_idx]
                predictions[aggregate_name][global_idx] = aggregate_label
                threshold = aggregate_thresholds.get(aggregate_name, float("inf"))
                predictions[f"{aggregate_name}_router"][global_idx] = (
                    aggregate_label
                    if float(aggregate_conf[local_idx]) >= threshold
                    else mixed_pred[local_idx]
                )
            predictions["max_any_probability"][global_idx] = classes[int(flat_class[local_idx])]
            predictions["confidence_router"][global_idx] = (
                candidates[int(best_self[local_idx])]
                if bool(use_extracted[local_idx])
                else mixed_pred[local_idx]
            )
            predictions["learned_candidate_selector"][global_idx] = candidates[
                int(learned_best[local_idx])
            ]
            predictions["learned_candidate_router"][global_idx] = (
                candidates[int(learned_best[local_idx])]
                if float(learned_conf[local_idx]) >= learned_threshold
                else mixed_pred[local_idx]
            )
            predictions["true_condition_classifier"][global_idx] = true_condition_pred
            any_candidate_contains_truth[global_idx] = bool(
                rows[global_idx].truth in set(str(item) for item in candidate_pred[local_idx])
            )
            diagnostic_rows.append(
                {
                    "index": int(global_idx),
                    "group": group,
                    "truth": rows[global_idx].truth,
                    "mixed_pred": mixed_pred[local_idx],
                    "mixed_conf": float(mixed_conf[local_idx]),
                    "mixed_margin": float(mixed_margin[local_idx]),
                    "best_candidate": candidates[int(best_self[local_idx])],
                    "best_candidate_conf": float(best_extracted_conf[local_idx]),
                    "best_candidate_margin": float(best_extracted_margin[local_idx]),
                    "learned_candidate": candidates[int(learned_best[local_idx])],
                    "learned_candidate_conf": float(learned_conf[local_idx]),
                    "learned_router_threshold": float(learned_threshold),
                    "router_uses_extracted": bool(use_extracted[local_idx]),
                }
            )

    scored = {
        name: _score_predictions(name, rows, labels) for name, labels in sorted(predictions.items())
    }
    truth = [row.truth for row in rows]
    scored["oracle_mixed_plus_best_candidate"] = {
        "name": "oracle_mixed_plus_best_candidate",
        "direct": {
            "examples": len(rows),
            "correct": sum(
                expected in (mixed, best)
                for expected, mixed, best in zip(
                    truth,
                    predictions["mixed"],
                    predictions["candidate_self_prob"],
                )
            ),
            "accuracy": (
                sum(
                    expected in (mixed, best)
                    for expected, mixed, best in zip(
                        truth,
                        predictions["mixed"],
                        predictions["candidate_self_prob"],
                    )
                )
                / len(rows)
                if rows
                else 0.0
            ),
        },
    }
    scored["oracle_mixed_plus_true_condition_classifier"] = {
        "name": "oracle_mixed_plus_true_condition_classifier",
        "direct": {
            "examples": len(rows),
            "correct": sum(
                expected in (mixed, true_condition)
                for expected, mixed, true_condition in zip(
                    truth,
                    predictions["mixed"],
                    predictions["true_condition_classifier"],
                )
            ),
            "accuracy": (
                sum(
                    expected in (mixed, true_condition)
                    for expected, mixed, true_condition in zip(
                        truth,
                        predictions["mixed"],
                        predictions["true_condition_classifier"],
                    )
                )
                / len(rows)
                if rows
                else 0.0
            ),
        },
    }
    scored["oracle_any_candidate_classifier"] = {
        "name": "oracle_any_candidate_classifier",
        "direct": {
            "examples": len(rows),
            "correct": sum(any_candidate_contains_truth),
            "accuracy": (sum(any_candidate_contains_truth) / len(rows)) if rows else 0.0,
        },
    }
    scored["oracle_mixed_plus_any_candidate_classifier"] = {
        "name": "oracle_mixed_plus_any_candidate_classifier",
        "direct": {
            "examples": len(rows),
            "correct": sum(
                expected == mixed or any_candidate
                for expected, mixed, any_candidate in zip(
                    truth,
                    predictions["mixed"],
                    any_candidate_contains_truth,
                )
            ),
            "accuracy": (
                sum(
                    expected == mixed or any_candidate
                    for expected, mixed, any_candidate in zip(
                        truth,
                        predictions["mixed"],
                        any_candidate_contains_truth,
                    )
                )
                / len(rows)
                if rows
                else 0.0
            ),
        },
    }
    scored["diagnostic_rows"] = diagnostic_rows
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deployable all-candidate selection for one-hot conditioned TasNet."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_onehot_s60_s2400.pt"),
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_candidate_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_conditioned_tasnet_candidate_results.json")
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
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=24)
    parser.add_argument("--conditioning", choices=("one_hot",), default="one_hot")
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    ordered, rows_by_window = _ordered_rows(rows)
    centroids = _conditioning_vectors(
        clean_bank_path=args.clean_bank.expanduser(),
        mode=str(args.conditioning),
    )
    model = _load_model(args, embedding_dim=len(CORE_SPEAKERS), device=device)
    if not args.embedding_output.expanduser().exists():
        _write_candidate_embeddings(
            ordered,
            model=model,
            centroids=centroids,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.embedding_output.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            row_batch_size=int(args.row_batch_size),
            embed_batch_size=int(args.embed_batch_size),
            device=device,
        )

    candidate_embeddings, candidates, embedding_rows = _load_candidate_embeddings(
        args.embedding_output.expanduser()
    )
    rows_by_window = defaultdict(list)
    for row in embedding_rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    mixed_embeddings = _mixed_same_rows(
        rows_by_window,
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    scored = _evaluate_candidate_embeddings(
        candidate_embeddings,
        candidates,
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        mixed_embeddings=mixed_embeddings,
    )
    diagnostic_rows = scored.pop("diagnostic_rows")
    payload = {
        "model": "conditioned_tasnet_all_candidate_selector",
        "source_model": str(args.model_output.expanduser()),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "candidates": list(candidates),
        "scores": scored,
        "diagnostic_rows": diagnostic_rows,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    for name, result in sorted(scored.items()):
        direct = result["direct"]
        print(
            ",".join(
                [
                    name,
                    str(direct["examples"]),
                    f"{float(direct['accuracy']):.4f}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
