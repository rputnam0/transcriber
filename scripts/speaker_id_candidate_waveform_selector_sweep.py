from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_candidate_selector_feature_sweep import (  # noqa: E402
    _candidate_rich_features,
    _clean_centroids,
)
from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _best_router_threshold,
    _candidate_features,
    _fit_lda,
    _load_candidate_embeddings,
    _rows_for_items,
)
from speaker_id_conditioned_tasnet_sweep import _conditioning_vectors, _extract_batch  # noqa: E402
from speaker_id_eval_stem_candidate_sweep import (  # noqa: E402
    _load_fold_model,
    _model_path_for_group,
    _split_group,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_audio,
    _load_clean_bank,
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _load_mixture_crops(
    rows: Sequence[MaskRow],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    payload_by_window: Dict[str, Mapping[str, np.ndarray]] = {}
    audio_by_window: Dict[str, np.ndarray] = {}
    samples = int(round(window_seconds * sample_rate))
    waves: List[np.ndarray] = []
    for row in rows:
        if row.window not in payload_by_window:
            payload_by_window[row.window] = _load_titanet_word_npz(
                titanet_cache_root,
                "reference",
                row.window,
                window_seconds,
            )
        if row.window not in audio_by_window:
            audio_by_window[row.window] = _load_audio(
                prepared_root / row.window / "mixed.wav", sample_rate
            )
        payload = payload_by_window[row.window]
        starts = np.asarray(payload["word_starts"], dtype=np.float32)
        ends = np.asarray(payload["word_ends"], dtype=np.float32)
        midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
        start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
        waves.append(_slice_wave(audio_by_window[row.window], start_sample, start_sample + samples))
    return np.stack(waves).astype(np.float32)


def _extract_candidate_waves(
    rows: Sequence[MaskRow],
    mixtures: np.ndarray,
    *,
    model_base_path: Path,
    split_mode: str,
    centroids: Mapping[str, np.ndarray],
    row_batch_size: int,
    device: str,
    args: argparse.Namespace,
) -> np.ndarray:
    candidates = tuple(CORE_SPEAKERS)
    embedding_dim = next(iter(centroids.values())).shape[0]
    model_cache = {}
    outputs = np.zeros((len(rows), len(candidates), mixtures.shape[-1]), dtype=np.float32)
    for offset in range(0, len(rows), row_batch_size):
        batch_rows = rows[offset : offset + row_batch_size]
        batch_waves = mixtures[offset : offset + row_batch_size]
        for group in sorted({_split_group(row, split_mode) for row in batch_rows}):
            local_indices = [
                index
                for index, row in enumerate(batch_rows)
                if _split_group(row, split_mode) == group
            ]
            if group not in model_cache:
                model_path = _model_path_for_group(model_base_path, group)
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                model_cache[group] = _load_fold_model(
                    model_path,
                    embedding_dim=embedding_dim,
                    device=device,
                    args=args,
                )
            group_waves = np.stack([batch_waves[index] for index in local_indices])
            expanded_waves = np.repeat(group_waves, len(candidates), axis=0)
            expanded_labels = [speaker for _ in local_indices for speaker in candidates]
            extracted = _extract_batch(
                model_cache[group],
                expanded_waves,
                expanded_labels,
                centroids=centroids,
                device=device,
            )
            extracted = extracted.reshape(len(local_indices), len(candidates), extracted.shape[-1])
            for extracted_index, local_index in enumerate(local_indices):
                outputs[offset + local_index] = extracted[extracted_index]
        print(
            f"waveform_selector_extracted {min(offset + row_batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
    return outputs


def _zero_crossing_rate(waves: np.ndarray) -> np.ndarray:
    signs = np.signbit(waves)
    return np.mean(signs[..., 1:] != signs[..., :-1], axis=-1).astype(np.float32)


def _band_energy_features(waves: np.ndarray, *, sample_rate: int) -> np.ndarray:
    window = np.hanning(waves.shape[-1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(waves * window, axis=-1)).astype(np.float32)
    power = np.square(spectrum)
    freqs = np.fft.rfftfreq(waves.shape[-1], d=1.0 / sample_rate).astype(np.float32)
    total = np.sum(power, axis=-1, keepdims=True) + 1e-8
    bands = [(0.0, 300.0), (300.0, 1000.0), (1000.0, 3000.0), (3000.0, 8000.0)]
    band_ratios: List[np.ndarray] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_ratios.append(
            (np.sum(power[..., mask], axis=-1) / total.squeeze(-1)).astype(np.float32)
        )
    centroid = (np.sum(power * freqs, axis=-1) / total.squeeze(-1)).astype(np.float32)
    bandwidth = np.sqrt(
        np.sum(power * np.square(freqs - centroid[..., None]), axis=-1) / total.squeeze(-1)
    ).astype(np.float32)
    flatness = (
        np.exp(np.mean(np.log(power + 1e-8), axis=-1)) / (np.mean(power + 1e-8, axis=-1) + 1e-8)
    ).astype(np.float32)
    return np.stack([*band_ratios, centroid / 8000.0, bandwidth / 8000.0, flatness], axis=-1)


def _waveform_features(
    mixtures: np.ndarray,
    candidate_waves: np.ndarray,
    *,
    candidates: Sequence[str],
    sample_rate: int,
) -> np.ndarray:
    del candidates
    eps = 1e-8
    mix = mixtures[:, None, :]
    output = candidate_waves
    mix_rms = np.sqrt(np.mean(np.square(mix), axis=-1) + eps)
    output_rms = np.sqrt(np.mean(np.square(output), axis=-1) + eps)
    mix_peak = np.max(np.abs(mix), axis=-1) + eps
    output_peak = np.max(np.abs(output), axis=-1) + eps
    dot = np.sum(output * mix, axis=-1)
    corr = dot / (
        np.sqrt(np.sum(np.square(output), axis=-1) + eps)
        * np.sqrt(np.sum(np.square(mix), axis=-1) + eps)
    )
    residual = mix - output
    residual_rms = np.sqrt(np.mean(np.square(residual), axis=-1) + eps)
    center_width = max(1, output.shape[-1] // 5)
    center_start = (output.shape[-1] - center_width) // 2
    center_end = center_start + center_width
    center_rms = np.sqrt(np.mean(np.square(output[..., center_start:center_end]), axis=-1) + eps)
    edge = np.concatenate([output[..., :center_start], output[..., center_end:]], axis=-1)
    edge_rms = np.sqrt(np.mean(np.square(edge), axis=-1) + eps)
    zcr = _zero_crossing_rate(output)
    band_features = _band_energy_features(
        output.reshape(-1, output.shape[-1]), sample_rate=sample_rate
    )
    band_features = band_features.reshape(output.shape[0], output.shape[1], -1)
    candidate_ids = np.tile(np.eye(output.shape[1], dtype=np.float32), (output.shape[0], 1, 1))
    features = np.concatenate(
        [
            np.log(output_rms + eps)[..., None],
            np.log(output_peak + eps)[..., None],
            (output_rms / mix_rms)[..., None],
            (output_peak / mix_peak)[..., None],
            (residual_rms / mix_rms)[..., None],
            corr[..., None],
            (center_rms / output_rms)[..., None],
            (edge_rms / output_rms)[..., None],
            (center_rms / (edge_rms + eps))[..., None],
            zcr[..., None],
            band_features,
            candidate_ids,
        ],
        axis=-1,
    )
    return features.reshape(output.shape[0] * output.shape[1], -1).astype(np.float32)


def _candidate_labels(rows: Sequence[MaskRow], candidates: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [1 if row.truth == candidate else 0 for row in rows for candidate in candidates],
        dtype=np.int64,
    )


def _fit_selector(kind: str, train_x: np.ndarray, train_y: np.ndarray, seed: int):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            ),
        ).fit(train_x, train_y)
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(train_x, train_y)
    raise ValueError(kind)


def _fit_row_selector(kind: str, train_x: np.ndarray, train_y: Sequence[str], seed: int):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            ),
        ).fit(train_x, list(train_y))
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(train_x, list(train_y))
    raise ValueError(kind)


def _probabilities(model, values: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(values)[:, 1], dtype=np.float32)


def _score_named_predictions(
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


def _evaluate_waveform_selectors(
    *,
    rows: Sequence[MaskRow],
    candidates: Sequence[str],
    candidate_embeddings: np.ndarray,
    mixed_embeddings: np.ndarray,
    waveform_features: np.ndarray,
    clean_bank,
    training_items,
    seed: int,
) -> Dict[str, object]:
    groups = sorted({_window_group(row.window) for row in rows})
    by_group: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[_window_group(row.window)].append(index)
    predictions: Dict[str, List[str]] = {
        "mixed": ["unknown"] * len(rows),
        "wave_logreg": ["unknown"] * len(rows),
        "wave_logreg_router": ["unknown"] * len(rows),
        "wave_extra_trees": ["unknown"] * len(rows),
        "wave_extra_trees_router": ["unknown"] * len(rows),
        "wave_plus_scalar_logreg": ["unknown"] * len(rows),
        "wave_plus_scalar_logreg_router": ["unknown"] * len(rows),
        "wave_plus_scalar_extra_trees": ["unknown"] * len(rows),
        "wave_plus_scalar_extra_trees_router": ["unknown"] * len(rows),
        "row_wave_logreg": ["unknown"] * len(rows),
        "row_wave_extra_trees": ["unknown"] * len(rows),
        "row_wave_plus_scalar_logreg": ["unknown"] * len(rows),
        "row_wave_plus_scalar_extra_trees": ["unknown"] * len(rows),
    }
    fold_diagnostics: List[Dict[str, object]] = []
    clean_centroids = _clean_centroids(clean_bank, candidates)

    for group in groups:
        test_indices = by_group[group]
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        train_items = [item for item in training_items.values() if item.window.group != group]
        train_x, train_y = _rows_for_items(train_items)
        train_x = np.vstack([clean_bank.embeddings, train_x]).astype(np.float32)
        train_y = list(clean_bank.labels) + train_y
        lda = _fit_lda(train_x, train_y)
        classes = [str(item) for item in lda.classes_.tolist()]

        mixed_train_probs = lda.predict_proba(mixed_embeddings[train_indices])
        mixed_test_probs = lda.predict_proba(mixed_embeddings[test_indices])
        mixed_train_pred = [classes[int(index)] for index in np.argmax(mixed_train_probs, axis=1)]
        mixed_test_pred = [classes[int(index)] for index in np.argmax(mixed_test_probs, axis=1)]
        for local_index, global_index in enumerate(test_indices):
            predictions["mixed"][global_index] = mixed_test_pred[local_index]

        train_candidate_probs = lda.predict_proba(
            candidate_embeddings[train_indices].reshape(
                len(train_indices) * len(candidates),
                candidate_embeddings.shape[-1],
            )
        ).reshape(len(train_indices), len(candidates), len(classes))
        test_candidate_probs = lda.predict_proba(
            candidate_embeddings[test_indices].reshape(
                len(test_indices) * len(candidates),
                candidate_embeddings.shape[-1],
            )
        ).reshape(len(test_indices), len(candidates), len(classes))
        train_candidate_pred = np.asarray(classes, dtype=object)[
            np.argmax(train_candidate_probs, axis=2)
        ]
        test_candidate_pred = np.asarray(classes, dtype=object)[
            np.argmax(test_candidate_probs, axis=2)
        ]
        train_scalar = _candidate_features(
            train_candidate_probs,
            train_candidate_pred,
            candidates,
            classes,
            mixed_train_probs,
        )
        test_scalar = _candidate_features(
            test_candidate_probs,
            test_candidate_pred,
            candidates,
            classes,
            mixed_test_probs,
        )
        train_rich = _candidate_rich_features(
            scalar_features=train_scalar,
            candidate_probs=train_candidate_probs,
            mixed_probs=mixed_train_probs,
            candidate_embeddings=candidate_embeddings[train_indices],
            mixed_embeddings=mixed_embeddings[train_indices],
            clean_centroids=clean_centroids,
            candidates=candidates,
        )["scalar_plus"]
        test_rich = _candidate_rich_features(
            scalar_features=test_scalar,
            candidate_probs=test_candidate_probs,
            mixed_probs=mixed_test_probs,
            candidate_embeddings=candidate_embeddings[test_indices],
            mixed_embeddings=mixed_embeddings[test_indices],
            clean_centroids=clean_centroids,
            candidates=candidates,
        )["scalar_plus"]

        train_wave = waveform_features.reshape(len(rows), len(candidates), -1)[train_indices]
        test_wave = waveform_features.reshape(len(rows), len(candidates), -1)[test_indices]
        train_wave = train_wave.reshape(len(train_indices) * len(candidates), -1)
        test_wave = test_wave.reshape(len(test_indices) * len(candidates), -1)
        feature_sets = {
            "wave": (train_wave, test_wave),
            "wave_plus_scalar": (
                np.hstack([train_wave, train_rich]).astype(np.float32),
                np.hstack([test_wave, test_rich]).astype(np.float32),
            ),
        }
        train_labels = _candidate_labels([rows[index] for index in train_indices], candidates)
        for feature_name, (feature_train, feature_test) in feature_sets.items():
            for kind in ("logreg", "extra_trees"):
                name = f"{feature_name}_{kind}"
                model = _fit_selector(kind, feature_train, train_labels, seed)
                train_scores = _probabilities(model, feature_train).reshape(
                    len(train_indices), len(candidates)
                )
                test_scores = _probabilities(model, feature_test).reshape(
                    len(test_indices), len(candidates)
                )
                train_best = np.argmax(train_scores, axis=1)
                test_best = np.argmax(test_scores, axis=1)
                train_conf = train_scores[np.arange(len(train_indices)), train_best]
                test_conf = test_scores[np.arange(len(test_indices)), test_best]
                train_candidate_pred = [candidates[int(index)] for index in train_best]
                threshold = _best_router_threshold(
                    train_conf,
                    train_candidate_pred,
                    mixed_train_pred,
                    [rows[index].truth for index in train_indices],
                )
                for local_index, global_index in enumerate(test_indices):
                    candidate = candidates[int(test_best[local_index])]
                    predictions[name][global_index] = candidate
                    predictions[f"{name}_router"][global_index] = (
                        candidate
                        if float(test_conf[local_index]) >= threshold
                        else mixed_test_pred[local_index]
                    )

        row_feature_sets = {
            "row_wave": (
                train_wave.reshape(len(train_indices), -1),
                test_wave.reshape(len(test_indices), -1),
            ),
            "row_wave_plus_scalar": (
                np.hstack(
                    [
                        train_wave.reshape(len(train_indices), -1),
                        train_rich.reshape(len(train_indices), -1),
                    ]
                ).astype(np.float32),
                np.hstack(
                    [
                        test_wave.reshape(len(test_indices), -1),
                        test_rich.reshape(len(test_indices), -1),
                    ]
                ).astype(np.float32),
            ),
        }
        for feature_name, (row_train, row_test) in row_feature_sets.items():
            for kind in ("logreg", "extra_trees"):
                name = f"{feature_name}_{kind}"
                row_model = _fit_row_selector(
                    kind,
                    row_train,
                    [rows[index].truth for index in train_indices],
                    seed,
                )
                row_predictions = row_model.predict(row_test)
                for local_index, global_index in enumerate(test_indices):
                    predictions[name][global_index] = str(row_predictions[local_index])

        fold_diagnostics.append(
            {
                "group": group,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "mixed_accuracy": _score_direct(
                    [rows[index].truth for index in test_indices],
                    mixed_test_pred,
                )["accuracy"],
            }
        )
        print(f"waveform_selector_fold {group} done", flush=True)

    return {
        "scores": {
            name: _score_named_predictions(name, rows, labels)
            for name, labels in sorted(predictions.items())
        },
        "fold_diagnostics": fold_diagnostics,
    }


def _rows_by_window(rows: Sequence[MaskRow]) -> Dict[str, List[MaskRow]]:
    by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        by_window[row.window].append(row)
    return by_window


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate waveform-level selectors over all-candidate target extraction outputs."
    )
    parser.add_argument(
        "--candidate-embeddings",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidates_lgo_s300_big_s1600_embeddings.npz"),
    )
    parser.add_argument(
        "--model-base-path",
        type=Path,
        default=Path("/tmp/codex_eval_stem_tasnet_lgo_s300_big_s1600.pt"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidate_waveform_features_s300_big_s1600.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidate_waveform_selector_s300_big_s1600.json"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument("--conditioning", choices=("one_hot",), default="one_hot")
    parser.add_argument(
        "--split-mode", choices=("leave_group", "leave_session"), default="leave_group"
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--enc-feats", type=int, default=192)
    parser.add_argument("--bottleneck", type=int, default=192)
    parser.add_argument("--cond-dim", type=int, default=96)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--stacks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    candidate_embeddings, candidates, rows = _load_candidate_embeddings(
        args.candidate_embeddings.expanduser()
    )
    if args.feature_cache.expanduser().exists():
        feature_payload = np.load(args.feature_cache.expanduser(), allow_pickle=False)
        waveform_features = np.asarray(feature_payload["waveform_features"], dtype=np.float32)
    else:
        mixtures = _load_mixture_crops(
            rows,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
        )
        centroids = _conditioning_vectors(
            clean_bank_path=args.clean_bank.expanduser(),
            mode=str(args.conditioning),
        )
        candidate_waves = _extract_candidate_waves(
            rows,
            mixtures,
            model_base_path=args.model_base_path.expanduser(),
            split_mode=str(args.split_mode),
            centroids=centroids,
            row_batch_size=int(args.row_batch_size),
            device=device,
            args=args,
        )
        waveform_features = _waveform_features(
            mixtures,
            candidate_waves,
            candidates=candidates,
            sample_rate=int(args.sample_rate),
        )
        args.feature_cache.expanduser().parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_cache.expanduser(),
            waveform_features=waveform_features.astype(np.float32),
            windows=np.asarray([row.window for row in rows]),
            indices=np.asarray([row.index for row in rows], dtype=np.int32),
            truths=np.asarray([row.truth for row in rows]),
            candidates=np.asarray(candidates),
        )

    rows_by_window = _rows_by_window(rows)
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
    result = _evaluate_waveform_selectors(
        rows=rows,
        candidates=candidates,
        candidate_embeddings=candidate_embeddings,
        mixed_embeddings=mixed_embeddings,
        waveform_features=waveform_features,
        clean_bank=clean_bank,
        training_items=training_items,
        seed=int(args.seed),
    )
    payload = {
        "model": "candidate_waveform_selector_sweep",
        "candidate_embeddings": str(args.candidate_embeddings.expanduser()),
        "feature_cache": str(args.feature_cache.expanduser()),
        "selected_rows": len(rows),
        "selected_speakers": dict(Counter(row.truth for row in rows)),
        "candidates": list(candidates),
        "scores": result["scores"],
        "fold_diagnostics": result["fold_diagnostics"],
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    for name, score in sorted(result["scores"].items()):
        direct = score["direct"]
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
