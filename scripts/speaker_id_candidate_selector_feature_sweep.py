from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _best_router_threshold,
    _candidate_features,
    _fit_lda,
    _load_candidate_embeddings,
    _rows_for_items,
    _score_predictions,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_clean_bank,
    _score_direct,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def _clean_centroids(clean_bank, candidates: Sequence[str]) -> np.ndarray:
    embeddings = _normalize_rows(clean_bank.embeddings)
    labels = np.asarray(clean_bank.labels, dtype=object)
    vectors: List[np.ndarray] = []
    for speaker in candidates:
        speaker_vectors = embeddings[labels == speaker]
        if speaker_vectors.size == 0:
            vectors.append(np.zeros(embeddings.shape[1], dtype=np.float32))
            continue
        centroid = speaker_vectors.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        vectors.append(centroid.astype(np.float32))
    return np.vstack(vectors).astype(np.float32)


def _candidate_rich_features(
    *,
    scalar_features: np.ndarray,
    candidate_probs: np.ndarray,
    mixed_probs: np.ndarray,
    candidate_embeddings: np.ndarray,
    mixed_embeddings: np.ndarray,
    clean_centroids: np.ndarray,
    candidates: Sequence[str],
) -> Dict[str, np.ndarray]:
    num_rows, num_candidates, embedding_dim = candidate_embeddings.shape
    candidate_norm = _normalize_rows(candidate_embeddings.reshape(-1, embedding_dim)).reshape(
        num_rows, num_candidates, embedding_dim
    )
    mixed_norm = _normalize_rows(mixed_embeddings)
    centroid_scores = np.einsum("ncd,sd->ncs", candidate_norm, clean_centroids).astype(np.float32)
    self_centroid_scores = np.stack(
        [
            centroid_scores[:, candidate_idx, candidate_idx]
            for candidate_idx in range(num_candidates)
        ],
        axis=1,
    )
    mixed_centroid_scores = np.matmul(mixed_norm, clean_centroids.T).astype(np.float32)
    mixed_candidate_scores = np.stack(
        [mixed_centroid_scores[:, candidate_idx] for candidate_idx in range(num_candidates)],
        axis=1,
    )
    candidate_mixed_cos = np.einsum("ncd,nd->nc", candidate_norm, mixed_norm).astype(np.float32)
    candidate_ids = np.tile(np.eye(num_candidates, dtype=np.float32), (num_rows, 1, 1))

    flat_scalar = scalar_features
    flat_probs = candidate_probs.reshape(num_rows * num_candidates, candidate_probs.shape[-1])
    flat_mixed_probs = np.repeat(mixed_probs, num_candidates, axis=0)
    flat_centroids = centroid_scores.reshape(num_rows * num_candidates, num_candidates)
    flat_mixed_centroids = np.repeat(mixed_centroid_scores, num_candidates, axis=0)
    flat_candidate_ids = candidate_ids.reshape(num_rows * num_candidates, num_candidates)
    flat_self_centroid = self_centroid_scores.reshape(-1, 1)
    flat_mixed_candidate = mixed_candidate_scores.reshape(-1, 1)
    flat_candidate_mixed = candidate_mixed_cos.reshape(-1, 1)

    scalar_plus = np.hstack(
        [
            flat_scalar,
            flat_probs,
            flat_mixed_probs,
            flat_probs - flat_mixed_probs,
            flat_centroids,
            flat_mixed_centroids,
            flat_centroids - flat_mixed_centroids,
            flat_self_centroid,
            flat_mixed_candidate,
            flat_self_centroid - flat_mixed_candidate,
            flat_candidate_mixed,
            flat_candidate_ids,
        ]
    ).astype(np.float32)
    flat_candidate_embeddings = candidate_embeddings.reshape(
        num_rows * num_candidates, embedding_dim
    ).astype(np.float32)
    flat_mixed_embeddings = np.repeat(mixed_embeddings.astype(np.float32), num_candidates, axis=0)
    return {
        "scalar": flat_scalar.astype(np.float32),
        "scalar_plus": scalar_plus,
        "scalar_plus_candidate_embedding": np.hstack(
            [scalar_plus, flat_candidate_embeddings]
        ).astype(np.float32),
        "scalar_plus_candidate_and_mixed_embedding": np.hstack(
            [scalar_plus, flat_candidate_embeddings, flat_mixed_embeddings]
        ).astype(np.float32),
    }


def _fit_binary_model(kind: str, train_x: np.ndarray, train_y: np.ndarray, seed: int):
    if kind.startswith("logreg"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        c_value = 0.2 if "c02" in kind else 1.0
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=500,
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
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(train_x, train_y)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=1e-3,
                early_stopping=True,
                max_iter=250,
                n_iter_no_change=20,
                random_state=seed,
            ),
        ).fit(train_x, train_y)
    raise ValueError(kind)


def _binary_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)
    probabilities = model.predict_proba(features)
    return np.asarray(probabilities[:, 1], dtype=np.float32)


def _fit_row_model(kind: str, train_x: np.ndarray, train_y: Sequence[str], seed: int):
    if kind.startswith("row_logreg"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        c_value = 0.2 if "c02" in kind else 1.0
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=500,
                random_state=seed,
                solver="liblinear",
            ),
        ).fit(train_x, list(train_y))
    if kind == "row_extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(train_x, list(train_y))
    raise ValueError(kind)


def _row_features(
    candidate_feature_map: Mapping[str, np.ndarray],
    feature_set: str,
    num_rows: int,
    num_candidates: int,
    mixed_probs: np.ndarray,
    mixed_embeddings: np.ndarray,
) -> np.ndarray:
    candidate_features = candidate_feature_map[feature_set].reshape(num_rows, num_candidates, -1)
    row_parts = [
        candidate_features.reshape(num_rows, -1),
        mixed_probs.astype(np.float32),
        mixed_embeddings.astype(np.float32),
    ]
    return np.hstack(row_parts).astype(np.float32)


def _candidate_labels(rows: Sequence[MaskRow], candidates: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [1 if row.truth == candidate else 0 for row in rows for candidate in candidates],
        dtype=np.int64,
    )


def _fill_binary_predictions(
    *,
    name: str,
    model_kind: str,
    feature_set: str,
    predictions: Dict[str, List[str]],
    router_predictions: Dict[str, List[str]],
    train_features: Mapping[str, np.ndarray],
    test_features: Mapping[str, np.ndarray],
    train_rows: Sequence[MaskRow],
    test_indices: Sequence[int],
    candidates: Sequence[str],
    mixed_train_pred: Sequence[str],
    mixed_test_pred: Sequence[str],
    seed: int,
) -> None:
    train_y = _candidate_labels(train_rows, candidates)
    if len(np.unique(train_y)) < 2:
        return
    num_candidates = len(candidates)
    train_x = train_features[feature_set]
    test_x = test_features[feature_set]
    model = _fit_binary_model(model_kind, train_x, train_y, seed)
    train_scores = _binary_scores(model, train_x).reshape(len(train_rows), num_candidates)
    test_scores = _binary_scores(model, test_x).reshape(len(test_indices), num_candidates)
    train_best = np.argmax(train_scores, axis=1)
    test_best = np.argmax(test_scores, axis=1)
    train_conf = train_scores[np.arange(len(train_rows)), train_best]
    test_conf = test_scores[np.arange(len(test_indices)), test_best]
    train_candidate_pred = [candidates[int(index)] for index in train_best]
    threshold = _best_router_threshold(
        train_conf,
        train_candidate_pred,
        mixed_train_pred,
        [row.truth for row in train_rows],
    )
    for local_idx, global_idx in enumerate(test_indices):
        candidate_pred = candidates[int(test_best[local_idx])]
        predictions[name][global_idx] = candidate_pred
        router_predictions[f"{name}_router"][global_idx] = (
            candidate_pred
            if float(test_conf[local_idx]) >= threshold
            else mixed_test_pred[local_idx]
        )


def _evaluate_selectors(
    *,
    candidate_embeddings: np.ndarray,
    candidates: Sequence[str],
    rows: Sequence[MaskRow],
    clean_bank,
    training_items,
    mixed_embeddings: np.ndarray,
    seed: int,
    profile: str,
) -> Dict[str, object]:
    groups = sorted({_window_group(row.window) for row in rows})
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_window_group(row.window)].append(index)

    fast_selector_specs = [
        ("logreg_scalar_plus", "logreg", "scalar_plus"),
        ("logreg_c02_scalar_plus", "logreg_c02", "scalar_plus"),
        ("logreg_candidate_embedding", "logreg_c02", "scalar_plus_candidate_embedding"),
        (
            "logreg_candidate_mixed_embedding",
            "logreg_c02",
            "scalar_plus_candidate_and_mixed_embedding",
        ),
    ]
    heavy_selector_specs = [
        ("extra_trees_scalar_plus", "extra_trees", "scalar_plus"),
        ("random_forest_scalar_plus", "random_forest", "scalar_plus"),
        ("mlp_scalar_plus", "mlp", "scalar_plus"),
        ("mlp_candidate_embedding", "mlp", "scalar_plus_candidate_embedding"),
    ]
    fast_row_specs = [
        ("row_logreg_scalar_plus", "row_logreg", "scalar_plus"),
        ("row_logreg_c02_scalar_plus", "row_logreg_c02", "scalar_plus"),
        ("row_logreg_candidate_embedding", "row_logreg_c02", "scalar_plus_candidate_embedding"),
    ]
    heavy_row_specs = [("row_extra_trees_scalar_plus", "row_extra_trees", "scalar_plus")]
    if profile == "micro":
        selector_specs = fast_selector_specs[:2]
        row_specs = fast_row_specs[:2]
    else:
        selector_specs = fast_selector_specs + (heavy_selector_specs if profile == "all" else [])
        row_specs = fast_row_specs + (heavy_row_specs if profile == "all" else [])
    predictions: Dict[str, List[str]] = {
        "mixed": ["unknown"] * len(rows),
        "self_prob": ["unknown"] * len(rows),
        "candidate_prob_argmax": ["unknown"] * len(rows),
    }
    router_predictions: Dict[str, List[str]] = {}
    for name, _kind, _feature_set in selector_specs:
        predictions[name] = ["unknown"] * len(rows)
        router_predictions[f"{name}_router"] = ["unknown"] * len(rows)
    for name, _kind, _feature_set in row_specs:
        predictions[name] = ["unknown"] * len(rows)
    any_candidate_contains_truth = [False] * len(rows)
    clean_centroids = _clean_centroids(clean_bank, candidates)
    fold_diagnostics: List[Dict[str, object]] = []

    for group in groups:
        print(f"fold {group}: start", flush=True)
        test_indices = by_group_index[group]
        test_index_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_index_set]
        train_rows = [rows[index] for index in train_indices]
        train_items = [item for item in training_items.values() if item.window.group != group]
        train_x, train_y = _rows_for_items(train_items)
        train_x = np.vstack([clean_bank.embeddings, train_x]).astype(np.float32)
        train_y = list(clean_bank.labels) + train_y
        lda = _fit_lda(train_x, train_y)
        classes = [str(item) for item in lda.classes_.tolist()]
        class_index = {speaker: idx for idx, speaker in enumerate(classes)}

        mixed_train_probs = lda.predict_proba(mixed_embeddings[train_indices])
        mixed_test_probs = lda.predict_proba(mixed_embeddings[test_indices])
        mixed_train_pred = [classes[int(index)] for index in np.argmax(mixed_train_probs, axis=1)]
        mixed_test_pred = [classes[int(index)] for index in np.argmax(mixed_test_probs, axis=1)]

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
        train_feature_map = _candidate_rich_features(
            scalar_features=train_scalar,
            candidate_probs=train_candidate_probs,
            mixed_probs=mixed_train_probs,
            candidate_embeddings=candidate_embeddings[train_indices],
            mixed_embeddings=mixed_embeddings[train_indices],
            clean_centroids=clean_centroids,
            candidates=candidates,
        )
        test_feature_map = _candidate_rich_features(
            scalar_features=test_scalar,
            candidate_probs=test_candidate_probs,
            mixed_probs=mixed_test_probs,
            candidate_embeddings=candidate_embeddings[test_indices],
            mixed_embeddings=mixed_embeddings[test_indices],
            clean_centroids=clean_centroids,
            candidates=candidates,
        )

        self_scores = np.zeros((len(test_indices), len(candidates)), dtype=np.float32)
        for candidate_idx, candidate in enumerate(candidates):
            idx = class_index.get(candidate)
            if idx is not None:
                self_scores[:, candidate_idx] = test_candidate_probs[:, candidate_idx, idx]
        self_best = np.argmax(self_scores, axis=1)
        flat_best = np.argmax(test_candidate_probs.reshape(len(test_indices), -1), axis=1)
        flat_class = flat_best % len(classes)

        for local_idx, global_idx in enumerate(test_indices):
            truth = rows[global_idx].truth
            predictions["mixed"][global_idx] = mixed_test_pred[local_idx]
            predictions["self_prob"][global_idx] = candidates[int(self_best[local_idx])]
            predictions["candidate_prob_argmax"][global_idx] = classes[int(flat_class[local_idx])]
            any_candidate_contains_truth[global_idx] = bool(
                truth in set(str(item) for item in test_candidate_pred[local_idx])
            )

        for name, model_kind, feature_set in selector_specs:
            print(f"fold {group}: fit {name}", flush=True)
            _fill_binary_predictions(
                name=name,
                model_kind=model_kind,
                feature_set=feature_set,
                predictions=predictions,
                router_predictions=router_predictions,
                train_features=train_feature_map,
                test_features=test_feature_map,
                train_rows=train_rows,
                test_indices=test_indices,
                candidates=candidates,
                mixed_train_pred=mixed_train_pred,
                mixed_test_pred=mixed_test_pred,
                seed=seed,
            )

        for name, model_kind, feature_set in row_specs:
            print(f"fold {group}: fit {name}", flush=True)
            train_row_features = _row_features(
                train_feature_map,
                feature_set,
                len(train_indices),
                len(candidates),
                mixed_train_probs,
                mixed_embeddings[train_indices],
            )
            test_row_features = _row_features(
                test_feature_map,
                feature_set,
                len(test_indices),
                len(candidates),
                mixed_test_probs,
                mixed_embeddings[test_indices],
            )
            row_model = _fit_row_model(
                model_kind,
                train_row_features,
                [row.truth for row in train_rows],
                seed,
            )
            row_predictions = row_model.predict(test_row_features)
            for local_idx, global_idx in enumerate(test_indices):
                predictions[name][global_idx] = str(row_predictions[local_idx])

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
        print(f"fold {group}: done", flush=True)

    all_predictions = {**predictions, **router_predictions}
    scored = {
        name: _score_predictions(name, rows, labels)
        for name, labels in sorted(all_predictions.items())
    }
    truths = [row.truth for row in rows]
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
                truth == mixed_pred or any_candidate
                for truth, mixed_pred, any_candidate in zip(
                    truths,
                    predictions["mixed"],
                    any_candidate_contains_truth,
                )
            ),
            "accuracy": (
                sum(
                    truth == mixed_pred or any_candidate
                    for truth, mixed_pred, any_candidate in zip(
                        truths,
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
    return {
        "scores": scored,
        "fold_diagnostics": fold_diagnostics,
    }


def _rows_by_window(rows: Sequence[MaskRow]) -> Dict[str, List[MaskRow]]:
    by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        by_window[row.window].append(row)
    return by_window


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep richer deployable selectors over all-candidate target extraction embeddings."
    )
    parser.add_argument(
        "--candidate-embeddings",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_onehot_candidates_full_embeddings.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_candidate_selector_features.json"),
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
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--profile",
        choices=("micro", "fast", "all"),
        default="micro",
        help="Use micro scalar selectors by default; 'fast' adds embedding linear selectors; 'all' adds slower trees and MLPs.",
    )
    args = parser.parse_args()

    candidate_embeddings, candidates, rows = _load_candidate_embeddings(
        args.candidate_embeddings.expanduser()
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
    result = _evaluate_selectors(
        candidate_embeddings=candidate_embeddings,
        candidates=candidates,
        rows=rows,
        clean_bank=clean_bank,
        training_items=training_items,
        mixed_embeddings=mixed_embeddings,
        seed=int(args.seed),
        profile=str(args.profile),
    )
    payload = {
        "model": "candidate_selector_feature_sweep",
        "candidate_embeddings": str(args.candidate_embeddings.expanduser()),
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
