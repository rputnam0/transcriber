from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_oracle_mask_sweep import _score_direct, _score_slices  # noqa: E402
from speaker_id_speechbrain_embedding_sweep import SpeakerRow, _load_embedding_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _speaker_centroids(clean_bank_path: Path, speakers: Sequence[str]) -> np.ndarray:
    payload = np.load(clean_bank_path, allow_pickle=False)
    embeddings = _normalize_rows(np.asarray(payload["embeddings"], dtype=np.float32))
    labels = np.asarray([str(item) for item in payload["labels"].tolist()], dtype=object)
    rows: List[np.ndarray] = []
    for speaker in speakers:
        speaker_rows = embeddings[labels == speaker]
        if speaker_rows.size == 0:
            raise ValueError(f"No clean-bank embeddings for {speaker}")
        centroid = speaker_rows.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        rows.append(centroid.astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def _pairwise_features(
    mixed_embeddings: np.ndarray,
    centroids: np.ndarray,
    *,
    include_embedding: bool,
) -> np.ndarray:
    mixed_norm = _normalize_rows(mixed_embeddings)
    speakers = centroids.shape[0]
    repeated_mixed = np.repeat(mixed_norm, speakers, axis=0)
    tiled_centroids = np.tile(centroids, (mixed_norm.shape[0], 1))
    cosine = np.sum(repeated_mixed * tiled_centroids, axis=1, keepdims=True)
    parts = [
        cosine,
        np.abs(repeated_mixed - tiled_centroids),
        repeated_mixed * tiled_centroids,
        np.tile(np.eye(speakers, dtype=np.float32), (mixed_norm.shape[0], 1)),
    ]
    if include_embedding:
        parts.insert(1, repeated_mixed)
        parts.insert(2, tiled_centroids)
    return np.hstack(parts).astype(np.float32)


def _pairwise_labels(rows: Sequence[SpeakerRow], speakers: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [1 if row.truth == speaker else 0 for row in rows for speaker in speakers],
        dtype=np.int64,
    )


def _fit_model(kind: str, train_x: np.ndarray, train_y: np.ndarray, seed: int):
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
                max_iter=800,
                random_state=seed,
                solver="liblinear",
            ),
        ).fit(train_x, train_y)
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(train_x, train_y)
    raise ValueError(kind)


def _positive_scores(model, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)


def _best_router_threshold(
    confidence: np.ndarray,
    pairwise_predictions: Sequence[str],
    mixed_predictions: Sequence[str],
    truths: Sequence[str],
) -> float:
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19, dtype=np.float32),
                np.asarray(confidence, dtype=np.float32),
            ]
        )
    )
    best_threshold = 1.01
    best_correct = -1
    for threshold in thresholds:
        routed = [
            pairwise if conf >= threshold else mixed
            for conf, pairwise, mixed in zip(confidence, pairwise_predictions, mixed_predictions)
        ]
        correct = sum(truth == pred for truth, pred in zip(truths, routed))
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(threshold)
    return best_threshold


def _score_subset(
    name: str,
    rows: Sequence[SpeakerRow],
    predictions: Sequence[str],
) -> Dict[str, object]:
    return {
        "name": name,
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
    }


def _evaluate(
    rows: Sequence[SpeakerRow],
    embeddings: np.ndarray,
    *,
    centroids: np.ndarray,
    model_kinds: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    speakers = sorted({row.truth for row in rows})
    groups = sorted({_window_group(row.window) for row in rows})
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_window_group(row.window)].append(index)

    predictions: Dict[str, List[str]] = {
        "mixed_titanet_lda_baseline": [row.mixed_pred for row in rows]
    }
    for include_embedding in (False, True):
        feature_name = "summary" if not include_embedding else "embedding"
        for kind in model_kinds:
            predictions[f"pairwise_{feature_name}/{kind}"] = ["unknown"] * len(rows)
            predictions[f"pairwise_{feature_name}/{kind}_router"] = ["unknown"] * len(rows)

    for group in groups:
        test_indices = by_group_index[group]
        train_indices = [
            index for index, row in enumerate(rows) if _window_group(row.window) != group
        ]
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        train_y = _pairwise_labels(train_rows, speakers)
        for include_embedding in (False, True):
            feature_name = "summary" if not include_embedding else "embedding"
            train_x = _pairwise_features(
                embeddings[train_indices],
                centroids,
                include_embedding=include_embedding,
            )
            test_x = _pairwise_features(
                embeddings[test_indices],
                centroids,
                include_embedding=include_embedding,
            )
            for kind in model_kinds:
                model = _fit_model(kind, train_x, train_y, seed)
                train_scores = _positive_scores(model, train_x).reshape(
                    len(train_rows), len(speakers)
                )
                test_scores = _positive_scores(model, test_x).reshape(len(test_rows), len(speakers))
                train_best = np.argmax(train_scores, axis=1)
                test_best = np.argmax(test_scores, axis=1)
                train_conf = train_scores[np.arange(len(train_rows)), train_best]
                test_conf = test_scores[np.arange(len(test_rows)), test_best]
                train_pred = [speakers[int(index)] for index in train_best]
                threshold = _best_router_threshold(
                    train_conf,
                    train_pred,
                    [row.mixed_pred for row in train_rows],
                    [row.truth for row in train_rows],
                )
                labels = predictions[f"pairwise_{feature_name}/{kind}"]
                router_labels = predictions[f"pairwise_{feature_name}/{kind}_router"]
                for local_idx, global_idx in enumerate(test_indices):
                    pred = speakers[int(test_best[local_idx])]
                    labels[global_idx] = pred
                    router_labels[global_idx] = (
                        pred
                        if float(test_conf[local_idx]) >= threshold
                        else rows[global_idx].mixed_pred
                    )

    all_scores = {
        name: _score_subset(name, rows, labels) for name, labels in sorted(predictions.items())
    }
    hard_indices = [index for index, row in enumerate(rows) if row.target_share <= 0.90]
    hard_rows = [rows[index] for index in hard_indices]
    hard_scores = {
        name: _score_subset(name, hard_rows, [labels[index] for index in hard_indices])
        for name, labels in sorted(predictions.items())
    }
    return {"all_rows": all_scores, "hard_rows": hard_scores}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise speaker-conditioned attribution over Titanet mixed embeddings."
    )
    parser.add_argument(
        "--mixed-cache",
        type=Path,
        default=Path("/tmp/codex_titanet_mixed_all_rows_for_ecapa_fusion.npz"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_pairwise_titanet_attribution_sweep.json"),
    )
    parser.add_argument("--models", default="logreg,logreg_c02,extra_trees")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    embeddings, rows = _load_embedding_rows(args.mixed_cache.expanduser())
    speakers = sorted({row.truth for row in rows})
    centroids = _speaker_centroids(args.clean_bank.expanduser(), speakers)
    model_kinds = [item.strip() for item in str(args.models).split(",") if item.strip()]
    scores = _evaluate(
        rows,
        embeddings,
        centroids=centroids,
        model_kinds=model_kinds,
        seed=int(args.seed),
    )
    payload = {
        "model": "pairwise_titanet_attribution_sweep",
        "mixed_cache": str(args.mixed_cache.expanduser()),
        "clean_bank": str(args.clean_bank.expanduser()),
        "selected_rows": len(rows),
        "selected_speakers": dict(Counter(row.truth for row in rows)),
        "models": model_kinds,
        "scores": scores,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("slice,name,examples,accuracy", flush=True)
    for slice_name, slice_scores in scores.items():
        for name, score in sorted(slice_scores.items()):
            direct = score["direct"]
            print(
                ",".join(
                    [
                        slice_name,
                        name,
                        str(direct["examples"]),
                        f"{float(direct['accuracy']):.4f}",
                    ]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
