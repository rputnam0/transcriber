from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_architecture_sweep import _fit_predict  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    _embed_waveforms,
    _load_titanet,
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
)
from speaker_id_speechbrain_embedding_sweep import SpeakerRow, _load_embedding_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _write_titanet_clean_cache(
    *,
    training_cache: Path,
    output_path: Path,
    batch_size: int,
    device: str,
) -> None:
    payload = np.load(training_cache, allow_pickle=False)
    targets = np.asarray(payload["targets"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    model = _load_titanet(device)
    embeddings = _embed_waveforms(
        model,
        list(targets),
        sample_rate=16000,
        batch_size=batch_size,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        labels=np.asarray(labels),
        source_training_cache=str(training_cache),
    )


def _write_titanet_mixed_cache(
    rows: Sequence[SpeakerRow],
    *,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
) -> None:
    payloads: Dict[str, object] = {}
    embeddings: List[np.ndarray] = []
    for row in rows:
        if row.window not in payloads:
            payloads[row.window] = _load_titanet_word_npz(
                titanet_cache_root,
                "reference",
                row.window,
                window_seconds,
            )
        payload = payloads[row.window]
        embeddings.append(np.asarray(payload["embeddings"][row.index], dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.vstack(embeddings).astype(np.float32),
        windows=np.asarray([row.window for row in rows]),
        indices=np.asarray([row.index for row in rows], dtype=np.int32),
        truths=np.asarray([row.truth for row in rows]),
        target_shares=np.asarray([row.target_share for row in rows], dtype=np.float32),
        active_5pct=np.asarray([row.active_5pct for row in rows], dtype=np.int16),
        mixed_pred=np.asarray([row.mixed_pred for row in rows]),
    )


def _load_clean(path: Path) -> Tuple[np.ndarray, List[str]]:
    payload = np.load(path, allow_pickle=False)
    return (
        np.asarray(payload["embeddings"], dtype=np.float32),
        [str(item) for item in payload["labels"].tolist()],
    )


def _fuse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _normalize_rows(np.hstack([_normalize_rows(left), _normalize_rows(right)]))


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


def _evaluate_feature_set(
    *,
    name: str,
    rows: Sequence[SpeakerRow],
    embeddings: np.ndarray,
    clean_embeddings: np.ndarray,
    clean_labels: Sequence[str],
    model_names: Sequence[str],
) -> Dict[str, List[str]]:
    groups = sorted({_window_group(row.window) for row in rows})
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_window_group(row.window)].append(index)
    predictions = {f"{name}/{model_name}": ["unknown"] * len(rows) for model_name in model_names}
    for group in groups:
        test_indices = by_group_index[group]
        train_indices = [
            index for index, row in enumerate(rows) if _window_group(row.window) != group
        ]
        train_x = np.vstack([clean_embeddings, embeddings[train_indices]]).astype(np.float32)
        train_y = list(clean_labels) + [rows[index].truth for index in train_indices]
        test_x = embeddings[test_indices]
        for model_name in model_names:
            labels = predictions[f"{name}/{model_name}"]
            for global_idx, label in zip(
                test_indices,
                _fit_predict(model_name, train_x, train_y, test_x),
            ):
                labels[global_idx] = label
    return predictions


def _score_predictions(
    rows: Sequence[SpeakerRow],
    predictions: Dict[str, List[str]],
) -> Dict[str, Dict[str, object]]:
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
        description="Evaluate Titanet+ECAPA fused speaker embeddings on flattened speaker-ID rows."
    )
    parser.add_argument(
        "--ecapa-mixed-cache",
        type=Path,
        default=Path("/tmp/codex_speechbrain_ecapa_mixed_all_rows.npz"),
    )
    parser.add_argument(
        "--ecapa-clean-cache",
        type=Path,
        default=Path("/tmp/codex_speechbrain_ecapa_clean_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--titanet-mixed-cache",
        type=Path,
        default=Path("/tmp/codex_titanet_mixed_all_rows_for_ecapa_fusion.npz"),
    )
    parser.add_argument(
        "--titanet-clean-cache",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_titanet_ecapa_fusion_sweep.json"),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        default="centroid_cosine,lda_shrinkage,linear_svc,logreg_balanced",
    )
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    ecapa_mixed, rows = _load_embedding_rows(args.ecapa_mixed_cache.expanduser())
    if not args.titanet_mixed_cache.expanduser().exists():
        _write_titanet_mixed_cache(
            rows,
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.titanet_mixed_cache.expanduser(),
            window_seconds=float(args.window_seconds),
        )
    if not args.titanet_clean_cache.expanduser().exists():
        _write_titanet_clean_cache(
            training_cache=args.training_cache.expanduser(),
            output_path=args.titanet_clean_cache.expanduser(),
            batch_size=int(args.batch_size),
            device=device,
        )

    titanet_mixed, titanet_rows = _load_embedding_rows(args.titanet_mixed_cache.expanduser())
    if [(row.window, row.index) for row in rows] != [
        (row.window, row.index) for row in titanet_rows
    ]:
        raise ValueError("ECAPA and Titanet mixed caches are not aligned")
    ecapa_clean, ecapa_labels = _load_clean(args.ecapa_clean_cache.expanduser())
    titanet_clean, titanet_labels = _load_clean(args.titanet_clean_cache.expanduser())
    if ecapa_labels != titanet_labels:
        raise ValueError("ECAPA and Titanet clean caches are not aligned")

    model_names = [item.strip() for item in str(args.models).split(",") if item.strip()]
    predictions: Dict[str, List[str]] = {
        "titanet_mixed_baseline_from_dominance": [row.mixed_pred for row in rows]
    }
    feature_sets = {
        "titanet_same_clean_cache": (
            _normalize_rows(titanet_mixed),
            _normalize_rows(titanet_clean),
        ),
        "ecapa_same_clean_cache": (
            _normalize_rows(ecapa_mixed),
            _normalize_rows(ecapa_clean),
        ),
        "titanet_ecapa_concat": (
            _fuse(titanet_mixed, ecapa_mixed),
            _fuse(titanet_clean, ecapa_clean),
        ),
    }
    for feature_name, (mixed_embeddings, clean_embeddings) in feature_sets.items():
        predictions.update(
            _evaluate_feature_set(
                name=feature_name,
                rows=rows,
                embeddings=mixed_embeddings,
                clean_embeddings=clean_embeddings,
                clean_labels=titanet_labels,
                model_names=model_names,
            )
        )

    scores = _score_predictions(rows, predictions)
    payload = {
        "model": "titanet_ecapa_fusion_sweep",
        "selected_rows": len(rows),
        "selected_speakers": dict(Counter(row.truth for row in rows)),
        "models": model_names,
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
