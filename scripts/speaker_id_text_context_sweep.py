from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_oracle_mask_sweep import (
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
)  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


@dataclass(frozen=True)
class TextRow:
    window: str
    index: int
    truth: str
    target_share: float
    active_5pct: int
    mixed_pred: str
    text: str
    context: str


def _load_raw_rows(path: Path) -> List[dict]:
    return list(json.loads(path.read_text(encoding="utf-8")).get("rows") or [])


def _load_text_rows(
    *,
    dominance_json: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    context_radius: int,
) -> List[TextRow]:
    raw_rows = _load_raw_rows(dominance_json)
    rows_by_window: Dict[str, List[dict]] = defaultdict(list)
    for raw in raw_rows:
        rows_by_window[str(raw["window"])].append(raw)

    rows: List[TextRow] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        texts = [str(item).strip() for item in payload["texts"].tolist()]
        for raw in sorted(window_rows, key=lambda item: int(item["index"])):
            index = int(raw["index"])
            start = max(0, index - context_radius)
            end = min(len(texts), index + context_radius + 1)
            context_words = texts[start:end]
            rows.append(
                TextRow(
                    window=window_name,
                    index=index,
                    truth=str(raw["truth"]),
                    target_share=float(raw.get("target_share") or 0.0),
                    active_5pct=int(raw.get("active_5pct") or 0),
                    mixed_pred=str(raw.get("mixed_pred") or "unknown"),
                    text=texts[index] if 0 <= index < len(texts) else "",
                    context=" ".join(context_words),
                )
            )
    return rows


def _score_subset(
    name: str,
    rows: Sequence[TextRow],
    predictions: Sequence[str],
) -> Dict[str, object]:
    return {
        "name": name,
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
    }


def _best_router_threshold(
    confidence: np.ndarray,
    text_predictions: Sequence[str],
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
            text if score >= threshold else mixed
            for score, text, mixed in zip(confidence, text_predictions, mixed_predictions)
        ]
        correct = sum(truth == pred for truth, pred in zip(truths, routed))
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(threshold)
    return best_threshold


def _speaker_one_hot(labels: Sequence[str], speakers: Sequence[str]) -> np.ndarray:
    speaker_to_idx = {speaker: index for index, speaker in enumerate(speakers)}
    values = np.zeros((len(labels), len(speakers)), dtype=np.float32)
    for row, label in enumerate(labels):
        idx = speaker_to_idx.get(str(label))
        if idx is not None:
            values[row, idx] = 1.0
    return values


def _evaluate_radius(rows: Sequence[TextRow], *, seed: int) -> Dict[str, object]:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = sorted({_window_group(row.window) for row in rows})
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_window_group(row.window)].append(index)
    speakers = sorted({row.truth for row in rows})
    predictions: Dict[str, List[str]] = {
        "mixed_audio_baseline": [row.mixed_pred for row in rows],
        "text_context_logreg": ["unknown"] * len(rows),
        "text_context_router": ["unknown"] * len(rows),
        "text_plus_audio_label_logreg": ["unknown"] * len(rows),
    }

    for group in groups:
        test_indices = by_group_index[group]
        train_indices = [
            index for index, row in enumerate(rows) if _window_group(row.window) != group
        ]
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=50000,
            lowercase=True,
        )
        train_text = vectorizer.fit_transform([row.context for row in train_rows])
        test_text = vectorizer.transform([row.context for row in test_rows])
        text_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        text_model.fit(train_text, [row.truth for row in train_rows])
        train_probs = text_model.predict_proba(train_text)
        test_probs = text_model.predict_proba(test_text)
        class_labels = [str(item) for item in text_model.classes_.tolist()]
        train_text_pred = [class_labels[int(index)] for index in np.argmax(train_probs, axis=1)]
        test_text_pred = [class_labels[int(index)] for index in np.argmax(test_probs, axis=1)]
        threshold = _best_router_threshold(
            np.max(train_probs, axis=1),
            train_text_pred,
            [row.mixed_pred for row in train_rows],
            [row.truth for row in train_rows],
        )

        train_audio_onehot = csr_matrix(
            _speaker_one_hot([row.mixed_pred for row in train_rows], speakers)
        )
        test_audio_onehot = csr_matrix(
            _speaker_one_hot([row.mixed_pred for row in test_rows], speakers)
        )
        fusion_train = hstack([train_text, train_audio_onehot], format="csr")
        fusion_test = hstack([test_text, test_audio_onehot], format="csr")
        fusion_model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
        fusion_model.fit(fusion_train, [row.truth for row in train_rows])
        fusion_pred = [str(item) for item in fusion_model.predict(fusion_test)]

        for local_idx, global_idx in enumerate(test_indices):
            predictions["text_context_logreg"][global_idx] = test_text_pred[local_idx]
            predictions["text_context_router"][global_idx] = (
                test_text_pred[local_idx]
                if float(np.max(test_probs[local_idx])) >= threshold
                else rows[global_idx].mixed_pred
            )
            predictions["text_plus_audio_label_logreg"][global_idx] = fusion_pred[local_idx]

    all_scores = {
        name: _score_subset(name, rows, labels) for name, labels in sorted(predictions.items())
    }
    hard_indices = [index for index, row in enumerate(rows) if row.target_share <= 0.90]
    hard_rows = [rows[index] for index in hard_indices]
    hard_scores = {
        name: _score_subset(name, hard_rows, [labels[index] for index in hard_indices])
        for name, labels in sorted(predictions.items())
    }
    return {
        "all_rows": all_scores,
        "hard_rows": hard_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reference-text context as a word-aware speaker attribution signal."
    )
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument(
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--context-radii", default="0,2,5,10")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, default=Path("/tmp/codex_text_context_sweep.json"))
    args = parser.parse_args()

    radii = [int(item.strip()) for item in str(args.context_radii).split(",") if item.strip()]
    results: Dict[str, object] = {}
    row_count = 0
    for radius in radii:
        rows = _load_text_rows(
            dominance_json=args.dominance_json.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
            context_radius=radius,
        )
        row_count = len(rows)
        results[f"radius_{radius}"] = _evaluate_radius(rows, seed=int(args.seed))

    payload = {
        "model": "reference_text_context_speaker_attribution",
        "note": "Uses reference-word text from the evaluation cache; treat as an optimistic word-aware diagnostic before predicted-ASR text integration.",
        "selected_rows": row_count,
        "context_radii": radii,
        "scores": results,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("radius,slice,name,examples,accuracy", flush=True)
    for radius_name, radius_scores in results.items():
        for slice_name, slice_scores in radius_scores.items():
            for name, score in sorted(slice_scores.items()):
                direct = score["direct"]
                print(
                    ",".join(
                        [
                            radius_name,
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
