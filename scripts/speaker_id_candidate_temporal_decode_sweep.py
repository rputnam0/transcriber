from __future__ import annotations

# ruff: noqa: E402

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

from speaker_id_candidate_selector_feature_sweep import _candidate_labels  # noqa: E402
from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _candidate_features,
    _fit_lda,
    _fit_selector,
    _load_candidate_embeddings,
    _rows_for_items,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_clean_bank,
    _score_direct,
    _score_slices,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _rows_by_window(rows: Sequence[MaskRow]) -> Dict[str, List[MaskRow]]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    return rows_by_window


def _group_indices(rows: Sequence[MaskRow]) -> Dict[str, List[int]]:
    by_group: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[_window_group(row.window)].append(index)
    return by_group


def _window_indices(rows: Sequence[MaskRow], indices: Sequence[int]) -> Dict[str, List[int]]:
    by_window: Dict[str, List[int]] = defaultdict(list)
    for index in indices:
        by_window[rows[index].window].append(index)
    for values in by_window.values():
        values.sort(key=lambda item: rows[item].index)
    return by_window


def _align_probs(
    probabilities: np.ndarray, classes: Sequence[str], candidates: Sequence[str]
) -> np.ndarray:
    aligned = np.full((probabilities.shape[0], len(candidates)), 1e-9, dtype=np.float64)
    candidate_index = {speaker: index for index, speaker in enumerate(candidates)}
    for column, speaker in enumerate(classes):
        if str(speaker) in candidate_index:
            aligned[:, candidate_index[str(speaker)]] = probabilities[:, column]
    return _normalize_emissions(aligned)


def _normalize_emissions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 1e-9)
    return values / values.sum(axis=1, keepdims=True)


def _self_scores(
    candidate_probs: np.ndarray,
    classes: Sequence[str],
    candidates: Sequence[str],
) -> np.ndarray:
    class_index = {speaker: index for index, speaker in enumerate(classes)}
    scores = np.full(candidate_probs.shape[:2], 1e-9, dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        if candidate in class_index:
            scores[:, candidate_index] = candidate_probs[:, candidate_index, class_index[candidate]]
    return _normalize_emissions(scores)


def _selector_scores(
    *,
    train_candidate_probs: np.ndarray,
    train_candidate_pred: np.ndarray,
    train_mixed_probs: np.ndarray,
    train_rows: Sequence[MaskRow],
    test_candidate_probs: np.ndarray,
    test_candidate_pred: np.ndarray,
    test_mixed_probs: np.ndarray,
    candidates: Sequence[str],
    classes: Sequence[str],
) -> np.ndarray:
    train_features = _candidate_features(
        train_candidate_probs,
        train_candidate_pred,
        candidates,
        classes,
        train_mixed_probs,
    )
    train_y = _candidate_labels(train_rows, candidates)
    test_features = _candidate_features(
        test_candidate_probs,
        test_candidate_pred,
        candidates,
        classes,
        test_mixed_probs,
    )
    if len(np.unique(train_y)) < 2:
        return _self_scores(test_candidate_probs, classes, candidates)
    selector = _fit_selector(train_features, train_y)
    scores = selector.predict_proba(test_features)[:, 1].reshape(
        test_candidate_probs.shape[0],
        len(candidates),
    )
    return _normalize_emissions(scores)


def _transition_logp(
    rows: Sequence[MaskRow],
    train_indices: Sequence[int],
    *,
    candidates: Sequence[str],
    alpha: float,
    sticky: float,
) -> np.ndarray:
    speaker_to_id = {speaker: index for index, speaker in enumerate(candidates)}
    counts = np.full((len(candidates), len(candidates)), float(alpha), dtype=np.float64)
    for indices in _window_indices(rows, train_indices).values():
        ids = [
            speaker_to_id[rows[index].truth]
            for index in indices
            if rows[index].truth in speaker_to_id
        ]
        for current, following in zip(ids, ids[1:]):
            counts[current, following] += 1.0
    for index in range(len(candidates)):
        counts[index, index] += float(sticky)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def _viterbi_fixed(emissions: np.ndarray, *, switch_penalty: float) -> np.ndarray:
    transition = np.full((emissions.shape[1], emissions.shape[1]), -float(switch_penalty))
    np.fill_diagonal(transition, 0.0)
    return _viterbi(emissions, transition)


def _viterbi(emissions: np.ndarray, transition_logp: np.ndarray) -> np.ndarray:
    logp = np.log(np.maximum(np.asarray(emissions, dtype=np.float64), 1e-9))
    if logp.shape[0] == 0:
        return np.zeros(0, dtype=np.int16)
    scores = np.empty_like(logp)
    previous = np.zeros(logp.shape, dtype=np.int16)
    scores[0] = logp[0]
    for row in range(1, logp.shape[0]):
        candidate_scores = scores[row - 1][:, None] + transition_logp
        previous[row] = np.argmax(candidate_scores, axis=0)
        scores[row] = logp[row] + candidate_scores[previous[row], np.arange(logp.shape[1])]
    path = np.zeros(logp.shape[0], dtype=np.int16)
    path[-1] = int(np.argmax(scores[-1]))
    for row in range(logp.shape[0] - 2, -1, -1):
        path[row] = previous[row + 1, path[row + 1]]
    return path


def _fill_decoded(
    predictions: List[str],
    rows: Sequence[MaskRow],
    test_indices: Sequence[int],
    emissions_by_index: Mapping[int, np.ndarray],
    candidates: Sequence[str],
    *,
    transition_logp: np.ndarray | None = None,
    switch_penalty: float | None = None,
) -> None:
    for indices in _window_indices(rows, test_indices).values():
        emissions = np.vstack([emissions_by_index[index] for index in indices])
        if transition_logp is not None:
            decoded = _viterbi(emissions, transition_logp)
        elif switch_penalty is not None:
            decoded = _viterbi_fixed(emissions, switch_penalty=float(switch_penalty))
        else:
            decoded = np.argmax(emissions, axis=1)
        for index, speaker_id in zip(indices, decoded):
            predictions[index] = str(candidates[int(speaker_id)])


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


def _evaluate_temporal(
    *,
    candidate_embeddings: np.ndarray,
    candidates: Sequence[str],
    rows: Sequence[MaskRow],
    clean_bank,
    training_items,
    mixed_embeddings: np.ndarray,
    switch_penalties: Sequence[float],
    transition_alpha: float,
    transition_sticky: float,
) -> Dict[str, object]:
    groups = sorted(_group_indices(rows))
    by_group = _group_indices(rows)
    emission_names = (
        "mixed",
        "candidate_self",
        "selector",
        "mixed_self_average",
        "mixed_selector_average",
        "mixed_self_max",
    )
    raw_predictions = {name: ["unknown"] * len(rows) for name in emission_names}
    hmm_predictions = {f"{name}_hmm": ["unknown"] * len(rows) for name in emission_names}
    penalty_predictions = {
        f"{name}_viterbi_penalty_{penalty:g}": ["unknown"] * len(rows)
        for name in emission_names
        for penalty in switch_penalties
    }
    fold_summaries: List[Dict[str, object]] = []

    for group in groups:
        test_indices = by_group[group]
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        train_rows = [rows[index] for index in train_indices]
        train_items = [item for item in training_items.values() if item.window.group != group]
        train_x, train_y = _rows_for_items(train_items)
        train_x = np.vstack([clean_bank.embeddings, train_x]).astype(np.float32)
        train_y = list(clean_bank.labels) + train_y
        lda = _fit_lda(train_x, train_y)
        classes = [str(item) for item in lda.classes_.tolist()]

        train_mixed_probs_raw = lda.predict_proba(mixed_embeddings[train_indices])
        test_mixed_probs_raw = lda.predict_proba(mixed_embeddings[test_indices])
        test_mixed_probs = _align_probs(test_mixed_probs_raw, classes, candidates)
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
        self_emissions = _self_scores(test_candidate_probs, classes, candidates)
        selector_emissions = _selector_scores(
            train_candidate_probs=train_candidate_probs,
            train_candidate_pred=train_candidate_pred,
            train_mixed_probs=train_mixed_probs_raw,
            train_rows=train_rows,
            test_candidate_probs=test_candidate_probs,
            test_candidate_pred=test_candidate_pred,
            test_mixed_probs=test_mixed_probs_raw,
            candidates=candidates,
            classes=classes,
        )
        emission_map = {
            "mixed": test_mixed_probs,
            "candidate_self": self_emissions,
            "selector": selector_emissions,
            "mixed_self_average": _normalize_emissions(test_mixed_probs + self_emissions),
            "mixed_selector_average": _normalize_emissions(test_mixed_probs + selector_emissions),
            "mixed_self_max": _normalize_emissions(np.maximum(test_mixed_probs, self_emissions)),
        }
        emissions_by_name: Dict[str, Dict[int, np.ndarray]] = {
            name: {
                global_index: emissions[local_index]
                for local_index, global_index in enumerate(test_indices)
            }
            for name, emissions in emission_map.items()
        }
        transition = _transition_logp(
            rows,
            train_indices,
            candidates=candidates,
            alpha=transition_alpha,
            sticky=transition_sticky,
        )
        for name in emission_names:
            _fill_decoded(
                raw_predictions[name],
                rows,
                test_indices,
                emissions_by_name[name],
                candidates,
            )
            _fill_decoded(
                hmm_predictions[f"{name}_hmm"],
                rows,
                test_indices,
                emissions_by_name[name],
                candidates,
                transition_logp=transition,
            )
            for penalty in switch_penalties:
                _fill_decoded(
                    penalty_predictions[f"{name}_viterbi_penalty_{penalty:g}"],
                    rows,
                    test_indices,
                    emissions_by_name[name],
                    candidates,
                    switch_penalty=penalty,
                )
        fold_summaries.append(
            {"group": group, "train_rows": len(train_indices), "test_rows": len(test_indices)}
        )

    scored = {
        name: _score_predictions(name, rows, predictions)
        for name, predictions in sorted(
            {**raw_predictions, **hmm_predictions, **penalty_predictions}.items()
        )
    }
    return {"scores": scored, "folds": fold_summaries}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply temporal decoding to all-candidate extractor speaker scores."
    )
    parser.add_argument(
        "--candidate-embeddings",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidates_lgo_full_big_s1600_embeddings.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_candidate_temporal_decode.json"),
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
    parser.add_argument(
        "--switch-penalties",
        default="0,0.25,0.5,1,2,3,5,8",
        help="Comma-separated fixed Viterbi switch penalties.",
    )
    parser.add_argument("--transition-alpha", type=float, default=0.5)
    parser.add_argument("--transition-sticky", type=float, default=5.0)
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
    switch_penalties = [
        float(item) for item in str(args.switch_penalties).split(",") if item.strip()
    ]
    result = _evaluate_temporal(
        candidate_embeddings=candidate_embeddings,
        candidates=candidates,
        rows=rows,
        clean_bank=clean_bank,
        training_items=training_items,
        mixed_embeddings=mixed_embeddings,
        switch_penalties=switch_penalties,
        transition_alpha=float(args.transition_alpha),
        transition_sticky=float(args.transition_sticky),
    )
    payload = {
        "model": "candidate_temporal_decode",
        "candidate_embeddings": str(args.candidate_embeddings.expanduser()),
        "selected_rows": len(rows),
        "selected_speakers": dict(Counter(row.truth for row in rows)),
        "candidates": list(candidates),
        "switch_penalties": switch_penalties,
        "transition_alpha": float(args.transition_alpha),
        "transition_sticky": float(args.transition_sticky),
        "scores": result["scores"],
        "folds": result["folds"],
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
