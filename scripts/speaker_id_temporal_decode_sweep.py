from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_architecture_sweep import CORE_SPEAKERS, TrainingSet  # noqa: E402
from speaker_id_word_window_sweep import (  # noqa: E402
    WordEmbeddingSet,
    _safe_name,
    _score_word_predictions,
    _smooth_majority,
    load_word_windows,
)
from transcriber.multitrack_eval import WordSpan  # noqa: E402


@dataclass
class ProbWindow:
    item: WordEmbeddingSet
    probabilities: np.ndarray


def _load_clean_bank(path: Path) -> TrainingSet:
    payload = np.load(path, allow_pickle=False)
    return TrainingSet(
        name=f"clean_bank:{path.name}",
        embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
        labels=[str(item) for item in payload["labels"].tolist()],
    )


def _load_titanet_items(
    *,
    prepared_root: Path,
    cache_root: Path,
    word_source: str,
    window_seconds: float,
    speakers: Sequence[str],
    tolerance_seconds: float,
) -> List[WordEmbeddingSet]:
    windows = load_word_windows(
        prepared_root,
        tolerance_seconds=tolerance_seconds,
        speakers=speakers,
        word_source=word_source,
    )
    items: List[WordEmbeddingSet] = []
    for window in windows:
        path = (
            cache_root
            / word_source
            / _safe_name(window.name)
            / f"titanet_small_{window_seconds:.2f}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        payload = np.load(path, allow_pickle=False)
        starts = np.asarray(payload["word_starts"], dtype=np.float32)
        ends = np.asarray(payload["word_ends"], dtype=np.float32)
        texts = [str(item) for item in payload["texts"].tolist()]
        predicted_words = [
            WordSpan(speaker="unknown", start=float(start), end=float(end), text=text)
            for start, end, text in zip(starts, ends, texts)
        ]
        items.append(
            WordEmbeddingSet(
                window=window,
                embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
                truths=[str(item) for item in payload["truths"].tolist()],
                predicted_words=predicted_words,
            )
        )
    return items


def _rows_for_items(items: Sequence[WordEmbeddingSet]) -> Tuple[np.ndarray, List[str]]:
    return (
        np.vstack([np.asarray(item.embeddings, dtype=np.float32) for item in items]),
        [truth for item in items for truth in item.truths],
    )


def _fit_lda_probabilities(
    *,
    train_items: Sequence[WordEmbeddingSet],
    test_items: Sequence[WordEmbeddingSet],
    clean_bank: TrainingSet,
    speakers: Sequence[str],
) -> List[ProbWindow]:
    train_x, train_y = _rows_for_items(train_items)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(
        np.vstack([clean_bank.embeddings, train_x]).astype(np.float32),
        list(clean_bank.labels) + train_y,
    )
    class_index = {speaker: index for index, speaker in enumerate(speakers)}
    prob_windows: List[ProbWindow] = []
    for item in test_items:
        raw = model.predict_proba(np.asarray(item.embeddings, dtype=np.float32))
        aligned = np.full((raw.shape[0], len(speakers)), 1e-9, dtype=np.float64)
        for column, speaker in enumerate(model.classes_):
            if speaker in class_index:
                aligned[:, class_index[str(speaker)]] = raw[:, column]
        aligned = aligned / aligned.sum(axis=1, keepdims=True)
        prob_windows.append(ProbWindow(item=item, probabilities=aligned))
    return prob_windows


def _viterbi_decode(
    probabilities: np.ndarray,
    *,
    speakers: Sequence[str],
    switch_penalty: float,
) -> List[str]:
    logp = np.log(np.maximum(np.asarray(probabilities, dtype=np.float64), 1e-9))
    rows, cols = logp.shape
    if rows == 0:
        return []
    transition = np.full((cols, cols), -float(switch_penalty), dtype=np.float64)
    np.fill_diagonal(transition, 0.0)
    scores = np.empty((rows, cols), dtype=np.float64)
    previous = np.zeros((rows, cols), dtype=np.int16)
    scores[0] = logp[0]
    for row in range(1, rows):
        candidate_scores = scores[row - 1][:, None] + transition
        previous[row] = np.argmax(candidate_scores, axis=0)
        scores[row] = logp[row] + candidate_scores[previous[row], np.arange(cols)]
    path = np.zeros(rows, dtype=np.int16)
    path[-1] = int(np.argmax(scores[-1]))
    for row in range(rows - 2, -1, -1):
        path[row] = previous[row + 1, path[row + 1]]
    return [str(speakers[index]) for index in path]


def _transition_logp(
    train_items: Sequence[WordEmbeddingSet],
    *,
    speakers: Sequence[str],
    alpha: float,
    sticky: float,
) -> np.ndarray:
    speaker_to_id = {speaker: index for index, speaker in enumerate(speakers)}
    counts = np.full((len(speakers), len(speakers)), float(alpha), dtype=np.float64)
    for item in train_items:
        ids = [speaker_to_id[truth] for truth in item.truths if truth in speaker_to_id]
        for current, following in zip(ids, ids[1:]):
            counts[current, following] += 1.0
    for index in range(len(speakers)):
        counts[index, index] += float(sticky)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def _hmm_decode(
    probabilities: np.ndarray,
    *,
    speakers: Sequence[str],
    transition_logp: np.ndarray,
) -> List[str]:
    logp = np.log(np.maximum(np.asarray(probabilities, dtype=np.float64), 1e-9))
    rows, cols = logp.shape
    if rows == 0:
        return []
    scores = np.empty((rows, cols), dtype=np.float64)
    previous = np.zeros((rows, cols), dtype=np.int16)
    scores[0] = logp[0]
    for row in range(1, rows):
        candidate_scores = scores[row - 1][:, None] + transition_logp
        previous[row] = np.argmax(candidate_scores, axis=0)
        scores[row] = logp[row] + candidate_scores[previous[row], np.arange(cols)]
    path = np.zeros(rows, dtype=np.int16)
    path[-1] = int(np.argmax(scores[-1]))
    for row in range(rows - 2, -1, -1):
        path[row] = previous[row + 1, path[row + 1]]
    return [str(speakers[index]) for index in path]


def _score_prob_windows(
    prob_windows: Sequence[ProbWindow],
    *,
    speakers: Sequence[str],
    switch_penalty: float,
) -> Dict[str, object]:
    predictions = {
        prob_window.item.window.name: _viterbi_decode(
            prob_window.probabilities,
            speakers=speakers,
            switch_penalty=switch_penalty,
        )
        for prob_window in prob_windows
    }
    return _score_word_predictions([prob_window.item for prob_window in prob_windows], predictions)


def _learned_transition_results(
    items: Sequence[WordEmbeddingSet],
    prob_windows: Sequence[ProbWindow],
    *,
    speakers: Sequence[str],
    alphas: Sequence[float],
    stickies: Sequence[float],
) -> List[Dict[str, object]]:
    by_window = {prob_window.item.window.name: prob_window for prob_window in prob_windows}
    groups = sorted({item.window.group for item in items})
    results: List[Dict[str, object]] = []
    for alpha in alphas:
        for sticky in stickies:
            predictions: Dict[str, List[str]] = {}
            for group in groups:
                train_items = [item for item in items if item.window.group != group]
                transition = _transition_logp(
                    train_items,
                    speakers=speakers,
                    alpha=alpha,
                    sticky=sticky,
                )
                for item in [item for item in items if item.window.group == group]:
                    predictions[item.window.name] = _hmm_decode(
                        by_window[item.window.name].probabilities,
                        speakers=speakers,
                        transition_logp=transition,
                    )
            results.append(
                {
                    "name": f"lda_hmm_alpha_{alpha:g}_sticky_{sticky:g}",
                    "selection": "global",
                    "transition_alpha": float(alpha),
                    "transition_sticky": float(sticky),
                    "word": _score_word_predictions(items, predictions),
                }
            )
    return results


def _leave_group_probabilities(
    items: Sequence[WordEmbeddingSet],
    *,
    clean_bank: TrainingSet,
    speakers: Sequence[str],
) -> List[ProbWindow]:
    prob_windows: List[ProbWindow] = []
    groups = sorted({item.window.group for item in items})
    for group in groups:
        train_items = [item for item in items if item.window.group != group]
        test_items = [item for item in items if item.window.group == group]
        prob_windows.extend(
            _fit_lda_probabilities(
                train_items=train_items,
                test_items=test_items,
                clean_bank=clean_bank,
                speakers=speakers,
            )
        )
    return sorted(prob_windows, key=lambda item: item.item.window.name)


def _raw_predictions(
    prob_windows: Sequence[ProbWindow],
    speakers: Sequence[str],
) -> Dict[str, List[str]]:
    return {
        prob_window.item.window.name: [
            str(speakers[index]) for index in np.argmax(prob_window.probabilities, axis=1)
        ]
        for prob_window in prob_windows
    }


def _global_penalty_results(
    prob_windows: Sequence[ProbWindow],
    *,
    speakers: Sequence[str],
    penalties: Sequence[float],
) -> List[Dict[str, object]]:
    items = [prob_window.item for prob_window in prob_windows]
    results: List[Dict[str, object]] = []
    raw_predictions = _raw_predictions(prob_windows, speakers)
    results.append(
        {
            "name": "lda_raw",
            "selection": "none",
            "word": _score_word_predictions(items, raw_predictions),
        }
    )
    for radius in (1, 2, 3, 4, 5):
        results.append(
            {
                "name": f"lda_majority_{(radius * 2) + 1}",
                "selection": "global",
                "word": _score_word_predictions(
                    items,
                    {
                        name: _smooth_majority(labels, radius)
                        for name, labels in raw_predictions.items()
                    },
                ),
            }
        )
    for penalty in penalties:
        predictions = {
            prob_window.item.window.name: _viterbi_decode(
                prob_window.probabilities,
                speakers=speakers,
                switch_penalty=penalty,
            )
            for prob_window in prob_windows
        }
        results.append(
            {
                "name": f"lda_viterbi_penalty_{penalty:g}",
                "selection": "global",
                "switch_penalty": float(penalty),
                "word": _score_word_predictions(items, predictions),
            }
        )
    return results


def _nested_penalty_result(
    items: Sequence[WordEmbeddingSet],
    *,
    clean_bank: TrainingSet,
    speakers: Sequence[str],
    penalties: Sequence[float],
) -> Dict[str, object]:
    groups = sorted({item.window.group for item in items})
    predictions: Dict[str, List[str]] = {}
    chosen: Dict[str, float] = {}
    inner_scores_by_outer: Dict[str, Dict[str, object]] = {}
    for outer_group in groups:
        outer_train = [item for item in items if item.window.group != outer_group]
        outer_test = [item for item in items if item.window.group == outer_group]
        inner_prob_windows: List[ProbWindow] = []
        inner_groups = sorted({item.window.group for item in outer_train})
        for inner_group in inner_groups:
            inner_train = [item for item in outer_train if item.window.group != inner_group]
            inner_test = [item for item in outer_train if item.window.group == inner_group]
            inner_prob_windows.extend(
                _fit_lda_probabilities(
                    train_items=inner_train,
                    test_items=inner_test,
                    clean_bank=clean_bank,
                    speakers=speakers,
                )
            )
        scored_penalties = [
            (
                penalty,
                _score_prob_windows(inner_prob_windows, speakers=speakers, switch_penalty=penalty),
            )
            for penalty in penalties
        ]
        penalty, score = max(
            scored_penalties,
            key=lambda item: (
                float(item[1]["accuracy"]),
                float(item[1]["direct_word_accuracy"]),
                -float(item[0]),
            ),
        )
        chosen[outer_group] = float(penalty)
        inner_scores_by_outer[outer_group] = {
            "switch_penalty": float(penalty),
            "word": score,
        }
        outer_prob_windows = _fit_lda_probabilities(
            train_items=outer_train,
            test_items=outer_test,
            clean_bank=clean_bank,
            speakers=speakers,
        )
        for prob_window in outer_prob_windows:
            predictions[prob_window.item.window.name] = _viterbi_decode(
                prob_window.probabilities,
                speakers=speakers,
                switch_penalty=penalty,
            )
    return {
        "name": "lda_viterbi_nested_penalty",
        "selection": "nested_leave_group_out",
        "chosen_penalties": chosen,
        "inner_scores_by_outer": inner_scores_by_outer,
        "word": _score_word_predictions(items, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal Viterbi decoding on cached Titanet word speaker embeddings."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window"))
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument("--word-source", choices=("predicted", "reference"), default="reference")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--switch-penalties",
        default="0,0.5,1,2,3,5,7.5,10,15,20,30",
    )
    parser.add_argument("--transition-alphas", default="0.1,1,5,10,25,50,100")
    parser.add_argument("--transition-stickies", default="0,10,50,100,250,500,1000,2000,5000")
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--speakers", default=",".join(CORE_SPEAKERS))
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    speakers = tuple(item.strip() for item in args.speakers.split(",") if item.strip())
    penalties = [
        float(item.strip()) for item in str(args.switch_penalties).split(",") if item.strip()
    ]
    transition_alphas = [
        float(item.strip()) for item in str(args.transition_alphas).split(",") if item.strip()
    ]
    transition_stickies = [
        float(item.strip()) for item in str(args.transition_stickies).split(",") if item.strip()
    ]
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    items = _load_titanet_items(
        prepared_root=args.prepared_root.expanduser(),
        cache_root=args.cache_root.expanduser(),
        word_source=str(args.word_source),
        window_seconds=float(args.window_seconds),
        speakers=speakers,
        tolerance_seconds=float(args.tolerance),
    )
    prob_windows = _leave_group_probabilities(items, clean_bank=clean_bank, speakers=speakers)
    results = _global_penalty_results(prob_windows, speakers=speakers, penalties=penalties)
    results.extend(
        _learned_transition_results(
            items,
            prob_windows,
            speakers=speakers,
            alphas=transition_alphas,
            stickies=transition_stickies,
        )
    )
    if not args.skip_nested:
        results.append(
            _nested_penalty_result(
                items,
                clean_bank=clean_bank,
                speakers=speakers,
                penalties=penalties,
            )
        )
    results.sort(
        key=lambda item: (
            float(item["word"]["accuracy"]),
            float(item["word"]["direct_word_accuracy"]),
        ),
        reverse=True,
    )
    payload = {
        "prepared_root": str(args.prepared_root.expanduser()),
        "cache_root": str(args.cache_root.expanduser()),
        "clean_bank": str(args.clean_bank.expanduser()),
        "word_source": str(args.word_source),
        "window_seconds": float(args.window_seconds),
        "speakers": speakers,
        "switch_penalties": penalties,
        "transition_alphas": transition_alphas,
        "transition_stickies": transition_stickies,
        "windows": [
            {
                "name": item.window.name,
                "group": item.window.group,
                "words": len(item.truths),
                "speakers": dict(Counter(item.truths)),
            }
            for item in items
        ],
        "results": results,
    }
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,selection,accuracy,matched_accuracy,coverage,direct_word_accuracy,correct")
    for result in results[:20]:
        word = result["word"]
        print(
            ",".join(
                [
                    str(result["name"]),
                    str(result.get("selection") or ""),
                    f"{float(word['accuracy']):.4f}",
                    f"{float(word['matched_accuracy']):.4f}",
                    f"{float(word['coverage']):.4f}",
                    f"{float(word['direct_word_accuracy']):.4f}",
                    str(word["correct_words"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
