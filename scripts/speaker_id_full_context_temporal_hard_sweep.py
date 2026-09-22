from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import CORE_SPEAKERS  # noqa: E402
from speaker_id_oracle_mask_sweep import _load_rows  # noqa: E402
from speaker_id_temporal_decode_sweep import (  # noqa: E402
    ProbWindow,
    _fit_lda_probabilities,
    _hmm_decode,
    _leave_group_probabilities,
    _load_clean_bank,
    _load_titanet_items,
    _raw_predictions,
    _transition_logp,
    _viterbi_decode,
)
from speaker_id_word_window_sweep import (  # noqa: E402
    WordEmbeddingSet,
    _score_word_predictions,
    _smooth_majority,
)


def _hard_keys(dominance_json: Path, max_target_share: float) -> set[tuple[str, int]]:
    return {(row.window, int(row.index)) for row in _load_rows(dominance_json, max_target_share)}


def _score_subset(
    items: Sequence[WordEmbeddingSet],
    predictions: Dict[str, List[str]],
    *,
    keys: set[tuple[str, int]],
) -> Dict[str, object]:
    total = 0
    correct = 0
    speakers: Counter[str] = Counter()
    for item in items:
        labels = predictions.get(item.window.name)
        if labels is None:
            raise KeyError(item.window.name)
        for index, (truth, pred) in enumerate(zip(item.truths, labels)):
            if (item.window.name, index) not in keys:
                continue
            total += 1
            correct += int(str(truth) == str(pred))
            speakers[str(truth)] += 1
    return {
        "examples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "speakers": dict(speakers),
    }


def _result_payload(
    name: str,
    *,
    selection: str,
    items: Sequence[WordEmbeddingSet],
    predictions: Dict[str, List[str]],
    hard: set[tuple[str, int]],
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    payload = {
        "name": name,
        "selection": selection,
        "full": _score_word_predictions(items, predictions),
        "hard": _score_subset(items, predictions, keys=hard),
    }
    if extra:
        payload.update(extra)
    return payload


def _global_results(
    prob_windows: Sequence[ProbWindow],
    *,
    speakers: Sequence[str],
    penalties: Sequence[float],
    hard: set[tuple[str, int]],
) -> List[Dict[str, object]]:
    items = [prob_window.item for prob_window in prob_windows]
    results: List[Dict[str, object]] = []
    raw_predictions = _raw_predictions(prob_windows, speakers)
    results.append(
        _result_payload(
            "lda_raw",
            selection="none",
            items=items,
            predictions=raw_predictions,
            hard=hard,
        )
    )
    for radius in (1, 2, 3, 4, 5):
        results.append(
            _result_payload(
                f"lda_majority_{(radius * 2) + 1}",
                selection="global",
                items=items,
                predictions={
                    name: _smooth_majority(labels, radius)
                    for name, labels in raw_predictions.items()
                },
                hard=hard,
            )
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
            _result_payload(
                f"lda_viterbi_penalty_{penalty:g}",
                selection="global",
                items=items,
                predictions=predictions,
                hard=hard,
                extra={"switch_penalty": float(penalty)},
            )
        )
    return results


def _learned_transition_hard_results(
    items: Sequence[WordEmbeddingSet],
    prob_windows: Sequence[ProbWindow],
    *,
    speakers: Sequence[str],
    alphas: Sequence[float],
    stickies: Sequence[float],
    hard: set[tuple[str, int]],
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
                _result_payload(
                    f"lda_hmm_alpha_{alpha:g}_sticky_{sticky:g}",
                    selection="global",
                    items=items,
                    predictions=predictions,
                    hard=hard,
                    extra={"transition_alpha": float(alpha), "transition_sticky": float(sticky)},
                )
            )
    return results


def _nested_hard_result(
    items: Sequence[WordEmbeddingSet],
    *,
    clean_bank,
    speakers: Sequence[str],
    penalties: Sequence[float],
    hard: set[tuple[str, int]],
) -> Dict[str, object]:
    groups = sorted({item.window.group for item in items})
    predictions: Dict[str, List[str]] = {}
    chosen: Dict[str, float] = {}
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
        scored_penalties = _global_results(
            inner_prob_windows,
            speakers=speakers,
            penalties=penalties,
            hard=hard,
        )
        viterbi = [item for item in scored_penalties if item["name"].startswith("lda_viterbi")]
        best = max(
            viterbi,
            key=lambda item: (
                float(item["hard"]["accuracy"]),
                float(item["full"]["accuracy"]),
                -float(item["switch_penalty"]),
            ),
        )
        penalty = float(best["switch_penalty"])
        chosen[outer_group] = penalty
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
    return _result_payload(
        "lda_viterbi_nested_penalty_hard_selected",
        selection="nested_leave_group_out_hard",
        items=items,
        predictions=predictions,
        hard=hard,
        extra={"chosen_penalties": chosen},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode full word sequences with mixed Titanet probabilities, then score hard-overlap "
            "rows to test whether easy words anchor overlap labels."
        )
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
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument("--word-source", choices=("reference",), default="reference")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument(
        "--switch-penalties",
        default="0,0.5,1,2,3,5,7.5,10,15,20,30",
    )
    parser.add_argument("--transition-alphas", default="0.1,1,5,10,25,50,100")
    parser.add_argument("--transition-stickies", default="0,10,50,100,250,500,1000,2000,5000")
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--speakers", default=",".join(CORE_SPEAKERS))
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_full_context_temporal_hard.json")
    )
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
    hard = _hard_keys(args.dominance_json.expanduser(), float(args.max_target_share))
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
    results = _global_results(prob_windows, speakers=speakers, penalties=penalties, hard=hard)
    results.extend(
        _learned_transition_hard_results(
            items,
            prob_windows,
            speakers=speakers,
            alphas=transition_alphas,
            stickies=transition_stickies,
            hard=hard,
        )
    )
    if not args.skip_nested:
        results.append(
            _nested_hard_result(
                items,
                clean_bank=clean_bank,
                speakers=speakers,
                penalties=penalties,
                hard=hard,
            )
        )
    results.sort(
        key=lambda item: (
            float(item["hard"]["accuracy"]),
            float(item["full"]["accuracy"]),
        ),
        reverse=True,
    )
    payload = {
        "prepared_root": str(args.prepared_root.expanduser()),
        "cache_root": str(args.cache_root.expanduser()),
        "clean_bank": str(args.clean_bank.expanduser()),
        "dominance_json": str(args.dominance_json.expanduser()),
        "word_source": str(args.word_source),
        "window_seconds": float(args.window_seconds),
        "speakers": speakers,
        "hard_rows": len(hard),
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
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,selection,full_accuracy,hard_accuracy,hard_correct,hard_examples")
    for result in results[:25]:
        print(
            ",".join(
                [
                    str(result["name"]),
                    str(result.get("selection") or ""),
                    f"{float(result['full']['accuracy']):.4f}",
                    f"{float(result['hard']['accuracy']):.4f}",
                    str(result["hard"]["correct"]),
                    str(result["hard"]["examples"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
