from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _fit_lda,
    _load_candidate_embeddings,
    _rows_for_items,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS  # noqa: E402
from speaker_id_oracle_mask_sweep import _score_direct  # noqa: E402
from speaker_id_temporal_decode_sweep import _load_clean_bank, _load_titanet_items  # noqa: E402
from speaker_id_word_window_sweep import (  # noqa: E402
    WordEmbeddingSet,
    _score_word_predictions,
)

OPTION_NAMES = (
    "mixed",
    "candidate_self",
    "candidate_class_sum",
    "candidate_class_max",
    "candidate_class_noisy_or",
)


@dataclass(frozen=True)
class CandidateEntry:
    features: np.ndarray
    options: Dict[str, str]


def _speaker_ids(speakers: Sequence[str]) -> Dict[str, int]:
    return {speaker: index for index, speaker in enumerate(speakers)}


def _align_probabilities(
    probabilities: np.ndarray,
    classes: Sequence[str],
    speakers: Sequence[str],
) -> np.ndarray:
    aligned = np.full((probabilities.shape[0], len(speakers)), 1e-8, dtype=np.float32)
    speaker_to_id = _speaker_ids(speakers)
    for column, speaker in enumerate(classes):
        idx = speaker_to_id.get(str(speaker))
        if idx is not None:
            aligned[:, idx] = probabilities[:, column]
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-8)
    return aligned.astype(np.float32)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] < 2:
        return np.ones(probabilities.shape[0], dtype=np.float32)
    sorted_probs = np.sort(probabilities, axis=1)
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)


def _dominance_maps(
    path: Path, hard_max_share: float
) -> tuple[Dict[tuple[str, int], float], set[tuple[str, int]]]:
    raw_rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    shares: Dict[tuple[str, int], float] = {}
    hard: set[tuple[str, int]] = set()
    for raw in raw_rows:
        key = (str(raw["window"]), int(raw["index"]))
        share = float(raw.get("target_share") or 0.0)
        shares[key] = share
        if share <= hard_max_share:
            hard.add(key)
    return shares, hard


def _candidate_feature_cache(
    *,
    candidate_embeddings: np.ndarray,
    candidate_rows,
    lda,
    classes: Sequence[str],
    speakers: Sequence[str],
) -> Dict[tuple[str, int], CandidateEntry]:
    flat = candidate_embeddings.reshape(
        candidate_embeddings.shape[0] * candidate_embeddings.shape[1],
        candidate_embeddings.shape[2],
    )
    probs = lda.predict_proba(flat).reshape(
        candidate_embeddings.shape[0],
        candidate_embeddings.shape[1],
        -1,
    )
    aligned = np.stack(
        [_align_probabilities(row, classes, speakers) for row in probs],
        axis=0,
    )
    speaker_count = len(speakers)
    self_scores = np.zeros((aligned.shape[0], speaker_count), dtype=np.float32)
    for speaker_index in range(speaker_count):
        if speaker_index < aligned.shape[1]:
            self_scores[:, speaker_index] = aligned[:, speaker_index, speaker_index]
    class_sum = aligned.sum(axis=1)
    class_max = aligned.max(axis=1)
    class_noisy_or = 1.0 - np.prod(1.0 - np.clip(aligned, 0.0, 1.0), axis=1)
    features = np.hstack(
        [
            np.ones((aligned.shape[0], 1), dtype=np.float32),
            self_scores,
            class_sum,
            class_max,
            class_noisy_or,
        ]
    ).astype(np.float32)
    speaker_array = np.asarray(speakers, dtype=object)
    option_ids = {
        "candidate_self": np.argmax(self_scores, axis=1),
        "candidate_class_sum": np.argmax(class_sum, axis=1),
        "candidate_class_max": np.argmax(class_max, axis=1),
        "candidate_class_noisy_or": np.argmax(class_noisy_or, axis=1),
    }
    return {
        (row.window, int(row.index)): CandidateEntry(
            features=features[index],
            options={
                name: str(speaker_array[int(values[index])]) for name, values in option_ids.items()
            },
        )
        for index, row in enumerate(candidate_rows)
    }


def _fit_scaler(train_items: Sequence[WordEmbeddingSet]) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.vstack([np.asarray(item.embeddings, dtype=np.float32) for item in train_items])
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def _build_item_features(
    item: WordEmbeddingSet,
    *,
    lda,
    classes: Sequence[str],
    speakers: Sequence[str],
    candidate_features: Mapping[tuple[str, int], CandidateEntry],
    include_embeddings: bool,
    embedding_mean: np.ndarray,
    embedding_std: np.ndarray,
) -> np.ndarray:
    mixed_probs = _align_probabilities(
        lda.predict_proba(np.asarray(item.embeddings, dtype=np.float32)),
        classes,
        speakers,
    )
    conf = mixed_probs.max(axis=1, keepdims=True)
    margin = _margin(mixed_probs)[:, None]
    empty_candidate = np.zeros((1 + (4 * len(speakers))), dtype=np.float32)
    rows: List[np.ndarray] = []
    for word_index in range(mixed_probs.shape[0]):
        key = (item.window.name, int(word_index))
        candidate = candidate_features.get(key)
        candidate_vector = candidate.features if candidate is not None else empty_candidate
        parts = [mixed_probs[word_index], conf[word_index], margin[word_index], candidate_vector]
        if include_embeddings:
            normalized = (
                np.asarray(item.embeddings[word_index : word_index + 1], dtype=np.float32)
                - embedding_mean
            ) / embedding_std
            parts.append(normalized.squeeze(0))
        rows.append(np.concatenate(parts).astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def _build_item_options(
    item: WordEmbeddingSet,
    *,
    lda,
    classes: Sequence[str],
    speakers: Sequence[str],
    candidate_features: Mapping[tuple[str, int], CandidateEntry],
) -> List[List[str]]:
    mixed_probs = _align_probabilities(
        lda.predict_proba(np.asarray(item.embeddings, dtype=np.float32)),
        classes,
        speakers,
    )
    speaker_array = np.asarray(speakers, dtype=object)
    mixed_predictions = [str(speaker_array[int(index)]) for index in np.argmax(mixed_probs, axis=1)]
    rows: List[List[str]] = []
    for word_index, mixed in enumerate(mixed_predictions):
        entry = candidate_features.get((item.window.name, int(word_index)))
        options = [mixed]
        for name in OPTION_NAMES[1:]:
            options.append(entry.options.get(name, mixed) if entry is not None else mixed)
        rows.append(options)
    return rows


class SequenceTagger(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int, layers: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.rnn = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            dropout=0.15 if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        output, _state = self.rnn(hidden)
        return self.head(output)


def _class_weights(labels: Sequence[int], classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights /= max(float(weights.mean()), 1e-6)
    return torch.from_numpy(weights.astype(np.float32))


def _train_sequence_model(
    train_sequences: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    input_dim: int,
    classes: int,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> SequenceTagger:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    model = SequenceTagger(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        classes=classes,
        layers=int(args.layers),
    ).to(device)
    all_labels = [int(label) for _x, y, _weights in train_sequences for label in y.tolist()]
    class_weights = _class_weights(all_labels, classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        total_loss = torch.zeros((), device=device)
        order = list(range(len(train_sequences)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        token_count = 0
        for index in order:
            features_np, labels_np, weights_np = train_sequences[index]
            features = torch.from_numpy(features_np).unsqueeze(0).to(device)
            labels = torch.from_numpy(labels_np).unsqueeze(0).to(device)
            token_weights = torch.from_numpy(weights_np).unsqueeze(0).to(device)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, classes),
                labels.reshape(-1),
                weight=class_weights,
                reduction="none",
            ).reshape_as(labels)
            weighted = (loss * token_weights).sum() / token_weights.sum().clamp_min(1.0)
            (weighted / max(len(order), 1)).backward()
            total_loss = total_loss + weighted.detach()
            token_count += int(labels.numel())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 5) == 0:
            print(
                f"sequence_step {step}/{args.train_steps} "
                f"loss={float(total_loss.cpu()) / max(len(order), 1):.4f} tokens={token_count}",
                flush=True,
            )
    model.eval()
    return model


def _predict_sequence(
    model: SequenceTagger,
    features: np.ndarray,
    *,
    speakers: Sequence[str],
    device: str,
) -> List[str]:
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).unsqueeze(0).to(device))
        ids = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
    return [str(speakers[int(index)]) for index in ids]


def _predict_actions(
    model: SequenceTagger,
    features: np.ndarray,
    options: Sequence[Sequence[str]],
    *,
    device: str,
) -> List[str]:
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).unsqueeze(0).to(device))
        ids = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
    return [
        str(row_options[min(int(action), len(row_options) - 1)])
        for action, row_options in zip(ids, options)
    ]


def _action_labels(item: WordEmbeddingSet, options: Sequence[Sequence[str]]) -> np.ndarray:
    labels: List[int] = []
    for truth, row_options in zip(item.truths, options):
        if str(row_options[0]) == str(truth):
            labels.append(0)
            continue
        chosen = 0
        for index, option in enumerate(row_options[1:], start=1):
            if str(option) == str(truth):
                chosen = index
                break
        labels.append(chosen)
    return np.asarray(labels, dtype=np.int64)


def _score_subset(
    items: Sequence[WordEmbeddingSet],
    predictions: Mapping[str, Sequence[str]],
    *,
    keys: set[tuple[str, int]],
) -> Dict[str, object]:
    truths: List[str] = []
    labels: List[str] = []
    for item in items:
        predicted = predictions[item.window.name]
        for index, (truth, label) in enumerate(zip(item.truths, predicted)):
            if (item.window.name, index) in keys:
                truths.append(str(truth))
                labels.append(str(label))
    direct = _score_direct(truths, labels)
    return {"direct": direct, "speakers": dict(Counter(truths))}


def _mixed_predictions(
    item_features: Mapping[str, np.ndarray],
    speakers: Sequence[str],
) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    for window_name, features in item_features.items():
        probs = features[:, : len(speakers)]
        predictions[window_name] = [str(speakers[int(index)]) for index in np.argmax(probs, axis=1)]
    return predictions


def _evaluate_sequence_tagger(
    *,
    items: Sequence[WordEmbeddingSet],
    candidate_embeddings: np.ndarray,
    candidates: Sequence[str],
    candidate_rows,
    clean_bank,
    dominance_json: Path,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, object]:
    del candidates
    speakers = list(args.speakers)
    speaker_to_id = _speaker_ids(speakers)
    _shares, hard_keys = _dominance_maps(dominance_json, float(args.hard_max_share))
    groups = sorted({item.window.group for item in items})
    predictions: Dict[str, List[str]] = {}
    router_predictions: Dict[str, List[str]] = {}
    mixed_all_predictions: Dict[str, List[str]] = {}
    fold_summaries: List[Dict[str, object]] = []

    for fold_index, group in enumerate(groups):
        train_items = [item for item in items if item.window.group != group]
        test_items = [item for item in items if item.window.group == group]
        train_x, train_y = _rows_for_items(train_items)
        lda = _fit_lda(
            np.vstack([clean_bank.embeddings, train_x]).astype(np.float32),
            list(clean_bank.labels) + train_y,
        )
        classes = [str(item) for item in lda.classes_.tolist()]
        candidate_cache = _candidate_feature_cache(
            candidate_embeddings=candidate_embeddings,
            candidate_rows=candidate_rows,
            lda=lda,
            classes=classes,
            speakers=speakers,
        )
        embed_mean, embed_std = _fit_scaler(train_items)
        train_sequences = []
        train_router_sequences = []
        for item in train_items:
            features = _build_item_features(
                item,
                lda=lda,
                classes=classes,
                speakers=speakers,
                candidate_features=candidate_cache,
                include_embeddings=bool(args.include_embeddings),
                embedding_mean=embed_mean,
                embedding_std=embed_std,
            )
            labels = np.asarray(
                [speaker_to_id[str(truth)] for truth in item.truths], dtype=np.int64
            )
            options = _build_item_options(
                item,
                lda=lda,
                classes=classes,
                speakers=speakers,
                candidate_features=candidate_cache,
            )
            action_labels = _action_labels(item, options)
            weights = np.asarray(
                [
                    float(args.hard_weight) if (item.window.name, index) in hard_keys else 1.0
                    for index in range(len(item.truths))
                ],
                dtype=np.float32,
            )
            train_sequences.append((features, labels, weights))
            train_router_sequences.append((features, action_labels, weights))
        input_dim = int(train_sequences[0][0].shape[1])
        print(
            f"sequence_fold group={group} train_windows={len(train_items)} "
            f"test_windows={len(test_items)} input_dim={input_dim}",
            flush=True,
        )
        model = None
        if not bool(args.skip_direct_tagger):
            model = _train_sequence_model(
                train_sequences,
                input_dim=input_dim,
                classes=len(speakers),
                args=args,
                device=device,
                seed=int(args.seed) + fold_index * 101,
            )
        router_model = _train_sequence_model(
            train_router_sequences,
            input_dim=input_dim,
            classes=len(OPTION_NAMES),
            args=args,
            device=device,
            seed=int(args.seed) + fold_index * 101 + 17,
        )
        test_feature_map: Dict[str, np.ndarray] = {}
        for item in test_items:
            features = _build_item_features(
                item,
                lda=lda,
                classes=classes,
                speakers=speakers,
                candidate_features=candidate_cache,
                include_embeddings=bool(args.include_embeddings),
                embedding_mean=embed_mean,
                embedding_std=embed_std,
            )
            test_feature_map[item.window.name] = features
            options = _build_item_options(
                item,
                lda=lda,
                classes=classes,
                speakers=speakers,
                candidate_features=candidate_cache,
            )
            if model is not None:
                predictions[item.window.name] = _predict_sequence(
                    model,
                    features,
                    speakers=speakers,
                    device=device,
                )
            else:
                predictions[item.window.name] = [row_options[0] for row_options in options]
            router_predictions[item.window.name] = _predict_actions(
                router_model,
                features,
                options,
                device=device,
            )
        mixed_all_predictions.update(_mixed_predictions(test_feature_map, speakers))
        fold_summaries.append(
            {
                "group": group,
                "train_windows": len(train_items),
                "test_windows": len(test_items),
                "train_tokens": int(sum(len(item.truths) for item in train_items)),
                "test_tokens": int(sum(len(item.truths) for item in test_items)),
            }
        )

    router_full = _score_word_predictions(items, router_predictions)
    router_hard = _score_subset(items, router_predictions, keys=hard_keys)
    mixed_full = _score_word_predictions(items, mixed_all_predictions)
    mixed_hard = _score_subset(items, mixed_all_predictions, keys=hard_keys)
    scores = {
        "sequence_action_router_full": router_full,
        "sequence_action_router_hard": router_hard,
        "mixed_lda_full": mixed_full,
        "mixed_lda_hard": mixed_hard,
    }
    if not bool(args.skip_direct_tagger):
        scores = {
            "sequence_tagger_full": _score_word_predictions(items, predictions),
            "sequence_tagger_hard": _score_subset(items, predictions, keys=hard_keys),
            **scores,
        }
    return {
        "scores": scores,
        "folds": fold_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leave-group sequence tagger over mixed word embeddings plus hard-row "
            "all-candidate extractor evidence."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--candidate-embeddings",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidates_lgo_full_big_s1600_embeddings.npz"),
    )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_sequence_candidate_tagger.json"),
    )
    parser.add_argument("--word-source", choices=("reference", "predicted"), default="reference")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--tolerance-seconds", type=float, default=0.35)
    parser.add_argument("--hard-max-share", type=float, default=0.90)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hard-weight", type=float, default=4.0)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--skip-direct-tagger", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.speakers = list(CORE_SPEAKERS)
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    candidate_embeddings, candidates, candidate_rows = _load_candidate_embeddings(
        args.candidate_embeddings.expanduser()
    )
    items = _load_titanet_items(
        prepared_root=args.prepared_root.expanduser(),
        cache_root=args.cache_root.expanduser(),
        word_source=str(args.word_source),
        window_seconds=float(args.window_seconds),
        speakers=args.speakers,
        tolerance_seconds=float(args.tolerance_seconds),
    )
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    result = _evaluate_sequence_tagger(
        items=items,
        candidate_embeddings=candidate_embeddings,
        candidates=candidates,
        candidate_rows=candidate_rows,
        clean_bank=clean_bank,
        dominance_json=args.dominance_json.expanduser(),
        args=args,
        device=device,
    )
    payload = {
        "model": "sequence_candidate_tagger",
        "word_source": str(args.word_source),
        "candidate_embeddings": str(args.candidate_embeddings.expanduser()),
        "window_count": len(items),
        "word_count": int(sum(len(item.truths) for item in items)),
        "hard_max_share": float(args.hard_max_share),
        "train_steps": int(args.train_steps),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "hard_weight": float(args.hard_weight),
        "include_embeddings": bool(args.include_embeddings),
        "skip_direct_tagger": bool(args.skip_direct_tagger),
        "scores": result["scores"],
        "folds": result["folds"],
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    for name, score in result["scores"].items():
        direct = score.get("direct", score)
        examples = direct.get("examples", direct.get("reference_words", 0))
        accuracy = direct.get("accuracy", direct.get("direct_word_accuracy", 0.0))
        print(
            ",".join(
                [
                    name,
                    str(examples),
                    f"{float(accuracy):.4f}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
