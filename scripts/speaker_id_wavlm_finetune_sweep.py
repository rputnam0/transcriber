from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_direct_word_source_sweep import (  # noqa: E402
    WordCrop,
    _load_word_crops,
    _score_mixed_baseline,
    _score_named,
    _split_value,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import _load_rows  # noqa: E402


class WavLMWordDataset(Dataset):
    def __init__(
        self,
        waves: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.waves = np.asarray(waves, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        source_index = int(self.indices[item])
        wave = np.array(self.waves[source_index], dtype=np.float32, copy=True)
        if self.augment:
            rng = np.random.default_rng(self.seed + source_index * 1009 + item)
            if rng.random() < 0.70:
                wave *= float(rng.uniform(0.80, 1.20))
            if rng.random() < 0.30:
                shift = int(rng.integers(-160, 161))
                wave = np.roll(wave, shift)
                if shift > 0:
                    wave[:shift] = 0.0
                elif shift < 0:
                    wave[shift:] = 0.0
            if rng.random() < 0.25:
                noise = rng.normal(0.0, float(rng.uniform(0.0005, 0.0035)), size=wave.shape)
                wave = wave + noise.astype(np.float32)
        wave = np.clip(wave, -1.0, 1.0).astype(np.float32)
        return torch.from_numpy(wave), torch.tensor(self.labels[source_index])


def _normalize_wave_batch(wave_batch: torch.Tensor) -> torch.Tensor:
    centered = wave_batch - wave_batch.mean(dim=1, keepdim=True)
    scale = torch.sqrt(torch.mean(centered * centered, dim=1, keepdim=True).clamp_min(1e-8))
    return centered / scale


class WavLMWordClassifier(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        pooling: str,
        unfreeze_last_layers: int,
    ) -> None:
        super().__init__()
        from transformers import WavLMForXVector

        pretrained = WavLMForXVector.from_pretrained(model_name)
        self.wavlm = pretrained.wavlm
        self.pooling = pooling
        hidden_size = int(pretrained.config.hidden_size)
        pooled_size = hidden_size if pooling.endswith("_mean") else hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )
        self._set_trainable_layers(int(unfreeze_last_layers))

    def _set_trainable_layers(self, unfreeze_last_layers: int) -> None:
        for parameter in self.wavlm.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.wavlm, "encoder", None), "layers", None)
        if layers is not None and unfreeze_last_layers > 0:
            for layer in layers[-unfreeze_last_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        # Keep final normalization trainable when present; it is tiny and helps domain adaptation.
        layer_norm = getattr(getattr(self.wavlm, "encoder", None), "layer_norm", None)
        if layer_norm is not None:
            for parameter in layer_norm.parameters():
                parameter.requires_grad = True

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        output = self.wavlm(
            _normalize_wave_batch(waves),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if self.pooling.startswith("last4"):
            hidden = torch.stack(list(hidden_states[-4:]), dim=0).mean(dim=0)
        elif self.pooling.startswith("last"):
            hidden = hidden_states[-1]
        else:
            raise ValueError(self.pooling)
        mean = hidden.mean(dim=1)
        if self.pooling.endswith("_mean"):
            pooled = mean
        else:
            std = hidden.std(dim=1, unbiased=False)
            pooled = torch.cat([mean, std], dim=1)
        return self.head(pooled)


def _class_weights(labels: np.ndarray, indices: Sequence[int], num_classes: int) -> torch.Tensor:
    selected = np.asarray(list(indices), dtype=np.int64)
    counts = np.bincount(labels[selected], minlength=num_classes)
    counts = np.maximum(counts.astype(np.float32), 1.0)
    weights = counts.sum() / counts
    weights = weights / max(float(np.mean(weights)), 1e-6)
    return torch.from_numpy(weights.astype(np.float32))


def _sampler(labels: np.ndarray, indices: Sequence[int], num_classes: int, seed: int):
    selected = np.asarray(list(indices), dtype=np.int64)
    counts = np.bincount(labels[selected], minlength=num_classes)
    counts = np.maximum(counts.astype(np.float64), 1.0)
    sample_weights = np.asarray([1.0 / counts[int(labels[index])] for index in selected])
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=max(len(selected), 1),
        replacement=True,
        generator=generator,
    )


def _train_fold(
    waves: np.ndarray,
    labels: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> tuple[List[int], Dict[str, object]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    num_classes = len(CORE_SPEAKERS)
    model = WavLMWordClassifier(
        model_name=str(args.model_name),
        num_classes=num_classes,
        pooling=str(args.pooling),
        unfreeze_last_layers=int(args.unfreeze_last_layers),
    ).to(device)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = WavLMWordDataset(waves, labels, train_indices, augment=True, seed=seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=_sampler(labels, train_indices, num_classes, seed),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(labels, train_indices, num_classes).to(device)
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and bool(args.amp)))
    model.train()
    losses: List[float] = []
    loader_iter = iter(loader)
    for step in range(1, int(args.train_steps) + 1):
        try:
            batch_wave, batch_label = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_wave, batch_label = next(loader_iter)
        batch_wave = batch_wave.to(device, non_blocking=True)
        batch_label = batch_label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda" and bool(args.amp))):
            logits = model(batch_wave)
            loss = criterion(logits, batch_label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, int(args.train_steps) // 5) == 0:
            with torch.no_grad():
                batch_acc = (logits.argmax(dim=1) == batch_label).float().mean().item()
            print(
                f"wavlm_step {step}/{args.train_steps} loss={losses[-1]:.4f} "
                f"batch_acc={batch_acc:.4f}",
                flush=True,
            )
    predictions = _predict_fold(model, waves, labels, test_indices, args=args, device=device)
    return predictions, {
        "mean_loss_last_20": float(np.mean(losses[-20:])) if losses else 0.0,
        "train_examples": len(train_indices),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
    }


def _predict_fold(
    model: nn.Module,
    waves: np.ndarray,
    labels: np.ndarray,
    test_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    device: str,
) -> List[int]:
    dataset = WavLMWordDataset(waves, labels, test_indices, augment=False, seed=int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    predictions: List[int] = []
    model.eval()
    with torch.inference_mode():
        for batch_wave, _batch_label in loader:
            wave = batch_wave.to(device, non_blocking=True)
            logits = model(wave)
            predictions.extend(int(index) for index in logits.argmax(dim=1).detach().cpu())
    return predictions


def _run_leave_out(
    waves: np.ndarray,
    crops: Sequence[WordCrop],
    *,
    split_mode: str,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, object]:
    speaker_to_id = {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}
    id_to_speaker = {index: speaker for speaker, index in speaker_to_id.items()}
    labels = np.asarray([speaker_to_id[crop.row.truth] for crop in crops], dtype=np.int64)
    split_values = sorted({_split_value(crop, split_mode) for crop in crops})
    if int(args.max_folds) > 0:
        split_values = split_values[: int(args.max_folds)]
    all_predictions = ["unknown"] * len(crops)
    evaluated_indices: List[int] = []
    fold_summaries: List[Dict[str, object]] = []
    for fold_index, split_value in enumerate(split_values):
        test_indices = [
            index
            for index, crop in enumerate(crops)
            if _split_value(crop, split_mode) == split_value
        ]
        test_set = set(test_indices)
        train_indices = [index for index in range(len(crops)) if index not in test_set]
        print(
            f"fold {fold_index + 1}/{len(split_values)} split={split_value} "
            f"train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        predicted_ids, fold_summary = _train_fold(
            waves,
            labels,
            train_indices,
            test_indices,
            args=args,
            device=device,
            seed=int(args.seed) + fold_index * 19,
        )
        for index, label_id in zip(test_indices, predicted_ids):
            all_predictions[index] = id_to_speaker[int(label_id)]
        evaluated_indices.extend(test_indices)
        fold_score = _score_named(
            f"wavlm_finetune/{split_mode}/{split_value}",
            crops,
            all_predictions,
            test_indices,
        )
        fold_summary.update(
            {
                "split": split_value,
                "test_examples": len(test_indices),
                "accuracy": fold_score["direct"]["accuracy"],
                "speaker_counts": dict(Counter(crops[index].row.truth for index in test_indices)),
            }
        )
        fold_summaries.append(fold_summary)

    hard_indices = [
        index
        for index in evaluated_indices
        if crops[index].row.target_share <= float(args.hard_max_share)
    ]
    return {
        "split_mode": split_mode,
        "folds": fold_summaries,
        "evaluated_examples": len(evaluated_indices),
        "all": _score_named(
            f"wavlm_finetune/{split_mode}/evaluated",
            crops,
            all_predictions,
            evaluated_indices,
        ),
        "hard": _score_named(
            f"wavlm_finetune/{split_mode}/hard_le_{args.hard_max_share:.2f}",
            crops,
            all_predictions,
            hard_indices,
        ),
        "mixed_baseline_all": _score_mixed_baseline(crops, evaluated_indices),
        "mixed_baseline_hard": _score_mixed_baseline(crops, hard_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune WavLM for direct known-speaker word ownership on mixed crops."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_wavlm_finetune_word_source.json"),
    )
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--pooling",
        choices=("last_mean", "last_mean_std", "last4_mean", "last4_mean_std"),
        default="last4_mean_std",
    )
    parser.add_argument("--crop-seconds", type=float, default=2.0)
    parser.add_argument("--word-window-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--hard-max-share", type=float, default=0.90)
    parser.add_argument("--split-mode", choices=("group", "session"), action="append", default=[])
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--unfreeze-last-layers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dominance = json.loads(args.dominance_json.expanduser().read_text(encoding="utf-8"))
    raw_rows = list(dominance.get("rows") or [])
    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    if int(args.eval_limit) > 0:
        rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    waves, crops = _load_word_crops(
        rows,
        raw_rows,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        sample_rate=int(args.sample_rate),
        crop_seconds=float(args.crop_seconds),
        word_window_seconds=float(args.word_window_seconds),
    )
    if not args.split_mode:
        args.split_mode = ["group"]
    print(
        f"loaded_wavlm_word_crops rows={len(crops)} waves={waves.shape} "
        f"speakers={dict(Counter(crop.row.truth for crop in crops))}",
        flush=True,
    )
    results = [
        _run_leave_out(waves, crops, split_mode=split_mode, args=args, device=device)
        for split_mode in args.split_mode
    ]
    payload = {
        "model": "wavlm_finetune_word_source",
        "model_name": str(args.model_name),
        "pooling": str(args.pooling),
        "crop_seconds": float(args.crop_seconds),
        "word_window_seconds": float(args.word_window_seconds),
        "max_target_share": float(args.max_target_share),
        "eval_limit": int(args.eval_limit),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "unfreeze_last_layers": int(args.unfreeze_last_layers),
        "speakers": list(CORE_SPEAKERS),
        "row_count": len(crops),
        "speaker_counts": dict(Counter(crop.row.truth for crop in crops)),
        "results": results,
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("split,subset,model_accuracy,baseline_accuracy,examples", flush=True)
    for result in results:
        for subset in ("all", "hard"):
            model_score = result[subset]["direct"]
            baseline_score = result[f"mixed_baseline_{subset}"]["direct"]
            print(
                ",".join(
                    [
                        str(result["split_mode"]),
                        subset,
                        f"{float(model_score['accuracy']):.4f}",
                        f"{float(baseline_score['accuracy']):.4f}",
                        str(model_score["examples"]),
                    ]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
