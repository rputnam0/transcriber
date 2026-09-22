from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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

from speaker_id_learned_mask_sweep import CORE_SPEAKERS  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _load_audio,
    _load_rows,
    _score_direct,
    _score_slices,
    _slice_wave,
)
from speaker_id_word_window_sweep import _safe_name, _window_group  # noqa: E402


@dataclass(frozen=True)
class WordCrop:
    row: MaskRow
    group: str
    session: str
    mixed_pred: str
    mixed_correct: bool


class WordCropDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
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

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[item])
        wave = np.array(self.waves[source_index], dtype=np.float32, copy=True)
        if self.augment:
            rng = np.random.default_rng(self.seed + source_index * 1009 + item)
            if rng.random() < 0.75:
                wave *= float(rng.uniform(0.75, 1.25))
            if rng.random() < 0.35:
                shift = int(rng.integers(-240, 241))
                wave = np.roll(wave, shift)
                if shift > 0:
                    wave[:shift] = 0.0
                elif shift < 0:
                    wave[shift:] = 0.0
            if rng.random() < 0.30:
                noise = rng.normal(0.0, float(rng.uniform(0.001, 0.006)), size=wave.shape)
                wave = wave + noise.astype(np.float32)
        wave = np.clip(wave, -1.0, 1.0).astype(np.float32)
        return torch.from_numpy(wave).unsqueeze(0), torch.tensor(self.labels[source_index])


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class RawWordSpeakerNet(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=21, stride=4, padding=10),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            ConvBlock(32, 48, stride=2),
            ConvBlock(48, 72, stride=2),
            ConvBlock(72, 96, stride=2),
            ConvBlock(96, 128, stride=2),
            ConvBlock(128, 160, stride=2),
        )
        self.head = nn.Sequential(
            nn.Linear(320, 160),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(160, classes),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        features = self.encoder(wave)
        mean = features.mean(dim=-1)
        std = features.std(dim=-1, unbiased=False)
        return self.head(torch.cat([mean, std], dim=1))


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: Tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(5, 5),
                stride=stride,
                padding=(2, 2),
            ),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LogMagSpeakerNet(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            Conv2dBlock(1, 32, stride=(2, 2)),
            Conv2dBlock(32, 48, stride=(2, 2)),
            Conv2dBlock(48, 72, stride=(2, 2)),
            Conv2dBlock(72, 96, stride=(2, 2)),
            Conv2dBlock(96, 128, stride=(2, 2)),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 160),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(160, classes),
        )

    def forward(self, logmag: torch.Tensor) -> torch.Tensor:
        features = self.encoder(logmag)
        mean = features.mean(dim=(-2, -1))
        std = features.std(dim=(-2, -1), unbiased=False)
        return self.head(torch.cat([mean, std], dim=1))


def _make_model(architecture: str, classes: int) -> nn.Module:
    if architecture == "raw_cnn":
        return RawWordSpeakerNet(classes=classes)
    if architecture == "logmag_cnn":
        return LogMagSpeakerNet(classes=classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def _model_input(
    wave: torch.Tensor,
    *,
    architecture: str,
    args: argparse.Namespace,
    window: torch.Tensor | None,
) -> torch.Tensor:
    if architecture == "raw_cnn":
        return wave
    if architecture != "logmag_cnn":
        raise ValueError(f"Unknown architecture: {architecture}")
    if window is None:
        raise ValueError("STFT window is required for logmag_cnn")
    stft = torch.stft(
        wave.squeeze(1),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        win_length=int(args.n_fft),
        window=window,
        return_complex=True,
    )
    logmag = torch.log1p(torch.abs(stft)).unsqueeze(1)
    mean = logmag.mean(dim=(-2, -1), keepdim=True)
    std = logmag.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return (logmag - mean) / std


def _session_for_window(window: str) -> str:
    if window.startswith("short_segment_slice/"):
        return "Session61"
    return window.split("/", maxsplit=1)[0]


def _normalize_crop(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if wave.size else 0.0
    if rms > 1e-6:
        wave = wave / rms
    peak = max(float(np.max(np.abs(wave))), 1e-6)
    if peak > 1.0:
        wave = wave / peak
    return wave.astype(np.float32)


def _word_cache_path(root: Path, window: str, window_seconds: float) -> Path:
    return root / "reference" / _safe_name(window) / f"titanet_small_{window_seconds:.2f}.npz"


def _load_word_crops(
    rows: Sequence[MaskRow],
    raw_rows: Sequence[Mapping[str, object]],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    sample_rate: int,
    crop_seconds: float,
    word_window_seconds: float,
) -> Tuple[np.ndarray, List[WordCrop]]:
    raw_by_key = {
        (str(raw["window"]), int(raw["index"])): raw
        for raw in raw_rows
        if "window" in raw and "index" in raw
    }
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    samples = int(round(crop_seconds * sample_rate))
    waves: List[np.ndarray] = []
    crops: List[WordCrop] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = np.load(
            _word_cache_path(titanet_cache_root, window_name, word_window_seconds),
            allow_pickle=False,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)
        for row in window_rows:
            raw = raw_by_key[(row.window, row.index)]
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - crop_seconds / 2.0) * sample_rate))
            wave = _slice_wave(mixed, start_sample, start_sample + samples)
            waves.append(_normalize_crop(wave))
            crops.append(
                WordCrop(
                    row=row,
                    group=_window_group(row.window),
                    session=_session_for_window(row.window),
                    mixed_pred=str(raw.get("mixed_pred") or "unknown"),
                    mixed_correct=bool(raw.get("mixed_correct")),
                )
            )
    return np.stack(waves).astype(np.float32), crops


def _split_value(crop: WordCrop, mode: str) -> str:
    if mode == "group":
        return crop.group
    if mode == "session":
        return crop.session
    raise ValueError(f"Unknown split mode: {mode}")


def _class_weights(labels: np.ndarray, indices: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels[np.asarray(list(indices), dtype=np.int64)], minlength=num_classes)
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
) -> Tuple[List[int], List[np.ndarray], Dict[str, object]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    num_classes = len(CORE_SPEAKERS)
    architecture = str(args.architecture)
    model = _make_model(architecture, classes=num_classes).to(device)
    stft_window = (
        torch.hann_window(int(args.n_fft), device=device) if architecture == "logmag_cnn" else None
    )
    train_dataset = WordCropDataset(waves, labels, train_indices, augment=True, seed=seed)
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
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
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
        logits = model(
            _model_input(batch_wave, architecture=architecture, args=args, window=stft_window)
        )
        loss = criterion(logits, batch_label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, int(args.train_steps) // 4) == 0:
            with torch.no_grad():
                train_acc = (logits.argmax(dim=1) == batch_label).float().mean().item()
            print(
                f"fold_train step={step}/{args.train_steps} "
                f"loss={losses[-1]:.4f} batch_acc={train_acc:.4f}",
                flush=True,
            )
    return (
        _predict_fold(model, waves, labels, test_indices, args=args, device=device),
        [],
        {
            "mean_loss_last_20": float(np.mean(losses[-20:])) if losses else 0.0,
            "train_examples": len(train_indices),
        },
    )


def _predict_fold(
    model: nn.Module,
    waves: np.ndarray,
    labels: np.ndarray,
    test_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    device: str,
) -> List[int]:
    dataset = WordCropDataset(waves, labels, test_indices, augment=False, seed=int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    predictions: List[int] = []
    model.eval()
    architecture = str(args.architecture)
    stft_window = (
        torch.hann_window(int(args.n_fft), device=device) if architecture == "logmag_cnn" else None
    )
    with torch.inference_mode():
        for batch_wave, _batch_label in loader:
            wave = batch_wave.to(device, non_blocking=True)
            logits = model(
                _model_input(wave, architecture=architecture, args=args, window=stft_window)
            )
            predictions.extend(int(index) for index in logits.argmax(dim=1).detach().cpu())
    return predictions


def _score_named(
    name: str,
    crops: Sequence[WordCrop],
    predictions: Sequence[str],
    indices: Sequence[int],
) -> Dict[str, object]:
    selected_crops = [crops[index] for index in indices]
    selected_predictions = [predictions[index] for index in indices]
    rows = [crop.row for crop in selected_crops]
    direct = _score_direct([row.truth for row in rows], selected_predictions)
    return {
        "name": name,
        "direct": direct,
        "share_slices": _score_slices(rows, selected_predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, selected_predictions, field="active_5pct"),
    }


def _score_mixed_baseline(crops: Sequence[WordCrop], indices: Sequence[int]) -> Dict[str, object]:
    predictions = [crops[index].mixed_pred for index in indices]
    rows = [crops[index].row for index in indices]
    return {
        "name": "mixed_titanet_lda_from_dominance_json",
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
    }


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
    all_predictions = ["unknown"] * len(crops)
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
        predicted_ids, _probs, fold_summary = _train_fold(
            waves,
            labels,
            train_indices,
            test_indices,
            args=args,
            device=device,
            seed=int(args.seed) + fold_index * 17,
        )
        for index, label_id in zip(test_indices, predicted_ids):
            all_predictions[index] = id_to_speaker[int(label_id)]
        fold_score = _score_named(
            f"direct_raw_cnn/{split_mode}/{split_value}",
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

    all_indices = list(range(len(crops)))
    hard_indices = [
        index for index, crop in enumerate(crops) if crop.row.target_share <= args.hard_max_share
    ]
    return {
        "split_mode": split_mode,
        "folds": fold_summaries,
        "all": _score_named(
            f"direct_{args.architecture}/{split_mode}/all",
            crops,
            all_predictions,
            all_indices,
        ),
        "hard": _score_named(
            f"direct_{args.architecture}/{split_mode}/hard_le_{args.hard_max_share:.2f}",
            crops,
            all_predictions,
            hard_indices,
        ),
        "mixed_baseline_all": _score_mixed_baseline(crops, all_indices),
        "mixed_baseline_hard": _score_mixed_baseline(crops, hard_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a direct raw-audio word-source classifier on flat mixed audio."
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
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/codex_direct_word_source.json"))
    parser.add_argument("--crop-seconds", type=float, default=2.0)
    parser.add_argument("--word-window-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--architecture", choices=("raw_cnn", "logmag_cnn"), default="logmag_cnn")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--hard-max-share", type=float, default=0.90)
    parser.add_argument("--split-mode", choices=("group", "session"), action="append", default=[])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dominance = json.loads(args.dominance_json.expanduser().read_text(encoding="utf-8"))
    raw_rows = list(dominance.get("rows") or [])
    rows = _load_rows(args.dominance_json.expanduser(), 1.0)
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
        f"loaded_word_crops rows={len(crops)} waves={waves.shape} "
        f"speakers={dict(Counter(crop.row.truth for crop in crops))}",
        flush=True,
    )
    results = [
        _run_leave_out(waves, crops, split_mode=split_mode, args=args, device=device)
        for split_mode in args.split_mode
    ]
    payload = {
        "model": f"direct_{args.architecture}",
        "prepared_root": str(args.prepared_root.expanduser()),
        "dominance_json": str(args.dominance_json.expanduser()),
        "crop_seconds": float(args.crop_seconds),
        "word_window_seconds": float(args.word_window_seconds),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
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
