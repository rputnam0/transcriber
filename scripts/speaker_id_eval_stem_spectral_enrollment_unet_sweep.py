from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_eval_stem_enrollment_extractor_sweep import _one_hot  # noqa: E402
from speaker_id_eval_stem_target_extractor_sweep import (  # noqa: E402
    _evaluate_embeddings_with_split,
    _load_embeddings,
    _load_eval_stem_pairs,
    _safe_group,
    _split_group,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_unet_mask_extractor_sweep import ConvBlock, _stft  # noqa: E402


class SpectralEnrollmentUNet(nn.Module):
    def __init__(self, *, input_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 3, kernel_size=4, stride=2, padding=1
        )
        self.enc3 = ConvBlock(base_channels * 3, base_channels * 3)
        self.down3 = nn.Conv2d(
            base_channels * 3, base_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up3 = ConvBlock(base_channels * 7, base_channels * 3)
        self.up2 = ConvBlock(base_channels * 5, base_channels * 2)
        self.up1 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Sequential(nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(value)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        hidden = self.bottleneck(self.down3(x3))
        hidden = torch.nn.functional.interpolate(
            hidden, size=x3.shape[-2:], mode="bilinear", align_corners=False
        )
        hidden = self.up3(torch.cat([hidden, x3], dim=1))
        hidden = torch.nn.functional.interpolate(
            hidden, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        hidden = self.up2(torch.cat([hidden, x2], dim=1))
        hidden = torch.nn.functional.interpolate(
            hidden, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )
        hidden = self.up1(torch.cat([hidden, x1], dim=1))
        return self.out(hidden)


def _speaker_indices(labels: Sequence[str], indices: Sequence[int]) -> Dict[str, List[int]]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index in indices:
        by_speaker[str(labels[index])].append(int(index))
    return by_speaker


def _balanced_indices(
    labels: Sequence[str],
    train_indices: Sequence[int],
    *,
    batch_size: int,
    rng: random.Random,
) -> List[int]:
    by_speaker = _speaker_indices(labels, train_indices)
    speakers = [speaker for speaker in CORE_SPEAKERS if by_speaker.get(speaker)]
    return [rng.choice(by_speaker[rng.choice(speakers)]) for _ in range(batch_size)]


def _random_enrollment_indices(
    labels: Sequence[str],
    train_indices: Sequence[int],
    target_labels: Sequence[str],
    *,
    count: int,
    rng: random.Random,
) -> tuple[List[List[int]], List[List[int]]]:
    by_speaker = _speaker_indices(labels, train_indices)
    positive: List[List[int]] = []
    negative: List[List[int]] = []
    for label in target_labels:
        pos_pool = by_speaker[str(label)]
        neg_speakers = [
            speaker for speaker in CORE_SPEAKERS if speaker != label and by_speaker.get(speaker)
        ]
        positive.append([rng.choice(pos_pool) for _ in range(count)])
        negative.append([rng.choice(by_speaker[rng.choice(neg_speakers)]) for _ in range(count)])
    return positive, negative


def _fixed_enrollment_indices(
    labels: Sequence[str],
    train_indices: Sequence[int],
    target_labels: Sequence[str],
    *,
    count: int,
) -> tuple[List[List[int]], List[List[int]]]:
    by_speaker = _speaker_indices(labels, train_indices)
    positive: List[List[int]] = []
    negative: List[List[int]] = []
    for label in target_labels:
        pos_pool = by_speaker[str(label)]
        neg_pools = [
            by_speaker[speaker]
            for speaker in CORE_SPEAKERS
            if speaker != label and by_speaker.get(speaker)
        ]
        positive.append([pos_pool[index % len(pos_pool)] for index in range(count)])
        neg_rows: List[int] = []
        for index in range(count):
            pool = neg_pools[index % len(neg_pools)]
            neg_rows.append(pool[(index // len(neg_pools)) % len(pool)])
        negative.append(neg_rows)
    return positive, negative


def _enrollment_source(source: str, mixtures: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if source == "target":
        return targets
    if source == "mixed":
        return mixtures
    raise ValueError(f"Unknown enrollment source: {source}")


def _gather_enrollments(source: np.ndarray, indices: Sequence[Sequence[int]]) -> np.ndarray:
    return np.stack([source[list(row_indices)] for row_indices in indices]).astype(np.float32)


def _spectral_context(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    frames: int,
) -> torch.Tensor:
    batch, count, samples = positive.shape
    stacked = torch.cat([positive, negative], dim=1).reshape(batch * count * 2, samples)
    logmag = torch.log1p(
        torch.abs(_stft(stacked, n_fft=n_fft, hop_length=hop_length, window=window))
    )
    freq = logmag.shape[1]
    logmag = logmag.reshape(batch, count * 2, freq, -1)
    pos = logmag[:, :count]
    neg = logmag[:, count:]
    pos_mean = pos.mean(dim=(1, 3))
    neg_mean = neg.mean(dim=(1, 3))
    pos_std = pos.flatten(1, 3).reshape(batch, count, freq, -1).std(dim=(1, 3))
    neg_std = neg.flatten(1, 3).reshape(batch, count, freq, -1).std(dim=(1, 3))
    channels = torch.stack(
        [
            pos_mean,
            neg_mean,
            pos_mean - neg_mean,
            pos_std,
            neg_std,
            pos_std - neg_std,
        ],
        dim=1,
    )
    return channels.unsqueeze(-1).expand(-1, -1, -1, frames)


def _model_input(
    mixture: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    one_hot: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    include_one_hot: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_stft = _stft(mixture, n_fft=n_fft, hop_length=hop_length, window=window)
    mix_mag = torch.abs(mix_stft)
    logmag = torch.log1p(mix_mag).unsqueeze(1)
    context = _spectral_context(
        positive,
        negative,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        frames=logmag.shape[-1],
    )
    channels = [logmag, context]
    if include_one_hot:
        batch, _classes = one_hot.shape
        _, _channel, freq, frames = logmag.shape
        channels.append(one_hot.view(batch, -1, 1, 1).expand(-1, -1, freq, frames))
    return torch.cat(channels, dim=1), mix_stft, mix_mag


def _input_channels(include_one_hot: bool) -> int:
    return 1 + 6 + (len(CORE_SPEAKERS) if include_one_hot else 0)


def _train_model(
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    train_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    model_path: Path,
    device: str,
) -> SpectralEnrollmentUNet:
    model = SpectralEnrollmentUNet(
        input_channels=_input_channels(bool(args.include_one_hot)),
        base_channels=int(args.base_channels),
    ).to(device)
    if model_path.exists():
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    enrollments = _enrollment_source(str(args.enrollment_source), mixtures, targets)
    window = torch.hann_window(int(args.n_fft), device=device)
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        batch_indices = _balanced_indices(
            labels,
            train_indices,
            batch_size=int(args.batch_size),
            rng=rng,
        )
        batch_labels = [labels[index] for index in batch_indices]
        positive_indices, negative_indices = _random_enrollment_indices(
            labels,
            train_indices,
            batch_labels,
            count=int(args.train_enrollment_count),
            rng=rng,
        )
        mixture = torch.from_numpy(mixtures[batch_indices]).to(device)
        target = torch.from_numpy(targets[batch_indices]).to(device)
        interferer = mixture - target
        positive = torch.from_numpy(_gather_enrollments(enrollments, positive_indices)).to(device)
        negative = torch.from_numpy(_gather_enrollments(enrollments, negative_indices)).to(device)
        speaker_one_hot = torch.from_numpy(_one_hot(batch_labels)).to(device)
        model_input, mix_stft, mix_mag = _model_input(
            mixture,
            positive,
            negative,
            speaker_one_hot,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
            include_one_hot=bool(args.include_one_hot),
        )
        target_stft = _stft(
            target,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
        )
        interferer_stft = _stft(
            interferer,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
        )
        target_mag = torch.abs(target_stft)
        interferer_mag = torch.abs(interferer_stft)
        true_mask = target_mag / (target_mag + interferer_mag).clamp_min(1e-5)
        predicted_mask = model(model_input).squeeze(1)
        estimated_mag = predicted_mask * mix_mag
        estimated_wave = torch.istft(
            mix_stft * predicted_mask,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            win_length=int(args.n_fft),
            window=window,
            length=mixture.shape[1],
        )
        mask_loss = torch.nn.functional.l1_loss(predicted_mask, true_mask)
        mag_loss = torch.nn.functional.l1_loss(torch.log1p(estimated_mag), torch.log1p(target_mag))
        wav_loss = torch.nn.functional.l1_loss(estimated_wave, target)
        loss = (
            mask_loss
            + float(args.mag_loss_weight) * mag_loss
            + float(args.wave_loss_weight) * wav_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"mask={mask_loss.item():.5f} mag={mag_loss.item():.5f} "
                f"wav={wav_loss.item():.5f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, model_path)
    return model


def _separate_batch(
    model: SpectralEnrollmentUNet,
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    train_indices: Sequence[int],
    batch_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    device: str,
) -> np.ndarray:
    enrollments = _enrollment_source(str(args.enrollment_source), mixtures, targets)
    batch_labels = [labels[index] for index in batch_indices]
    positive_indices, negative_indices = _fixed_enrollment_indices(
        labels,
        train_indices,
        batch_labels,
        count=int(args.eval_enrollment_count),
    )
    mixture = torch.from_numpy(mixtures[batch_indices]).to(device)
    positive = torch.from_numpy(_gather_enrollments(enrollments, positive_indices)).to(device)
    negative = torch.from_numpy(_gather_enrollments(enrollments, negative_indices)).to(device)
    speaker_one_hot = torch.from_numpy(_one_hot(batch_labels)).to(device)
    window = torch.hann_window(int(args.n_fft), device=device)
    with torch.inference_mode():
        model_input, mix_stft, _mix_mag = _model_input(
            mixture,
            positive,
            negative,
            speaker_one_hot,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
            include_one_hot=bool(args.include_one_hot),
        )
        mask = model(model_input).squeeze(1)
        separated = torch.istft(
            mix_stft * mask,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            win_length=int(args.n_fft),
            window=window,
            length=mixture.shape[1],
        )
    return separated.detach().cpu().numpy().astype(np.float32)


def _write_leave_group_embeddings(
    rows: Sequence[MaskRow],
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    *,
    args: argparse.Namespace,
    output_path: Path,
    sample_rate: int,
    device: str,
) -> List[Dict[str, object]]:
    titanet = _load_titanet(device)
    groups = sorted({_split_group(row, str(args.split_mode)) for row in rows})
    embeddings_by_index: Dict[int, np.ndarray] = {}
    fold_summaries: List[Dict[str, object]] = []
    base_model_path = args.model_output.expanduser()
    all_indices = list(range(len(rows)))
    for group in groups:
        test_indices = [
            index
            for index, row in enumerate(rows)
            if _split_group(row, str(args.split_mode)) == group
        ]
        test_set = set(test_indices)
        train_indices = [index for index in all_indices if index not in test_set]
        fold_model_path = base_model_path.with_name(
            f"{base_model_path.stem}_{_safe_group(group)}{base_model_path.suffix}"
        )
        print(
            f"spectral_enrollment_unet_fold group={group} train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        model = _train_model(
            mixtures,
            targets,
            labels,
            train_indices,
            args=args,
            model_path=fold_model_path,
            device=device,
        )
        fold_embeddings: List[np.ndarray] = []
        for offset in range(0, len(test_indices), int(args.eval_batch_size)):
            batch_indices = test_indices[offset : offset + int(args.eval_batch_size)]
            enhanced = _separate_batch(
                model,
                mixtures,
                targets,
                labels,
                train_indices,
                batch_indices,
                args=args,
                device=device,
            )
            fold_embeddings.append(
                _embed_waveforms(
                    titanet,
                    list(enhanced),
                    sample_rate=sample_rate,
                    batch_size=int(args.eval_batch_size),
                    device=device,
                )
            )
            print(
                f"embedded_spectral_enrollment_unet group={group} "
                f"{min(offset + int(args.eval_batch_size), len(test_indices))}/{len(test_indices)}",
                flush=True,
            )
        fold_matrix = np.vstack(fold_embeddings).astype(np.float32)
        for index, embedding in zip(test_indices, fold_matrix):
            embeddings_by_index[index] = embedding
        fold_summaries.append(
            {
                "group": group,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_speakers": dict(Counter(labels[index] for index in train_indices)),
                "test_speakers": dict(Counter(labels[index] for index in test_indices)),
                "model_path": str(fold_model_path),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.vstack([embeddings_by_index[index] for index in range(len(rows))]).astype(
            np.float32
        ),
        windows=np.asarray([row.window for row in rows]),
        indices=np.asarray([row.index for row in rows], dtype=np.int32),
        truths=np.asarray([row.truth for row in rows]),
        target_shares=np.asarray([row.target_share for row in rows], dtype=np.float32),
        active_5pct=np.asarray([row.active_5pct for row in rows], dtype=np.int16),
    )
    return fold_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an eval-stem STFT U-Net with positive/negative spectral enrollment."
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
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument(
        "--pair-cache", type=Path, default=Path("/tmp/codex_eval_stem_pairs_s300.npz")
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_spectral_enrollment_unet.pt"),
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_spectral_enrollment_unet_embeddings.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_spectral_enrollment_unet.json"),
    )
    parser.add_argument(
        "--split-mode", choices=("leave_group", "leave_session"), default="leave_group"
    )
    parser.add_argument("--evaluation-split", choices=("window", "session"), default="window")
    parser.add_argument("--enrollment-source", choices=("target", "mixed"), default="mixed")
    parser.add_argument("--include-one-hot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mag-loss-weight", type=float, default=0.40)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--train-enrollment-count", type=int, default=2)
    parser.add_argument("--eval-enrollment-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    mixtures, targets, labels, pair_rows = _load_eval_stem_pairs(
        rows,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        output_path=args.pair_cache.expanduser(),
        window_seconds=float(args.window_seconds),
        sample_rate=int(args.sample_rate),
    )
    print(
        f"loaded_eval_stem_pairs rows={len(pair_rows)} speakers={dict(Counter(labels))}",
        flush=True,
    )
    fold_summaries: List[Dict[str, object]] = []
    if not args.embedding_output.expanduser().exists():
        fold_summaries = _write_leave_group_embeddings(
            pair_rows,
            mixtures,
            targets,
            labels,
            args=args,
            output_path=args.embedding_output.expanduser(),
            sample_rate=int(args.sample_rate),
            device=device,
        )
    embeddings, embedding_rows = _load_embeddings(args.embedding_output.expanduser())
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in embedding_rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    result = _evaluate_embeddings_with_split(
        embeddings,
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
        split_mode=str(args.evaluation_split),
    )
    result.pop("predictions", None)
    mixed_result = _evaluate_embeddings_with_split(
        _mixed_same_rows(
            rows_by_window,
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
        ),
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
        split_mode=str(args.evaluation_split),
    )
    mixed_result.pop("predictions", None)
    payload = {
        "model": "eval_stem_spectral_enrollment_unet",
        "split_mode": str(args.split_mode),
        "evaluation_split": str(args.evaluation_split),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "enrollment_source": str(args.enrollment_source),
        "include_one_hot": bool(args.include_one_hot),
        "train_steps": int(args.train_steps),
        "train_enrollment_count": int(args.train_enrollment_count),
        "eval_enrollment_count": int(args.eval_enrollment_count),
        "folds": fold_summaries,
        "target_extractor": result,
        "mixed_same_rows": mixed_result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    print(
        ",".join(
            [
                "eval_stem_spectral_enrollment_unet/lda_shrinkage",
                str(result["direct"]["examples"]),
                f"{float(result['direct']['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )
    print(
        ",".join(
            [
                "mixed_same_rows/lda_shrinkage",
                str(mixed_result["direct"]["examples"]),
                f"{float(mixed_result['direct']['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
