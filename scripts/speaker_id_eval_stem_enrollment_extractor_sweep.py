from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    FilmTcnBlock,
    _match_length,
    _normalize_batch,
    _si_snr,
)
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


class EnrollmentEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.GroupNorm(4, 32),
            nn.PReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.GroupNorm(8, 64),
            nn.PReLU(),
            nn.Conv1d(64, 96, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(8, 96),
            nn.PReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out = nn.Sequential(
            nn.Linear(96, output_dim),
            nn.SiLU(),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        hidden = self.net(wave.unsqueeze(1)).squeeze(-1)
        return self.out(hidden)


class PositiveNegativeTasNetExtractor(nn.Module):
    def __init__(
        self,
        *,
        enc_feats: int = 128,
        bottleneck: int = 128,
        cond_dim: int = 64,
        enrollment_dim: int = 64,
        enc_kernel: int = 16,
        layers: int = 6,
        stacks: int = 2,
        include_one_hot: bool = True,
    ) -> None:
        super().__init__()
        self.enc_kernel = int(enc_kernel)
        self.enc_stride = int(enc_kernel) // 2
        self.include_one_hot = bool(include_one_hot)
        self.encoder = nn.Conv1d(1, enc_feats, kernel_size=enc_kernel, stride=self.enc_stride)
        self.enrollment_encoder = EnrollmentEncoder(output_dim=enrollment_dim)
        one_hot_dim = len(CORE_SPEAKERS) if include_one_hot else 0
        self.cond = nn.Sequential(
            nn.Linear(enrollment_dim * 3 + one_hot_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cond_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(enc_feats, bottleneck, kernel_size=1),
            nn.GroupNorm(8, bottleneck),
            nn.PReLU(),
        )
        self.blocks = nn.ModuleList(
            FilmTcnBlock(bottleneck, cond_dim, dilation=2**layer)
            for _stack in range(stacks)
            for layer in range(layers)
        )
        self.mask = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck, enc_feats, kernel_size=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.ConvTranspose1d(
            enc_feats,
            1,
            kernel_size=enc_kernel,
            stride=self.enc_stride,
        )

    def forward(
        self,
        mixture: torch.Tensor,
        positive_enrollment: torch.Tensor,
        negative_enrollment: torch.Tensor,
        one_hot: torch.Tensor,
    ) -> torch.Tensor:
        length = mixture.shape[-1]
        encoded = torch.relu(self.encoder(mixture.unsqueeze(1)))
        pos = self.enrollment_encoder(positive_enrollment)
        neg = self.enrollment_encoder(negative_enrollment)
        features = [pos, neg, pos - neg]
        if self.include_one_hot:
            features.append(one_hot)
        condition = self.cond(torch.cat(features, dim=1))
        hidden = self.input_proj(encoded)
        for block in self.blocks:
            hidden = block(hidden, condition)
        decoded = self.decoder(encoded * self.mask(hidden)).squeeze(1)
        return _match_length(decoded, length)


def _one_hot(labels: Sequence[str]) -> np.ndarray:
    index_by_speaker = {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}
    values = np.zeros((len(labels), len(CORE_SPEAKERS)), dtype=np.float32)
    for row, label in enumerate(labels):
        values[row, index_by_speaker[str(label)]] = 1.0
    return values


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


def _enrollment_indices(
    labels: Sequence[str],
    train_indices: Sequence[int],
    target_labels: Sequence[str],
    *,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    by_speaker = _speaker_indices(labels, train_indices)
    positive: List[int] = []
    negative: List[int] = []
    for label in target_labels:
        pos_pool = by_speaker[str(label)]
        neg_speakers = [
            speaker for speaker in CORE_SPEAKERS if speaker != label and by_speaker.get(speaker)
        ]
        positive.append(rng.choice(pos_pool))
        negative.append(rng.choice(by_speaker[rng.choice(neg_speakers)]))
    return positive, negative


def _fixed_enrollment_indices(
    labels: Sequence[str],
    train_indices: Sequence[int],
    target_labels: Sequence[str],
) -> Tuple[List[int], List[int]]:
    by_speaker = _speaker_indices(labels, train_indices)
    positive: List[int] = []
    negative: List[int] = []
    for label in target_labels:
        pos_pool = by_speaker[str(label)]
        neg_speakers = [
            speaker for speaker in CORE_SPEAKERS if speaker != label and by_speaker.get(speaker)
        ]
        positive.append(pos_pool[0])
        negative.append(by_speaker[neg_speakers[0]][0])
    return positive, negative


def _enrollment_source(
    source: str,
    mixtures: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if source == "target":
        return targets
    if source == "mixed":
        return mixtures
    raise ValueError(f"Unknown enrollment source: {source}")


def _train_model(
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    train_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    device: str,
    model_path: Path,
) -> PositiveNegativeTasNetExtractor:
    model = PositiveNegativeTasNetExtractor(
        enc_feats=int(args.enc_feats),
        bottleneck=int(args.bottleneck),
        cond_dim=int(args.cond_dim),
        enrollment_dim=int(args.enrollment_dim),
        enc_kernel=int(args.enc_kernel),
        layers=int(args.layers),
        stacks=int(args.stacks),
        include_one_hot=bool(args.include_one_hot),
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
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        batch_indices = _balanced_indices(
            labels,
            train_indices,
            batch_size=int(args.batch_size),
            rng=rng,
        )
        batch_labels = [labels[index] for index in batch_indices]
        positive_indices, negative_indices = _enrollment_indices(
            labels,
            train_indices,
            batch_labels,
            rng=rng,
        )
        mixture = torch.from_numpy(mixtures[batch_indices]).to(device)
        target = torch.from_numpy(targets[batch_indices]).to(device)
        positive = torch.from_numpy(enrollments[positive_indices]).to(device)
        negative = torch.from_numpy(enrollments[negative_indices]).to(device)
        speaker_one_hot = torch.from_numpy(_one_hot(batch_labels)).to(device)
        mixture_norm, scale = _normalize_batch(mixture)
        target_norm = target / scale
        positive_norm, _positive_scale = _normalize_batch(positive)
        negative_norm, _negative_scale = _normalize_batch(negative)
        estimate = model(mixture_norm, positive_norm, negative_norm, speaker_one_hot)
        si_loss = -_si_snr(estimate, target_norm).mean()
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        loss = si_loss + float(args.wave_loss_weight) * wav_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, model_path)
    return model


def _extract_batch(
    model: PositiveNegativeTasNetExtractor,
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
    )
    mixture = torch.from_numpy(mixtures[batch_indices]).to(device)
    positive = torch.from_numpy(enrollments[positive_indices]).to(device)
    negative = torch.from_numpy(enrollments[negative_indices]).to(device)
    speaker_one_hot = torch.from_numpy(_one_hot(batch_labels)).to(device)
    with torch.inference_mode():
        mixture_norm, scale = _normalize_batch(mixture)
        positive_norm, _positive_scale = _normalize_batch(positive)
        negative_norm, _negative_scale = _normalize_batch(negative)
        estimate = model(mixture_norm, positive_norm, negative_norm, speaker_one_hot) * scale
    return estimate.detach().cpu().numpy().astype(np.float32)


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
            f"enrollment_extractor_fold group={group} train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        model = _train_model(
            mixtures,
            targets,
            labels,
            train_indices,
            args=args,
            device=device,
            model_path=fold_model_path,
        )
        fold_embeddings: List[np.ndarray] = []
        for offset in range(0, len(test_indices), int(args.eval_batch_size)):
            batch_indices = test_indices[offset : offset + int(args.eval_batch_size)]
            enhanced = _extract_batch(
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
                f"embedded_enrollment_lgo group={group} "
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
        description="Train a positive/negative enrollment-conditioned eval-stem target extractor."
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
        "--model-output", type=Path, default=Path("/tmp/codex_eval_stem_enrollment_tasnet.pt")
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_enrollment_tasnet_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_eval_stem_enrollment_tasnet.json")
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
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enrollment-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
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
        "model": "eval_stem_positive_negative_enrollment_tasnet",
        "split_mode": str(args.split_mode),
        "diagnostic_leakage": f"extractor trained {args.split_mode.replace('_', '-')}-out on selected eval-stem rows",
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "enrollment_source": str(args.enrollment_source),
        "include_one_hot": bool(args.include_one_hot),
        "evaluation_split": str(args.evaluation_split),
        "train_steps": int(args.train_steps),
        "folds": fold_summaries,
        "target_extractor": result,
        "mixed_same_rows": mixed_result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    print(
        ",".join(
            [
                "eval_stem_posneg_enrollment_tasnet/lda_shrinkage",
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
