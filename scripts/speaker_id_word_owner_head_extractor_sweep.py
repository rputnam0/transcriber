from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
    _evaluate_candidate_embeddings,
    _load_candidate_embeddings,
)
from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    FilmTcnBlock,
    _conditioning_vectors,
    _match_length,
    _normalize_batch,
    _si_snr,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _score_direct,
    _score_slices,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_owner_extractor_sweep import (  # noqa: E402
    _energy_predictions,
    _energy_router_predictions,
    _load_word_owner_cache,
    _mixed_predictions_from_diagnostics,
    _model_path_for_group,
    _sample_pair_indices,
    _split_group,
)


def _sample_row_indices_by_speaker(
    rows: Sequence[MaskRow],
    *,
    batch_size: int,
    rng: random.Random,
) -> List[int]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_speaker[row.truth].append(index)
    speakers = [speaker for speaker in CORE_SPEAKERS if by_speaker.get(speaker)]
    return [rng.choice(by_speaker[rng.choice(speakers)]) for _ in range(batch_size)]


class OwnerHeadTasNetExtractor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        enc_feats: int = 128,
        bottleneck: int = 128,
        cond_dim: int = 64,
        enc_kernel: int = 16,
        layers: int = 6,
        stacks: int = 2,
    ) -> None:
        super().__init__()
        self.enc_kernel = int(enc_kernel)
        self.enc_stride = int(enc_kernel) // 2
        self.encoder = nn.Conv1d(1, enc_feats, kernel_size=enc_kernel, stride=self.enc_stride)
        self.cond = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cond_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(enc_feats, bottleneck, kernel_size=1),
            nn.GroupNorm(8, bottleneck),
            nn.PReLU(),
        )
        blocks: List[FilmTcnBlock] = []
        for _stack in range(stacks):
            for layer in range(layers):
                blocks.append(FilmTcnBlock(bottleneck, cond_dim, dilation=2**layer))
        self.blocks = nn.ModuleList(blocks)
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
        owner_features = bottleneck * 2 + enc_feats * 2 + cond_dim
        self.owner_head = nn.Sequential(
            nn.Linear(owner_features, bottleneck),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, 1),
        )

    def forward(
        self, mixture: torch.Tensor, enrollment: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        length = mixture.shape[-1]
        encoded = torch.relu(self.encoder(mixture.unsqueeze(1)))
        condition = self.cond(enrollment)
        hidden = self.input_proj(encoded)
        for block in self.blocks:
            hidden = block(hidden, condition)
        mask = self.mask(hidden)
        decoded = self.decoder(encoded * mask).squeeze(1)

        hidden_mean = hidden.mean(dim=-1)
        hidden_std = hidden.std(dim=-1, unbiased=False)
        mask_mean = mask.mean(dim=-1)
        mask_std = mask.std(dim=-1, unbiased=False)
        owner_features = torch.cat(
            [hidden_mean, hidden_std, mask_mean, mask_std, condition],
            dim=1,
        )
        owner_logit = self.owner_head(owner_features).squeeze(1)
        return _match_length(decoded, length), owner_logit


def _load_model_from_path(
    path: Path,
    *,
    embedding_dim: int,
    device: str,
    args: argparse.Namespace,
) -> OwnerHeadTasNetExtractor:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = dict(payload.get("args") or {})
    model = OwnerHeadTasNetExtractor(
        embedding_dim=embedding_dim,
        enc_feats=int(saved_args.get("enc_feats", args.enc_feats)),
        bottleneck=int(saved_args.get("bottleneck", args.bottleneck)),
        cond_dim=int(saved_args.get("cond_dim", args.cond_dim)),
        enc_kernel=int(saved_args.get("enc_kernel", args.enc_kernel)),
        layers=int(saved_args.get("layers", args.layers)),
        stacks=int(saved_args.get("stacks", args.stacks)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _train_fold_model(
    mixtures: np.ndarray,
    candidate_sources: np.ndarray,
    candidate_source_rms: np.ndarray,
    rows: Sequence[MaskRow],
    train_indices: Sequence[int],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    model_path: Path,
    device: str,
) -> OwnerHeadTasNetExtractor:
    embedding_dim = next(iter(centroids.values())).shape[0]
    if model_path.exists():
        return _load_model_from_path(
            model_path,
            embedding_dim=embedding_dim,
            device=device,
            args=args,
        )

    model = OwnerHeadTasNetExtractor(
        embedding_dim=embedding_dim,
        enc_feats=int(args.enc_feats),
        bottleneck=int(args.bottleneck),
        cond_dim=int(args.cond_dim),
        enc_kernel=int(args.enc_kernel),
        layers=int(args.layers),
        stacks=int(args.stacks),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_rows = [rows[index] for index in train_indices]
    train_mixtures = mixtures[train_indices]
    train_sources = candidate_sources[train_indices]
    train_rms = candidate_source_rms[train_indices]
    rng = random.Random(int(args.seed) + len(train_indices))
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        if str(args.owner_loss_mode) == "row_softmax":
            row_indices = _sample_row_indices_by_speaker(
                train_rows,
                batch_size=int(args.batch_size),
                rng=rng,
            )
            speaker_to_id = {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}
            truth_indices = np.asarray(
                [speaker_to_id[train_rows[index].truth] for index in row_indices],
                dtype=np.int64,
            )
            candidate_indices = np.tile(np.arange(len(CORE_SPEAKERS)), len(row_indices))
            flat_row_indices = np.repeat(
                np.asarray(row_indices, dtype=np.int64), len(CORE_SPEAKERS)
            )
            is_positive = candidate_indices == np.repeat(truth_indices, len(CORE_SPEAKERS))
            batch_mixture = torch.from_numpy(train_mixtures[flat_row_indices]).to(device)
            batch_target = np.zeros_like(train_mixtures[flat_row_indices], dtype=np.float32)
            if np.any(is_positive):
                batch_target[is_positive] = train_sources[
                    flat_row_indices[is_positive],
                    candidate_indices[is_positive],
                ]
            labels = [CORE_SPEAKERS[index] for index in candidate_indices]
            target = torch.from_numpy(batch_target).to(device)
            enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(
                device
            )
            positive_mask = torch.from_numpy(is_positive).to(device)
            truth_tensor = torch.from_numpy(truth_indices).to(device)
        else:
            row_indices, candidate_indices, is_positive = _sample_pair_indices(
                train_rows,
                train_rms,
                batch_size=int(args.batch_size),
                positive_fraction=float(args.positive_fraction),
                hard_negative_probability=float(args.hard_negative_probability),
                active_rms_threshold=float(args.active_rms_threshold),
                rng=rng,
            )
            batch_mixture = torch.from_numpy(train_mixtures[row_indices]).to(device)
            batch_target = np.zeros_like(train_mixtures[row_indices], dtype=np.float32)
            if np.any(is_positive):
                positive_rows = np.asarray(row_indices, dtype=np.int64)[is_positive]
                positive_candidates = np.asarray(candidate_indices, dtype=np.int64)[is_positive]
                batch_target[is_positive] = train_sources[positive_rows, positive_candidates]
            target = torch.from_numpy(batch_target).to(device)
            labels = [CORE_SPEAKERS[index] for index in candidate_indices]
            enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(
                device
            )
            owner_target = torch.from_numpy(is_positive.astype(np.float32)).to(device)
            positive_mask = torch.from_numpy(is_positive).to(device)

        mixture_norm, scale = _normalize_batch(batch_mixture)
        target_norm = target / scale
        estimate, owner_logit = model(mixture_norm, enrollment)
        if bool(positive_mask.any()):
            si_loss = -_si_snr(estimate[positive_mask], target_norm[positive_mask]).mean()
        else:
            si_loss = torch.zeros((), device=device)
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        if bool((~positive_mask).any()):
            silence_loss = torch.mean(torch.abs(estimate[~positive_mask]))
        else:
            silence_loss = torch.zeros((), device=device)
        if str(args.owner_loss_mode) == "row_softmax":
            owner_logits_by_row = owner_logit.reshape(len(row_indices), len(CORE_SPEAKERS))
            owner_loss = torch.nn.functional.cross_entropy(owner_logits_by_row, truth_tensor)
            owner_acc = (owner_logits_by_row.argmax(dim=1) == truth_tensor).float().mean()
        else:
            owner_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                owner_logit,
                owner_target,
            )
            owner_acc = ((owner_logit.detach() >= 0.0) == (owner_target >= 0.5)).float().mean()
        loss = (
            float(args.si_loss_weight) * si_loss
            + float(args.wave_loss_weight) * wav_loss
            + float(args.silence_loss_weight) * silence_loss
            + float(args.owner_loss_weight) * owner_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f} "
                f"silence={silence_loss.item():.5f} owner={owner_loss.item():.5f} "
                f"owner_acc={owner_acc.item():.4f} pos={int(is_positive.sum())}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "speakers": CORE_SPEAKERS,
            "objective": "word_owner_positive_target_else_silence_with_owner_head",
        },
        model_path,
    )
    return model


def _extract_batch(
    model: OwnerHeadTasNetExtractor,
    mixtures: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mixture = torch.from_numpy(np.asarray(mixtures, dtype=np.float32)).to(device)
    enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(device)
    with torch.inference_mode():
        mixture_norm, scale = _normalize_batch(mixture)
        estimate, owner_logit = model(mixture_norm, enrollment)
        estimate = estimate * scale
    return (
        estimate.detach().cpu().numpy().astype(np.float32),
        owner_logit.detach().cpu().numpy().astype(np.float32),
    )


def _write_candidate_outputs(
    mixtures: np.ndarray,
    rows: Sequence[MaskRow],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    output_path: Path,
    sample_rate: int,
    device: str,
) -> List[Dict[str, object]]:
    candidates = tuple(CORE_SPEAKERS)
    embedding_dim = next(iter(centroids.values())).shape[0]
    titanet = _load_titanet(device)
    groups = (
        ["all"]
        if str(args.split_mode) == "leaky"
        else sorted({_split_group(row, str(args.split_mode)) for row in rows})
    )
    model_cache: Dict[str, OwnerHeadTasNetExtractor] = {}
    embeddings_by_index: Dict[int, np.ndarray] = {}
    rms_by_index: Dict[int, np.ndarray] = {}
    peak_by_index: Dict[int, np.ndarray] = {}
    owner_logits_by_index: Dict[int, np.ndarray] = {}
    fold_summaries: List[Dict[str, object]] = []
    base_model_path = args.model_output.expanduser()

    for group in groups:
        if str(args.split_mode) == "leaky":
            test_indices = list(range(len(rows)))
        else:
            test_indices = [
                index
                for index, row in enumerate(rows)
                if _split_group(row, str(args.split_mode)) == group
            ]
        model_path = (
            base_model_path
            if str(args.split_mode) == "leaky"
            else _model_path_for_group(base_model_path, group)
        )
        if group not in model_cache:
            model_cache[group] = _load_model_from_path(
                model_path,
                embedding_dim=embedding_dim,
                device=device,
                args=args,
            )

        for offset in range(0, len(test_indices), int(args.row_batch_size)):
            batch_indices = test_indices[offset : offset + int(args.row_batch_size)]
            batch_waves = mixtures[batch_indices]
            expanded_waves = np.repeat(batch_waves, len(candidates), axis=0)
            expanded_labels = [speaker for _row in batch_indices for speaker in candidates]
            extracted, owner_logits = _extract_batch(
                model_cache[group],
                expanded_waves,
                expanded_labels,
                centroids=centroids,
                device=device,
            )
            embedded = _embed_waveforms(
                titanet,
                list(extracted),
                sample_rate=sample_rate,
                batch_size=int(args.embed_batch_size),
                device=device,
            )
            extracted_by_row = extracted.reshape(len(batch_indices), len(candidates), -1)
            embedded_by_row = embedded.reshape(len(batch_indices), len(candidates), -1)
            owner_by_row = owner_logits.reshape(len(batch_indices), len(candidates))
            row_rms = np.sqrt(np.mean(np.square(extracted_by_row), axis=-1)).astype(np.float32)
            row_peak = np.max(np.abs(extracted_by_row), axis=-1).astype(np.float32)
            for index, embedding, rms, peak, logits in zip(
                batch_indices,
                embedded_by_row,
                row_rms,
                row_peak,
                owner_by_row,
            ):
                embeddings_by_index[index] = embedding.astype(np.float32)
                rms_by_index[index] = rms.astype(np.float32)
                peak_by_index[index] = peak.astype(np.float32)
                owner_logits_by_index[index] = logits.astype(np.float32)
            print(
                f"embedded_word_owner_head group={group} "
                f"{min(offset + int(args.row_batch_size), len(test_indices))}/{len(test_indices)}",
                flush=True,
            )
        fold_summaries.append({"group": group, "test_rows": len(test_indices)})

    missing = [index for index in range(len(rows)) if index not in embeddings_by_index]
    if missing:
        raise RuntimeError(f"Missing candidate outputs for {missing[:10]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.stack([embeddings_by_index[index] for index in range(len(rows))]).astype(
            np.float32
        ),
        candidates=np.asarray(candidates),
        output_rms=np.stack([rms_by_index[index] for index in range(len(rows))]).astype(np.float32),
        output_peak=np.stack([peak_by_index[index] for index in range(len(rows))]).astype(
            np.float32
        ),
        owner_logits=np.stack([owner_logits_by_index[index] for index in range(len(rows))]).astype(
            np.float32
        ),
        windows=np.asarray([row.window for row in rows]),
        indices=np.asarray([row.index for row in rows], dtype=np.int32),
        truths=np.asarray([row.truth for row in rows]),
        target_shares=np.asarray([row.target_share for row in rows], dtype=np.float32),
        active_5pct=np.asarray([row.active_5pct for row in rows], dtype=np.int16),
    )
    return fold_summaries


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


def _load_output_metadata(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return (
        np.asarray(payload["output_rms"], dtype=np.float32),
        np.asarray(payload["output_peak"], dtype=np.float32),
        np.asarray(payload["owner_logits"], dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a word-owner calibrated target extractor with an explicit owner/presence head."
        )
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
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--pair-cache",
        type=Path,
        default=Path("/tmp/codex_word_owner_sources_s300.npz"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/tmp/codex_word_owner_head_tasnet_s300.pt"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("/tmp/codex_word_owner_head_candidate_embeddings_s300.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_word_owner_head_tasnet_s300.json"),
    )
    parser.add_argument("--conditioning", choices=("one_hot",), default="one_hot")
    parser.add_argument(
        "--split-mode",
        choices=("leaky", "leave_group", "leave_session"),
        default="leave_group",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=24)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--hard-negative-probability", type=float, default=0.8)
    parser.add_argument("--active-rms-threshold", type=float, default=0.002)
    parser.add_argument("--si-loss-weight", type=float, default=1.0)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument("--silence-loss-weight", type=float, default=0.25)
    parser.add_argument("--owner-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--owner-loss-mode",
        choices=("pair_bce", "row_softmax"),
        default="pair_bce",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--enc-feats", type=int, default=192)
    parser.add_argument("--bottleneck", type=int, default=192)
    parser.add_argument("--cond-dim", type=int, default=96)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--stacks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    centroids = _conditioning_vectors(
        clean_bank_path=args.clean_bank.expanduser(),
        mode=str(args.conditioning),
    )
    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    mixtures, candidate_sources, candidate_source_rms, cache_rows = _load_word_owner_cache(
        rows,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        output_path=args.pair_cache.expanduser(),
        window_seconds=float(args.window_seconds),
        sample_rate=int(args.sample_rate),
    )
    print(
        f"loaded_word_owner_cache rows={len(cache_rows)} speakers="
        f"{dict(Counter(row.truth for row in cache_rows))}",
        flush=True,
    )

    groups = (
        ["all"]
        if str(args.split_mode) == "leaky"
        else sorted({_split_group(row, str(args.split_mode)) for row in cache_rows})
    )
    fold_summaries: List[Dict[str, object]] = []
    for group in groups:
        if str(args.split_mode) == "leaky":
            train_indices = list(range(len(cache_rows)))
            test_indices = train_indices
            model_path = args.model_output.expanduser()
        else:
            test_indices = [
                index
                for index, row in enumerate(cache_rows)
                if _split_group(row, str(args.split_mode)) == group
            ]
            test_set = set(test_indices)
            train_indices = [index for index in range(len(cache_rows)) if index not in test_set]
            model_path = _model_path_for_group(args.model_output.expanduser(), group)
        print(
            f"word_owner_head_fold group={group} train={len(train_indices)} "
            f"test={len(test_indices)}",
            flush=True,
        )
        _train_fold_model(
            mixtures,
            candidate_sources,
            candidate_source_rms,
            cache_rows,
            train_indices,
            centroids=centroids,
            args=args,
            model_path=model_path,
            device=device,
        )
        fold_summaries.append(
            {
                "group": group,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_speakers": dict(Counter(cache_rows[index].truth for index in train_indices)),
                "test_speakers": dict(Counter(cache_rows[index].truth for index in test_indices)),
                "model_path": str(model_path),
            }
        )

    if not args.candidate_output.expanduser().exists():
        output_folds = _write_candidate_outputs(
            mixtures,
            cache_rows,
            centroids=centroids,
            args=args,
            output_path=args.candidate_output.expanduser(),
            sample_rate=int(args.sample_rate),
            device=device,
        )
        for summary, output_summary in zip(fold_summaries, output_folds):
            summary.update(output_summary)

    candidate_embeddings, candidates, embedding_rows = _load_candidate_embeddings(
        args.candidate_output.expanduser()
    )
    output_rms, output_peak, owner_logits = _load_output_metadata(
        args.candidate_output.expanduser()
    )
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
    mixed_embeddings = _mixed_same_rows(
        rows_by_window,
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    scored = _evaluate_candidate_embeddings(
        candidate_embeddings,
        candidates,
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        mixed_embeddings=mixed_embeddings,
    )
    diagnostics = scored.pop("diagnostic_rows")
    mixed_pred = _mixed_predictions_from_diagnostics(embedding_rows, diagnostics)
    rms_pred, rms_conf, rms_margin = _energy_predictions(output_rms, candidates)
    peak_pred, peak_conf, peak_margin = _energy_predictions(output_peak, candidates)
    owner_pred, owner_conf, owner_margin = _energy_predictions(owner_logits, candidates)
    owner_prob = 1.0 / (1.0 + np.exp(-np.clip(owner_logits, -30.0, 30.0)))
    owner_prob_pred, owner_prob_conf, owner_prob_margin = _energy_predictions(
        owner_prob, candidates
    )

    scored["owner_logit_argmax"] = _score_predictions(
        "owner_logit_argmax",
        embedding_rows,
        owner_pred,
    )
    scored["owner_logit_router"] = _score_predictions(
        "owner_logit_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, owner_pred, owner_conf, mixed_pred),
    )
    scored["owner_logit_margin_router"] = _score_predictions(
        "owner_logit_margin_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, owner_pred, owner_margin, mixed_pred),
    )
    scored["owner_probability_argmax"] = _score_predictions(
        "owner_probability_argmax",
        embedding_rows,
        owner_prob_pred,
    )
    scored["owner_probability_router"] = _score_predictions(
        "owner_probability_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, owner_prob_pred, owner_prob_conf, mixed_pred),
    )
    scored["owner_probability_margin_router"] = _score_predictions(
        "owner_probability_margin_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, owner_prob_pred, owner_prob_margin, mixed_pred),
    )
    scored["output_rms_argmax"] = _score_predictions(
        "output_rms_argmax",
        embedding_rows,
        rms_pred,
    )
    scored["output_peak_argmax"] = _score_predictions(
        "output_peak_argmax",
        embedding_rows,
        peak_pred,
    )
    scored["output_rms_router"] = _score_predictions(
        "output_rms_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, rms_pred, rms_conf, mixed_pred),
    )
    scored["output_peak_router"] = _score_predictions(
        "output_peak_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, peak_pred, peak_conf, mixed_pred),
    )
    scored["output_rms_margin_router"] = _score_predictions(
        "output_rms_margin_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, rms_pred, rms_margin, mixed_pred),
    )
    scored["output_peak_margin_router"] = _score_predictions(
        "output_peak_margin_router",
        embedding_rows,
        _energy_router_predictions(embedding_rows, peak_pred, peak_margin, mixed_pred),
    )

    payload = {
        "model": "word_owner_head_conditioned_tasnet",
        "objective": "truth_candidate_source_else_silence_plus_owner_head",
        "split_mode": str(args.split_mode),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "conditioning": str(args.conditioning),
        "train_steps": int(args.train_steps),
        "owner_loss_weight": float(args.owner_loss_weight),
        "folds": fold_summaries,
        "candidate_embedding_cache": str(args.candidate_output.expanduser()),
        "scores": scored,
        "diagnostic_rows": diagnostics[:200],
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    for name, score in sorted(scored.items()):
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
