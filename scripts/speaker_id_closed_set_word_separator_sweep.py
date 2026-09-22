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

from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _evaluate_candidate_embeddings,
    _load_candidate_embeddings,
)
from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    FilmTcnBlock,
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


class ClosedSetTasNetSeparator(nn.Module):
    def __init__(
        self,
        *,
        speakers: int,
        enc_feats: int = 192,
        bottleneck: int = 192,
        enc_kernel: int = 16,
        layers: int = 8,
        stacks: int = 3,
    ) -> None:
        super().__init__()
        self.speakers = int(speakers)
        self.enc_kernel = int(enc_kernel)
        self.enc_stride = int(enc_kernel) // 2
        self.encoder = nn.Conv1d(1, enc_feats, kernel_size=enc_kernel, stride=self.enc_stride)
        self.input_proj = nn.Sequential(
            nn.Conv1d(enc_feats, bottleneck, kernel_size=1),
            nn.GroupNorm(8, bottleneck),
            nn.PReLU(),
        )
        blocks: List[FilmTcnBlock] = []
        for _stack in range(stacks):
            for layer in range(layers):
                blocks.append(FilmTcnBlock(bottleneck, cond_dim=1, dilation=2**layer))
        self.blocks = nn.ModuleList(blocks)
        self.mask = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck, speakers * enc_feats, kernel_size=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.ConvTranspose1d(
            enc_feats,
            1,
            kernel_size=enc_kernel,
            stride=self.enc_stride,
        )
        self.owner_head = nn.Sequential(
            nn.Linear(bottleneck * 2, bottleneck),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, speakers),
        )

    def forward(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length = mixture.shape[-1]
        encoded = torch.relu(self.encoder(mixture.unsqueeze(1)))
        hidden = self.input_proj(encoded)
        dummy_condition = hidden.new_zeros((hidden.shape[0], 1))
        for block in self.blocks:
            hidden = block(hidden, dummy_condition)
        masks = self.mask(hidden).reshape(
            hidden.shape[0],
            self.speakers,
            encoded.shape[1],
            encoded.shape[2],
        )
        masked = (encoded.unsqueeze(1) * masks).reshape(
            hidden.shape[0] * self.speakers,
            encoded.shape[1],
            encoded.shape[2],
        )
        decoded = self.decoder(masked).squeeze(1)
        decoded = _match_length(decoded, length).reshape(hidden.shape[0], self.speakers, length)
        pooled = torch.cat(
            [hidden.mean(dim=-1), hidden.std(dim=-1, unbiased=False)],
            dim=1,
        )
        return decoded, self.owner_head(pooled)


def _speaker_to_id() -> Dict[str, int]:
    return {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}


def _build_targets(
    sources: np.ndarray,
    rows: Sequence[MaskRow],
    *,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    speaker_ids = _speaker_to_id()
    truth_indices = np.asarray([speaker_ids[row.truth] for row in rows], dtype=np.int64)
    if mode == "all_sources":
        return sources.astype(np.float32), truth_indices
    targets = np.zeros_like(sources, dtype=np.float32)
    for row_index, speaker_index in enumerate(truth_indices):
        targets[row_index, speaker_index] = sources[row_index, speaker_index]
    return targets.astype(np.float32), truth_indices


def _load_model_from_path(
    path: Path,
    *,
    device: str,
    args: argparse.Namespace,
) -> ClosedSetTasNetSeparator:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = dict(payload.get("args") or {})
    model = ClosedSetTasNetSeparator(
        speakers=len(CORE_SPEAKERS),
        enc_feats=int(saved_args.get("enc_feats", args.enc_feats)),
        bottleneck=int(saved_args.get("bottleneck", args.bottleneck)),
        enc_kernel=int(saved_args.get("enc_kernel", args.enc_kernel)),
        layers=int(saved_args.get("layers", args.layers)),
        stacks=int(saved_args.get("stacks", args.stacks)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _train_fold_model(
    mixtures: np.ndarray,
    targets: np.ndarray,
    truth_indices: np.ndarray,
    rows: Sequence[MaskRow],
    train_indices: Sequence[int],
    *,
    args: argparse.Namespace,
    model_path: Path,
    device: str,
) -> ClosedSetTasNetSeparator:
    if model_path.exists():
        return _load_model_from_path(model_path, device=device, args=args)

    model = ClosedSetTasNetSeparator(
        speakers=len(CORE_SPEAKERS),
        enc_feats=int(args.enc_feats),
        bottleneck=int(args.bottleneck),
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
    train_targets = targets[train_indices]
    train_truth = truth_indices[train_indices]
    rng = random.Random(int(args.seed) + len(train_indices))
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        row_indices = _sample_row_indices_by_speaker(
            train_rows,
            batch_size=int(args.batch_size),
            rng=rng,
        )
        mixture = torch.from_numpy(train_mixtures[row_indices]).to(device)
        target = torch.from_numpy(train_targets[row_indices]).to(device)
        truth = torch.from_numpy(train_truth[row_indices]).to(device)

        mixture_norm, scale = _normalize_batch(mixture)
        target_norm = target / scale.unsqueeze(1)
        estimate, owner_logits = model(mixture_norm)
        selected_estimate = estimate[torch.arange(len(row_indices), device=device), truth]
        selected_target = target_norm[torch.arange(len(row_indices), device=device), truth]
        si_loss = -_si_snr(selected_estimate, selected_target).mean()
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        if str(args.target_mode) == "word_owner":
            negative_mask = torch.ones_like(estimate, dtype=torch.bool)
            negative_mask[torch.arange(len(row_indices), device=device), truth] = False
            silence_loss = torch.mean(torch.abs(estimate[negative_mask]))
        else:
            silence_loss = torch.zeros((), device=device)
        owner_loss = torch.nn.functional.cross_entropy(owner_logits, truth)
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
            owner_acc = (owner_logits.argmax(dim=1) == truth).float().mean()
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f} "
                f"silence={silence_loss.item():.5f} owner={owner_loss.item():.5f} "
                f"owner_acc={owner_acc.item():.4f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "speakers": CORE_SPEAKERS,
            "objective": f"closed_set_{args.target_mode}_separator",
        },
        model_path,
    )
    return model


def _extract_batch(
    model: ClosedSetTasNetSeparator,
    mixtures: np.ndarray,
    *,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mixture = torch.from_numpy(np.asarray(mixtures, dtype=np.float32)).to(device)
    with torch.inference_mode():
        mixture_norm, scale = _normalize_batch(mixture)
        estimate, owner_logits = model(mixture_norm)
        estimate = estimate * scale.unsqueeze(1)
    return (
        estimate.detach().cpu().numpy().astype(np.float32),
        owner_logits.detach().cpu().numpy().astype(np.float32),
    )


def _write_candidate_outputs(
    mixtures: np.ndarray,
    rows: Sequence[MaskRow],
    *,
    args: argparse.Namespace,
    output_path: Path,
    sample_rate: int,
    device: str,
) -> List[Dict[str, object]]:
    candidates = tuple(CORE_SPEAKERS)
    titanet = _load_titanet(device)
    groups = (
        ["all"]
        if str(args.split_mode) == "leaky"
        else sorted({_split_group(row, str(args.split_mode)) for row in rows})
    )
    model_cache: Dict[str, ClosedSetTasNetSeparator] = {}
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
            model_cache[group] = _load_model_from_path(model_path, device=device, args=args)

        for offset in range(0, len(test_indices), int(args.row_batch_size)):
            batch_indices = test_indices[offset : offset + int(args.row_batch_size)]
            extracted, owner_logits = _extract_batch(
                model_cache[group],
                mixtures[batch_indices],
                device=device,
            )
            embedded = _embed_waveforms(
                titanet,
                list(extracted.reshape(-1, extracted.shape[-1])),
                sample_rate=sample_rate,
                batch_size=int(args.embed_batch_size),
                device=device,
            )
            embedded_by_row = embedded.reshape(len(batch_indices), len(candidates), -1)
            row_rms = np.sqrt(np.mean(np.square(extracted), axis=-1)).astype(np.float32)
            row_peak = np.max(np.abs(extracted), axis=-1).astype(np.float32)
            for index, embedding, rms, peak, logits in zip(
                batch_indices,
                embedded_by_row,
                row_rms,
                row_peak,
                owner_logits,
            ):
                embeddings_by_index[index] = embedding.astype(np.float32)
                rms_by_index[index] = rms.astype(np.float32)
                peak_by_index[index] = peak.astype(np.float32)
                owner_logits_by_index[index] = logits.astype(np.float32)
            print(
                f"embedded_closed_set group={group} "
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
        description="Train a fixed-channel closed-set separator/word-owner model."
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
        default=Path("/tmp/codex_closed_set_word_separator_s300.pt"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("/tmp/codex_closed_set_word_separator_candidates_s300.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_closed_set_word_separator_s300.json"),
    )
    parser.add_argument(
        "--target-mode", choices=("word_owner", "all_sources"), default="word_owner"
    )
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=24)
    parser.add_argument("--si-loss-weight", type=float, default=1.0)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument("--silence-loss-weight", type=float, default=0.25)
    parser.add_argument("--owner-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--enc-feats", type=int, default=192)
    parser.add_argument("--bottleneck", type=int, default=192)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--stacks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    mixtures, candidate_sources, _candidate_source_rms, cache_rows = _load_word_owner_cache(
        rows,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        output_path=args.pair_cache.expanduser(),
        window_seconds=float(args.window_seconds),
        sample_rate=int(args.sample_rate),
    )
    targets, truth_indices = _build_targets(
        candidate_sources, cache_rows, mode=str(args.target_mode)
    )
    print(
        f"loaded_closed_set_cache rows={len(cache_rows)} speakers="
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
            f"closed_set_fold group={group} train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        _train_fold_model(
            mixtures,
            targets,
            truth_indices,
            cache_rows,
            train_indices,
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
    owner_prob = torch.softmax(torch.from_numpy(owner_logits), dim=1).numpy()
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
        "model": "closed_set_tasnet_separator",
        "objective": f"fixed_speaker_channels_{args.target_mode}",
        "split_mode": str(args.split_mode),
        "target_mode": str(args.target_mode),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
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
