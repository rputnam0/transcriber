from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_sweep import _conditioning_vectors  # noqa: E402
from speaker_id_eval_stem_target_extractor_sweep import (  # noqa: E402
    _evaluate_embeddings_with_split,
    _load_embeddings,
    _load_eval_stem_pairs,
    _safe_group,
    _session_group,
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
from speaker_id_unet_mask_extractor_sweep import (  # noqa: E402
    OneHotMaskUNet,
    _separate_batch,
    _stft,
)
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _split_group(row: MaskRow, mode: str) -> str:
    if mode == "leave_group":
        return _window_group(row.window)
    if mode == "leave_session":
        return _session_group(row.window)
    if mode == "leaky":
        return "all"
    raise ValueError(mode)


def _model_path_for_group(base_path: Path, group: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{_safe_group(group)}{base_path.suffix}")


def _sample_indices(labels: Sequence[str], *, batch_size: int, rng) -> List[int]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_speaker[str(label)].append(index)
    speakers = [speaker for speaker in CORE_SPEAKERS if by_speaker.get(speaker)]
    return [rng.choice(by_speaker[rng.choice(speakers)]) for _ in range(batch_size)]


def _train_model(
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    model_path: Path,
    device: str,
) -> OneHotMaskUNet:
    embedding_dim = next(iter(centroids.values())).shape[0]
    model = OneHotMaskUNet(
        embedding_dim=embedding_dim,
        cond_channels=int(args.cond_channels),
        base_channels=int(args.base_channels),
    ).to(device)
    if model_path.exists():
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    import random

    rng = random.Random(int(args.seed) + mixtures.shape[0])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    window = torch.hann_window(int(args.n_fft), device=device)
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        indices = _sample_indices(labels, batch_size=int(args.batch_size), rng=rng)
        mixture = torch.from_numpy(mixtures[indices]).to(device)
        target = torch.from_numpy(targets[indices]).to(device)
        interferer = mixture - target
        enrollment = torch.from_numpy(np.stack([centroids[labels[index]] for index in indices])).to(
            device
        )
        mix_stft = _stft(
            mixture,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
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
        mix_mag = torch.abs(mix_stft)
        target_mag = torch.abs(target_stft)
        interferer_mag = torch.abs(interferer_stft)
        true_mask = target_mag / (target_mag + interferer_mag).clamp_min(1e-5)
        predicted_mask = model(torch.log1p(mix_mag).unsqueeze(1), enrollment).squeeze(1)
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
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "speakers": CORE_SPEAKERS,
            "objective": "eval_stem_stft_mask",
        },
        model_path,
    )
    return model


def _load_model(
    model_path: Path,
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> OneHotMaskUNet:
    embedding_dim = next(iter(centroids.values())).shape[0]
    model = OneHotMaskUNet(
        embedding_dim=embedding_dim,
        cond_channels=int(args.cond_channels),
        base_channels=int(args.base_channels),
    ).to(device)
    payload = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _write_embeddings_for_models(
    mixtures: np.ndarray,
    labels: Sequence[str],
    rows: Sequence[MaskRow],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    output_path: Path,
    sample_rate: int,
    device: str,
) -> List[Dict[str, object]]:
    titanet = _load_titanet(device)
    groups = (
        ["all"]
        if str(args.split_mode) == "leaky"
        else sorted({_split_group(row, str(args.split_mode)) for row in rows})
    )
    embeddings_by_index: Dict[int, np.ndarray] = {}
    fold_summaries: List[Dict[str, object]] = []
    base_model_path = args.model_output.expanduser()
    for group in groups:
        if str(args.split_mode) == "leaky":
            test_indices = list(range(len(rows)))
            model_path = base_model_path
        else:
            test_indices = [
                index
                for index, row in enumerate(rows)
                if _split_group(row, str(args.split_mode)) == group
            ]
            model_path = _model_path_for_group(base_model_path, group)
        model = _load_model(
            model_path,
            centroids=centroids,
            args=args,
            device=device,
        )
        fold_embeddings: List[np.ndarray] = []
        for offset in range(0, len(test_indices), int(args.eval_batch_size)):
            batch_indices = test_indices[offset : offset + int(args.eval_batch_size)]
            enhanced = _separate_batch(
                model,
                mixtures[batch_indices],
                [labels[index] for index in batch_indices],
                centroids=centroids,
                n_fft=int(args.n_fft),
                hop_length=int(args.hop_length),
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
                f"embedded_eval_unet group={group} "
                f"{min(offset + int(args.eval_batch_size), len(test_indices))}/{len(test_indices)}",
                flush=True,
            )
        matrix = np.vstack(fold_embeddings).astype(np.float32)
        for index, embedding in zip(test_indices, matrix):
            embeddings_by_index[index] = embedding
        fold_summaries.append({"group": group, "test_rows": len(test_indices)})

    missing = [index for index in range(len(rows)) if index not in embeddings_by_index]
    if missing:
        raise RuntimeError(f"Missing embeddings for {missing[:10]}")
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
        description="Train a one-hot STFT U-Net directly on flattened eval-window stems."
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
        default=Path("/tmp/codex_eval_stem_pairs_s300.npz"),
    )
    parser.add_argument("--model-output", type=Path, default=Path("/tmp/codex_eval_unet.pt"))
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_eval_unet_embeddings.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/codex_eval_unet_results.json"))
    parser.add_argument(
        "--conditioning",
        choices=("one_hot",),
        default="one_hot",
    )
    parser.add_argument(
        "--split-mode",
        choices=("leaky", "leave_group", "leave_session"),
        default="leave_group",
    )
    parser.add_argument("--evaluation-split", choices=("window", "session"), default="window")
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
    parser.add_argument("--cond-channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
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

    groups = (
        ["all"]
        if str(args.split_mode) == "leaky"
        else sorted({_split_group(row, str(args.split_mode)) for row in pair_rows})
    )
    fold_summaries: List[Dict[str, object]] = []
    for group in groups:
        if str(args.split_mode) == "leaky":
            train_indices = list(range(len(pair_rows)))
            test_indices = train_indices
            model_path = args.model_output.expanduser()
        else:
            test_indices = [
                index
                for index, row in enumerate(pair_rows)
                if _split_group(row, str(args.split_mode)) == group
            ]
            test_set = set(test_indices)
            train_indices = [index for index in range(len(pair_rows)) if index not in test_set]
            model_path = _model_path_for_group(args.model_output.expanduser(), group)
        print(
            f"eval_unet_fold group={group} train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        _train_model(
            mixtures[train_indices],
            targets[train_indices],
            [labels[index] for index in train_indices],
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
                "train_speakers": dict(Counter(labels[index] for index in train_indices)),
                "test_speakers": dict(Counter(labels[index] for index in test_indices)),
                "model_path": str(model_path),
            }
        )

    if not args.embedding_output.expanduser().exists():
        embed_summaries = _write_embeddings_for_models(
            mixtures,
            labels,
            pair_rows,
            centroids=centroids,
            args=args,
            output_path=args.embedding_output.expanduser(),
            sample_rate=int(args.sample_rate),
            device=device,
        )
        for summary, embed_summary in zip(fold_summaries, embed_summaries):
            summary.update(embed_summary)

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
        "model": "eval_stem_one_hot_stft_unet_mask",
        "split_mode": str(args.split_mode),
        "evaluation_split": str(args.evaluation_split),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
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
                "eval_stem_unet_true_target/lda_shrinkage",
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
