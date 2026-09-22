from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    ConditionedTasNetExtractor,
    _conditioning_vectors,
    _normalize_batch,
    _si_snr,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
    evaluate_mask_embeddings,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _load_eval_stem_pairs(
    rows: Sequence[MaskRow],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[MaskRow]]:
    if output_path.exists():
        payload = np.load(output_path, allow_pickle=False)
        pair_rows = [
            MaskRow(
                window=str(window),
                index=int(index),
                truth=str(truth),
                target_file=str(target_file),
                target_share=float(share),
                active_5pct=int(active),
            )
            for window, index, truth, target_file, share, active in zip(
                payload["windows"].tolist(),
                payload["indices"].tolist(),
                payload["truths"].tolist(),
                payload["target_files"].tolist(),
                payload["target_shares"].tolist(),
                payload["active_5pct"].tolist(),
            )
        ]
        return (
            np.asarray(payload["mixtures"], dtype=np.float32),
            np.asarray(payload["targets"], dtype=np.float32),
            [str(item) for item in payload["truths"].tolist()],
            pair_rows,
        )

    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    samples = int(round(window_seconds * sample_rate))
    mixtures: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    ordered: List[MaskRow] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        window_dir = prepared_root / window_name
        mixed = _load_audio(window_dir / "mixed.wav", sample_rate)
        source_cache: Dict[str, np.ndarray] = {}
        for row in window_rows:
            if row.target_file not in source_cache:
                source_cache[row.target_file] = _load_audio(
                    window_dir / "clips" / row.target_file,
                    sample_rate,
                )
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            mixtures.append(_slice_wave(mixed, start_sample, start_sample + samples))
            targets.append(
                _slice_wave(
                    source_cache[row.target_file],
                    start_sample,
                    start_sample + samples,
                )
            )
            ordered.append(row)
        print(f"loaded_eval_pairs {window_name}: {len(window_rows)} rows", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mixtures=np.stack(mixtures).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        windows=np.asarray([row.window for row in ordered]),
        indices=np.asarray([row.index for row in ordered], dtype=np.int32),
        truths=np.asarray([row.truth for row in ordered]),
        target_files=np.asarray([row.target_file for row in ordered]),
        target_shares=np.asarray([row.target_share for row in ordered], dtype=np.float32),
        active_5pct=np.asarray([row.active_5pct for row in ordered], dtype=np.int16),
    )
    return (
        np.stack(mixtures).astype(np.float32),
        np.stack(targets).astype(np.float32),
        [row.truth for row in ordered],
        ordered,
    )


def _sample_indices(labels: Sequence[str], *, batch_size: int, rng: random.Random) -> List[int]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_speaker[str(label)].append(index)
    speakers = [speaker for speaker in CORE_SPEAKERS if by_speaker.get(speaker)]
    return [rng.choice(by_speaker[rng.choice(speakers)]) for _ in range(batch_size)]


def _load_supplement_pairs(
    path: Path | None,
    *,
    max_per_speaker: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if path is None or not path.exists():
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            [],
        )
    payload = np.load(path, allow_pickle=False)
    targets = np.asarray(payload["targets"], dtype=np.float32)
    interferers = np.asarray(payload["interferers"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    indices = list(range(len(labels)))
    if max_per_speaker > 0:
        rng = random.Random(seed)
        by_speaker: Dict[str, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            by_speaker[label].append(index)
        indices = []
        for speaker in sorted(by_speaker):
            speaker_indices = list(by_speaker[speaker])
            rng.shuffle(speaker_indices)
            indices.extend(speaker_indices[:max_per_speaker])
        indices.sort()
    target_rows = targets[indices].astype(np.float32)
    interferer_rows = interferers[indices].astype(np.float32)
    mixtures = target_rows + interferer_rows
    peaks = np.maximum(np.max(np.abs(mixtures), axis=1, keepdims=True), 1e-6)
    scale = np.minimum(0.95 / peaks, 1.0).astype(np.float32)
    mixtures = mixtures * scale
    target_rows = target_rows * scale
    return (
        mixtures.astype(np.float32),
        target_rows.astype(np.float32),
        [labels[index] for index in indices],
    )


def _train_model(
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
    model_path: Path | None = None,
) -> ConditionedTasNetExtractor:
    embedding_dim = next(iter(centroids.values())).shape[0]
    model = ConditionedTasNetExtractor(
        embedding_dim=embedding_dim,
        enc_feats=int(args.enc_feats),
        bottleneck=int(args.bottleneck),
        cond_dim=int(args.cond_dim),
        enc_kernel=int(args.enc_kernel),
        layers=int(args.layers),
        stacks=int(args.stacks),
        mask_activation=str(args.mask_activation),
    ).to(device)
    if model_path is None:
        model_path = args.model_output.expanduser()
    if model_path.exists():
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    window = torch.hann_window(int(args.n_fft), device=device)
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        indices = _sample_indices(labels, batch_size=int(args.batch_size), rng=rng)
        mixture = torch.from_numpy(mixtures[indices]).to(device)
        target = torch.from_numpy(targets[indices]).to(device)
        enrollment = torch.from_numpy(np.stack([centroids[labels[index]] for index in indices])).to(
            device
        )
        mixture_norm, scale = _normalize_batch(mixture)
        target_norm = target / scale
        estimate = model(mixture_norm, enrollment)
        si_loss = -_si_snr(estimate, target_norm).mean()
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        if float(args.stft_loss_weight) > 0:
            est_stft = torch.stft(
                estimate,
                n_fft=int(args.n_fft),
                hop_length=int(args.hop_length),
                win_length=int(args.n_fft),
                window=window,
                return_complex=True,
            )
            target_stft = torch.stft(
                target_norm,
                n_fft=int(args.n_fft),
                hop_length=int(args.hop_length),
                win_length=int(args.n_fft),
                window=window,
                return_complex=True,
            )
            stft_loss = torch.nn.functional.l1_loss(
                torch.log1p(torch.abs(est_stft)),
                torch.log1p(torch.abs(target_stft)),
            )
        else:
            stft_loss = torch.zeros((), device=device)
        loss = (
            si_loss
            + float(args.wave_loss_weight) * wav_loss
            + float(args.stft_loss_weight) * stft_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f} "
                f"stft={stft_loss.item():.5f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, model_path)
    return model


def _extract_batch(
    model: ConditionedTasNetExtractor,
    mixtures: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    device: str,
) -> np.ndarray:
    mixture = torch.from_numpy(np.asarray(mixtures, dtype=np.float32)).to(device)
    enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(device)
    with torch.inference_mode():
        mixture_norm, scale = _normalize_batch(mixture)
        estimate = model(mixture_norm, enrollment) * scale
    return estimate.detach().cpu().numpy().astype(np.float32)


def _write_embeddings(
    rows: Sequence[MaskRow],
    mixtures: np.ndarray,
    labels: Sequence[str],
    *,
    model: ConditionedTasNetExtractor,
    centroids: Mapping[str, np.ndarray],
    output_path: Path,
    sample_rate: int,
    batch_size: int,
    device: str,
) -> None:
    del sample_rate
    titanet = _load_titanet(device)
    embeddings: List[np.ndarray] = []
    for offset in range(0, len(rows), batch_size):
        enhanced = _extract_batch(
            model,
            mixtures[offset : offset + batch_size],
            labels[offset : offset + batch_size],
            centroids=centroids,
            device=device,
        )
        embeddings.append(
            _embed_waveforms(
                titanet,
                list(enhanced),
                sample_rate=16000,
                batch_size=batch_size,
                device=device,
            )
        )
        print(f"embedded_eval_stem {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.vstack(embeddings).astype(np.float32),
        windows=np.asarray([row.window for row in rows]),
        indices=np.asarray([row.index for row in rows], dtype=np.int32),
        truths=np.asarray([row.truth for row in rows]),
        target_shares=np.asarray([row.target_share for row in rows], dtype=np.float32),
        active_5pct=np.asarray([row.active_5pct for row in rows], dtype=np.int16),
    )


def _safe_group(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _session_group(window_name: str) -> str:
    if window_name.startswith("short_segment_slice/"):
        return "Session61"
    return window_name.split("/", 1)[0]


def _split_group(row: MaskRow, mode: str) -> str:
    if mode == "leave_group":
        return _window_group(row.window)
    if mode == "leave_session":
        return _session_group(row.window)
    raise ValueError(f"Unsupported split mode for grouped folds: {mode}")


def _write_leave_group_embeddings(
    rows: Sequence[MaskRow],
    mixtures: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
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
    supplement_path = (
        args.supplement_training_cache.expanduser()
        if args.supplement_training_cache is not None
        else None
    )
    supplement_mix, supplement_targets, supplement_labels = _load_supplement_pairs(
        supplement_path,
        max_per_speaker=int(args.supplement_max_per_speaker),
        seed=int(args.seed),
    )
    for group in groups:
        test_indices = [
            index
            for index, row in enumerate(rows)
            if _split_group(row, str(args.split_mode)) == group
        ]
        train_indices = [index for index in range(len(rows)) if index not in set(test_indices)]
        fold_model_path = base_model_path.with_name(
            f"{base_model_path.stem}_{_safe_group(group)}{base_model_path.suffix}"
        )
        print(
            f"extractor_fold group={group} train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        train_mixtures = mixtures[train_indices]
        train_targets = targets[train_indices]
        train_labels = [labels[index] for index in train_indices]
        if supplement_labels:
            train_mixtures = np.concatenate([train_mixtures, supplement_mix], axis=0)
            train_targets = np.concatenate([train_targets, supplement_targets], axis=0)
            train_labels = train_labels + list(supplement_labels)
        model = _train_model(
            train_mixtures,
            train_targets,
            train_labels,
            centroids=centroids,
            args=args,
            device=device,
            model_path=fold_model_path,
        )
        fold_embeddings: List[np.ndarray] = []
        for offset in range(0, len(test_indices), int(args.eval_batch_size)):
            batch_indices = test_indices[offset : offset + int(args.eval_batch_size)]
            enhanced = _extract_batch(
                model,
                mixtures[batch_indices],
                [labels[index] for index in batch_indices],
                centroids=centroids,
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
                f"embedded_lgo group={group} {min(offset + int(args.eval_batch_size), len(test_indices))}/{len(test_indices)}",
                flush=True,
            )
        fold_matrix = np.vstack(fold_embeddings).astype(np.float32)
        for index, embedding in zip(test_indices, fold_matrix):
            embeddings_by_index[index] = embedding
        fold_summaries.append(
            {
                "group": group,
                "train_rows": len(train_indices),
                "supplement_rows": len(supplement_labels),
                "test_rows": len(test_indices),
                "train_speakers": dict(Counter(labels[index] for index in train_indices)),
                "supplement_speakers": dict(Counter(supplement_labels)),
                "test_speakers": dict(Counter(labels[index] for index in test_indices)),
                "model_path": str(fold_model_path),
            }
        )

    missing = [index for index in range(len(rows)) if index not in embeddings_by_index]
    if missing:
        raise RuntimeError(f"Missing leave-group embeddings for {missing[:10]}")
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


def _load_embeddings(path: Path) -> Tuple[np.ndarray, List[MaskRow]]:
    payload = np.load(path, allow_pickle=False)
    rows = [
        MaskRow(
            window=str(window),
            index=int(index),
            truth=str(truth),
            target_file="",
            target_share=float(share),
            active_5pct=int(active),
        )
        for window, index, truth, share, active in zip(
            payload["windows"].tolist(),
            payload["indices"].tolist(),
            payload["truths"].tolist(),
            payload["target_shares"].tolist(),
            payload["active_5pct"].tolist(),
        )
    ]
    return np.asarray(payload["embeddings"], dtype=np.float32), rows


def _rows_for_items(items) -> Tuple[np.ndarray, List[str]]:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for item in items:
        vectors.extend(np.asarray(item.embeddings, dtype=np.float32))
        labels.extend(item.truths)
    return np.vstack(vectors), labels


def _evaluate_embeddings_with_split(
    embeddings: np.ndarray,
    rows: Sequence[MaskRow],
    *,
    clean_bank,
    training_items: Mapping[str, object],
    model_name: str,
    split_mode: str,
) -> Dict[str, object]:
    if split_mode == "window":
        return evaluate_mask_embeddings(
            embeddings,
            rows,
            clean_bank=clean_bank,
            training_items=training_items,
            model_name=model_name,
        )
    if split_mode != "session":
        raise ValueError(f"Unknown evaluation split: {split_mode}")

    from speaker_id_architecture_sweep import _fit_predict

    groups = sorted({_session_group(row.window) for row in rows})
    predictions = ["unknown"] * len(rows)
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_session_group(row.window)].append(index)

    for group in groups:
        train_items = [
            item for item in training_items.values() if _session_group(item.window.name) != group
        ]
        train_x, train_y = _rows_for_items(train_items)
        train_x = np.vstack([clean_bank.embeddings, train_x]).astype(np.float32)
        train_y = list(clean_bank.labels) + train_y
        test_indices = by_group_index[group]
        predicted = _fit_predict(model_name, train_x, train_y, embeddings[test_indices])
        for index, label in zip(test_indices, predicted):
            predictions[index] = label

    return {
        "name": f"eval_split_{split_mode}/{model_name}",
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a target extractor directly on flattened eval-window stems."
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
    parser.add_argument("--model-output", type=Path, default=Path("/tmp/codex_eval_stem_tasnet.pt"))
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_tasnet_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_eval_stem_tasnet_results.json")
    )
    parser.add_argument(
        "--conditioning",
        choices=("clean_bank_centroid", "one_hot"),
        default="one_hot",
    )
    parser.add_argument(
        "--split-mode",
        choices=("leaky", "leave_group", "leave_session"),
        default="leaky",
    )
    parser.add_argument("--evaluation-split", choices=("window", "session"), default="window")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument("--stft-loss-weight", type=float, default=0.0)
    parser.add_argument("--supplement-training-cache", type=Path)
    parser.add_argument("--supplement-max-per-speaker", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
    parser.add_argument(
        "--mask-activation",
        choices=("sigmoid", "relu", "softplus"),
        default="sigmoid",
    )
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
    fold_summaries: List[Dict[str, object]] = []
    if str(args.split_mode) == "leaky":
        model = _train_model(
            mixtures,
            targets,
            labels,
            centroids=centroids,
            args=args,
            device=device,
        )
        if not args.embedding_output.expanduser().exists():
            _write_embeddings(
                pair_rows,
                mixtures,
                labels,
                model=model,
                centroids=centroids,
                output_path=args.embedding_output.expanduser(),
                sample_rate=int(args.sample_rate),
                batch_size=int(args.eval_batch_size),
                device=device,
            )
    elif not args.embedding_output.expanduser().exists():
        fold_summaries = _write_leave_group_embeddings(
            pair_rows,
            mixtures,
            targets,
            labels,
            centroids=centroids,
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
        "model": "eval_stem_conditioned_tasnet_target_extractor",
        "split_mode": str(args.split_mode),
        "diagnostic_leakage": (
            "extractor trained on the same selected eval-stem rows"
            if str(args.split_mode) == "leaky"
            else f"extractor trained {args.split_mode.replace('_', '-')}-out on selected eval-stem rows"
        ),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "conditioning": str(args.conditioning),
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
                "eval_stem_tasnet_true_target/lda_shrinkage",
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
