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

from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _best_router_threshold,
    _evaluate_candidate_embeddings,
    _load_candidate_embeddings,
)
from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    ConditionedTasNetExtractor,
    _conditioning_vectors,
    _normalize_batch,
    _si_snr,
)
from speaker_id_eval_stem_target_extractor_sweep import _session_group  # noqa: E402
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
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _safe_group(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _split_group(row: MaskRow, mode: str) -> str:
    if mode == "leave_group":
        return _window_group(row.window)
    if mode == "leave_session":
        return _session_group(row.window)
    if mode == "leaky":
        return "all"
    raise ValueError(mode)


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _rms_rows(waves: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(waves, dtype=np.float32), axis=-1)).astype(np.float32)


def _speaker_file_map(window_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    clips_path = window_dir / "reference" / "clips" / "clips.jsonl"
    for line in clips_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        speaker = str(raw.get("speaker") or "")
        file_name = str(raw.get("file") or "")
        if speaker and file_name and speaker not in mapping:
            mapping[speaker] = file_name
    return mapping


def _load_word_owner_cache(
    rows: Sequence[MaskRow],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[MaskRow]]:
    if output_path.exists():
        payload = np.load(output_path, allow_pickle=False)
        cache_rows = [
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
            np.asarray(payload["candidate_sources"], dtype=np.float32),
            np.asarray(payload["candidate_source_rms"], dtype=np.float32),
            cache_rows,
        )

    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    samples = int(round(window_seconds * sample_rate))
    mixtures: List[np.ndarray] = []
    candidate_sources: List[np.ndarray] = []
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
        speaker_files = _speaker_file_map(window_dir)
        source_cache: Dict[str, np.ndarray] = {}
        for speaker, file_name in speaker_files.items():
            if speaker in CORE_SPEAKERS:
                source_cache[speaker] = _load_audio(window_dir / "clips" / file_name, sample_rate)

        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            end_sample = start_sample + samples
            mixtures.append(_slice_wave(mixed, start_sample, end_sample))
            candidate_sources.append(
                np.stack(
                    [
                        (
                            _slice_wave(source_cache[speaker], start_sample, end_sample)
                            if speaker in source_cache
                            else np.zeros(samples, dtype=np.float32)
                        )
                        for speaker in CORE_SPEAKERS
                    ]
                ).astype(np.float32)
            )
            ordered.append(row)
        print(f"loaded_word_owner_sources {window_name}: {len(window_rows)} rows", flush=True)

    mixture_arr = np.stack(mixtures).astype(np.float32)
    source_arr = np.stack(candidate_sources).astype(np.float32)
    source_rms = _rms_rows(source_arr.reshape(-1, source_arr.shape[-1])).reshape(
        source_arr.shape[:2]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mixtures=mixture_arr,
        candidate_sources=source_arr.astype(np.float16),
        candidate_source_rms=source_rms.astype(np.float32),
        candidates=np.asarray(CORE_SPEAKERS),
        windows=np.asarray([row.window for row in ordered]),
        indices=np.asarray([row.index for row in ordered], dtype=np.int32),
        truths=np.asarray([row.truth for row in ordered]),
        target_files=np.asarray([row.target_file for row in ordered]),
        target_shares=np.asarray([row.target_share for row in ordered], dtype=np.float32),
        active_5pct=np.asarray([row.active_5pct for row in ordered], dtype=np.int16),
    )
    return mixture_arr, source_arr, source_rms, ordered


def _sample_pair_indices(
    rows: Sequence[MaskRow],
    candidate_source_rms: np.ndarray,
    *,
    batch_size: int,
    positive_fraction: float,
    hard_negative_probability: float,
    active_rms_threshold: float,
    rng: random.Random,
) -> Tuple[List[int], List[int], np.ndarray]:
    speaker_to_id = {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_speaker[row.truth].append(index)
    speakers = [speaker for speaker in CORE_SPEAKERS if by_speaker.get(speaker)]
    row_indices: List[int] = []
    candidate_indices: List[int] = []
    is_positive: List[bool] = []
    for _ in range(batch_size):
        if rng.random() < positive_fraction:
            speaker = rng.choice(speakers)
            row_index = rng.choice(by_speaker[speaker])
            candidate_index = speaker_to_id[speaker]
            row_indices.append(row_index)
            candidate_indices.append(candidate_index)
            is_positive.append(True)
            continue

        row_index = rng.randrange(len(rows))
        truth_index = speaker_to_id[rows[row_index].truth]
        negatives = [index for index in range(len(CORE_SPEAKERS)) if index != truth_index]
        hard_negatives = [
            index
            for index in negatives
            if float(candidate_source_rms[row_index, index]) >= active_rms_threshold
        ]
        if hard_negatives and rng.random() < hard_negative_probability:
            candidate_index = rng.choice(hard_negatives)
        else:
            candidate_index = rng.choice(negatives)
        row_indices.append(row_index)
        candidate_indices.append(candidate_index)
        is_positive.append(False)
    return row_indices, candidate_indices, np.asarray(is_positive, dtype=bool)


def _model_path_for_group(base_path: Path, group: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{_safe_group(group)}{base_path.suffix}")


def _load_model_from_path(
    path: Path,
    *,
    embedding_dim: int,
    device: str,
    args: argparse.Namespace,
) -> ConditionedTasNetExtractor:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = dict(payload.get("args") or {})
    model = ConditionedTasNetExtractor(
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
) -> ConditionedTasNetExtractor:
    embedding_dim = next(iter(centroids.values())).shape[0]
    if model_path.exists():
        return _load_model_from_path(
            model_path,
            embedding_dim=embedding_dim,
            device=device,
            args=args,
        )

    model = ConditionedTasNetExtractor(
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
        enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(device)
        positive_mask = torch.from_numpy(is_positive).to(device)

        mixture_norm, scale = _normalize_batch(batch_mixture)
        target_norm = target / scale
        estimate = model(mixture_norm, enrollment)
        if bool(positive_mask.any()):
            si_loss = -_si_snr(estimate[positive_mask], target_norm[positive_mask]).mean()
        else:
            si_loss = torch.zeros((), device=device)
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        if bool((~positive_mask).any()):
            silence_loss = torch.mean(torch.abs(estimate[~positive_mask]))
        else:
            silence_loss = torch.zeros((), device=device)
        loss = (
            float(args.si_loss_weight) * si_loss
            + float(args.wave_loss_weight) * wav_loss
            + float(args.silence_loss_weight) * silence_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f} "
                f"silence={silence_loss.item():.5f} pos={int(is_positive.sum())}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "speakers": CORE_SPEAKERS,
            "objective": "word_owner_positive_target_else_silence",
        },
        model_path,
    )
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
    model_cache: Dict[str, ConditionedTasNetExtractor] = {}
    embeddings_by_index: Dict[int, np.ndarray] = {}
    rms_by_index: Dict[int, np.ndarray] = {}
    peak_by_index: Dict[int, np.ndarray] = {}
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
            extracted = _extract_batch(
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
            row_rms = _rms_rows(extracted_by_row.reshape(-1, extracted_by_row.shape[-1])).reshape(
                len(batch_indices),
                len(candidates),
            )
            row_peak = np.max(np.abs(extracted_by_row), axis=-1).astype(np.float32)
            for index, embedding, rms, peak in zip(
                batch_indices,
                embedded_by_row,
                row_rms,
                row_peak,
            ):
                embeddings_by_index[index] = embedding.astype(np.float32)
                rms_by_index[index] = rms.astype(np.float32)
                peak_by_index[index] = peak.astype(np.float32)
            print(
                f"embedded_word_owner group={group} "
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


def _energy_predictions(
    values: np.ndarray,
    candidates: Sequence[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    best = order[:, -1]
    second = order[:, -2] if values.shape[1] > 1 else best
    predictions = [candidates[int(index)] for index in best]
    confidence = values[np.arange(values.shape[0]), best]
    margin = confidence - values[np.arange(values.shape[0]), second]
    return predictions, confidence.astype(np.float32), margin.astype(np.float32)


def _energy_router_predictions(
    rows: Sequence[MaskRow],
    energy_pred: Sequence[str],
    energy_confidence: np.ndarray,
    mixed_pred: Sequence[str],
) -> List[str]:
    predictions = ["unknown"] * len(rows)
    by_group: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[_window_group(row.window)].append(index)
    for group, test_indices in by_group.items():
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        threshold = _best_router_threshold(
            energy_confidence[train_indices],
            [energy_pred[index] for index in train_indices],
            [mixed_pred[index] for index in train_indices],
            [rows[index].truth for index in train_indices],
        )
        for index in test_indices:
            predictions[index] = (
                energy_pred[index]
                if float(energy_confidence[index]) >= threshold
                else mixed_pred[index]
            )
        print(f"energy_router group={group} threshold={threshold:.6f}", flush=True)
    return predictions


def _load_output_metadata(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return (
        np.asarray(payload["output_rms"], dtype=np.float32),
        np.asarray(payload["output_peak"], dtype=np.float32),
    )


def _mixed_predictions_from_diagnostics(
    rows: Sequence[MaskRow],
    diagnostics: Sequence[Mapping[str, object]],
) -> List[str]:
    predictions = ["unknown"] * len(rows)
    for item in diagnostics:
        predictions[int(item["index"])] = str(item["mixed_pred"])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a word-owner calibrated target extractor: true speaker outputs the source "
            "crop; all other candidate speakers output silence."
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
        default=Path("/tmp/codex_word_owner_tasnet_s300.pt"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("/tmp/codex_word_owner_candidate_embeddings_s300.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_word_owner_tasnet_s300.json"),
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
    parser.add_argument("--wave-loss-weight", type=float, default=0.1)
    parser.add_argument("--silence-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
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
            f"word_owner_fold group={group} train={len(train_indices)} test={len(test_indices)}",
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
    output_rms, output_peak = _load_output_metadata(args.candidate_output.expanduser())
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
        "model": "word_owner_conditioned_tasnet",
        "objective": "truth_candidate_source_else_silence",
        "split_mode": str(args.split_mode),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "conditioning": str(args.conditioning),
        "train_steps": int(args.train_steps),
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
