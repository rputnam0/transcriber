from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torchaudio.models import ConvTasNet

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    _limit_rows_by_group,
    _load_clean_records,
    _make_snippet_cache,
    _normalize_wave,
    _slice_wave,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _load_titanet_word_npz,
    evaluate_mask_embeddings,
)


def _load_or_create_snippets(args: argparse.Namespace) -> Tuple[np.ndarray, List[str]]:
    snippet_path = args.snippet_cache.expanduser()
    if not snippet_path.exists():
        sessions = [item.strip() for item in str(args.train_sessions).split(",") if item.strip()]
        records_by_speaker = _load_clean_records(
            args.quality_records.expanduser(),
            sessions=sessions,
            speakers=CORE_SPEAKERS,
            min_duration=float(args.snippet_seconds),
        )
        _make_snippet_cache(
            records_by_speaker=records_by_speaker,
            audio_root=args.audio_root.expanduser(),
            cache_root=args.cache_root.expanduser(),
            output_path=snippet_path,
            sample_rate=int(args.sample_rate),
            duration=float(args.snippet_seconds),
            max_per_speaker=int(args.max_snippets_per_speaker),
            seed=int(args.seed),
        )
    payload = np.load(snippet_path, allow_pickle=False)
    return np.asarray(payload["waves"], dtype=np.float32), [
        str(item) for item in payload["labels"].tolist()
    ]


def _synth_separation_batch(
    waves: np.ndarray,
    labels: Sequence[str],
    *,
    batch_size: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_speaker[str(label)].append(index)

    mixtures: List[np.ndarray] = []
    sources: List[np.ndarray] = []
    for _ in range(batch_size):
        speaker_a, speaker_b = rng.sample(list(CORE_SPEAKERS), 2)
        source_a = _normalize_wave(waves[rng.choice(by_speaker[speaker_a])])
        source_b = _normalize_wave(waves[rng.choice(by_speaker[speaker_b])])
        snr_db = rng.uniform(-8.0, 8.0)
        source_b = source_b * (10.0 ** (-snr_db / 20.0))
        source_pair = np.stack([source_a, source_b]).astype(np.float32)
        mixture = np.sum(source_pair, axis=0)
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.95:
            scale = 0.95 / peak
            mixture = mixture * scale
            source_pair = source_pair * scale
        mixtures.append(mixture.astype(np.float32))
        sources.append(source_pair.astype(np.float32))
    return np.stack(mixtures), np.stack(sources)


def _si_snr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        torch.sum(estimate * target, dim=-1, keepdim=True)
        / torch.sum(target * target, dim=-1, keepdim=True).clamp_min(eps)
    ) * target
    noise = estimate - projection
    ratio = torch.sum(projection * projection, dim=-1).clamp_min(eps) / torch.sum(
        noise * noise, dim=-1
    ).clamp_min(eps)
    return 10.0 * torch.log10(ratio)


def _pit_si_snr_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if estimate.shape[-1] != target.shape[-1]:
        samples = min(estimate.shape[-1], target.shape[-1])
        estimate = estimate[..., :samples]
        target = target[..., :samples]
    same = _si_snr(estimate[:, 0], target[:, 0]) + _si_snr(estimate[:, 1], target[:, 1])
    swapped = _si_snr(estimate[:, 0], target[:, 1]) + _si_snr(estimate[:, 1], target[:, 0])
    return -torch.maximum(same, swapped).mean() / 2.0


def _build_model(args: argparse.Namespace) -> ConvTasNet:
    return ConvTasNet(
        num_sources=2,
        enc_kernel_size=int(args.enc_kernel_size),
        enc_num_feats=int(args.enc_num_feats),
        msk_kernel_size=3,
        msk_num_feats=int(args.mask_num_feats),
        msk_num_hidden_feats=int(args.mask_hidden_feats),
        msk_num_layers=int(args.mask_layers),
        msk_num_stacks=int(args.mask_stacks),
        msk_activate="relu",
    )


def _train_separator(
    waves: np.ndarray,
    labels: Sequence[str],
    *,
    args: argparse.Namespace,
    device: str,
) -> ConvTasNet:
    model = _build_model(args).to(device)
    model_path = args.model_output.expanduser()
    if model_path.exists():
        payload = torch.load(model_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-5
    )
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        mixture_np, sources_np = _synth_separation_batch(
            waves,
            labels,
            batch_size=int(args.batch_size),
            rng=rng,
        )
        mixture = torch.from_numpy(mixture_np).to(device)
        sources = torch.from_numpy(sources_np).to(device)
        estimate = model(mixture.unsqueeze(1))
        loss = _pit_si_snr_loss(estimate, sources)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(f"train_step {step}/{args.train_steps} pit_si_snr_loss={loss.item():.4f}")

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
        },
        model_path,
    )
    return model


def _separate_eval_rows(
    rows: Sequence[MaskRow],
    *,
    model: ConvTasNet,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
    device: str,
) -> Tuple[List[List[np.ndarray]], List[MaskRow], Dict[str, List[MaskRow]]]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    samples = int(round(window_seconds * sample_rate))
    crops: List[np.ndarray] = []
    ordered: List[MaskRow] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(payload["word_starts"], dtype=np.float32)
        ends = np.asarray(payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            crops.append(_slice_wave(mixed, start_sample, start_sample + samples))
            ordered.append(row)
        print(f"queued {window_name}: {len(window_rows)} rows", flush=True)

    source_waves: List[List[np.ndarray]] = [[], []]
    with torch.inference_mode():
        for offset in range(0, len(crops), batch_size):
            batch_np = np.stack(crops[offset : offset + batch_size]).astype(np.float32)
            peaks = np.maximum(np.max(np.abs(batch_np), axis=1, keepdims=True), 1e-6)
            batch = torch.from_numpy(batch_np / peaks).to(device)
            estimate = model(batch.unsqueeze(1)).detach().cpu().numpy()
            if estimate.shape[-1] != samples:
                estimate = estimate[..., :samples]
            estimate = estimate * peaks[:, None, :]
            for row in range(estimate.shape[0]):
                for source_index in range(2):
                    source_waves[source_index].append(
                        estimate[row, source_index].astype(np.float32)
                    )
            print(f"separated {min(offset + batch_size, len(crops))}/{len(crops)}", flush=True)

    return source_waves, ordered, rows_by_window


def _mixed_embeddings_for_rows(
    rows_by_window: Dict[str, List[MaskRow]],
    *,
    titanet_cache_root: Path,
    window_seconds: float,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for window_name, rows in sorted(rows_by_window.items()):
        payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        window_embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        for row in rows:
            embeddings.append(window_embeddings[row.index])
    return np.stack(embeddings).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small domain ConvTasNet separator and score speaker-ID after separation."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--quality-records",
        type=Path,
        default=Path(
            ".outputs/speaker_id_baseline_prod_graph/artifacts/bank/"
            "1e2a7a014a23be63/dataset/quality_records.jsonl"
        ),
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/prod/Audio"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/tmp/codex_domain_separator_cache")
    )
    parser.add_argument(
        "--snippet-cache", type=Path, default=Path("/tmp/codex_domain_separator_snippets.npz")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("/tmp/codex_domain_convtasnet.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_domain_convtasnet_results.json")
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
        "--train-sessions",
        default="Session 50,Session 51,Session 52,Session 53,Session 54,Session 55,"
        "Session 56,Session 57,Session 58,Session 59,Session 60",
    )
    parser.add_argument("--snippet-seconds", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--max-snippets-per-speaker", type=int, default=120)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enc-kernel-size", type=int, default=16)
    parser.add_argument("--enc-num-feats", type=int, default=128)
    parser.add_argument("--mask-num-feats", type=int, default=64)
    parser.add_argument("--mask-hidden-feats", type=int, default=128)
    parser.add_argument("--mask-layers", type=int, default=4)
    parser.add_argument("--mask-stacks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    waves, labels = _load_or_create_snippets(args)
    counts = Counter(labels)
    missing = [speaker for speaker in CORE_SPEAKERS if counts.get(speaker, 0) < 2]
    if missing:
        raise RuntimeError(f"Not enough snippets for {missing}: {dict(counts)}")
    print(f"loaded_snippets {dict(counts)}", flush=True)

    model = _train_separator(waves, labels, args=args, device=device)

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    source_waves, ordered_rows, rows_by_window = _separate_eval_rows(
        rows,
        model=model,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
        sample_rate=int(args.sample_rate),
        batch_size=int(args.eval_batch_size),
        device=device,
    )

    titanet = _load_titanet(device)
    source_results = []
    source_predictions = []
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    for source_index, waves_for_source in enumerate(source_waves):
        embeddings = _embed_waveforms(
            titanet,
            waves_for_source,
            sample_rate=int(args.sample_rate),
            batch_size=16,
            device=device,
        )
        result = evaluate_mask_embeddings(
            embeddings,
            ordered_rows,
            clean_bank=clean_bank,
            training_items=training_items,
            model_name="lda_shrinkage",
        )
        source_predictions.append(result.pop("predictions"))
        source_results.append(result)
        print(f"source{source_index} {result['direct']}", flush=True)

    mixed_result = evaluate_mask_embeddings(
        _mixed_embeddings_for_rows(
            rows_by_window,
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
        ),
        ordered_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
    )
    mixed_predictions = mixed_result.pop("predictions")
    truth = [row.truth for row in ordered_rows]
    oracle_sep_correct = sum(
        1
        for expected, pred_a, pred_b in zip(truth, source_predictions[0], source_predictions[1])
        if expected in (pred_a, pred_b)
    )
    oracle_plus_correct = sum(
        1
        for expected, pred_a, pred_b, pred_mixed in zip(
            truth, source_predictions[0], source_predictions[1], mixed_predictions
        )
        if expected in (pred_a, pred_b, pred_mixed)
    )
    payload = {
        "model": "domain_convtasnet",
        "selected_rows": len(ordered_rows),
        "selected_speakers": dict(Counter(truth)),
        "snippet_counts": dict(counts),
        "train_steps": int(args.train_steps),
        "source0": source_results[0],
        "source1": source_results[1],
        "oracle_sep_sources": {
            "examples": len(truth),
            "correct": oracle_sep_correct,
            "accuracy": oracle_sep_correct / len(truth) if truth else 0.0,
        },
        "mixed_same_rows": mixed_result,
        "oracle_mixed_plus_sep": {
            "examples": len(truth),
            "correct": oracle_plus_correct,
            "accuracy": oracle_plus_correct / len(truth) if truth else 0.0,
        },
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
