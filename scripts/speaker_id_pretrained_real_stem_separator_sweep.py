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
import torchaudio
from torchaudio.models import ConvTasNet

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_domain_separator_sweep import (  # noqa: E402
    _mixed_embeddings_for_rows,
    _pit_si_snr_loss,
)
from speaker_id_learned_mask_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    _limit_rows_by_group,
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
from speaker_id_real_stem_separator_sweep import _real_stem_batch  # noqa: E402
from speaker_id_target_extractor_sweep import _load_or_create_real_stems  # noqa: E402


def _load_pretrained_convtasnet(device: str) -> ConvTasNet:
    bundle = torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX
    if int(bundle.sample_rate) != 8000:
        raise RuntimeError(f"Expected Libri2Mix ConvTasNet at 8 kHz, got {bundle.sample_rate}")
    model = bundle.get_model().to(device)
    model.eval()
    return model


def _train_separator(
    targets: np.ndarray,
    interferers: np.ndarray,
    *,
    args: argparse.Namespace,
    device: str,
) -> ConvTasNet:
    model_path = args.model_output.expanduser()
    model = _load_pretrained_convtasnet(device)
    if model_path.exists():
        payload = torch.load(model_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    if int(args.train_steps) <= 0:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, model_path)
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        mixture_np, sources_np = _real_stem_batch(
            targets,
            interferers,
            batch_size=int(args.batch_size),
            rng=rng,
            min_target_snr_db=float(args.min_target_snr_db),
            max_target_snr_db=float(args.max_target_snr_db),
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
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, model_path)
    return model


def _resample_wave_list(
    waves: Sequence[np.ndarray],
    *,
    source_rate: int,
    target_rate: int,
    batch_size: int,
    device: str,
) -> List[np.ndarray]:
    if int(source_rate) == int(target_rate):
        return [np.asarray(wave, dtype=np.float32) for wave in waves]
    output: List[np.ndarray] = []
    for offset in range(0, len(waves), batch_size):
        batch_np = np.stack(waves[offset : offset + batch_size]).astype(np.float32)
        batch = torch.from_numpy(batch_np).to(device)
        with torch.inference_mode():
            resampled = torchaudio.functional.resample(
                batch,
                orig_freq=int(source_rate),
                new_freq=int(target_rate),
            )
        output.extend(row.astype(np.float32) for row in resampled.detach().cpu().numpy())
    return output


def _separate_eval_rows(
    rows: Sequence[MaskRow],
    *,
    model: ConvTasNet,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    separator_sample_rate: int,
    embedding_sample_rate: int,
    batch_size: int,
    device: str,
) -> Tuple[List[List[np.ndarray]], List[MaskRow], Dict[str, List[MaskRow]]]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    samples = int(round(window_seconds * separator_sample_rate))
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
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", separator_sample_rate)
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(
                math.floor((midpoint - window_seconds / 2.0) * separator_sample_rate)
            )
            crops.append(_slice_wave(mixed, start_sample, start_sample + samples))
            ordered.append(row)
        print(f"queued {window_name}: {len(window_rows)} rows", flush=True)

    separated_by_source: List[List[np.ndarray]] = [[], []]
    model.eval()
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
                    separated_by_source[source_index].append(
                        estimate[row, source_index].astype(np.float32)
                    )
            print(f"separated {min(offset + batch_size, len(crops))}/{len(crops)}", flush=True)

    resampled_by_source = [
        _resample_wave_list(
            waves,
            source_rate=separator_sample_rate,
            target_rate=embedding_sample_rate,
            batch_size=batch_size,
            device=device,
        )
        for waves in separated_by_source
    ]
    return resampled_by_source, ordered, rows_by_window


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained Libri2Mix ConvTasNet on real Discord stems."
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
        "--cache-root", type=Path, default=Path("/tmp/codex_pretrained_separator_cache")
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/tmp/codex_pretrained_real_stem_pairs_8k.npz"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/tmp/codex_pretrained_real_stem_convtasnet.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_pretrained_real_stem_convtasnet_results.json"),
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
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--max-crops-per-speaker", type=int, default=120)
    parser.add_argument("--min-interferer-ratio", type=float, default=0.05)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--min-target-snr-db", type=float, default=-10.0)
    parser.add_argument("--max-target-snr-db", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--embedding-sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    targets, interferers, labels = _load_or_create_real_stems(args)
    counts = Counter(labels)
    missing = [speaker for speaker in CORE_SPEAKERS if counts.get(speaker, 0) < 2]
    if missing:
        raise RuntimeError(f"Not enough real-stem crops for {missing}: {dict(counts)}")
    print(f"loaded_real_stem_pairs_8k {dict(counts)}", flush=True)

    model = _train_separator(targets, interferers, args=args, device=device)
    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    source_waves, ordered_rows, rows_by_window = _separate_eval_rows(
        rows,
        model=model,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
        separator_sample_rate=int(args.sample_rate),
        embedding_sample_rate=int(args.embedding_sample_rate),
        batch_size=int(args.eval_batch_size),
        device=device,
    )

    titanet = _load_titanet(device)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )

    source_results = []
    source_predictions = []
    for source_index, waves in enumerate(source_waves):
        embeddings = _embed_waveforms(
            titanet,
            waves,
            sample_rate=int(args.embedding_sample_rate),
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
        "model": "pretrained_libri2mix_convtasnet_real_stem_finetune",
        "selected_rows": len(ordered_rows),
        "selected_speakers": dict(Counter(truth)),
        "training_counts": dict(counts),
        "train_steps": int(args.train_steps),
        "separator_sample_rate": int(args.sample_rate),
        "embedding_sample_rate": int(args.embedding_sample_rate),
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
