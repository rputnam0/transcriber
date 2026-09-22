from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torchaudio.models import ConvTasNet

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_domain_separator_sweep import (  # noqa: E402
    _build_model,
    _mixed_embeddings_for_rows,
    _pit_si_snr_loss,
    _separate_eval_rows,
)
from speaker_id_learned_mask_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    _limit_rows_by_group,
    _normalize_wave,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    _collect_training_items,
    _embed_waveforms,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    evaluate_mask_embeddings,
)
from speaker_id_target_extractor_sweep import _load_or_create_real_stems  # noqa: E402


def _real_stem_batch(
    targets: np.ndarray,
    interferers: np.ndarray,
    *,
    batch_size: int,
    rng: random.Random,
    min_target_snr_db: float,
    max_target_snr_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mixtures: List[np.ndarray] = []
    sources: List[np.ndarray] = []
    for _ in range(batch_size):
        index = rng.randrange(targets.shape[0])
        target = _normalize_wave(targets[index])
        interferer = _normalize_wave(interferers[index])
        snr_db = rng.uniform(min_target_snr_db, max_target_snr_db)
        interferer = interferer * (10.0 ** (-snr_db / 20.0))
        source_pair = np.stack([target, interferer]).astype(np.float32)
        mixture = np.sum(source_pair, axis=0)
        peak = max(float(np.max(np.abs(mixture))), 1e-6)
        if peak > 0.95:
            scale = 0.95 / peak
            mixture = mixture * scale
            source_pair = source_pair * scale
        mixtures.append(mixture.astype(np.float32))
        sources.append(source_pair.astype(np.float32))
    return np.stack(mixtures).astype(np.float32), np.stack(sources).astype(np.float32)


def _train_separator(
    targets: np.ndarray,
    interferers: np.ndarray,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ConvTasNet separator on real production Discord stem crops."
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
        "--cache-root", type=Path, default=Path("/tmp/codex_real_stem_separator_cache")
    )
    parser.add_argument(
        "--training-cache", type=Path, default=Path("/tmp/codex_real_stem_separator_pairs.npz")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("/tmp/codex_real_stem_convtasnet.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_real_stem_convtasnet_results.json")
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
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-target-snr-db", type=float, default=-10.0)
    parser.add_argument("--max-target-snr-db", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enc-kernel-size", type=int, default=16)
    parser.add_argument("--enc-num-feats", type=int, default=128)
    parser.add_argument("--mask-num-feats", type=int, default=64)
    parser.add_argument("--mask-hidden-feats", type=int, default=128)
    parser.add_argument("--mask-layers", type=int, default=4)
    parser.add_argument("--mask-stacks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
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
    print(f"loaded_real_stem_pairs {dict(counts)}", flush=True)

    model = _train_separator(targets, interferers, args=args, device=device)
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
        "model": "real_stem_convtasnet",
        "selected_rows": len(ordered_rows),
        "selected_speakers": dict(Counter(truth)),
        "training_counts": dict(counts),
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
