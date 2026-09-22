from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from run_streaming_usef_tp_word_owner import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REVISION,
    MODEL_ARGS,
    IdentityHead,
    load_model,
)
from train_ps4_activity_word_owner import (
    EnrollmentSelector,
    _evaluate_chunks,
    _evaluate_full_groups,
    _frame_loss_and_metrics,
    _group_rows,
    _load_group_chunk,
    _metric_summary,
    _sha256,
    _subset_batch_candidates,
    _word_owner_loss,
)
from train_sequence_tsvad_word_owner_baseline import _summarize_groups
from train_tsvad_word_owner_baseline import _write_jsonl
from train_usef_tse_domain_adapter import (
    ForcedReferenceIndex,
    StemCache,
    _forced_reference_coverage,
    _load_rows,
    _parse_csv_set,
    _validate_forced_reference_coverage,
)


def extraction_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    length = min(estimate.shape[-1], target.shape[-1], target_masks.shape[-1])
    estimate = estimate[..., :length]
    target = target[..., :length]
    masks = target_masks[..., :length]
    raw_estimate = estimate
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (estimate * target).sum(dim=-1, keepdim=True)
        * target
        / target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    noise = estimate - projection
    si_sdr = 10.0 * torch.log10(
        projection.square().sum(dim=-1).clamp_min(eps) / noise.square().sum(dim=-1).clamp_min(eps)
    )
    active = masks.mean(dim=-1) >= 0.01
    active_loss = -si_sdr
    silence_loss = 10.0 * torch.log10(raw_estimate.square().mean(dim=-1).clamp_min(eps))
    return torch.where(active, active_loss, silence_loss.clamp_min(-40.0)).mean()


def run_training_batch(
    *,
    extractor,
    batch: Mapping[str, object],
    device: torch.device,
    frame_loss_weight: float,
    word_loss_weight: float,
    extraction_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    mixture = torch.as_tensor(batch["mixture"]).to(device)
    enrollments = torch.as_tensor(batch["enrollments"]).to(device)
    masks = torch.as_tensor(batch["target_masks"]).to(device)
    if extraction_loss_weight > 0.0:
        targets = torch.as_tensor(batch["target_sources"]).to(device)
        estimate, raw_logits = extractor.model(mixture, enrollments)
        logits = raw_logits.squeeze(1)
        tse_loss = extraction_loss(estimate, targets, masks)
    else:
        logits = extractor(mixture, enrollments)
        tse_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    frame_loss, frame_metrics = _frame_loss_and_metrics(logits, masks)
    word_loss, word_metrics = _word_owner_loss(
        logits,
        words=list(batch["words"]),
        speakers=list(batch["speakers"]),
        chunk_start=float(batch["chunk_start"]),
        chunk_seconds=float(batch["chunk_seconds"]),
        frame_hop_seconds=0.008,
    )
    total = (
        frame_loss_weight * frame_loss
        + word_loss_weight * word_loss
        + extraction_loss_weight * tse_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "frame_loss": float(frame_loss.detach().cpu()),
        "word_loss": float(word_loss.detach().cpu()),
        "extraction_loss": float(tse_loss.detach().cpu()),
        **frame_metrics,
        **word_metrics,
    }


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    dev: Mapping[str, object],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "dev": dict(dev),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain-adapt a joint streaming USEF-TP extraction and personal-VAD model."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--initial-adapter-checkpoint", type=Path)
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--stems-wav-root", type=Path)
    parser.add_argument(
        "--stem-cache-sample-rate",
        type=int,
        default=16000,
        help="On-disk stem rate; chunks are resampled to the model's fixed 8 kHz rate after loading.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--train-sessions")
    parser.add_argument("--dev-sessions")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-dev-rows", type=int)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--dev-batches", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=10.0)
    parser.add_argument("--train-candidates-per-step", type=int, default=4)
    parser.add_argument("--active-probability", type=float, default=0.9)
    parser.add_argument("--train-overlap-probability", type=float, default=0.65)
    parser.add_argument("--dev-overlap-probability", type=float, default=0.65)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--word-loss-weight", type=float, default=1.0)
    parser.add_argument("--extraction-loss-weight", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--min-mixture-rms", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-dev-eval", action="store_true")
    parser.add_argument("--max-full-eval-groups", type=int, default=0)
    parser.add_argument("--inference-chunk-seconds", type=float, default=15.0)
    parser.add_argument("--inference-candidate-batch-size", type=int, default=2)
    args = parser.parse_args()

    sample_rate = 8000
    args.sample_rate = sample_rate
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_path or args.source_root / DEFAULT_CHECKPOINT
    device = torch.device(args.device)
    extractor, load_summary = load_model(
        source_root=args.source_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = extractor.model
    initial_step = 0
    if args.initial_adapter_checkpoint:
        initial = torch.load(
            args.initial_adapter_checkpoint,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(initial["model"], strict=True)
        initial_step = int(initial.get("step") or 0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    references = ForcedReferenceIndex(args.forced_reference_jsonl)
    train_rows = _load_rows(
        args.manifest,
        {str(args.train_split)},
        sessions=_parse_csv_set(args.train_sessions),
        max_rows=args.max_train_rows,
    )
    dev_rows = _load_rows(
        args.manifest,
        {str(args.dev_split)},
        sessions=_parse_csv_set(args.dev_sessions),
        max_rows=args.max_dev_rows,
    )
    coverage = {
        "train": _forced_reference_coverage(train_rows, references),
        "dev": _forced_reference_coverage(dev_rows, references),
    }
    for label, report in coverage.items():
        _validate_forced_reference_coverage(
            label=label,
            coverage=report,
            require_forced_reference=True,
            min_row_coverage=1.0,
            min_word_coverage=0.99,
        )
    train_groups = _group_rows(train_rows)
    dev_groups = _group_rows(dev_rows)
    cache = StemCache(
        root=args.stems_cache_root or args.output_dir / "_stems",
        wav_root=args.stems_wav_root or args.output_dir / "_stems8",
        sample_rate=args.stem_cache_sample_rate,
    )
    enrollment_selector = EnrollmentSelector(
        seconds=args.enrollment_seconds,
        sample_rate=sample_rate,
        min_rms=1e-4,
    )
    head = IdentityHead().to(device)
    metadata = {
        "args": vars(args),
        "source_repo": "VMoorjani/Streaming-USEF-TP",
        "source_revision": DEFAULT_REVISION,
        "license_status": "undeclared-research-feasibility-only",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_args": MODEL_ARGS,
        "load": load_summary,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "forced_reference_coverage": coverage,
    }
    (args.output_dir / "streaming_usef_tp_training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    def evaluate_chunks(step: int) -> dict[str, float | int]:
        metrics = _evaluate_chunks(
            groups=dev_groups,
            cache=cache,
            enrollment_selector=enrollment_selector,
            references=references,
            extractor=extractor,
            head=head,
            device=device,
            seed=args.seed + 1,
            batches=args.dev_batches,
            args=args,
        )
        print(json.dumps({"step": step, "dev": metrics}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics

    baseline_dev = evaluate_chunks(initial_step)
    best_accuracy = float(baseline_dev["word_accuracy"])
    best_path = args.output_dir / "best_streaming_usef_tp_adapter.pt"
    save_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        step=initial_step,
        dev=baseline_dev,
    )
    rng = random.Random(args.seed)
    history = [{"step": initial_step, "dev": baseline_dev}]
    train_records = []
    successful_steps = 0
    attempts = 0
    while successful_steps < args.train_steps and attempts < args.train_steps * 40:
        attempts += 1
        try:
            batch = _load_group_chunk(
                train_groups[rng.randrange(len(train_groups))],
                cache=cache,
                enrollment_selector=enrollment_selector,
                references=references,
                rng=rng,
                chunk_seconds=args.chunk_seconds,
                sample_rate=sample_rate,
                overlap_probability=args.train_overlap_probability,
                active_probability=args.active_probability,
                min_mixture_rms=args.min_mixture_rms,
            )
        except (OSError, RuntimeError, ValueError):
            continue
        batch = _subset_batch_candidates(
            batch,
            limit=args.train_candidates_per_step,
            rng=rng,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, record = run_training_batch(
            extractor=extractor,
            batch=batch,
            device=device,
            frame_loss_weight=args.frame_loss_weight,
            word_loss_weight=args.word_loss_weight,
            extraction_loss_weight=args.extraction_loss_weight,
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        successful_steps += 1
        train_records.append(record)
        global_step = initial_step + successful_steps
        if successful_steps == 1 or successful_steps % args.log_every == 0:
            dev = evaluate_chunks(global_step)
            aggregate_train = _metric_summary(train_records[-args.log_every :])
            aggregate_train["extraction_loss"] = float(
                np.mean([item["extraction_loss"] for item in train_records[-args.log_every :]])
            )
            history.append({"step": global_step, "train": aggregate_train, "dev": dev})
            if float(dev["word_accuracy"]) > best_accuracy:
                best_accuracy = float(dev["word_accuracy"])
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    dev=dev,
                )
    if successful_steps < args.train_steps:
        raise RuntimeError(
            f"Only completed {successful_steps}/{args.train_steps} steps after {attempts} attempts"
        )

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    summary = {
        "model": "streaming-usef-tp-domain-adapter",
        "initial_step": initial_step,
        "completed_steps": successful_steps,
        "best_step": int(best["step"]),
        "best_chunk_dev": best["dev"],
        "history": history,
        "attempts": attempts,
        "elapsed_seconds": time.time() - started,
    }
    if args.full_dev_eval:
        group_results, word_records = _evaluate_full_groups(
            dev_groups,
            references=references,
            manifest_dir=args.manifest.resolve().parent,
            extractor=extractor,
            head=head,
            device=device,
            args=args,
        )
        summary["full_dev"] = _summarize_groups(group_results)
        _write_jsonl(args.output_dir / "streaming_usef_tp_dev_groups.jsonl", group_results)
        _write_jsonl(args.output_dir / "streaming_usef_tp_dev_words.jsonl", word_records)
    (args.output_dir / "streaming_usef_tp_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
