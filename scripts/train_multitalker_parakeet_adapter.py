from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
TRAINABLE_PREFIXES = {
    "kernels": ("spk_kernels.", "bg_spk_kernels."),
    "kernels-decoder": ("spk_kernels.", "bg_spk_kernels.", "decoder.", "joint."),
    "decoder": ("decoder.", "joint."),
}


def configure_trainable_parameters(model: Any, mode: str) -> dict:
    prefixes = TRAINABLE_PREFIXES.get(mode)
    encoder_tail_layers = 0
    if mode.startswith("asr-tail-"):
        encoder_tail_layers = int(mode.removeprefix("asr-tail-"))
        layer_ids = sorted(
            {
                int(name.split(".")[2])
                for name, _ in model.named_parameters()
                if name.startswith("encoder.layers.")
            }
        )
        prefixes = ("decoder.", "joint.") + tuple(
            f"encoder.layers.{layer_id}." for layer_id in layer_ids[-encoder_tail_layers:]
        )
    trainable = 0
    total = 0
    names = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        enabled = mode == "all" or (prefixes is not None and name.startswith(prefixes))
        parameter.requires_grad = enabled
        if enabled:
            trainable += parameter.numel()
            names.append(name)
    if not trainable:
        raise ValueError(f"No parameters selected for trainable mode {mode!r}")
    return {
        "mode": mode,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "trainable_tensors": names,
        "encoder_tail_layers": encoder_tail_layers,
    }


def _as_plain_mapping(config: Any) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def build_training_mapping(
    base_config: Any,
    *,
    cuts_path: Path,
    batch_size: int,
    num_workers: int,
    max_duration: float,
) -> Any:
    config = _as_plain_mapping(base_config)
    config.update(
        {
            "use_lhotse": True,
            "manifest_filepath": None,
            "cuts_path": str(cuts_path.resolve()),
            "input_cfg": None,
            "sample_rate": 16000,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": False,
            "max_duration": max_duration,
            "min_duration": 0.1,
            "drop_last": False,
            "use_bucketing": False,
            "defer_setup": False,
            "num_speakers": 4,
            "num_sample_per_mel_frame": 160,
            "num_mel_frame_per_asr_frame": 8,
            "shuffle_spk_mapping": False,
        }
    )
    for key in (
        "batch_duration",
        "quadratic_duration",
        "bucketing_batch_size",
        "bucketing_strategy",
        "bucket_buffer_size",
        "shuffle_buffer_size",
    ):
        config.pop(key, None)
    return config


def build_training_config(
    base_config: Any,
    *,
    cuts_path: Path,
    batch_size: int,
    num_workers: int,
    max_duration: float,
) -> Any:
    from omegaconf import OmegaConf

    config = build_training_mapping(
        base_config,
        cuts_path=cuts_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_duration=max_duration,
    )
    return OmegaConf.create(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune multitalker Parakeet on domain mono mixtures and speaker turns."
    )
    parser.add_argument("--train-cuts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--restore-path", type=Path)
    parser.add_argument(
        "--trainable",
        choices=("kernels", "kernels-decoder", "decoder", "asr-tail-2", "asr-tail-4", "all"),
        default="kernels-decoder",
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=31.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-predictions", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--probability-masks",
        action="store_true",
        help="Use raw Sortformer probabilities when cuts provide them.",
    )
    args = parser.parse_args()

    import lightning.pytorch as pl
    import torch
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf

    if not args.train_cuts.exists():
        raise FileNotFoundError(args.train_cuts)
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=args.max_steps,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=args.progress_bar,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        log_every_n_steps=1,
    )
    if args.restore_path:
        model = ASRModel.restore_from(
            restore_path=str(args.restore_path), map_location=map_location, trainer=trainer
        )
    else:
        model = ASRModel.from_pretrained(args.model, map_location=map_location, trainer=trainer)
    if hasattr(model, "wer"):
        model.wer.log_prediction = args.log_predictions
    parameter_summary = configure_trainable_parameters(model, args.trainable)
    model.cfg.optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": args.learning_rate,
            "betas": [0.9, 0.98],
            "weight_decay": args.weight_decay,
        }
    )
    train_config = build_training_config(
        model.cfg.train_ds,
        cuts_path=args.train_cuts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_duration=args.max_duration,
    )
    if args.probability_masks:
        import nemo.collections.asr.models.multitalker_asr_models as multitalker_models

        from transcriber.multitalker_probability_dataset import build_probability_dataset_class

        multitalker_models.LhotseSpeechToTextSpkBpeDataset = build_probability_dataset_class()
    model.setup_training_data(train_config)
    trainer.fit(model)

    checkpoint_path = args.output_dir / "multitalker_parakeet_domain_adapter_final.nemo"
    model.save_to(str(checkpoint_path))
    summary = {
        "train_cuts": str(args.train_cuts),
        "checkpoint_path": str(checkpoint_path),
        "model": args.model,
        "restore_path": str(args.restore_path) if args.restore_path else None,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "precision": args.precision,
        "probability_masks": args.probability_masks,
        "parameter_summary": parameter_summary,
        "logged_metrics": {
            key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            for key, value in trainer.logged_metrics.items()
        },
    }
    (args.output_dir / "multitalker_parakeet_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
