from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_NAME = "mago-ai/ultra_diar_streaming_sortformer_8spk_v1"


def _as_plain_mapping(config: Any) -> dict:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        if hasattr(config, "items"):
            return dict(config.items())
        raise
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def _make_config(config: Mapping[str, object]) -> Any:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return dict(config)
    return OmegaConf.create(config)


def build_diarization_data_config(
    base_config: Any,
    *,
    manifest_path: Path,
    sample_rate: int,
    num_speakers: int,
    session_len_seconds: float,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Any:
    config = _as_plain_mapping(base_config)
    config.update(
        {
            "manifest_filepath": str(manifest_path.resolve()),
            "sample_rate": int(sample_rate),
            "num_spks": int(num_speakers),
            "session_len_sec": float(session_len_seconds),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": False,
            "shuffle": bool(shuffle),
            "use_lhotse": False,
            "use_bucketing": False,
            "drop_last": False,
            "soft_label_thres": float(config.get("soft_label_thres", 0.5)),
            "soft_targets": bool(config.get("soft_targets", False)),
        }
    )
    return _make_config(config)


def _set_optimizer_config(
    model: Any, *, learning_rate: float, min_learning_rate: float | None
) -> None:
    model.cfg.optim.lr = float(learning_rate)
    if "sched" in model.cfg.optim and model.cfg.optim.sched is not None:
        model.cfg.optim.sched.warmup_steps = 0
        model.cfg.optim.sched.warmup_ratio = None
        model.cfg.optim.sched.min_lr = float(
            min_learning_rate if min_learning_rate is not None else learning_rate
        )


def _load_model(
    *,
    model_name: str,
    restore_path: Path | None,
    device: str,
    trainer: Any,
) -> Any:
    from nemo.collections.asr.models import SortformerEncLabelModel

    if restore_path is not None:
        return SortformerEncLabelModel.restore_from(
            restore_path=str(restore_path),
            map_location=device,
            trainer=trainer,
            strict=False,
        )
    return SortformerEncLabelModel.from_pretrained(
        model_name,
        map_location=device,
        trainer=trainer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a NeMo Sortformer diarization checkpoint on exported Drive manifests."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--restore-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-speakers", type=int, default=8)
    parser.add_argument("--session-len-seconds", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import torch
    import lightning.pytorch as pl

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(int(args.seed), workers=True)
    accelerator = (
        "gpu" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_steps=int(args.max_steps),
        limit_val_batches=int(args.limit_val_batches),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    model = _load_model(
        model_name=str(args.model_name),
        restore_path=args.restore_path,
        device=str(args.device),
        trainer=trainer,
    )
    _set_optimizer_config(
        model,
        learning_rate=float(args.learning_rate),
        min_learning_rate=args.min_learning_rate,
    )
    train_config = build_diarization_data_config(
        model.cfg.train_ds,
        manifest_path=args.train_manifest,
        sample_rate=int(args.sample_rate),
        num_speakers=int(args.num_speakers),
        session_len_seconds=float(args.session_len_seconds),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=True,
    )
    model.setup_training_data(train_config)
    validation_config = None
    if args.validation_manifest is not None and int(args.limit_val_batches) > 0:
        validation_config = build_diarization_data_config(
            model.cfg.validation_ds,
            manifest_path=args.validation_manifest,
            sample_rate=int(args.sample_rate),
            num_speakers=int(args.num_speakers),
            session_len_seconds=float(args.session_len_seconds),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            shuffle=False,
        )
        model.setup_validation_data(validation_config)

    trainer.fit(model)
    checkpoint_path = args.output_dir / "sortformer_domain_adapter_final.nemo"
    model.save_to(str(checkpoint_path))
    summary = {
        "train_manifest": str(args.train_manifest),
        "validation_manifest": str(args.validation_manifest) if args.validation_manifest else None,
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_name": str(args.model_name),
        "restore_path": str(args.restore_path) if args.restore_path else None,
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "num_speakers": int(args.num_speakers),
        "session_len_seconds": float(args.session_len_seconds),
        "learning_rate": float(args.learning_rate),
        "min_learning_rate": (
            float(args.min_learning_rate) if args.min_learning_rate is not None else None
        ),
        "limit_val_batches": int(args.limit_val_batches),
        "seed": int(args.seed),
        "logged_metrics": {
            key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            for key, value in trainer.logged_metrics.items()
        },
    }
    (args.output_dir / "sortformer_domain_adapter_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
