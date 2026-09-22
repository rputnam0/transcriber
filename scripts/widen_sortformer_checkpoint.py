from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch


DEFAULT_SOURCE_MODEL_NAME = "devsy0117/ultra_diar_streaming_sortformer_5spk_v1"
DEFAULT_TARGET_MODEL_NAME = "mago-ai/ultra_diar_streaming_sortformer_8spk_v1"


@dataclass
class TransplantSummary:
    exact_copied: list[str]
    widened: list[dict[str, object]]
    missing_in_source: list[str]
    shape_mismatches: list[dict[str, object]]


def _can_widen(source: torch.Tensor, target: torch.Tensor) -> bool:
    return (
        source.ndim == target.ndim
        and source.ndim >= 1
        and source.shape[0] < target.shape[0]
        and tuple(source.shape[1:]) == tuple(target.shape[1:])
    )


def transplant_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], TransplantSummary]:
    """Copy a smaller-speaker Sortformer state into a larger-speaker target state."""
    output_state: dict[str, torch.Tensor] = {}
    summary = TransplantSummary(
        exact_copied=[],
        widened=[],
        missing_in_source=[],
        shape_mismatches=[],
    )
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            output_state[key] = target_tensor.detach().clone()
            summary.missing_in_source.append(key)
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            output_state[key] = source_tensor.detach().to(
                dtype=target_tensor.dtype,
                device=target_tensor.device,
            )
            summary.exact_copied.append(key)
            continue
        if _can_widen(source_tensor, target_tensor):
            widened_tensor = target_tensor.detach().clone()
            copy_rows = int(source_tensor.shape[0])
            widened_tensor[:copy_rows] = source_tensor.detach().to(
                dtype=target_tensor.dtype,
                device=target_tensor.device,
            )
            output_state[key] = widened_tensor
            summary.widened.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                    "copied_rows": copy_rows,
                    "kept_target_rows": int(target_tensor.shape[0] - copy_rows),
                }
            )
            continue
        output_state[key] = target_tensor.detach().clone()
        summary.shape_mismatches.append(
            {
                "key": key,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
        )
    return output_state, summary


def _load_model(
    *,
    model_name: str,
    restore_path: Path | None,
    device: str,
    trainer: object | None = None,
):
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


def _set_num_speakers(model: object, num_speakers: int) -> None:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for section in ("train_ds", "validation_ds", "test_ds"):
            if hasattr(cfg, section):
                section_cfg = getattr(cfg, section)
                if hasattr(section_cfg, "num_spks"):
                    section_cfg.num_spks = int(num_speakers)
        if hasattr(cfg, "sortformer_modules") and hasattr(cfg.sortformer_modules, "num_spks"):
            cfg.sortformer_modules.num_spks = int(num_speakers)
    modules = getattr(model, "sortformer_modules", None)
    if modules is not None and hasattr(modules, "num_spks"):
        modules.num_spks = int(num_speakers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a wider Sortformer checkpoint by transplanting a smaller-speaker "
            "checkpoint into a larger-speaker target checkpoint."
        )
    )
    parser.add_argument("--source-model-name", default=DEFAULT_SOURCE_MODEL_NAME)
    parser.add_argument("--source-restore-path", type=Path)
    parser.add_argument("--target-model-name", default=DEFAULT_TARGET_MODEL_NAME)
    parser.add_argument("--target-restore-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-speakers", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_model = _load_model(
        model_name=str(args.source_model_name),
        restore_path=args.source_restore_path,
        device=str(args.device),
    )
    target_model = _load_model(
        model_name=str(args.target_model_name),
        restore_path=args.target_restore_path,
        device=str(args.device),
    )
    output_state, summary = transplant_state(source_model.state_dict(), target_model.state_dict())
    load_result = target_model.load_state_dict(output_state, strict=True)
    _set_num_speakers(target_model, int(args.num_speakers))

    checkpoint_path = args.output_dir / "sortformer_widened_final.nemo"
    target_model.save_to(str(checkpoint_path))
    result = {
        "source_model_name": str(args.source_model_name),
        "source_restore_path": str(args.source_restore_path) if args.source_restore_path else None,
        "target_model_name": str(args.target_model_name),
        "target_restore_path": str(args.target_restore_path) if args.target_restore_path else None,
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_speakers": int(args.num_speakers),
        "load_missing_keys": list(getattr(load_result, "missing_keys", [])),
        "load_unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "transplant": asdict(summary),
    }
    summary_path = args.output_dir / "sortformer_widened_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
