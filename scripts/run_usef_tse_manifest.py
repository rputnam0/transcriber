from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
import types
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


DEFAULT_HF_REPO = "ZBang/USEF-TSE"
DEFAULT_TFGRIDNET_CONFIG = "chkpt/USEF-TFGridNet/config.yaml"
DEFAULT_TFGRIDNET_CHECKPOINT = "chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar"
DEFAULT_LAURA_HF_REPO = "ZBang/USEF-Laura-TSE"
DEFAULT_LAURA_CHECKPOINT = "usef_front_end.pth"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _install_tfgridnet_import_shims() -> None:
    # USEF-TFGridNet imports python_speech_features through a shared feature module,
    # but the TFGridNet path only needs STFT/iSTFT. This keeps the experiment
    # runnable without adding an unused repo dependency.
    if "python_speech_features" not in sys.modules:
        module = types.ModuleType("python_speech_features")
        sigproc = types.ModuleType("python_speech_features.sigproc")
        module.sigproc = sigproc
        sys.modules["python_speech_features"] = module
        sys.modules["python_speech_features.sigproc"] = sigproc


def _load_hyperpyyaml_config(config_path: Path) -> Mapping[str, object]:
    _install_tfgridnet_import_shims()
    import ruamel.yaml
    from hyperpyyaml import load_hyperpyyaml

    if not hasattr(ruamel.yaml.Loader, "max_depth"):
        ruamel.yaml.Loader.max_depth = 0
    with config_path.open("r", encoding="utf-8") as handle:
        return load_hyperpyyaml(handle.read())


def _load_usef_tfgridnet(
    *,
    usef_repo: Path,
    hf_repo: str,
    checkpoint_file: str,
    checkpoint_path: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(usef_repo))
    config_path = usef_repo / DEFAULT_TFGRIDNET_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"USEF-TFGridNet config not found: {config_path}")
    hparams = _load_hyperpyyaml_config(config_path)
    model = hparams["modules"]["masknet"]
    ckpt_path = checkpoint_path or Path(hf_hub_download(hf_repo, checkpoint_file))
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = OrderedDict(
        (
            key.replace("module.", "").replace("convolution_", "convolution_module."),
            value,
        )
        for key, value in payload["model_state_dict"].items()
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected USEF checkpoint keys: missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model


def _load_usef_laura_front(
    *,
    usef_repo: Path,
    checkpoint_path: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(usef_repo))
    config_path = usef_repo / "configs/front_end.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"USEF-Laura front-end config not found: {config_path}")
    from usef_laura_tse.config import load_config
    from usef_laura_tse.front_end.build import build_front_end

    model = build_front_end(load_config(config_path))
    resolved_checkpoint = checkpoint_path or Path(
        hf_hub_download(DEFAULT_LAURA_HF_REPO, DEFAULT_LAURA_CHECKPOINT)
    )
    safe_globals = [ReduceLROnPlateau, Adam, defaultdict, dict]
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(
            resolved_checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"USEF-Laura checkpoint has no state dictionary: {resolved_checkpoint}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wave, sample_rate = torchaudio.load(str(path))
    if wave.ndim > 1:
        wave = wave.mean(dim=0)
    return wave.to(torch.float32).flatten(), int(sample_rate)


def _resample(wave: torch.Tensor, *, source_rate: int, target_rate: int) -> torch.Tensor:
    wave = wave.flatten().to(torch.float32)
    if int(source_rate) == int(target_rate):
        return wave
    return torchaudio.functional.resample(
        wave.unsqueeze(0),
        orig_freq=int(source_rate),
        new_freq=int(target_rate),
    ).squeeze(0)


def _concat_enrollments(
    paths: Sequence[Path],
    *,
    base_dir: Path,
    max_seconds: float,
) -> tuple[torch.Tensor, int]:
    waves: list[torch.Tensor] = []
    rates: list[int] = []
    for path in paths:
        wave, sample_rate = _load_mono(base_dir / path)
        waves.append(wave)
        rates.append(sample_rate)
    if not waves:
        raise ValueError("No positive enrollment paths available")
    sample_rate = rates[0]
    if any(rate != sample_rate for rate in rates):
        waves = [
            _resample(wave, source_rate=rate, target_rate=sample_rate)
            for wave, rate in zip(waves, rates)
        ]
    merged = torch.cat(waves)
    limit = int(round(max_seconds * sample_rate))
    return (merged[:limit] if limit > 0 else merged), sample_rate


def _run_chunked(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    enrollment: torch.Tensor,
    *,
    source_rate: int,
    enrollment_rate: int,
    model_rate: int,
    output_rate: int,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
    device: torch.device,
) -> torch.Tensor:
    mix_model = _resample(mixture, source_rate=source_rate, target_rate=model_rate)
    aux_model = _resample(enrollment, source_rate=enrollment_rate, target_rate=model_rate)
    chunk_samples = int(round(chunk_seconds * model_rate))
    if chunk_samples <= 0:
        raise ValueError("--chunk-seconds must be positive")
    overlap_samples = max(0, int(round(chunk_overlap_seconds * model_rate)))
    overlap_samples = min(overlap_samples, max(chunk_samples - 1, 0))
    hop_samples = chunk_samples - overlap_samples
    if hop_samples <= 0:
        raise ValueError("--chunk-overlap-seconds must be smaller than --chunk-seconds")

    output = torch.zeros_like(mix_model)
    weights = torch.zeros_like(mix_model)
    total_chunks = max(1, math.ceil(max(mix_model.numel() - overlap_samples, 1) / hop_samples))
    with torch.inference_mode():
        for index in range(total_chunks):
            start = index * hop_samples
            stop = min(mix_model.numel(), start + chunk_samples)
            chunk = mix_model[start:stop]
            if chunk.numel() == 0:
                continue
            estimate = model(
                chunk.unsqueeze(0).to(device),
                aux_model.unsqueeze(0).to(device),
            )
            estimate = estimate.detach().cpu().reshape(-1)[: chunk.numel()]
            window = _stitch_window(
                estimate.numel(),
                overlap_samples=overlap_samples,
                has_left_context=start > 0,
                has_right_context=stop < mix_model.numel(),
            )
            output[start:stop] += estimate * window
            weights[start:stop] += window
            if stop >= mix_model.numel():
                break
    merged = (
        output / weights.clamp_min(1e-8)
        if mix_model.numel()
        else torch.zeros(0, dtype=torch.float32)
    )
    return _resample(merged, source_rate=model_rate, target_rate=output_rate)


def _stitch_window(
    samples: int,
    *,
    overlap_samples: int,
    has_left_context: bool,
    has_right_context: bool,
) -> torch.Tensor:
    window = torch.ones(samples, dtype=torch.float32)
    if overlap_samples <= 0 or samples <= 1:
        return window
    fade = min(overlap_samples, max(samples // 2, 1))
    if has_left_context:
        window[:fade] = torch.linspace(0.0, 1.0, steps=fade)
    if has_right_context:
        window[-fade:] = torch.linspace(1.0, 0.0, steps=fade)
    return window


def _valid_audio(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        metadata = sf.info(path)
    except Exception:
        return False
    return int(metadata.frames or 0) > 0 and int(metadata.samplerate or 0) > 0


def _relative_path(path_value: object) -> Path:
    return Path(str(path_value))


def _row_is_selected(
    row: Mapping[str, object],
    *,
    row_ids: set[str],
    split_ids: set[str],
    max_rows: int | None,
    count: int,
) -> bool:
    row_id = str(row.get("row_id") or "")
    if row_ids and row_id not in row_ids:
        return False
    if split_ids and str(row.get("split_id") or "") not in split_ids:
        return False
    if max_rows is not None and count >= max_rows:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run USEF-TFGridNet target-speaker extraction over materialized manifest rows."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--usef-repo", type=Path, required=True)
    parser.add_argument("--backend", choices=("legacy", "laura-front"), default="legacy")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--checkpoint-file", default=DEFAULT_TFGRIDNET_CHECKPOINT)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=2.0)
    parser.add_argument("--max-enrollment-seconds", type=float, default=30.0)
    parser.add_argument("--model-sample-rate", type=int)
    parser.add_argument("--output-sample-rate", type=int, default=16000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--row-id", action="append", default=[])
    parser.add_argument("--split-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_dir = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "laura-front":
        model = _load_usef_laura_front(
            usef_repo=args.usef_repo,
            checkpoint_path=args.checkpoint_path,
            device=device,
        )
        model_sample_rate = int(args.model_sample_rate or 16000)
    else:
        model = _load_usef_tfgridnet(
            usef_repo=args.usef_repo,
            hf_repo=args.hf_repo,
            checkpoint_file=args.checkpoint_file,
            checkpoint_path=args.checkpoint_path,
            device=device,
        )
        model_sample_rate = int(args.model_sample_rate or 8000)

    row_ids = set(args.row_id)
    split_ids = set(args.split_id)
    processed = 0
    metadata = []
    for row in _read_jsonl(args.manifest):
        if not _row_is_selected(
            row,
            row_ids=row_ids,
            split_ids=split_ids,
            max_rows=args.max_rows,
            count=processed,
        ):
            continue
        materialized = dict(row.get("materialized") or {})
        mixture_path = materialized.get("mixture_path")
        enrollment_paths = materialized.get("positive_enrollment_paths") or []
        if not mixture_path or not enrollment_paths:
            continue
        row_id = str(row.get("row_id") or "")
        out_path = args.output_dir / f"{row_id}.wav"
        if out_path.exists() and not args.force:
            if not _valid_audio(out_path):
                out_path.unlink()
            else:
                item = {
                    "row_id": row_id,
                    "speaker_id": row.get("speaker_id"),
                    "session": row.get("session"),
                    "estimate_path": str(out_path),
                    "output_sample_rate": args.output_sample_rate,
                    "chunk_seconds": args.chunk_seconds,
                    "chunk_overlap_seconds": args.chunk_overlap_seconds,
                    "model_sample_rate": model_sample_rate,
                    "backend": args.backend,
                    "positive_enrollment_count": len(enrollment_paths),
                    "status": "skipped_existing",
                }
                metadata.append(item)
                processed += 1
                print(json.dumps(item), flush=True)
                continue
        mixture, source_rate = _load_mono(base_dir / _relative_path(mixture_path))
        enrollment, enrollment_rate = _concat_enrollments(
            [_relative_path(path) for path in enrollment_paths],
            base_dir=base_dir,
            max_seconds=args.max_enrollment_seconds,
        )
        estimate = _run_chunked(
            model,
            mixture,
            enrollment,
            source_rate=source_rate,
            enrollment_rate=enrollment_rate,
            model_rate=model_sample_rate,
            output_rate=args.output_sample_rate,
            chunk_seconds=args.chunk_seconds,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
            device=device,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, estimate.numpy().astype(np.float32), args.output_sample_rate)
        processed += 1
        item = {
            "row_id": row_id,
            "speaker_id": row.get("speaker_id"),
            "session": row.get("session"),
            "estimate_path": str(out_path),
            "source_sample_rate": source_rate,
            "enrollment_sample_rate": enrollment_rate,
            "output_sample_rate": args.output_sample_rate,
            "model_sample_rate": model_sample_rate,
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "backend": args.backend,
            "positive_enrollment_count": len(enrollment_paths),
            "status": "processed",
        }
        metadata.append(item)
        print(json.dumps(item), flush=True)

    metadata_path = args.output_dir / "usef_tfgridnet_manifest_run.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "processed_rows": processed,
                "backend": args.backend,
                "model_sample_rate": model_sample_rate,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
