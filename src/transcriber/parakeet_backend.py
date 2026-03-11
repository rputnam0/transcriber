from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_MLX_MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"
_MODEL_CACHE: Dict[Tuple[str, str, str, str | None], "ParakeetModelHandle"] = {}
_MODEL_ALIASES = {
    "large-v3": DEFAULT_MODEL_NAME,
    "medium.en": DEFAULT_MODEL_NAME,
    "parakeet": DEFAULT_MODEL_NAME,
    "parakeet-tdt-0.6b-v3": DEFAULT_MODEL_NAME,
    DEFAULT_MLX_MODEL_NAME: DEFAULT_MODEL_NAME,
}


@dataclass(frozen=True)
class ParakeetModelHandle:
    runtime: str
    model: object


def resolve_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL_NAME
    return _MODEL_ALIASES.get(model_name, model_name)


def _supports_mlx_runtime() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _supports_nemo_runtime() -> bool:
    return platform.system() == "Linux"


def _resolve_torch_device(device: str | None) -> str:
    if device in {"cpu", "cuda"}:
        return device
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_mlx_dtype(compute_type: str) -> tuple[object, str]:
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(
            "The Apple Silicon Parakeet runtime requires `parakeet-mlx` and `mlx`."
        ) from exc

    compute = (compute_type or "").lower()
    if compute in {"float32", "fp32"}:
        return mx.float32, "float32"
    if compute in {"float16", "fp16"}:
        return mx.float16, "float16"
    return mx.bfloat16, "bfloat16"


def _resolve_dtype(compute_type: str) -> tuple[object, str]:
    """Backward-compatible alias for existing tests and callers."""
    return _resolve_mlx_dtype(compute_type)


def _load_mlx_model(
    model_name: str,
    *,
    compute_type: str,
    device: str | None,
    download_root: str | None,
    local_files_only: bool,
) -> ParakeetModelHandle:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dtype, dtype_name = _resolve_mlx_dtype(compute_type)
    cache_key = ("mlx", model_name, dtype_name, download_root)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if device in {"cpu", "cuda"}:
        logger.warning(
            "Parakeet MLX does not expose explicit device selection; ignoring device=%s.",
            device,
        )

    try:
        from parakeet_mlx import from_pretrained  # type: ignore
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(
            "Failed to import `parakeet_mlx`. Install `parakeet-mlx` on Apple Silicon "
            "or choose another backend."
        ) from exc

    resolved_model = DEFAULT_MLX_MODEL_NAME if model_name == DEFAULT_MODEL_NAME else model_name
    model = from_pretrained(
        resolved_model,
        dtype=dtype,
        cache_dir=download_root,
    )
    handle = ParakeetModelHandle(runtime="mlx", model=model)
    _MODEL_CACHE[cache_key] = handle
    return handle


def _load_nemo_model(
    model_name: str,
    *,
    device: str | None,
    download_root: str | None,
    local_files_only: bool,
) -> ParakeetModelHandle:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch_device_name = _resolve_torch_device(device)
    cache_key = ("nemo", model_name, torch_device_name, download_root)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(
            "Failed to import `nemo.collections.asr`. Install `nemo-toolkit[asr]` on Linux "
            "to use the Parakeet backend there."
        ) from exc

    load_kwargs = {"model_name": model_name}
    if download_root:
        load_kwargs["cache_dir"] = download_root
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("cache_dir", None)
        model = nemo_asr.models.ASRModel.from_pretrained(**load_kwargs)

    try:
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256],
        )
    except Exception:
        pass

    try:
        import torch

        torch_device = torch.device(torch_device_name)
    except Exception:
        torch_device = torch_device_name

    try:
        model = model.to(torch_device)
    except Exception:
        if torch_device_name == "cuda" and hasattr(model, "cuda"):
            model = model.cuda()
        elif torch_device_name == "cpu" and hasattr(model, "cpu"):
            model = model.cpu()

    try:
        model.eval()
    except Exception:
        pass

    handle = ParakeetModelHandle(runtime="nemo", model=model)
    _MODEL_CACHE[cache_key] = handle
    return handle


def load_model(
    model_name: str,
    compute_type: str = "float16",
    device: str = "auto",
    download_root: str | None = None,
    local_files_only: bool = False,
) -> ParakeetModelHandle:
    """Initialise a Parakeet runtime for the current platform."""
    resolved_model = resolve_model_name(model_name)

    if _supports_mlx_runtime():
        return _load_mlx_model(
            resolved_model,
            compute_type=compute_type,
            device=device,
            download_root=download_root,
            local_files_only=local_files_only,
        )

    if _supports_nemo_runtime():
        if compute_type not in {"auto", "int8", "float32", "fp32"}:
            logger.info(
                "Parakeet NeMo ignores compute_type=%s; runtime precision is controlled by PyTorch.",
                compute_type,
            )
        return _load_nemo_model(
            resolved_model,
            device=device,
            download_root=download_root,
            local_files_only=local_files_only,
        )

    raise RuntimeError(
        "Parakeet currently supports Apple Silicon via `parakeet-mlx` and Linux via "
        "`nemo-toolkit[asr]`. Use the faster backend on other platforms."
    )


def _result_to_segments(result: object) -> List[dict]:
    sentences = getattr(result, "sentences", None) or []
    segments: List[dict] = []
    for sentence in sentences:
        text = getattr(sentence, "text", "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(getattr(sentence, "start", 0.0) or 0.0),
                "end": float(getattr(sentence, "end", 0.0) or 0.0),
                "text": text,
                "speaker": None,
            }
        )

    if segments:
        return segments

    timestamps = getattr(result, "timestamp", None)
    if timestamps is None and isinstance(result, dict):
        timestamps = result.get("timestamp")
    if isinstance(timestamps, dict):
        for item in timestamps.get("segment") or timestamps.get("segments") or []:
            if isinstance(item, dict):
                text = str(item.get("segment") or item.get("text") or "").strip()
                start = float(item.get("start") or 0.0)
                end = float(item.get("end") or 0.0)
            else:
                text = str(
                    getattr(item, "segment", None) or getattr(item, "text", "") or ""
                ).strip()
                start = float(getattr(item, "start", 0.0) or 0.0)
                end = float(getattr(item, "end", 0.0) or 0.0)
            if not text:
                continue
            segments.append({"start": start, "end": end, "text": text, "speaker": None})

    if segments:
        return segments

    text = str(getattr(result, "text", "") or "").strip()
    if not text and isinstance(result, dict):
        text = str(result.get("text") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "text": text, "speaker": None}]


def transcribe_file(
    path: str,
    model: ParakeetModelHandle | object,
    batch_size: int = 32,
    *,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
) -> List[dict]:
    """Run Parakeet transcription and return the repo's standard segment shape."""
    del batch_size

    runtime = getattr(model, "runtime", None)
    runtime_model = getattr(model, "model", model)
    audio_path = str(Path(path))

    if runtime == "nemo":
        try:
            result = runtime_model.transcribe([audio_path], timestamps=True)
        except TypeError:
            result = runtime_model.transcribe(paths2audio_files=[audio_path], timestamps=True)
        if isinstance(result, list):
            result = result[0] if result else None
        return _result_to_segments(result)

    result = runtime_model.transcribe(
        Path(path),
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )
    return _result_to_segments(result)
