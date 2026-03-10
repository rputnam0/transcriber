from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"
_MODEL_CACHE: Dict[Tuple[str, str, str | None], object] = {}
_MODEL_ALIASES = {
    "large-v3": DEFAULT_MODEL_NAME,
    "medium.en": DEFAULT_MODEL_NAME,
    "parakeet": DEFAULT_MODEL_NAME,
    "parakeet-tdt-0.6b-v3": DEFAULT_MODEL_NAME,
}


def resolve_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL_NAME
    return _MODEL_ALIASES.get(model_name, model_name)


def _resolve_dtype(compute_type: str) -> tuple[object, str]:
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(
            "The parakeet backend requires `parakeet-mlx` and `mlx`, which are only "
            "available on Apple Silicon. Install the macOS dependencies and retry."
        ) from exc

    compute = (compute_type or "").lower()
    if compute in {"float32", "fp32"}:
        return mx.float32, "float32"
    if compute in {"float16", "fp16"}:
        return mx.float16, "float16"
    return mx.bfloat16, "bfloat16"


def load_model(
    model_name: str,
    compute_type: str = "float16",
    device: str = "auto",
    download_root: str | None = None,
    local_files_only: bool = False,
):
    """Initialise a Parakeet MLX model for Apple Silicon."""
    resolved_model = resolve_model_name(model_name)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dtype, dtype_name = _resolve_dtype(compute_type)
    cache_key = (resolved_model, dtype_name, download_root)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if device in {"cpu", "cuda"}:
        logger.warning(
            "Parakeet MLX does not expose explicit device selection; ignoring device=%s.",
            device,
        )

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        logger.warning(
            "Parakeet MLX is designed for Apple Silicon. Current platform is %s/%s.",
            platform.system(),
            platform.machine(),
        )

    try:
        from parakeet_mlx import from_pretrained  # type: ignore
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(
            "Failed to import `parakeet_mlx`. Install `parakeet-mlx` on Apple Silicon "
            "or choose another backend."
        ) from exc

    model = from_pretrained(
        resolved_model,
        dtype=dtype,
        cache_dir=download_root,
    )
    _MODEL_CACHE[cache_key] = model
    return model


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

    text = str(getattr(result, "text", "") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "text": text, "speaker": None}]


def transcribe_file(
    path: str,
    model,
    batch_size: int = 32,
    *,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
) -> List[dict]:
    """Run Parakeet transcription and return the repo's standard segment shape."""
    del batch_size  # Parakeet MLX uses chunking instead of Whisper-style batch sizes.
    result = model.transcribe(
        Path(path),
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )
    return _result_to_segments(result)
