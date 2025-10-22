from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Callable
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os

_MODEL_CACHE: Dict[Tuple[str, str, str, Optional[str], bool], object] = {}
_ALIGN_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[object, object]] = {}
_DIAR_CACHE: Dict[Tuple[str, Optional[str]], object] = {}

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            try:
                if torch.cuda.device_count() > 0:
                    # Enable TF32 where available for a speed boost on Ampere+
                    try:
                        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    return "cuda"
            except Exception as exc:  # noqa: BLE001
                logger.warning("CUDA device query failed, falling back to CPU: %s", exc)
    except Exception:
        pass
    return "cpu"


def _cudnn_usable() -> bool:
    """Return True if PyTorch can use cuDNN successfully on this system."""
    try:
        import torch

        return bool(torch.backends.cudnn.is_available())
    except Exception:
        return False


@contextmanager
def _silence_stdio(quiet: bool):
    """Silence stdout/stderr from third-party libs when quiet.

    Keep this context tight around library calls so tqdm output from the CLI remains visible.
    Also tries to clamp ONNX Runtime logger severity to ERROR.
    """
    if not quiet:
        yield
        return
    # Best-effort: set ORT severity
    try:
        import onnxruntime as _ort  # type: ignore

        try:
            _ort.set_default_logger_severity(3)
        except Exception:
            pass
    except Exception:
        pass
    with open(os.devnull, "w") as _devnull:
        with redirect_stdout(_devnull), redirect_stderr(_devnull):
            yield


def _get_model(
    model_name: str,
    device: str,
    compute_type: str,
    cache_root: Optional[str],
    local_files_only: bool,
    *,
    strict_cuda: bool = False,
):
    # Lazy import to honor env vars set by the CLI
    import whisperx  # type: ignore
    cache_key = (model_name, device, compute_type, cache_root, local_files_only)
    if cache_key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[cache_key] = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=cache_root,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # noqa: BLE001
            # Two fallback strategies (disabled when strict_cuda is True):
            # 1) If running on GPU, try CPU.
            # 2) If compute_type looks incompatible for the device (e.g., float16 on CPU),
            #    try a conservative CPU-friendly compute type.
            if device != "cpu" and not strict_cuda:
                logger.warning(
                    "Model load failed on %s (%s); retrying on CPU.",
                    device,
                    exc,
                )
                fallback_key = (model_name, "cpu", compute_type, cache_root, local_files_only)
                if fallback_key not in _MODEL_CACHE:
                    safe_compute = compute_type
                    if "float16" in compute_type:
                        safe_compute = "int8"
                    _MODEL_CACHE[fallback_key] = whisperx.load_model(
                        model_name,
                        device="cpu",
                        compute_type=safe_compute,
                        download_root=cache_root,
                        local_files_only=local_files_only,
                    )
                model = _MODEL_CACHE[fallback_key]
                _MODEL_CACHE[cache_key] = model
                return model
            # Already on CPU: attempt a conservative compute_type if not already using one
            cpu_safe_types = {"int8", "int8_float32", "int8_np"}
            if compute_type not in cpu_safe_types and not strict_cuda:
                cpu_key = (model_name, "cpu", "int8", cache_root, local_files_only)
                if cpu_key not in _MODEL_CACHE:
                    logger.info(
                        "Retrying model load on CPU with compute_type=int8 (was %s).",
                        compute_type,
                    )
                    _MODEL_CACHE[cpu_key] = whisperx.load_model(
                        model_name,
                        device="cpu",
                        compute_type="int8",
                        download_root=cache_root,
                        local_files_only=local_files_only,
                    )
                model = _MODEL_CACHE[cpu_key]
                _MODEL_CACHE[cache_key] = model
                return model
            # Give up
            raise
    return _MODEL_CACHE[cache_key]


def _get_align_model(language: str, device: str, cache_root: Optional[str]):
    import whisperx  # type: ignore
    cache_key = (language, device, cache_root)
    if cache_key not in _ALIGN_CACHE:
        _ALIGN_CACHE[cache_key] = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_dir=cache_root,
        )
    return _ALIGN_CACHE[cache_key]


def _get_diar_pipeline(device: str, hf_token: Optional[str]):
    """Return a diarization pipeline compatible with installed whisperx.

    Supports multiple API shapes across whisperx versions:
      - whisperx.DiarizationPipeline(use_auth_token=..., device=...)
      - whisperx.load_diarize_model(device=..., use_auth_token=...)
      - whisperx.load_diarization_model(device=..., use_auth_token=...)
    """
    import whisperx  # type: ignore
    cache_key = (device, hf_token)
    if cache_key in _DIAR_CACHE:
        return _DIAR_CACHE[cache_key]

    pipeline = None
    # 1) Newer/older direct class
    try:
        pipeline = getattr(whisperx, "DiarizationPipeline")(use_auth_token=hf_token, device=device)
    except AttributeError:
        pipeline = None
    except TypeError:
        # Some builds may not accept use_auth_token kwarg
        try:
            pipeline = getattr(whisperx, "DiarizationPipeline")(device=device)
        except Exception:
            pipeline = None

    # 2) Loader helpers under different names
    if pipeline is None:
        for fn_name in ("load_diarize_model", "load_diarization_model"):
            fn = getattr(whisperx, fn_name, None)
            if fn is None:
                continue
            try:
                pipeline = fn(device=device, use_auth_token=hf_token)
                break
            except TypeError:
                # Try minimal signature
                try:
                    pipeline = fn(device=device)
                    break
                except Exception:
                    continue

    if pipeline is None:
        raise RuntimeError("No diarization pipeline available in installed whisperx version")

    _DIAR_CACHE[cache_key] = pipeline
    return pipeline


def transcribe_with_whisperx(
    audio_path: str,
    model_name: str,
    compute_type: str,
    batch_size: int,
    hf_token: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    model_cache_dir: Optional[str],
    align_cache_dir: Optional[str],
    local_files_only: bool,
    vad_on_cpu: bool = False,
    pyannote_on_cpu: bool = False,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    quiet: bool = True,
    force_device: Optional[str] = None,
    strict_cuda: bool = False,
) -> Tuple[List[dict], Optional[List[dict]]]:
    """Run ASR + alignment + diarization via WhisperX for a single file."""
    # Lazy import here as well for direct function callers
    import whisperx  # type: ignore

    device = force_device if force_device in {"cpu", "cuda"} else _detect_device()
    # Elevate to WARNING so it appears in quiet/systemd logs
    try:
        import torch  # type: ignore
        cuda_ok = bool(torch.cuda.is_available())
        dev_count = int(torch.cuda.device_count()) if cuda_ok else 0
        cuda_ver = getattr(torch.version, "cuda", "?")
    except Exception:
        cuda_ok, dev_count, cuda_ver = False, 0, "?"
    logger.warning(
        "ASR Device: %s (torch cuda=%s, count=%s, cuda_ver=%s) compute=%s file=%s",
        device,
        cuda_ok,
        dev_count,
        cuda_ver,
        compute_type,
        audio_path,
    )
    with _silence_stdio(quiet):
        model = _get_model(
            model_name,
            device,
            compute_type,
            model_cache_dir,
            local_files_only,
            strict_cuda=strict_cuda,
        )

    # Attempt to force VAD to CPU if requested and supported by installed whisperx.
    # Adaptive batch size to avoid OOM while utilizing VRAM
    attempt_bs = max(1, int(batch_size or 1))
    while True:
        with _silence_stdio(quiet):
            try:
                if vad_on_cpu:
                    result = model.transcribe(  # type: ignore[call-arg]
                        audio_path, batch_size=attempt_bs, vad_on_cpu=True
                    )
                else:
                    result = model.transcribe(audio_path, batch_size=attempt_bs)
                break
            except TypeError:
                # Older whisperx versions may not support `vad_on_cpu`; fall back and warn.
                if vad_on_cpu:
                    logger.warning(
                        "Installed whisperx does not support `vad_on_cpu`; proceeding without it."
                    )
                result = model.transcribe(audio_path, batch_size=attempt_bs)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                is_oom = (
                    "out of memory" in msg
                    or "cuda error" in msg
                    or "failed to allocate" in msg
                    or "cublas_status_alloc_failed" in msg
                )
                if not is_oom or attempt_bs <= 1:
                    raise
                new_bs = max(1, attempt_bs // 2)
                logger.warning(
                    "Transcribe OOM at batch_size=%d; retrying with %d.", attempt_bs, new_bs
                )
                attempt_bs = new_bs
    segments = result.get("segments", [])
    language = result.get("language", "en")
    logger.info("Detected language=%s with %d base segment(s)", language, len(segments))
    if progress_cb:
        try:
            progress_cb("asr", 1, 3)
        except Exception:
            pass

    aligned_segments: List[dict] = segments
    try:
        with _silence_stdio(quiet):
            align_model, metadata = _get_align_model(language, device, align_cache_dir)
            aligned = whisperx.align(
                segments,
                align_model,
                metadata,
                audio_path,
                device=device,
            )
            aligned_segments = aligned.get("segments", segments) or segments
    except Exception as exc:  # noqa: BLE001 - offline / cache miss
        logger.warning("Alignment failed for %s: %s", audio_path, exc)
    finally:
        if progress_cb:
            try:
                progress_cb("align", 2, 3)
            except Exception:
                pass

    diar_segments: Optional[List[dict]] = None
    try:
        diar_device = "cpu" if pyannote_on_cpu else device
        if diar_device == "cuda" and not _cudnn_usable():
            if strict_cuda:
                raise RuntimeError(
                    "cuDNN unavailable: strict CUDA mode forbids CPU diarization fallback"
                )
            logger.warning(
                "cuDNN not available or failed to load; falling back to CPU for diarization."
            )
            diar_device = "cpu"
        with _silence_stdio(quiet):
            diar_pipeline = _get_diar_pipeline(diar_device, hf_token)
            diar_result = diar_pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diar_segments = diar_result.get("segments", [])
            try:
                speaker_aligned = whisperx.assign_word_speakers(
                    diar_result,
                    {"segments": aligned_segments},
                )
            except TypeError:
                # Some versions expect the diarization model/pipeline as a third arg
                speaker_aligned = whisperx.assign_word_speakers(
                    diar_result,
                    {"segments": aligned_segments},
                    diar_pipeline,
                )
            aligned_segments = speaker_aligned.get("segments", aligned_segments) or aligned_segments
    except Exception as exc:  # noqa: BLE001 - best effort diarization
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
    finally:
        if progress_cb:
            try:
                progress_cb("diar", 3, 3)
            except Exception:
                pass

    structured_segments: List[dict] = []
    for seg in aligned_segments:
        structured_segments.append(
            {
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker"),
            }
        )

    logger.info("Produced %d aligned segment(s) for %s", len(structured_segments), audio_path)
    return structured_segments, diar_segments
