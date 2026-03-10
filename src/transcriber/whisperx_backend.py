from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple, Callable
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
import concurrent.futures
import math

import numpy as np

from dataclasses import dataclass

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
      - whisperx.diarize.DiarizationPipeline(...)
    """
    import whisperx  # type: ignore
    cache_key = (device, hf_token)
    if cache_key in _DIAR_CACHE:
        return _DIAR_CACHE[cache_key]

    def _instantiate(ctor):
        """Try constructing a pipeline with progressively simpler signatures."""
        if ctor is None:
            return None
        attempts = []
        if hf_token:
            attempts.append({"use_auth_token": hf_token, "device": device})
        attempts.append({"device": device})
        attempts.append({})
        for kwargs in attempts:
            try:
                return ctor(**kwargs)
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialise diarization pipeline %s: %s", ctor, exc)
        return None

    pipeline = None
    # 1) Newer/older direct class on the top-level package
    pipeline = _instantiate(getattr(whisperx, "DiarizationPipeline", None))

    # 2) Module-scoped class (whisperx.diarize.DiarizationPipeline) introduced in newer releases
    if pipeline is None:
        try:
            from whisperx import diarize as _wx_diarize  # type: ignore

            pipeline = _instantiate(getattr(_wx_diarize, "DiarizationPipeline", None))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to import whisperx.diarize.DiarizationPipeline: %s", exc)

    # 3) Loader helpers under different names
    if pipeline is None:
        for fn_name in ("load_diarize_model", "load_diarization_model"):
            fn = getattr(whisperx, fn_name, None)
            if fn is None:
                # Some builds expose the helpers from whisperx.diarize instead
                try:
                    from whisperx import diarize as _wx_diarize  # type: ignore

                    fn = getattr(_wx_diarize, fn_name, None)
                except Exception:  # noqa: BLE001
                    fn = None
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


def _normalize_embeddings(payload) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    if isinstance(payload, dict):
        logger.debug(
            "Normalizing %d speaker embedding(s) from payload keys=%s",
            len(payload),
            list(payload.keys()),
        )
        for key, value in payload.items():
            vec = np.asarray(value, dtype=np.float32).flatten()
            if vec.size == 0:
                logger.debug("Skipping empty embedding for label=%s", key)
                continue
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                logger.debug("Skipping zero-norm embedding for label=%s", key)
                continue
            embeddings[str(key)] = (vec / norm).astype(np.float32)
            logger.debug(
                "Normalised embedding label=%s dims=%d original_norm=%.4f",
                key,
                vec.size,
                norm,
            )
    else:
        logger.debug("Embedding payload type=%s is not a dict; skipping normalization.", type(payload).__name__)
    return embeddings


@dataclass
class SegmentDescriptor:
    start: float
    end: float
    speaker: str
    index: int


@dataclass
class SegmentEmbeddingResult:
    speaker: str
    start: float
    end: float
    index: int
    embedding: np.ndarray


def extract_embeddings_for_segments(
    audio_path: str,
    segments: Iterable[Tuple[float, float, str]],
    hf_token: Optional[str],
    *,
    force_device: Optional[str] = None,
    quiet: bool = True,
    pre_pad: float = 0.15,
    post_pad: float = 0.15,
    batch_size: int = 16,
    workers: int = 4,
) -> Tuple[List[SegmentEmbeddingResult], Dict[str, object]]:
    """Extract embeddings for pre-labelled segments from an audio file.

    Returns (embeddings, summary) where embeddings is a list of SegmentEmbeddingResult
    and summary contains diagnostic counts.
    """

    from whisperx.audio import load_audio, SAMPLE_RATE as WX_SAMPLE_RATE  # type: ignore

    import torch

    if batch_size <= 0:
        batch_size = 1
    if workers <= 0:
        workers = 1

    device = force_device if force_device in {"cpu", "cuda"} else _detect_device()
    torch_device = torch.device(device)
    segment_items = list(segments)
    logger.debug(
        "Segment embedding extraction start file=%s device=%s segments=%d",
        audio_path,
        device,
        len(segment_items),
    )

    # Materialize segments once since we need multiple passes
    seg_list: List[SegmentDescriptor] = [
        SegmentDescriptor(start=float(s or 0.0), end=float(e or 0.0), speaker=str(label or ""), index=idx)
        for idx, (s, e, label) in enumerate(segment_items)
    ]
    if not seg_list:
        return [], {"embedded": 0, "skipped": 0, "total": 0}

    diar_pipeline = _get_diar_pipeline(device, hf_token)

    # Resolve embedder from pipeline (supports whisperx wrappers and pyannote pipeline)
    embedder = None
    attr_candidates = ("_embedding", "embedding_model")
    for attr in attr_candidates:
        embedder = getattr(diar_pipeline, attr, None)
        if embedder is not None:
            break
    if embedder is None and hasattr(diar_pipeline, "model"):
        for attr in attr_candidates:
            embedder = getattr(diar_pipeline.model, attr, None)
            if embedder is not None:
                break

    if embedder is None:
        try:
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )

            embedding_name = getattr(diar_pipeline, "embedding", None)
            if embedding_name is None and hasattr(diar_pipeline, "model"):
                embedding_name = getattr(diar_pipeline.model, "embedding", None)
            if embedding_name:
                embedder = PretrainedSpeakerEmbedding(
                    embedding_name,
                    device=torch_device,
                    use_auth_token=hf_token,
                )
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("Failed to instantiate standalone speaker embedder: %s", exc)
            embedder = None

    if embedder is None:
        raise RuntimeError("Unable to resolve speaker embedding model from diarization pipeline")

    try:
        embedder = embedder.to(torch_device)  # type: ignore[attr-defined]
    except Exception:
        # embedder may not support to(); ignore
        pass

    sample_rate = getattr(embedder, "sample_rate", WX_SAMPLE_RATE)

    base_waveform = load_audio(audio_path)
    base_waveform = np.asarray(base_waveform, dtype=np.float32)
    base_sr = WX_SAMPLE_RATE

    if sample_rate != base_sr:
        try:
            import torchaudio.functional as AF  # type: ignore

            waveform_t = torch.from_numpy(base_waveform).unsqueeze(0)
            resampled = AF.resample(waveform_t, base_sr, sample_rate)
            base_waveform = resampled.squeeze(0).numpy()
            base_sr = sample_rate
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "Failed to resample audio from %s Hz to %s Hz: %s (continuing with original rate)",
                base_sr,
                sample_rate,
                exc,
            )
            sample_rate = base_sr

    audio_total_samples = base_waveform.shape[0]
    audio_duration = audio_total_samples / float(base_sr)

    # Precompute crops (optionally in parallel)
    def _crop_segment(seg: SegmentDescriptor) -> Optional[Tuple[SegmentDescriptor, np.ndarray]]:
        if seg.end <= seg.start:
            return None
        start = max(seg.start - pre_pad, 0.0)
        end = min(seg.end + post_pad, audio_duration)
        if end <= start:
            return None
        start_idx = int(math.floor(start * base_sr))
        end_idx = int(math.ceil(end * base_sr))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, audio_total_samples)
        if end_idx <= start_idx:
            return None
        segment_wave = base_waveform[start_idx:end_idx]
        if segment_wave.size == 0:
            return None
        return seg, segment_wave

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            cropped = list(filter(None, pool.map(_crop_segment, seg_list)))
    else:
        cropped = []
        for seg in seg_list:
            item = _crop_segment(seg)
            if item is not None:
                cropped.append(item)

    if not cropped:
        logger.warning("No valid segments produced for %s", audio_path)
        return [], {"embedded": 0, "skipped": len(seg_list), "total": len(seg_list)}

    cropped.sort(key=lambda item: item[0].index)

    embedded_results: List[SegmentEmbeddingResult] = []
    skipped = len(seg_list) - len(cropped)

    batch: List[Tuple[SegmentDescriptor, np.ndarray]] = []

    def _flush_batch():
        nonlocal embedded_results, skipped
        if not batch:
            return
        lengths = [wave.shape[0] for _, wave in batch]
        max_len = max(lengths)
        if max_len == 0:
            batch.clear()
            return
        import torch

        wave_batch = torch.zeros((len(batch), 1, max_len), dtype=torch.float32, device=torch_device)
        mask_batch = torch.zeros((len(batch), max_len), dtype=torch.float32, device=torch_device)
        for i, (seg, wave) in enumerate(batch):
            tensor = torch.from_numpy(wave.astype(np.float32))
            wave_batch[i, 0, : tensor.shape[0]] = tensor
            mask_batch[i, : tensor.shape[0]] = 1.0

        with torch.no_grad():
            try:
                embedding_batch = embedder(wave_batch, masks=mask_batch)
            except TypeError:
                # Some embedders might not accept masks
                embedding_batch = embedder(wave_batch)

        if not isinstance(embedding_batch, np.ndarray):
            embedding_batch = np.asarray(embedding_batch)

        for (seg, _), vec in zip(batch, embedding_batch):
            vec = np.asarray(vec, dtype=np.float32).flatten()
            if not np.isfinite(vec).all():
                skipped += 1
                continue
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                skipped += 1
                continue
            vec = vec / norm
            embedded_results.append(
                SegmentEmbeddingResult(
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                    index=seg.index,
                    embedding=vec.astype(np.float32),
                )
            )
        batch.clear()

    for item in cropped:
        batch.append(item)
        if len(batch) >= batch_size:
            _flush_batch()
    _flush_batch()

    embedded_results.sort(key=lambda entry: entry.index)

    summary = {
        "embedded": len(embedded_results),
        "skipped": skipped,
        "total": len(seg_list),
        "device": device,
        "sample_rate": sample_rate,
    }
    logger.debug(
        "Segment embedding extraction complete file=%s embedded=%d skipped=%d",
        audio_path,
        summary["embedded"],
        summary["skipped"],
    )

    return embedded_results, summary


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
    enable_diarization: bool = True,
) -> Tuple[List[dict], Optional[List[dict]], Dict[str, np.ndarray]]:
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
    speaker_embedding_map: Dict[str, np.ndarray] = {}
    try:
        if not enable_diarization:
            logger.info("Diarization disabled by configuration; skipping for %s", audio_path)
        else:
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
            diar_result = None
            diar_kwargs = {
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
            logger.debug(
                "Invoking diarization pipeline device=%s kwargs=%s file=%s",
                diar_device,
                diar_kwargs,
                audio_path,
            )
            with _silence_stdio(quiet):
                diar_pipeline = _get_diar_pipeline(diar_device, hf_token)
                try:
                    diar_result = diar_pipeline(
                        audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        return_embeddings=True,
                    )
                except TypeError:
                    diar_result = diar_pipeline(
                        audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )

            def _is_dataframe(obj) -> bool:
                try:
                    import pandas as pd  # type: ignore

                    return isinstance(obj, pd.DataFrame)
                except Exception:
                    return False

            def _segments_from_dataframe(df) -> List[dict]:
                try:
                    records = df.to_dict(orient="records")
                except Exception:
                    return []
                serialised: List[dict] = []
                for row in records:
                    start = row.get("start")
                    end = row.get("end")
                    if (start is None or end is None) and "segment" in row:
                        seg_obj = row.get("segment")
                        start = getattr(seg_obj, "start", start)
                        end = getattr(seg_obj, "end", end)
                    serialised.append(
                        {
                            "start": float(start or 0.0),
                            "end": float(end or 0.0),
                            "speaker": row.get("speaker") or row.get("label"),
                        }
                    )
                return serialised

            assign_source = diar_result
            speaker_embeddings = None
            if isinstance(diar_result, tuple):
                if diar_result:
                    assign_source = diar_result[0]
                logger.debug(
                    "Diarization result tuple(len=%d) types=%s for %s",
                    len(diar_result),
                    tuple(type(item).__name__ for item in diar_result),
                    audio_path,
                )
                if len(diar_result) > 1 and isinstance(diar_result[1], dict):
                    speaker_embeddings = diar_result[1]
                    speaker_embedding_map = _normalize_embeddings(speaker_embeddings)
                    logger.debug(
                        "Diarization returned %d normalized embedding(s) for %s labels=%s",
                        len(speaker_embedding_map),
                        audio_path,
                        list(speaker_embedding_map.keys()),
                    )

            if _is_dataframe(assign_source):
                diar_segments = _segments_from_dataframe(assign_source)
            elif isinstance(diar_result, dict):
                diar_segments = list(diar_result.get("segments") or [])
            elif isinstance(diar_result, list):
                diar_segments = list(diar_result)
            else:
                diar_segments = []

            assign_candidates = []
            if _is_dataframe(assign_source):
                if speaker_embeddings is not None:
                    assign_candidates.append(
                        (assign_source, {"segments": aligned_segments}, speaker_embeddings)
                    )
                assign_candidates.append((assign_source, {"segments": aligned_segments}))
            assign_candidates.append(
                (diar_result, {"segments": aligned_segments}, diar_pipeline)
            )
            assign_candidates.append((diar_result, {"segments": aligned_segments}))
            logger.debug(
                "Prepared %d diarization assignment candidate(s) for %s",
                len(assign_candidates),
                audio_path,
            )

            speaker_aligned = None
            for args in assign_candidates:
                try:
                    speaker_aligned = whisperx.assign_word_speakers(*args)
                    break
                except TypeError:
                    continue
            if speaker_aligned:
                aligned_segments = (
                    speaker_aligned.get("segments", aligned_segments) or aligned_segments
                )
                logger.debug(
                    "Applied diarization word-speaker alignment for %s via payload type=%s",
                    audio_path,
                    type(speaker_aligned).__name__,
                )
            else:
                logger.debug(
                    "Word-speaker alignment did not modify segments for %s",
                    audio_path,
                )
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
    logger.debug(
        "Returning from transcribe_with_whisperx file=%s segments=%d diar_segments=%d embeddings=%d",
        audio_path,
        len(structured_segments),
        len(diar_segments or []),
        len(speaker_embedding_map),
    )
    return structured_segments, diar_segments, speaker_embedding_map


def extract_speaker_embeddings(
    audio_path: str,
    hf_token: Optional[str],
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    pyannote_on_cpu: bool = False,
    force_device: Optional[str] = None,
    quiet: bool = True,
) -> Tuple[Dict[str, np.ndarray], Optional[List[dict]]]:
    """Return normalised speaker embeddings (cosine space) for an audio file."""
    diar_device = force_device if force_device in {"cpu", "cuda"} else _detect_device()
    if pyannote_on_cpu:
        diar_device = "cpu"
    if diar_device == "cuda" and not _cudnn_usable():
        diar_device = "cpu"
    logger = logging.getLogger(__name__)
    logger.warning(
        "Speaker embedding extraction: device=%s min=%s max=%s file=%s",
        diar_device,
        min_speakers,
        max_speakers,
        audio_path,
    )
    diar_segments: Optional[List[dict]] = None
    embeddings: Dict[str, np.ndarray] = {}
    with _silence_stdio(quiet):
        diar_pipeline = _get_diar_pipeline(diar_device, hf_token)
        try:
            diar_result = diar_pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )
        except TypeError:
            diar_result = diar_pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
    def _is_dataframe(obj) -> bool:
        try:
            import pandas as pd  # type: ignore

            return isinstance(obj, pd.DataFrame)
        except Exception:
            return False

    def _segments_from_dataframe(df) -> List[dict]:
        try:
            records = df.to_dict(orient="records")
        except Exception:
            return []
        serialised: List[dict] = []
        for row in records:
            start = row.get("start")
            end = row.get("end")
            if (start is None or end is None) and "segment" in row:
                seg_obj = row.get("segment")
                start = getattr(seg_obj, "start", start)
                end = getattr(seg_obj, "end", end)
            serialised.append(
                {
                    "start": float(start or 0.0),
                    "end": float(end or 0.0),
                    "speaker": row.get("speaker") or row.get("label"),
                }
            )
        return serialised

    assign_source = diar_result
    if isinstance(diar_result, tuple):
        if diar_result:
            assign_source = diar_result[0]
        if len(diar_result) > 1 and isinstance(diar_result[1], dict):
            embeddings = _normalize_embeddings(diar_result[1])
            logger.info(
                "Speaker embedding extraction: received %d embedding(s) from pipeline",
                len(embeddings),
            )
    if _is_dataframe(assign_source):
        diar_segments = _segments_from_dataframe(assign_source)
    elif isinstance(assign_source, dict):
        diar_segments = list(assign_source.get("segments") or [])
    elif isinstance(assign_source, list):
        diar_segments = list(assign_source)
    if not embeddings:
        logger.warning("Speaker embedding extraction returned 0 embeddings for %s", audio_path)
    return embeddings, diar_segments
