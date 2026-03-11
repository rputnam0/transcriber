from __future__ import annotations

import gc
import importlib.metadata
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .audio import load_audio_mono

logger = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

_PIPELINE_CACHE: Dict[Tuple[str, Optional[str], str], object] = {}
_EMBEDDER_CACHE: Dict[Tuple[str, Optional[str], str], object] = {}


@dataclass(frozen=True)
class DiarizationTurn:
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class DiarizationResult:
    segments: List[DiarizationTurn]
    exclusive_segments: List[DiarizationTurn]
    metadata: dict


@dataclass
class SegmentEmbeddingResult:
    speaker: str
    start: float
    end: float
    index: int
    embedding: np.ndarray


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _cudnn_usable() -> bool:
    try:
        import torch

        return bool(torch.backends.cudnn.is_available())
    except Exception:
        return False


def _pyannote_version() -> str:
    try:
        return importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _annotation_to_segments(annotation: object) -> List[DiarizationTurn]:
    if annotation is None:
        return []
    if isinstance(annotation, list):
        turns: List[DiarizationTurn] = []
        for item in annotation:
            if not isinstance(item, dict):
                continue
            turns.append(
                DiarizationTurn(
                    start=float(item.get("start") or 0.0),
                    end=float(item.get("end") or 0.0),
                    speaker=str(item.get("speaker") or item.get("label") or "unknown"),
                )
            )
        turns.sort(key=lambda item: (item.start, item.end, item.speaker))
        return turns

    if isinstance(annotation, dict):
        if "speaker_diarization" in annotation:
            return _annotation_to_segments(annotation.get("speaker_diarization"))
        if "exclusive_speaker_diarization" in annotation:
            return _annotation_to_segments(annotation.get("exclusive_speaker_diarization"))
        if "segments" in annotation:
            return _annotation_to_segments(annotation.get("segments"))

    speaker_diarization = getattr(annotation, "speaker_diarization", None)
    if speaker_diarization is not None:
        return _annotation_to_segments(speaker_diarization)

    if hasattr(annotation, "itertracks"):
        turns = []
        for segment, _track, label in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    start=float(getattr(segment, "start", 0.0) or 0.0),
                    end=float(getattr(segment, "end", 0.0) or 0.0),
                    speaker=str(label or "unknown"),
                )
            )
        turns.sort(key=lambda item: (item.start, item.end, item.speaker))
        return turns

    return []


def _load_pipeline(model_name: str, *, device: str, hf_token: Optional[str]) -> object:
    from pyannote.audio import Pipeline

    cache_key = (device, hf_token, model_name)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    attempts = []
    if hf_token:
        attempts.append({"token": hf_token})
        attempts.append({"use_auth_token": hf_token})
    attempts.append({})

    pipeline = None
    last_error: Optional[BaseException] = None
    for kwargs in attempts:
        try:
            pipeline = Pipeline.from_pretrained(model_name, **kwargs)
            if pipeline is not None:
                break
        except TypeError as exc:
            last_error = exc
            if "unexpected keyword argument 'plda'" in str(exc):
                raise RuntimeError(
                    f"{model_name} requires a newer pyannote.audio than {_pyannote_version()}. "
                    "Upgrade pyannote.audio to a Community-1 compatible version."
                ) from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if pipeline is None:
        raise RuntimeError(f"Failed to load diarization model {model_name}: {last_error}")

    try:
        import torch

        torch_device = torch.device(device)
    except Exception:
        torch_device = device

    try:
        pipeline = pipeline.to(torch_device)
    except Exception:
        pass

    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def diarize_audio(
    audio_path: str,
    *,
    model_name: Optional[str] = None,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[str] = None,
) -> DiarizationResult:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("torch is required for diarization") from exc

    diar_model_name = model_name or DEFAULT_DIARIZATION_MODEL
    diar_device = device or _detect_device()
    if diar_device == "cuda" and not _cudnn_usable():
        diar_device = "cpu"

    pipeline = _load_pipeline(diar_model_name, device=diar_device, hf_token=hf_token)
    call_kwargs = {}
    if min_speakers is not None and max_speakers is not None and min_speakers == max_speakers:
        call_kwargs["num_speakers"] = int(min_speakers)
    else:
        if min_speakers is not None:
            call_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            call_kwargs["max_speakers"] = int(max_speakers)

    waveform = load_audio_mono(audio_path, sample_rate=16000)
    audio_input = {
        "waveform": torch.from_numpy(np.array(waveform, dtype=np.float32, copy=True)).unsqueeze(0),
        "sample_rate": 16000,
    }

    result = pipeline(audio_input, **call_kwargs)
    exclusive = getattr(result, "exclusive_speaker_diarization", None)
    if exclusive is None and isinstance(result, dict):
        exclusive = result.get("exclusive_speaker_diarization")

    segments = _annotation_to_segments(result)
    exclusive_segments = _annotation_to_segments(exclusive) or list(segments)

    return DiarizationResult(
        segments=segments,
        exclusive_segments=exclusive_segments,
        metadata={
            "model_name": diar_model_name,
            "device": diar_device,
            "num_speakers": call_kwargs.get("num_speakers"),
            "min_speakers": call_kwargs.get("min_speakers"),
            "max_speakers": call_kwargs.get("max_speakers"),
        },
    )


def _resolve_embedder(
    *,
    model_name: str,
    hf_token: Optional[str],
    device: str,
) -> object:
    cache_key = (device, hf_token, model_name)
    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]

    pipeline = _load_pipeline(model_name, device=device, hf_token=hf_token)
    embedder = None
    attr_candidates = ("_embedding", "embedding_model")
    for attr in attr_candidates:
        embedder = getattr(pipeline, attr, None)
        if embedder is not None:
            break
    if embedder is None and hasattr(pipeline, "model"):
        for attr in attr_candidates:
            embedder = getattr(pipeline.model, attr, None)
            if embedder is not None:
                break

    if embedder is None:
        try:
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError("Unable to import PretrainedSpeakerEmbedding") from exc

        embedding_name = getattr(pipeline, "embedding", None)
        if embedding_name is None and hasattr(pipeline, "model"):
            embedding_name = getattr(pipeline.model, "embedding", None)
        if embedding_name is None:
            raise RuntimeError("Unable to resolve embedding model from diarization pipeline")

        try:
            import torch

            torch_device = torch.device(device)
        except Exception:
            torch_device = device

        init_attempts = []
        if hf_token:
            init_attempts.append({"use_auth_token": hf_token})
            init_attempts.append({"token": hf_token})
        init_attempts.append({})

        last_error: Optional[BaseException] = None
        for kwargs in init_attempts:
            try:
                embedder = PretrainedSpeakerEmbedding(
                    embedding_name,
                    device=torch_device,
                    **kwargs,
                )
                break
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if embedder is None:
            raise RuntimeError(f"Unable to instantiate speaker embedder: {last_error}")

    try:
        import torch

        embedder = embedder.to(torch.device(device))  # type: ignore[attr-defined]
    except Exception:
        pass

    _EMBEDDER_CACHE[cache_key] = embedder
    return embedder


def release_runtime_caches() -> None:
    _PIPELINE_CACHE.clear()
    _EMBEDDER_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _embedding_batch_to_numpy(embedding_batch: object) -> np.ndarray:
    if isinstance(embedding_batch, np.ndarray):
        return embedding_batch

    if (
        hasattr(embedding_batch, "detach")
        and hasattr(embedding_batch, "cpu")
        and hasattr(embedding_batch, "numpy")
    ):
        try:
            return np.asarray(embedding_batch.detach().cpu().numpy(), dtype=np.float32)
        except Exception:
            pass

    return np.asarray(embedding_batch)


def extract_embeddings_for_segments(
    audio_path: str,
    segments: Iterable[Tuple[float, float, str]],
    hf_token: Optional[str],
    *,
    diarization_model_name: Optional[str] = None,
    force_device: Optional[str] = None,
    quiet: bool = True,
    pre_pad: float = 0.15,
    post_pad: float = 0.15,
    batch_size: int = 16,
    workers: int = 4,
    waveform_transform: Optional[Callable[[np.ndarray, int, str, int], np.ndarray]] = None,
) -> Tuple[List[SegmentEmbeddingResult], Dict[str, object]]:
    del quiet  # retained for API compatibility
    del workers  # cropping is in-memory now

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("torch is required for speaker embedding extraction") from exc

    if batch_size <= 0:
        batch_size = 1

    device = force_device if force_device in {"cpu", "cuda"} else _detect_device()
    if device == "cuda" and not _cudnn_usable():
        device = "cpu"
    diar_model_name = diarization_model_name or DEFAULT_DIARIZATION_MODEL
    embedder = _resolve_embedder(model_name=diar_model_name, hf_token=hf_token, device=device)
    sample_rate = int(getattr(embedder, "sample_rate", 16000) or 16000)
    base_waveform = load_audio_mono(audio_path, sample_rate=16000)
    base_sr = 16000

    if sample_rate != base_sr:
        try:
            import torchaudio.functional as AF  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "torchaudio is required when resampling speaker-embedding audio"
            ) from exc
        waveform_t = torch.from_numpy(base_waveform).unsqueeze(0)
        resampled = AF.resample(waveform_t, base_sr, sample_rate)
        base_waveform = resampled.squeeze(0).numpy()
        base_sr = sample_rate

    total_samples = base_waveform.shape[0]
    audio_duration = total_samples / float(base_sr)

    segment_items = [
        (index, float(start or 0.0), float(end or 0.0), str(label or ""))
        for index, (start, end, label) in enumerate(list(segments))
    ]
    if not segment_items:
        return [], {"embedded": 0, "skipped": 0, "total": 0}

    cropped: List[Tuple[int, float, float, str, np.ndarray]] = []
    for index, start, end, speaker in segment_items:
        if end <= start:
            continue
        crop_start = max(start - pre_pad, 0.0)
        crop_end = min(end + post_pad, audio_duration)
        if crop_end <= crop_start:
            continue
        start_idx = max(0, int(math.floor(crop_start * base_sr)))
        end_idx = min(total_samples, int(math.ceil(crop_end * base_sr)))
        if end_idx <= start_idx:
            continue
        wave = np.asarray(base_waveform[start_idx:end_idx], dtype=np.float32)
        if wave.size == 0:
            continue
        if waveform_transform is not None:
            wave = np.asarray(
                waveform_transform(wave.copy(), int(base_sr), speaker, index),
                dtype=np.float32,
            ).flatten()
            if wave.size == 0:
                continue
        cropped.append((index, start, end, speaker, wave))

    if not cropped:
        return [], {"embedded": 0, "skipped": len(segment_items), "total": len(segment_items)}

    try:
        torch_device = torch.device(device)
    except Exception:
        torch_device = device

    results: List[SegmentEmbeddingResult] = []
    skipped = len(segment_items) - len(cropped)
    batch: List[Tuple[int, float, float, str, np.ndarray]] = []

    def _flush_batch() -> None:
        nonlocal skipped
        if not batch:
            return
        lengths = [wave.shape[0] for *_meta, wave in batch]
        max_len = max(lengths)
        if max_len <= 0:
            batch.clear()
            return
        wave_batch = torch.zeros((len(batch), 1, max_len), dtype=torch.float32, device=torch_device)
        mask_batch = torch.zeros((len(batch), max_len), dtype=torch.float32, device=torch_device)
        for row_idx, (_index, _start, _end, _speaker, wave) in enumerate(batch):
            tensor = torch.from_numpy(wave.astype(np.float32))
            wave_batch[row_idx, 0, : tensor.shape[0]] = tensor
            mask_batch[row_idx, : tensor.shape[0]] = 1.0

        with torch.no_grad():
            try:
                embedding_batch = embedder(wave_batch, masks=mask_batch)
            except TypeError:
                embedding_batch = embedder(wave_batch)
        embedding_batch = _embedding_batch_to_numpy(embedding_batch)

        for (index, start, end, speaker, _wave), vec in zip(batch, embedding_batch):
            vector = np.asarray(vec, dtype=np.float32).flatten()
            if vector.size == 0 or not np.isfinite(vector).all():
                skipped += 1
                continue
            norm = np.linalg.norm(vector)
            if norm == 0.0:
                skipped += 1
                continue
            results.append(
                SegmentEmbeddingResult(
                    speaker=speaker,
                    start=start,
                    end=end,
                    index=index,
                    embedding=(vector / norm).astype(np.float32),
                )
            )
        batch.clear()

    for item in cropped:
        batch.append(item)
        if len(batch) >= batch_size:
            _flush_batch()
    _flush_batch()

    results.sort(key=lambda item: item.index)
    return results, {
        "embedded": len(results),
        "skipped": skipped,
        "total": len(segment_items),
        "device": device,
        "sample_rate": sample_rate,
    }


def _merge_adjacent_segments(
    segments: List[DiarizationTurn],
    *,
    max_gap_seconds: float = 0.20,
    min_duration: float = 0.80,
) -> List[DiarizationTurn]:
    if not segments:
        return []
    merged: List[DiarizationTurn] = []
    current = segments[0]
    for item in segments[1:]:
        if item.speaker == current.speaker and item.start - current.end <= max_gap_seconds:
            current = DiarizationTurn(start=current.start, end=item.end, speaker=current.speaker)
            continue
        if current.end - current.start >= min_duration:
            merged.append(current)
        current = item
    if current.end - current.start >= min_duration:
        merged.append(current)
    return merged


def extract_speaker_embeddings(
    audio_path: str,
    hf_token: Optional[str],
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    pyannote_on_cpu: bool = False,
    diarization_model_name: Optional[str] = None,
    force_device: Optional[str] = None,
    quiet: bool = True,
) -> Tuple[Dict[str, np.ndarray], Optional[List[dict]]]:
    device = force_device if force_device in {"cpu", "cuda"} else _detect_device()
    if pyannote_on_cpu:
        device = "cpu"
    diarization = diarize_audio(
        audio_path,
        model_name=diarization_model_name,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
    )
    segments = _merge_adjacent_segments(diarization.exclusive_segments or diarization.segments)
    payload = [(segment.start, segment.end, segment.speaker) for segment in segments]
    embed_results, _summary = extract_embeddings_for_segments(
        audio_path,
        payload,
        hf_token,
        diarization_model_name=diarization_model_name,
        force_device=device,
        quiet=quiet,
    )
    by_speaker: Dict[str, List[np.ndarray]] = {}
    for result in embed_results:
        by_speaker.setdefault(result.speaker, []).append(result.embedding)

    embeddings: Dict[str, np.ndarray] = {}
    for speaker, vectors in by_speaker.items():
        matrix = np.stack(vectors)
        centroid = np.mean(matrix, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            continue
        embeddings[speaker] = (centroid / norm).astype(np.float32)

    return embeddings, [turn.to_dict() for turn in diarization.segments]
