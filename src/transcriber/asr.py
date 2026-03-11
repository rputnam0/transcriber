from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[Tuple[str, str, str, Optional[str], bool], WhisperModel] = {}


@dataclass(frozen=True)
class AsrWord:
    word: str
    start: float
    end: float
    score: Optional[float] = None

    def to_dict(self) -> dict:
        payload = {
            "word": self.word,
            "start": float(self.start),
            "end": float(self.end),
        }
        if self.score is not None:
            payload["score"] = float(self.score)
        return payload


@dataclass(frozen=True)
class AsrSegment:
    start: float
    end: float
    text: str
    words: List[AsrWord] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "start": float(self.start),
            "end": float(self.end),
            "text": self.text.strip(),
            "speaker": None,
        }
        if self.words:
            payload["words"] = [word.to_dict() for word in self.words]
        return payload


@dataclass(frozen=True)
class AsrResult:
    segments: List[AsrSegment]
    language: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict_segments(self) -> List[dict]:
        return [segment.to_dict() for segment in self.segments]


class AsrEngine(Protocol):
    def transcribe(self, path: str, *, batch_size: int = 32) -> AsrResult: ...


def load_faster_whisper_model(
    model_name: str,
    *,
    compute_type: str = "float16",
    device: str = "cuda",
    download_root: str | None = None,
    local_files_only: bool = False,
) -> WhisperModel:
    cache_key = (model_name, device, compute_type, download_root, local_files_only)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
        )
    return _MODEL_CACHE[cache_key]


class FasterWhisperEngine:
    def __init__(
        self,
        model_name: str,
        *,
        compute_type: str = "float16",
        device: str = "cuda",
        download_root: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = str(model_name)
        self.compute_type = str(compute_type)
        self.device = str(device)
        self.download_root = download_root
        self.local_files_only = bool(local_files_only)
        self.model = load_faster_whisper_model(
            self.model_name,
            compute_type=self.compute_type,
            device=self.device,
            download_root=self.download_root,
            local_files_only=self.local_files_only,
        )

    def transcribe(self, path: str, *, batch_size: int = 32) -> AsrResult:
        attempt_bs = max(1, int(batch_size or 1))
        while True:
            try:
                segments_iter, info = self.model.transcribe(
                    path,
                    batch_size=attempt_bs,
                    vad_filter=True,
                    word_timestamps=True,
                )
                break
            except TypeError:
                segments_iter, info = self.model.transcribe(
                    path,
                    vad_filter=True,
                    word_timestamps=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                is_oom = (
                    "out of memory" in msg
                    or "failed to allocate" in msg
                    or "cublas_status_alloc_failed" in msg
                    or "cuda error" in msg
                )
                if not is_oom or attempt_bs <= 1:
                    raise
                new_bs = max(1, attempt_bs // 2)
                logger.warning(
                    "Faster-whisper OOM at batch_size=%d; retrying with %d", attempt_bs, new_bs
                )
                attempt_bs = new_bs

        segments: List[AsrSegment] = []
        for segment in segments_iter:
            words: List[AsrWord] = []
            raw_words = getattr(segment, "words", None) or []
            for word in raw_words:
                text = str(getattr(word, "word", "") or "").strip()
                if not text:
                    continue
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                if start is None or end is None:
                    continue
                score = getattr(word, "probability", None)
                if score is None:
                    score = getattr(word, "score", None)
                words.append(
                    AsrWord(
                        word=text,
                        start=float(start),
                        end=float(end),
                        score=(float(score) if score is not None else None),
                    )
                )
            segments.append(
                AsrSegment(
                    start=float(getattr(segment, "start", 0.0) or 0.0),
                    end=float(getattr(segment, "end", 0.0) or 0.0),
                    text=str(getattr(segment, "text", "") or "").strip(),
                    words=words,
                )
            )

        language = getattr(info, "language", None)
        return AsrResult(
            segments=segments,
            language=(str(language) if language is not None else None),
            metadata={"batch_size": attempt_bs},
        )


def transcribe_with_faster_whisper(
    path: str,
    *,
    model_name: str,
    compute_type: str = "float16",
    device: str = "cuda",
    download_root: str | None = None,
    local_files_only: bool = False,
    batch_size: int = 32,
) -> AsrResult:
    engine = FasterWhisperEngine(
        model_name,
        compute_type=compute_type,
        device=device,
        download_root=download_root,
        local_files_only=local_files_only,
    )
    return engine.transcribe(path, batch_size=batch_size)
