"""Local large-v3 recognition for isolated tracks on Apple Silicon.

Decode bounded windows with real context. Timed words belong to exactly one core
window; sentence splitting remains the exporter's responsibility. Only digital
silence is skipped, so the guard does not impose a VAD threshold on quiet speech.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import logging
import re
from pathlib import Path
import time

import numpy as np

from .audio import load_audio_mono

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
SAMPLE_RATE = 16000
CORE_SECONDS = 60
CONTEXT_SECONDS = 3
logger = logging.getLogger(__name__)


def resolve_model_name(name: str | None) -> str:
    if name in {None, "large-v3", "whisper", "whisper-large-v3", DEFAULT_MODEL}:
        return DEFAULT_MODEL
    if name == "parakeet":
        raise ValueError("Use --backend parakeet for a Parakeet model")
    return str(name)


def owned_segments(raw: list[dict], offset: float, core_start: float, core_end: float):
    """Keep original word strings/times; ownership removes context duplication."""
    result = []
    for segment in raw:
        words = []
        original = re.sub(r"\s+", "", segment.get("text", ""))
        timed = re.sub(r"\s+", "", "".join(w["word"] for w in segment.get("words", [])))
        if "text" in segment and segment.get("words") and original != timed:
            raise ValueError("MLX Whisper word timestamps do not cover the recognized text")
        if segment.get("text", "").strip() and not segment.get("words"):
            raise ValueError("MLX Whisper returned text without required word timestamps")
        for word in segment.get("words", []):
            start, end = offset + float(word["start"]), offset + float(word["end"])
            if not np.isfinite(start) or not np.isfinite(end) or end < start:
                raise ValueError("Invalid MLX Whisper word timestamps")
            if core_start <= (start + end) / 2 < core_end:
                words.append({**word, "start": start, "end": end})
        if words:
            result.append(
                {
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "text": "".join(w["word"] for w in words).strip(),
                    "words": words,
                    "speaker": None,
                }
            )
    return result


def join_decoder_segments(segments: list[dict]) -> list[dict]:
    """Let the exporter split at sentences/pauses instead of decoder boundaries."""
    words = [word for segment in segments for word in segment["words"]]
    if not words:
        return []
    return [
        {
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": "".join(w["word"] for w in words).strip(),
            "words": words,
            "speaker": None,
        }
    ]


def transcribe_file(
    path: str,
    *,
    model_name: str = "large-v3",
    compute_type: str = "float16",
    download_root: str | None = None,
    local_files_only: bool = False,
    join_segments: bool = False,
) -> list[dict]:
    import mlx_whisper
    from huggingface_hub import snapshot_download

    if compute_type not in {"float16", "float32"}:
        raise ValueError("MLX Whisper supports float16 or float32 compute")
    name = resolve_model_name(model_name)
    revision = DEFAULT_REVISION if name == DEFAULT_MODEL else None
    if Path(name).is_dir():
        model_path = str(Path(name).resolve())
    else:
        model_path = snapshot_download(name, revision=revision, local_files_only=local_files_only)
    # A resolved snapshot or local model's actual files, rather than its mutable alias.
    model_hash = hashlib.sha256()
    for item in sorted(Path(model_path).iterdir()):
        if item.suffix in {".json", ".npz", ".safetensors"}:
            model_hash.update(item.name.encode())
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    model_hash.update(block)
    settings = {
        "model": name,
        "model_sha256": model_hash.hexdigest(),
        "revision": revision,
        "compute_type": compute_type,
        "language": "en",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2.0,
        "core_seconds": CORE_SECONDS,
        "context_seconds": CONTEXT_SECONDS,
        "schema": 1,
        "mlx_whisper_version": version("mlx-whisper"),
    }
    wave = load_audio_mono(path)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Input must contain finite audio samples")
    source_hash = hashlib.sha256(wave).hexdigest()
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cache = (
        Path(download_root) / "transcriptions" / source_hash / signature if download_root else None
    )
    segments = []
    started = time.monotonic()
    chunks = (len(wave) + CORE_SECONDS * SAMPLE_RATE - 1) // (CORE_SECONDS * SAMPLE_RATE)
    skipped = 0
    for index, lo in enumerate(range(0, len(wave), CORE_SECONDS * SAMPLE_RATE)):
        hi = min(len(wave), lo + CORE_SECONDS * SAMPLE_RATE)
        core = wave[lo:hi]
        # Craig uses exact zero for gated microphones. Retain every nonzero sample.
        if not np.any(core):
            skipped += 1
            continue
        audio_lo = max(0, lo - CONTEXT_SECONDS * SAMPLE_RATE)
        audio_hi = min(len(wave), hi + CONTEXT_SECONDS * SAMPLE_RATE)
        clip = wave[audio_lo:audio_hi]
        digest = hashlib.sha256(clip).hexdigest()
        target = cache / f"{index:05d}.json" if cache else None
        if target and target.exists():
            cached = json.loads(target.read_text())
            if cached["settings"] != settings or cached["audio_sha256"] != digest:
                raise ValueError("ASR cache provenance mismatch")
            raw = cached["segments"]
        else:
            raw = mlx_whisper.transcribe(
                clip,
                path_or_hf_repo=model_path,
                language="en",
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
                hallucination_silence_threshold=2.0,
                fp16=compute_type == "float16",
                verbose=None,
            )["segments"]
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"settings": settings, "audio_sha256": digest, "segments": raw})
                )
                temporary.replace(target)
        segments.extend(
            owned_segments(raw, audio_lo / SAMPLE_RATE, lo / SAMPLE_RATE, hi / SAMPLE_RATE)
        )
        if (index + 1) % 20 == 0:
            logger.warning("MLX Whisper: %d/%d windows", index + 1, chunks)
    logger.warning(
        "MLX Whisper: %.1fs audio in %.1fs; %d/%d digital-silence windows skipped",
        len(wave) / SAMPLE_RATE,
        time.monotonic() - started,
        skipped,
        chunks,
    )
    return join_decoder_segments(segments) if join_segments else segments
