"""Conservative local speech regions with original-time coordinates.

The recurrent ONNX interface follows Silero's official OnnxWrapper. Segmentation
uses hysteresis, retains short responses, and pads boundaries before ASR.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

import numpy as np

VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
VAD_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "v6.2.2/src/silero_vad/data/silero_vad.onnx"
)
SETTINGS = dict(
    threshold=0.35,
    release_threshold=0.20,
    min_speech_seconds=0.064,
    min_silence_seconds=0.50,
    pad_seconds=0.25,
)


def resolve_vad(cache_root: str | None, local_files_only: bool) -> Path:
    root = Path(cache_root) if cache_root else Path.home() / ".cache/transcriber"
    path = root / "vad/silero-vad-6.2.2.onnx"
    if not path.exists():
        if local_files_only:
            raise ValueError(f"Speech detector missing at {path}; run once without offline mode")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = urllib.request.urlopen(VAD_URL, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != VAD_SHA256:
            raise ValueError("Downloaded speech detector checksum mismatch")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != VAD_SHA256:
        raise ValueError("Speech detector checksum mismatch")
    return path


def probabilities(wave: np.ndarray, model_path: Path) -> np.ndarray:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, 64), np.float32)
    scores = np.empty((len(wave) + 511) // 512, np.float32)
    for i in range(len(scores)):
        frame = np.asarray(wave[i * 512 : (i + 1) * 512], dtype=np.float32)
        frame = np.pad(frame, (0, 512 - len(frame)))[None, :]
        output, state = session.run(
            None,
            dict(
                input=np.concatenate([context, frame], axis=1),
                state=state,
                sr=np.array(16000, np.int64),
            ),
        )
        context = frame[:, -64:]
        scores[i] = output[0, 0]
    return scores


def speech_regions(
    scores,
    samples: int,
    *,
    threshold=0.35,
    release_threshold=0.20,
    min_speech_seconds=0.064,
    min_silence_seconds=0.50,
    pad_seconds=0.25,
):
    """Return merged, padded [start, end) sample intervals on the source timeline."""
    rate, step = 16000, 512
    intervals = []
    start = quiet = None
    for i, score in enumerate(scores):
        position = i * step
        if start is None:
            if score >= threshold:
                start = position
        elif score >= release_threshold:
            quiet = None
        else:
            if quiet is None:
                quiet = position
            if position + step - quiet >= min_silence_seconds * rate:
                intervals.append((start, quiet))
                start = quiet = None
    if start is not None:
        intervals.append((start, quiet if quiet is not None else samples))
    result = []
    pad = round(pad_seconds * rate)
    for start, end in intervals:
        end = min(end, samples)
        if end - start < min_speech_seconds * rate:
            continue
        lo, hi = max(0, start - pad), min(samples, end + pad)
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result
