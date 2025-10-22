from __future__ import annotations

from typing import List

from faster_whisper import WhisperModel


def load_model(
    model_name: str,
    compute_type: str = "float16",
    device: str = "cuda",
    download_root: str | None = None,
    local_files_only: bool = False,
) -> WhisperModel:
    """Initialise a faster-whisper model using the requested device and precision."""
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
        local_files_only=local_files_only,
    )


def transcribe_file(path: str, model: WhisperModel, batch_size: int = 32) -> List[dict]:
    """Run transcription and return a list of segment dictionaries.

    Uses adaptive batch size: if an OOM occurs, halves the batch size and retries.
    """
    attempt_bs = max(1, int(batch_size or 1))
    while True:
        try:
            try:
                # Newer faster-whisper supports batch_size
                segments, _ = model.transcribe(path, batch_size=attempt_bs, vad_filter=True)
            except TypeError:
                # Older versions don't accept batch_size
                segments, _ = model.transcribe(path, vad_filter=True)
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
            attempt_bs = max(1, attempt_bs // 2)
    output: List[dict] = []
    for seg in segments:
        output.append(
            {
                "start": float(seg.start or 0.0),
                "end": float(seg.end or 0.0),
                "text": seg.text.strip(),
                "speaker": seg.speaker if hasattr(seg, "speaker") else None,
            }
        )
    return output
