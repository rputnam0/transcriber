from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .consolidate import choose_speaker


@dataclass
class TrainingSegment:
    audio_file: str
    start: float
    end: float
    speaker: str
    speaker_raw: str


@dataclass
class SegmentWindow:
    audio_file: str
    start: float
    end: float
    speaker: str
    speaker_raw: str
    segment_index: int
    window_index: int


def _resolve_label(raw_label: str, mapping: Dict[str, str]) -> Tuple[str, bool]:
    """Return canonical speaker label using mapping; fall back to raw label."""
    raw_label = (raw_label or "").strip()
    if not raw_label:
        return choose_speaker("", mapping, return_match=True)
    canonical, matched = choose_speaker(raw_label, mapping, return_match=True)
    if not matched and raw_label.lower() not in {"", "unknown"}:
        canonical = raw_label
    return canonical, matched


def load_segments_from_jsonl(path: Path, mapping: Dict[str, str]) -> List[TrainingSegment]:
    segments: List[TrainingSegment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            start = float(payload.get("start") or 0.0)
            end = float(payload.get("end") or 0.0)
            audio_file = str(payload.get("file") or path.stem)
            speaker_raw = str(payload.get("speaker_raw") or payload.get("speaker") or "")
            speaker, _matched = _resolve_label(speaker_raw, mapping)
            segments.append(
                TrainingSegment(
                    audio_file=audio_file,
                    start=start,
                    end=end,
                    speaker=speaker,
                    speaker_raw=speaker_raw,
                )
            )
    return segments


def load_segments_from_json(path: Path, mapping: Dict[str, str]) -> List[TrainingSegment]:
    segments: List[TrainingSegment] = []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and any(isinstance(v, list) for v in payload.values()):
        items = payload.items()
    elif isinstance(payload, list):
        # treat as unnamed list referencing the path stem
        items = [(path.stem, payload)]
    else:
        return segments
    for audio_file, entries in items:
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            start = float(entry.get("start") or 0.0)
            end = float(entry.get("end") or 0.0)
            speaker_raw = str(entry.get("speaker") or "")
            speaker, _matched = _resolve_label(speaker_raw, mapping)
            segments.append(
                TrainingSegment(
                    audio_file=str(audio_file),
                    start=start,
                    end=end,
                    speaker=speaker,
                    speaker_raw=speaker_raw,
                )
            )
    return segments


def load_segments_file(path: Path, mapping: Dict[str, str]) -> List[TrainingSegment]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_segments_from_jsonl(path, mapping)
    if suffix == ".json":
        return load_segments_from_json(path, mapping)
    raise ValueError(f"Unsupported segment file type: {path}")


def generate_windows_for_segments(
    segments: Iterable[TrainingSegment],
    *,
    min_duration: float,
    max_duration: float,
    window_size: float,
    window_stride: float,
) -> List[SegmentWindow]:
    """Split training segments into windows obeying duration constraints."""

    windows: List[SegmentWindow] = []
    segments_list = sorted(list(segments), key=lambda seg: (seg.audio_file, seg.start))
    if not segments_list:
        return windows

    min_duration = max(min_duration, 0.0)
    max_duration = max_duration if max_duration > 0 else None
    window_size = window_size if window_size > 0 else 0.0
    window_stride = window_stride if window_stride > 0 else 0.0

    for idx, seg in enumerate(segments_list):
        duration = max(seg.end - seg.start, 0.0)
        if duration <= 0:
            continue
        if duration < min_duration:
            continue

        effective_window = window_size if window_size > 0 else duration
        if max_duration is not None:
            effective_window = min(effective_window, max_duration)
        if effective_window <= 0:
            effective_window = duration
        if duration <= effective_window * 1.05:
            # single window covering whole segment
            windows.append(
                SegmentWindow(
                    audio_file=seg.audio_file,
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    speaker_raw=seg.speaker_raw,
                    segment_index=idx,
                    window_index=0,
                )
            )
            continue

        stride = window_stride if window_stride > 0 else effective_window
        stride = max(stride, min_duration)
        local_index = 0
        pos = seg.start
        last_start = None
        while pos < seg.end:
            window_end = min(pos + effective_window, seg.end)
            if window_end - pos < min_duration:
                break
            windows.append(
                SegmentWindow(
                    audio_file=seg.audio_file,
                    start=pos,
                    end=window_end,
                    speaker=seg.speaker,
                    speaker_raw=seg.speaker_raw,
                    segment_index=idx,
                    window_index=local_index,
                )
            )
            last_start = pos
            local_index += 1
            if window_end >= seg.end:
                break
            pos += stride

        if last_start is not None and windows:
            last_window = windows[-1]
            if (
                last_window.audio_file == seg.audio_file
                and abs(last_window.end - seg.end) > 1e-3
                and seg.end - last_window.start >= min_duration
            ):
                windows[-1] = SegmentWindow(
                    audio_file=last_window.audio_file,
                    start=last_window.start,
                    end=seg.end,
                    speaker=last_window.speaker,
                    speaker_raw=last_window.speaker_raw,
                    segment_index=last_window.segment_index,
                    window_index=last_window.window_index,
                )

    return windows
