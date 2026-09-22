from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_multitalker_parakeet_drive_dataset import (  # noqa: E402
    activity_statistics,
    align_text_to_vad,
    deduplicate_windows,
    energy_vad_regions,
    select_chunk_supervisions,
)


def test_deduplicate_windows_removes_target_speaker_rows() -> None:
    rows = [
        {
            "split_id": "train",
            "session": "Session 1",
            "window_start": 0.0,
            "window_end": 300.0,
            "speaker_id": speaker,
        }
        for speaker in ("Alice", "Bob")
    ]

    windows = deduplicate_windows(rows, split="train")

    assert len(windows) == 1


def test_select_chunk_supervisions_preserves_overlap_and_boundary_text() -> None:
    spans = [
        {"speaker": "Alice", "start": 4.0, "end": 8.0, "text": "long phrase here"},
        {"speaker": "Bob", "start": 6.0, "end": 7.0, "text": "yes"},
        {"speaker": "Alice", "start": 28.5, "end": 30.5, "text": "boundary phrase"},
        {"speaker": "Carol", "start": 31.0, "end": 32.0, "text": "next chunk"},
    ]

    selected, summary = select_chunk_supervisions(
        spans,
        chunk_start=0.0,
        chunk_duration=30.0,
        collar_seconds=0.0,
    )
    activity = activity_statistics(selected, duration=30.0)

    assert [span["text"] for span in selected] == [
        "long phrase here",
        "yes",
        "boundary phrase",
    ]
    assert summary["boundary_clipped_spans"] == 1
    assert activity["overlap_seconds"] == pytest.approx(1.04)


def test_energy_vad_and_text_alignment_preserve_target_words() -> None:
    import numpy as np

    waveform = np.zeros(16000 * 3, dtype=np.float32)
    waveform[16000:24000] = 0.1
    regions = energy_vad_regions(waveform, sample_rate=16000)
    aligned = align_text_to_vad(
        [{"speaker": "Alice", "start": 0.9, "end": 1.7, "text": "hello there"}],
        {"Alice": regions},
        duration=3.0,
        collar_seconds=0.0,
    )

    assert len(regions) == 1
    assert regions[0][0] == pytest.approx(0.92)
    assert regions[0][1] == pytest.approx(1.58)
    assert aligned[0]["text"] == "hello there"
