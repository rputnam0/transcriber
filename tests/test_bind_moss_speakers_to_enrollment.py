from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from bind_moss_speakers_to_enrollment import stream_waveforms  # noqa: E402


def test_stream_waveforms_use_only_selected_mono_regions() -> None:
    mixture = np.arange(100, dtype=np.float32)
    segments = [
        {"speaker": "S01", "start": 0.0, "end": 2.0},
        {"speaker": "S02", "start": 3.0, "end": 5.0},
    ]

    speakers, waves, metadata = stream_waveforms(
        mixture,
        segments,
        sample_rate=10,
        minimum_segment_seconds=0.1,
        minimum_stream_seconds=0.1,
        maximum_stream_seconds=10.0,
    )

    assert speakers == ["S01", "S02"]
    np.testing.assert_array_equal(waves[0], mixture[:20])
    np.testing.assert_array_equal(waves[1], mixture[30:50])
    assert metadata[0]["exclusive_only"] is True


def test_stream_waveforms_can_reject_generated_overlap() -> None:
    mixture = np.ones(100, dtype=np.float32)
    segments = [
        {"speaker": "S01", "start": 0.0, "end": 2.0},
        {"speaker": "S02", "start": 1.0, "end": 3.0},
        {"speaker": "S02", "start": 4.0, "end": 5.0},
    ]

    speakers, waves, _metadata = stream_waveforms(
        mixture,
        segments,
        sample_rate=10,
        minimum_segment_seconds=0.1,
        minimum_stream_seconds=0.1,
        maximum_stream_seconds=10.0,
        exclusive_only=True,
    )

    assert speakers == ["S02"]
    assert len(waves[0]) == 10
