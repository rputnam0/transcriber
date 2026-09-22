from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_moss_transcribe_diarize import normalized_segments  # noqa: E402


def test_normalized_segments_clips_duration_and_drops_invalid_rows() -> None:
    segments = [
        SimpleNamespace(start=3.0, end=5.0, speaker="S02", text=" second "),
        SimpleNamespace(start=-1.0, end=2.0, speaker="S01", text="first"),
        SimpleNamespace(start=6.0, end=7.0, speaker="S03", text="padding"),
        SimpleNamespace(start=2.0, end=3.0, speaker="S01", text="  "),
    ]

    assert normalized_segments(segments, duration=5.5) == [
        {"start": 0.0, "end": 2.0, "speaker": "S01", "text": "first"},
        {"start": 3.0, "end": 5.0, "speaker": "S02", "text": "second"},
    ]
