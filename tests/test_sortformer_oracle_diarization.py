from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_sortformer_oracle_diarization import _parse_sortformer_segments  # noqa: E402


def test_parse_sortformer_segments_sorts_and_skips_bad_lines():
    parsed = _parse_sortformer_segments(
        [
            "1.000 2.000 speaker_1",
            "0.000 0.500 speaker_0",
            "bad line",
            "3.000 2.000 speaker_2",
        ]
    )

    assert parsed == [
        {"start": 0.0, "end": 0.5, "speaker": "speaker_0"},
        {"start": 1.0, "end": 2.0, "speaker": "speaker_1"},
    ]


def test_parse_sortformer_segments_accepts_mapping_rows():
    parsed = _parse_sortformer_segments([{"start": 0, "end": 1.25, "speaker": "spk"}])

    assert parsed == [{"start": 0.0, "end": 1.25, "speaker": "spk"}]
