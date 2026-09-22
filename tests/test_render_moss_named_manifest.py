from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from render_moss_named_manifest import (  # noqa: E402
    cut_offset_seconds,
    render_named_segments,
)


def test_cut_offset_seconds_reads_manifest_cut_id() -> None:
    assert cut_offset_seconds("session_34_w000240_c030000") == 30.0


def test_render_named_segments_marks_low_margin_overlap_for_review() -> None:
    records = [
        {
            "cut_id": "session_34_w000240_c030000",
            "activity_frame_hz": 2.0,
            "activity_overlap_probabilities": [0.1, 0.8, 0.9, 0.1],
            "segments": [{"speaker": "S02", "start": 0.5, "end": 1.5, "text": "yeah"}],
        }
    ]
    bindings = [
        {
            "cut_id": "session_34_w000240_c030000",
            "one_to_one_mapping": {"S02": "Alice"},
            "binding_evidence": {"S02": {"scores": {"Alice": 0.55, "Bob": 0.45}}},
            "stream_metadata": [
                {
                    "speaker": "S02",
                    "selected_intervals": [{"start": 0.5, "end": 1.5, "overlapped": True}],
                }
            ],
        }
    ]

    segments = render_named_segments(
        records,
        bindings,
        mode="one_to_one",
        binding_margin_threshold=0.25,
        binding_score_threshold=0.4,
        overlap_review_threshold=0.665,
    )

    assert segments[0]["speaker"] == "Alice"
    assert segments[0]["start"] == pytest.approx(30.5)
    assert segments[0]["binding_margin"] == pytest.approx(0.1)
    assert segments[0]["needs_review"] is True
    assert segments[0]["review_reasons"] == [
        "low-speaker-margin",
        "overlap-only-identity-evidence",
        "ambiguous-crosstalk",
    ]


def test_render_named_segments_keeps_strong_exclusive_binding_high_confidence() -> None:
    records = [
        {
            "cut_id": "session_34_w000240_c000000",
            "segments": [{"speaker": "S01", "start": 1.0, "end": 2.0, "text": "hello"}],
        }
    ]
    bindings = [
        {
            "cut_id": "session_34_w000240_c000000",
            "one_to_one_mapping": {"S01": "Alice"},
            "binding_evidence": {"S01": {"scores": {"Alice": 0.8, "Bob": 0.2}}},
            "stream_metadata": [
                {
                    "speaker": "S01",
                    "selected_intervals": [{"start": 1.0, "end": 2.0, "overlapped": False}],
                }
            ],
        }
    ]

    segments = render_named_segments(
        records,
        bindings,
        mode="one_to_one",
        binding_margin_threshold=0.25,
        binding_score_threshold=0.4,
        overlap_review_threshold=0.665,
    )

    assert segments[0]["confidence"] == "high"
    assert segments[0]["review_reasons"] == []


def test_render_named_segments_accepts_strong_lexical_anchor() -> None:
    records = [
        {
            "cut_id": "session_34_w000240_c000000",
            "segments": [{"speaker": "S01", "start": 1.0, "end": 2.0, "text": "hello"}],
        }
    ]
    bindings = [
        {
            "cut_id": "session_34_w000240_c000000",
            "one_to_one_mapping": {"S01": "Alice"},
            "identity_evidence": {
                "S01": {
                    "source": "long-transcript-lexical-anchor",
                    "lexical_coverage": 0.9,
                    "lexical_margin": 0.7,
                }
            },
            "stream_metadata": [
                {
                    "speaker": "S01",
                    "selected_intervals": [{"start": 1.0, "end": 2.0, "overlapped": True}],
                }
            ],
        }
    ]

    segments = render_named_segments(
        records,
        bindings,
        mode="one_to_one",
        binding_margin_threshold=0.25,
        binding_score_threshold=0.4,
        overlap_review_threshold=0.665,
    )

    assert segments[0]["confidence"] == "high"
    assert segments[0]["identity_source"] == "long-transcript-lexical-anchor"
    assert segments[0]["review_reasons"] == []
