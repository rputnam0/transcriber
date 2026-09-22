from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from render_se_dicow_named_transcript import (  # noqa: E402
    activity_evidence,
    assigned_binding_evidence,
    clip_segment_to_duration,
    mark_duplicate_hypotheses,
    parse_timestamped_segments,
)


def test_clip_segment_to_duration_drops_padding_and_clamps_tail() -> None:
    assert clip_segment_to_duration({"start": 31.0, "end": 32.0}, 30.5) is None
    assert clip_segment_to_duration({"start": 30.0, "end": 32.0}, 30.5) == {
        "start": 30.0,
        "end": 30.5,
    }


def test_parse_timestamped_segments_extracts_closed_intervals() -> None:
    parsed = parse_timestamped_segments("<|0.00|>hello there<|1.20|><|2.00|>again<|3.00|>")

    assert parsed == [
        {"start": 0.0, "end": 1.2, "text": "hello there"},
        {"start": 2.0, "end": 3.0, "text": "again"},
    ]


def test_assigned_binding_margin_uses_assigned_speaker_not_independent_winner() -> None:
    record = {
        "one_to_one_mapping": {"speaker_0": "Alice"},
        "binding_evidence": {"speaker_0": {"scores": {"Alice": 0.6, "Bob": 0.7}}},
    }

    evidence = assigned_binding_evidence(record, speaker="Alice", mode="one_to_one")

    assert evidence["assigned_margin"] == pytest.approx(-0.1)


def test_activity_evidence_marks_probabilistic_overlap() -> None:
    evidence = activity_evidence(
        np.array([[0.8, 0.5], [0.6, 0.7]], dtype=np.float32),
        slot="speaker_0",
        start=0.0,
        end=30.0,
    )

    assert evidence["mean_overlap_probability"] == pytest.approx(0.41)


def test_duplicate_overlapping_hypotheses_are_kept_and_marked() -> None:
    segments = [
        {"speaker": "Alice", "start": 1.0, "end": 2.0, "text": "hello there"},
        {"speaker": "Bob", "start": 1.2, "end": 2.1, "text": "hello there"},
    ]

    mark_duplicate_hypotheses(segments)

    assert segments[0]["review_reasons"] == ["duplicate-overlapping-hypothesis"]
    assert segments[1]["review_reasons"] == ["duplicate-overlapping-hypothesis"]
