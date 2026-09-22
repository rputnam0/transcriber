from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from augment_multitalker_parakeet_masks import (  # noqa: E402
    activity_iou,
    corrupt_activity_supervisions,
    inherit_transcript_spans,
)


def test_corruption_preserves_complete_text_and_speaker_roster() -> None:
    source = [
        {"speaker": "Alice", "start": 1.0, "end": 2.0, "text": "hello"},
        {"speaker": "Bob", "start": 1.8, "end": 2.5, "text": "yes"},
        {"speaker": "Alice", "start": 3.0, "end": 3.4, "text": "there"},
    ]

    corrupted = corrupt_activity_supervisions(
        source,
        duration=5.0,
        profile="strong",
        rng=random.Random(7),
    )

    assert {span["speaker"] for span in corrupted} == {"Alice", "Bob"}
    assert " ".join(span["text"] for span in corrupted if span["speaker"] == "Alice").strip() == (
        "hello there"
    )
    assert " ".join(span["text"] for span in corrupted if span["speaker"] == "Bob").strip() == (
        "yes"
    )
    assert all(0.0 <= span["start"] < span["end"] <= 5.0 for span in corrupted)


def test_activity_iou_detects_mask_mismatch() -> None:
    source = [{"speaker": "Alice", "start": 1.0, "end": 2.0, "text": "hello"}]
    shifted = [{"speaker": "Alice", "start": 1.5, "end": 2.5, "text": "hello"}]

    assert activity_iou(source, source, duration=4.0) == 1.0
    assert activity_iou(source, shifted, duration=4.0) == pytest.approx(1 / 3, abs=0.04)


def test_unknown_corruption_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown mask corruption profile"):
        corrupt_activity_supervisions(
            [],
            duration=5.0,
            profile="mystery",
            rng=random.Random(1),
        )


def test_append_inherits_transcript_spans_without_overwriting_existing_labels() -> None:
    inherited = inherit_transcript_spans(
        {"activity_mask_source": "mono-sortformer"},
        [{"speaker": "Alice", "start": 1.0, "end": 2.0, "text": "hello"}],
    )
    preserved = inherit_transcript_spans(
        {"transcript_spans": [{"text": "existing"}]},
        [{"text": "replacement"}],
    )

    assert inherited["transcript_spans"][0]["text"] == "hello"
    assert preserved["transcript_spans"][0]["text"] == "existing"
