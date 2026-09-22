from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_moss_brief_overlap_crops import (  # noqa: E402
    render_weighted_target,
    segment_is_impossible,
    stable_session_speaker_ids,
)


def test_rejects_only_multiword_impossible_rate() -> None:
    assert segment_is_impossible(
        {"start": 0.0, "end": 0.5, "words": ["a", "b", "c", "d"]},
        maximum_words_per_second=6.0,
        minimum_words_for_rate_check=4,
    )
    assert not segment_is_impossible(
        {"start": 0.0, "end": 0.05, "words": ["yeah"]},
        maximum_words_per_second=6.0,
        minimum_words_for_rate_check=4,
    )


def test_session_speaker_ids_are_stable_by_first_onset() -> None:
    rows = [
        {
            "session": "Session 1",
            "window_start": 100.0,
            "words": [
                {"speaker": "Bob", "start": 2.0},
                {"speaker": "Alice", "start": 1.0},
            ],
        },
        {
            "session": "Session 1",
            "window_start": 0.0,
            "words": [{"speaker": "Bob", "start": 1.0}],
        },
    ]

    assert stable_session_speaker_ids(rows) == {"Session 1": {"Bob": "S01", "Alice": "S02"}}


def test_rendered_overlap_has_deterministic_order_and_weight_spans() -> None:
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 3.0, "words": ["primary", "turn"]},
        {"speaker": "Bob", "start": 1.0, "end": 1.5, "words": ["yes"]},
    ]

    target, spans, activity = render_weighted_target(
        segments,
        {"Alice": "S02", "Bob": "S01"},
    )

    assert target == "[0.00][S02] primary turn[3.00][1.00][S01] yes[1.50]"
    brief = [span for span in spans if span["kind"] == "brief_overlap_word"]
    assert target[brief[0]["start"] : brief[0]["end"]] == "yes"
    assert brief[0]["weight"] == 4.0
    assert [item["speaker"] for item in activity] == ["S02", "S01"]
