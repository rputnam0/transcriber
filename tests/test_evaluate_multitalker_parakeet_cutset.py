from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_multitalker_parakeet_cutset import (  # noqa: E402
    oracle_rttm_lines_from_supervisions,
    reference_words_from_supervisions,
    summarize_scores,
)


def test_reference_words_preserve_speaker_text_and_interpolate_times() -> None:
    words = reference_words_from_supervisions(
        [
            {
                "speaker": "Alice",
                "start": 2.0,
                "duration": 1.0,
                "text": "Hello there",
            }
        ]
    )

    assert [word["normalized"] for word in words] == ["hello", "there"]
    assert {word["speaker"] for word in words} == {"Alice"}
    assert [(word["start"], word["end"]) for word in words] == [(2.0, 2.5), (2.5, 3.0)]


def test_summarize_scores_is_micro_averaged() -> None:
    summary = summarize_scores(
        [
            {
                "reference_words": 90,
                "predicted_words": 80,
                "oracle_one_to_one_matched_words": 72,
            },
            {
                "reference_words": 10,
                "predicted_words": 20,
                "oracle_one_to_one_matched_words": 8,
            },
        ]
    )

    assert summary["oracle_one_to_one_attributed_word_recall"] == 0.8
    assert summary["oracle_one_to_one_prediction_precision"] == 0.8


def test_oracle_rttm_lines_use_stable_anonymous_speaker_ids() -> None:
    class Supervision:
        def __init__(self, speaker: str, start: float, duration: float) -> None:
            self.speaker = speaker
            self.start = start
            self.duration = duration

    lines = oracle_rttm_lines_from_supervisions(
        [
            Supervision("Bob Smith", 1.25, 0.5),
            Supervision("Alice Jones", 2.0, 0.75),
        ],
        recording_id="cut-1",
    )

    assert lines == [
        "SPEAKER cut-1 1 1.250 0.500 <NA> <NA> speaker_01 <NA> <NA>",
        "SPEAKER cut-1 1 2.000 0.750 <NA> <NA> speaker_00 <NA> <NA>",
    ]
