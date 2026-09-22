from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_moss_synthetic_overlap_dataset import (  # noqa: E402
    group_high_quality_utterances,
)


def _word(text: str, start: float, end: float, *, score: float = 0.9) -> dict:
    return {
        "speaker": "A",
        "start": start,
        "end": end,
        "text": text,
        "score": score,
        "alignment_source": "mms",
    }


def test_groups_quality_words_and_breaks_on_large_gap() -> None:
    utterances = group_high_quality_utterances(
        [
            _word("hello", 0.0, 0.3),
            _word("there", 0.35, 0.7),
            _word("again", 2.0, 2.4),
        ],
        minimum_score=0.45,
        maximum_word_seconds=1.25,
        maximum_gap_seconds=0.55,
        maximum_utterance_seconds=5.5,
        maximum_words_per_second=6.0,
    )

    assert [[word["text"] for word in item["words"]] for item in utterances] == [
        ["hello", "there"],
        ["again"],
    ]


def test_rejects_low_confidence_long_and_impossible_rate_words() -> None:
    utterances = group_high_quality_utterances(
        [
            _word("low", 0.0, 0.2, score=0.1),
            _word("long", 1.0, 3.0),
            _word("one", 4.0, 4.05),
            _word("two", 4.06, 4.11),
            _word("three", 4.12, 4.17),
        ],
        minimum_score=0.45,
        maximum_word_seconds=1.25,
        maximum_gap_seconds=0.55,
        maximum_utterance_seconds=5.5,
        maximum_words_per_second=6.0,
    )

    assert utterances == []
