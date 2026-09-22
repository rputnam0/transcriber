from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_multitalker_parakeet import _best_one_to_one_mapping, score  # noqa: E402


def test_best_one_to_one_mapping_keeps_speaker_slots_unique() -> None:
    matches = {
        ("speaker_0", "Alice"): 1,
        ("speaker_0", "Bob"): 5,
        ("speaker_1", "Alice"): 4,
        ("speaker_1", "Bob"): 1,
    }

    mapping = _best_one_to_one_mapping(
        matches,
        ["speaker_0", "speaker_1"],
        ["Alice", "Bob"],
    )

    assert mapping == {"speaker_0": "Bob", "speaker_1": "Alice"}


def test_score_reports_overlap_and_non_overlap_recovery() -> None:
    reference = [
        {"speaker": "Alice", "start": 0.0, "end": 0.4, "text": "hello"},
        {"speaker": "Alice", "start": 0.5, "end": 1.0, "text": "there"},
        {"speaker": "Bob", "start": 0.7, "end": 1.1, "text": "yes"},
        {"speaker": "Bob", "start": 1.2, "end": 1.5, "text": "okay"},
    ]
    predictions = [
        {"speaker": "speaker_0", "start_time": 0.6, "end_time": 1.5, "words": "yes okay"},
        {"speaker": "speaker_1", "start_time": 0.0, "end_time": 1.0, "words": "hello there"},
    ]

    result = score(reference, predictions)

    assert result["oracle_one_to_one_mapping"] == {
        "speaker_0": "Bob",
        "speaker_1": "Alice",
    }
    assert result["oracle_one_to_one_attributed_word_recall"] == 1.0
    assert result["overlap_reference_words"] == 2
    assert result["overlap_attributed_word_recall"] == 1.0
