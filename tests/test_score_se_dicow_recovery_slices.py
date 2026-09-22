from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_se_dicow_recovery_slices import score_recovery_slices  # noqa: E402


def test_score_recovery_slices_separates_overlap_and_brief_turns() -> None:
    cuts = [
        {
            "id": "session_64_w000100_c000000",
            "custom": {
                "transcript_spans": [
                    {"speaker": "A", "start": 0.0, "end": 1.0, "text": "hello there"},
                    {"speaker": "B", "start": 0.5, "end": 1.5, "text": "yes"},
                    {"speaker": "A", "start": 3.0, "end": 7.0, "text": "ordinary words"},
                ]
            },
            "supervisions": [
                {"speaker": "A", "start": 0.0, "duration": 1.0},
                {"speaker": "B", "start": 0.0, "duration": 0.75},
                {"speaker": "A", "start": 3.0, "duration": 4.0},
            ],
        }
    ]
    results = [
        {
            "cut_id": "session_64_w000100_c000000",
            "records": [
                {"speaker": "A", "prediction": "hello ordinary words"},
                {"speaker": "B", "prediction": ""},
            ],
        }
    ]

    score = score_recovery_slices(cuts, results, brief_turn_seconds=2.0)

    assert score["categories"]["all"] == {
        "reference_words": 5,
        "matched_words": 3,
        "recall": 0.6,
    }
    assert score["categories"]["ordinary_nonoverlap"]["recall"] == 0.5
    assert score["categories"]["overlap"]["recall"] == 1.0
    assert score["categories"]["brief_turn"]["recall"] == 1 / 3
    assert score["categories"]["brief_overlap"]["recall"] == 1.0
