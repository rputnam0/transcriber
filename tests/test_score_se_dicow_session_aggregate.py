from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_se_dicow_session_aggregate import aggregate_records  # noqa: E402


def test_aggregate_scoring_recovers_words_shifted_across_cut_boundary() -> None:
    results = [
        {
            "cut_id": "session_1_w000100_c000000",
            "session": "Session 1",
            "records": [{"speaker": "Alice", "reference": "one two", "prediction": "one"}],
        },
        {
            "cut_id": "session_1_w000100_c030000",
            "session": "Session 1",
            "records": [{"speaker": "Alice", "reference": "three", "prediction": "two three"}],
        },
    ]

    scored = aggregate_records(results)

    assert scored["sequence_matches"] == 3
    assert scored["attributed_sequence_recall"] == 1.0
    assert scored["attributed_sequence_precision"] == 1.0
