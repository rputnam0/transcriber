from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from materialize_cross_session_enrollment_profiles import select_diverse_rows  # noqa: E402


def test_select_diverse_rows_uses_longest_clip_from_distinct_sessions() -> None:
    rows = [
        {
            "session": "Session 1",
            "positive_enrollment_spans": [{"duration": 2.0}],
            "id": "short",
        },
        {
            "session": "Session 1",
            "positive_enrollment_spans": [{"duration": 5.0}],
            "id": "long",
        },
        {
            "session": "Session 2",
            "positive_enrollment_spans": [{"duration": 4.0}],
            "id": "other",
        },
    ]

    selected = select_diverse_rows(rows, max_clips=2)

    assert [row["id"] for row in selected] == ["long", "other"]
