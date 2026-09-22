from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from reconcile_moss_manifest_bindings import reconcile_bindings  # noqa: E402


def test_reconcile_bindings_anchors_strong_same_time_text() -> None:
    records = [
        {
            "cut_id": "session_34_w000240_c030000",
            "duration": 30.0,
            "segments": [
                {
                    "speaker": "S01",
                    "start": 1.0,
                    "end": 3.0,
                    "text": "the correct teleportation circle",
                }
            ],
        }
    ]
    bindings = [
        {
            "cut_id": "session_34_w000240_c030000",
            "independent_mapping": {"S01": "Bob"},
            "one_to_one_mapping": {"S01": "Bob"},
        }
    ]
    primary = [
        {
            "speaker": "P01",
            "start": 31.0,
            "end": 34.0,
            "text": "we need the correct teleportation circle",
        },
        {"speaker": "P02", "start": 31.0, "end": 34.0, "text": "hello there"},
    ]

    result = reconcile_bindings(
        records,
        bindings,
        primary,
        {"P01": "Alice", "P02": "Bob"},
        minimum_matches=3,
        minimum_coverage=0.6,
        minimum_margin=0.25,
    )

    assert result[0]["one_to_one_mapping"]["S01"] == "Alice"
    assert result[0]["identity_evidence"]["S01"]["source"] == "long-transcript-lexical-anchor"


def test_reconcile_bindings_leaves_ambiguous_short_interruption_on_enrollment() -> None:
    records = [
        {
            "cut_id": "session_34_w000240_c000000",
            "duration": 30.0,
            "segments": [{"speaker": "S01", "start": 1.0, "end": 1.2, "text": "yeah"}],
        }
    ]
    bindings = [
        {
            "cut_id": "session_34_w000240_c000000",
            "independent_mapping": {"S01": "Bob"},
            "one_to_one_mapping": {"S01": "Bob"},
        }
    ]

    result = reconcile_bindings(
        records,
        bindings,
        [
            {"speaker": "P01", "start": 1.0, "end": 2.0, "text": "yeah"},
            {"speaker": "P02", "start": 1.0, "end": 2.0, "text": "yeah"},
        ],
        {"P01": "Alice", "P02": "Bob"},
        minimum_matches=3,
        minimum_coverage=0.6,
        minimum_margin=0.25,
    )

    assert result[0]["one_to_one_mapping"]["S01"] == "Bob"
    assert result[0]["identity_evidence"]["S01"]["source"] == "historical-enrollment"
