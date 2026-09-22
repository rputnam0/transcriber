from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_moss_named_manifest import (  # noqa: E402
    named_reference_segments,
    score_named_outputs,
)


def _manifest_row() -> dict:
    return {
        "conversation": [
            {"role": "user", "message_type": "text", "content": "transcribe"},
            {"role": "user", "message_type": "audio", "content": "mono.wav"},
            {
                "role": "assistant",
                "message_type": "text",
                "content": "[0.00][S01] hello[1.00][0.50][S02] yeah[0.80]",
            },
        ],
        "metadata": {
            "cut_id": "session_34_w000240_c000000",
            "stable_session_speaker_ids": {"Alice": "S01", "Bob": "S02"},
        },
    }


def test_named_reference_segments_replaces_anonymous_ids() -> None:
    segments = named_reference_segments(_manifest_row())

    assert [segment["speaker"] for segment in segments] == ["Alice", "Bob"]


def test_score_named_outputs_uses_enrollment_mapping_for_brief_overlap() -> None:
    result = score_named_outputs(
        [_manifest_row()],
        [
            {
                "cut_id": "session_34_w000240_c000000",
                "segments": [
                    {"speaker": "S02", "start": 0.0, "end": 1.0, "text": "hello"},
                    {"speaker": "S01", "start": 0.5, "end": 0.8, "text": "yeah"},
                ],
            }
        ],
        [
            {
                "cut_id": "session_34_w000240_c000000",
                "one_to_one_mapping": {"S01": "Bob", "S02": "Alice"},
            }
        ],
        binding_mode="one_to_one",
        brief_turn_seconds=2.0,
    )

    assert result["attributed_bag_recall"] == pytest.approx(1.0)
    assert result["categories"]["brief_overlap"]["matched_words"] == 2
