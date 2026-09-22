from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_moss_target_speaker_dataset import (  # noqa: E402
    parse_serialized_segments,
    render_target,
    select_cross_session_rows,
)


def test_parse_and_render_target_shifts_timestamps_and_uses_target_slot() -> None:
    segments = parse_serialized_segments(
        "[0.20][S02] primary words[1.40][0.75][S01] yeah right[1.10]"
    )
    segments[1]["brief_overlap"] = True

    target, loss_spans, references = render_target(
        segments,
        source_speaker_id="S01",
        timestamp_shift=4.5,
    )

    assert target == "[5.25][S01] yeah right[5.60]"
    assert references == [{"start": 0.75, "end": 1.1, "text": "yeah right", "brief_overlap": True}]
    assert {span["kind"]: span["weight"] for span in loss_spans} == {
        "timestamp": 2.0,
        "speaker_tag": 3.0,
        "brief_overlap_word": 4.0,
    }


def test_cross_session_enrollment_never_selects_excluded_session() -> None:
    rows = [
        {
            "row_id": f"row-{session}",
            "session": session,
            "speaker_id": "Known Speaker",
            "source_zip": f"{session}.zip",
            "target_member": "speaker.ogg",
            "positive_enrollment_spans": [{"duration": duration}],
        }
        for session, duration in (("Session 43", 40.0), ("Session 44", 30.0), ("Session 45", 20.0))
    ]

    selected = select_cross_session_rows(
        rows,
        speaker="Known Speaker",
        excluded_session="Session 43",
        max_clips=2,
    )

    assert [row["session"] for row in selected] == ["Session 44", "Session 45"]


def test_target_rendering_drops_other_speakers() -> None:
    segments = parse_serialized_segments(
        "[0.00][S01] one[0.50][0.20][S02] two[0.70][0.80][S01] three[1.20]"
    )

    target, _loss_spans, references = render_target(
        segments,
        source_speaker_id="S01",
        timestamp_shift=4.5,
    )

    assert target == "[4.50][S01] one[5.00][5.30][S01] three[5.70]"
    assert [item["text"] for item in references] == ["one", "three"]
