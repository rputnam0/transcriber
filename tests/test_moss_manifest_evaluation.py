from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_moss_manifest import (  # noqa: E402
    DEFAULT_MODEL,
    resolve_processor_source,
    select_unique_records,
)
from score_moss_manifest import (  # noqa: E402
    parse_target,
    score_outputs,
    temporal_attributed_score,
)


def _manifest_row(cut_id: str, target: str) -> dict:
    return {
        "conversation": [
            {"content": "Transcribe."},
            {"content": "/tmp/audio.wav"},
            {"content": target},
        ],
        "metadata": {"cut_id": cut_id, "session": "Session 63"},
    }


def test_manifest_selection_deduplicates_overlap_repeats() -> None:
    row = _manifest_row("cut-1", "[0.00][S01] hi[1.00]")
    row["metadata"]["activity"] = [{"start": 0.0, "end": 1.0}]
    row["metadata"]["isolated_stem"] = "/tmp/privileged.wav"

    selected = select_unique_records([row, row], max_records=0)

    assert len(selected) == 1
    assert selected[0]["cut_id"] == "cut-1"
    assert set(selected[0]) == {"cut_id", "session", "prompt", "audio"}
    assert DEFAULT_MODEL == "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def test_manifest_selection_balances_session_quota() -> None:
    rows = [
        {
            **_manifest_row(f"cut-{session}-{index}", "[0.00][S01] hi[1.00]"),
            "metadata": {
                "cut_id": f"cut-{session}-{index}",
                "session": f"Session {session}",
            },
        }
        for session in (63, 66)
        for index in range(3)
    ]

    selected = select_unique_records(rows, max_records_per_session=2)

    assert len(selected) == 4
    assert [row["session"] for row in selected].count("Session 63") == 2
    assert [row["session"] for row in selected].count("Session 66") == 2


def test_local_checkpoint_defaults_to_pinned_base_processor(tmp_path: Path) -> None:
    assert resolve_processor_source(str(tmp_path), None) == DEFAULT_MODEL
    assert resolve_processor_source(str(tmp_path), "custom") == "custom"


def test_parse_target_accepts_overlapped_nonmonotonic_segments() -> None:
    target = "[0.00][S01] hello there[1.00][0.20][S02] yes[0.60]"

    segments = parse_target(target)

    assert [segment["speaker"] for segment in segments] == ["S01", "S02"]
    assert segments[1]["start"] == 0.2


def test_temporal_score_does_not_credit_repeated_word_at_wrong_time() -> None:
    reference = [
        {"start": 1.0, "end": 1.5, "speaker": "S01", "text": "yeah"},
        {"start": 9.0, "end": 9.5, "speaker": "S01", "text": "yeah"},
    ]
    prediction = [
        {"start": 1.1, "end": 1.4, "speaker": "P01", "text": "yeah yeah"},
    ]

    score = temporal_attributed_score(
        reference,
        prediction,
        {"P01": "S01"},
        brief_turn_seconds=2.0,
        tolerance_seconds=0.5,
    )

    assert score["reference_words"] == 2
    assert score["predicted_words"] == 2
    assert score["matched_words"] == 1
    assert score["recall"] == 0.5
    assert score["precision"] == 0.5


def test_temporal_score_requires_correct_speaker_during_overlap() -> None:
    reference = [
        {"start": 2.0, "end": 5.0, "speaker": "S01", "text": "keep talking"},
        {"start": 3.5, "end": 4.0, "speaker": "S02", "text": "wait"},
    ]
    prediction = [
        {"start": 2.0, "end": 5.0, "speaker": "P01", "text": "keep talking wait"},
    ]

    score = temporal_attributed_score(
        reference,
        prediction,
        {"P01": "S01"},
        brief_turn_seconds=2.0,
        tolerance_seconds=0.5,
    )

    assert score["matched_words"] == 2
    assert score["categories"]["brief_overlap"] == {
        "reference_words": 1,
        "matched_words": 0,
        "recall": 0.0,
    }


def test_manifest_authored_overlap_flag_overrides_spanning_segment_geometry() -> None:
    row = _manifest_row(
        "cut-1",
        "[0.00][S01] I agree[1.20][0.50][S02] yes[0.80]",
    )
    row["metadata"]["reference_segments"] = [
        {"speaker": "S01", "start": 0.0, "end": 1.2, "overlap": False, "brief": True},
        {"speaker": "S02", "start": 0.5, "end": 0.8, "overlap": False, "brief": True},
    ]
    outputs = [
        {
            "cut_id": "cut-1",
            "segments": [
                {"start": 0.0, "end": 1.2, "speaker": "S01", "text": "I agree"},
                {"start": 0.5, "end": 0.8, "speaker": "S02", "text": "yes"},
            ],
        }
    ]

    score = score_outputs([row], outputs)

    assert score["temporal_categories"]["overlap"]["reference_words"] == 0
    assert score["temporal_categories"]["brief_overlap"]["reference_words"] == 0


def test_manifest_word_spans_assign_overlap_per_word() -> None:
    row = _manifest_row(
        "cut-1",
        "[0.00][S01] before during[1.20][0.50][S02] yes[0.80]",
    )
    row["metadata"]["reference_segments"] = [
        {
            "speaker": "S01",
            "start": 0.0,
            "end": 1.2,
            "overlap": True,
            "brief": True,
            "word_spans": [
                {"text": "before", "start": 0.0, "end": 0.4, "overlap": False, "brief": True},
                {"text": "during", "start": 0.6, "end": 1.0, "overlap": True, "brief": True},
            ],
        },
        {
            "speaker": "S02",
            "start": 0.5,
            "end": 0.8,
            "overlap": True,
            "brief": True,
            "word_spans": [
                {"text": "yes", "start": 0.5, "end": 0.8, "overlap": True, "brief": True}
            ],
        },
    ]
    outputs = [
        {
            "cut_id": "cut-1",
            "segments": [
                {"start": 0.0, "end": 1.2, "speaker": "S01", "text": "before during"},
                {"start": 0.5, "end": 0.8, "speaker": "S02", "text": "yes"},
            ],
        }
    ]

    score = score_outputs([row], outputs)

    assert score["temporal_categories"]["ordinary_nonoverlap"]["reference_words"] == 1
    assert score["temporal_categories"]["brief_overlap"]["reference_words"] == 2


def test_score_outputs_uses_per_cut_speaker_permutation_and_slices() -> None:
    manifest = [
        _manifest_row(
            "cut-1",
            "[0.00][S01] hello there[1.00][0.20][S02] yes[0.60]",
        )
    ]
    outputs = [
        {
            "cut_id": "cut-1",
            "segments": [
                {"start": 0.2, "end": 0.6, "speaker": "S01", "text": "yes"},
                {"start": 0.0, "end": 1.0, "speaker": "S02", "text": "hello there"},
            ],
        }
    ]

    score = score_outputs(manifest, outputs)

    assert score["attributed_bag_recall"] == 1.0
    assert score["attributed_bag_precision"] == 1.0
    assert score["attributed_temporal_recall"] == 1.0
    assert score["attributed_temporal_precision"] == 1.0
    assert score["categories"]["overlap"]["recall"] == 1.0
    assert score["categories"]["brief_overlap"]["recall"] == 1.0
    assert score["temporal_categories"]["brief_overlap"]["recall"] == 1.0
