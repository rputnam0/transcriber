from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_moss_target_speaker import score_target_outputs  # noqa: E402


def _row(cut_id: str, *, negative: bool, segments: list[dict]) -> dict:
    return {
        "metadata": {
            "cut_id": cut_id,
            "session": "Session 63",
            "target_speaker": "Known Speaker",
            "negative": negative,
            "reference_segments": segments,
        }
    }


def test_scores_target_recovery_overlap_and_absent_speaker_leakage() -> None:
    manifest = [
        _row(
            "positive",
            negative=False,
            segments=[
                {"text": "ordinary words", "brief_overlap": False},
                {"text": "yeah right", "brief_overlap": True},
            ],
        ),
        _row("negative", negative=True, segments=[]),
    ]
    outputs = [
        {
            "cut_id": "positive",
            "segments": [{"start": 0.0, "end": 1.0, "text": "ordinary yeah right"}],
        },
        {
            "cut_id": "negative",
            "segments": [{"start": 0.0, "end": 1.0, "text": "wrong voice"}],
        },
    ]

    score = score_target_outputs(manifest, outputs)

    assert score["reference_words"] == 4
    assert score["sequence_matches"] == 3
    assert score["brief_overlap_reference_words"] == 2
    assert score["brief_overlap_sequence_matches"] == 2
    assert score["negative_record_false_positive_rate"] == 1.0
    assert score["negative_false_positive_words"] == 2
