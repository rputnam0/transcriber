from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_moss_activity_head import (  # noqa: E402
    reference_overlap_frames,
    score_activity,
    threshold_metrics,
)


def test_reference_overlap_requires_two_speakers() -> None:
    activity = [
        {"speaker_index": 0, "start": 0.0, "end": 2.0},
        {"speaker_index": 1, "start": 1.0, "end": 2.0},
    ]

    assert reference_overlap_frames(activity, frame_count=2, frame_hz=1.0) == [False, True]


def test_threshold_metrics() -> None:
    metrics = threshold_metrics([0.1, 0.9], [False, True], threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_score_selects_high_recall_operating_point() -> None:
    manifest = [
        {
            "metadata": {
                "cut_id": "one",
                "activity": [
                    {"speaker_index": 0, "start": 0.0, "end": 2.0},
                    {"speaker_index": 1, "start": 1.0, "end": 2.0},
                ],
            }
        }
    ]
    records = [
        {
            "cut_id": "one",
            "activity_frame_hz": 1.0,
            "activity_overlap_probabilities": [0.1, 0.9],
        }
    ]

    score = score_activity(manifest, records, target_recall=0.9)

    assert score["best_f1"]["f1"] == 1.0
    assert score["target_recall_operating_point"]["precision"] == 1.0
