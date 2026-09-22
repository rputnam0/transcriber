from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_moss_target_activity import (  # noqa: E402
    score_target_activity,
    target_activity_frames,
)


def test_target_activity_frames_uses_only_mono_interval() -> None:
    indices, truth = target_activity_frames(
        [{"start": 4.7, "end": 4.9}],
        frame_count=12,
        frame_hz=2.0,
        valid_start=4.5,
        valid_end=6.0,
    )

    assert indices == [9, 10, 11]
    assert truth == [True, False, False]


def test_score_target_activity_separates_presence_and_ignores_prefix() -> None:
    manifest = [
        {
            "metadata": {
                "cut_id": "positive",
                "activity": [{"start": 4.5, "end": 5.0}],
                "activity_valid_start": 4.5,
                "activity_valid_end": 5.5,
            }
        },
        {
            "metadata": {
                "cut_id": "negative",
                "activity": [],
                "activity_valid_start": 4.5,
                "activity_valid_end": 5.5,
            }
        },
    ]
    outputs = [
        {
            "cut_id": "positive",
            "target_activity_frame_hz": 2.0,
            "target_activity_probabilities": [0.99] * 9 + [0.9, 0.1],
        },
        {
            "cut_id": "negative",
            "target_activity_frame_hz": 2.0,
            "target_activity_probabilities": [0.99] * 9 + [0.2, 0.1],
        },
    ]

    score = score_target_activity(
        manifest,
        outputs,
        target_recall=0.9,
        presence_top_frames=1,
    )

    assert score["valid_mono_frames"] == 4
    assert score["target_active_frames"] == 1
    assert score["positive_presence_score_mean"] == 0.9
    assert score["negative_presence_score_mean"] == 0.2
    assert score["target_presence"]["best_f1"]["f1"] == 1.0
