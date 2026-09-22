from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_tsvad_word_owner_baseline import (  # noqa: E402
    _frame_labels,
    _score_word_owners,
)


def test_frame_labels_preserve_overlap_as_independent_speaker_activity():
    frame_centers = np.asarray([0.01, 0.03, 0.05, 0.07, 0.09], dtype=np.float32)
    words = [
        {"speaker": "Ada", "start": 0.02, "end": 0.08, "text": "one"},
        {"speaker": "Ben", "start": 0.04, "end": 0.10, "text": "two"},
    ]

    ada = _frame_labels(words, "Ada", frame_centers)
    ben = _frame_labels(words, "Ben", frame_centers)

    assert ada.tolist() == [False, True, True, True, False]
    assert ben.tolist() == [False, False, True, True, True]
    assert np.any(ada & ben)


def test_score_word_owners_uses_interval_posterior_not_global_winner():
    frame_centers = np.asarray([0.01, 0.03, 0.05, 0.07], dtype=np.float32)
    words = [
        {"speaker": "Ada", "start": 0.0, "end": 0.04, "text": "hello"},
        {"speaker": "Ben", "start": 0.04, "end": 0.08, "text": "there"},
    ]
    posteriors = {
        "Ada": np.asarray([0.9, 0.8, 0.1, 0.2], dtype=np.float32),
        "Ben": np.asarray([0.2, 0.1, 0.7, 0.8], dtype=np.float32),
    }

    result, records = _score_word_owners(
        words=words,
        speakers=["Ada", "Ben"],
        frame_centers=frame_centers,
        speaker_posteriors=posteriors,
    )

    assert result["correct_words"] == 2
    assert result["accuracy"] == 1.0
    assert [record["predicted"] for record in records] == ["Ada", "Ben"]
