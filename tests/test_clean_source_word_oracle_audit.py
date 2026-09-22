from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_clean_source_word_oracle import (  # noqa: E402
    _best_threshold,
    _word_overlap_speakers,
)


def test_word_overlap_speakers_preserves_multi_label_activity():
    words = [
        {"speaker": "Ada", "start": 0.0, "end": 1.0},
        {"speaker": "Ben", "start": 0.5, "end": 1.2},
        {"speaker": "Cy", "start": 1.3, "end": 1.5},
    ]

    assert _word_overlap_speakers(words[0], words, min_overlap_seconds=0.02) == {"Ada", "Ben"}


def test_best_threshold_prefers_f1_then_accuracy():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    result = _best_threshold(scores, labels)

    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0
    assert 0.2 < result["threshold"] <= 0.81
