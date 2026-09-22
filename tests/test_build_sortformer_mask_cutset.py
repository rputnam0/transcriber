from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_sortformer_mask_cutset import (  # noqa: E402
    mapped_supervisions,
    match_probability_speakers,
    match_predicted_speakers,
)


def test_match_predicted_speakers_recovers_permuted_slots() -> None:
    reference = [
        {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "hello"},
        {"start": 0.8, "end": 1.8, "speaker": "Bob", "text": "yes"},
    ]
    predicted = [
        {"start": 0.05, "end": 1.0, "speaker": "slot_1"},
        {"start": 0.8, "end": 1.75, "speaker": "slot_0"},
    ]

    mapping, metrics = match_predicted_speakers(reference, predicted, duration=2.0)

    assert mapping == {"slot_0": "Bob", "slot_1": "Alice"}
    assert metrics["reference_speaker_recall"] == 1.0
    assert metrics["mean_matched_f1"] > 0.9


def test_low_agreement_slot_is_not_given_a_teacher_identity() -> None:
    reference = [{"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "hello"}]
    predicted = [{"start": 3.0, "end": 4.0, "speaker": "slot_0"}]

    mapping, metrics = match_predicted_speakers(
        reference,
        predicted,
        duration=5.0,
        min_f1=0.1,
    )

    assert mapping == {}
    assert metrics["reference_speaker_recall"] == 0.0


def test_mapped_supervisions_use_mono_masks_but_keep_all_target_words() -> None:
    reference = [
        {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "hello"},
        {"start": 2.0, "end": 3.0, "speaker": "Alice", "text": "there"},
    ]
    predicted = [
        {"start": 0.2, "end": 0.9, "speaker": "slot_3"},
        {"start": 2.3, "end": 2.8, "speaker": "slot_3"},
    ]

    output = mapped_supervisions(reference, predicted, {"slot_3": "Alice"})

    assert [(span["start"], span["end"]) for span in output] == [(0.2, 0.9), (2.3, 2.8)]
    assert " ".join(span["text"] for span in output).strip() == "hello there"
    assert {span["speaker"] for span in output} == {"Alice"}


def test_probability_matching_keeps_brief_speaker_without_thresholded_turn() -> None:
    import numpy as np

    reference = [
        {"start": 0.0, "end": 0.8, "speaker": "Alice", "text": "long turn"},
        {"start": 0.4, "end": 0.56, "speaker": "Bob", "text": "brief"},
    ]
    probabilities = np.asarray(
        [
            [0.9, 0.01],
            [0.8, 0.10],
            [0.8, 0.35],
            [0.7, 0.15],
            [0.7, 0.02],
        ],
        dtype=np.float32,
    )

    mapping, metrics = match_probability_speakers(
        reference,
        probabilities,
        duration=0.8,
    )

    assert mapping == {"speaker_0": "Alice", "speaker_1": "Bob"}
    assert metrics["reference_speaker_recall"] == 1.0
    assert metrics["mean_matched_f1"] > 0.3
