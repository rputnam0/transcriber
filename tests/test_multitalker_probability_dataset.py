from __future__ import annotations

import numpy as np
import pytest

from transcriber.multitalker_probability_dataset import speaker_slot_index, soft_target_masks


@pytest.mark.parametrize(
    ("label", "expected"),
    [("speaker_0", 0), ("spk7", 7), ("slot_12", 12)],
)
def test_speaker_slot_index(label: str, expected: int) -> None:
    assert speaker_slot_index(label) == expected


def test_soft_target_masks_keep_target_and_union_background_probabilities() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.2, 0.5],
            [0.1, 0.4, 0.0],
        ],
        dtype=np.float32,
    )

    target, background = soft_target_masks(probabilities, target_slot=0)

    assert target.tolist() == pytest.approx([0.9, 0.1])
    assert background.tolist() == pytest.approx([0.6, 0.4])


def test_soft_target_masks_reject_bad_shape_or_slot() -> None:
    with pytest.raises(ValueError, match="Expected"):
        soft_target_masks(np.zeros(3), target_slot=0)
    with pytest.raises(ValueError, match="outside"):
        soft_target_masks(np.zeros((3, 2)), target_slot=2)
