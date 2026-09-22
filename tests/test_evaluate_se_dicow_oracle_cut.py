from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_se_dicow_oracle_cut import (  # noqa: E402
    attributed_bag_metrics,
    attributed_sequence_metrics,
    broaden_stno_masks,
    build_diarization_mask,
    enrollment_stno_mask,
    _enrollment_roster,
    load_enrollment_slot_mapping,
    load_cut_audio,
    sortformer_soft_stno_masks,
    sortformer_stno_masks,
    stno_mask,
)


def test_load_cut_audio_honors_offset_in_shared_recording(tmp_path: Path) -> None:
    path = tmp_path / "shared.wav"
    sf.write(path, np.concatenate((np.zeros(80), np.ones(80))), 80, subtype="FLOAT")

    wave, sample_rate = load_cut_audio(path, {"start": 1.0, "duration": 1.0})

    assert sample_rate == 80
    assert len(wave) == 80
    assert np.allclose(wave, 1.0)


def test_stno_mask_distinguishes_target_other_and_overlap() -> None:
    diarization = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    mask = stno_mask(diarization, 0)

    assert torch.equal(mask.argmax(dim=0), torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(mask.sum(dim=0), torch.ones(4))


def test_broaden_stno_masks_turns_all_detected_speech_into_target_prior() -> None:
    masks = torch.eye(4).unsqueeze(0)

    broadened = broaden_stno_masks(masks, alpha=1.0)

    assert torch.equal(broadened.argmax(dim=1), torch.tensor([[0, 1, 1, 1]]))
    assert torch.allclose(broadened.sum(dim=1), torch.ones(1, 4))


def test_build_diarization_mask_uses_only_reference_intervals() -> None:
    supervisions = [
        {"speaker": "A", "start": 0.0, "duration": 0.5},
        {"speaker": "B", "start": 0.25, "duration": 0.5},
    ]

    mask = build_diarization_mask(supervisions, ["A", "B"], duration=1.0, frame_hz=4)

    assert torch.equal(mask, torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]))


def test_enrollment_stno_marks_padded_tail_as_silence() -> None:
    wave = np.ones(800, dtype=np.float32) * 0.1

    mask = enrollment_stno_mask(wave, sample_rate=800, frame_hz=10, output_frames=20)

    assert torch.all(mask[1, :10] == 1.0)
    assert torch.all(mask[0, 10:] == 1.0)
    assert torch.allclose(mask.sum(dim=0), torch.ones(20))


def test_sortformer_stno_masks_use_mono_slot_probabilities() -> None:
    probabilities = np.array(
        [
            [0.0, 0.0],
            [0.9, 0.0],
            [0.0, 0.9],
            [0.9, 0.9],
        ],
        dtype=np.float32,
    )

    masks = sortformer_stno_masks(
        probabilities,
        {"speaker_1": "Alice", "speaker_0": "Bob"},
        ["Alice", "Bob"],
        threshold=0.5,
        output_frames=4,
    )

    assert torch.equal(masks[0].argmax(dim=0), torch.tensor([0, 2, 1, 3]))
    assert torch.equal(masks[1].argmax(dim=0), torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(masks.sum(dim=1), torch.ones(2, 4))


def test_sortformer_stno_masks_allow_unmapped_enrollment_speaker() -> None:
    probabilities = np.array([[0.9], [0.0]], dtype=np.float32)

    masks = sortformer_stno_masks(
        probabilities,
        {"speaker_0": "Alice"},
        ["Alice", "Bob"],
        threshold=0.5,
        output_frames=2,
        require_all_speakers=False,
    )

    assert torch.equal(masks[1].argmax(dim=0), torch.tensor([2, 0]))


def test_sortformer_soft_stno_masks_preserve_probability_mass() -> None:
    probabilities = np.array(
        [[0.0, 0.0], [0.8, 0.0], [0.0, 0.9], [0.8, 0.9]],
        dtype=np.float32,
    )

    masks = sortformer_soft_stno_masks(
        probabilities,
        {"speaker_0": "Alice", "speaker_1": "Bob"},
        ["Alice", "Bob", "Absent"],
        output_frames=4,
        require_all_speakers=False,
    )

    assert torch.equal(masks[0].argmax(dim=0), torch.tensor([0, 1, 2, 3]))
    assert torch.equal(masks[2].argmax(dim=0), torch.tensor([0, 2, 2, 2]))
    assert torch.allclose(masks.sum(dim=1), torch.ones(3, 4))


def test_load_enrollment_slot_mapping_selects_cut_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "bindings.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cut_id": "cut-a",
                        "independent_mapping": {"speaker_0": "Alice"},
                        "one_to_one_mapping": {"speaker_0": "Bob"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    mapping = load_enrollment_slot_mapping(path, cut_id="cut-a", mode="one_to_one")

    assert mapping == {"speaker_0": "Bob"}


def test_enrollment_roster_includes_only_materialized_profiles(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "session": "Session 1",
            "speaker_id": "Alice",
            "materialized": {"positive_enrollment_paths": ["alice.wav"]},
        },
        {
            "session": "Session 1",
            "speaker_id": "Bob",
            "materialized": {},
        },
        {
            "session": "Session 2",
            "speaker_id": "Carol",
            "materialized": {"positive_enrollment_paths": ["carol.wav"]},
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert _enrollment_roster(manifest, session="Session 1") == ["Alice"]


def test_attributed_bag_metrics_preserves_speaker_specific_counts() -> None:
    metrics = attributed_bag_metrics("hello there hello", "hello hello extra")

    assert metrics["reference_words"] == 3
    assert metrics["predicted_words"] == 3
    assert metrics["bag_matches"] == 2
    assert metrics["recall"] == 2 / 3
    assert metrics["precision"] == 2 / 3


def test_sequence_metrics_penalize_reordered_words() -> None:
    metrics = attributed_sequence_metrics(
        "one two three four",
        "three four one two",
    )

    assert metrics["sequence_matches"] == 2
    assert metrics["sequence_recall"] == 0.5
    assert metrics["sequence_precision"] == 0.5


def test_sequence_metrics_tolerate_short_word_boundary_changes() -> None:
    metrics = attributed_sequence_metrics(
        "what i said",
        "whati said",
    )

    assert metrics["sequence_matches"] == 3
    assert metrics["sequence_prediction_matches"] == 2
    assert metrics["sequence_recall"] == 1.0
    assert metrics["sequence_precision"] == 1.0
