from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_se_dicow_cutset_adapter import (  # noqa: E402
    broaden_stno_to_all_speech,
    build_cutset_examples,
    choose_enrollment_row,
    choose_wrong_enrollment_speaker,
    enrollment_cross_gate_values,
    enrollment_index,
    enrollment_ranking_loss,
    sample_training_assignment,
    scale_enrollment_cross_gates,
    set_enrollment_cross_gates,
    soft_stno_from_sortformer,
    stno_from_spans,
    target_text,
    target_timestamped_text,
)


def test_set_enrollment_cross_gates_opens_only_speaker_conditioning_path() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ca_enrolls = torch.nn.ModuleList(
                [torch.nn.ModuleDict({"cross_gate": torch.nn.Linear(1, 1, bias=False)})]
            )
            self.ca_enrolls[0].cross_gate.register_parameter(
                "gate", torch.nn.Parameter(torch.tensor([0.0]))
            )

    model = TinyModel()

    values = set_enrollment_cross_gates(model, 0.1)

    assert list(values.values()) == [pytest.approx(0.1)]
    assert enrollment_cross_gate_values(model) == values


def test_scale_enrollment_cross_gates_preserves_sign_pattern() -> None:
    class GateModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ca_enrolls = torch.nn.Module()
            self.ca_enrolls.cross_gate = torch.nn.Module()
            self.ca_enrolls.cross_gate.gate = torch.nn.Parameter(torch.tensor([-0.02, 0.01]))

    model = GateModel()

    values = scale_enrollment_cross_gates(model, 5.0)

    assert list(values.values()) == [pytest.approx(-0.025)]
    assert torch.allclose(model.ca_enrolls.cross_gate.gate, torch.tensor([-0.1, 0.05]))


def test_stno_from_spans_preserves_overlap_and_absent_target() -> None:
    spans = [
        {"speaker": "A", "start": 0.0, "end": 0.75},
        {"speaker": "B", "start": 0.5, "end": 1.0},
    ]

    mask = stno_from_spans(
        spans,
        target_speaker="A",
        duration=1.5,
        frame_rate=4,
    )
    absent = stno_from_spans(
        spans,
        target_speaker="C",
        duration=1.5,
        frame_rate=4,
    )

    assert np.array_equal(mask.argmax(axis=0), np.array([1, 1, 3, 2, 0, 0]))
    assert np.array_equal(absent.argmax(axis=0), np.array([2, 2, 2, 2, 0, 0]))


def test_soft_sortformer_stno_uses_mono_probabilities_and_handles_absent_target() -> None:
    probabilities = np.array(
        [
            [0.0, 0.0],
            [0.8, 0.0],
            [0.0, 0.9],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )

    target = soft_stno_from_sortformer(
        probabilities,
        {"speaker_0": "A", "speaker_1": "B"},
        target_speaker="A",
        duration=1.0,
        frame_rate=4,
    )
    absent = soft_stno_from_sortformer(
        probabilities,
        {"speaker_0": "A", "speaker_1": "B"},
        target_speaker="C",
        duration=1.0,
        frame_rate=4,
    )

    assert np.array_equal(target.argmax(axis=0), np.array([0, 1, 2, 3]))
    assert np.array_equal(absent.argmax(axis=0), np.array([0, 2, 2, 2]))
    assert np.allclose(target.sum(axis=0), 1.0)


def test_broaden_stno_to_all_speech_removes_target_identity_but_keeps_silence() -> None:
    stno = np.eye(4, dtype=np.float32)

    broadened = broaden_stno_to_all_speech(stno, alpha=1.0)

    assert np.array_equal(
        broadened.argmax(axis=0),
        np.array([0, 1, 1, 1]),
    )
    assert np.allclose(broadened.sum(axis=0), 1.0)


def test_target_text_keeps_only_requested_speaker_in_time_order() -> None:
    spans = [
        {"speaker": "A", "start": 0.0, "text": "hello"},
        {"speaker": "B", "start": 0.1, "text": "wrong"},
        {"speaker": "A", "start": 0.2, "text": "there"},
    ]

    assert target_text(spans, "A") == "hello there"


def test_target_timestamped_text_keeps_transcript_timing_separate_from_activity() -> None:
    spans = [
        {"speaker": "A", "start": 0.011, "end": 1.011, "text": "hello"},
        {"speaker": "B", "start": 0.5, "end": 1.5, "text": "wrong"},
        {"speaker": "A", "start": 2.0, "end": 3.0, "text": "there"},
    ]

    assert target_timestamped_text(spans, "A") == ("<|0.02|>hello<|1.02|><|2.00|>there<|3.00|>")


def test_build_cutset_examples_filters_roster_and_activity_source() -> None:
    cut = {
        "id": "session_1_w000000_c000000-mask-sortformer",
        "duration": 30.0,
        "recording": {"sources": [{"source": "mono.wav"}]},
        "supervisions": [
            {"speaker": "A", "start": 0.0, "duration": 1.0, "text": "hello"},
            {"speaker": "NPC", "start": 1.0, "duration": 1.0, "text": "no"},
        ],
        "custom": {"activity_mask_source": "mono-sortformer"},
    }

    examples = build_cutset_examples(
        [cut],
        allowed_speakers={"A", "B"},
        activity_sources={"mono-sortformer"},
    )

    assert len(examples) == 1
    assert examples[0]["target_speaker"] == "A"
    assert examples[0]["active_speakers"] == ["A"]


def test_build_cutset_examples_prefers_preserved_transcript_spans() -> None:
    cut = {
        "id": "session_1_w000000_c000000-mask-clean",
        "duration": 30.0,
        "recording": {"sources": [{"source": "mono.wav"}]},
        "supervisions": [
            {"speaker": "A", "start": 0.0, "duration": 0.8, "text": "all the words"},
            {"speaker": "A", "start": 1.0, "duration": 0.8, "text": ""},
        ],
        "custom": {
            "activity_mask_source": "isolated-track-teacher",
            "transcript_spans": [
                {"speaker": "A", "start": 0.0, "end": 5.0, "text": "all the words"}
            ],
        },
    }

    examples = build_cutset_examples(
        [cut],
        allowed_speakers={"A"},
        activity_sources={"isolated-track-teacher"},
    )

    assert examples[0]["spans"][0]["end"] == 0.8
    assert examples[0]["transcript_spans"][0]["end"] == 5.0
    assert target_timestamped_text(examples[0]["transcript_spans"], "A") == (
        "<|0.00|>all the words<|5.00|>"
    )


def test_enrollment_index_deduplicates_rows_and_cross_session_choice() -> None:
    base = {
        "split_id": "train",
        "speaker_id": "A",
        "target_member": "a.ogg",
        "positive_enrollment_spans": [
            {"start": 1.0, "duration": 2.0},
        ],
    }
    rows = [
        {**base, "session": "Session 1"},
        {**base, "session": "Session 1"},
        {**base, "session": "Session 2"},
    ]

    indexed = enrollment_index(rows, split="train")
    selected = choose_enrollment_row(
        indexed["A"],
        exclude_session="Session 1",
        rng=__import__("random").Random(3),
    )

    assert len(indexed["A"]) == 2
    assert selected["session"] == "Session 2"


def test_wrong_enrollment_negative_keeps_active_mask_speaker() -> None:
    example = {
        "target_speaker": "A",
        "active_speakers": ["A", "B"],
    }

    activity, enrollment, transcript, negative_type = sample_training_assignment(
        example,
        roster={"A", "B", "C"},
        positive_transcript="hello",
        absent_probability=0.0,
        wrong_enrollment_probability=1.0,
        rng=__import__("random").Random(3),
    )

    assert activity == "A"
    assert enrollment == "C"
    assert transcript == ""
    assert negative_type == "wrong-enrollment-active-mask"


def test_wrong_enrollment_ranking_prefers_speaker_absent_from_mixture() -> None:
    wrong = choose_wrong_enrollment_speaker(
        {"target_speaker": "A", "active_speakers": ["A", "B"]},
        roster={"A", "B", "C"},
        rng=__import__("random").Random(3),
    )

    assert wrong == "C"


def test_enrollment_ranking_loss_pushes_only_wrong_transcript_nll_up() -> None:
    positive = torch.tensor(1.0, requires_grad=True)
    wrong = torch.tensor(1.1, requires_grad=True)

    loss = enrollment_ranking_loss(positive, wrong, margin=0.5)
    loss.backward()

    assert positive.grad is None
    assert wrong.grad is not None
    assert float(wrong.grad) < 0.0
