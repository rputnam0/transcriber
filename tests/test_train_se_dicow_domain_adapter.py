from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT_PATH = SCRIPTS_DIR / "train_se_dicow_domain_adapter.py"
SPEC = importlib.util.spec_from_file_location("train_se_dicow_domain_adapter", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_target_transcript_emits_target_only_timestamp_segments() -> None:
    words = [
        {"speaker": "A", "start": 30.10, "end": 30.30, "token": "hello"},
        {"speaker": "B", "start": 30.20, "end": 30.40, "token": "wrong"},
        {"speaker": "A", "start": 30.32, "end": 30.60, "token": "there"},
        {"speaker": "A", "start": 32.00, "end": 32.20, "token": "again"},
    ]

    transcript = MODULE.build_target_transcript(words, speaker="A", chunk_start=30.0)

    assert transcript == "<|0.10|>hello there<|0.60|><|2.00|>again<|2.20|>"
    assert "wrong" not in transcript


def test_build_target_transcript_is_empty_for_absent_target() -> None:
    transcript = MODULE.build_target_transcript(
        [{"speaker": "B", "start": 0.0, "end": 0.2, "token": "hello"}],
        speaker="A",
        chunk_start=0.0,
    )

    assert transcript == ""


def test_build_training_examples_includes_absent_candidate_passes() -> None:
    rows = [
        {
            "session": "Session 1",
            "window_start": 0.0,
            "window_end": 30.0,
            "duration": 30.0,
            "split_id": "train",
            "speaker_id": speaker,
        }
        for speaker in ("A", "B")
    ]
    references = {
        ("Session 1", 0.0, 30.0): {
            "words": [
                {"speaker": "A", "start": 0.0, "end": 0.2, "text": "hello"},
            ]
        }
    }

    examples = MODULE.build_training_examples(rows, references, split="train")

    assert len(examples) == 2
    assert {example["speaker"]: example["target_word_count"] for example in examples} == {
        "A": 1,
        "B": 0,
    }


def test_positive_enrollment_filter_accepts_spans_or_materialized_paths() -> None:
    assert MODULE.has_positive_enrollment(
        {"row": {"positive_enrollment_spans": [{"start": 1.0, "end": 2.0}]}}
    )
    assert MODULE.has_positive_enrollment(
        {"row": {"materialized": {"positive_enrollment_paths": ["enroll.wav"]}}}
    )
    assert not MODULE.has_positive_enrollment({"row": {}})


def test_strip_peft_prefix_produces_base_model_key() -> None:
    name = "base_model.model.model.encoder.fddts.0.target_linear.weight"

    assert MODULE.strip_peft_prefix(name) == "model.encoder.fddts.0.target_linear.weight"


def test_session_stem_dir_matches_materializer_layout(tmp_path: Path) -> None:
    path = MODULE._session_stem_dir(tmp_path, "Session 49")

    assert path == tmp_path / "_stems" / "session_49"


def test_strip_peft_prefix_handles_full_scb_parameter() -> None:
    name = "base_model.model.model.encoder.ca_enrolls.0.cae.ffn.0.weight"

    assert MODULE.strip_peft_prefix(name) == "model.encoder.ca_enrolls.0.cae.ffn.0.weight"


def test_strip_peft_prefix_unwraps_lora_base_layer() -> None:
    name = "base_model.model.model.encoder.ca_enrolls.0.cae.cross_attn.q_proj." "base_layer.weight"

    assert MODULE.strip_peft_prefix(name) == (
        "model.encoder.ca_enrolls.0.cae.cross_attn.q_proj.weight"
    )


def test_promote_trainable_parameters_to_float32_leaves_frozen_weights_alone() -> None:
    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, dtype=torch.bfloat16),
        torch.nn.Linear(2, 2, dtype=torch.bfloat16),
    )
    for parameter in model[1].parameters():
        parameter.requires_grad = False

    promoted = MODULE.promote_trainable_parameters_to_float32(model)

    assert promoted == sum(parameter.numel() for parameter in model[0].parameters())
    assert {parameter.dtype for parameter in model[0].parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in model[1].parameters()} == {torch.bfloat16}


def test_activity_labels_preserve_target_non_target_and_overlap() -> None:
    import numpy as np

    stno = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    labels = MODULE.activity_labels_from_stno(stno)

    np.testing.assert_array_equal(
        labels,
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )


def test_multilabel_activity_loss_rewards_correct_logits() -> None:
    import torch

    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    correct = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])
    inverted = -correct

    assert MODULE.multilabel_activity_loss(correct, labels) < MODULE.multilabel_activity_loss(
        inverted,
        labels,
    )
