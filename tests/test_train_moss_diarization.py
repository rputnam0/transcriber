from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_moss_diarization import (  # noqa: E402
    activity_bce_loss,
    load_samples,
    loss_weights_from_offsets,
    promote_trainable_parameters_to_float32,
    weighted_causal_loss,
)


def _row(audio: Path, *, session: str, mono_input_only: bool = True) -> dict:
    return {
        "conversation": [
            {"role": "user", "message_type": "text", "content": "Transcribe."},
            {"role": "user", "message_type": "audio", "content": str(audio)},
            {"role": "assistant", "message_type": "text", "content": "[0.00][S01] hi[1.00]"},
        ],
        "metadata": {"session": session, "mono_input_only": mono_input_only},
    }


def test_load_samples_accepts_declared_mono_training_audio(tmp_path: Path) -> None:
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"fixture")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(_row(audio, session="Session 49")) + "\n")

    samples = load_samples(str(manifest), forbidden_sessions={"Session 34"})

    assert samples[0]["audio"] == str(audio)
    assert samples[0]["session"] == "Session 49"


@pytest.mark.parametrize(
    ("session", "mono_input_only", "match"),
    [
        ("Session 34", True, "forbidden holdout"),
        ("Session 49", False, "not declared mono-only"),
    ],
)
def test_load_samples_rejects_leakage(
    tmp_path: Path,
    session: str,
    mono_input_only: bool,
    match: str,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(_row(audio, session=session, mono_input_only=mono_input_only)) + "\n"
    )

    with pytest.raises(ValueError, match=match):
        load_samples(str(manifest), forbidden_sessions={"Session 34"})


def test_character_spans_weight_brief_overlap_tokens() -> None:
    weights = loss_weights_from_offsets(
        [(0, 5), (6, 9), (10, 13)],
        [{"start": 6, "end": 9, "weight": 4.0}],
    )

    assert weights == [1.0, 4.0, 1.0]


def test_explicit_timestamp_downweight_is_not_silently_clamped_to_one() -> None:
    assert loss_weights_from_offsets(
        [(0, 5), (5, 10), (10, 13)],
        [{"start": 0, "end": 5, "weight": 0.2}, {"start": 5, "end": 10, "weight": 0.0}],
    ) == [0.2, 0.0, 1.0]


def test_synthetic_secondary_source_cannot_leak_holdout(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    row = _row(audio, session="43")
    row["metadata"]["source_sessions"] = [43, 37]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="forbidden holdout"):
        load_samples(str(manifest), forbidden_sessions={"Session 37"})


def test_weighted_causal_loss_emphasizes_selected_label() -> None:
    logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]]])
    labels = torch.tensor([[-100, 0, 1]])
    ordinary = weighted_causal_loss(logits, labels, torch.ones_like(labels, dtype=torch.float32))
    weighted = weighted_causal_loss(
        logits,
        labels,
        torch.tensor([[0.0, 1.0, 4.0]]),
    )

    assert weighted > ordinary


def test_masked_padding_preserves_weighted_loss_and_valid_token_gradients() -> None:
    logits = torch.tensor([[[3.0, 0.0], [1.0, 2.0], [2.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 0, 1]])
    weights = torch.tensor([[0.0, 1.0, 4.0]])
    loss = weighted_causal_loss(logits, labels, weights)
    grad = torch.autograd.grad(loss, logits)[0]
    padded = torch.cat([logits.detach(), torch.zeros(1, 5, 2)], dim=1).requires_grad_(True)
    padded_loss = weighted_causal_loss(
        padded,
        torch.nn.functional.pad(labels, (0, 5), value=-100),
        torch.nn.functional.pad(weights, (0, 5), value=0),
    )
    padded_grad = torch.autograd.grad(padded_loss, padded)[0]
    torch.testing.assert_close(loss, padded_loss)
    torch.testing.assert_close(grad, padded_grad[:, :3])
    assert not padded_grad[:, 3:].any()


def test_activity_loss_upweights_overlap_frames() -> None:
    logits = [torch.zeros(1, 2, 2)]
    overlap_logits = [torch.zeros(1, 2, 1)]
    targets = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    loss = activity_bce_loss(logits, overlap_logits, targets, mask)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_activity_loss_is_invariant_to_speaker_slot_permutation() -> None:
    logits = [torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]])]
    overlap_logits = [torch.full((1, 2, 1), -8.0)]
    speech_logits = [torch.full((1, 2, 1), 8.0)]
    targets = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    permuted = activity_bce_loss(
        logits,
        overlap_logits,
        targets,
        mask,
        speech_logits,
    )
    aligned = activity_bce_loss(
        logits,
        overlap_logits,
        targets.flip(-1),
        mask,
        speech_logits,
    )

    torch.testing.assert_close(permuted, aligned)


def test_promotes_only_trainable_parameters_to_float32() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)).to(torch.bfloat16)
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    summary = promote_trainable_parameters_to_float32(model)

    assert summary["promoted_parameters"] == 2
    assert summary["promoted_elements"] == 3
    assert next(model[0].parameters()).dtype == torch.bfloat16
    assert next(model[1].parameters()).dtype == torch.float32
