from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from moss_target_speaker_conditioning import install_target_speaker_conditioning  # noqa: E402
from train_moss_diarization import target_activity_bce_loss  # noqa: E402


class _Adaptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))


def test_target_conditioning_masks_enrollment_and_emits_frame_logits() -> None:
    model = SimpleNamespace(
        model=SimpleNamespace(vq_adaptor=_Adaptor()),
        config=SimpleNamespace(),
    )
    wrapper = install_target_speaker_conditioning(
        model,
        profile_seconds=0.16,
        gap_seconds=0.08,
        frame_hz=12.5,
        attention_heads=2,
    )
    values = torch.randn(1, 7, 8)

    output = wrapper(values)

    assert output.shape == values.shape
    assert wrapper.profile_tokens == 2
    assert wrapper.context_tokens == 3
    torch.testing.assert_close(output[:, 0], output[:, 1])
    assert wrapper.last_target_activity_logits[0].shape == (1, 7, 1)
    assert model.config.target_speaker_conditioning is True


def test_target_activity_loss_rewards_correct_enrollment_activity() -> None:
    targets = torch.tensor([[[0.0], [1.0], [0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    correct = [torch.tensor([[[-8.0], [8.0], [-8.0]]])]
    wrong = [torch.tensor([[[8.0], [-8.0], [8.0]]])]

    assert target_activity_bce_loss(correct, targets, mask) < target_activity_bce_loss(
        wrong, targets, mask
    )
