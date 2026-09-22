from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_streaming_usef_tp_domain_adapter import extraction_loss  # noqa: E402


def test_extraction_loss_prefers_correct_active_source() -> None:
    target = torch.tensor([[0.0, 1.0, -1.0, 0.5, -0.5]])
    mask = torch.ones_like(target)

    correct = extraction_loss(target, target, mask)
    wrong = extraction_loss(torch.flip(target, dims=(-1,)), target, mask)

    assert correct < wrong


def test_extraction_loss_rewards_silence_for_inactive_target() -> None:
    target = torch.zeros(1, 8)
    mask = torch.zeros_like(target)

    silent = extraction_loss(torch.zeros_like(target), target, mask)
    leaking = extraction_loss(torch.ones_like(target), target, mask)

    assert silent < leaking
