from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_streaming_usef_tp_word_owner import StreamingUsefTpActivity  # noqa: E402


class FakeUsefTp(nn.Module):
    def forward(
        self, mixture: torch.Tensor, enrollment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del enrollment
        logits = mixture[:, None, ::2]
        return mixture, logits


class FakeStft(nn.Module):
    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor]:
        return (wave.unsqueeze(2).to(torch.complex64),)


class FakeCrossAttention(nn.Module):
    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        return mixture + enrollment


class FakePvadDecoder(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(1, 3), keepdim=False).unsqueeze(1)


class FakeModularUsefTp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stft = FakeStft()
        self.encoder = nn.Identity()
        self.cmha = FakeCrossAttention()
        self.separator = nn.Identity()
        self.pvad_decoder = FakePvadDecoder()

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor):
        del mixture, enrollment
        raise AssertionError("activity-only inference should skip waveform reconstruction")


def test_activity_wrapper_returns_personal_vad_logits() -> None:
    activity = StreamingUsefTpActivity(FakeUsefTp())
    mixture = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    logits = activity(mixture, torch.zeros_like(mixture))

    assert logits.shape == (2, 3)
    assert torch.equal(logits, mixture[:, ::2])
    assert activity.frame_hop_seconds == 0.008


def test_activity_wrapper_skips_waveform_decoder_for_modular_model() -> None:
    activity = StreamingUsefTpActivity(FakeModularUsefTp())
    mixture = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    logits = activity(mixture, torch.ones_like(mixture))

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
