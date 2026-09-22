from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from moss_activity_conditioning import (  # noqa: E402
    install_activity_conditioning,
    invariant_activity_probabilities,
)


class _Adaptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


def test_activity_conditioning_is_identity_at_initialization() -> None:
    model = SimpleNamespace(
        model=SimpleNamespace(vq_adaptor=_Adaptor()),
        config=SimpleNamespace(),
    )
    values = torch.randn(2, 3, 4)
    expected = model.model.vq_adaptor.layers(values)

    wrapper = install_activity_conditioning(model, max_speakers=3, version=2)
    actual = wrapper(values)

    torch.testing.assert_close(actual, expected)
    assert wrapper.last_activity_logits[0].shape == (2, 3, 3)
    assert wrapper.last_overlap_logits[0].shape == (2, 3, 1)
    assert model.config.activity_conditioning is True
    assert model.config.activity_conditioning_version == 2


def test_version_three_uses_direct_invariant_heads() -> None:
    model = SimpleNamespace(
        model=SimpleNamespace(vq_adaptor=_Adaptor()),
        config=SimpleNamespace(),
    )
    values = torch.randn(2, 3, 4)
    expected = model.model.vq_adaptor.layers(values)

    wrapper = install_activity_conditioning(model, max_speakers=3, version=3)
    actual = wrapper(values)

    torch.testing.assert_close(actual, expected)
    assert wrapper.last_speech_logits[0].shape == (2, 3, 1)
    assert wrapper.last_overlap_logits[0].shape == (2, 3, 1)
    assert model.config.activity_conditioning_version == 3


def test_invariant_probabilities_report_two_confident_speakers_as_overlap() -> None:
    logits = torch.tensor([[8.0, 8.0, -8.0], [8.0, -8.0, -8.0]])

    probabilities = invariant_activity_probabilities(logits)

    assert probabilities["overlap"][0] > 0.99
    assert probabilities["overlap"][1] < 0.01
