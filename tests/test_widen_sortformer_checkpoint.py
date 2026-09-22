from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from widen_sortformer_checkpoint import transplant_state  # noqa: E402


def test_transplant_state_copies_equal_shapes_and_widens_first_dimension():
    source_state = {
        "encoder.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "sortformer_modules.hidden_to_spks.weight": torch.arange(10, dtype=torch.float32).view(
            5, 2
        ),
        "sortformer_modules.hidden_to_spks.bias": torch.arange(5, dtype=torch.float32),
    }
    target_state = {
        "encoder.weight": torch.zeros(2, 2),
        "sortformer_modules.hidden_to_spks.weight": torch.full((8, 2), -1.0),
        "sortformer_modules.hidden_to_spks.bias": torch.full((8,), -2.0),
    }

    output_state, summary = transplant_state(source_state, target_state)

    assert torch.equal(output_state["encoder.weight"], source_state["encoder.weight"])
    assert torch.equal(
        output_state["sortformer_modules.hidden_to_spks.weight"][:5],
        source_state["sortformer_modules.hidden_to_spks.weight"],
    )
    assert torch.equal(
        output_state["sortformer_modules.hidden_to_spks.weight"][5:],
        target_state["sortformer_modules.hidden_to_spks.weight"][5:],
    )
    assert torch.equal(
        output_state["sortformer_modules.hidden_to_spks.bias"][:5],
        source_state["sortformer_modules.hidden_to_spks.bias"],
    )
    assert torch.equal(
        output_state["sortformer_modules.hidden_to_spks.bias"][5:],
        target_state["sortformer_modules.hidden_to_spks.bias"][5:],
    )
    assert summary.exact_copied == ["encoder.weight"]
    assert [item["key"] for item in summary.widened] == [
        "sortformer_modules.hidden_to_spks.weight",
        "sortformer_modules.hidden_to_spks.bias",
    ]


def test_transplant_state_keeps_target_for_missing_and_incompatible_shapes():
    source_state = {
        "missing.target": torch.ones(2, 2),
        "shape.mismatch": torch.ones(3, 4),
    }
    target_state = {
        "target.only": torch.full((2, 2), 3.0),
        "shape.mismatch": torch.full((4, 5), 7.0),
    }

    output_state, summary = transplant_state(source_state, target_state)

    assert torch.equal(output_state["target.only"], target_state["target.only"])
    assert torch.equal(output_state["shape.mismatch"], target_state["shape.mismatch"])
    assert summary.missing_in_source == ["target.only"]
    assert summary.shape_mismatches == [
        {"key": "shape.mismatch", "source_shape": [3, 4], "target_shape": [4, 5]}
    ]
