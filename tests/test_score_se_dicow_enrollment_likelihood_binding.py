from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_se_dicow_enrollment_likelihood_binding import (  # noqa: E402
    per_sequence_cross_entropy,
    rank_enrollments,
)


def test_per_sequence_cross_entropy_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0]],
            [[0.0, 4.0], [4.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, -100], [0, 1]])

    losses = per_sequence_cross_entropy(logits, labels)

    assert float(losses[0]) < float(losses[1])


def test_rank_enrollments_reports_truth_margin_and_rank() -> None:
    ranked = rank_enrollments([0.9, 0.2, 0.5], ["A", "B", "C"], truth="C")

    assert ranked["predicted_speaker"] == "B"
    assert ranked["truth_rank"] == 2
    assert ranked["truth_margin"] == -0.3
