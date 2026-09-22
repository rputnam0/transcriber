from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_listwise_word_owner_baseline import (  # noqa: E402
    _CandidateScorer,
    _pad_candidate_items,
    _standardize_candidate_tensor,
)


def test_pad_candidate_items_masks_variable_candidate_counts():
    items = [
        np.ones((2, 3), dtype=np.float32),
        np.full((4, 3), 2.0, dtype=np.float32),
    ]

    features, mask, labels, weights = _pad_candidate_items(items, [1, 3], [0.5, 1.0])

    assert features.shape == (2, 4, 3)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert labels.tolist() == [1, 3]
    assert np.allclose(weights, [0.5, 1.0])
    assert np.allclose(features[0, 2:], 0.0)


def test_standardize_candidate_tensor_ignores_padded_candidates():
    features = np.array(
        [
            [[1.0, 3.0], [2.0, 5.0], [999.0, 999.0]],
            [[3.0, 7.0], [4.0, 9.0], [5.0, 11.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, True, False], [True, True, True]])

    standardized, mean, std = _standardize_candidate_tensor(features, mask)

    assert np.allclose(mean, [3.0, 7.0])
    assert np.allclose(standardized[0, 2], 0.0)
    assert np.all(std > 0.0)


def test_candidate_scorer_returns_one_logit_per_candidate():
    model = _CandidateScorer(input_dim=5, hidden_dim=16, dropout=0.0)
    features = torch.randn(3, 4, 5)

    logits = model(features)

    assert logits.shape == (3, 4)
