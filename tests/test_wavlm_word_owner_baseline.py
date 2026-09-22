from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_wavlm_word_owner_baseline import (  # noqa: E402
    _apply_projection,
    _frame_times,
    _profile_from_feature_chunks,
    _projection_matrix,
)


def test_frame_times_cover_chunk_midpoints():
    times = _frame_times(10.0, 2.0, 4)

    assert np.allclose(times, np.array([10.25, 10.75, 11.25, 11.75], dtype=np.float32))


def test_projection_matrix_is_deterministic_and_shapes_features():
    projection = _projection_matrix(input_dim=4, output_dim=2, seed=11)
    same_projection = _projection_matrix(input_dim=4, output_dim=2, seed=11)
    features = np.ones((3, 4), dtype=np.float32)

    projected = _apply_projection(features, projection)

    assert projection is not None
    assert projected.shape == (3, 2)
    assert np.allclose(projection, same_projection)


def test_profile_from_feature_chunks_returns_mean_and_std():
    first = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.array([[5.0, 6.0]], dtype=np.float32)

    profile = _profile_from_feature_chunks([first, second], output_dim=2)

    assert profile.shape == (4,)
    assert np.allclose(profile[:2], np.array([3.0, 4.0], dtype=np.float32))
    assert np.all(profile[2:] > 0.0)
