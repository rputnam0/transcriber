from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_sequence_tsvad_word_owner_baseline import (  # noqa: E402
    SequenceTsvadModel,
    _sample_chunk_starts,
    _slice_or_pad,
)


def test_slice_or_pad_extends_short_sequences():
    values = np.arange(6, dtype=np.float32).reshape(3, 2)

    chunk = _slice_or_pad(values, start=1, frames=4)

    assert chunk.shape == (4, 2)
    assert np.allclose(chunk[:2], values[1:])
    assert np.allclose(chunk[2:], 0.0)


def test_sample_chunk_starts_prefers_positive_frames_when_requested():
    labels = np.zeros(100, dtype=np.float32)
    labels[50] = 1.0
    starts = _sample_chunk_starts(
        labels,
        chunk_frames=20,
        chunks_per_sequence=10,
        positive_probability=1.0,
        rng=random.Random(7),
    )

    assert len(starts) == 10
    assert all(31 <= start <= 50 for start in starts)


def test_sequence_tsvad_model_preserves_frame_count():
    model = SequenceTsvadModel(input_dim=12, channels=16, layers=3, dropout=0.0)
    batch = torch.randn(2, 25, 12)

    logits = model(batch)

    assert logits.shape == (2, 25)
