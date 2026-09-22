from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_tse_word_ownership import _span_energy, _track_energy_scale  # noqa: E402


def test_track_p95_normalization_is_invariant_to_output_gain():
    wave = np.concatenate(
        [
            np.zeros(800, dtype=np.float32),
            np.ones(800, dtype=np.float32),
            np.zeros(800, dtype=np.float32),
        ]
    )
    louder = wave * 7.0

    energy = _span_energy(wave, sample_rate=8000, start=0.1, end=0.2, min_seconds=0.1)
    louder_energy = _span_energy(
        louder,
        sample_rate=8000,
        start=0.1,
        end=0.2,
        min_seconds=0.1,
    )
    scale = _track_energy_scale(wave, sample_rate=8000, normalization="track-p95")
    louder_scale = _track_energy_scale(
        louder,
        sample_rate=8000,
        normalization="track-p95",
    )

    assert np.isclose(energy / scale, louder_energy / louder_scale)
