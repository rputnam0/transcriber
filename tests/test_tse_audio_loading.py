from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchaudio

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_usef_tse_domain_adapter import _load_chunk, _valid_cached_wav  # noqa: E402


def test_load_chunk_seeks_at_source_rate_before_resampling(tmp_path: Path) -> None:
    sample_rate = 16000
    wave = torch.cat((torch.zeros(sample_rate), torch.full((sample_rate,), 0.5)))
    path = tmp_path / "source.wav"
    torchaudio.save(str(path), wave.unsqueeze(0), sample_rate)

    chunk = _load_chunk(
        path,
        start_seconds=1.0,
        duration_seconds=1.0,
        sample_rate=8000,
    )

    assert chunk.shape == (8000,)
    assert torch.allclose(chunk.mean(), torch.tensor(0.5), atol=1e-3)


def test_valid_cached_wav_checks_requested_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    torchaudio.save(str(path), torch.zeros(1, 2000), 8000)

    assert _valid_cached_wav(path, sample_rate=8000)
    assert not _valid_cached_wav(path, sample_rate=16000)


def test_load_chunk_after_end_returns_silence_without_deleting_cache(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    torchaudio.save(str(path), torch.full((1, 8000), 0.5), 8000)

    chunk = _load_chunk(
        path,
        start_seconds=2.0,
        duration_seconds=1.0,
        sample_rate=8000,
    )

    assert chunk.shape == (8000,)
    assert not torch.any(chunk)
    assert path.exists()
