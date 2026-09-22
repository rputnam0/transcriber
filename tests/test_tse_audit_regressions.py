from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_usef_tse_manifest import (  # noqa: E402
    _concat_enrollments,
    _row_is_selected,
    _stitch_window,
)
from score_speaker_tse_manifest import _match_length  # noqa: E402
from score_tse_speaker_attributed_asr import _cache_identity, _cache_path  # noqa: E402


def test_match_length_pads_shorter_signal_instead_of_truncating():
    reference = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    estimate = np.array([1.0, 2.0], dtype=np.float32)

    matched_reference, matched_estimate = _match_length(reference, estimate)

    assert matched_reference.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert matched_estimate.tolist() == [1.0, 2.0, 0.0, 0.0]


def test_asr_cache_identity_changes_with_model_and_decode_config(tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"not really wav, but enough identity bytes")
    row = {"row_id": "row-1"}

    small_identity = _cache_identity(
        row,
        audio_path,
        model_name="small",
        compute_type="float16",
        device="cuda",
        batch_size=8,
    )
    large_identity = _cache_identity(
        row,
        audio_path,
        model_name="large-v3",
        compute_type="float16",
        device="cuda",
        batch_size=8,
    )

    assert small_identity["decode_options"]["word_timestamps"] is True
    assert small_identity["decode_options"]["vad_filter"] is True
    assert _cache_path(tmp_path, small_identity) != _cache_path(tmp_path, large_identity)


def test_concat_enrollments_resamples_each_clip_from_its_own_rate(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, np.ones(800, dtype=np.float32), 8000)
    sf.write(second, np.ones(1600, dtype=np.float32), 16000)

    merged, sample_rate = _concat_enrollments(
        [Path("first.wav"), Path("second.wav")],
        base_dir=tmp_path,
        max_seconds=1.0,
    )

    assert sample_rate == 8000
    assert merged.numel() == 1600


def test_row_selection_filters_manifest_split_before_max_rows():
    row = {"row_id": "row-1", "split_id": "dev"}

    assert _row_is_selected(
        row,
        row_ids=set(),
        split_ids={"dev"},
        max_rows=1,
        count=0,
    )
    assert not _row_is_selected(
        row,
        row_ids=set(),
        split_ids={"test"},
        max_rows=1,
        count=0,
    )


def test_stitch_window_fades_only_contextual_edges():
    window = _stitch_window(
        8,
        overlap_samples=4,
        has_left_context=True,
        has_right_context=True,
    )

    assert torch.isclose(window[0], torch.tensor(0.0))
    assert torch.isclose(window[3], torch.tensor(1.0))
    assert torch.isclose(window[-1], torch.tensor(0.0))
