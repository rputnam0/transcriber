"""Prevent reference leakage and unstable identity targets in local MOSS training."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_moss_mac_training import (
    TRAIN,
    DEV,
    TEST,
    authored_record,
    training_duration,
    mono_window,
)  # noqa: E402


def test_splits_are_session_disjoint():
    assert not (set(TRAIN) & set(DEV) or set(TRAIN) & set(TEST) or set(DEV) & set(TEST))


def test_short_participant_track_does_not_truncate_session_or_misalign_mix():
    meta = {"tracks": {"a": {"duration": 10}, "b": {"duration": 9800}}}
    assert training_duration(meta) == 9800
    waves = {"a": np.ones(16000, np.float32), "b": np.full(48000, 2, np.float32)}
    assert np.array_equal(mono_window(waves, 0.5, 1), np.r_[np.full(8000, 3), np.full(8000, 2)])


def test_identity_targets_do_not_change_when_speaking_order_changes(tmp_path):
    segments = [
        dict(speaker="kinglizard7958", start=0, end=1, words=["first"]),
        dict(speaker="bfschmity", start=2, end=3, words=["second"]),
    ]
    record = authored_record("cut", tmp_path / "audio.wav", segments, "43", "identity")
    mapping = record["metadata"]["stable_session_speaker_ids"]
    assert mapping == {"kinglizard7958": "S04", "bfschmity": "S01"}
    onset = authored_record("cut", tmp_path / "audio.wav", segments, "43", "onset")
    assert onset["metadata"]["stable_session_speaker_ids"]["kinglizard7958"] == "S01"
