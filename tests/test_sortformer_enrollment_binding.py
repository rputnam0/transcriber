from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_sortformer_enrollment_binding import (  # noqa: E402
    assignment_maps,
    _load_wave,
    enrollment_leave_one_out_score,
    enrollment_provenance,
    slot_waveforms,
    trim_clean_enrollment,
)


def test_load_wave_honors_cut_offset_in_shared_recording(tmp_path: Path) -> None:
    path = tmp_path / "shared.wav"
    sf.write(path, np.concatenate((np.zeros(80), np.ones(80))), 80, subtype="FLOAT")

    wave = _load_wave(
        path,
        sample_rate=80,
        start_seconds=1.0,
        duration_seconds=1.0,
    )

    assert len(wave) == 80
    assert np.allclose(wave, 1.0)


def test_slot_waveforms_keep_exclusive_mono_regions() -> None:
    mixture = np.arange(32, dtype=np.float32)
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.7],
            [0.1, 0.9],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    slots, waves, metadata = slot_waveforms(
        mixture,
        probabilities,
        sample_rate=8,
        activity_threshold=0.5,
        exclusivity_margin=0.2,
        min_slot_seconds=0.5,
        min_wave_seconds=0.5,
    )

    assert slots == ["speaker_0", "speaker_1"]
    assert np.array_equal(waves[0], mixture[:8])
    assert np.array_equal(waves[1], mixture[16:24])
    assert metadata[0]["activity_seconds"] == 2.0


def test_assignment_maps_support_independent_and_one_to_one_binding() -> None:
    similarity = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.7],
        ],
        dtype=np.float32,
    )

    independent, one_to_one, evidence = assignment_maps(
        similarity,
        ["speaker_0", "speaker_1"],
        ["Alice", "Bob"],
    )

    assert independent == {"speaker_0": "Alice", "speaker_1": "Alice"}
    assert one_to_one == {"speaker_0": "Alice", "speaker_1": "Bob"}
    assert np.isclose(evidence["speaker_0"]["margin"], 0.8)


def test_trim_clean_enrollment_removes_isolated_track_silence() -> None:
    wave = np.concatenate(
        (
            np.zeros(80, dtype=np.float32),
            np.ones(80, dtype=np.float32) * 0.2,
            np.zeros(80, dtype=np.float32),
        )
    )

    trimmed = trim_clean_enrollment(
        wave,
        sample_rate=80,
        frame_seconds=0.25,
        minimum_seconds=0.5,
    )

    assert len(trimmed) == 80
    assert np.allclose(trimmed, 0.2)


def test_enrollment_leave_one_out_score_separates_clean_clusters() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    score = enrollment_leave_one_out_score(
        embeddings,
        ["Alice", "Alice", "Bob", "Bob"],
    )

    assert score["accuracy"] == 1.0


def test_enrollment_provenance_marks_cross_session_training_profiles(tmp_path: Path) -> None:
    manifest = tmp_path / "enrollment.jsonl"
    manifest.write_text(
        '{"session":"Session 64","enrollment_source_split":"train",'
        '"enrollment_source_sessions":["Session 41","Session 44"],'
        '"uses_evaluation_session_audio":false}\n',
        encoding="utf-8",
    )

    provenance = enrollment_provenance(manifest, session="Session 64")

    assert provenance == {
        "source": "cross-session-training-profile",
        "source_splits": ["train"],
        "source_sessions": ["Session 41", "Session 44"],
        "uses_evaluation_session_audio": False,
    }
