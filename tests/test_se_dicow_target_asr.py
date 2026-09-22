from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_se_dicow_target_asr.py"
SPEC = importlib.util.spec_from_file_location("run_se_dicow_target_asr", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_path_finds_repo_relative_file_from_manifest_ancestor(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "outputs" / "experiment"
    audio_path = tmp_path / "outputs" / "experiment" / "materialized" / "mixture.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    resolved = MODULE._resolve_path(
        "outputs/experiment/materialized/mixture.wav",
        manifest_dir=manifest_dir,
    )

    assert resolved == audio_path


def test_build_stno_mask_preserves_all_four_activity_classes() -> None:
    words = [
        {"speaker": "A", "start": 1.0, "end": 2.0},
        {"speaker": "B", "start": 1.5, "end": 2.5},
    ]

    mask = MODULE.build_stno_mask(
        words,
        speaker="A",
        chunk_start=0.0,
        chunk_seconds=3.0,
        frame_rate=10,
        collar_seconds=0.0,
    )

    assert mask.shape == (4, 30)
    np.testing.assert_array_equal(mask.sum(axis=0), np.ones(30))
    assert mask[0, 0] == 1.0
    assert mask[1, 11] == 1.0
    assert mask[3, 16] == 1.0
    assert mask[2, 21] == 1.0


def test_vad_stno_marks_padding_as_silence() -> None:
    samples = np.zeros(1_000, dtype=np.float32)
    samples[100:400] = 0.5

    mask = MODULE.build_vad_stno_mask(
        samples,
        sample_rate=1_000,
        chunk_seconds=2.0,
        frame_rate=10,
    )

    assert mask.shape == (4, 20)
    np.testing.assert_array_equal(mask.sum(axis=0), np.ones(20))
    assert mask[1].sum() > 0
    assert mask[0, -1] == 1.0


def test_activity_probabilities_preserve_predicted_overlap() -> None:
    probabilities = np.array(
        [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]],
        dtype=np.float32,
    )

    mask = MODULE.build_stno_from_activity_probabilities(
        probabilities,
        mixture_active=np.array([False, True, True, True]),
    )

    np.testing.assert_array_equal(mask.argmax(axis=0), np.array([0, 1, 2, 3]))


def test_activity_score_reports_precision_recall_and_ambiguity() -> None:
    oracle = np.eye(4, dtype=np.float32)
    probabilities = np.array(
        [[0.1, 0.9], [0.9, 0.1], [0.5, 0.9], [0.9, 0.9]],
        dtype=np.float32,
    )

    score = MODULE.score_activity_probabilities(probabilities, oracle)

    assert score["target_frame_recall"] == 1.0
    assert score["non_target_frame_recall"] == 1.0
    assert score["ambiguous_frame_fraction"] == 0.25


def test_score_candidate_reports_target_recall_overlap_and_leakage() -> None:
    words = [
        {"speaker": "A", "start": 0.0, "end": 0.5, "token": "hello"},
        {"speaker": "A", "start": 1.0, "end": 1.5, "token": "there"},
        {"speaker": "B", "start": 1.2, "end": 1.7, "token": "wrong"},
    ]

    score = MODULE.score_candidate(
        reference_words=words,
        speaker="A",
        predicted_text="hello there wrong",
    )

    assert score["matched_target_words"] == 2
    assert score["target_recall"] == 1.0
    assert score["target_precision"] == 2 / 3
    assert score["overlap_reference_words"] == 1
    assert score["overlap_recall"] == 1.0
    assert score["wrong_speaker_match_proxy"] == 1


def test_score_candidate_recovers_timestamp_boundary_word_concatenation() -> None:
    words = [
        {"speaker": "A", "start": 0.0, "end": 0.2, "token": "what"},
        {"speaker": "A", "start": 0.2, "end": 0.4, "token": "i"},
        {"speaker": "A", "start": 0.4, "end": 0.6, "token": "said"},
    ]

    score = MODULE.score_candidate(
        reference_words=words,
        speaker="A",
        predicted_text="whati said",
    )

    assert score["matched_target_words"] == 3
    assert score["matched_prediction_units"] == 2
    assert score["target_recall"] == 1.0
    assert score["target_precision"] == 1.0


def test_score_candidate_penalizes_words_for_absent_target() -> None:
    score = MODULE.score_candidate(
        reference_words=[{"speaker": "B", "start": 0.0, "end": 0.5, "token": "hello"}],
        speaker="A",
        predicted_text="hello again",
    )

    assert score["reference_words"] == 0
    assert score["predicted_words"] == 2
    assert score["no_speech_false_positive_words"] == 2
    assert score["target_precision"] == 0.0


def test_clip_words_assigns_boundary_word_by_midpoint_once() -> None:
    words = [
        {"speaker": "A", "start": 29.9, "end": 30.1, "text": "boundary"},
        {"speaker": "A", "start": 30.1, "end": 30.2, "text": "later"},
    ]

    first = MODULE._clip_words(words, chunk_start=0.0, chunk_seconds=30.0)
    second = MODULE._clip_words(words, chunk_start=30.0, chunk_seconds=30.0)

    assert [word["token"] for word in first] == []
    assert [word["token"] for word in second] == ["boundary", "later"]


def test_aggregate_scores_uses_word_weighted_metrics() -> None:
    summary = MODULE.aggregate_scores(
        [
            {
                "reference_words": 3,
                "predicted_words": 4,
                "matched_target_words": 2,
                "matched_prediction_units": 1,
                "wrong_speaker_match_proxy": 1,
                "overlap_reference_words": 2,
                "overlap_matched_words": 1,
                "no_speech_false_positive_words": 0,
            },
            {
                "reference_words": 1,
                "predicted_words": 1,
                "matched_target_words": 1,
                "matched_prediction_units": 1,
                "wrong_speaker_match_proxy": 0,
                "overlap_reference_words": 0,
                "overlap_matched_words": 0,
                "no_speech_false_positive_words": 0,
            },
        ]
    )

    assert summary["speaker_attributed_word_recall"] == 0.75
    assert summary["target_prediction_precision"] == 0.4
    assert summary["overlap_word_recall"] == 0.5


def test_select_max_energy_window_skips_quiet_prefix() -> None:
    samples = np.zeros(4_000, dtype=np.float32)
    samples[2_500:3_500] = 0.5

    selected = MODULE.select_max_energy_window(samples, frames=1_000)

    assert np.sqrt(np.mean(selected**2)) > 0.45


def test_decode_timestamped_text_preserves_word_boundaries() -> None:
    class Tokenizer:
        @staticmethod
        def batch_decode(_sequences, *, skip_special_tokens):
            assert not skip_special_tokens
            return ["<|en|><|0.00|>what<|0.20|> i<|0.40|> said<|eot|>"]

    text = MODULE._decode_timestamped_text(Tokenizer(), [[1, 2, 3]])

    assert text == "what i said"
