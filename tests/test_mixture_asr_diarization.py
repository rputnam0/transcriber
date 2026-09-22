from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_mixture_asr_diarization import (  # noqa: E402
    _aggregate,
    _manifest_group_key,
    _score_group,
)


def test_manifest_group_key_parses_sortformer_audio_path():
    row = {
        "audio_filepath": (
            "/tmp/outputs/sortformer_nemo_export_devtest_v1/audio/"
            "session_64_004800000_005100000.wav"
        )
    }

    assert _manifest_group_key(row) == ("Session 64", 4800.0, 5100.0)


def test_score_group_counts_lexical_coverage_and_oracle_diarization_bound():
    diarization_words = [
        {
            "speaker": "Alice",
            "cluster": "speaker_0",
            "start": 0.0,
            "end": 0.3,
            "text": "hello",
            "duration": 0.3,
            "score": 0.9,
            "overlap": False,
        },
        {
            "speaker": "Bob",
            "cluster": "speaker_1",
            "start": 0.4,
            "end": 0.7,
            "text": "dragon",
            "duration": 0.3,
            "score": 0.8,
            "overlap": True,
        },
        {
            "speaker": "Alice",
            "cluster": "speaker_0",
            "start": 0.8,
            "end": 1.1,
            "text": "again",
            "duration": 0.3,
            "score": 0.01,
            "overlap": False,
        },
    ]
    asr_words = [
        {"word": "hello", "start": 0.0, "end": 0.2},
        {"word": "wrong", "start": 0.2, "end": 0.4},
        {"word": "dragon", "start": 0.4, "end": 0.7},
        {"word": "again", "start": 0.8, "end": 1.1},
    ]

    score = _score_group(
        key=("Session 1", 0.0, 10.0),
        diarization_words=diarization_words,
        asr_words=asr_words,
    )

    assert score["reference_words"] == 3
    assert score["predicted_words"] == 4
    assert score["lexical_matched_words"] == 3
    assert score["many_to_one_oracle_diarization_accuracy"] == 1.0
    assert score["many_to_one_accuracy"] == 1.0
    assert score["many_to_one_prediction_precision_proxy"] == 0.75
    assert score["many_to_one_overlap_accuracy"] == 1.0
    assert score["quality_filters"]["score_ge_0_05"]["words"] == 2
    assert score["quality_filters"]["score_ge_0_05"]["many_to_one_accuracy"] == 1.0


def test_score_group_wrong_or_missing_asr_words_reduce_primary_accuracy():
    diarization_words = [
        {"speaker": "Alice", "cluster": "speaker_0", "text": "hello", "overlap": False},
        {"speaker": "Bob", "cluster": "speaker_1", "text": "dragon", "overlap": False},
    ]
    asr_words = [{"word": "hello", "start": 0.0, "end": 0.2}]

    score = _score_group(
        key=("Session 1", 0.0, 10.0),
        diarization_words=diarization_words,
        asr_words=asr_words,
    )
    summary = _aggregate([score])

    assert score["lexical_coverage"] == 0.5
    assert score["many_to_one_accuracy"] == 0.5
    assert summary["many_to_one_accuracy"] == 0.5
    assert summary["many_to_one_oracle_diarization_accuracy"] == 1.0
    assert summary["quality_filters"]["all"]["lexical_coverage"] == 0.5
