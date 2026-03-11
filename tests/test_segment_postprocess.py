from __future__ import annotations

from transcriber.segment_postprocess import smooth_short_speaker_flips


def test_smooth_short_speaker_flips_relabeled_single_short_sandwich():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Alice",
            "words": [{"start": 0.0, "end": 1.0, "word": "hi"}],
        },
        {
            "start": 2.0,
            "end": 2.8,
            "speaker": "Bob",
            "speaker_match_score": 0.2,
            "words": [{"start": 2.0, "end": 2.6, "word": "oops"}],
        },
        {
            "start": 2.8,
            "end": 5.0,
            "speaker": "Alice",
            "words": [{"start": 3.0, "end": 4.0, "word": "again"}],
        },
    ]

    relabeled = smooth_short_speaker_flips(segments, max_total_duration=1.0, max_score=0.3)

    assert relabeled[1]["speaker"] == "Alice"
    assert relabeled[1]["words"][0]["speaker"] == "Alice"
    assert relabeled[1]["speaker_postprocess_source"] == "smooth_short_speaker_flips"


def test_smooth_short_speaker_flips_preserves_high_confidence_run():
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "Alice"},
        {"start": 2.0, "end": 2.8, "speaker": "Bob", "speaker_match_score": 0.6},
        {"start": 2.8, "end": 5.0, "speaker": "Alice"},
    ]

    relabeled = smooth_short_speaker_flips(segments, max_total_duration=1.0, max_score=0.4)

    assert relabeled[1]["speaker"] == "Bob"


def test_smooth_short_speaker_flips_handles_two_segment_runs():
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "Alice"},
        {"start": 2.0, "end": 2.6, "speaker": "Bob", "speaker_match_score": 0.2},
        {"start": 2.6, "end": 3.2, "speaker": "Carol", "speaker_match_score": 0.2},
        {"start": 3.2, "end": 5.0, "speaker": "Alice"},
    ]

    relabeled = smooth_short_speaker_flips(
        segments,
        max_total_duration=1.5,
        max_score=0.3,
        max_run_segments=2,
    )

    assert [segment["speaker"] for segment in relabeled] == ["Alice", "Alice", "Alice", "Alice"]
