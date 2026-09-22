from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_transcribe_with_faster_pipeline_assigns_exclusive_speakers(monkeypatch):
    from transcriber.asr import AsrResult, AsrSegment, AsrWord
    from transcriber.diarization import DiarizationResult, DiarizationTurn, SegmentEmbeddingResult
    from transcriber import transcript_pipeline as pipeline

    def fake_asr(*args, **kwargs):
        return AsrResult(
            segments=[
                AsrSegment(
                    start=0.0,
                    end=1.2,
                    text="hello there",
                    words=[
                        AsrWord(word="hello", start=0.0, end=0.4, score=0.9),
                        AsrWord(word="there", start=0.7, end=1.1, score=0.8),
                    ],
                )
            ],
            language="en",
            metadata={"batch_size": 2},
        )

    def fake_diarize(*args, **kwargs):
        return DiarizationResult(
            segments=[
                DiarizationTurn(start=0.0, end=0.6, speaker="SPEAKER_00"),
                DiarizationTurn(start=0.6, end=1.2, speaker="SPEAKER_01"),
            ],
            exclusive_segments=[
                DiarizationTurn(start=0.0, end=0.5, speaker="SPEAKER_00"),
                DiarizationTurn(start=0.5, end=1.2, speaker="SPEAKER_01"),
            ],
            metadata={"model_name": "community-1"},
        )

    def fake_embeddings(*args, **kwargs):
        return (
            [
                SegmentEmbeddingResult(
                    speaker="SPEAKER_00",
                    start=0.0,
                    end=0.5,
                    index=0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
                SegmentEmbeddingResult(
                    speaker="SPEAKER_01",
                    start=0.5,
                    end=1.2,
                    index=1,
                    embedding=np.array([0.0, 1.0], dtype=np.float32),
                ),
            ],
            {"embedded": 2, "skipped": 0, "total": 2},
        )

    monkeypatch.setattr(pipeline, "transcribe_with_faster_whisper", fake_asr)
    monkeypatch.setattr(pipeline, "diarize_audio", fake_diarize)
    monkeypatch.setattr(pipeline, "extract_embeddings_for_segments", fake_embeddings)

    result = pipeline.transcribe_with_faster_pipeline(
        "dummy.wav",
        model_name="large-v3",
        compute_type="int8",
        force_device="cpu",
    )

    assert [segment["speaker"] for segment in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0]["words"][0]["speaker"] == "SPEAKER_00"
    assert result.segments[1]["words"][0]["speaker"] == "SPEAKER_01"
    assert len(result.exclusive_diarization_segments) == 2
    assert sorted(result.speaker_embeddings) == ["SPEAKER_00", "SPEAKER_01"]


def test_word_uses_overlapping_fallback_before_distant_exclusive_turn():
    from transcriber.transcript_pipeline import _choose_turn_label
    from transcriber.diarization import DiarizationTurn as Turn

    assert _choose_turn_label(10, 10.4, [Turn(0, 1, "A")], [Turn(10, 11, "B")]) == "B"
    assert _choose_turn_label(10, 10.4, [Turn(0, 1, "A")], []) is None
    assert _choose_turn_label(1.05, 1.1, [Turn(0, 1, "A")], []) == "A"
    # Parakeet can give the trailing word a long span beginning just after speech.
    assert _choose_turn_label(1.1, 2.0, [Turn(0, 1, "A")], []) == "A"
    assert _choose_turn_label(1.36, 2.0, [Turn(0, 1, "A")], []) is None
    assert _choose_turn_label(1.1, 1.3, [Turn(0, 1, "A"), Turn(1, 1.017, "B")], []) == "A"


def test_identity_excerpts_keep_brief_speakers_and_exclude_other_voices(monkeypatch):
    from transcriber import transcript_pipeline as pipeline
    from transcriber.diarization import DiarizationResult, DiarizationTurn as Turn

    # B has only a half-second clean interjection. C overlaps A and has no clean evidence.
    regular = [Turn(0, 3, "A"), Turn(1, 2, "C"), Turn(3, 3.5, "B")]
    exclusive = [Turn(0, 3, "A"), Turn(3, 3.5, "B")]
    captured = {}

    def fake_embeddings(path, segments, token, **kwargs):
        captured["segments"] = segments
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(pipeline, "extract_embeddings_for_segments", fake_embeddings)
    pipeline._aggregate_speaker_embeddings(
        "unused.wav",
        DiarizationResult(regular, exclusive, {}),
        hf_token=None,
        diarization_model_name=None,
        force_device="cpu",
        quiet=True,
    )
    assert {speaker for _, _, speaker in captured["segments"]} == {"A", "B"}
    assert all(end <= 1 or start >= 2 for start, end, _ in captured["segments"])
    assert captured["pre_pad"] == captured["post_pad"] == 0


def test_diarization_failure_is_reported_to_caller(monkeypatch):
    import pytest
    from transcriber import transcript_pipeline as pipeline
    from transcriber.asr import AsrResult

    monkeypatch.setattr(pipeline, "transcribe_with_faster_whisper", lambda *a, **k: AsrResult([]))

    def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(pipeline, "diarize_audio", fail)
    with pytest.raises(RuntimeError, match="diarization"):
        pipeline.transcribe_with_faster_pipeline(
            "unused.wav", model_name="tiny", force_device="cpu"
        )
