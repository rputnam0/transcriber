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
