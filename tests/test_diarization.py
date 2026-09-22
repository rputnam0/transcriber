from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_annotation_to_segments_reads_speaker_diarization_attribute():
    from transcriber.diarization import _annotation_to_segments

    class FakeAnnotation:
        def itertracks(self, yield_label: bool = True):
            yield SimpleSegment(0.0, 0.5), None, "SPEAKER_00"
            yield SimpleSegment(0.5, 1.0), None, "SPEAKER_01"

    class SimpleSegment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class FakeDiarizeOutput:
        def __init__(self):
            self.speaker_diarization = FakeAnnotation()

    turns = _annotation_to_segments(FakeDiarizeOutput())

    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].start == 0.0
    assert turns[1].end == 1.0


def test_extract_embeddings_for_segments_accepts_tensor_like_batches(monkeypatch):
    from transcriber import diarization

    class FakeTensorBatch:
        def __init__(self, values: list[list[float]]):
            self._values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class FakeEmbedder:
        sample_rate = 16000

        def __call__(self, wave_batch, masks=None):
            return FakeTensorBatch([[1.0, 0.0]])

    monkeypatch.setattr(diarization, "_resolve_embedder", lambda **kwargs: FakeEmbedder())
    monkeypatch.setattr(
        diarization,
        "load_audio_mono",
        lambda *args, **kwargs: np.ones(16000, dtype=np.float32),
    )

    results, summary = diarization.extract_embeddings_for_segments(
        "dummy.wav",
        [(0.0, 0.5, "SPEAKER_00")],
        hf_token=None,
        force_device="cpu",
    )

    assert summary["embedded"] == 1
    assert len(results) == 1
    assert results[0].speaker == "SPEAKER_00"
    assert np.allclose(results[0].embedding, np.array([1.0, 0.0], dtype=np.float32))


def test_embedding_does_not_change_with_a_longer_batch_neighbor(monkeypatch):
    from transcriber import diarization

    class CenteringEmbedder:
        sample_rate = 16000

        def __call__(self, waves, masks=None):
            # Like WeSpeaker's frontend, statistics are computed before pooling masks.
            values = waves[:, 0].numpy()
            return np.stack([values.mean(axis=1), values.std(axis=1)], axis=1)

    monkeypatch.setattr(diarization, "_resolve_embedder", lambda **kwargs: CenteringEmbedder())
    kwargs = dict(
        hf_token=None,
        force_device="cpu",
        pre_pad=0,
        post_pad=0,
        audio_waveform=np.ones(16000, dtype=np.float32),
        audio_sample_rate=16000,
    )
    alone, _ = diarization.extract_embeddings_for_segments("", [(0, 0.5, "A")], **kwargs)
    batched, _ = diarization.extract_embeddings_for_segments(
        "", [(0, 0.5, "A"), (0, 1, "B")], **kwargs
    )
    assert np.allclose(alone[0].embedding, batched[0].embedding)


def test_embedding_skips_crops_shorter_than_model_minimum(monkeypatch):
    from transcriber import diarization

    class Embedder:
        sample_rate = 16000
        min_num_samples = 400

        def __call__(self, waves, masks=None):
            assert waves.shape[-1] >= self.min_num_samples
            return np.ones((len(waves), 2), dtype=np.float32)

    monkeypatch.setattr(diarization, "_resolve_embedder", lambda **kwargs: Embedder())
    results, summary = diarization.extract_embeddings_for_segments(
        "",
        [(0, 0.01, "too_short"), (0, 0.5, "valid")],
        None,
        force_device="cpu",
        pre_pad=0,
        post_pad=0,
        audio_waveform=np.ones(16000, dtype=np.float32),
        audio_sample_rate=16000,
    )
    assert [result.speaker for result in results] == ["valid"]
    assert summary["skipped"] == 1


def test_explicit_mps_embedding_device_is_not_silently_replaced(monkeypatch):
    from transcriber import diarization

    devices = []

    class Embedder:
        sample_rate = 16000

    def resolve(**kwargs):
        devices.append(kwargs["device"])
        return Embedder()

    monkeypatch.setattr(diarization, "_resolve_embedder", resolve)
    # Empty input avoids requiring Apple hardware on Linux CI while exercising device routing.
    results, _ = diarization.extract_embeddings_for_segments(
        "", [], None, force_device="mps", audio_waveform=np.zeros(16000, np.float32)
    )
    assert results == []
    assert devices == ["mps"]
