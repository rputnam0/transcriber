from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from transcriber.segments import (
    TrainingSegment,
    generate_windows_for_segments,
    load_segments_from_jsonl,
)
from transcriber.whisperx_backend import extract_embeddings_for_segments


def test_load_segments_from_jsonl(tmp_path):
    payload = {
        "file": "session_01.wav",
        "start": 0.5,
        "end": 4.5,
        "speaker": "SPEAKER_00",
        "speaker_raw": "SPEAKER_00",
    }
    path = tmp_path / "segments.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    mapping = {"SPEAKER_00": "Alice"}
    segments = load_segments_from_jsonl(path, mapping)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.audio_file == "session_01.wav"
    assert seg.speaker == "Alice"
    assert seg.start == payload["start"]
    assert seg.end == payload["end"]


def test_generate_windows_from_segment():
    segment = TrainingSegment(
        audio_file="sample.wav",
        start=0.0,
        end=25.0,
        speaker="Alice",
        speaker_raw="SPEAKER_00",
    )
    windows = generate_windows_for_segments(
        [segment],
        min_duration=5.0,
        max_duration=12.0,
        window_size=10.0,
        window_stride=7.5,
    )
    assert windows, "Expected at least one window"
    assert all(win.speaker == "Alice" for win in windows)
    assert windows[0].start == 0.0
    assert windows[-1].end <= segment.end + 1e-6


def test_extract_embeddings_for_segments_mocked(monkeypatch, tmp_path):
    # Stub whisperx.audio
    audio_module = types.SimpleNamespace(
        load_audio=lambda path: np.ones(32000, dtype=np.float32),
        SAMPLE_RATE=16000,
    )
    monkeypatch.setitem(sys.modules, "whisperx", types.SimpleNamespace(audio=audio_module))
    monkeypatch.setitem(sys.modules, "whisperx.audio", audio_module)

    class DummyEmbedder:
        sample_rate = 16000

        def to(self, device):
            return self

        def __call__(self, wave_batch, masks=None):
            batch = wave_batch.shape[0]
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (batch, 1))

    class DummyPipeline:
        def __init__(self):
            self.model = types.SimpleNamespace(_embedding=DummyEmbedder(), embedding="dummy")

    monkeypatch.setattr(
        "transcriber.whisperx_backend._get_diar_pipeline",
        lambda device, token: DummyPipeline(),
    )

    segments = [(0.0, 1.0, "SpeakerA"), (1.0, 2.0, "SpeakerB")]
    results, summary = extract_embeddings_for_segments(
        audio_path="dummy.wav",
        segments=segments,
        hf_token=None,
        force_device="cpu",
        quiet=True,
        pre_pad=0.0,
        post_pad=0.0,
        batch_size=2,
        workers=1,
    )

    assert summary["embedded"] == 2
    assert len(results) == 2
    assert {res.speaker for res in results} == {"SpeakerA", "SpeakerB"}
    for res in results:
        assert np.isclose(np.linalg.norm(res.embedding), 1.0)
