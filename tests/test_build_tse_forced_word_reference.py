from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_tse_forced_word_reference as forced_ref  # noqa: E402


def test_fallback_word_spans_distributes_tokens_across_source_span():
    words = forced_ref._fallback_word_spans(
        span={
            "speaker": "Dungeon Master",
            "start": 10.0,
            "end": 12.0,
        },
        units=[
            {"text": "hello", "normalized": "hello"},
            {"text": "there", "normalized": "there"},
        ],
        reason="RuntimeError",
    )

    assert words == [
        {
            "speaker": "Dungeon Master",
            "start": 10.0,
            "end": 11.0,
            "text": "hello",
            "normalized": "hello",
            "score": None,
            "source_span_start": 10.0,
            "source_span_end": 12.0,
            "alignment_source": "source_span_fallback",
            "fallback_reason": "RuntimeError",
        },
        {
            "speaker": "Dungeon Master",
            "start": 11.0,
            "end": 12.0,
            "text": "there",
            "normalized": "there",
            "score": None,
            "source_span_start": 10.0,
            "source_span_end": 12.0,
            "alignment_source": "source_span_fallback",
            "fallback_reason": "RuntimeError",
        },
    ]


def test_load_track_window_uses_keyword_frame_offset_and_pads(monkeypatch, tmp_path):
    calls = []

    def fake_load(uri, *, frame_offset=0, num_frames=-1):
        calls.append(
            {
                "uri": uri,
                "frame_offset": frame_offset,
                "num_frames": num_frames,
            }
        )
        return torch.ones(1, 4), 16000

    monkeypatch.setattr(forced_ref.torchaudio, "load", fake_load)

    wave = forced_ref._load_track_window(
        tmp_path / "speaker.wav",
        start_seconds=1.0,
        duration_seconds=0.001,
        sample_rate=16000,
    )

    assert calls == [
        {
            "uri": str(tmp_path / "speaker.wav"),
            "frame_offset": 16000,
            "num_frames": 16,
        }
    ]
    assert tuple(wave.shape) == (1, 16)
    assert torch.equal(wave[:, :4], torch.ones(1, 4))
    assert torch.equal(wave[:, 4:], torch.zeros(1, 12))


def test_speaker_window_units_preserve_order_and_source_spans() -> None:
    spans = [
        {"speaker": "Alice", "start": 5.0, "end": 6.0, "text": "later"},
        {"speaker": "Bob", "start": 1.0, "end": 2.0, "text": "ignore"},
        {"speaker": "Alice", "start": 1.0, "end": 2.0, "text": "first two"},
    ]

    units = forced_ref._speaker_units(
        spans,
        speaker="Alice",
        allowed=set("abcdefghijklmnopqrstuvwxyz'"),
    )

    assert [unit["normalized"] for unit in units] == ["first", "two", "later"]
    assert [unit["source_span_start"] for unit in units] == [1.0, 1.0, 5.0]
