from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from transcriber import mlx_whisper_backend as backend


def test_context_words_have_one_owner_and_preserve_punctuation():
    raw = [
        {
            "text": "Hello, world!",
            "words": [
                {"word": " Hello,", "start": 0.0, "end": 0.8},
                {"word": " world!", "start": 0.8, "end": 1.4},
            ],
        }
    ]
    first = backend.owned_segments(raw, 0, 0, 1)
    second = backend.owned_segments(raw, 0, 1, 2)
    assert first[0]["text"] == "Hello,"
    assert second[0]["text"] == "world!"
    assert first[0]["words"] + second[0]["words"] == raw[0]["words"]


def test_missing_or_invalid_word_times_fail_instead_of_losing_text():
    with pytest.raises(ValueError, match="word timestamps"):
        backend.owned_segments([{"text": "speech", "words": []}], 0, 0, 1)
    with pytest.raises(ValueError, match="Invalid"):
        backend.owned_segments([{"words": [{"word": "speech", "start": 1, "end": 0}]}], 0, 0, 1)

    with pytest.raises(ValueError, match="cover the recognized text"):
        backend.owned_segments(
            [{"text": "quiet speech", "words": [{"word": "speech", "start": 0, "end": 1}]}],
            0,
            0,
            1,
        )


def test_quiet_speech_is_retained_silence_skipped_and_cache_verified(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    wave = np.concatenate([np.zeros(16000), np.full(16000, 1e-9)]).astype(np.float32)
    calls = []

    def transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {
            "segments": [
                {"text": "Quiet.", "words": [{"word": " Quiet.", "start": 0.1, "end": 0.9}]}
            ]
        }

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(backend, "load_audio_mono", lambda _: wave)
    monkeypatch.setattr(backend, "version", lambda _: "0.4.3")
    monkeypatch.setattr(backend, "CORE_SECONDS", 1)
    monkeypatch.setattr(backend, "CONTEXT_SECONDS", 0)
    cache = tmp_path / "cache"
    result = backend.transcribe_file("audio", model_name=str(model), download_root=str(cache))
    assert len(calls) == 1
    assert result[0]["start"] == 1.1
    assert result[0]["text"] == "Quiet."
    assert calls[0]["condition_on_previous_text"] is False
    assert (
        backend.transcribe_file("audio", model_name=str(model), download_root=str(cache)) == result
    )
    assert len(calls) == 1
    target = next(cache.rglob("*.json"))
    cached = json.loads(target.read_text())
    cached["audio_sha256"] = "changed"
    target.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="provenance"):
        backend.transcribe_file("audio", model_name=str(model), download_root=str(cache))


def test_whisper_alias_is_pinned_and_parakeet_requires_explicit_backend():
    assert backend.resolve_model_name("large-v3") == backend.DEFAULT_MODEL
    with pytest.raises(ValueError, match="backend parakeet"):
        backend.resolve_model_name("parakeet")


def test_sentence_continuation_crosses_decoder_boundary_without_losing_words():
    from transcriber.consolidate import split_segments

    words = [
        {"word": " This", "start": 59.5, "end": 59.9},
        {"word": " continues.", "start": 60.0, "end": 60.5},
        {"word": " Next.", "start": 62.0, "end": 62.5},
    ]
    raw = [{"words": words[:1]}, {"words": words[1:]}]
    merged = backend.join_decoder_segments(raw)
    result = split_segments(merged, pause_threshold=0.8)
    assert [r["text"] for r in result] == ["This continues.", "Next."]
    assert [w for r in result for w in r["words"]] == words
    assert result[0]["start"] == 59.5 and result[0]["end"] == 60.5
