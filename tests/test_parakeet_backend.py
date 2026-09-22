from __future__ import annotations

import sys
import types

from transcriber.parakeet_backend import DEFAULT_MODEL_NAME, resolve_model_name, transcribe_file


class _FakeSentence:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end


class _FakeResult:
    def __init__(self, sentences, text: str = "") -> None:
        self.sentences = sentences
        self.text = text


class _FakeNemoResult:
    def __init__(self, segments) -> None:
        self.timestamp = {"segment": segments}


class _FakeModel:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self.result


def test_resolve_model_name_maps_default_whisper_aliases():
    assert resolve_model_name(None) == DEFAULT_MODEL_NAME
    assert resolve_model_name("large-v3") == DEFAULT_MODEL_NAME
    assert resolve_model_name("medium.en") == DEFAULT_MODEL_NAME
    assert resolve_model_name("parakeet-tdt-0.6b-v3") == DEFAULT_MODEL_NAME
    assert resolve_model_name("mlx-community/parakeet-tdt-0.6b-v3") == DEFAULT_MODEL_NAME
    assert resolve_model_name("custom/model") == "custom/model"


def test_transcribe_file_maps_sentences_to_repo_segments():
    result = _FakeResult(
        [
            _FakeSentence(" Hello there. ", 0.25, 1.5),
            _FakeSentence(" ", 1.5, 2.0),
            _FakeSentence("General Kenobi.", 2.0, 3.0),
        ]
    )
    model = _FakeModel(result)

    segs = transcribe_file("sample.wav", model)

    assert segs == [
        {"start": 0.25, "end": 1.5, "text": "Hello there.", "speaker": None},
        {"start": 2.0, "end": 3.0, "text": "General Kenobi.", "speaker": None},
    ]
    assert model.calls[0][1]["chunk_duration"] == 120.0
    assert model.calls[0][1]["overlap_duration"] == 15.0


def test_parakeet_keeps_word_times_and_joins_subword_pieces():
    sentence = _FakeSentence("Hello everyone.", 0.1, 1.4)
    sentence.tokens = [
        types.SimpleNamespace(text=" Hello", start=0.1, end=0.4, confidence=0.9),
        types.SimpleNamespace(text=" every", start=0.5, end=0.8, confidence=0.8),
        types.SimpleNamespace(text="one", start=0.8, end=1.2, confidence=0.7),
        types.SimpleNamespace(text=".", start=1.2, end=1.4, confidence=0.9),
    ]
    segments = transcribe_file("unused.wav", _FakeModel(_FakeResult([sentence])))
    assert [w["word"] for w in segments[0]["words"]] == ["Hello", "everyone."]
    assert segments[0]["words"][1]["start"] == 0.5
    assert segments[0]["words"][1]["end"] == 1.2


def test_delayed_punctuation_does_not_extend_word_into_next_speaker():
    from transcriber.parakeet_backend import _tokens_to_words

    tokens = [
        types.SimpleNamespace(text=" on", start=6.24, end=6.56),
        types.SimpleNamespace(text=".", start=10.0, end=10.32),
        types.SimpleNamespace(text=" Um", start=10.32, end=10.64),
    ]
    words = _tokens_to_words(tokens)
    assert words[0] == {"word": "on.", "start": 6.24, "end": 6.56}
    assert words[1]["start"] == 10.32


def test_transcribe_file_maps_nemo_timestamps_to_repo_segments():
    result = _FakeNemoResult(
        [
            {"start": 0.1, "end": 0.9, "segment": "Hello there"},
            {"start": 1.2, "end": 1.8, "segment": "General Kenobi"},
        ]
    )
    handle = types.SimpleNamespace(runtime="nemo", model=_FakeModel([result]))

    segs = transcribe_file("sample.wav", handle)

    assert segs == [
        {"start": 0.1, "end": 0.9, "text": "Hello there", "speaker": None},
        {"start": 1.2, "end": 1.8, "text": "General Kenobi", "speaker": None},
    ]


def test_parakeet_keeps_separate_whitespace_token_before_number():
    from transcriber.parakeet_backend import _tokens_to_words

    tokens = [
        types.SimpleNamespace(text=text, start=i, end=i + 1)
        for i, text in enumerate(["cost", " ", "64", "0", " gold"])
    ]
    words = _tokens_to_words(tokens)
    assert [word["word"] for word in words] == ["cost", "640", "gold"]
    assert words[1]["start"] == 2


def test_resolve_dtype_respects_float16_alias(monkeypatch):
    fake_core = types.SimpleNamespace(
        float16="float16-dtype",
        float32="float32-dtype",
        bfloat16="bfloat16-dtype",
    )
    fake_mlx = types.SimpleNamespace(core=fake_core)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    from transcriber import parakeet_backend

    assert parakeet_backend._resolve_dtype("float16") == ("float16-dtype", "float16")
    assert parakeet_backend._resolve_dtype("fp16") == ("float16-dtype", "float16")
    assert parakeet_backend._resolve_dtype("float32") == ("float32-dtype", "float32")
    assert parakeet_backend._resolve_dtype("int8") == ("bfloat16-dtype", "bfloat16")


def test_unknown_decoder_markers_never_become_transcript_words():
    from types import SimpleNamespace
    from transcriber.parakeet_backend import _result_to_segments

    sentence = SimpleNamespace(
        text="<unk>" * 200, start=1, end=2, tokens=[SimpleNamespace(text="<unk>", start=1, end=2)]
    )
    assert _result_to_segments(SimpleNamespace(sentences=[sentence])) == []
