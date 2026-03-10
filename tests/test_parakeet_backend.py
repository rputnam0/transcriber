from __future__ import annotations

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
