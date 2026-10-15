import numpy as np
from types import SimpleNamespace
from transcriber.speech_activity import speech_regions
from transcriber.qwen_backend import transcribe_wave


def test_silence_and_brief_speech_with_padding():
    assert speech_regions(np.zeros(100), 100 * 512) == []
    scores = np.zeros(100)
    scores[30:33] = 0.8  # Retain a 96 ms response, even amid a long pause.
    assert speech_regions(scores, 100 * 512) == [[30 * 512 - 4000, 33 * 512 + 4000]]
    scores[:] = 0.8
    assert speech_regions(scores, 100 * 512) == [[0, 100 * 512]]


def test_nearby_phrases_merge_without_losing_quiet_tail():
    scores = np.zeros(100)
    scores[10:20] = 0.8
    scores[20:23] = 0.25  # Hysteresis keeps lower-confidence speech.
    scores[30:35] = 0.8
    assert speech_regions(scores, 100 * 512) == [[10 * 512 - 4000, 35 * 512 + 4000]]


def test_disjoint_speech_keeps_original_offsets_and_skips_silence(tmp_path):
    calls = []
    model = SimpleNamespace(
        generate=lambda wave, **kw: calls.append(len(wave))
        or SimpleNamespace(text="Yes.", generation_tokens=2)
    )
    aligner = SimpleNamespace(
        generate=lambda **kw: SimpleNamespace(segments=[dict(text="Yes", start=0.2, end=0.5)])
    )
    wave = np.ones(100 * 16000)
    result = transcribe_wave(
        wave,
        model,
        aligner,
        vocabulary={"terms": []},
        cache=tmp_path,
        provenance={},
        regions=[[10 * 16000, 11 * 16000], [90 * 16000, 91 * 16000]],
    )
    assert calls == [16000, 16000]
    assert [s["start"] for s in result] == [10.2, 90.2]
    assert (
        transcribe_wave(
            wave,
            None,
            None,
            vocabulary={"terms": []},
            cache=tmp_path / "silent",
            provenance={},
            regions=[],
        )
        == []
    )
