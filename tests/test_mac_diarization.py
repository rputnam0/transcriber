from transcriber.diarization import DiarizationResult, DiarizationTurn


def test_explicit_parakeet_runs_diarization_with_word_times(monkeypatch, tmp_path):
    from transcriber import cli, parakeet_backend, transcript_pipeline

    audio = tmp_path / "conversation.wav"
    audio.touch()
    monkeypatch.setattr(cli, "_tqdm_enabled", lambda: False)
    monkeypatch.setattr(parakeet_backend, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        parakeet_backend,
        "transcribe_file",
        lambda *a, **k: [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello goodbye",
                "speaker": None,
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "goodbye", "start": 1.0, "end": 1.5},
                ],
            }
        ],
    )
    called = []

    def diarize(path, **kwargs):
        called.append(kwargs)
        turns = [DiarizationTurn(0, 0.8, "A"), DiarizationTurn(1, 2, "B")]
        return DiarizationResult(turns, turns, {})

    monkeypatch.setattr(transcript_pipeline, "diarize_audio", diarize)
    monkeypatch.setattr(transcript_pipeline, "_aggregate_speaker_embeddings", lambda *a, **k: {})
    outputs = {}

    def save(**kwargs):
        outputs.update(kwargs)
        return tmp_path / "out"

    monkeypatch.setattr(cli, "save_outputs", save)
    cli.run_transcribe(
        str(audio),
        backend="parakeet",
        output_dir=str(tmp_path / "out"),
        min_speakers=2,
        max_speakers=2,
        quiet=True,
    )
    assert called and called[0]["min_speakers"] == 2
    segments = outputs["per_file_segments"][0][1]
    assert [segment["speaker"] for segment in segments] == ["A", "B"]
    assert outputs["diar_by_file"][str(audio)]


def test_parakeet_known_single_speaker_does_not_load_diarization(monkeypatch, tmp_path):
    from transcriber import cli, parakeet_backend, transcript_pipeline

    audio = tmp_path / "one-speaker.wav"
    audio.touch()
    monkeypatch.setattr(cli, "_tqdm_enabled", lambda: False)
    monkeypatch.setattr(parakeet_backend, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        parakeet_backend,
        "transcribe_file",
        lambda *a, **k: [{"start": 0, "end": 1, "text": "Hello", "speaker": None}],
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("A known isolated voice does not need diarization")

    monkeypatch.setattr(transcript_pipeline, "diarize_audio", unexpected)
    outputs = {}
    monkeypatch.setattr(cli, "save_outputs", lambda **kw: outputs.update(kw) or tmp_path)
    cli.run_transcribe(
        str(audio),
        backend="parakeet",
        single_file_speaker="Known",
        cache_mode="env",
        output_dir=str(tmp_path),
    )
    assert outputs["per_file_segments"][0][1][0]["speaker"] == "Known"
