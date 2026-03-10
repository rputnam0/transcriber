from __future__ import annotations

from transcriber import cli


def test_single_file_speaker_overrides_backend_labels(monkeypatch, tmp_path):
    captured: dict = {}

    monkeypatch.setattr(cli, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli, "_preload_cudnn_libs", lambda: None)
    monkeypatch.setattr(cli, "_resolve_cache_root", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "gather_inputs", lambda path: ([str(tmp_path / "input.wav")], None))

    def fake_save_outputs(**kwargs):
        captured["per_file_segments"] = kwargs["per_file_segments"]
        out_dir = tmp_path / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(cli, "save_outputs", fake_save_outputs)

    from transcriber import parakeet_backend

    monkeypatch.setattr(parakeet_backend, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        parakeet_backend,
        "transcribe_file",
        lambda *args, **kwargs: [
            {"start": 0.0, "end": 1.0, "text": "hello world", "speaker": "from-backend"}
        ],
    )

    cli.run_transcribe(
        input_path=str(tmp_path / "input.wav"),
        backend="parakeet",
        output_dir=str(tmp_path / "outputs"),
        single_file_speaker="Narrator",
        quiet=True,
        speaker_bank_config=None,
    )

    segs = captured["per_file_segments"][0][1]
    assert segs[0]["speaker"] == "Narrator"
