from __future__ import annotations

import argparse
from pathlib import Path

from transcriber import cli
from transcriber.postprocess import PostProcessConfig
from transcriber.speaker_bank import SpeakerBankConfig


def test_auto_backend_prefers_parakeet_for_zip_stems_on_linux_cpu(monkeypatch):
    monkeypatch.setattr(cli, "_supports_parakeet_cpu_runtime", lambda: True)

    resolved = cli._resolve_backend_choice(
        "auto",
        files=["track1.ogg", "track2.ogg"],
        tmp_root="/tmp/extracted",
        min_speakers=None,
        max_speakers=None,
        single_file_speaker=None,
        device="cpu",
        speaker_bank_config=None,
    )

    assert resolved == "parakeet"


def test_auto_backend_prefers_parakeet_for_explicit_single_speaker_cpu(monkeypatch):
    monkeypatch.setattr(cli, "_supports_parakeet_cpu_runtime", lambda: True)

    resolved = cli._resolve_backend_choice(
        "auto",
        files=["input.wav"],
        tmp_root=None,
        min_speakers=None,
        max_speakers=None,
        single_file_speaker="Narrator",
        device="cpu",
        speaker_bank_config=None,
    )

    assert resolved == "parakeet"


def test_auto_backend_prefers_faster_for_multi_speaker_runs(monkeypatch):
    monkeypatch.setattr(cli, "_supports_parakeet_cpu_runtime", lambda: True)

    resolved = cli._resolve_backend_choice(
        "auto",
        files=["input.m4a"],
        tmp_root=None,
        min_speakers=2,
        max_speakers=6,
        single_file_speaker=None,
        device="cpu",
        speaker_bank_config=SpeakerBankConfig(enabled=True),
    )

    assert resolved == "faster"


def test_auto_backend_treats_single_file_zip_like_regular_audio(monkeypatch):
    monkeypatch.setattr(cli, "_supports_parakeet_cpu_runtime", lambda: True)

    resolved = cli._resolve_backend_choice(
        "auto",
        files=["input.wav"],
        tmp_root="/tmp/extracted",
        min_speakers=None,
        max_speakers=None,
        single_file_speaker=None,
        device="cpu",
        speaker_bank_config=None,
    )

    assert resolved == "faster"


def test_apply_config_defaults_honors_backend():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "faster", "parakeet"], default="auto")

    cli._apply_config_defaults(parser, {"backend": "parakeet"})

    assert parser.parse_args([]).backend == "parakeet"


def test_single_file_speaker_overrides_faster_labels(monkeypatch, tmp_path):
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

    from transcriber.transcript_pipeline import TranscriptPipelineResult

    def fake_transcribe_with_faster_pipeline(*args, **kwargs):
        return TranscriptPipelineResult(
            segments=[{"start": 0.0, "end": 1.0, "text": "hello world", "speaker": "SPEAKER_00"}],
            diarization_segments=[],
            exclusive_diarization_segments=[],
            speaker_embeddings={},
            metadata={},
        )

    monkeypatch.setattr(
        __import__("transcriber.transcript_pipeline", fromlist=["x"]),
        "transcribe_with_faster_pipeline",
        fake_transcribe_with_faster_pipeline,
    )
    monkeypatch.setattr(
        __import__("transcriber.diarization", fromlist=["x"]), "_detect_device", lambda: "cpu"
    )

    cli.run_transcribe(
        input_path=str(tmp_path / "input.wav"),
        backend="faster",
        output_dir=str(tmp_path / "outputs"),
        single_file_speaker="Narrator",
        quiet=True,
        speaker_bank_config=None,
    )

    segs = captured["per_file_segments"][0][1]
    assert segs[0]["speaker"] == "Narrator"


def test_watch_task_kind_requests_postprocess_backfill(tmp_path):
    transcript_root = tmp_path / "outputs" / "Session 32"
    transcript_root.mkdir(parents=True, exist_ok=True)
    (transcript_root / "Session 32.txt").write_text("hello", encoding="utf-8")

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Session 32.wav"),
        str(tmp_path / "outputs"),
        postprocess_config,
    )

    assert action == "postprocess"

    marker = Path(postprocess_config.summaries_dir) / "Session 32" / "session_32.postprocess.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")

    assert (
        cli._watch_task_kind(
            str(tmp_path / "incoming" / "Session 32.wav"),
            str(tmp_path / "outputs"),
            postprocess_config,
        )
        is None
    )
