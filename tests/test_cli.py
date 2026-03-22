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


def test_apply_config_defaults_honors_non_session_output_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-session-output-dir")

    cli._apply_config_defaults(
        parser,
        {"non_session_output_dir": "/tmp/odds-and-ends"},
    )

    assert parser.parse_args([]).non_session_output_dir == "/tmp/odds-and-ends"


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
    (transcript_root / "Session 32.jsonl").write_text("{}", encoding="utf-8")

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
        watch_postprocess_backfill=True,
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
            watch_postprocess_backfill=True,
        )
        is None
    )


def test_iter_candidate_media_honors_exclude_globs(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    (root / "Session 62.zip").write_text("zip", encoding="utf-8")
    processed_dir = root / "processed"
    processed_dir.mkdir()
    (processed_dir / "Session 70.zip").write_text("zip", encoding="utf-8")
    session_dir = root / "Session 9"
    session_dir.mkdir()
    (session_dir / "1-joeeeenathan_0.wav").write_text("wav", encoding="utf-8")
    (session_dir / "Session 9.wav").write_text("wav", encoding="utf-8")

    found = cli._iter_candidate_media(root, ["**/[0-9]-*.wav"])

    assert str(root / "Session 62.zip") in found
    assert str(session_dir / "Session 9.wav") in found
    assert str(session_dir / "1-joeeeenathan_0.wav") not in found
    assert str(processed_dir / "Session 70.zip") not in found


def test_archive_processed_watch_input_moves_audio_and_manifest_pair(tmp_path):
    root = tmp_path / "incoming"
    root.mkdir()
    zip_path = root / "Live Smoke Test.zip"
    json_path = root / "Live Smoke Test.json"
    zip_path.write_text("zip", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")

    archived = cli._archive_processed_watch_input(zip_path, root)

    assert archived == [
        root / "processed" / "Live Smoke Test.zip",
        root / "processed" / "Live Smoke Test.json",
    ]
    assert not zip_path.exists()
    assert not json_path.exists()


def test_move_watch_input_bundle_moves_audio_and_manifest_pair_to_quarantine(tmp_path):
    root = tmp_path / "incoming"
    quarantine_dir = root / "quarantine"
    root.mkdir()
    quarantine_dir.mkdir()
    zip_path = root / "MmCQtqke7kB5.zip"
    json_path = root / "MmCQtqke7kB5.json"
    zip_path.write_text("zip", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")

    moved = cli._move_watch_input_bundle(zip_path, quarantine_dir)

    assert moved == [
        quarantine_dir / "MmCQtqke7kB5.zip",
        quarantine_dir / "MmCQtqke7kB5.json",
    ]
    assert not zip_path.exists()
    assert not json_path.exists()


def test_archive_processed_watch_input_avoids_name_collisions(tmp_path):
    root = tmp_path / "incoming"
    root.mkdir()
    processed_dir = root / "processed"
    processed_dir.mkdir()
    (processed_dir / "Live Smoke Test.zip").write_text("old zip", encoding="utf-8")
    (processed_dir / "Live Smoke Test.json").write_text("old json", encoding="utf-8")

    zip_path = root / "Live Smoke Test.zip"
    json_path = root / "Live Smoke Test.json"
    zip_path.write_text("zip", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")

    archived = cli._archive_processed_watch_input(zip_path, root)

    assert archived == [
        root / "processed" / "Live Smoke Test__2.zip",
        root / "processed" / "Live Smoke Test__2.json",
    ]


def test_find_existing_transcript_for_input_matches_historical_names(tmp_path):
    output_dir = tmp_path / "Transcripts"
    transcript_dir = output_dir / "Session 13"
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "Session 13 Transcript.txt"
    transcript.write_text("done", encoding="utf-8")

    found = cli._find_existing_transcript_for_input(
        str(tmp_path / "Audio" / "Session 13.zip"),
        str(output_dir),
    )

    assert found == transcript


def test_find_existing_transcript_for_nested_session_input(tmp_path):
    output_dir = tmp_path / "Transcripts"
    transcript_dir = output_dir / "Session 4"
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "Copy of Session 4_transcription.txt"
    transcript.write_text("done", encoding="utf-8")

    found = cli._find_existing_transcript_for_input(
        str(tmp_path / "Audio" / "Session 4" / "Raw Audio.zip"),
        str(output_dir),
    )

    assert found == transcript


def test_find_existing_transcript_for_non_session_input_prefers_alternate_output_root(tmp_path):
    output_dir = tmp_path / "Transcripts"
    alternate_output_dir = tmp_path / "Odds and Ends Transcripts"
    transcript_dir = alternate_output_dir / "Live Smoke Test"
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "Live Smoke Test.txt"
    transcript.write_text("done", encoding="utf-8")

    found = cli._find_existing_transcript_for_input(
        str(tmp_path / "Audio" / "Live Smoke Test.zip"),
        str(output_dir),
        str(alternate_output_dir),
    )

    assert found == transcript


def test_raw_recording_id_does_not_count_as_session_identity():
    assert not cli._input_has_session_identity("/tmp/1j637PZBwVBg.zip")
    assert not cli._input_has_session_identity("/tmp/MmCQtqke7kB5.zip")
    assert cli._input_has_session_identity("/tmp/Session 63 Craig Copy Smoke.zip")


def test_watch_task_kind_skips_historical_transcript_without_backfill(tmp_path):
    output_dir = tmp_path / "Transcripts"
    transcript_dir = output_dir / "Session 7"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "Session 7_transcription.txt").write_text("done", encoding="utf-8")

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "Audio" / "Session 7" / "ALgqyzm1qUOD_data.zip"),
        str(output_dir),
        postprocess_config,
    )

    assert action is None


def test_watch_task_kind_retries_postprocess_for_marked_transcript(tmp_path):
    transcript_root = tmp_path / "outputs" / "Session 44"
    transcript_root.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_root / "Session 44.txt"
    transcript_path.write_text("hello", encoding="utf-8")
    cli._transcription_completion_marker_path(transcript_path).write_text(
        '{"status":"completed"}',
        encoding="utf-8",
    )

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Session 44.wav"),
        str(tmp_path / "outputs"),
        postprocess_config,
        watch_postprocess_backfill=False,
    )

    assert action == "postprocess"


def test_watch_task_kind_retranscribes_expected_transcript_with_incomplete_marker(tmp_path):
    transcript_root = tmp_path / "outputs" / "Session 51"
    transcript_root.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_root / "Session 51.txt"
    transcript_path.write_text("partial", encoding="utf-8")
    cli._transcription_completion_marker_path(transcript_path).write_text(
        '{"status":"in_progress"}',
        encoding="utf-8",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Session 51.wav"),
        str(tmp_path / "outputs"),
        None,
    )

    assert action == "transcribe"


def test_watch_task_kind_skips_legacy_expected_transcript_without_marker(tmp_path):
    transcript_root = tmp_path / "outputs" / "Session 52"
    transcript_root.mkdir(parents=True, exist_ok=True)
    (transcript_root / "Session 52.txt").write_text("done", encoding="utf-8")

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Session 52.wav"),
        str(tmp_path / "outputs"),
        None,
    )

    assert action is None


def test_watch_task_kind_waits_for_all_split_parts_before_postprocess(tmp_path):
    transcript_root = tmp_path / "outputs" / "Session 28 1_2"
    transcript_root.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_root / "Session 28 1_2.txt"
    transcript_path.write_text("part one", encoding="utf-8")
    cli._transcription_completion_marker_path(transcript_path).write_text(
        '{"status":"completed"}',
        encoding="utf-8",
    )

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Session 28 1_2.wav"),
        str(tmp_path / "outputs"),
        postprocess_config,
        watch_postprocess_backfill=False,
    )

    assert action is None


def test_watch_task_kind_skips_postprocess_for_non_session_transcript(tmp_path):
    transcript_root = tmp_path / "outputs" / "Craig Copy Local Smoke"
    transcript_root.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_root / "Craig Copy Local Smoke.txt"
    transcript_path.write_text("done", encoding="utf-8")
    cli._transcription_completion_marker_path(transcript_path).write_text(
        '{"status":"completed"}',
        encoding="utf-8",
    )

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Craig Copy Local Smoke.zip"),
        str(tmp_path / "outputs"),
        postprocess_config,
        watch_postprocess_backfill=False,
    )

    assert action is None


def test_watch_task_kind_skips_non_session_transcript_in_alternate_root(tmp_path):
    main_output_dir = tmp_path / "outputs"
    alternate_output_dir = tmp_path / "Odds and Ends Transcripts"
    transcript_root = alternate_output_dir / "Live Smoke Test"
    transcript_root.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_root / "Live Smoke Test.txt"
    transcript_path.write_text("done", encoding="utf-8")
    cli._transcription_completion_marker_path(transcript_path).write_text(
        '{"status":"completed"}',
        encoding="utf-8",
    )

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Live Smoke Test.zip"),
        str(main_output_dir),
        postprocess_config,
        non_session_output_dir=str(alternate_output_dir),
        watch_postprocess_backfill=False,
    )

    assert action is None


def test_should_quarantine_watch_failure_recognizes_empty_transcript_bundle():
    assert cli._should_quarantine_watch_failure(
        "Refusing to run post-processing with an empty transcript bundle for /tmp/out.txt"
    )


def test_should_quarantine_watch_failure_ignores_retryable_errors():
    assert not cli._should_quarantine_watch_failure("temporary google api timeout")


def test_watch_task_kind_waits_for_completed_split_part_markers(tmp_path):
    part_one_root = tmp_path / "outputs" / "Session 28 1_2"
    part_two_root = tmp_path / "outputs" / "Session 28 2_2"
    part_one_root.mkdir(parents=True, exist_ok=True)
    part_two_root.mkdir(parents=True, exist_ok=True)
    part_one_path = part_one_root / "Session 28 1_2.txt"
    part_two_path = part_two_root / "Session 28 2_2.txt"
    part_one_path.write_text("part one", encoding="utf-8")
    part_two_path.write_text("part two", encoding="utf-8")
    cli._transcription_completion_marker_path(part_one_path).write_text(
        '{"status":"completed"}',
        encoding="utf-8",
    )
    cli._transcription_completion_marker_path(part_two_path).write_text(
        '{"status":"in_progress"}',
        encoding="utf-8",
    )

    postprocess_config = PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    action = cli._watch_task_kind(
        str(tmp_path / "incoming" / "Session 28 1_2.wav"),
        str(tmp_path / "outputs"),
        postprocess_config,
        watch_postprocess_backfill=False,
    )

    assert action is None
