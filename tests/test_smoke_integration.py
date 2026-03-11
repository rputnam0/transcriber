from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from transcriber.consolidate import consolidate, save_outputs


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _smoke_enabled() -> bool:
    return os.getenv("TRANSCRIBER_RUN_SMOKE") == "1"


def _require_smoke_enabled() -> None:
    if not _smoke_enabled():
        pytest.skip("Set TRANSCRIBER_RUN_SMOKE=1 to run integration smoke tests.")


def _fixture_audio_path() -> Path:
    configured = os.getenv("TRANSCRIBER_SMOKE_AUDIO")
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        candidate = (Path(__file__).resolve().parents[1] / "audio" / "Session 15.m4a").resolve()

    if not candidate.exists():
        pytest.skip(f"Smoke audio fixture not found: {candidate}")
    return candidate


def _smoke_seconds() -> int:
    value = int(os.getenv("TRANSCRIBER_SMOKE_SECONDS", "300"))
    return max(1, value)


def _smoke_local_files_only() -> bool:
    return os.getenv("TRANSCRIBER_SMOKE_LOCAL_ONLY") == "1"


def _trim_audio_fixture(source: Path, target: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for smoke integration tests.")

    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            str(_smoke_seconds()),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ],
        check=True,
    )
    return target


def _assert_transcript_shape(out_dir: Path, base_stem: str, *, expect_words: bool) -> None:
    txt_path = out_dir / f"{base_stem}.txt"
    srt_path = out_dir / f"{base_stem}.srt"
    jsonl_path = out_dir / f"{base_stem}.jsonl"

    assert txt_path.exists()
    assert srt_path.exists()
    assert jsonl_path.exists()

    txt_lines = [
        line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert txt_lines

    jsonl_records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert jsonl_records

    first = jsonl_records[0]
    assert set(first).issuperset({"file", "start", "end", "speaker", "text"})
    assert isinstance(first["text"], str)
    assert first["end"] >= first["start"]
    if expect_words:
        assert "words" in first


def test_faster_whisper_audio_to_transcript_smoke(tmp_path):
    _require_smoke_enabled()

    from transcriber.transcript_pipeline import transcribe_with_faster_pipeline

    clip_path = _trim_audio_fixture(_fixture_audio_path(), tmp_path / "smoke.wav")
    result = transcribe_with_faster_pipeline(
        str(clip_path),
        model_name=os.getenv("TRANSCRIBER_SMOKE_FASTER_MODEL", "tiny"),
        compute_type=os.getenv("TRANSCRIBER_SMOKE_FASTER_COMPUTE", "int8"),
        batch_size=int(os.getenv("TRANSCRIBER_SMOKE_FASTER_BATCH_SIZE", "4")),
        force_device=os.getenv("TRANSCRIBER_SMOKE_FASTER_DEVICE", "cpu"),
        local_files_only=_smoke_local_files_only(),
        enable_diarization=False,
    )

    assert result.segments
    assert any(segment.get("text", "").strip() for segment in result.segments)

    per_file_segments = [(str(clip_path), result.segments)]
    out_dir = save_outputs(
        base_stem="faster_smoke",
        output_dir=str(tmp_path / "out_faster"),
        per_file_segments=per_file_segments,
        consolidated_pairs=consolidate(per_file_segments),
        diar_by_file=None,
        exclusive_diar_by_file=None,
        write_srt_file=True,
        write_jsonl_file=True,
    )

    _assert_transcript_shape(out_dir, "faster_smoke", expect_words=True)


def test_parakeet_audio_to_transcript_smoke(tmp_path):
    _require_smoke_enabled()

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine != "arm64":
        pytest.skip("Parakeet MLX smoke tests require Apple Silicon on macOS.")
    if system not in {"Darwin", "Linux"}:
        pytest.skip("Parakeet smoke tests are only supported on Darwin and Linux.")

    from transcriber.parakeet_backend import load_model, transcribe_file

    clip_path = _trim_audio_fixture(_fixture_audio_path(), tmp_path / "smoke.wav")
    model = load_model(
        os.getenv("TRANSCRIBER_SMOKE_PARAKEET_MODEL", "parakeet"),
        compute_type=os.getenv("TRANSCRIBER_SMOKE_PARAKEET_COMPUTE", "float16"),
        device=os.getenv("TRANSCRIBER_SMOKE_PARAKEET_DEVICE", "auto"),
        local_files_only=_smoke_local_files_only(),
    )
    segments = transcribe_file(str(clip_path), model, batch_size=1)

    assert segments
    assert any(segment.get("text", "").strip() for segment in segments)

    per_file_segments = [(str(clip_path), segments)]
    out_dir = save_outputs(
        base_stem="parakeet_smoke",
        output_dir=str(tmp_path / "out_parakeet"),
        per_file_segments=per_file_segments,
        consolidated_pairs=consolidate(per_file_segments),
        diar_by_file=None,
        exclusive_diar_by_file=None,
        write_srt_file=True,
        write_jsonl_file=True,
    )

    _assert_transcript_shape(out_dir, "parakeet_smoke", expect_words=False)
