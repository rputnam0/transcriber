from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from docx import Document

from transcriber import postprocess as postprocess_mod


def _write_prompt_files(prompts_dir: Path) -> None:
    prompt_files = {
        "analysis_system_prompt.txt": "You are the analyst.",
        "analysis_prompt_template.txt": (
            "BG={avarias_bg}\nOVW={campaign_ovw}\nTRANS={session_trans}\n"
            "INST={session_analysis_inst}\nEX={analysis_example}"
        ),
        "avarias_background.txt": "Background",
        "campaign_overview.txt": "Base overview",
        "session_analysis_instructions.txt": "Analyze it",
        "analysis_example.txt": "Example",
        "co_prompt_template.txt": (
            "PREV={previous_campaign_overview}\nANALYSIS={session_analysis}"
        ),
        "summary_system_prompt.txt": "You are the summarizer.",
        "summary_prompt_template.txt": (
            "BG={avarias_bg}\nOVW={campaign_ovw}\nTRANS={session_trans}\n"
            "ANALYSIS={session_analysis}\nINST={summary_inst}"
        ),
        "summary_instructions.txt": "Summarize it",
    }
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in prompt_files.items():
        (prompts_dir / filename).write_text(content, encoding="utf-8")


def _write_transcription_marker(transcript_path: Path, status: str) -> None:
    marker_path = transcript_path.parent / f"{transcript_path.stem}.transcribe.json"
    marker_path.write_text(json.dumps({"status": status}), encoding="utf-8")


def test_run_postprocess_for_transcript_creates_outputs(monkeypatch, tmp_path):
    prompts_dir = tmp_path / "prompts"
    summaries_dir = tmp_path / "summaries"
    transcript_dir = tmp_path / "transcripts" / "Session 32"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / "Session 32.txt"
    transcript_path.write_text("Speaker 00 00:00:00 Hello there\n", encoding="utf-8")
    _write_prompt_files(prompts_dir)

    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=prompts_dir,
        summaries_dir=summaries_dir,
        api_key_env="GOOGLE_API_KEY",
        calls_per_minute=60,
    )

    class DummyGenerator:
        calls: list[tuple[str, str | None]] = []

        def __init__(self, cfg):
            assert cfg is config

        def generate(self, *, prompt: str, system_instruction: str | None = None) -> str:
            self.calls.append((prompt, system_instruction))
            if len(self.calls) == 1:
                return "# Analysis\n\nThe party solved a puzzle."
            if len(self.calls) == 2:
                return (
                    "<campaign_overview># Campaign Overview\n\nThings changed.</campaign_overview>"
                )
            return "<Summary># Summary\n\nSession recap.</Summary>"

    monkeypatch.setattr(postprocess_mod, "_GoogleTextGenerator", DummyGenerator)

    result = postprocess_mod.run_postprocess_for_transcript(transcript_path, config)

    assert result.created_marker is True
    assert result.paths.analysis_txt.exists()
    assert result.paths.analysis_docx.exists()
    assert result.paths.campaign_overview_txt.exists()
    assert result.paths.campaign_overview_docx.exists()
    assert result.paths.summary_txt.exists()
    assert result.paths.summary_docx.exists()
    assert result.paths.completion_marker.exists()

    marker = json.loads(result.paths.completion_marker.read_text(encoding="utf-8"))
    assert marker["status"] == "completed"
    assert marker["session_number"] == 32
    assert marker["provider"] == "google"

    summary_doc = Document(result.paths.summary_docx)
    joined_text = "\n".join(paragraph.text for paragraph in summary_doc.paragraphs)
    assert "Summary" in joined_text
    assert "Session recap." in joined_text

    second_result = postprocess_mod.run_postprocess_for_transcript(transcript_path, config)
    assert second_result.created_marker is False
    assert len(DummyGenerator.calls) == 3


def test_expected_completion_marker_path_uses_session_folder(tmp_path):
    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
    )

    marker = postprocess_mod.expected_completion_marker_path(
        tmp_path / "transcripts" / "Session 7" / "Session 7.txt",
        config,
    )

    assert marker == tmp_path / "summaries" / "Session 7" / "session_7.postprocess.json"


def test_postprocess_delay_seconds_allows_no_throttle(tmp_path):
    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
        calls_per_minute=0,
    )

    assert config.delay_seconds == 0.0


def test_run_postprocess_for_non_session_transcript_raises_clear_error(tmp_path):
    prompts_dir = tmp_path / "prompts"
    summaries_dir = tmp_path / "summaries"
    transcript_dir = tmp_path / "transcripts" / "Live Smoke Test"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / "Live Smoke Test.txt"
    transcript_path.write_text("Speaker 00 00:00:00 Hello there\n", encoding="utf-8")
    _write_prompt_files(prompts_dir)

    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=prompts_dir,
        summaries_dir=summaries_dir,
    )

    with pytest.raises(postprocess_mod.UnsupportedTranscriptPostprocessError):
        postprocess_mod.run_postprocess_for_transcript(transcript_path, config)


def test_can_postprocess_transcript_requires_explicit_session_name() -> None:
    assert postprocess_mod.can_postprocess_transcript("Session 3.txt")
    assert postprocess_mod.can_postprocess_transcript("Session 28 1_2.txt")
    assert not postprocess_mod.can_postprocess_transcript("kZWVHz3eCBaf.txt")
    assert not postprocess_mod.can_postprocess_transcript(
        "Odds and Ends Transcripts/kZWVHz3eCBaf.txt"
    )


def test_resolve_postprocess_config_accepts_thinking_level(tmp_path):
    cfg = {
        "postprocess": {
            "enabled": True,
            "provider": "google",
            "model": "gemini-3-flash-preview",
            "thinking_level": "high",
            "prompts_dir": str(tmp_path / "prompts"),
            "summaries_dir": str(tmp_path / "summaries"),
        }
    }

    config = postprocess_mod.resolve_postprocess_config(cfg)

    assert config is not None
    assert config.model == "gemini-3-flash-preview"
    assert config.thinking_level == "high"


def test_google_text_generator_passes_thinking_level_for_gemini_3(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            captured["thinking_config_kwargs"] = kwargs
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["generate_config_kwargs"] = kwargs
            self.kwargs = kwargs

    class FakePart:
        @staticmethod
        def from_text(*, text: str):
            return {"text": text}

    class FakeContent:
        def __init__(self, *, role: str, parts: list[object]):
            self.role = role
            self.parts = parts

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return types.SimpleNamespace(text="ok")

    class FakeClient:
        def __init__(self, *, api_key: str):
            captured["api_key"] = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        ThinkingConfig=FakeThinkingConfig,
        GenerateContentConfig=FakeGenerateContentConfig,
        Content=FakeContent,
        Part=FakePart,
    )
    fake_genai = types.SimpleNamespace(Client=FakeClient, types=fake_types)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="gemini-3-flash-preview",
        prompts_dir=tmp_path / "prompts",
        summaries_dir=tmp_path / "summaries",
        thinking_level="high",
    )

    generator = postprocess_mod._GoogleTextGenerator(config)
    text = generator.generate(prompt="hello", system_instruction="system")

    assert text == "ok"
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["thinking_config_kwargs"] == {"thinking_level": "high"}
    assert (
        captured["generate_config_kwargs"]["thinking_config"]
        is captured["config"].kwargs["thinking_config"]
    )


def test_split_session_ready_for_postprocess_requires_all_parts(tmp_path):
    transcript_path = tmp_path / "transcripts" / "Session 28 1_2" / "Session 28 1_2.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("part one", encoding="utf-8")

    assert postprocess_mod.split_session_ready_for_postprocess(transcript_path) is False


def test_split_session_ready_for_postprocess_requires_completed_part_markers(tmp_path):
    transcript_root = tmp_path / "transcripts"
    part_one = transcript_root / "Session 28 1_2" / "Session 28 1_2.txt"
    part_two = transcript_root / "Session 28 2_2" / "Session 28 2_2.txt"
    part_one.parent.mkdir(parents=True, exist_ok=True)
    part_two.parent.mkdir(parents=True, exist_ok=True)
    part_one.write_text("part one", encoding="utf-8")
    part_two.write_text("part two", encoding="utf-8")
    _write_transcription_marker(part_one, "completed")
    _write_transcription_marker(part_two, "in_progress")

    assert postprocess_mod.split_session_ready_for_postprocess(part_one) is False


def test_run_postprocess_for_split_transcript_raises_until_all_parts_exist(monkeypatch, tmp_path):
    prompts_dir = tmp_path / "prompts"
    summaries_dir = tmp_path / "summaries"
    transcript_path = tmp_path / "transcripts" / "Session 28 1_2" / "Session 28 1_2.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("part one", encoding="utf-8")
    _write_prompt_files(prompts_dir)

    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=prompts_dir,
        summaries_dir=summaries_dir,
    )

    class DummyGenerator:
        def __init__(self, _cfg):
            pytest.fail("Generator should not run while split-session parts are missing.")

    monkeypatch.setattr(postprocess_mod, "_GoogleTextGenerator", DummyGenerator)

    with pytest.raises(postprocess_mod.SplitSessionPendingError):
        postprocess_mod.run_postprocess_for_transcript(transcript_path, config)


def test_run_postprocess_for_split_transcript_waits_for_completed_part_markers(
    monkeypatch,
    tmp_path,
):
    prompts_dir = tmp_path / "prompts"
    summaries_dir = tmp_path / "summaries"
    transcript_root = tmp_path / "transcripts"
    part_one = transcript_root / "Session 28 1_2" / "Session 28 1_2.txt"
    part_two = transcript_root / "Session 28 2_2" / "Session 28 2_2.txt"
    part_one.parent.mkdir(parents=True, exist_ok=True)
    part_two.parent.mkdir(parents=True, exist_ok=True)
    part_one.write_text("part one", encoding="utf-8")
    part_two.write_text("part two", encoding="utf-8")
    _write_prompt_files(prompts_dir)
    _write_transcription_marker(part_one, "completed")
    _write_transcription_marker(part_two, "in_progress")

    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=prompts_dir,
        summaries_dir=summaries_dir,
    )

    class DummyGenerator:
        def __init__(self, _cfg):
            pytest.fail("Generator should not run while a split-session part is still in progress.")

    monkeypatch.setattr(postprocess_mod, "_GoogleTextGenerator", DummyGenerator)

    with pytest.raises(postprocess_mod.SplitSessionPendingError):
        postprocess_mod.run_postprocess_for_transcript(part_one, config)


def test_run_postprocess_for_split_transcript_stitches_parts(monkeypatch, tmp_path):
    prompts_dir = tmp_path / "prompts"
    summaries_dir = tmp_path / "summaries"
    transcript_root = tmp_path / "transcripts"
    part_one = transcript_root / "Session 28 1_2" / "Session 28 1_2.txt"
    part_two = transcript_root / "Session 28 2_2" / "Session 28 2_2.txt"
    part_one.parent.mkdir(parents=True, exist_ok=True)
    part_two.parent.mkdir(parents=True, exist_ok=True)
    part_one.write_text("Speaker 00 00:00:00 Part one\n", encoding="utf-8")
    part_two.write_text("Speaker 00 00:05:00 Part two\n", encoding="utf-8")
    _write_prompt_files(prompts_dir)
    _write_transcription_marker(part_one, "completed")
    _write_transcription_marker(part_two, "completed")

    config = postprocess_mod.PostProcessConfig(
        enabled=True,
        provider="google",
        model="test-model",
        prompts_dir=prompts_dir,
        summaries_dir=summaries_dir,
    )

    class DummyGenerator:
        calls: list[tuple[str, str | None]] = []

        def __init__(self, cfg):
            assert cfg is config

        def generate(self, *, prompt: str, system_instruction: str | None = None) -> str:
            self.calls.append((prompt, system_instruction))
            if len(self.calls) == 1:
                assert "Part one" in prompt
                assert "Part two" in prompt
                assert prompt.index("Part one") < prompt.index("Part two")
                return "# Analysis\n\nSplit session stitched."
            if len(self.calls) == 2:
                return "<campaign_overview># Campaign Overview\n\nUpdated.</campaign_overview>"
            return "<Summary># Summary\n\nCombined session.</Summary>"

    monkeypatch.setattr(postprocess_mod, "_GoogleTextGenerator", DummyGenerator)

    result = postprocess_mod.run_postprocess_for_transcript(part_two, config)

    assert result.paths.session_folder == summaries_dir / "Session 28"
    assert result.paths.summary_txt == (
        summaries_dir / "Session 28" / "raw_txt" / "session_28_summary_output.txt"
    )
    marker = json.loads(result.paths.completion_marker.read_text(encoding="utf-8"))
    assert marker["source_transcript_paths"] == [str(part_one), str(part_two)]
    assert marker["split_session"] == {"part_total": 2, "source_transcript_count": 2}
