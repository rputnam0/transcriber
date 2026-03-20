from __future__ import annotations

import json
from pathlib import Path

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
