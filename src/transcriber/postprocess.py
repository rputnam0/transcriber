from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

logger = logging.getLogger(__name__)

_PROMPT_FILES = {
    "analysis_system_prompt": "analysis_system_prompt.txt",
    "analysis_prompt_template": "analysis_prompt_template.txt",
    "avarias_background": "avarias_background.txt",
    "base_campaign_overview": "campaign_overview.txt",
    "session_analysis_instructions": "session_analysis_instructions.txt",
    "analysis_example": "analysis_example.txt",
    "campaign_overview_prompt_template": "co_prompt_template.txt",
    "summary_system_prompt": "summary_system_prompt.txt",
    "summary_prompt_template": "summary_prompt_template.txt",
    "summary_instructions": "summary_instructions.txt",
}


@dataclass(frozen=True)
class PostProcessConfig:
    enabled: bool
    provider: str
    model: str
    prompts_dir: Path
    summaries_dir: Path
    api_key_env: str = "GOOGLE_API_KEY"
    raw_text_subdir: str = "raw_txt"
    calls_per_minute: int = 5
    skip_existing: bool = True

    @property
    def delay_seconds(self) -> float:
        return 60.0 / max(1, int(self.calls_per_minute))


@dataclass(frozen=True)
class PostProcessPaths:
    transcript_path: Path
    session_number: int
    session_folder: Path
    raw_text_folder: Path
    analysis_txt: Path
    analysis_docx: Path
    campaign_overview_txt: Path
    campaign_overview_docx: Path
    summary_txt: Path
    summary_docx: Path
    completion_marker: Path


@dataclass(frozen=True)
class PostProcessResult:
    paths: PostProcessPaths
    created_marker: bool


class _GoogleTextGenerator:
    def __init__(self, config: PostProcessConfig) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - guarded by dependency install
            raise RuntimeError(
                "Google post-processing requires the google-genai package. Run `uv sync`."
            ) from exc

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key for post-processing. Set ${config.api_key_env} in the environment."
            )

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._config = config
        self._last_completed_at: Optional[float] = None

    def generate(self, *, prompt: str, system_instruction: Optional[str] = None) -> str:
        if self._last_completed_at is not None:
            elapsed = time.monotonic() - self._last_completed_at
            wait_for = self._config.delay_seconds - elapsed
            if wait_for > 0:
                time.sleep(wait_for)

        config_kwargs = {"response_mime_type": "text/plain"}
        if system_instruction:
            config_kwargs["system_instruction"] = [system_instruction]

        response = self._client.models.generate_content(
            model=self._config.model,
            contents=[
                self._types.Content(
                    role="user",
                    parts=[self._types.Part.from_text(text=prompt)],
                )
            ],
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        self._last_completed_at = time.monotonic()

        text = (getattr(response, "text", None) or "").strip()
        if text:
            return text

        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            raise RuntimeError(f"Google post-processing blocked the request: {block_reason}")
        raise RuntimeError("Google post-processing returned an empty response.")


def resolve_postprocess_config(cfg: Mapping[str, object] | None) -> Optional[PostProcessConfig]:
    if not isinstance(cfg, Mapping):
        return None

    raw_cfg = cfg.get("postprocess") or {}
    if not isinstance(raw_cfg, Mapping):
        return None
    if not bool(raw_cfg.get("enabled")):
        return None

    prompts_dir = str(raw_cfg.get("prompts_dir") or "").strip()
    summaries_dir = str(raw_cfg.get("summaries_dir") or "").strip()
    model = str(raw_cfg.get("model") or "").strip()
    provider = str(raw_cfg.get("provider") or "google").strip().lower() or "google"
    api_key_env = str(raw_cfg.get("api_key_env") or "GOOGLE_API_KEY").strip() or "GOOGLE_API_KEY"
    raw_text_subdir = str(raw_cfg.get("raw_text_subdir") or "raw_txt").strip() or "raw_txt"
    calls_per_minute = int(raw_cfg.get("calls_per_minute") or 5)
    skip_existing = bool(raw_cfg.get("skip_existing", True))

    if provider != "google":
        raise SystemExit(f"Unsupported postprocess.provider={provider!r}; expected 'google'.")
    if not prompts_dir:
        raise SystemExit("postprocess.prompts_dir is required when post-processing is enabled.")
    if not summaries_dir:
        raise SystemExit("postprocess.summaries_dir is required when post-processing is enabled.")
    if not model:
        raise SystemExit("postprocess.model is required when post-processing is enabled.")

    return PostProcessConfig(
        enabled=True,
        provider=provider,
        model=model,
        prompts_dir=Path(prompts_dir).expanduser(),
        summaries_dir=Path(summaries_dir).expanduser(),
        api_key_env=api_key_env,
        raw_text_subdir=raw_text_subdir,
        calls_per_minute=max(1, calls_per_minute),
        skip_existing=skip_existing,
    )


def expected_completion_marker_path(
    transcript_path: str | Path,
    config: PostProcessConfig,
) -> Path:
    return build_postprocess_paths(transcript_path, config).completion_marker


def build_postprocess_paths(
    transcript_path: str | Path,
    config: PostProcessConfig,
) -> PostProcessPaths:
    transcript = Path(transcript_path).expanduser()
    session_number = _extract_session_number(transcript)
    session_prefix = f"session_{session_number}"
    session_folder = config.summaries_dir / f"Session {session_number}"
    raw_text_folder = session_folder / config.raw_text_subdir
    return PostProcessPaths(
        transcript_path=transcript,
        session_number=session_number,
        session_folder=session_folder,
        raw_text_folder=raw_text_folder,
        analysis_txt=raw_text_folder / f"{session_prefix}_analysis_output.txt",
        analysis_docx=session_folder / f"{session_prefix}_analysis_output.docx",
        campaign_overview_txt=raw_text_folder / f"{session_prefix}_campaign_overview.txt",
        campaign_overview_docx=session_folder / f"{session_prefix}_campaign_overview.docx",
        summary_txt=raw_text_folder / f"{session_prefix}_summary_output.txt",
        summary_docx=session_folder / f"{session_prefix}_summary_output.docx",
        completion_marker=session_folder / f"{session_prefix}.postprocess.json",
    )


def run_postprocess_for_transcript(
    transcript_path: str | Path,
    config: PostProcessConfig,
) -> PostProcessResult:
    paths = build_postprocess_paths(transcript_path, config)
    if not paths.transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript not found for post-processing: {paths.transcript_path}"
        )

    if config.skip_existing and _postprocess_complete(paths):
        logger.info("Post-process already complete for session %s", paths.session_number)
        return PostProcessResult(paths=paths, created_marker=False)

    prompts = _load_prompt_texts(config.prompts_dir)
    previous_campaign_overview = _resolve_previous_campaign_overview(paths, config, prompts)
    transcript_text = _read_text(paths.transcript_path)
    client = _GoogleTextGenerator(config)

    analysis_text, _ = _materialize_text_and_docx(
        text_path=paths.analysis_txt,
        docx_path=paths.analysis_docx,
        skip_existing=config.skip_existing,
        generate_text=lambda: _generate_analysis(
            client=client,
            prompts=prompts,
            transcript_text=transcript_text,
            previous_campaign_overview=previous_campaign_overview,
        ),
    )
    campaign_overview_text, _ = _materialize_text_and_docx(
        text_path=paths.campaign_overview_txt,
        docx_path=paths.campaign_overview_docx,
        skip_existing=config.skip_existing,
        generate_text=lambda: _generate_campaign_overview(
            client=client,
            prompts=prompts,
            previous_campaign_overview=previous_campaign_overview,
            analysis_text=analysis_text,
        ),
    )
    summary_text, _ = _materialize_text_and_docx(
        text_path=paths.summary_txt,
        docx_path=paths.summary_docx,
        skip_existing=config.skip_existing,
        generate_text=lambda: _generate_summary(
            client=client,
            prompts=prompts,
            transcript_text=transcript_text,
            analysis_text=analysis_text,
            campaign_overview_text=campaign_overview_text,
        ),
    )

    metadata = {
        "status": "completed",
        "provider": config.provider,
        "model": config.model,
        "session_number": paths.session_number,
        "transcript_path": str(paths.transcript_path),
        "previous_campaign_overview_path": str(
            _resolve_previous_campaign_overview_path(paths, config)
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs": {
            "analysis_txt": str(paths.analysis_txt),
            "analysis_docx": str(paths.analysis_docx),
            "campaign_overview_txt": str(paths.campaign_overview_txt),
            "campaign_overview_docx": str(paths.campaign_overview_docx),
            "summary_txt": str(paths.summary_txt),
            "summary_docx": str(paths.summary_docx),
        },
        "summary_preview": summary_text[:200],
    }
    _atomic_write_text(paths.completion_marker, json.dumps(metadata, indent=2) + "\n")
    logger.warning(
        "Post-process complete for session %s (summary=%s)",
        paths.session_number,
        paths.summary_docx,
    )
    return PostProcessResult(paths=paths, created_marker=True)


def save_markdown_as_docx(markdown_text: str, docx_path: str | Path) -> None:
    try:
        import markdown as markdown_lib
        from bs4 import BeautifulSoup, NavigableString, Tag
        from docx import Document
        from docx.oxml.ns import qn
    except ImportError as exc:  # pragma: no cover - guarded by dependency install
        raise RuntimeError(
            "DOCX export requires markdown, beautifulsoup4, and python-docx. Run `uv sync`."
        ) from exc

    def set_run_mono(run) -> None:
        run.font.name = "Courier New"
        try:
            r_fonts = run._element.rPr.rFonts
            r_fonts.set(qn("w:eastAsia"), "Courier New")
            r_fonts.set(qn("w:ascii"), "Courier New")
            r_fonts.set(qn("w:hAnsi"), "Courier New")
        except Exception:
            pass

    def add_inline(paragraph, node) -> None:
        if isinstance(node, NavigableString):
            text = str(node)
            if text:
                paragraph.add_run(text)
            return
        if not isinstance(node, Tag):
            return

        name = node.name.lower()
        if name in {"strong", "b"}:
            run = paragraph.add_run(node.get_text())
            run.bold = True
            return
        if name in {"em", "i"}:
            run = paragraph.add_run(node.get_text())
            run.italic = True
            return
        if name == "code":
            run = paragraph.add_run(node.get_text())
            set_run_mono(run)
            return
        for child in node.children:
            add_inline(paragraph, child)

    def add_paragraph(document, paragraph_tag) -> None:
        paragraph = document.add_paragraph()
        for child in paragraph_tag.children:
            add_inline(paragraph, child)

    def add_list(document, list_tag, *, ordered: bool) -> None:
        style_name = "List Number" if ordered else "List Bullet"
        for item in list_tag.find_all("li", recursive=False):
            paragraph = document.add_paragraph(style=style_name)
            for child in item.children:
                if isinstance(child, Tag) and child.name.lower() in {"ul", "ol"}:
                    continue
                add_inline(paragraph, child)
            for nested in item.find_all(["ul", "ol"], recursive=False):
                add_list(document, nested, ordered=nested.name.lower() == "ol")

    def add_code_block(document, text: str) -> None:
        for line in text.splitlines() or [""]:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(line)
            set_run_mono(run)

    def add_table(document, table_tag) -> None:
        rows = table_tag.find_all("tr")
        if not rows:
            return
        first_row = rows[0].find_all(["th", "td"])
        table = document.add_table(rows=len(rows), cols=len(first_row))
        for row_index, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            for col_index, cell in enumerate(cells):
                table.cell(row_index, col_index).text = cell.get_text()

    html = markdown_lib.markdown(
        markdown_text,
        extensions=["fenced_code", "tables", "sane_lists"],
    )
    soup = BeautifulSoup(html, "html.parser")
    document = Document()

    for element in soup.contents:
        if isinstance(element, NavigableString):
            if str(element).strip():
                document.add_paragraph(str(element).strip())
            continue
        if not isinstance(element, Tag):
            continue

        name = element.name.lower()
        if name in {"h1", "h2", "h3", "h4"}:
            document.add_heading(element.get_text(), level=int(name[1]))
        elif name == "p":
            add_paragraph(document, element)
        elif name == "ul":
            add_list(document, element, ordered=False)
        elif name == "ol":
            add_list(document, element, ordered=True)
        elif name == "blockquote":
            try:
                document.add_paragraph(element.get_text(), style="Intense Quote")
            except Exception:
                paragraph = document.add_paragraph()
                run = paragraph.add_run(element.get_text())
                run.italic = True
        elif name == "pre":
            code_node = element.find("code")
            add_code_block(document, code_node.get_text() if code_node else element.get_text())
        elif name == "table":
            add_table(document, element)
        else:
            document.add_paragraph(element.get_text())

    _atomic_save_docx(document, Path(docx_path).expanduser())


def _extract_session_number(path: Path) -> int:
    session_pattern = re.compile(r"session[\s_-]*(\d+)", re.IGNORECASE)
    plain_number_pattern = re.compile(r"(?<!\d)(\d+)(?!\d)")
    candidates = [
        path.stem,
        path.name,
        path.parent.name,
        path.parent.parent.name,
    ]
    for candidate in candidates:
        match = session_pattern.search(candidate)
        if match:
            return int(match.group(1))
    for candidate in candidates:
        match = plain_number_pattern.search(candidate)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer a session number from transcript path: {path}")


def _load_prompt_texts(prompts_dir: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for key, filename in _PROMPT_FILES.items():
        prompt_path = prompts_dir / filename
        loaded[key] = _read_text(prompt_path)
    return loaded


def _resolve_previous_campaign_overview(
    paths: PostProcessPaths,
    config: PostProcessConfig,
    prompts: Mapping[str, str],
) -> str:
    previous_path = _resolve_previous_campaign_overview_path(paths, config)
    if previous_path.exists():
        return _read_text(previous_path)
    logger.info(
        "Previous campaign overview missing for session %s; using base prompt file.",
        paths.session_number,
    )
    return prompts["base_campaign_overview"]


def _resolve_previous_campaign_overview_path(
    paths: PostProcessPaths,
    config: PostProcessConfig,
) -> Path:
    if paths.session_number <= 1:
        return config.prompts_dir / _PROMPT_FILES["base_campaign_overview"]
    previous_session = paths.session_number - 1
    return (
        config.summaries_dir
        / f"Session {previous_session}"
        / config.raw_text_subdir
        / f"session_{previous_session}_campaign_overview.txt"
    )


def _generate_analysis(
    *,
    client: _GoogleTextGenerator,
    prompts: Mapping[str, str],
    transcript_text: str,
    previous_campaign_overview: str,
) -> str:
    prompt = prompts["analysis_prompt_template"].format(
        avarias_bg=prompts["avarias_background"],
        campaign_ovw=previous_campaign_overview,
        session_trans=transcript_text,
        session_analysis_inst=prompts["session_analysis_instructions"],
        analysis_example=prompts["analysis_example"],
    )
    return client.generate(
        prompt=prompt,
        system_instruction=prompts["analysis_system_prompt"],
    )


def _generate_campaign_overview(
    *,
    client: _GoogleTextGenerator,
    prompts: Mapping[str, str],
    previous_campaign_overview: str,
    analysis_text: str,
) -> str:
    prompt = prompts["campaign_overview_prompt_template"].format(
        previous_campaign_overview=previous_campaign_overview,
        session_analysis=analysis_text,
    )
    response_text = client.generate(prompt=prompt)
    extracted = _extract_tagged_text(response_text, "campaign_overview")
    if extracted:
        return extracted
    logger.warning(
        "Campaign overview response was missing <campaign_overview> tags; using full text."
    )
    return response_text


def _generate_summary(
    *,
    client: _GoogleTextGenerator,
    prompts: Mapping[str, str],
    transcript_text: str,
    analysis_text: str,
    campaign_overview_text: str,
) -> str:
    prompt = prompts["summary_prompt_template"].format(
        avarias_bg=prompts["avarias_background"],
        campaign_ovw=campaign_overview_text,
        session_trans=transcript_text,
        session_analysis=analysis_text,
        summary_inst=prompts["summary_instructions"],
    )
    response_text = client.generate(
        prompt=prompt,
        system_instruction=prompts["summary_system_prompt"],
    )
    extracted = _extract_tagged_text(response_text, "summary")
    if extracted:
        return extracted
    logger.warning("Summary response was missing <Summary> tags; using full text.")
    return response_text


def _extract_tagged_text(text: str, tag_name: str) -> str:
    pattern = re.compile(
        rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _materialize_text_and_docx(
    *,
    text_path: Path,
    docx_path: Path,
    skip_existing: bool,
    generate_text: Callable[[], str],
) -> tuple[str, bool]:
    if skip_existing and text_path.exists():
        text = _read_text(text_path)
        if not docx_path.exists():
            save_markdown_as_docx(text, docx_path)
        return text, False

    text = generate_text().strip()
    if not text:
        raise RuntimeError(f"Refusing to write empty post-process output: {text_path}")
    _atomic_write_text(text_path, text + "\n")
    save_markdown_as_docx(text, docx_path)
    return text, True


def _postprocess_complete(paths: PostProcessPaths) -> bool:
    required_files = (
        paths.analysis_txt,
        paths.analysis_docx,
        paths.campaign_overview_txt,
        paths.campaign_overview_docx,
        paths.summary_txt,
        paths.summary_docx,
        paths.completion_marker,
    )
    return all(path.exists() for path in required_files)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file:
            tmp_file.write(text)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_save_docx(document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".docx.tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        document.save(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
