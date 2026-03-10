from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Keep environment predictable for run_transcribe tests."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)


def _fake_segments_payload() -> Dict[str, Any]:
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "speaker": "SPEAKER_00",
            }
        ],
        "extra": {"note": "metadata that should be ignored"},
    }


def test_run_transcribe_normalises_segment_payload(monkeypatch, tmp_path):
    """Ensure dict-based segment payloads are converted before consolidation."""
    from transcriber import cli as cli_mod

    fake_input = tmp_path / "input.zip"
    fake_input.write_text("dummy")

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(tmp_path / "track1.wav")], None)

    def fake_save_outputs(
        *,
        base_stem: str,
        output_dir: str,
        per_file_segments: List[Tuple[str, List[Dict[str, Any]]]],
        consolidated_pairs: List[Tuple[str, str, str]],
        diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        write_srt_file: bool,
        write_jsonl_file: bool,
    ) -> Path:
        # Expect normalised tuples with list-of-dict segments
        assert per_file_segments and len(per_file_segments[0]) == 2
        filename, segments = per_file_segments[0]
        assert filename.endswith("track1.wav")
        assert isinstance(segments, list)
        assert segments[0]["text"] == "hello world"
        # consolidated data should match the normalised structure
        assert consolidated_pairs[0][2] == "hello world"
        if diar_by_file:
            diar_entry = diar_by_file[filename][0]
            assert diar_entry["speaker"] == "SPEAKER_00"
        return Path(output_dir)

    def fake_transcribe_with_whisperx(*args: Any, **kwargs: Any):
        segments = _fake_segments_payload()
        diar = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}
        speaker_embeddings: Dict[str, Any] = {}
        return segments, diar, speaker_embeddings

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr("transcriber.whisperx_backend.transcribe_with_whisperx", fake_transcribe_with_whisperx)
    monkeypatch.setattr("transcriber.whisperx_backend._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="whisperx",
        model_name="tiny",
        compute_type="int8",
        batch_size=4,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=None,
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=None,
    )
