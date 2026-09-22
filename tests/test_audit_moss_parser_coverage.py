import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_moss_parser_coverage import audit  # noqa: E402


def test_audit_detects_dropped_dialogue_and_unprocessed_audio_without_rewriting(tmp_path):
    records = [
        {"cut_id": "00000", "duration": 30, "sha256": "first-audio"},
        {"cut_id": "00001", "duration": 30, "sha256": "second-audio"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(records))
    (tmp_path / "predictions").mkdir()
    first = {"start": 1.0, "end": 2.0, "speaker": "S01", "text": "Before."}
    prediction = {
        "sha256": "first-audio",
        "raw_text": "[1][S01] Before.[2][3][S02] This must survive.[4]",
        "segments": [first],
    }
    path = tmp_path / "predictions/00000.json"
    path.write_text(json.dumps(prediction))
    original = path.read_bytes()
    report = audit(tmp_path, lambda raw: [SimpleNamespace(**first)])
    assert report["missing_chunks"] == ["00001"]
    assert report["errors"][0]["expected_turns"] == 2
    assert "cached_segments_differ_from_reparsed_output" in report["errors"][0]["issues"]
    assert path.read_bytes() == original


def test_audit_accepts_complete_repaired_cache_with_provenance(tmp_path):
    record = {"cut_id": "00000", "duration": 30, "sha256": "audio"}
    (tmp_path / "manifest.json").write_text(json.dumps([record]))
    (tmp_path / "predictions").mkdir()
    first = {"start": 1.0, "end": 2.0, "speaker": "S01", "text": "Before."}
    second = {"start": 3.0, "end": 4.0, "speaker": "S02", "text": "This must survive."}
    prediction = {
        "sha256": "audio",
        "raw_text": "[1][S01] Before.[2][3][S02} This must survive.[4]",
        "segments": [first, second],
        "parser_repair": {"rule": "recover bounded turns and speaker closing braces"},
    }
    (tmp_path / "predictions/00000.json").write_text(json.dumps(prediction))
    report = audit(tmp_path, lambda raw: [SimpleNamespace(**first)])
    assert report["audited_chunks"] == report["expected_chunks"] == 1
    assert report["errors"] == report["missing_chunks"] == []
