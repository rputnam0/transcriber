from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from export_sortformer_training_data import (  # noqa: E402
    _clamp_and_merge_intervals,
    _write_rttm,
    export_sortformer_manifests,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_write_rttm_sanitizes_speaker_names(tmp_path):
    rttm = tmp_path / "sample.rttm"

    speaker_map = _write_rttm(
        rttm,
        file_id="sample",
        intervals=[
            {"speaker": "Dungeon Master", "start": 0.0, "end": 1.25},
            {"speaker": "Cyrus Schwert", "start": 1.5, "end": 2.0},
        ],
    )

    assert speaker_map == {
        "Cyrus Schwert": "cyrus_schwert",
        "Dungeon Master": "dungeon_master",
    }
    assert rttm.read_text(encoding="utf-8").splitlines() == [
        "SPEAKER sample 1 0.000 1.250 <NA> <NA> dungeon_master <NA> <NA>",
        "SPEAKER sample 1 1.500 0.500 <NA> <NA> cyrus_schwert <NA> <NA>",
    ]


def test_clamp_and_merge_intervals_merges_per_speaker_only():
    merged = _clamp_and_merge_intervals(
        [
            {"speaker": "A", "start": -1.0, "end": 0.2},
            {"speaker": "A", "start": 0.25, "end": 0.5},
            {"speaker": "B", "start": 0.3, "end": 0.4},
            {"speaker": "A", "start": 2.0, "end": 10.0},
        ],
        duration=3.0,
        merge_gap_seconds=0.1,
    )

    assert merged == [
        {"speaker": "A", "start": 0.0, "end": 0.5},
        {"speaker": "B", "start": 0.3, "end": 0.4},
        {"speaker": "A", "start": 2.0, "end": 3.0},
    ]


def test_export_sortformer_manifests_prefers_forced_reference_and_groups_rows(tmp_path):
    mixture = tmp_path / "mixture.wav"
    mixture.write_bytes(b"fake wav for manifest export")
    manifest = tmp_path / "speaker_manifest.jsonl"
    refs = tmp_path / "refs.jsonl"
    rows = [
        {
            "row_id": "row-a",
            "split_id": "dev",
            "session": "Session 1",
            "window_start": 10.0,
            "window_end": 20.0,
            "duration": 10.0,
            "speaker_id": "A",
            "materialized": {"mixture_path": str(mixture)},
            "word_spans": [{"speaker": "A", "start": 0.0, "end": 9.0}],
        },
        {
            "row_id": "row-b",
            "split_id": "dev",
            "session": "Session 1",
            "window_start": 10.0,
            "window_end": 20.0,
            "duration": 10.0,
            "speaker_id": "B",
            "materialized": {"mixture_path": str(mixture)},
            "word_spans": [{"speaker": "B", "start": 0.0, "end": 9.0}],
        },
    ]
    _write_jsonl(manifest, rows)
    _write_jsonl(
        refs,
        [
            {
                "session": "Session 1",
                "window_start": 10.0,
                "window_end": 20.0,
                "words": [
                    {"speaker": "A", "start": 0.0, "end": 0.5, "text": "hello"},
                    {"speaker": "B", "start": 0.25, "end": 0.75, "text": "there"},
                ],
            }
        ],
    )

    summary = export_sortformer_manifests(
        manifest_path=manifest,
        reference_jsonl=refs,
        output_dir=tmp_path / "export",
        splits={"dev"},
    )

    assert summary["groups"] == 1
    assert summary["by_split"]["dev"]["groups"] == 1
    assert summary["reference_sources"] == {"forced": 1}

    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "export" / "dev_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["num_speakers"] == 2
    assert Path(manifest_rows[0]["audio_filepath"]).exists()
    rttm_text = Path(manifest_rows[0]["rttm_filepath"]).read_text(encoding="utf-8")
    assert " a " in rttm_text
    assert " b " in rttm_text
