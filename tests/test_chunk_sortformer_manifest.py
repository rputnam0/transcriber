from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from chunk_sortformer_manifest import chunk_rows  # noqa: E402


def test_chunk_rows_offsets_against_relative_rttm(tmp_path):
    rttm = tmp_path / "sample.rttm"
    rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 10.000 <NA> <NA> a <NA> <NA>",
                "SPEAKER sample 1 20.000 20.000 <NA> <NA> b <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows, summary = chunk_rows(
        [
            {
                "audio_filepath": str(tmp_path / "sample.wav"),
                "offset": 0.0,
                "duration": 60.0,
                "num_speakers": 2,
                "rttm_filepath": str(rttm),
            }
        ],
        chunk_seconds=30.0,
        min_active_speakers=1,
    )

    assert [row["offset"] for row in rows] == [0.0, 30.0]
    assert [row["duration"] for row in rows] == [30.0, 30.0]
    assert [row["num_speakers"] for row in rows] == [2, 1]
    assert summary["speaker_count_distribution"] == {"1": 1, "2": 1}


def test_chunk_rows_can_filter_silent_chunks(tmp_path):
    rttm = tmp_path / "sample.rttm"
    rttm.write_text("SPEAKER sample 1 0.000 5.000 <NA> <NA> a <NA> <NA>\n", encoding="utf-8")

    rows, summary = chunk_rows(
        [
            {
                "audio_filepath": str(tmp_path / "sample.wav"),
                "offset": 0.0,
                "duration": 60.0,
                "num_speakers": 1,
                "rttm_filepath": str(rttm),
            }
        ],
        chunk_seconds=30.0,
        min_active_speakers=1,
    )

    assert len(rows) == 1
    assert rows[0]["offset"] == 0.0
    assert summary["skipped"] == {"too_few_active_speakers": 1}
