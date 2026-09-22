from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_clean_speaker_asr_dataset import round_robin_rows, target_speaker_spans  # noqa: E402


def test_target_speaker_spans_excludes_other_speakers() -> None:
    row = {
        "speaker_id": "Alice",
        "word_spans": [
            {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "hello"},
            {"speaker": "Bob", "start": 0.5, "end": 1.5, "text": "yes"},
        ],
    }

    assert [span["text"] for span in target_speaker_spans(row)] == ["hello"]


def test_round_robin_rows_interleaves_speakers_and_filters_split() -> None:
    rows = [
        {"split_id": "train", "speaker_id": "Alice", "row_id": "a1"},
        {"split_id": "train", "speaker_id": "Alice", "row_id": "a2"},
        {"split_id": "train", "speaker_id": "Bob", "row_id": "b1"},
        {"split_id": "dev", "speaker_id": "Bob", "row_id": "b2"},
    ]

    assert [row["row_id"] for row in round_robin_rows(rows, split="train")] == [
        "a1",
        "b1",
        "a2",
    ]
