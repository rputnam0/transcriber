from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_moss_speaker_attributed import (  # noqa: E402
    _optimal_unique_mapping,
    materialize_cut_results,
    score_mapping,
)


def test_oracle_mapping_uses_whole_stream_lexical_ownership() -> None:
    references = {"Alice": "red apple yes", "Bob": "blue pear no"}
    predictions = {"S01": "blue pear", "S02": "red apple yes"}

    mapping = _optimal_unique_mapping(predictions, references)
    score = score_mapping(predictions, references, mapping)

    assert mapping == {"S01": "Bob", "S02": "Alice"}
    assert score["bag_matches"] == 5
    assert score["attributed_bag_recall"] == 5 / 6
    assert score["attributed_bag_precision"] == 1.0


def test_unmapped_stream_counts_against_precision() -> None:
    score = score_mapping(
        {"S01": "hello", "S02": "intruding words"},
        {"Alice": "hello"},
        {"S01": "Alice"},
    )

    assert score["reference_words"] == 1
    assert score["predicted_words"] == 3
    assert score["bag_matches"] == 1
    assert score["attributed_bag_precision"] == 1 / 3


def test_materialized_results_use_contiguous_timeline_offsets(tmp_path: Path) -> None:
    cuts = [
        {
            "id": "session_34_w000240_c000000-mask-sortformer",
            "duration": 30.0,
            "custom": {
                "transcript_spans": [
                    {"speaker": "Alice", "text": "first"},
                ]
            },
        },
        {
            "id": "session_34_w000240_c030000-mask-sortformer",
            "duration": 30.0,
            "custom": {
                "transcript_spans": [
                    {"speaker": "Alice", "text": "second"},
                ]
            },
        },
    ]
    segments = [
        {"start": 1.0, "end": 2.0, "speaker": "S01", "text": "first"},
        {"start": 31.0, "end": 32.0, "speaker": "S01", "text": "second"},
    ]

    materialize_cut_results(
        cuts,
        segments,
        {"S01": "Alice"},
        tmp_path,
        mapping_is_oracle=False,
    )

    first = (tmp_path / "session_34_w000240_c000000.json").read_text()
    second = (tmp_path / "session_34_w000240_c030000.json").read_text()
    assert '"prediction": "first"' in first
    assert '"prediction": "second"' in second
    assert '"mapping_is_oracle": false' in first
