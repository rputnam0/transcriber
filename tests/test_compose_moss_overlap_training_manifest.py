from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from compose_moss_overlap_training_manifest import compose_records  # noqa: E402


def test_compose_keeps_specialist_and_unique_ordinary_guardrails() -> None:
    overlap = [{"metadata": {"cut_id": "o1", "word_count": 10, "brief_overlap_word_count": 4}}]
    ordinary = [
        {"metadata": {"cut_id": "a", "word_count": 10, "has_overlap": False}},
        {"metadata": {"cut_id": "a", "word_count": 10, "has_overlap": False}},
        {"metadata": {"cut_id": "b", "word_count": 10, "has_overlap": True}},
    ]

    records, summary = compose_records(
        overlap,
        ordinary,
        ordinary_examples=5,
        seed=1,
    )

    assert len(records) == 2
    assert summary["ordinary_examples"] == 1
    assert summary["brief_overlap_word_fraction"] == 0.2


def test_compose_can_use_low_overlap_conversation_as_guardrail() -> None:
    overlap = [{"metadata": {"cut_id": "o1", "word_count": 10, "brief_overlap_word_count": 4}}]
    ordinary = [
        {
            "metadata": {
                "cut_id": "a",
                "word_count": 10,
                "has_overlap": False,
                "brief_overlap_word_fraction": 0.0,
            }
        },
        {
            "metadata": {
                "cut_id": "b",
                "word_count": 10,
                "has_overlap": True,
                "brief_overlap_word_fraction": 0.03,
            }
        },
        {
            "metadata": {
                "cut_id": "c",
                "word_count": 10,
                "has_overlap": True,
                "brief_overlap_word_fraction": 0.20,
            }
        },
    ]

    records, summary = compose_records(
        overlap,
        ordinary,
        ordinary_examples=5,
        seed=1,
        maximum_ordinary_brief_overlap_fraction=0.05,
    )

    assert {row["metadata"]["cut_id"] for row in records} == {"o1", "a", "b"}
    assert summary["ordinary_examples"] == 2
    assert summary["maximum_ordinary_brief_overlap_fraction"] == 0.05
