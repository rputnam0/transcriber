from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from analyze_multitalker_parakeet_results import (  # noqa: E402
    lexical_decomposition,
    summarize,
)


def test_lexical_decomposition_separates_recognition_from_ownership() -> None:
    reference = [
        {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "hello"},
        {"speaker": "Bob", "start": 1.0, "end": 2.0, "text": "there"},
    ]
    predicted = [{"speaker": "S0", "start_time": 0.0, "end_time": 1.0, "words": "hello there"}]

    result = lexical_decomposition(reference, predicted)

    assert result["bag_lexical_matched_words"] == 2
    assert result["bag_one_to_one_attributed_matched_words"] == 1
    assert result["bag_many_to_one_stream_matched_words_proxy"] == 1


def test_summarize_reports_owner_fraction_given_recognized_words() -> None:
    report = summarize(
        [
            {
                "reference_words": 10,
                "predicted_words": 8,
                "bag_lexical_matched_words": 6,
                "bag_one_to_one_attributed_matched_words": 5,
                "bag_many_to_one_stream_matched_words_proxy": 5,
            }
        ]
    )

    assert report["bag_lexical_recall_upper_bound"] == 0.6
    assert report["bag_owner_fraction_given_lexical_match"] == 5 / 6
