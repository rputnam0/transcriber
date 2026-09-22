"""Guard against optimistic overlap and repeated-word scores in the Mac experiment."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_moss_mac import reference_words, combine, named_edit_errors  # noqa: E402
from score_moss_manifest import temporal_attributed_score  # noqa: E402


def test_word_intervals_do_not_label_silent_sentence_gaps_as_overlap():
    transcripts = {
        "alice": [
            {
                "words": [
                    {"word": "one", "start": 0, "end": 0.4},
                    {"word": "two", "start": 3, "end": 3.4},
                ]
            }
        ],
        "bob": [
            {
                "words": [
                    {"word": "gap", "start": 1, "end": 1.4},
                    {"word": "interrupt", "start": 3.1, "end": 3.3},
                ]
            }
        ],
    }
    masks = {name: np.ones(1000, bool) for name in transcripts}
    words = reference_words(transcripts, masks, 0, 0, 30)
    assert {w["text"] for w in words if w["overlap"]} == {"two", "interrupt"}
    assert all(w["brief"] for w in words)


def test_duplicate_output_cannot_improve_recall_and_missed_words_stay_in_denominator():
    refs = [
        dict(start=0, end=0.5, speaker="a", text="yes", overlap=True, brief=True),
        dict(start=2, end=2.5, speaker="a", text="missed", overlap=False, brief=True),
    ]
    pred = [dict(start=0, end=0.5, speaker="a", text="yes yes")]
    score = temporal_attributed_score(
        refs, pred, {"a": "a"}, brief_turn_seconds=2, tolerance_seconds=0.5
    )
    result = combine([score, score])
    assert result["matched_words"] == 2
    assert result["reference_words"] == 4
    assert result["recall"] == result["precision"] == 0.5


def test_named_edit_metric_counts_wrong_owner_as_missed_and_inserted():
    refs = [dict(start=0, end=1, speaker="alice", text="yes")]
    prediction = [dict(start=8, end=9, speaker="S01", text="yes")]
    assert named_edit_errors(refs, prediction, {"S01": "alice"}) == 0
    assert named_edit_errors(refs, prediction, {"S01": "bob"}) == 2
    assert named_edit_errors(refs, prediction, {}) == 2
