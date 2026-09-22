"""Keep test labels and reference-name oracles out of checkpoint selection."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finish_moss_mac_experiment import promotion, select_candidate  # noqa: E402


def report(split="dev", cut="39_00_00"):
    return {
        "split": split,
        "records": [{"cut_id": cut, "reference": [{"text": "hello", "speaker": "alice"}]}],
        "metrics": {
            "moss_oracle_names": {"named_word_error_rate": 0, "macro_f1": 1},
            "moss_named": {"named_word_error_rate": 0.3, "macro_f1": 0.8},
            "moss_hybrid_names": {"named_word_error_rate": 0.2, "macro_f1": 0.79},
        },
    }


def test_oracle_is_never_selected_and_test_labels_are_rejected():
    selected, _ = select_candidate([(Path("candidate.json"), report())])
    assert selected["method"] == "moss_hybrid_names"
    with pytest.raises(ValueError, match="only use development"):
        select_candidate([(Path("candidate.json"), report(split="test"))])


def test_candidates_must_share_identical_references():
    with pytest.raises(ValueError, match="references or clip sets differ"):
        select_candidate([(Path("a"), report()), (Path("b"), report(cut="39_00_30"))])


def test_more_overlap_or_lower_loss_cannot_override_primary_error_and_speaker_guard():
    public = {"named_word_error_rate": 0.24, "macro_f1": 0.77}
    assert promotion({"named_word_error_rate": 0.23, "macro_f1": 0.76}, public)
    assert not promotion({"named_word_error_rate": 0.23, "macro_f1": 0.75}, public)
    assert not promotion({"named_word_error_rate": 0.24, "macro_f1": 0.90}, public)
