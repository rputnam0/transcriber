import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "annotation_analysis", Path(__file__).parents[1] / "scripts/analyze_speaker_annotations.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def make_review(tmp_path, verdict="wrong", slot=None):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    dict(
                        start=1,
                        end=2,
                        speaker="Alice",
                        text="Yes.",
                        cut_id="00000",
                        local_speaker="S01",
                    )
                ]
            }
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
                "roster": ["Alice", "Bob"],
                "reviews": {"0": dict(verdict=verdict, speaker_slot=slot)},
            }
        )
    )
    return review, transcript


def test_wrong_without_identity_is_not_positive_training_label(tmp_path):
    review, transcript = make_review(tmp_path)
    original = review.read_bytes()
    rows, _ = module.load_review(
        review, transcript, 2, {"a": "Alice", "b": "Bob"}, tmp_path / "snapshot"
    )
    assert rows[0]["truth"] is None
    assert rows[0]["positive_label_eligible"] is False
    assert rows[0]["verdict"] == "wrong"
    assert review.read_bytes() == original


def test_explicit_correction_is_separate_from_original(tmp_path):
    review, transcript = make_review(tmp_path, slot=1)
    rows, _ = module.load_review(
        review, transcript, 2, {"a": "Alice", "b": "Bob"}, tmp_path / "snapshot"
    )
    assert rows[0]["truth"] == "Bob"
    assert rows[0]["predicted"] == "Alice"
    assert rows[0]["positive_label_eligible"]


def test_stale_review_must_not_join_by_turn_number(tmp_path):
    review, transcript = make_review(tmp_path)
    transcript.write_text(transcript.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.load_review(review, transcript, 2, {}, tmp_path / "snapshot")


def test_overlap_union_does_not_double_count_three_speakers():
    assert module.union_seconds([(0, 2), (1, 3), (1.5, 2.5)]) == 3


def test_unsure_with_retained_correction_is_not_a_training_target(tmp_path):
    review, transcript = make_review(tmp_path, verdict="unsure", slot=1)
    rows, _ = module.load_review(
        review, transcript, 2, {"a": "Alice", "b": "Bob"}, tmp_path / "snapshot"
    )
    assert rows[0]["truth"] is None
    assert not rows[0]["positive_label_eligible"]


def test_two_true_identities_in_one_cluster_require_at_least_one_error(tmp_path):
    review, transcript = make_review(tmp_path, verdict="correct")
    data = json.loads(transcript.read_text())
    data["segments"].append({**data["segments"][0], "start": 3, "end": 4})
    transcript.write_text(json.dumps(data))
    grades = json.loads(review.read_text())
    grades["transcript_sha256"] = hashlib.sha256(transcript.read_bytes()).hexdigest()
    grades["reviews"]["1"] = {"verdict": "wrong", "speaker_slot": 1}
    review.write_text(json.dumps(grades))
    rows, source = module.load_review(
        review, transcript, 2, {"a": "Alice", "b": "Bob"}, tmp_path / "snapshot"
    )
    report = module.summarize(rows, [source])
    assert report["minimum_cluster_constant_errors"] == 1
    assert len(report["mixed_identity_clusters"]) == 1
