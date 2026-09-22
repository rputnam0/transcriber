"""Utterance overrides must not propagate across decoder IDs or weaken confidence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refine_named_turns as local


def evidence(monkeypatch, duration=1.5, posterior=0.98, cosine=0.8):
    monkeypatch.setattr(local, "probabilities", lambda m, x: np.array([[1 - posterior, posterior]]))
    model = dict(names=np.array(["bob", "alice"]), centroids=np.array([[0.0], [cosine]]))
    features = dict(
        vectors=np.array([[1.0]]), indices=np.array([1]), durations=np.array([duration])
    )
    return features, model, dict(enabled=True, minimum_seconds=1, posterior=0.95)


def test_only_the_individual_turn_changes_even_with_shared_decoder_id(monkeypatch):
    features, model, policy = evidence(monkeypatch)
    original = ["bob", "bob", "decoder_fallback"]
    names, audit = local.apply_local_identity(original, features, model, policy, ["alice", "bob"])
    assert names == ["bob", "alice", "decoder_fallback"]
    assert original == ["bob", "bob", "decoder_fallback"]
    assert audit[0]["turn_index"] == 1
    assert audit[0]["before"] == "bob" and audit[0]["after"] == "alice"


@pytest.mark.parametrize("kwargs", [dict(duration=0.75), dict(posterior=0.94), dict(cosine=0.39)])
def test_insufficient_evidence_preserves_fallback(monkeypatch, kwargs):
    f, m, p = evidence(monkeypatch, **kwargs)
    assert local.apply_local_identity(["bob", "fallback"], f, m, p) == (["bob", "fallback"], [])


def test_absent_winner_does_not_renormalize_an_eligible_runner_up(monkeypatch):
    f, m, p = evidence(monkeypatch)
    assert local.apply_local_identity(["bob", "fallback"], f, m, p, ["bob"]) == (
        ["bob", "fallback"],
        [],
    )


def test_disabled_or_missing_features_keep_names(monkeypatch):
    f, m, p = evidence(monkeypatch)
    assert local.apply_local_identity(["bob", "bob"], f, m, dict(enabled=False))[1] == []
    f["vectors"] = np.empty((0, 1))
    assert local.apply_local_identity(["bob", "bob"], f, m, p)[0] == ["bob", "bob"]


def test_merged_paragraph_retains_correction_review_flag():
    from export_moss_transcripts import paragraphs, reader

    common = dict(
        speaker_handle="alice",
        speaker="Alice",
        cut_id="00000",
        local_speaker="S01",
        roster_review_required=False,
    )
    first = dict(common, start=0, end=2, text="one", turn_id=0)
    second = dict(
        common, start=2.1, end=4, text="two", turn_id=1, local_identity_correction={"before": "bob"}
    )
    result = paragraphs([first, second])
    assert len(result) == 1
    assert "check speaker" in reader("test", result, "")
    assert "local_identity_review_required" not in first
