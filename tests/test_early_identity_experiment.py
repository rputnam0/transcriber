"""Numerical checks for the reproducible named-diarization experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from attribute_cached_diarization import bind_clusters  # noqa: E402
from evaluate_early_named_diarization import activity_metrics  # noqa: E402
from prepare_early_domain_corpus import NAMES  # noqa: E402
from refine_named_turns import corrected_names, turn_features  # noqa: E402


def test_cluster_bindings_are_json_serializable_for_ambiguous_evidence():
    features = dict(
        vectors=np.eye(6, dtype=np.float32)[:2],
        owners=np.array(["A", "A"]),
        labels=np.array(["A"]),
        clean_seconds=np.array([6.0]),
    )
    model = dict(
        weights=np.eye(6) * 10, bias=np.zeros(6), centroids=np.eye(6), names=np.array(NAMES)
    )
    bindings = bind_clusters(features, model)
    assert json.loads(json.dumps(bindings))["A"]["review_required"] is True


def test_named_activity_error_counts_confusion_once_and_overlap_miss_once():
    reference = np.zeros((3, 6), bool)
    reference[0, 0] = True
    reference[1, :2] = True
    predicted = np.zeros_like(reference)
    predicted[0, 1] = True  # One identity confusion, not one miss plus one false alarm.
    predicted[1, 0] = True  # One missed overlapping speaker.
    predicted[2, 2] = True  # One false alarm in silence.
    metrics = activity_metrics(reference, predicted)
    assert metrics["weak_named_activity_error_rate"] == 1.0
    assert metrics["confused_speaker_seconds"] == 0.02
    assert metrics["missed_speaker_seconds"] == 0.02
    assert metrics["false_speaker_seconds"] == 0.02


def test_local_identity_never_embeds_detected_overlap(monkeypatch):
    import refine_named_turns

    wave = np.ones(3 * 16000, np.float32)
    wave[16000:32000] = 99  # Overlap must not enter the identity crop.
    seen = []

    def fake_embed(clips, embedder):
        seen.extend(clips)
        return np.ones((len(clips), 6), np.float32)

    monkeypatch.setattr(refine_named_turns, "embed", fake_embed)
    turns = [dict(start=0.0, end=3.0, speaker="A"), dict(start=1.0, end=2.0, speaker="B")]
    features = turn_features(wave, turns, object())
    assert features["indices"].tolist() == [0]
    assert all(np.max(clip) == 1 for clip in seen)


def test_disabled_local_policy_keeps_global_attribution():
    turns = [dict(start=0.0, end=1.0, speaker="A")]
    result = corrected_names(
        turns, {"A": {"proposed_speaker": NAMES[0]}}, {}, {}, dict(enabled=False)
    )
    assert result == [NAMES[0]]
