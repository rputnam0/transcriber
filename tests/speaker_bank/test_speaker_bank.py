from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from transcriber.speaker_bank import SpeakerBank, SpeakerBankConfig
from transcriber.cli import _resolve_speaker_bank_paths


def test_speaker_bank_persistence_and_matching(tmp_path):
    root = tmp_path / "bank"
    bank = SpeakerBank(
        root,
        profile="campaign",
        cluster_method="dbscan",
        dbscan_eps=0.3,
        dbscan_min_samples=1,
    )
    alice_primary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    alice_variant = np.array([0.96, 0.25, 0.0], dtype=np.float32)
    alice_character = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    bank.extend(
        [
            ("alice", alice_primary, "scene1.wav", {"diar_label": "SPEAKER_00"}),
            ("alice", alice_variant, "scene2.wav", {"diar_label": "SPEAKER_01"}),
            ("alice", alice_character, "scene3.wav", {"diar_label": "SPEAKER_02"}),
        ]
    )
    bank.save()

    # Reload to ensure persistence is working
    reopened = SpeakerBank(
        root,
        profile="campaign",
        cluster_method="dbscan",
        dbscan_eps=0.3,
        dbscan_min_samples=1,
    )
    summary = reopened.summary()
    assert summary["entries"] == 3
    assert "alice" in summary["speakers"]

    match = reopened.match(np.array([1.0, 0.01, 0], dtype=np.float32), threshold=0.5)
    assert match is not None
    assert match["speaker"] == "alice"
    assert match["score"] > 0.8

    no_match = reopened.match(np.array([-1.0, 0.0, 0.0], dtype=np.float32), threshold=0.9)
    assert no_match is None

    manifest = json.loads((reopened.profile_dir / "bank.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "campaign"

    pca_path = reopened.render_pca(tmp_path / "pca.png")
    if pca_path:
        assert Path(pca_path).exists()


def test_resolve_speaker_bank_paths(tmp_path, monkeypatch):
    # Force HF root via monkeypatch for reproducibility
    override_root = tmp_path / "bank_root"
    cfg = SpeakerBankConfig()
    root, profile, profile_dir = _resolve_speaker_bank_paths(cfg, str(override_root), None)
    assert root == (override_root / "speaker_bank").resolve()
    assert profile == "default"
    assert profile_dir == root / "default"


def test_match_with_single_embedding_relies_on_cosine(tmp_path):
    bank = SpeakerBank(
        tmp_path,
        profile="solo",
        cluster_method="dbscan",
        dbscan_eps=0.3,
        dbscan_min_samples=1,
    )
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    bank.extend([("alice", anchor, "solo.wav", {"diar_label": "SPEAKER_00"})])
    bank.save()

    query = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    match = bank.match(query, threshold=0.7)
    assert match is not None
    assert match["speaker"] == "alice"


def test_match_enforces_margin(tmp_path):
    bank = SpeakerBank(
        tmp_path,
        profile="margin",
        cluster_method="dbscan",
        dbscan_eps=0.3,
        dbscan_min_samples=1,
        prototypes_enabled=True,
        prototypes_per_cluster=1,
    )
    bank.extend(
        [
            ("alice", np.array([1.0, 0.0], dtype=np.float32), "a.wav", {}),
            ("bob", np.array([0.95, 0.3], dtype=np.float32), "b.wav", {}),
        ]
    )
    bank.save()

    query = np.array([0.97, 0.24], dtype=np.float32)
    loose = bank.match(query, threshold=0.5, margin=0.0)
    assert loose is not None

    strict = bank.match(query, threshold=0.5, margin=0.1)
    assert strict is None


def test_score_candidates_uses_prototypes(tmp_path):
    bank = SpeakerBank(
        tmp_path,
        profile="proto",
        cluster_method="dbscan",
        dbscan_eps=1.2,
        dbscan_min_samples=1,
        prototypes_enabled=True,
        prototypes_per_cluster=2,
    )
    bank.extend(
        [
            ("alice", np.array([1.0, 0.0], dtype=np.float32), "a1.wav", {}),
            ("alice", np.array([0.0, 1.0], dtype=np.float32), "a2.wav", {}),
        ]
    )
    bank.save()

    results = bank.score_candidates(np.array([0.0, 1.0], dtype=np.float32))
    assert results
    top = results[0]
    assert top["speaker"] == "alice"
    assert top["source"] == "prototype"
    assert top["score"] > 0.95
