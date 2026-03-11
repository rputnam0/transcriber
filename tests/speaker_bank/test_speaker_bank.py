from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_resolve_speaker_bank_paths_prefers_existing_repo_bank(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    (repo_root / ".hf_cache" / "speaker_bank" / "default").mkdir(parents=True)

    cfg = SpeakerBankConfig()
    root, profile, profile_dir = _resolve_speaker_bank_paths(
        cfg,
        root_override=None,
        hf_cache_root=str(tmp_path / "model_cache"),
    )

    assert root == (repo_root / ".hf_cache" / "speaker_bank").resolve()
    assert profile == "default"
    assert profile_dir == root / "default"


def test_resolve_speaker_bank_paths_normalizes_hub_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    home_root = tmp_path / "hf_home"
    (home_root / "speaker_bank" / "default").mkdir(parents=True)
    (home_root / "hub").mkdir(parents=True)

    cfg = SpeakerBankConfig()
    root, profile, profile_dir = _resolve_speaker_bank_paths(
        cfg,
        root_override=None,
        hf_cache_root=str(home_root / "hub"),
    )

    assert root == (home_root / "speaker_bank").resolve()
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


def test_match_per_segment_defaults_enabled():
    cfg = SpeakerBankConfig()
    assert cfg.match_per_segment is True


def test_classifier_defaults_follow_promoted_knn_profile():
    cfg = SpeakerBankConfig()
    assert cfg.classifier_model == "knn"
    assert cfg.classifier_n_neighbors == 7
    assert cfg.classifier_training_mode == "mixed"
    assert cfg.threshold == pytest.approx(0.35)
    assert cfg.scoring_margin == pytest.approx(0.0)
    assert cfg.scoring_whiten is True
    assert cfg.classifier_min_margin == pytest.approx(0.03)
    assert cfg.min_segments_per_label == 1


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


def test_score_candidates_supports_adaptive_s_norm(tmp_path):
    bank = SpeakerBank(
        tmp_path,
        profile="asnorm",
        cluster_method="dbscan",
        dbscan_eps=0.4,
        dbscan_min_samples=1,
        prototypes_enabled=True,
        prototypes_per_cluster=1,
    )
    bank.extend(
        [
            ("alice", np.array([1.0, 0.0], dtype=np.float32), "a.wav", {}),
            ("alice", np.array([0.98, 0.2], dtype=np.float32), "a2.wav", {}),
            ("bob", np.array([0.6, 0.8], dtype=np.float32), "b.wav", {}),
            ("charlie", np.array([-0.8, 0.6], dtype=np.float32), "c.wav", {}),
        ]
    )
    bank.save()

    query = np.array([0.99, 0.1], dtype=np.float32)
    raw_results = bank.score_candidates(query)
    norm_results = bank.score_candidates(query, as_norm_enabled=True, as_norm_cohort_size=2)

    assert raw_results
    assert norm_results
    assert raw_results[0]["speaker"] == "alice"
    assert norm_results[0]["speaker"] == "alice"
    assert norm_results[0]["score_mode"] == "as_norm"
    assert norm_results[0]["raw_score"] == pytest.approx(raw_results[0]["score"], abs=1e-2)
    assert norm_results[0]["score"] != raw_results[0]["score"]


def test_score_candidates_whitening_expands_margin(tmp_path):
    entries = [
        ("alice", np.array([5.0, 1.0], dtype=np.float32), "a1.wav", {}),
        ("alice", np.array([15.0, 1.1], dtype=np.float32), "a2.wav", {}),
        ("bob", np.array([5.0, -1.0], dtype=np.float32), "b1.wav", {}),
        ("bob", np.array([15.0, -1.1], dtype=np.float32), "b2.wav", {}),
    ]
    query = np.array([12.0, -1.0], dtype=np.float32)

    raw_bank = SpeakerBank(
        tmp_path / "raw",
        profile="campaign",
        cluster_method="dbscan",
        dbscan_eps=0.5,
        dbscan_min_samples=1,
        prototypes_enabled=False,
        scoring_whiten=False,
    )
    raw_bank.extend(entries)
    raw_bank.save()
    raw_scores = raw_bank.score_candidates(query, radius_factor=100.0)

    white_bank = SpeakerBank(
        tmp_path / "white",
        profile="campaign",
        cluster_method="dbscan",
        dbscan_eps=0.5,
        dbscan_min_samples=1,
        prototypes_enabled=False,
        scoring_whiten=True,
    )
    white_bank.extend(entries)
    white_bank.save()
    white_scores = white_bank.score_candidates(query, radius_factor=100.0)

    assert raw_scores[0]["speaker"] == "bob"
    assert white_scores[0]["speaker"] == "bob"
    raw_margin = raw_scores[0]["score"] - raw_scores[1]["score"]
    white_margin = white_scores[0]["score"] - white_scores[1]["score"]
    assert white_margin > raw_margin


def test_loading_whitened_bank_recomputes_cluster_variance(tmp_path):
    bank = SpeakerBank(
        tmp_path,
        profile="campaign",
        cluster_method="dbscan",
        dbscan_eps=1.2,
        dbscan_min_samples=1,
        prototypes_enabled=False,
        scoring_whiten=False,
    )
    bank.extend(
        [
            ("alice", np.array([1.0, 0.0], dtype=np.float32), "a1.wav", {}),
            ("alice", np.array([0.7, 0.3], dtype=np.float32), "a2.wav", {}),
            ("alice", np.array([0.6, 0.4], dtype=np.float32), "a3.wav", {}),
            ("alice", np.array([0.2, 0.8], dtype=np.float32), "a4.wav", {}),
        ]
    )
    bank.save()

    manifest = json.loads((bank.profile_dir / "bank.json").read_text(encoding="utf-8"))
    stored_variance = manifest["clusters"]["alice"][0]["variance"]

    reopened = SpeakerBank(
        tmp_path,
        profile="campaign",
        cluster_method="dbscan",
        dbscan_eps=1.2,
        dbscan_min_samples=1,
        prototypes_enabled=False,
        scoring_whiten=True,
    )
    cluster = reopened._clusters["alice"][0]
    recomputed_variance = reopened._cluster_variance(cluster.centroid, cluster.member_indices)

    assert recomputed_variance != pytest.approx(stored_variance)
    assert cluster.variance == pytest.approx(recomputed_variance)
