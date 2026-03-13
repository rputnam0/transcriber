from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from transcriber.session_reassignment import (
    DiarizationRepairConfig,
    SessionSegmentEmbedding,
    apply_profile_to_segments,
    repair_diarization_segments,
)


class _GraphSpeakerBank:
    def score_candidates(self, embedding, **kwargs):  # noqa: ANN003, ARG002
        vector = np.asarray(embedding, dtype=np.float32)
        alice_score = float(vector[0])
        bob_score = float(vector[1])
        total = max(alice_score + bob_score, 1e-6)
        return [
            {
                "speaker": "Alice",
                "score": alice_score / total,
                "cluster_id": 1,
                "distance": 1.0 - (alice_score / total),
                "source": "centroid",
            },
            {
                "speaker": "Bob",
                "score": bob_score / total,
                "cluster_id": 2,
                "distance": 1.0 - (bob_score / total),
                "source": "centroid",
            },
        ]

    def match(self, embedding, **kwargs):  # noqa: ANN003
        candidates = self.score_candidates(embedding, **kwargs)
        top1 = candidates[0]
        top2 = candidates[1]
        threshold = float(kwargs.get("threshold") or 0.0)
        margin = float(kwargs.get("margin") or 0.0)
        if float(top1["score"]) >= threshold and (float(top1["score"]) - float(top2["score"])) >= margin:
            return dict(top1)
        return None


class _NamedSpeakerBank:
    def __init__(self, speakers: tuple[str, str]):
        self.left, self.right = speakers

    def score_candidates(self, embedding, **kwargs):  # noqa: ANN003, ARG002
        vector = np.asarray(embedding, dtype=np.float32)
        left_score = float(vector[0])
        right_score = float(vector[1])
        total = max(left_score + right_score, 1e-6)
        return [
            {
                "speaker": self.left,
                "score": left_score / total,
                "cluster_id": 1,
                "distance": 1.0 - (left_score / total),
                "source": "centroid",
            },
            {
                "speaker": self.right,
                "score": right_score / total,
                "cluster_id": 2,
                "distance": 1.0 - (right_score / total),
                "source": "centroid",
            },
        ]

    def match(self, embedding, **kwargs):  # noqa: ANN003
        candidates = self.score_candidates(embedding, **kwargs)
        top1 = candidates[0]
        top2 = candidates[1]
        threshold = float(kwargs.get("threshold") or 0.0)
        margin = float(kwargs.get("margin") or 0.0)
        if float(top1["score"]) >= threshold and (float(top1["score"]) - float(top2["score"])) >= margin:
            return dict(top1)
        return None


def test_repair_diarization_segments_splits_trims_and_merges():
    segments = [
        {
            "start": 0.0,
            "end": 1.15,
            "speaker": "SPEAKER_00",
            "speaker_raw": "SPEAKER_00",
            "text": "alpha beta gamma",
            "words": [
                {"start": 0.00, "end": 0.25, "word": "alpha", "speaker": "SPEAKER_00"},
                {"start": 0.30, "end": 0.55, "word": "beta", "speaker": "SPEAKER_00"},
                {"start": 1.00, "end": 1.15, "word": "gamma", "speaker": "SPEAKER_01"},
            ],
        },
        {
            "start": 1.10,
            "end": 1.45,
            "speaker": "SPEAKER_01",
            "speaker_raw": "SPEAKER_01",
            "text": "delta epsilon",
            "words": [
                {"start": 1.10, "end": 1.22, "word": "delta", "speaker": "SPEAKER_01"},
                {"start": 1.28, "end": 1.45, "word": "epsilon", "speaker": "SPEAKER_01"},
            ],
        },
    ]

    repaired, summary = repair_diarization_segments(
        segments,
        config=DiarizationRepairConfig(enabled=True),
    )

    assert len(repaired) == 2
    assert summary["split_segments"] == 1
    assert summary["trimmed_overlaps"] == 1
    assert summary["merged_segments"] >= 1
    assert repaired[0]["speaker_raw"] == "SPEAKER_00"
    assert repaired[1]["speaker_raw"] == "SPEAKER_01"
    assert repaired[1]["speaker_repair_overlap_heavy"] is True


def test_repair_diarization_segments_marks_only_boundary_chunks_overlap_heavy():
    segments = [
        {
            "start": 0.0,
            "end": 1.90,
            "speaker": "SPEAKER_00",
            "speaker_raw": "SPEAKER_00",
            "text": "alpha beta gamma delta",
            "words": [
                {"start": 0.00, "end": 0.20, "word": "alpha", "speaker": "SPEAKER_00"},
                {"start": 0.25, "end": 0.45, "word": "beta", "speaker": "SPEAKER_00"},
                {"start": 0.85, "end": 1.25, "word": "gamma", "speaker": "SPEAKER_00"},
                {"start": 1.30, "end": 1.90, "word": "delta", "speaker": "SPEAKER_01"},
            ],
        },
    ]

    repaired, summary = repair_diarization_segments(
        segments,
        config=DiarizationRepairConfig(enabled=True, max_seed_overlap_seconds=0.50),
    )

    assert len(repaired) == 3
    assert summary["split_segments"] == 1
    assert repaired[0]["speaker_raw"] == "SPEAKER_00"
    assert repaired[1]["speaker_raw"] == "SPEAKER_00"
    assert repaired[2]["speaker_raw"] == "SPEAKER_01"
    assert repaired[0]["speaker_repair_overlap_heavy"] is False
    assert repaired[1]["speaker_repair_overlap_heavy"] is True
    assert repaired[2]["speaker_repair_overlap_heavy"] is True


def test_apply_profile_to_segments_session_graph_rescues_unknown_segment():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "alpha",
            "words": [{"start": 0.0, "end": 0.8, "word": "alpha", "speaker": "SPEAKER_00"}],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "SPEAKER_01",
            "text": "beta",
            "words": [{"start": 1.0, "end": 1.8, "word": "beta", "speaker": "SPEAKER_01"}],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "SPEAKER_02",
            "text": "gamma",
            "words": [{"start": 2.0, "end": 2.8, "word": "gamma", "speaker": "SPEAKER_02"}],
        },
    ]

    relabeled, summary, _ = apply_profile_to_segments(
        audio_path="session.wav",
        segments=segments,
        label_embeddings={},
        speaker_bank=_GraphSpeakerBank(),
        speaker_bank_config=SimpleNamespace(
            use_existing=True,
            threshold=0.60,
            scoring_margin=0.12,
            radius_factor=0.0,
            classifier_min_confidence=0.0,
            classifier_min_margin=0.0,
            classifier_fusion_mode="fallback",
            classifier_fusion_weight=0.70,
            classifier_bank_weight=0.30,
            match_aggregation="mean",
            min_segments_per_label=1,
            scoring_as_norm_enabled=False,
            scoring_as_norm_cohort_size=0,
            repair_enabled=False,
            session_graph_enabled=True,
            session_graph_override_min_confidence=0.55,
            session_graph_override_min_margin=0.12,
        ),
        segment_classifier=None,
        extract_embeddings_for_segments_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected embedding extraction")
        ),
        precomputed_segment_embeddings=[
            SessionSegmentEmbedding(
                segment_index=0,
                raw_label="SPEAKER_00",
                start=0.0,
                end=1.0,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            SessionSegmentEmbedding(
                segment_index=1,
                raw_label="SPEAKER_01",
                start=1.0,
                end=2.0,
                embedding=np.asarray([0.95, 0.05], dtype=np.float32),
            ),
            SessionSegmentEmbedding(
                segment_index=2,
                raw_label="SPEAKER_02",
                start=2.0,
                end=3.0,
                embedding=np.asarray([0.54, 0.46], dtype=np.float32),
            ),
        ],
    )

    assert [segment["speaker"] for segment in relabeled] == ["Alice", "Alice", "Alice"]
    assert relabeled[0]["speaker_graph_seed"] is True
    assert relabeled[1]["speaker_graph_seed"] is True
    assert relabeled[2]["speaker_graph_override"] is True
    assert relabeled[2]["speaker_match_source"] == "session_graph"
    assert summary["graph"]["seed_segments"] == 2
    assert summary["graph"]["rescued_unknowns"] == 1
    assert summary["graph"]["overrides"] == 1


def test_apply_profile_to_segments_pair_override_can_enable_permissive_graph_flip():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "alpha",
            "words": [{"start": 0.0, "end": 0.8, "word": "alpha", "speaker": "SPEAKER_00"}],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "SPEAKER_01",
            "text": "beta",
            "words": [{"start": 1.0, "end": 1.8, "word": "beta", "speaker": "SPEAKER_01"}],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "SPEAKER_02",
            "text": "gamma",
            "words": [{"start": 2.0, "end": 2.8, "word": "gamma", "speaker": "SPEAKER_02"}],
        },
    ]

    relabeled, summary, _ = apply_profile_to_segments(
        audio_path="session.wav",
        segments=segments,
        label_embeddings={},
        speaker_bank=_NamedSpeakerBank(("Cyrus Schwert", "Cletus Cobbington")),
        speaker_bank_config=SimpleNamespace(
            use_existing=True,
            threshold=0.60,
            scoring_margin=0.12,
            radius_factor=0.0,
            classifier_min_confidence=0.0,
            classifier_min_margin=0.0,
            classifier_fusion_mode="fallback",
            classifier_fusion_weight=0.70,
            classifier_bank_weight=0.30,
            match_aggregation="mean",
            min_segments_per_label=1,
            scoring_as_norm_enabled=False,
            scoring_as_norm_cohort_size=0,
            repair_enabled=False,
            session_graph_enabled=True,
            session_graph_override_min_confidence=0.90,
            session_graph_override_min_margin=0.12,
            session_graph_pair_overrides={
                "Cletus Cobbington::Cyrus Schwert": {
                    "override_min_confidence": 0.55,
                }
            },
        ),
        segment_classifier=None,
        extract_embeddings_for_segments_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected embedding extraction")
        ),
        precomputed_segment_embeddings=[
            SessionSegmentEmbedding(
                segment_index=0,
                raw_label="SPEAKER_00",
                start=0.0,
                end=1.0,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            SessionSegmentEmbedding(
                segment_index=1,
                raw_label="SPEAKER_01",
                start=1.0,
                end=2.0,
                embedding=np.asarray([0.95, 0.05], dtype=np.float32),
            ),
            SessionSegmentEmbedding(
                segment_index=2,
                raw_label="SPEAKER_02",
                start=2.0,
                end=3.0,
                embedding=np.asarray([0.54, 0.46], dtype=np.float32),
            ),
        ],
    )

    assert relabeled[2]["speaker"] == "Cyrus Schwert"
    assert relabeled[2]["speaker_graph_override"] is True
    assert relabeled[2]["speaker_graph_pair"] == "Cletus Cobbington::Cyrus Schwert"
    assert summary["graph"]["pairs"]["Cletus Cobbington::Cyrus Schwert"]["overrides_accepted"] == 1


def test_apply_profile_to_segments_pair_override_can_block_conservative_graph_flip():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "alpha",
            "words": [{"start": 0.0, "end": 0.8, "word": "alpha", "speaker": "SPEAKER_00"}],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "SPEAKER_01",
            "text": "beta",
            "words": [{"start": 1.0, "end": 1.8, "word": "beta", "speaker": "SPEAKER_01"}],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "SPEAKER_02",
            "text": "gamma",
            "words": [{"start": 2.0, "end": 2.8, "word": "gamma", "speaker": "SPEAKER_02"}],
        },
    ]

    relabeled, summary, _ = apply_profile_to_segments(
        audio_path="session.wav",
        segments=segments,
        label_embeddings={},
        speaker_bank=_NamedSpeakerBank(("Dungeon Master", "Kaladen Shash")),
        speaker_bank_config=SimpleNamespace(
            use_existing=True,
            threshold=0.60,
            scoring_margin=0.12,
            radius_factor=0.0,
            classifier_min_confidence=0.0,
            classifier_min_margin=0.0,
            classifier_fusion_mode="fallback",
            classifier_fusion_weight=0.70,
            classifier_bank_weight=0.30,
            match_aggregation="mean",
            min_segments_per_label=1,
            scoring_as_norm_enabled=False,
            scoring_as_norm_cohort_size=0,
            repair_enabled=False,
            session_graph_enabled=True,
            session_graph_override_min_confidence=0.0,
            session_graph_override_min_margin=0.0,
            session_graph_pair_overrides={
                "Dungeon Master::Kaladen Shash": {
                    "override_min_confidence": 0.90,
                    "override_min_margin": 0.20,
                }
            },
        ),
        segment_classifier=None,
        extract_embeddings_for_segments_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected embedding extraction")
        ),
        precomputed_segment_embeddings=[
            SessionSegmentEmbedding(
                segment_index=0,
                raw_label="SPEAKER_00",
                start=0.0,
                end=1.0,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            SessionSegmentEmbedding(
                segment_index=1,
                raw_label="SPEAKER_01",
                start=1.0,
                end=2.0,
                embedding=np.asarray([0.95, 0.05], dtype=np.float32),
            ),
            SessionSegmentEmbedding(
                segment_index=2,
                raw_label="SPEAKER_02",
                start=2.0,
                end=3.0,
                embedding=np.asarray([0.54, 0.46], dtype=np.float32),
            ),
        ],
    )

    assert relabeled[2]["speaker"] == "unknown"
    assert relabeled[2]["speaker_graph_override"] is False
    assert relabeled[2]["speaker_graph_pair"] == "Dungeon Master::Kaladen Shash"
    assert summary["graph"]["pairs"]["Dungeon Master::Kaladen Shash"]["overrides_accepted"] == 0
