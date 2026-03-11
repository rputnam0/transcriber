from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from transcriber.multitrack_eval import (
    CachedSegmentEmbedding,
    WordSpan,
    apply_profile_to_cached_segments,
    compute_segment_purity,
    extract_session_stems,
    score_word_speaker_alignment,
    select_candidate_windows,
)


def test_select_candidate_windows_prefers_many_speakers_and_turns():
    records = [
        {"start": 0.0, "end": 20.0, "speaker": "DM", "text": "one two three four"},
        {"start": 25.0, "end": 40.0, "speaker": "Alice", "text": "five six seven"},
        {"start": 45.0, "end": 60.0, "speaker": "Bob", "text": "eight nine ten"},
        {"start": 65.0, "end": 80.0, "speaker": "Carol", "text": "eleven twelve"},
        {"start": 310.0, "end": 330.0, "speaker": "DM", "text": "lonely window"},
    ]

    windows = select_candidate_windows(
        records,
        window_seconds=120.0,
        hop_seconds=60.0,
        top_k=2,
        min_speakers=3,
    )

    assert len(windows) == 1
    assert windows[0]["start"] == 0.0
    assert windows[0]["speaker_count"] == 4
    assert "Carol" in windows[0]["speakers"]


def test_score_word_speaker_alignment_uses_word_timestamps():
    reference = [
        WordSpan(speaker="Alice", start=0.0, end=0.4, text="hello"),
        WordSpan(speaker="Bob", start=0.5, end=0.9, text="there"),
        WordSpan(speaker="Alice", start=1.0, end=1.4, text="again"),
    ]
    predicted = [
        WordSpan(speaker="Alice", start=0.0, end=0.5, text="hello"),
        WordSpan(speaker="Bob", start=0.45, end=0.95, text="there"),
        WordSpan(speaker="Carol", start=1.0, end=1.5, text="again"),
    ]

    metrics = score_word_speaker_alignment(reference, predicted, tolerance_seconds=0.2)

    assert metrics["reference_words"] == 3
    assert metrics["matched_words"] == 3
    assert metrics["correct_words"] == 2
    assert metrics["accuracy"] == 2 / 3
    assert metrics["confusion"]["Alice"]["Carol"] == 1


class _FakeSpeakerBank:
    def __init__(self):
        self._scores = {
            "Alice": [
                {
                    "speaker": "Alice",
                    "score": 0.91,
                    "cluster_id": 1,
                    "distance": 0.1,
                    "source": "centroid",
                }
            ],
            "Bob": [
                {
                    "speaker": "Bob",
                    "score": 0.93,
                    "cluster_id": 2,
                    "distance": 0.08,
                    "source": "centroid",
                }
            ],
        }

    def score_candidates(self, embedding, **kwargs):
        speaker = "Alice" if float(embedding[0]) > float(embedding[1]) else "Bob"
        return list(self._scores[speaker])

    def match(self, embedding, **kwargs):
        candidates = self.score_candidates(embedding, **kwargs)
        return dict(candidates[0]) if candidates else None


class _FakeClassifier:
    def score_candidates(self, embedding):
        if float(embedding[0]) > float(embedding[1]):
            return [
                {
                    "speaker": "Alice",
                    "score": 0.88,
                    "source": "segment_classifier",
                    "cluster_id": None,
                    "distance": None,
                },
                {
                    "speaker": "Bob",
                    "score": 0.12,
                    "source": "segment_classifier",
                    "cluster_id": None,
                    "distance": None,
                },
            ]
        return [
            {
                "speaker": "Bob",
                "score": 0.86,
                "source": "segment_classifier",
                "cluster_id": None,
                "distance": None,
            },
            {
                "speaker": "Alice",
                "score": 0.14,
                "source": "segment_classifier",
                "cluster_id": None,
                "distance": None,
            },
        ]

    def predict(self, embedding, *, min_confidence, min_margin):
        if float(embedding[0]) > float(embedding[1]):
            return SimpleNamespace(
                speaker="Alice",
                score=0.88,
                margin=0.76,
                second_best=0.12,
                candidates=self.score_candidates(embedding),
            )
        return SimpleNamespace(
            speaker="Bob",
            score=0.86,
            margin=0.72,
            second_best=0.14,
            candidates=self.score_candidates(embedding),
        )


def test_apply_profile_to_cached_segments_relabels_words_and_segments():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "hello there",
            "words": [
                {"start": 0.0, "end": 0.4, "word": "hello", "speaker": "SPEAKER_00"},
                {"start": 0.5, "end": 0.9, "word": "there", "speaker": "SPEAKER_00"},
            ],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "SPEAKER_01",
            "text": "general kenobi",
            "words": [
                {"start": 1.0, "end": 1.4, "word": "general", "speaker": "SPEAKER_01"},
                {"start": 1.5, "end": 1.9, "word": "kenobi", "speaker": "SPEAKER_01"},
            ],
        },
    ]
    relabeled, summary = apply_profile_to_cached_segments(
        segments,
        label_embeddings={
            "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
            "SPEAKER_01": np.array([0.0, 1.0], dtype=np.float32),
        },
        segment_embeddings=[
            CachedSegmentEmbedding(
                0, "SPEAKER_00", 0.0, 1.0, np.array([1.0, 0.0], dtype=np.float32)
            ),
            CachedSegmentEmbedding(
                1, "SPEAKER_01", 1.0, 2.0, np.array([0.0, 1.0], dtype=np.float32)
            ),
        ],
        speaker_bank=_FakeSpeakerBank(),
        speaker_bank_config=SimpleNamespace(
            use_existing=True,
            threshold=0.4,
            scoring_margin=0.0,
            radius_factor=0.0,
            classifier_min_confidence=0.0,
            classifier_min_margin=0.0,
            match_aggregation="mean",
            min_segments_per_label=1,
            scoring_as_norm_enabled=False,
            scoring_as_norm_cohort_size=0,
        ),
        segment_classifier=_FakeClassifier(),
    )

    assert summary["matched"] == 2
    assert relabeled[0]["speaker"] == "Alice"
    assert relabeled[1]["speaker"] == "Bob"
    assert relabeled[0]["words"][0]["speaker"] == "Alice"
    assert relabeled[1]["words"][1]["speaker"] == "Bob"
    assert relabeled[0]["speaker_raw"] == "SPEAKER_00"


def test_extract_session_stems_reuses_cached_audio_without_zip(tmp_path):
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    cached = stems_dir / "speaker1.ogg"
    cached.write_bytes(b"audio")

    stems = extract_session_stems(tmp_path / "missing.zip", stems_dir)

    assert stems == [cached]


class _FusionSpeakerBank:
    def score_candidates(self, embedding, **kwargs):
        return [
            {
                "speaker": "Bob",
                "score": 0.95,
                "cluster_id": 2,
                "distance": 0.05,
                "source": "prototype",
            },
            {
                "speaker": "Alice",
                "score": 0.30,
                "cluster_id": 1,
                "distance": 0.40,
                "source": "prototype",
            },
        ]

    def match(self, embedding, **kwargs):
        return dict(self.score_candidates(embedding, **kwargs)[0])


class _FusionClassifier:
    def score_candidates(self, embedding):
        return [
            {
                "speaker": "Alice",
                "score": 0.70,
                "source": "segment_classifier",
                "cluster_id": None,
                "distance": None,
            },
            {
                "speaker": "Bob",
                "score": 0.30,
                "source": "segment_classifier",
                "cluster_id": None,
                "distance": None,
            },
        ]

    def predict(self, embedding, *, min_confidence, min_margin):
        return SimpleNamespace(
            speaker="Alice",
            score=0.70,
            margin=0.40,
            second_best=0.30,
            candidates=self.score_candidates(embedding),
        )


def test_compute_segment_purity_reports_buckets_and_rejections():
    stem_arrays = [
        ("Alice", np.concatenate([np.ones(1600), np.full(1600, 0.8)]).astype(np.float32)),
        ("Bob", np.concatenate([np.zeros(1600), np.ones(1600)]).astype(np.float32)),
    ]
    segments = [
        {"start": 0.0, "end": 0.1, "speaker": "SPEAKER_00"},
        {"start": 0.1, "end": 0.2, "speaker": "SPEAKER_01"},
    ]

    purity = compute_segment_purity(
        segments,
        stem_arrays,
        sample_rate=16000,
        min_power=1e-4,
        min_training_segment_dur=0.05,
        max_training_segment_dur=2.0,
    )

    assert purity["segments"] == 2
    assert purity["accepted_training_segments"] == 1
    assert purity["bucket_counts"]["high"] == 1
    assert purity["bucket_counts"]["low"] == 1
    assert purity["rejection_counts"]["low_share"] == 1


def test_apply_profile_to_cached_segments_supports_score_fusion():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "speaker": "SPEAKER_00"}],
        }
    ]

    relabeled, summary = apply_profile_to_cached_segments(
        segments,
        label_embeddings={"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)},
        segment_embeddings=[
            CachedSegmentEmbedding(
                0, "SPEAKER_00", 0.0, 1.0, np.array([1.0, 0.0], dtype=np.float32)
            )
        ],
        speaker_bank=_FusionSpeakerBank(),
        speaker_bank_config=SimpleNamespace(
            use_existing=True,
            threshold=0.40,
            scoring_margin=0.0,
            radius_factor=0.0,
            classifier_min_confidence=0.0,
            classifier_min_margin=0.0,
            classifier_fusion_mode="score_sum",
            classifier_fusion_weight=0.30,
            classifier_bank_weight=0.70,
            match_aggregation="mean",
            min_segments_per_label=1,
            scoring_as_norm_enabled=False,
            scoring_as_norm_cohort_size=0,
        ),
        segment_classifier=_FusionClassifier(),
    )

    assert summary["matched"] == 1
    assert relabeled[0]["speaker"] == "Bob"
    assert relabeled[0]["speaker_match_source"] == "segment_fusion"


class _ShortSegmentFallbackBank:
    def score_candidates(self, embedding, **kwargs):
        return []

    def match(self, embedding, **kwargs):
        return None


class _ShortSegmentFallbackClassifier:
    def score_candidates(self, embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        if float(vector[0]) > 0.9:
            return [
                {
                    "speaker": "Cyrus Schwert",
                    "score": 0.82,
                    "source": "segment_classifier",
                    "cluster_id": None,
                    "distance": None,
                },
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.18,
                    "source": "segment_classifier",
                    "cluster_id": None,
                    "distance": None,
                },
            ]
        return [
            {
                "speaker": "Cyrus Schwert",
                "score": 0.55,
                "source": "segment_classifier",
                "cluster_id": None,
                "distance": None,
            },
            {
                "speaker": "Cletus Cobbington",
                "score": 0.45,
                "source": "segment_classifier",
                "cluster_id": None,
                "distance": None,
            },
        ]

    def predict(self, embedding, *, min_confidence, min_margin):
        candidates = self.score_candidates(embedding)
        top1 = candidates[0]
        top2 = candidates[1]["score"]
        margin = float(top1["score"]) - float(top2)
        if float(top1["score"]) < float(min_confidence) or margin < float(min_margin):
            return None
        return SimpleNamespace(
            speaker=top1["speaker"],
            score=top1["score"],
            margin=margin,
            second_best=top2,
            candidates=candidates,
        )


def test_apply_profile_to_cached_segments_uses_label_classifier_for_short_segments():
    segments = [
        {
            "start": 0.0,
            "end": 0.6,
            "speaker": "SPEAKER_00",
            "text": "hey",
            "words": [{"start": 0.0, "end": 0.5, "word": "hey", "speaker": "SPEAKER_00"}],
        },
        {
            "start": 0.6,
            "end": 1.2,
            "speaker": "SPEAKER_00",
            "text": "yo",
            "words": [{"start": 0.6, "end": 1.1, "word": "yo", "speaker": "SPEAKER_00"}],
        },
    ]

    relabeled, summary = apply_profile_to_cached_segments(
        segments,
        label_embeddings={"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)},
        segment_embeddings=[
            CachedSegmentEmbedding(
                0, "SPEAKER_00", 0.0, 0.6, np.array([0.4, 0.6], dtype=np.float32)
            ),
            CachedSegmentEmbedding(
                1, "SPEAKER_00", 0.6, 1.2, np.array([0.4, 0.6], dtype=np.float32)
            ),
        ],
        speaker_bank=_ShortSegmentFallbackBank(),
        speaker_bank_config=SimpleNamespace(
            use_existing=True,
            threshold=0.40,
            scoring_margin=0.03,
            radius_factor=0.0,
            classifier_min_confidence=0.0,
            classifier_min_margin=0.03,
            match_aggregation="mean",
            min_segments_per_label=1,
            scoring_as_norm_enabled=False,
            scoring_as_norm_cohort_size=0,
        ),
        segment_classifier=_ShortSegmentFallbackClassifier(),
    )

    assert summary["matched"] == 1
    assert relabeled[0]["speaker"] == "Cyrus Schwert"
    assert relabeled[1]["speaker"] == "Cyrus Schwert"
    assert relabeled[0]["speaker_match_source"] == "label_classifier"
    assert relabeled[1]["speaker_match_source"] == "label_classifier"
