from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from transcriber.hard_negatives import build_hard_negative_dataset, discover_confusion_pairs
from transcriber.prep_artifacts import save_candidate_pool


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _build_eval_summary(tmp_path: Path) -> dict:
    predicted_path = _write_jsonl(
        tmp_path / "predicted.jsonl",
        [
            {
                "speaker_match": {"margin": 0.03},
                "speaker_match_candidates": [
                    {"speaker": "Cyrus Schwert", "score": 0.58},
                    {"speaker": "Cletus Cobbington", "score": 0.55},
                ],
            },
            {
                "speaker_match": {"margin": 0.11},
                "speaker_match_candidates": [
                    {"speaker": "Cyrus Schwert", "score": 0.54},
                    {"speaker": "Cletus Cobbington", "score": 0.43},
                ],
            },
        ],
    )
    purity_path = tmp_path / "purity.json"
    purity_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "speaker": "Cyrus Schwert",
                        "second_speaker": "Cletus Cobbington",
                        "dominant_share": 0.64,
                        "active_speakers": 2,
                        "top1_power": 0.90,
                        "top2_power": 0.72,
                        "start": 0.0,
                        "end": 1.0,
                        "session": "Session 61",
                    },
                    {
                        "speaker": "Cyrus Schwert",
                        "second_speaker": "Cletus Cobbington",
                        "dominant_share": 0.57,
                        "active_speakers": 2,
                        "top1_power": 0.82,
                        "top2_power": 0.69,
                        "start": 1.0,
                        "end": 2.0,
                        "session": "Session 61",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    segment_embedding_path = tmp_path / "segment_embeddings.npz"
    np.savez_compressed(
        segment_embedding_path,
        segment_indices=np.asarray([0, 1], dtype=np.int32),
        embeddings=np.asarray([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32),
    )
    return {
        "session_zip": str(tmp_path / "Session_61.zip"),
        "results": [
            {
                "predicted_jsonl": str(predicted_path),
                "diarization_purity_path": str(purity_path),
                "segment_embedding_path": str(segment_embedding_path),
                "metrics": {
                    "confusion": {
                        "Cyrus Schwert": {"Cletus Cobbington": 4},
                        "Dungeon Master": {"Leopold Magnus": 1},
                    }
                },
            }
        ],
    }


def test_discover_confusion_pairs_preserves_seed_and_adds_top_confusions():
    summaries = [
        {
            "results": [
                {
                    "metrics": {
                        "confusion": {
                            "Dungeon Master": {
                                "Leopold Magnus": 3,
                                "<unmatched>": 2,
                                "unknown": 4,
                            },
                            "Leopold Magnus": {"Dungeon Master": 1},
                            "unknown": {"Cyrus Schwert": 10},
                            "Alice": {"Alice": 5},
                        }
                    }
                }
            ]
        }
    ]

    pairs = discover_confusion_pairs(
        summaries,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_k=1,
    )

    assert ("Cletus Cobbington", "Cyrus Schwert") in pairs
    assert ("Dungeon Master", "Leopold Magnus") in pairs
    assert all("unknown" not in pair for pair in pairs)


def test_build_hard_negative_dataset_applies_per_pair_and_global_caps(tmp_path):
    eval_summary = _build_eval_summary(tmp_path / "eval")
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 2,
                "top1_power": 0.81,
                "top2_power": 0.70,
                "window_index": 6,
                "session": "Session 48",
                "start": 11.0,
                "end": 12.0,
            },
            {
                "speaker": "Cletus Cobbington",
                "second_speaker": "Cyrus Schwert",
                "rejection": "low_share",
                "dominant_share": 0.61,
                "active_speakers": 2,
                "top1_power": 0.79,
                "top2_power": 0.68,
                "window_index": 7,
                "session": "Session 61",
                "start": 12.0,
                "end": 13.0,
            }
        ],
        embeddings=[
            np.asarray([0.8, 0.2], dtype=np.float32),
            np.asarray([0.2, 0.8], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[eval_summary],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=10,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=1,
        max_eval_margin=0.12,
        min_mixed_dominant_share=0.55,
        per_pair_cap=1,
        max_fraction=0.20,
    )

    assert dataset is not None
    assert dataset.samples == 2
    assert summary["selected"] == 2
    assert summary["limits"]["global_cap"] == 2
    assert sorted(dataset.labels) == ["Cletus Cobbington", "Cyrus Schwert"]
    assert dataset.sources == ["hard_negative", "hard_negative"]
    assert {record["source_kind"] for record in records} == {"mixed_candidate_pool"}
    assert summary["heuristics"] == {"uses_eval_confusions": True, "uses_eval_examples": False}

    cyrus_records = [record for record in records if record["speaker"] == "Cyrus Schwert"]
    assert len(cyrus_records) == 1
    assert cyrus_records[0]["score_margin"] is None


def test_build_hard_negative_dataset_does_not_train_on_eval_segments(tmp_path):
    eval_summary = _build_eval_summary(tmp_path / "eval")

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[eval_summary],
        candidate_pool_dirs=[],
        base_dataset_samples=10,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=1,
        max_eval_margin=0.12,
        min_mixed_dominant_share=0.55,
        per_pair_cap=2,
        max_fraction=1.0,
    )

    assert dataset is None
    assert records == []
    assert summary["tracked_pairs"] == [("Cletus Cobbington", "Cyrus Schwert")]
    assert summary["heuristics"] == {"uses_eval_confusions": True, "uses_eval_examples": False}


def test_build_hard_negative_dataset_applies_per_speaker_cap(tmp_path):
    eval_summary = _build_eval_summary(tmp_path / "eval")
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 2,
                "top1_power": 0.81,
                "top2_power": 0.70,
                "window_index": 6,
                "session": "Session 48",
                "start": 11.0,
                "end": 12.0,
            },
            {
                "speaker": "Cletus Cobbington",
                "second_speaker": "Cyrus Schwert",
                "rejection": "low_share",
                "dominant_share": 0.61,
                "active_speakers": 2,
                "top1_power": 0.79,
                "top2_power": 0.68,
                "window_index": 7,
                "session": "Session 61",
                "start": 12.0,
                "end": 13.0,
            }
        ],
        embeddings=[
            np.asarray([0.8, 0.2], dtype=np.float32),
            np.asarray([0.2, 0.8], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[eval_summary],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=50,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=1,
        max_eval_margin=0.12,
        min_mixed_dominant_share=0.55,
        per_pair_cap=2,
        per_speaker_cap=1,
        max_fraction=1.0,
    )

    assert dataset is not None
    assert dataset.samples == 2
    assert summary["limits"]["per_speaker_cap"] == 1
    assert sorted(dataset.labels) == ["Cletus Cobbington", "Cyrus Schwert"]

    cyrus_records = [record for record in records if record["speaker"] == "Cyrus Schwert"]
    assert len(cyrus_records) == 1
    assert cyrus_records[0]["score_margin"] is None


def test_build_hard_negative_dataset_respects_zero_per_speaker_cap(tmp_path):
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 2,
                "top1_power": 0.81,
                "top2_power": 0.70,
                "window_index": 1,
                "session": "Session 48",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "speaker": "Cletus Cobbington",
                "second_speaker": "Cyrus Schwert",
                "rejection": "not_dominant",
                "dominant_share": 0.61,
                "active_speakers": 2,
                "top1_power": 0.80,
                "top2_power": 0.71,
                "window_index": 2,
                "session": "Session 49",
                "start": 1.0,
                "end": 2.0,
            },
        ],
        embeddings=[
            np.asarray([0.9, 0.1], dtype=np.float32),
            np.asarray([0.1, 0.9], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=50,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=0,
        min_mixed_dominant_share=0.55,
        per_pair_cap=2,
        per_speaker_cap=0,
        max_fraction=1.0,
    )

    assert dataset is None
    assert records == []
    assert summary["selected"] == 0
    assert summary["limits"]["per_speaker_cap"] == 0


def test_build_hard_negative_dataset_applies_pair_specific_caps(tmp_path):
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Dungeon Master",
                "second_speaker": "Kaladen Shash",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 2,
                "top1_power": 0.81,
                "top2_power": 0.70,
                "window_index": 1,
                "session": "Session 48",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "speaker": "Dungeon Master",
                "second_speaker": "Kaladen Shash",
                "rejection": "not_dominant",
                "dominant_share": 0.61,
                "active_speakers": 2,
                "top1_power": 0.80,
                "top2_power": 0.71,
                "window_index": 2,
                "session": "Session 49",
                "start": 1.0,
                "end": 2.0,
            },
            {
                "speaker": "Kaladen Shash",
                "second_speaker": "Dungeon Master",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 2,
                "top1_power": 0.79,
                "top2_power": 0.70,
                "window_index": 3,
                "session": "Session 50",
                "start": 2.0,
                "end": 3.0,
            },
            {
                "speaker": "Kaladen Shash",
                "second_speaker": "Dungeon Master",
                "rejection": "not_dominant",
                "dominant_share": 0.62,
                "active_speakers": 2,
                "top1_power": 0.82,
                "top2_power": 0.72,
                "window_index": 4,
                "session": "Session 51",
                "start": 3.0,
                "end": 4.0,
            },
        ],
        embeddings=[
            np.asarray([0.9, 0.1], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
            np.asarray([0.1, 0.9], dtype=np.float32),
            np.asarray([0.2, 0.8], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=50,
        seed_pairs=[["Dungeon Master", "Kaladen Shash"]],
        top_confusion_pairs=0,
        min_mixed_dominant_share=0.55,
        per_pair_cap=2,
        pair_caps={("Dungeon Master", "Kaladen Shash"): 1},
        max_fraction=1.0,
    )

    assert dataset is not None
    assert dataset.samples == 2
    assert sorted(dataset.labels) == ["Dungeon Master", "Kaladen Shash"]
    assert summary["limits"]["pair_caps"] == {"Dungeon Master::Kaladen Shash": 1}


def test_build_hard_negative_dataset_style_selection_respects_pair_cap(tmp_path):
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.58,
                "active_speakers": 6,
                "top1_power": 0.82,
                "top2_power": 0.76,
                "window_index": 3,
                "session": "Session 61",
                "start": 8.0,
                "end": 9.0,
                "style_profile": "session61_like",
                "style_score": 0.92,
            },
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.59,
                "active_speakers": 6,
                "top1_power": 0.84,
                "top2_power": 0.78,
                "window_index": 4,
                "session": "Session 61",
                "start": 10.0,
                "end": 11.0,
                "style_profile": "session61_like",
                "style_score": 0.91,
            },
        ],
        embeddings=[
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.95, 0.05], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=20,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=0,
        min_mixed_dominant_share=0.55,
        per_pair_cap=1,
        max_fraction=1.0,
        style_profile_name="session61_like",
        style_score_threshold=0.70,
        min_style_samples_per_pair=5,
    )

    assert dataset is not None
    assert dataset.samples == 1
    assert len(records) == 1
    assert summary["selected"] == 1
    assert summary["by_style_threshold"] == {"at_or_above_threshold": 1}


def test_build_hard_negative_dataset_tracks_pair_caps_without_seed_pairs(tmp_path):
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cletus Cobbington",
                "second_speaker": "Dungeon Master",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 2,
                "top1_power": 0.81,
                "top2_power": 0.70,
                "window_index": 1,
                "session": "Session 48",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "speaker": "Dungeon Master",
                "second_speaker": "Cletus Cobbington",
                "rejection": "not_dominant",
                "dominant_share": 0.61,
                "active_speakers": 2,
                "top1_power": 0.80,
                "top2_power": 0.71,
                "window_index": 2,
                "session": "Session 49",
                "start": 1.0,
                "end": 2.0,
            },
        ],
        embeddings=[
            np.asarray([0.9, 0.1], dtype=np.float32),
            np.asarray([0.1, 0.9], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=50,
        seed_pairs=[],
        top_confusion_pairs=0,
        min_mixed_dominant_share=0.55,
        per_pair_cap=5,
        pair_caps={("Cletus Cobbington", "Dungeon Master"): 1},
        max_fraction=1.0,
    )

    assert dataset is not None
    assert dataset.samples == 2
    assert sorted(dataset.labels) == ["Cletus Cobbington", "Dungeon Master"]
    assert summary["tracked_pairs"] == [("Cletus Cobbington", "Dungeon Master")]
    assert summary["limits"]["pair_caps"] == {"Cletus Cobbington::Dungeon Master": 1}
    assert {record["confusion_partner"] for record in records} == {
        "Cletus Cobbington",
        "Dungeon Master",
    }


def test_build_hard_negative_dataset_prioritizes_session61_like_candidates(tmp_path):
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.58,
                "active_speakers": 6,
                "top1_power": 0.82,
                "top2_power": 0.76,
                "window_index": 3,
                "session": "Session 61",
                "start": 8.0,
                "end": 9.0,
                "style_profile": "generic",
                "style_score": 0.25,
            },
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.59,
                "active_speakers": 6,
                "top1_power": 0.84,
                "top2_power": 0.78,
                "window_index": 4,
                "session": "Session 61",
                "start": 10.0,
                "end": 11.0,
                "style_profile": "session61_like",
                "style_score": 0.92,
            },
        ],
        embeddings=[
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.95, 0.05], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=20,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=0,
        min_mixed_dominant_share=0.55,
        per_pair_cap=1,
        max_fraction=1.0,
        style_profile_name="session61_like",
        style_score_threshold=0.70,
        min_style_samples_per_pair=1,
    )

    assert dataset is not None
    assert dataset.samples == 1
    assert records[0]["style_profile"] == "session61_like"
    assert summary["by_style_profile"] == {"session61_like": 1}
    assert summary["by_style_threshold"] == {"at_or_above_threshold": 1}
    assert summary["by_pair_style_threshold"] == {
        "Cyrus Schwert::Cletus Cobbington": {"at_or_above_threshold": 1}
    }


def test_build_hard_negative_dataset_uses_style_score_threshold_even_if_profile_is_generic(tmp_path):
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.58,
                "active_speakers": 6,
                "top1_power": 0.82,
                "top2_power": 0.76,
                "window_index": 3,
                "session": "Session 61",
                "start": 8.0,
                "end": 9.0,
                "style_profile": "generic",
                "style_score": 0.46,
            },
            {
                "speaker": "Cyrus Schwert",
                "second_speaker": "Cletus Cobbington",
                "rejection": "low_share",
                "dominant_share": 0.60,
                "active_speakers": 6,
                "top1_power": 0.80,
                "top2_power": 0.72,
                "window_index": 4,
                "session": "Session 12",
                "start": 10.0,
                "end": 11.0,
                "style_profile": "generic",
                "style_score": 0.20,
            },
        ],
        embeddings=[
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ],
    )

    dataset, records, summary = build_hard_negative_dataset(
        eval_summaries=[],
        candidate_pool_dirs=[candidate_pool_dir],
        base_dataset_samples=20,
        seed_pairs=[["Cyrus Schwert", "Cletus Cobbington"]],
        top_confusion_pairs=0,
        min_mixed_dominant_share=0.55,
        per_pair_cap=1,
        max_fraction=1.0,
        style_profile_name="session61_like",
        style_score_threshold=0.45,
        min_style_samples_per_pair=1,
    )

    assert dataset is not None
    assert dataset.samples == 1
    assert records[0]["window_index"] == 3
    assert summary["by_style_profile"] == {"generic": 1}
    assert summary["by_style_threshold"] == {"at_or_above_threshold": 1}
    assert summary["by_pair_style_threshold"] == {
        "Cyrus Schwert::Cletus Cobbington": {"at_or_above_threshold": 1}
    }
