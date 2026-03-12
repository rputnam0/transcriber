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
                            },
                            "Leopold Magnus": {"Dungeon Master": 1},
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


def test_build_hard_negative_dataset_applies_per_pair_and_global_caps(tmp_path):
    eval_summary = _build_eval_summary(tmp_path / "eval")
    candidate_pool_dir = tmp_path / "mixed_candidates"
    save_candidate_pool(
        candidate_pool_dir,
        records=[
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
        embeddings=[np.asarray([0.2, 0.8], dtype=np.float32)],
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
    assert {record["source_kind"] for record in records} == {"eval", "mixed_candidate_pool"}

    cyrus_records = [record for record in records if record["speaker"] == "Cyrus Schwert"]
    assert len(cyrus_records) == 1
    assert cyrus_records[0]["score_margin"] == 0.03
