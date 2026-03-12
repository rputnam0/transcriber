from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import transcriber.baseline_prep as baseline_prep
from transcriber.segment_classifier import ClassifierDataset, load_classifier_dataset, save_classifier_dataset


def _dataset(
    labels: list[str],
    *,
    domains: list[str],
    sources: list[str],
    sessions: list[str],
) -> ClassifierDataset:
    rows = []
    for index, label in enumerate(labels):
        if label.endswith("Alice"):
            rows.append([1.0, 0.1 + (index * 0.01)])
        else:
            rows.append([0.1 + (index * 0.01), 1.0])
    count = len(labels)
    return ClassifierDataset(
        embeddings=np.asarray(rows, dtype=np.float32),
        labels=list(labels),
        domains=list(domains),
        sources=list(sources),
        sessions=list(sessions),
        durations=np.full(count, 1.0, dtype=np.float32),
        dominant_shares=np.full(count, 0.9, dtype=np.float32),
        top1_powers=np.full(count, 0.3, dtype=np.float32),
        top2_powers=np.full(count, 0.05, dtype=np.float32),
        active_speakers=np.full(count, 1, dtype=np.int32),
    )


def test_prepare_baseline_stages_resume_and_materialize_variants_from_mixed_base(
    tmp_path, monkeypatch
):
    training_root = tmp_path / "training"
    session_dir = training_root / "Session_10"
    session_dir.mkdir(parents=True)
    (session_dir / "alice.ogg").write_bytes(b"audio")
    transcripts_root = tmp_path / "transcripts"
    transcripts_root.mkdir()

    speaker_mapping_path = tmp_path / "speaker_mapping.json"
    speaker_mapping_path.write_text("{}", encoding="utf-8")
    base_config_path = tmp_path / "transcriber.json"
    base_config_path.write_text("{}", encoding="utf-8")
    eval_zip = tmp_path / "Session_22.zip"
    eval_zip.write_bytes(b"zip")
    eval_transcript = tmp_path / "Session_22.jsonl"
    eval_transcript.write_text("{}", encoding="utf-8")

    recipe_path = tmp_path / "baseline_recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "output_root": str(tmp_path / "baseline_output"),
                "hf_cache_root": str(tmp_path / "hf_cache"),
                "metrics_log_path": str(tmp_path / "baseline_output" / "stage_metrics.jsonl"),
                "diarization_model": "pyannote/speaker-diarization-community-1",
                "speaker_mapping": str(speaker_mapping_path),
                "base_config": str(base_config_path),
                "training_inputs": [str(training_root)],
                "transcript_roots": [str(transcripts_root)],
                "core_speakers": ["Player Alice", "Player Bob"],
                "eval": [
                    {
                        "name": "Session22",
                        "session_zip": str(eval_zip),
                        "transcript": str(eval_transcript),
                    }
                ],
                "baseline_pack": ["mixed_raw", "light_x1", "discord_x1"],
                "build_variants": ["mixed_raw", "light_x1", "discord_x1"],
                "resume": True,
            }
        ),
        encoding="utf-8",
    )

    clean_dataset = _dataset(
        ["Player Alice", "Player Bob"],
        domains=["clean", "clean"],
        sources=["clean_raw", "clean_raw"],
        sessions=["Session_10", "Session_10"],
    )
    mixed_base_dataset = _dataset(
        ["Player Alice", "Player Bob"],
        domains=["mixed", "mixed"],
        sources=["mixed_raw", "mixed_raw"],
        sessions=["Session_10", "Session_10"],
    )
    light_dataset = _dataset(
        ["Player Alice", "Player Bob"],
        domains=["mixed_aug", "mixed_aug"],
        sources=["mixed_aug", "mixed_aug"],
        sessions=["Session_10", "Session_10"],
    )
    discord_dataset = _dataset(
        ["Player Alice", "Player Bob"],
        domains=["mixed_aug", "mixed_aug"],
        sources=["mixed_aug", "mixed_aug"],
        sessions=["Session_10", "Session_10"],
    )
    hard_negative_dataset = _dataset(
        ["Player Alice"],
        domains=["hard_negative"],
        sources=["hard_negative"],
        sessions=["Session_61"],
    )

    build_calls = {"multitrack": 0, "materialize": 0}

    def _write_dataset_artifacts(dataset_cache_dir: Path, dataset: ClassifierDataset, *, mixed_base: bool) -> dict:
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "quality_filters": {
                "clipping_fraction_max": 0.005,
                "silence_fraction_max": 0.80,
            },
            "source_groups": {},
            "breakdown": {"by_source": {}},
            "cache_hits": {},
        }
        save_classifier_dataset(dataset_cache_dir, dataset, summary=summary)
        (dataset_cache_dir / "quality_report.json").write_text(
            json.dumps({"records": dataset.samples, "accepted": dataset.samples}),
            encoding="utf-8",
        )
        (dataset_cache_dir / "purity.jsonl").write_text("", encoding="utf-8")
        if mixed_base:
            (dataset_cache_dir / "prepared_windows.jsonl").write_text(
                json.dumps(
                    {
                        "session": "Session_10",
                        "mixed_path": str(dataset_cache_dir / "window_01.wav"),
                        "accepted_segments": [
                            {
                                "start": 0.0,
                                "end": 1.0,
                                "duration": 1.0,
                                "speaker": "Player Alice",
                                "raw_label": "SPEAKER_00",
                                "dominant_share": 0.9,
                                "top1_power": 0.3,
                                "top2_power": 0.05,
                                "active_speakers": 1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (dataset_cache_dir / "window_01.wav").write_bytes(b"fake")
            (dataset_cache_dir / "candidate_pool.jsonl").write_text("", encoding="utf-8")
            np.savez_compressed(
                dataset_cache_dir / "candidate_pool_embeddings.npz",
                embeddings=np.zeros((0, 2), dtype=np.float32),
            )
        return summary

    def fake_build_classifier_dataset_from_multitrack(
        *, dataset_cache_dir, training_mode, **kwargs
    ):
        build_calls["multitrack"] += 1
        dataset = clean_dataset if str(training_mode) == "clean" else mixed_base_dataset
        summary = _write_dataset_artifacts(
            Path(dataset_cache_dir),
            dataset,
            mixed_base=str(training_mode) == "mixed",
        )
        return dataset, summary

    def fake_materialize_classifier_dataset_from_mixed_base(
        *, dataset_cache_dir, augmentation_profile="none", **kwargs
    ):
        build_calls["materialize"] += 1
        dataset = light_dataset if str(augmentation_profile) == "light" else discord_dataset
        summary = _write_dataset_artifacts(Path(dataset_cache_dir), dataset, mixed_base=False)
        summary["materialization_mode"] = "mixed_base_derived"
        summary["base_artifact_id"] = "mixed-base-artifact"
        return dataset, summary

    class FakeSpeakerBank:
        def __init__(self, bank_root: Path, profile: str):
            self.profile_dir = Path(bank_root) / profile
            self.rows: list[tuple[str, np.ndarray, str, dict]] = []

        def extend(self, rows):
            self.rows.extend(rows)

        def save(self) -> None:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            entries = []
            embeddings = []
            for speaker, embedding, _session, metadata in self.rows:
                entries.append({"speaker": speaker, "source": metadata.get("source")})
                embeddings.append(np.asarray(embedding, dtype=np.float32))
            (self.profile_dir / "bank.json").write_text(
                json.dumps({"entries": entries}),
                encoding="utf-8",
            )
            np.save(
                self.profile_dir / "embeddings.npy",
                np.vstack(embeddings).astype(np.float32),
            )

        def render_pca(self, output_path: Path) -> Path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-pca")
            return output_path

    def fake_build_classifier_dataset_from_bank(**kwargs):
        summary = {
            "quality_filters": {
                "clipping_fraction_max": 0.005,
                "silence_fraction_max": 0.80,
            },
            "source_groups": {},
        }
        return clean_dataset, summary

    def fake_train_segment_classifier_from_dataset(*, profile_dir, base_summary=None, **kwargs):
        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "segment_classifier.meta.json").write_text(
            json.dumps(base_summary or {}),
            encoding="utf-8",
        )
        return {"artifacts": {"meta": str(profile_dir / "segment_classifier.meta.json")}}

    def fake_evaluate_multitrack_session(*, session_zip, output_dir, windows_override=None, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predicted_path = output_dir / "predicted.jsonl"
        predicted_path.write_text(
            json.dumps(
                {
                    "speaker_match": {"margin": 0.05},
                    "speaker_match_candidates": [
                        {"speaker": "Player Alice", "score": 0.55},
                        {"speaker": "Player Bob", "score": 0.50},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        purity_path = output_dir / "purity.json"
        purity_path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "speaker": "Player Alice",
                            "second_speaker": "Player Bob",
                            "dominant_share": 0.62,
                            "active_speakers": 2,
                            "top1_power": 0.8,
                            "top2_power": 0.7,
                            "start": 0.0,
                            "end": 1.0,
                            "session": Path(session_zip).stem,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        segment_embedding_path = output_dir / "segment_embeddings.npz"
        np.savez_compressed(
            segment_embedding_path,
            segment_indices=np.asarray([0], dtype=np.int32),
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        summary = {
            "session_zip": str(session_zip),
            "results": [
                {
                    "predicted_jsonl": str(predicted_path),
                    "diarization_purity_path": str(purity_path),
                    "segment_embedding_path": str(segment_embedding_path),
                    "metrics": {
                        "accuracy": 0.8,
                        "coverage": 0.9,
                        "matched_accuracy": 0.88,
                        "confusion": {"Player Alice": {"Player Bob": 1}},
                    },
                }
            ],
            "summary_path": str(output_dir / "summary.json"),
            "windows_override_used": bool(windows_override),
        }
        Path(summary["summary_path"]).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fake_build_hard_negative_dataset(**kwargs):
        records = [
            {
                "speaker": "Player Alice",
                "confusion_partner": "Player Bob",
                "source_session": "Session_61",
                "window_index": 1,
                "dominant_share": 0.61,
                "active_speakers": 2,
                "top1_power": 0.8,
                "top2_power": 0.7,
                "score_margin": 0.05,
                "selection_reason": "eval_top2_margin",
                "source_kind": "eval",
                "start": 0.0,
                "end": 1.0,
            }
        ]
        return hard_negative_dataset, records, {"tracked_pairs": [("Player Alice", "Player Bob")], "selected": 1}

    monkeypatch.setattr(
        baseline_prep,
        "build_classifier_dataset_from_multitrack",
        fake_build_classifier_dataset_from_multitrack,
    )
    monkeypatch.setattr(
        baseline_prep,
        "materialize_classifier_dataset_from_mixed_base",
        fake_materialize_classifier_dataset_from_mixed_base,
    )
    monkeypatch.setattr(baseline_prep, "SpeakerBank", FakeSpeakerBank)
    monkeypatch.setattr(
        baseline_prep, "build_classifier_dataset_from_bank", fake_build_classifier_dataset_from_bank
    )
    monkeypatch.setattr(
        baseline_prep, "train_segment_classifier_from_dataset", fake_train_segment_classifier_from_dataset
    )
    monkeypatch.setattr(baseline_prep, "evaluate_multitrack_session", fake_evaluate_multitrack_session)
    monkeypatch.setattr(
        baseline_prep,
        "_derive_short_segment_slice",
        lambda eval_specs, **kwargs: (
            eval_specs[0],
            {"start": 0.0, "end": 30.0, "speaker_count": 2},
        ),
    )
    monkeypatch.setattr(baseline_prep, "build_hard_negative_dataset", fake_build_hard_negative_dataset)
    monkeypatch.setattr(baseline_prep, "current_git_commit", lambda **kwargs: "deadbeef")

    first_summary = baseline_prep.prepare_baseline(recipe_path=recipe_path)
    second_summary = baseline_prep.prepare_baseline(recipe_path=recipe_path)

    assert build_calls["multitrack"] == 2
    assert build_calls["materialize"] == 2
    assert Path(first_summary["bank_manifest_path"]).exists()
    assert Path(first_summary["mixed_base_manifest_path"]).exists()
    assert Path(first_summary["base_training_manifest_path"]).exists()
    assert Path(first_summary["final_training_manifest_path"]).exists()
    assert Path(first_summary["eval_manifest_path"]).exists()
    assert Path(first_summary["hard_negative_manifest_path"]).exists()
    assert Path(first_summary["narrow_doe_recipe_path"]).exists()
    assert Path(first_summary["stage_metrics_path"]).exists()
    assert "short_segment_slice" in first_summary["canonical_eval"]
    assert set(first_summary["stage_manifests"]) == {
        "bank",
        "mixed_base",
        "variants",
        "hard_negatives",
        "train",
        "eval",
    }
    assert second_summary["stage_manifests"] == first_summary["stage_manifests"]

    final_dataset, final_summary = load_classifier_dataset(
        Path(first_summary["final_training_manifest_path"]).parent
    )
    assert "hard_negative" in final_dataset.sources
    assert "quality_filters" in final_summary
    assert "source_groups" in final_summary
