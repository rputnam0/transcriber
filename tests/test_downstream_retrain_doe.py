from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import transcriber.downstream_retrain_doe as downstream_retrain_doe
from transcriber.segment_classifier import ClassifierDataset, save_classifier_dataset


def _dataset(
    labels: list[str],
    *,
    domains: list[str],
    sources: list[str],
    sessions: list[str],
) -> ClassifierDataset:
    rows = []
    for index, label in enumerate(labels):
        if "Alice" in label:
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


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _baseline_fixture(tmp_path: Path) -> dict[str, Path]:
    hf_cache_root = tmp_path / ".hf_cache"
    output_root = tmp_path / ".outputs" / "baseline"
    bank_profile_dir = hf_cache_root / "speaker_bank" / "baseline_bank_profile"
    bank_profile_dir.mkdir(parents=True)
    (bank_profile_dir / "bank.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    np.save(bank_profile_dir / "embeddings.npy", np.zeros((1, 2), dtype=np.float32))

    base_training_dir = output_root / "artifacts" / "datasets" / "baseline_pack" / "base"
    final_training_dir = output_root / "artifacts" / "datasets" / "final_training" / "final"
    hard_negative_dir = output_root / "artifacts" / "datasets" / "hard_negative" / "hn"
    mixed_raw_dir = output_root / "artifacts" / "datasets" / "mixed_base" / "mixed"
    for path in [base_training_dir, final_training_dir, hard_negative_dir, mixed_raw_dir]:
        path.mkdir(parents=True, exist_ok=True)

    dataset = _dataset(
        ["Player Alice", "Player Bob"],
        domains=["mixed", "mixed"],
        sources=["mixed_raw", "mixed_raw"],
        sessions=["Session_10", "Session_10"],
    )
    save_classifier_dataset(
        base_training_dir,
        dataset,
        summary={"artifact_id": "base-artifact", "parent_artifacts": [], "source_groups": {}},
    )
    save_classifier_dataset(
        final_training_dir,
        dataset,
        summary={
            "artifact_id": "final-artifact",
            "parent_artifacts": ["base-artifact"],
            "source_groups": {},
        },
    )
    save_classifier_dataset(
        hard_negative_dir,
        dataset,
        summary={"artifact_id": "hn-artifact", "parent_artifacts": ["base-artifact"]},
    )

    base_config_path = tmp_path / "transcriber.json"
    base_config_path.write_text("{}", encoding="utf-8")
    speaker_mapping_path = tmp_path / "speaker_mapping.json"
    speaker_mapping_path.write_text("{}", encoding="utf-8")

    session22_zip = tmp_path / "Session 22.zip"
    session22_zip.write_bytes(b"zip")
    session61_zip = tmp_path / "Session 61.zip"
    session61_zip.write_bytes(b"zip")
    session58_zip = tmp_path / "Session 58.zip"
    session58_zip.write_bytes(b"zip")
    session22_txt = tmp_path / "Session 22.txt"
    session22_txt.write_text("{}", encoding="utf-8")
    session61_txt = tmp_path / "Session 61.txt"
    session61_txt.write_text("{}", encoding="utf-8")
    session58_txt = tmp_path / "Session 58.txt"
    session58_txt.write_text("{}", encoding="utf-8")

    recipe_path = tmp_path / "recipe.json"
    _write_json(
        recipe_path,
        {
            "output_root": str(output_root),
            "hf_cache_root": str(hf_cache_root),
            "base_config": str(base_config_path),
            "speaker_mapping": str(speaker_mapping_path),
            "diarization_model": "pyannote/speaker-diarization-community-1",
            "eval": [
                {
                    "name": "Session22",
                    "session_zip": str(session22_zip),
                    "transcript": str(session22_txt),
                },
                {
                    "name": "Session61",
                    "session_zip": str(session61_zip),
                    "transcript": str(session61_txt),
                },
            ],
            "mining_heuristic_eval": [
                {
                    "name": "Session22",
                    "session_zip": str(session22_zip),
                    "transcript": str(session22_txt),
                },
                {
                    "name": "Session61",
                    "session_zip": str(session61_zip),
                    "transcript": str(session61_txt),
                },
            ],
            "eval_final": [
                {
                    "name": "Session58",
                    "session_zip": str(session58_zip),
                    "transcript": str(session58_txt),
                }
            ],
            "hard_negative_candidate_variants": ["mixed_raw"],
            "seed_confusion_pairs": [["Cyrus Schwert", "Cletus Cobbington"]],
            "top_confusion_pairs": 0,
            "hard_negative_per_pair_cap": 60,
            "hard_negative_pair_caps": [
                {"pair": ["Kaladen Shash", "Dungeon Master"], "value": 20},
                {"pair": ["Kaladen Shash", "Leopold Magnus"], "value": 30},
                {"pair": ["David Tanglethorn", "Leopold Magnus"], "value": 40},
            ],
            "hard_negative_per_speaker_cap": 160,
            "hard_negative_max_fraction": 0.20,
            "hard_negative_style_profile_name": "session61_like",
            "hard_negative_style_score_threshold": 0.45,
            "hard_negative_min_style_samples_per_pair": 10,
            "speaker_bank_overrides": {
                "repair": {"enabled": True},
                "session_graph": {"enabled": True, "alpha": 0.8},
            },
            "local_files_only": True,
        },
    )

    eval_manifest_path = output_root / "artifacts" / "eval" / "canonical_baseline" / "manifest" / "eval_manifest.json"
    _write_json(
        eval_manifest_path,
        {
            "canonical_suite": {
                "sessions": ["Session22", "Session61"],
                "short_segment_slice": {
                    "source_session": "Session61",
                    "window": {
                        "start": 0.0,
                        "end": 300.0,
                        "score": 100.0,
                        "speaker_count": 6,
                        "speaker_turns": 111,
                        "total_words": 902,
                        "speakers": [
                            "Cletus Cobbington",
                            "Cyrus Schwert",
                            "David Tanglethorn",
                            "Dungeon Master",
                            "Kaladen Shash",
                            "Leopold Magnus",
                        ],
                    },
                },
            }
        },
    )

    narrow_doe_recipe_path = output_root / "narrow_doe_recipe.json"
    _write_json(
        narrow_doe_recipe_path,
        {
            "bank_profile_dir": str(bank_profile_dir),
            "baseline_profile_dir": str(bank_profile_dir),
            "base_training_dataset_dir": str(base_training_dir),
            "training_dataset_dir": str(final_training_dir),
            "hard_negative_dataset_dir": str(hard_negative_dir),
            "variant_dataset_dirs": {"mixed_raw": str(mixed_raw_dir)},
            "classifier": {
                "model_name": "knn",
                "classifier_c": 1.0,
                "classifier_n_neighbors": 7,
                "classifier_min_margin": 0.06,
                "threshold": 0.36,
                "match_aggregation": "vote",
                "min_segments_per_label": 2,
            },
            "speaker_bank_overrides": {
                "repair": {"enabled": True},
                "session_graph": {
                    "enabled": True,
                    "alpha": 0.8,
                    "override_min_confidence": 0.1,
                    "override_min_margin": 0.0,
                    "override_min_delta": 0.03,
                    "same_raw_label_weight": 0.15,
                },
            },
        },
    )

    baseline_summary_path = output_root / "baseline_summary.json"
    _write_json(
        baseline_summary_path,
        {
            "recipe_path": str(recipe_path),
            "output_root": str(output_root),
            "narrow_doe_recipe_path": str(narrow_doe_recipe_path),
            "eval_manifest_path": str(eval_manifest_path),
            "canonical_eval": {
                "Session22": {
                    "mean_accuracy": 0.762718353946025,
                    "mean_coverage": 0.9820262256193102,
                    "mean_matched_accuracy": 0.7770244686285506,
                },
                "Session61": {
                    "mean_accuracy": 0.6333446527877175,
                    "mean_coverage": 0.9723723432296487,
                    "mean_matched_accuracy": 0.6515010400294632,
                },
                "short_segment_slice": {
                    "mean_accuracy": 0.6663078579117331,
                    "mean_coverage": 0.9634015069967707,
                    "mean_matched_accuracy": 0.6916201117318436,
                },
            },
        },
    )
    return {
        "baseline_summary_path": baseline_summary_path,
        "output_root": output_root,
        "recipe_path": recipe_path,
    }


def _spec_path(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "spec.json"
    _write_json(path, payload)
    return path


def _evaluate_stub(metric_map: dict[tuple[str, str], dict[str, float]]):
    experiment_names = sorted({key[0] for key in metric_map}, key=len, reverse=True)

    def fake_evaluate_multitrack_session(*, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_text = str(output_dir)
        experiment_name = next(
            (name for name in experiment_names if f"/{name}/" in output_text or output_text.endswith(name)),
            "baseline",
        )
        session_name = output_dir.name
        metrics = dict(metric_map[(experiment_name, session_name)])
        summary = {
            "results": [
                {
                    "metrics": metrics,
                    "metrics_pre_graph": {
                        "accuracy": max(metrics["accuracy"] - 0.01, 0.0),
                        "matched_accuracy": max(metrics["matched_accuracy"] - 0.01, 0.0),
                    },
                }
            ]
        }
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    return fake_evaluate_multitrack_session


def _training_stub(*, dataset, profile_dir, model_name, classifier_c, classifier_n_neighbors, base_summary=None):
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "segment_classifier.meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "classifier_c": classifier_c,
                "classifier_n_neighbors": classifier_n_neighbors,
            }
        ),
        encoding="utf-8",
    )
    return {
        **dict(base_summary or {}),
        "model_name": model_name,
        "classifier": {
            "n_neighbors": classifier_n_neighbors,
            "c": classifier_c,
        },
        "samples": dataset.samples,
    }


def test_load_experiment_spec_supports_both_modes():
    spec = downstream_retrain_doe._load_experiment_spec(
        Path(__file__).resolve().parents[1] / "config" / "downstream_retrain_doe.dev.yaml"
    )
    assert spec["dev_only"] is True
    assert len(spec["phase_a"]["family_sweep"]) == 9
    assert len(spec["phase_b"]["experiments"]) == 5
    assert spec["phase_b"]["candidate_variants"] == ["mixed_raw"]


def test_sort_results_prefers_session61_then_short_slice_then_session22():
    ranked = downstream_retrain_doe._sort_results(
        [
            {
                "name": "a",
                "session_results": {
                    "Session22": {"mean_accuracy": 0.80, "mean_matched_accuracy": 0.81},
                    "Session61": {"mean_matched_accuracy": 0.66},
                    "short_segment_slice": {"mean_matched_accuracy": 0.69},
                },
            },
            {
                "name": "b",
                "session_results": {
                    "Session22": {"mean_accuracy": 0.79, "mean_matched_accuracy": 0.80},
                    "Session61": {"mean_matched_accuracy": 0.67},
                    "short_segment_slice": {"mean_matched_accuracy": 0.68},
                },
            },
        ]
    )
    assert ranked[0]["name"] == "b"


def test_phase_a_classifier_only_writes_report_and_skips_hard_negative_builder(
    tmp_path, monkeypatch
):
    fixture = _baseline_fixture(tmp_path)
    spec_path = _spec_path(
        tmp_path,
        {
            "dev_only": True,
            "acceptance": {
                "session61_matched_accuracy_gain": 0.01,
                "session22_accuracy_regression_max": 0.01,
            },
            "phase_a": {
                "family_sweep": [
                    {"name": "knn_k7", "classifier": {"model_name": "knn", "classifier_n_neighbors": 7}},
                    {
                        "name": "lda_knn_k11",
                        "classifier": {"model_name": "lda_knn", "classifier_n_neighbors": 11},
                    },
                ],
                "calibration": {"top_n": 2, "thresholds": [0.38], "classifier_min_margins": [0.04]},
            },
            "phase_b": {"candidate_variants": ["mixed_raw"], "experiments": []},
        },
    )

    metric_map = {
        ("knn_k7", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("lda_knn_k11", "Session22"): {"accuracy": 0.7560, "coverage": 0.98, "matched_accuracy": 0.7700},
        ("lda_knn_k11", "Session61"): {"accuracy": 0.6450, "coverage": 0.97, "matched_accuracy": 0.6640},
        ("lda_knn_k11", "short_segment_slice"): {"accuracy": 0.6750, "coverage": 0.96, "matched_accuracy": 0.7010},
        ("lda_knn_k11_thr_0_38_margin_0_04", "Session22"): {
            "accuracy": 0.7565,
            "coverage": 0.98,
            "matched_accuracy": 0.7705,
        },
        ("lda_knn_k11_thr_0_38_margin_0_04", "Session61"): {
            "accuracy": 0.6480,
            "coverage": 0.97,
            "matched_accuracy": 0.6660,
        },
        ("lda_knn_k11_thr_0_38_margin_0_04", "short_segment_slice"): {
            "accuracy": 0.6770,
            "coverage": 0.96,
            "matched_accuracy": 0.7040,
        },
        ("knn_k7_thr_0_38_margin_0_04", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7_thr_0_38_margin_0_04", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7_thr_0_38_margin_0_04", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
    }

    monkeypatch.setattr(
        downstream_retrain_doe,
        "evaluate_multitrack_session",
        _evaluate_stub(metric_map),
    )
    monkeypatch.setattr(
        downstream_retrain_doe,
        "train_segment_classifier_from_dataset",
        _training_stub,
    )

    def fail_hard_negative_builder(*args, **kwargs):
        raise AssertionError("hard-negative builder should not run during classifier-only promotion")

    monkeypatch.setattr(
        downstream_retrain_doe,
        "build_hard_negative_dataset",
        fail_hard_negative_builder,
    )

    report = downstream_retrain_doe.run_downstream_retrain_doe(
        baseline_summary_path=fixture["baseline_summary_path"],
        spec_path=spec_path,
        output_dir=tmp_path / "doe_output",
        device="cpu",
    )

    assert report["phase_b"]["skipped"] is True
    assert report["promoted_result"]["name"] == "lda_knn_k11_thr_0_38_margin_0_04"
    assert report["promoted_result"]["mode"] == "classifier_only"
    assert (tmp_path / "doe_output" / "downstream_retrain_doe_report.json").exists()


def test_phase_b_hard_negative_refresh_promotes_best_experiment(tmp_path, monkeypatch):
    fixture = _baseline_fixture(tmp_path)
    spec_path = _spec_path(
        tmp_path,
        {
            "dev_only": True,
            "acceptance": {
                "session61_matched_accuracy_gain": 0.01,
                "session22_accuracy_regression_max": 0.01,
            },
            "phase_a": {
                "family_sweep": [
                    {"name": "knn_k7", "classifier": {"model_name": "knn", "classifier_n_neighbors": 7}},
                ],
                "calibration": {"top_n": 1, "thresholds": [0.36], "classifier_min_margins": [0.06]},
            },
            "phase_b": {
                "candidate_variants": ["mixed_raw"],
                "experiments": [
                    {"name": "control_current", "hard_negative": {}},
                    {
                        "name": "combined_precision_v1",
                        "hard_negative": {
                            "pair_caps": [
                                {"pair": ["Cletus Cobbington", "Cyrus Schwert"], "value": 80}
                            ],
                            "hard_negative_style_score_threshold": 0.55,
                        },
                    },
                ],
            },
        },
    )

    metric_map = {
        ("knn_k7", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("knn_k7_thr_0_36_margin_0_06", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7_thr_0_36_margin_0_06", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7_thr_0_36_margin_0_06", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("control_current", "Session22"): {"accuracy": 0.7625, "coverage": 0.98, "matched_accuracy": 0.7765},
        ("control_current", "Session61"): {"accuracy": 0.6350, "coverage": 0.97, "matched_accuracy": 0.6520},
        ("control_current", "short_segment_slice"): {"accuracy": 0.6670, "coverage": 0.96, "matched_accuracy": 0.6920},
        ("combined_precision_v1", "Session22"): {"accuracy": 0.7580, "coverage": 0.98, "matched_accuracy": 0.7730},
        ("combined_precision_v1", "Session61"): {"accuracy": 0.6460, "coverage": 0.97, "matched_accuracy": 0.6630},
        ("combined_precision_v1", "short_segment_slice"): {"accuracy": 0.6760, "coverage": 0.96, "matched_accuracy": 0.7040},
    }
    monkeypatch.setattr(
        downstream_retrain_doe,
        "evaluate_multitrack_session",
        _evaluate_stub(metric_map),
    )
    monkeypatch.setattr(
        downstream_retrain_doe,
        "train_segment_classifier_from_dataset",
        _training_stub,
    )

    def fake_build_hard_negative_dataset(**kwargs):
        experiment_dir = kwargs["candidate_pool_dirs"][0]
        dataset = _dataset(
            ["Player Alice"],
            domains=["hard_negative"],
            sources=["hard_negative"],
            sessions=["Session_10"],
        )
        records = [
            {
                "speaker": "Player Alice",
                "source_session": "Session_10",
                "tracked_pair": ["Cyrus Schwert", "Cletus Cobbington"],
            }
        ]
        summary = {
            "selected": 1,
            "tracked_pairs": [["Cyrus Schwert", "Cletus Cobbington"]],
            "candidate_variant_hint": str(experiment_dir),
        }
        return dataset, records, summary

    monkeypatch.setattr(
        downstream_retrain_doe,
        "build_hard_negative_dataset",
        fake_build_hard_negative_dataset,
    )

    report = downstream_retrain_doe.run_downstream_retrain_doe(
        baseline_summary_path=fixture["baseline_summary_path"],
        spec_path=spec_path,
        output_dir=tmp_path / "doe_output",
        device="cpu",
    )

    assert report["phase_b"]["skipped"] is False
    assert report["promoted_result"]["name"] == "combined_precision_v1"
    assert report["promoted_result"]["mode"] == "hard_negative_refresh"


def test_hard_negative_refresh_rejects_eval_session_records(tmp_path, monkeypatch):
    fixture = _baseline_fixture(tmp_path)
    spec_path = _spec_path(
        tmp_path,
        {
            "dev_only": True,
            "phase_a": {
                "family_sweep": [{"name": "knn_k7", "classifier": {"model_name": "knn", "classifier_n_neighbors": 7}}],
                "calibration": {"top_n": 1, "thresholds": [0.36], "classifier_min_margins": [0.06]},
            },
            "phase_b": {
                "candidate_variants": ["mixed_raw"],
                "experiments": [{"name": "control_current", "hard_negative": {}}],
            },
        },
    )

    metric_map = {
        ("knn_k7", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("knn_k7_thr_0_36_margin_0_06", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7_thr_0_36_margin_0_06", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7_thr_0_36_margin_0_06", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("control_current", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("control_current", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("control_current", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
    }
    monkeypatch.setattr(
        downstream_retrain_doe,
        "evaluate_multitrack_session",
        _evaluate_stub(metric_map),
    )
    monkeypatch.setattr(
        downstream_retrain_doe,
        "train_segment_classifier_from_dataset",
        _training_stub,
    )

    def leaky_build_hard_negative_dataset(**kwargs):
        dataset = _dataset(
            ["Player Alice"],
            domains=["hard_negative"],
            sources=["hard_negative"],
            sessions=["Session_61"],
        )
        records = [{"speaker": "Player Alice", "source_session": "Session61"}]
        return dataset, records, {"selected": 1, "tracked_pairs": []}

    monkeypatch.setattr(
        downstream_retrain_doe,
        "build_hard_negative_dataset",
        leaky_build_hard_negative_dataset,
    )

    with pytest.raises(RuntimeError, match="train on eval sessions"):
        downstream_retrain_doe.run_downstream_retrain_doe(
            baseline_summary_path=fixture["baseline_summary_path"],
            spec_path=spec_path,
            output_dir=tmp_path / "doe_output",
            device="cpu",
        )


def test_hard_negative_refresh_rejects_final_eval_session_records(tmp_path, monkeypatch):
    fixture = _baseline_fixture(tmp_path)
    spec_path = _spec_path(
        tmp_path,
        {
            "dev_only": True,
            "phase_a": {
                "family_sweep": [{"name": "knn_k7", "classifier": {"model_name": "knn", "classifier_n_neighbors": 7}}],
                "calibration": {"top_n": 1, "thresholds": [0.36], "classifier_min_margins": [0.06]},
            },
            "phase_b": {
                "candidate_variants": ["mixed_raw"],
                "experiments": [{"name": "control_current", "hard_negative": {}}],
            },
        },
    )

    metric_map = {
        ("knn_k7", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("knn_k7_thr_0_36_margin_0_06", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("knn_k7_thr_0_36_margin_0_06", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("knn_k7_thr_0_36_margin_0_06", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
        ("control_current", "Session22"): {"accuracy": 0.7627, "coverage": 0.98, "matched_accuracy": 0.7770},
        ("control_current", "Session61"): {"accuracy": 0.6333, "coverage": 0.97, "matched_accuracy": 0.6515},
        ("control_current", "short_segment_slice"): {"accuracy": 0.6663, "coverage": 0.96, "matched_accuracy": 0.6916},
    }
    monkeypatch.setattr(
        downstream_retrain_doe,
        "evaluate_multitrack_session",
        _evaluate_stub(metric_map),
    )
    monkeypatch.setattr(
        downstream_retrain_doe,
        "train_segment_classifier_from_dataset",
        _training_stub,
    )

    def leaky_build_hard_negative_dataset(**kwargs):
        dataset = _dataset(
            ["Player Alice"],
            domains=["hard_negative"],
            sources=["hard_negative"],
            sessions=["Session_58"],
        )
        records = [{"speaker": "Player Alice", "source_session": "Session58"}]
        return dataset, records, {"selected": 1, "tracked_pairs": []}

    monkeypatch.setattr(
        downstream_retrain_doe,
        "build_hard_negative_dataset",
        leaky_build_hard_negative_dataset,
    )

    with pytest.raises(RuntimeError, match="train on eval sessions"):
        downstream_retrain_doe.run_downstream_retrain_doe(
            baseline_summary_path=fixture["baseline_summary_path"],
            spec_path=spec_path,
            output_dir=tmp_path / "doe_output",
            device="cpu",
        )
