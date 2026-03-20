from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import transcriber.autoresearch_speaker_id as autoresearch
import transcriber.downstream_retrain_doe as downstream_retrain_doe
from transcriber.segment_classifier import ClassifierDataset, save_classifier_dataset


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _init_git_repo(workspace: Path) -> None:
    subprocess.run(["git", "init"], cwd=str(workspace), check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(workspace), check=True)
    subprocess.run(["git", "add", "."], cwd=str(workspace), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(workspace), check=True, capture_output=True, text=True)


def _dataset(labels: list[str]) -> ClassifierDataset:
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
        domains=["mixed"] * count,
        sources=["mixed_raw"] * count,
        sessions=["Session_10"] * count,
        durations=np.full(count, 1.0, dtype=np.float32),
        dominant_shares=np.full(count, 0.9, dtype=np.float32),
        top1_powers=np.full(count, 0.3, dtype=np.float32),
        top2_powers=np.full(count, 0.05, dtype=np.float32),
        active_speakers=np.full(count, 1, dtype=np.int32),
    )


def _baseline_fixture(workspace: Path, *, include_final_metrics: bool = True) -> dict[str, Path]:
    output_root = workspace / ".outputs" / "baseline"
    hf_cache_root = workspace / ".hf_cache"
    bank_profile_dir = hf_cache_root / "speaker_bank" / "baseline_bank_profile"
    bank_profile_dir.mkdir(parents=True)
    _write_json(bank_profile_dir / "bank.json", {"entries": []})
    np.save(bank_profile_dir / "embeddings.npy", np.zeros((1, 2), dtype=np.float32))

    base_training_dir = output_root / "artifacts" / "datasets" / "baseline_pack" / "base"
    final_training_dir = output_root / "artifacts" / "datasets" / "final_training" / "final"
    hard_negative_dir = output_root / "artifacts" / "datasets" / "hard_negative" / "hn"
    mixed_raw_dir = output_root / "artifacts" / "datasets" / "mixed_base" / "mixed"
    for path in [base_training_dir, final_training_dir, hard_negative_dir, mixed_raw_dir]:
        path.mkdir(parents=True, exist_ok=True)

    dataset = _dataset(["Player Alice", "Player Bob"])
    save_classifier_dataset(
        base_training_dir,
        dataset,
        summary={"artifact_id": "base-artifact", "parent_artifacts": [], "source_groups": {}},
    )
    save_classifier_dataset(
        final_training_dir,
        dataset,
        summary={"artifact_id": "final-artifact", "parent_artifacts": ["base-artifact"], "source_groups": {}},
    )
    save_classifier_dataset(
        hard_negative_dir,
        dataset,
        summary={"artifact_id": "hn-artifact", "parent_artifacts": ["base-artifact"]},
    )

    config_dir = workspace / "config"
    _write_json(config_dir / "transcriber.yaml", {})
    _write_json(config_dir / "speaker_mapping.yaml", {})
    recipe_path = config_dir / "speaker_recipe.json"
    _write_json(
        recipe_path,
        {
            "base_config": str(config_dir / "transcriber.yaml"),
            "speaker_mapping": str(config_dir / "speaker_mapping.yaml"),
            "hf_cache_root": str(hf_cache_root),
            "diarization_model": "pyannote/speaker-diarization-community-1",
            "eval_dev": [
                {"name": "Session22", "session_zip": str(workspace / "Session 22.zip"), "transcript": str(workspace / "Session 22.txt")},
                {"name": "Session61", "session_zip": str(workspace / "Session 61.zip"), "transcript": str(workspace / "Session 61.txt")},
            ],
            "mining_heuristic_eval": [
                {"name": "Session22", "session_zip": str(workspace / "Session 22.zip"), "transcript": str(workspace / "Session 22.txt")},
                {"name": "Session61", "session_zip": str(workspace / "Session 61.zip"), "transcript": str(workspace / "Session 61.txt")},
            ],
            "eval_final": [
                {"name": "Session58", "session_zip": str(workspace / "Session 58.zip"), "transcript": str(workspace / "Session 58.txt")},
            ],
            "hard_negative_candidate_variants": ["mixed_raw"],
            "seed_confusion_pairs": [["Cyrus Schwert", "Cletus Cobbington"]],
            "top_confusion_pairs": 0,
            "hard_negative_per_pair_cap": 60,
            "hard_negative_pair_caps": [{"pair": ["Kaladen Shash", "Dungeon Master"], "value": 20}],
            "hard_negative_per_speaker_cap": 160,
            "hard_negative_max_fraction": 0.20,
            "hard_negative_style_profile_name": "session61_like",
            "hard_negative_style_score_threshold": 0.45,
            "hard_negative_min_style_samples_per_pair": 10,
            "current_winner": {
                "model_name": "knn",
                "classifier_c": 1.0,
                "classifier_n_neighbors": 9,
                "classifier_min_margin": 0.08,
                "threshold": 0.34,
                "match_aggregation": "vote",
                "min_segments_per_label": 2,
            },
            "speaker_bank_overrides": {
                "repair": {"enabled": True},
                "session_graph": {"enabled": True, "alpha": 0.8},
            },
            "local_files_only": True,
        },
    )

    for name in ["Session 22.zip", "Session 61.zip", "Session 58.zip"]:
        (workspace / name).write_bytes(b"zip")
    for name in ["Session 22.txt", "Session 61.txt", "Session 58.txt"]:
        _write_text(workspace / name, "{}")

    session_summary_dir = output_root / "artifacts" / "eval" / "canonical_dev" / "manifest"
    session22_summary = _write_json(
        session_summary_dir / "Session22" / "summary.json",
        {"graph_pair_diagnostics": {"Cletus Cobbington::Cyrus Schwert": {"post_confusions": 7}}},
    )
    session61_summary = _write_json(
        session_summary_dir / "Session61" / "summary.json",
        {"graph_pair_diagnostics": {"Dungeon Master::Kaladen Shash": {"post_confusions": 11}}},
    )
    eval_manifest_path = _write_json(
        session_summary_dir / "eval_manifest.json",
        {
            "canonical_suite": {
                "short_segment_slice": {
                    "source_session": "Session61",
                    "window": {"start": 0.0, "end": 300.0},
                }
            },
            "summary_paths": [str(session22_summary), str(session61_summary)],
        },
    )

    narrow_doe_recipe_path = _write_json(
        output_root / "narrow_doe_recipe.json",
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
                "classifier_n_neighbors": 9,
                "classifier_min_margin": 0.08,
                "threshold": 0.34,
                "match_aggregation": "vote",
                "min_segments_per_label": 2,
            },
            "speaker_bank_overrides": {"repair": {"enabled": True}, "session_graph": {"enabled": True, "alpha": 0.8}},
        },
    )

    baseline_summary = {
        "recipe_path": str(recipe_path),
        "output_root": str(output_root),
        "narrow_doe_recipe_path": str(narrow_doe_recipe_path),
        "dev_eval_manifest_path": str(eval_manifest_path),
        "eval_manifest_path": str(eval_manifest_path),
        "dev_eval": {
            "Session22": {"mean_accuracy": 0.7627, "mean_matched_accuracy": 0.7770},
            "Session61": {"mean_accuracy": 0.6333, "mean_matched_accuracy": 0.6515},
            "short_segment_slice": {"mean_matched_accuracy": 0.6916},
        },
    }
    if include_final_metrics:
        baseline_summary["final_eval"] = {
            "Session58": {"mean_accuracy": 0.60, "mean_matched_accuracy": 0.70}
        }
    baseline_summary_path = _write_json(output_root / "baseline_summary.json", baseline_summary)
    return {
        "baseline_summary_path": baseline_summary_path,
        "output_root": output_root,
        "recipe_path": recipe_path,
    }


def _config_path(workspace: Path, *, output_root: str = ".outputs/autoresearch") -> Path:
    config_path = workspace / "config" / "autopilot.json"
    _write_json(
        config_path,
        {
            "mode": "dev_autopilot_no_rebuild",
            "workspace_root": str(workspace),
            "output_root": output_root,
            "program_path": "speaker_id_program.md",
            "bootstrap": {
                "dev_champion_baseline_summary": ".outputs/baseline/baseline_summary.json"
            },
            "families": {
                "eval_only": {"enabled": True, "order": 0},
                "classifier_only": {"enabled": True, "order": 1},
                "hard_negative_refresh": {"enabled": True, "order": 2},
                "full_rebuild": {"enabled": True, "order": 3},
            },
            "protected_paths": [
                "config/autoresearch_speaker_id.yaml",
                "speaker_id_program.md",
                "src/transcriber/autoresearch_objective.py",
                ".outputs/autoresearch",
            ],
        },
    )
    return config_path


class _WriteFileBackend:
    def __init__(self, relative_path: str, text: str = "change\n") -> None:
        self.relative_path = relative_path
        self.text = text

    def run(self, *, worktree_root: Path, log_path: Path, output_last_message_path: Path, **kwargs) -> dict[str, object]:
        target = worktree_root / self.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.text, encoding="utf-8")
        _write_text(log_path, "mutated")
        _write_text(output_last_message_path, "done")
        return {"exit_code": 0, "timed_out": False}


class _NoChangeBackend:
    def run(self, *, log_path: Path, output_last_message_path: Path, **kwargs) -> dict[str, object]:
        _write_text(log_path, "no changes")
        _write_text(output_last_message_path, "done")
        return {"exit_code": 0, "timed_out": False}


class _StaticExecutor:
    def __init__(self, *, dev_metrics: dict[str, dict[str, float]], final_metrics: dict[str, dict[str, float]] | None = None) -> None:
        self.dev_metrics = dev_metrics
        self.final_metrics = final_metrics or {}

    def run(self, *, run_dir: Path, champion: autoresearch.ChampionState, **kwargs) -> autoresearch.ExperimentOutcome:
        summary_path = _write_json(
            run_dir / "candidate_baseline_summary.json",
            {
                "recipe_path": str(champion.baseline_summary_path),
                "narrow_doe_recipe_path": str(champion.baseline_summary_path),
                "dev_eval": self.dev_metrics,
                "canonical_eval": self.dev_metrics,
                "final_eval": self.final_metrics,
            },
        )
        return autoresearch.ExperimentOutcome(
            family=str(kwargs["family"]),
            result={"session_results": self.dev_metrics, "artifact_paths": {}},
            candidate_summary_path=summary_path,
            dev_metrics=self.dev_metrics,
            final_metrics=self.final_metrics,
            artifact_paths={},
        )


def test_load_config_parses_budgets_and_disables_full_rebuild_in_dev_mode(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace / "speaker_id_program.md", "program")
    config_path = _config_path(workspace)

    config = autoresearch._load_autoresearch_config(config_path)

    assert config.budgets_seconds["classifier_only"] == 90 * 60
    assert "full_rebuild" not in autoresearch._enabled_families(config)


def test_render_prompt_excludes_hidden_final_metrics(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace / "speaker_id_program.md", "base program")
    fixture = _baseline_fixture(workspace)
    config_path = _config_path(workspace)
    config = autoresearch._load_autoresearch_config(config_path)
    champion = autoresearch._champion_from_summary(
        summary_path=fixture["baseline_summary_path"],
        source="bootstrap",
        run_id="bootstrap",
        branch="bootstrap",
        config=config,
        commit_sha="abc123",
    )
    prepared = autoresearch._prepare_run_inputs(
        champion=champion,
        config=config,
        worktree_root=workspace,
        run_dir=workspace / ".outputs" / "prompt_run",
    )

    prompt = autoresearch._render_prompt(
        run_id="run1",
        family="classifier_only",
        champion=champion,
        config=config,
        visible_history=[],
        error_themes=["Session61 remains the primary objective."],
        prepared_inputs=prepared,
    )

    assert "Session58" not in prompt
    assert "0.70" not in prompt
    assert "Session61" in prompt
    visible_summary = json.loads(prepared.run_summary_path.read_text(encoding="utf-8"))
    assert "final_eval" not in visible_summary
    assert "final_eval_manifest_path" not in visible_summary


def test_run_loop_rejects_protected_path_touches(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace / "speaker_id_program.md", "program")
    _baseline_fixture(workspace)
    config_path = _config_path(workspace)
    _write_text(workspace / "README.md", "repo")
    _write_text(workspace / "config" / "autoresearch_speaker_id.yaml", "protected")
    _init_git_repo(workspace)

    report = autoresearch.run_autoresearch_loop(
        config_path=config_path,
        once=True,
        mutation_backend=_WriteFileBackend("config/autoresearch_speaker_id.yaml", "bad\n"),
        experiment_executor=_StaticExecutor(
            dev_metrics={
                "Session22": {"mean_accuracy": 0.80, "mean_matched_accuracy": 0.81},
                "Session61": {"mean_accuracy": 0.70, "mean_matched_accuracy": 0.71},
                "short_segment_slice": {"mean_matched_accuracy": 0.72},
            }
        ),
    )

    assert report["runs"][0]["accepted"] is False
    assert "protected_paths_touched" in str(report["runs"][0]["reason"])


def test_scheduler_escalates_after_five_rejections_and_resets_after_acceptance(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace / "speaker_id_program.md", "program")
    _baseline_fixture(workspace)
    config_path = _config_path(workspace)
    _write_text(workspace / "README.md", "repo")
    _init_git_repo(workspace)

    autoresearch.run_autoresearch_loop(
        config_path=config_path,
        max_runs=5,
        mutation_backend=_NoChangeBackend(),
        experiment_executor=_StaticExecutor(dev_metrics={}),
    )
    state = json.loads((workspace / ".outputs" / "autoresearch" / "state.json").read_text(encoding="utf-8"))
    assert state["current_family"] == "hard_negative_refresh"

    improved_metrics = {
        "Session22": {"mean_accuracy": 0.7530, "mean_matched_accuracy": 0.7680},
        "Session61": {"mean_accuracy": 0.6501, "mean_matched_accuracy": 0.6687},
        "short_segment_slice": {"mean_matched_accuracy": 0.6872},
    }
    autoresearch.run_autoresearch_loop(
        config_path=config_path,
        once=True,
        mutation_backend=_WriteFileBackend("README.md", "new champion\n"),
        experiment_executor=_StaticExecutor(dev_metrics=improved_metrics),
    )
    state = json.loads((workspace / ".outputs" / "autoresearch" / "state.json").read_text(encoding="utf-8"))
    assert state["current_family"] == "eval_only"
    assert all(value == 0 for value in state["consecutive_rejections_by_family"].values())


def test_run_loop_promotes_champion_and_records_ledgers(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace / "speaker_id_program.md", "program")
    _baseline_fixture(workspace)
    config_path = _config_path(workspace)
    _write_text(workspace / "README.md", "repo")
    _init_git_repo(workspace)

    improved_metrics = {
        "Session22": {"mean_accuracy": 0.7530, "mean_matched_accuracy": 0.7680},
        "Session61": {"mean_accuracy": 0.6501, "mean_matched_accuracy": 0.6687},
        "short_segment_slice": {"mean_matched_accuracy": 0.6872},
    }

    report = autoresearch.run_autoresearch_loop(
        config_path=config_path,
        once=True,
        mutation_backend=_WriteFileBackend("README.md", "new champion\n"),
        experiment_executor=_StaticExecutor(dev_metrics=improved_metrics),
    )

    assert report["runs"][0]["accepted"] is True
    champions = json.loads((workspace / ".outputs" / "autoresearch" / "champions.json").read_text(encoding="utf-8"))
    assert champions["dev_champion"]["run_id"] == report["runs"][0]["run_id"]
    assert (workspace / ".outputs" / "autoresearch" / "results.tsv").exists()
    assert (workspace / ".outputs" / "autoresearch" / "results.jsonl").exists()


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
    return {
        **dict(base_summary or {}),
        "model_name": model_name,
        "classifier": {"n_neighbors": classifier_n_neighbors, "c": classifier_c},
        "samples": dataset.samples,
    }


def test_executor_reuses_downstream_artifacts_for_classifier_and_hard_negative_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace / "speaker_id_program.md", "program")
    fixture = _baseline_fixture(workspace)
    config_path = _config_path(workspace)
    config = autoresearch._load_autoresearch_config(config_path)
    champion = autoresearch._champion_from_summary(
        summary_path=fixture["baseline_summary_path"],
        source="bootstrap",
        run_id="bootstrap",
        branch="bootstrap",
        config=config,
        commit_sha="abc123",
    )
    worktree = tmp_path / "worktree"
    shutil.copytree(workspace, worktree)

    metric_map = {
        ("autopilot_classifier_only", "Session22"): {"accuracy": 0.7573, "coverage": 0.98, "matched_accuracy": 0.7714},
        ("autopilot_classifier_only", "Session61"): {"accuracy": 0.6400, "coverage": 0.97, "matched_accuracy": 0.6583},
        ("autopilot_classifier_only", "short_segment_slice"): {"accuracy": 0.6523, "coverage": 0.96, "matched_accuracy": 0.6771},
        ("autopilot_hard_negative_refresh", "Session22"): {"accuracy": 0.7528, "coverage": 0.98, "matched_accuracy": 0.7668},
        ("autopilot_hard_negative_refresh", "Session61"): {"accuracy": 0.6501, "coverage": 0.97, "matched_accuracy": 0.6687},
        ("autopilot_hard_negative_refresh", "short_segment_slice"): {"accuracy": 0.6620, "coverage": 0.96, "matched_accuracy": 0.6872},
    }
    monkeypatch.setattr(
        downstream_retrain_doe,
        "evaluate_multitrack_session",
        _evaluate_stub(metric_map),
    )
    monkeypatch.setattr(downstream_retrain_doe, "train_segment_classifier_from_dataset", _training_stub)
    monkeypatch.setattr(
        autoresearch,
        "prepare_baseline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full rebuild should not run")),
    )

    def fake_build_hard_negative_dataset(**kwargs):
        dataset = _dataset(["Player Alice"])
        records = [{"speaker": "Player Alice", "source_session": "Session_10"}]
        summary = {"selected": 1, "tracked_pairs": [["Cyrus Schwert", "Cletus Cobbington"]]}
        return dataset, records, summary

    monkeypatch.setattr(downstream_retrain_doe, "build_hard_negative_dataset", fake_build_hard_negative_dataset)

    executor = autoresearch.AutoresearchExperimentExecutor()
    classifier_outcome = executor.run(
        family="classifier_only",
        champion=champion,
        config=config,
        worktree_root=worktree,
        run_dir=tmp_path / "classifier_run",
        device="cpu",
        run_budget_seconds=60,
    )
    hard_negative_outcome = executor.run(
        family="hard_negative_refresh",
        champion=champion,
        config=config,
        worktree_root=worktree,
        run_dir=tmp_path / "hard_negative_run",
        device="cpu",
        run_budget_seconds=60,
    )

    assert classifier_outcome.dev_metrics["Session61"]["mean_matched_accuracy"] == pytest.approx(0.6583)
    assert hard_negative_outcome.dev_metrics["Session61"]["mean_matched_accuracy"] == pytest.approx(0.6687)
    candidate_summary = json.loads(hard_negative_outcome.candidate_summary_path.read_text(encoding="utf-8"))
    narrow_doe = json.loads(Path(str(candidate_summary["narrow_doe_recipe_path"])).read_text(encoding="utf-8"))
    assert "training_dataset_dir" in narrow_doe
