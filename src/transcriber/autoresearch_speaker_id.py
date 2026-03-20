from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .autoresearch_objective import (
    DevAcceptanceGate,
    FinalGate,
    ObjectiveWeights,
    compute_dev_score,
    evaluate_dev_candidate,
    evaluate_final_gate,
    load_dev_acceptance_gate,
    load_final_gate,
    load_objective_weights,
)
from .baseline_prep import prepare_baseline
from .cli import _load_yaml_or_json
from .downstream_retrain_doe import (
    _evaluate_dev_suite,
    _load_context as _load_downstream_context,
    run_downstream_retrain_doe,
)
from .prep_artifacts import current_git_commit


FAMILY_ORDER = ("eval_only", "classifier_only", "hard_negative_refresh", "full_rebuild")
DEFAULT_RESULTS_HISTORY = 10


@dataclass(frozen=True)
class ChampionState:
    run_id: str
    commit_sha: str
    branch: str
    baseline_summary_path: Path
    dev_metrics: Dict[str, Dict[str, float]]
    dev_score: float
    final_metrics: Dict[str, Dict[str, float]]
    source: str


@dataclass(frozen=True)
class AutoresearchConfig:
    mode: str
    workspace_root: Path
    output_root: Path
    program_path: Path
    bootstrap_dev_baseline_summary: Path
    bootstrap_release_baseline_summary: Optional[Path]
    splits: Dict[str, List[str]]
    objective_weights: ObjectiveWeights
    dev_acceptance: DevAcceptanceGate
    final_gate: FinalGate
    families: Dict[str, Dict[str, object]]
    budgets_seconds: Dict[str, int]
    scheduler_start_family: str
    rejection_streak_to_escalate: int
    protected_paths: Tuple[str, ...]
    codex_binary: str
    codex_exec_args: Tuple[str, ...]
    device: str
    results_history_limit: int
    config_path: Path


@dataclass(frozen=True)
class PreparedRunInputs:
    parent_summary: Dict[str, object]
    run_summary_path: Path
    run_recipe_path: Path
    run_narrow_doe_recipe_path: Path
    canonical_recipe_path: Path
    remapped_recipe: Dict[str, object]
    remapped_narrow_doe_recipe: Dict[str, object]


@dataclass(frozen=True)
class ExperimentOutcome:
    family: str
    result: Dict[str, object]
    candidate_summary_path: Path
    dev_metrics: Dict[str, Dict[str, float]]
    final_metrics: Dict[str, Dict[str, float]]
    artifact_paths: Dict[str, str]


class MutationBackend(Protocol):
    def run(
        self,
        *,
        prompt_text: str,
        worktree_root: Path,
        log_path: Path,
        output_last_message_path: Path,
        config: AutoresearchConfig,
        run_budget_seconds: int,
    ) -> Dict[str, object]:
        ...


class ExperimentExecutor(Protocol):
    def run(
        self,
        *,
        family: str,
        champion: ChampionState,
        config: AutoresearchConfig,
        worktree_root: Path,
        run_dir: Path,
        device: str,
        run_budget_seconds: int,
    ) -> ExperimentOutcome:
        ...


def _json_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


def _jsonl_append(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload)) + "\n")
    return path


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_metric_map(raw: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    parsed: Dict[str, Dict[str, float]] = {}
    for session_name, metrics in dict(raw).items():
        if not isinstance(metrics, Mapping):
            continue
        parsed[str(session_name)] = {
            key: float(value)
            for key, value in dict(metrics).items()
            if isinstance(value, (int, float))
        }
    return parsed


def _sanitize_visible_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    hidden_keys = {
        "final_eval",
        "final_eval_manifest_path",
        "comparison_summary_path",
    }
    return {key: value for key, value in dict(summary).items() if key not in hidden_keys}


def _summary_dev_metrics(summary: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    return _normalize_metric_map(dict(summary.get("dev_eval") or summary.get("canonical_eval") or {}))


def _summary_final_metrics(summary: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    return _normalize_metric_map(dict(summary.get("final_eval") or {}))


def _parse_duration_seconds(value: object, default_seconds: int) -> int:
    if value is None:
        return int(default_seconds)
    if isinstance(value, (int, float)):
        return int(float(value))
    text = str(value).strip().lower()
    if not text:
        return int(default_seconds)
    multiplier = 1
    if text.endswith("h"):
        multiplier = 3600
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 60
        text = text[:-1]
    elif text.endswith("s"):
        multiplier = 1
        text = text[:-1]
    return int(float(text) * multiplier)


def _resolve_path(value: object, *, workspace_root: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _load_autoresearch_config(config_path: Path) -> AutoresearchConfig:
    raw = _load_yaml_or_json(str(config_path.expanduser().resolve())) or {}
    workspace_root = _resolve_path(raw.get("workspace_root") or ".", workspace_root=Path.cwd())
    output_root = _resolve_path(
        raw.get("output_root") or ".outputs/autoresearch_speaker_id",
        workspace_root=workspace_root,
    )
    bootstrap = dict(raw.get("bootstrap") or {})
    codex = dict(raw.get("codex") or {})
    scheduler = dict(raw.get("scheduler") or {})
    families_raw = dict(raw.get("families") or {})
    families = {
        family: {
            "enabled": bool(dict(families_raw.get(family) or {}).get("enabled", family != "full_rebuild")),
            "order": int(dict(families_raw.get(family) or {}).get("order", FAMILY_ORDER.index(family))),
        }
        for family in FAMILY_ORDER
    }
    budgets_raw = dict(raw.get("budgets") or {})
    budgets_seconds = {
        "eval_only": _parse_duration_seconds(budgets_raw.get("eval_only"), 30 * 60),
        "classifier_only": _parse_duration_seconds(budgets_raw.get("classifier_only"), 90 * 60),
        "hard_negative_refresh": _parse_duration_seconds(
            budgets_raw.get("hard_negative_refresh"), 2 * 60 * 60
        ),
        "full_rebuild": _parse_duration_seconds(budgets_raw.get("full_rebuild"), 6 * 60 * 60),
    }
    protected_paths = tuple(
        str(item).strip()
        for item in list(
            raw.get("protected_paths")
            or [
                "config/autoresearch_speaker_id.yaml",
                "speaker_id_program.md",
                "src/transcriber/autoresearch_objective.py",
                ".outputs/autoresearch_speaker_id",
            ]
        )
        if str(item).strip()
    )
    splits = {
        "mining_heuristic_eval": [str(item) for item in list(dict(raw.get("splits") or {}).get("mining_heuristic_eval") or ["Session22", "Session61"])],
        "eval_dev": [str(item) for item in list(dict(raw.get("splits") or {}).get("eval_dev") or ["Session22", "Session61", "short_segment_slice"])],
        "eval_final": [str(item) for item in list(dict(raw.get("splits") or {}).get("eval_final") or ["Session58", "Session52", "Session48"])],
    }
    return AutoresearchConfig(
        mode=str(raw.get("mode") or "dev_autopilot_no_rebuild"),
        workspace_root=workspace_root,
        output_root=output_root,
        program_path=_resolve_path(raw.get("program_path") or "speaker_id_program.md", workspace_root=workspace_root),
        bootstrap_dev_baseline_summary=_resolve_path(
            bootstrap.get("dev_champion_baseline_summary")
            or ".outputs/speaker_id_baseline_prod_graph/baseline_summary.json",
            workspace_root=workspace_root,
        ),
        bootstrap_release_baseline_summary=(
            _resolve_path(bootstrap.get("release_champion_baseline_summary"), workspace_root=workspace_root)
            if bootstrap.get("release_champion_baseline_summary")
            else None
        ),
        splits=splits,
        objective_weights=load_objective_weights(dict(raw.get("objective") or {}).get("weights")),
        dev_acceptance=load_dev_acceptance_gate(
            dict(raw.get("objective") or {}).get("dev_acceptance")
        ),
        final_gate=load_final_gate(dict(raw.get("objective") or {}).get("final_gate")),
        families=families,
        budgets_seconds=budgets_seconds,
        scheduler_start_family=str(scheduler.get("start_family") or "classifier_only"),
        rejection_streak_to_escalate=int(scheduler.get("rejection_streak_to_escalate", 5)),
        protected_paths=protected_paths,
        codex_binary=str(codex.get("binary") or "codex"),
        codex_exec_args=tuple(
            str(item)
            for item in list(codex.get("exec_args") or ["--full-auto"])
        ),
        device=str(raw.get("device") or "cuda"),
        results_history_limit=int(raw.get("results_history_limit", DEFAULT_RESULTS_HISTORY)),
        config_path=config_path.expanduser().resolve(),
    )


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _remap_repo_path(path: Path, *, workspace_root: Path, worktree_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return (worktree_root / candidate).resolve()
    try:
        relative = candidate.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return candidate.resolve()
    return (worktree_root / relative).resolve()


def _resolve_run_dependency_path(
    path: Path,
    *,
    workspace_root: Path,
    worktree_root: Path,
) -> Path:
    remapped = _remap_repo_path(path, workspace_root=workspace_root, worktree_root=worktree_root)
    if remapped.exists():
        return remapped
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return remapped


def _enabled_families(config: AutoresearchConfig) -> List[str]:
    enabled: List[str] = []
    for family, settings in sorted(config.families.items(), key=lambda item: int(item[1]["order"])):
        if not bool(settings.get("enabled", True)):
            continue
        if config.mode == "dev_autopilot_no_rebuild" and family == "full_rebuild":
            continue
        enabled.append(family)
    return enabled


def _next_family(current_family: str, config: AutoresearchConfig) -> str:
    families = _enabled_families(config)
    if current_family not in families:
        return families[0]
    index = families.index(current_family)
    return families[min(index + 1, len(families) - 1)]


def _load_state(output_root: Path, *, config: AutoresearchConfig) -> Dict[str, object]:
    state_path = output_root / "state.json"
    if state_path.exists():
        return _read_json(state_path)
    return {
        "mode": config.mode,
        "current_family": config.scheduler_start_family,
        "consecutive_rejections_by_family": {family: 0 for family in FAMILY_ORDER},
        "runs_completed": 0,
        "last_run_id": None,
    }


def _save_state(output_root: Path, state: Mapping[str, object]) -> Path:
    return _json_write(output_root / "state.json", state)


def _champion_from_summary(
    *,
    summary_path: Path,
    source: str,
    run_id: str,
    branch: str,
    config: AutoresearchConfig,
    commit_sha: Optional[str] = None,
) -> ChampionState:
    summary = _read_json(summary_path)
    dev_metrics = _summary_dev_metrics(summary)
    final_metrics = _summary_final_metrics(summary)
    return ChampionState(
        run_id=run_id,
        commit_sha=commit_sha or str(summary.get("git_commit") or current_git_commit(cwd=config.workspace_root) or ""),
        branch=branch,
        baseline_summary_path=summary_path,
        dev_metrics=dev_metrics,
        dev_score=compute_dev_score(dev_metrics, weights=config.objective_weights),
        final_metrics=final_metrics,
        source=source,
    )


def _load_champions(output_root: Path, *, config: AutoresearchConfig) -> Dict[str, Optional[ChampionState]]:
    champions_path = output_root / "champions.json"
    if champions_path.exists():
        raw = _read_json(champions_path)
        return {
            "dev_champion": (
                ChampionState(
                    run_id=str(dict(raw.get("dev_champion") or {})["run_id"]),
                    commit_sha=str(dict(raw.get("dev_champion") or {})["commit_sha"]),
                    branch=str(dict(raw.get("dev_champion") or {})["branch"]),
                    baseline_summary_path=Path(
                        str(dict(raw.get("dev_champion") or {})["baseline_summary_path"])
                    ).expanduser(),
                    dev_metrics=_normalize_metric_map(
                        dict(dict(raw.get("dev_champion") or {}).get("dev_metrics") or {})
                    ),
                    dev_score=float(dict(raw.get("dev_champion") or {})["dev_score"]),
                    final_metrics=_normalize_metric_map(
                        dict(dict(raw.get("dev_champion") or {}).get("final_metrics") or {})
                    ),
                    source=str(dict(raw.get("dev_champion") or {})["source"]),
                )
                if raw.get("dev_champion")
                else None
            ),
            "release_champion": (
                ChampionState(
                    run_id=str(dict(raw.get("release_champion") or {})["run_id"]),
                    commit_sha=str(dict(raw.get("release_champion") or {})["commit_sha"]),
                    branch=str(dict(raw.get("release_champion") or {})["branch"]),
                    baseline_summary_path=Path(
                        str(dict(raw.get("release_champion") or {})["baseline_summary_path"])
                    ).expanduser(),
                    dev_metrics=_normalize_metric_map(
                        dict(dict(raw.get("release_champion") or {}).get("dev_metrics") or {})
                    ),
                    dev_score=float(dict(raw.get("release_champion") or {})["dev_score"]),
                    final_metrics=_normalize_metric_map(
                        dict(dict(raw.get("release_champion") or {}).get("final_metrics") or {})
                    ),
                    source=str(dict(raw.get("release_champion") or {})["source"]),
                )
                if raw.get("release_champion")
                else None
            ),
        }

    dev_champion = _champion_from_summary(
        summary_path=config.bootstrap_dev_baseline_summary,
        source="bootstrap",
        run_id="bootstrap",
        branch="bootstrap",
        config=config,
    )
    release_champion: Optional[ChampionState] = None
    if config.bootstrap_release_baseline_summary is not None:
        release_champion = _champion_from_summary(
            summary_path=config.bootstrap_release_baseline_summary,
            source="bootstrap",
            run_id="bootstrap_release",
            branch="bootstrap_release",
            config=config,
        )
    elif config.mode == "checkpoint_release_mode" and dev_champion.final_metrics:
        release_champion = replace(dev_champion, run_id="bootstrap_release", source="bootstrap")
    champions = {
        "dev_champion": dev_champion,
        "release_champion": release_champion,
    }
    _save_champions(output_root, champions)
    return champions


def _save_champions(
    output_root: Path,
    champions: Mapping[str, Optional[ChampionState]],
) -> Path:
    payload: Dict[str, object] = {}
    for key, champion in champions.items():
        payload[key] = asdict(champion) if champion is not None else None
        if champion is not None:
            payload[key]["baseline_summary_path"] = str(champion.baseline_summary_path)
    return _json_write(output_root / "champions.json", payload)


def _load_results_history(output_root: Path, *, limit: int) -> List[Dict[str, object]]:
    results_path = output_root / "results.jsonl"
    if not results_path.exists():
        return []
    lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]


def _visible_history_entries(output_root: Path, *, limit: int) -> List[Dict[str, object]]:
    entries = []
    for item in _load_results_history(output_root, limit=limit):
        entries.append(
            {
                "run_id": item.get("run_id"),
                "family": item.get("family"),
                "accepted": item.get("accepted"),
                "dev_score": item.get("dev_score"),
                "score_gain": item.get("score_gain"),
                "session61_matched_accuracy": item.get("session61_matched_accuracy"),
                "session22_accuracy": item.get("session22_accuracy"),
                "final_gate_status": item.get("final_gate_status"),
                "reason": item.get("reason"),
            }
        )
    return entries


def _extract_error_themes(summary_path: Path) -> List[str]:
    summary = _read_json(summary_path)
    manifest_path = summary.get("dev_eval_manifest_path") or summary.get("eval_manifest_path")
    if not manifest_path:
        return [
            "Primary objective is Session61 matched_accuracy.",
            "Session22 accuracy is the non-regression guardrail.",
        ]
    manifest = _read_json(Path(str(manifest_path)).expanduser())
    themes: List[str] = []
    for summary_path_value in list(manifest.get("summary_paths") or []):
        summary_path = Path(str(summary_path_value or "")).expanduser()
        if not str(summary_path_value or "").strip():
            continue
        if not summary_path.exists() or not summary_path.is_file():
            continue
        session_summary = _read_json(summary_path)
        graph_pairs = dict(session_summary.get("graph_pair_diagnostics") or {})
        ranked_pairs = sorted(
            (
                (pair_key, dict(pair_metrics or {}).get("post_confusions", 0))
                for pair_key, pair_metrics in graph_pairs.items()
            ),
            key=lambda item: int(item[1]),
            reverse=True,
        )
        for pair_key, count in ranked_pairs[:2]:
            if int(count) <= 0:
                continue
            themes.append(f"{summary_path.parent.name}: {pair_key} still has {int(count)} confusions.")
    if themes:
        return themes[:5]
    return [
        "Primary objective is Session61 matched_accuracy.",
        "Session22 accuracy is the non-regression guardrail.",
    ]


def _render_prompt(
    *,
    run_id: str,
    family: str,
    champion: ChampionState,
    config: AutoresearchConfig,
    visible_history: Sequence[Mapping[str, object]],
    error_themes: Sequence[str],
    prepared_inputs: PreparedRunInputs,
) -> str:
    program_text = config.program_path.read_text(encoding="utf-8").strip()
    visible_metrics = json.dumps(champion.dev_metrics, indent=2)
    recent_history = json.dumps(list(visible_history), indent=2)
    experiment_artifacts = {
        "baseline_summary_path": str(prepared_inputs.run_summary_path),
        "recipe_path": str(prepared_inputs.run_recipe_path),
        "narrow_doe_recipe_path": str(prepared_inputs.run_narrow_doe_recipe_path),
        "artifact_roots": {
            key: prepared_inputs.remapped_narrow_doe_recipe.get(key)
            for key in (
                "base_training_dataset_dir",
                "training_dataset_dir",
                "hard_negative_dataset_dir",
                "baseline_profile_dir",
                "bank_profile_dir",
                "variant_dataset_dirs",
            )
        },
    }
    error_text = "\n".join(f"- {item}" for item in error_themes)
    return (
        f"{program_text}\n\n"
        f"## Run Context\n"
        f"- Run id: `{run_id}`\n"
        f"- Active family: `{family}`\n"
        f"- Workspace root: `{config.workspace_root}`\n"
        f"- Protected paths: `{', '.join(config.protected_paths)}`\n"
        f"- Current dev champion commit: `{champion.commit_sha}`\n"
        f"- Current dev champion score: `{champion.dev_score:.6f}`\n\n"
        f"## Visible Dev Metrics\n"
        f"```json\n{visible_metrics}\n```\n\n"
        f"## Dominant Error Themes\n"
        f"{error_text}\n\n"
        f"## Recent Visible Run History\n"
        f"```json\n{recent_history}\n```\n\n"
        f"## Fixed Experiment Inputs\n"
        f"```json\n{json.dumps(experiment_artifacts, indent=2)}\n```\n\n"
        f"## Deliverable\n"
        f"Make one coherent repo change that should improve the visible dev objective for `{family}`.\n"
        f"Prefer small diffs. Do not modify protected paths. Do not change split semantics. "
        f"Do not surface or seek final-holdout metrics. The outer harness will run the experiment and score it.\n"
    )


def _prepare_run_inputs(
    *,
    champion: ChampionState,
    config: AutoresearchConfig,
    worktree_root: Path,
    run_dir: Path,
) -> PreparedRunInputs:
    parent_summary = _read_json(champion.baseline_summary_path)
    canonical_recipe_path = Path(str(parent_summary["recipe_path"])).expanduser()
    source_recipe_path = _resolve_run_dependency_path(
        canonical_recipe_path,
        workspace_root=config.workspace_root,
        worktree_root=worktree_root,
    )
    remapped_recipe = _load_yaml_or_json(str(source_recipe_path)) or {}
    remapped_recipe = {
        **dict(remapped_recipe),
            "base_config": str(
                _resolve_run_dependency_path(
                    Path(str(remapped_recipe.get("base_config") or "config/transcriber.yaml")),
                    workspace_root=config.workspace_root,
                    worktree_root=worktree_root,
                )
            ),
            "speaker_mapping": str(
                _resolve_run_dependency_path(
                    Path(str(remapped_recipe.get("speaker_mapping") or "config/speaker_mapping.yaml")),
                    workspace_root=config.workspace_root,
                    worktree_root=worktree_root,
            )
        ),
        "output_root": str(run_dir / "experiment_outputs"),
        "metrics_log_path": str(run_dir / "experiment_outputs" / "stage_metrics.jsonl"),
        "resume": False,
    }
    run_recipe_path = _json_write(run_dir / "inputs" / "recipe.json", remapped_recipe)

    parent_narrow_doe = _read_json(Path(str(parent_summary["narrow_doe_recipe_path"])).expanduser())
    current_winner = dict(remapped_recipe.get("current_winner") or {})
    speaker_bank_overrides = dict(remapped_recipe.get("speaker_bank_overrides") or {})
    remapped_narrow_doe = {
        **dict(parent_narrow_doe),
        "classifier": {
            **dict(parent_narrow_doe.get("classifier") or {}),
            **current_winner,
        },
        "speaker_bank_overrides": {
            **dict(parent_narrow_doe.get("speaker_bank_overrides") or {}),
            **speaker_bank_overrides,
        },
    }
    run_narrow_doe_recipe_path = _json_write(
        run_dir / "inputs" / "narrow_doe_recipe.json",
        remapped_narrow_doe,
    )
    run_summary = _sanitize_visible_summary(parent_summary)
    run_summary.update(
        {
            "recipe_path": str(run_recipe_path),
            "narrow_doe_recipe_path": str(run_narrow_doe_recipe_path),
        }
    )
    run_summary_path = _json_write(run_dir / "inputs" / "baseline_summary.json", run_summary)
    return PreparedRunInputs(
        parent_summary=parent_summary,
        run_summary_path=run_summary_path,
        run_recipe_path=run_recipe_path,
        run_narrow_doe_recipe_path=run_narrow_doe_recipe_path,
        canonical_recipe_path=canonical_recipe_path,
        remapped_recipe=remapped_recipe,
        remapped_narrow_doe_recipe=remapped_narrow_doe,
    )


def _evaluate_final_suite(
    *,
    candidate_summary_path: Path,
    device: str,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    context = _load_downstream_context(baseline_summary_path=candidate_summary_path)
    if not context.eval_final_specs:
        return {}
    final_context = replace(
        context,
        eval_specs=list(context.eval_final_specs),
        short_slice_session="",
        short_slice_window={},
    )
    session_results, _ = _evaluate_dev_suite(
        context=final_context,
        experiment_dir=output_dir,
        profile_name=Path(str(context.baseline_profile_dir)).name,
        classifier_config=dict(context.classifier),
        speaker_bank_overrides=context.speaker_bank_overrides,
        device=device,
    )
    return _normalize_metric_map(session_results)


def _git(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def _create_worktree(
    *,
    workspace_root: Path,
    branch_name: str,
    commit_sha: str,
    worktree_root: Path,
) -> None:
    worktree_root.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", "-b", branch_name, str(worktree_root), commit_sha], cwd=workspace_root)


def _remove_worktree(
    *,
    workspace_root: Path,
    branch_name: str,
    worktree_root: Path,
    keep_branch: bool,
) -> None:
    if worktree_root.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_root)],
            cwd=str(workspace_root),
            check=False,
            text=True,
            capture_output=True,
        )
    if not keep_branch:
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(workspace_root),
            check=False,
            text=True,
            capture_output=True,
        )


def _collect_touched_files(worktree_root: Path) -> List[str]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "--relative", "HEAD"],
        cwd=str(worktree_root),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(worktree_root),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    touched = sorted({item.strip() for item in tracked + untracked if item.strip()})
    return touched


def _write_patch_diff(worktree_root: Path, output_path: Path) -> Path:
    subprocess.run(["git", "add", "-N", "."], cwd=str(worktree_root), check=True, text=True)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=str(worktree_root),
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diff, encoding="utf-8")
    return output_path


def _path_is_protected(path_text: str, protected_paths: Sequence[str]) -> bool:
    normalized = Path(path_text).as_posix().lstrip("./")
    for protected in protected_paths:
        protected_norm = Path(protected).as_posix().lstrip("./")
        if normalized == protected_norm or normalized.startswith(f"{protected_norm}/"):
            return True
    return False


def _reject_if_protected_paths_touched(
    *,
    touched_files: Sequence[str],
    config: AutoresearchConfig,
) -> Optional[str]:
    protected_hits = [
        path_text
        for path_text in touched_files
        if _path_is_protected(path_text, config.protected_paths)
    ]
    if protected_hits:
        return "protected_paths_touched: " + ", ".join(sorted(protected_hits))
    return None


def _record_results_tsv(output_root: Path, row: Mapping[str, object]) -> Path:
    path = output_root / "results.tsv"
    headers = [
        "run_id",
        "parent_commit",
        "tier",
        "dev_score",
        "session22_accuracy",
        "session22_matched_accuracy",
        "session61_accuracy",
        "session61_matched_accuracy",
        "short_slice_matched_accuracy",
        "accepted",
        "final_gate_status",
        "runtime_seconds",
        "commit_sha",
    ]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\t".join(headers) + "\n", encoding="utf-8")
    values = [str(row.get(header, "")) for header in headers]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(values) + "\n")
    return path


class CodexExecMutationBackend:
    def run(
        self,
        *,
        prompt_text: str,
        worktree_root: Path,
        log_path: Path,
        output_last_message_path: Path,
        config: AutoresearchConfig,
        run_budget_seconds: int,
    ) -> Dict[str, object]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            config.codex_binary,
            "exec",
            *config.codex_exec_args,
            "-C",
            str(worktree_root),
            "--output-last-message",
            str(output_last_message_path),
            "-",
        ]
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + " ".join(command) + "\n")
            try:
                completed = subprocess.run(
                    command,
                    input=prompt_text,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    timeout=max(run_budget_seconds, 1),
                    check=False,
                )
                return {
                    "exit_code": int(completed.returncode),
                    "timed_out": False,
                    "elapsed_seconds": time.monotonic() - started,
                }
            except subprocess.TimeoutExpired:
                return {
                    "exit_code": None,
                    "timed_out": True,
                    "elapsed_seconds": time.monotonic() - started,
                }


class AutoresearchExperimentExecutor:
    def _result_metrics(self, result: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
        return _normalize_metric_map(dict(result.get("session_results") or {}))

    def _build_candidate_summary(
        self,
        *,
        family: str,
        outcome_result: Mapping[str, object],
        prepared_inputs: PreparedRunInputs,
        run_dir: Path,
        rebuild_summary_path: Optional[Path] = None,
    ) -> Path:
        if rebuild_summary_path is not None:
            summary = _sanitize_visible_summary(_read_json(rebuild_summary_path))
            summary["recipe_path"] = str(prepared_inputs.canonical_recipe_path)
            return _json_write(run_dir / "candidate_baseline_summary.json", summary)

        summary = _sanitize_visible_summary(prepared_inputs.parent_summary)
        summary["recipe_path"] = str(prepared_inputs.canonical_recipe_path)
        candidate_metrics = dict(outcome_result.get("session_results") or {})
        summary["dev_eval"] = candidate_metrics
        summary["canonical_eval"] = candidate_metrics
        narrow_doe = dict(prepared_inputs.remapped_narrow_doe_recipe)
        result_classifier = dict(outcome_result.get("classifier") or {})
        if result_classifier:
            narrow_doe["classifier"] = result_classifier
        profile_dir = outcome_result.get("profile_dir")
        if profile_dir:
            narrow_doe["baseline_profile_dir"] = str(profile_dir)

        if family == "hard_negative_refresh":
            artifact_paths = dict(outcome_result.get("artifact_paths") or {})
            hard_negative_dataset_dir = Path(
                str(artifact_paths.get("hard_negative_dataset_dir") or "")
            ).expanduser()
            final_training_dataset_dir = Path(
                str(artifact_paths.get("final_training_dataset_dir") or "")
            ).expanduser()
            if hard_negative_dataset_dir.exists():
                narrow_doe["hard_negative_dataset_dir"] = str(hard_negative_dataset_dir)
            if final_training_dataset_dir.exists():
                narrow_doe["training_dataset_dir"] = str(final_training_dataset_dir)
        narrow_doe_path = _json_write(run_dir / "candidate_narrow_doe_recipe.json", narrow_doe)
        summary["narrow_doe_recipe_path"] = str(narrow_doe_path)
        candidate_summary_path = _json_write(run_dir / "candidate_baseline_summary.json", summary)
        return candidate_summary_path

    def run(
        self,
        *,
        family: str,
        champion: ChampionState,
        config: AutoresearchConfig,
        worktree_root: Path,
        run_dir: Path,
        device: str,
        run_budget_seconds: int,
    ) -> ExperimentOutcome:
        del run_budget_seconds
        prepared_inputs = _prepare_run_inputs(
            champion=champion,
            config=config,
            worktree_root=worktree_root,
            run_dir=run_dir,
        )

        if family == "eval_only":
            context = _load_downstream_context(baseline_summary_path=prepared_inputs.run_summary_path)
            experiment_dir = run_dir / "eval_only"
            session_results, config_path = _evaluate_dev_suite(
                context=context,
                experiment_dir=experiment_dir,
                profile_name=Path(str(context.baseline_profile_dir)).name,
                classifier_config=dict(context.classifier),
                speaker_bank_overrides=context.speaker_bank_overrides,
                device=device,
            )
            result = {
                "name": "autopilot_eval_only",
                "mode": "eval_only",
                "classifier": dict(context.classifier),
                "profile_dir": str(context.baseline_profile_dir),
                "artifact_paths": {
                    "config_path": str(config_path),
                    "eval_root": str(experiment_dir / "eval"),
                },
                "session_results": session_results,
            }
        elif family == "classifier_only":
            recipe = dict(prepared_inputs.remapped_recipe)
            current_winner = dict(recipe.get("current_winner") or {})
            spec_path = _json_write(
                run_dir / "classifier_only_spec.json",
                {
                    "dev_only": True,
                    "acceptance": {
                        "session61_matched_accuracy_gain": 999.0,
                        "session22_accuracy_regression_max": 0.0,
                    },
                    "phase_a": {
                        "family_sweep": [
                            {"name": "autopilot_classifier_only", "classifier": current_winner}
                        ],
                        "calibration": {"top_n": 0, "thresholds": [], "classifier_min_margins": []},
                    },
                    "phase_b": {"candidate_variants": ["mixed_raw"], "experiments": []},
                },
            )
            report = run_downstream_retrain_doe(
                baseline_summary_path=prepared_inputs.run_summary_path,
                spec_path=spec_path,
                output_dir=run_dir / "classifier_only",
                device=device,
            )
            result = dict(report["phase_a"]["results"][0])
        elif family == "hard_negative_refresh":
            recipe = dict(prepared_inputs.remapped_recipe)
            hard_negative = {
                "seed_confusion_pairs": list(
                    recipe.get("seed_confusion_pairs") or [["Cyrus Schwert", "Cletus Cobbington"]]
                ),
                "top_confusion_pairs": int(recipe.get("top_confusion_pairs", 0)),
                "hard_negative_max_margin": float(recipe.get("hard_negative_max_margin", 0.12)),
                "hard_negative_min_dominant_share": float(
                    recipe.get("hard_negative_min_dominant_share", 0.55)
                ),
                "hard_negative_per_pair_cap": int(recipe.get("hard_negative_per_pair_cap", 60)),
                "pair_caps": list(recipe.get("hard_negative_pair_caps") or []),
                "hard_negative_per_speaker_cap": recipe.get("hard_negative_per_speaker_cap"),
                "hard_negative_max_fraction": float(recipe.get("hard_negative_max_fraction", 0.20)),
                "hard_negative_style_profile_name": str(
                    recipe.get("hard_negative_style_profile_name") or "session61_like"
                ),
                "hard_negative_style_score_threshold": float(
                    recipe.get("hard_negative_style_score_threshold", 0.45)
                ),
                "hard_negative_min_style_samples_per_pair": int(
                    recipe.get("hard_negative_min_style_samples_per_pair", 10)
                ),
            }
            spec_path = _json_write(
                run_dir / "hard_negative_refresh_spec.json",
                {
                    "dev_only": True,
                    "acceptance": {
                        "session61_matched_accuracy_gain": 999.0,
                        "session22_accuracy_regression_max": 0.0,
                    },
                    "phase_a": {
                        "family_sweep": [],
                        "calibration": {"top_n": 0, "thresholds": [], "classifier_min_margins": []},
                    },
                    "phase_b": {
                        "candidate_variants": list(
                            recipe.get("hard_negative_candidate_variants") or ["mixed_raw"]
                        ),
                        "experiments": [
                            {
                                "name": "autopilot_hard_negative_refresh",
                                "hard_negative": hard_negative,
                            }
                        ],
                    },
                },
            )
            report = run_downstream_retrain_doe(
                baseline_summary_path=prepared_inputs.run_summary_path,
                spec_path=spec_path,
                output_dir=run_dir / "hard_negative_refresh",
                device=device,
            )
            result = dict(report["phase_b"]["results"][0])
        elif family == "full_rebuild":
            summary = prepare_baseline(
                recipe_path=prepared_inputs.run_recipe_path,
                output_root=run_dir / "full_rebuild",
                allow_legacy_reuse=False,
                resume=False,
            )
            rebuilt_narrow_doe = _read_json(Path(str(summary["narrow_doe_recipe_path"])).expanduser())
            result = {
                "name": "autopilot_full_rebuild",
                "mode": "full_rebuild",
                "classifier": dict(
                    rebuilt_narrow_doe.get("classifier")
                    or prepared_inputs.remapped_narrow_doe_recipe.get("classifier")
                    or {}
                ),
                "profile_dir": str(rebuilt_narrow_doe.get("baseline_profile_dir") or ""),
                "artifact_paths": {
                    "baseline_summary_path": str(
                        run_dir / "full_rebuild" / "baseline_summary.json"
                    )
                },
                "session_results": _summary_dev_metrics(summary),
                "final_metrics": _summary_final_metrics(summary),
            }
        else:
            raise RuntimeError(f"Unsupported experiment family: {family}")

        if family != "full_rebuild":
            candidate_summary_path = self._build_candidate_summary(
                family=family,
                outcome_result=result,
                prepared_inputs=prepared_inputs,
                run_dir=run_dir,
            )
            final_metrics = {}
        else:
            candidate_summary_path = self._build_candidate_summary(
                family=family,
                outcome_result=result,
                prepared_inputs=prepared_inputs,
                run_dir=run_dir,
                rebuild_summary_path=run_dir / "full_rebuild" / "baseline_summary.json",
            )
            final_metrics = _summary_final_metrics(_read_json(candidate_summary_path))

        return ExperimentOutcome(
            family=family,
            result=result,
            candidate_summary_path=candidate_summary_path,
            dev_metrics=self._result_metrics(result),
            final_metrics=final_metrics,
            artifact_paths=dict(result.get("artifact_paths") or {}),
        )


def _commit_worktree(
    *,
    worktree_root: Path,
    run_id: str,
) -> str:
    subprocess.run(["git", "add", "-A"], cwd=str(worktree_root), check=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@local",
            "commit",
            "-m",
            f"Autoresearch speaker ID run {run_id}",
        ],
        cwd=str(worktree_root),
        check=True,
        text=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(worktree_root),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _session_metric(metrics: Mapping[str, Mapping[str, float]], session_name: str, key: str) -> float:
    return float(dict(metrics.get(session_name) or {}).get(key) or 0.0)


def _write_result_artifacts(
    *,
    output_root: Path,
    run_entry: Mapping[str, object],
) -> None:
    _jsonl_append(output_root / "results.jsonl", run_entry)
    _record_results_tsv(output_root, run_entry)


def _run_autoresearch_once(
    *,
    config: AutoresearchConfig,
    state: Dict[str, object],
    champions: Dict[str, Optional[ChampionState]],
    mutation_backend: MutationBackend,
    experiment_executor: ExperimentExecutor,
    device: str,
) -> Dict[str, object]:
    dev_champion = champions.get("dev_champion")
    if dev_champion is None:
        raise RuntimeError("Autoresearch requires a dev champion to start")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    family = str(state.get("current_family") or config.scheduler_start_family)
    if family not in _enabled_families(config):
        family = _enabled_families(config)[0]
    branch_name = f"codex/autoresearch/{run_id}"
    run_dir = config.output_root / "runs" / run_id
    worktree_root = config.output_root / "worktrees" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _create_worktree(
        workspace_root=config.workspace_root,
        branch_name=branch_name,
        commit_sha=dev_champion.commit_sha,
        worktree_root=worktree_root,
    )

    keep_branch = False
    started = time.monotonic()
    try:
        run_budget_seconds = int(config.budgets_seconds[family])
        prepared_inputs = _prepare_run_inputs(
            champion=dev_champion,
            config=config,
            worktree_root=worktree_root,
            run_dir=run_dir,
        )
        prompt_text = _render_prompt(
            run_id=run_id,
            family=family,
            champion=dev_champion,
            config=config,
            visible_history=_visible_history_entries(
                config.output_root,
                limit=config.results_history_limit,
            ),
            error_themes=_extract_error_themes(dev_champion.baseline_summary_path),
            prepared_inputs=prepared_inputs,
        )
        prompt_path = run_dir / "prompt.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        _json_write(
            run_dir / "experiment_manifest.json",
            {
                "run_id": run_id,
                "family": family,
                "branch_name": branch_name,
                "parent_champion_commit": dev_champion.commit_sha,
                "run_budget_seconds": run_budget_seconds,
                "baseline_summary_path": str(prepared_inputs.run_summary_path),
                "recipe_path": str(prepared_inputs.run_recipe_path),
                "narrow_doe_recipe_path": str(prepared_inputs.run_narrow_doe_recipe_path),
            },
        )
        mutation_result = mutation_backend.run(
            prompt_text=prompt_text,
            worktree_root=worktree_root,
            log_path=run_dir / "codex.log",
            output_last_message_path=run_dir / "codex_last_message.txt",
            config=config,
            run_budget_seconds=run_budget_seconds,
        )
        touched_files = _collect_touched_files(worktree_root)
        (run_dir / "touched_files.txt").write_text("\n".join(touched_files) + ("\n" if touched_files else ""), encoding="utf-8")
        _write_patch_diff(worktree_root, run_dir / "patch.diff")

        rejection_reason = None
        if mutation_result.get("timed_out"):
            rejection_reason = "mutation_timed_out"
        elif mutation_result.get("exit_code") not in {0}:
            rejection_reason = f"mutation_failed:{mutation_result.get('exit_code')}"
        elif not touched_files:
            rejection_reason = "no_changes"
        else:
            rejection_reason = _reject_if_protected_paths_touched(
                touched_files=touched_files,
                config=config,
            )

        outcome: Optional[ExperimentOutcome] = None
        dev_eval_result: Dict[str, object] = {
            "accepted": False,
            "candidate_score": dev_champion.dev_score,
            "champion_score": dev_champion.dev_score,
            "score_gain": 0.0,
            "session61_matched_accuracy_gain": 0.0,
            "session22_accuracy_delta": 0.0,
        }
        final_gate_result: Optional[Dict[str, object]] = None
        final_gate_status = "not_run"
        commit_sha = ""

        if rejection_reason is None:
            outcome = experiment_executor.run(
                family=family,
                champion=dev_champion,
                config=config,
                worktree_root=worktree_root,
                run_dir=run_dir,
                device=device,
                run_budget_seconds=run_budget_seconds,
            )
            _json_write(run_dir / "dev_metrics.json", outcome.dev_metrics)
            dev_eval_result = evaluate_dev_candidate(
                outcome.dev_metrics,
                champion_metrics=dev_champion.dev_metrics,
                weights=config.objective_weights,
                gate=config.dev_acceptance,
            )
            if dev_eval_result["accepted"]:
                commit_sha = _commit_worktree(worktree_root=worktree_root, run_id=run_id)
                keep_branch = True
                new_dev_champion = ChampionState(
                    run_id=run_id,
                    commit_sha=commit_sha,
                    branch=branch_name,
                    baseline_summary_path=outcome.candidate_summary_path,
                    dev_metrics=outcome.dev_metrics,
                    dev_score=float(dev_eval_result["candidate_score"]),
                    final_metrics=outcome.final_metrics,
                    source=family,
                )
                champions["dev_champion"] = new_dev_champion
                if config.mode == "checkpoint_release_mode":
                    release_champion = champions.get("release_champion")
                    if release_champion is None:
                        raise RuntimeError(
                            "checkpoint_release_mode requires a release_champion bootstrap with final metrics"
                        )
                    candidate_final_metrics = outcome.final_metrics
                    if not candidate_final_metrics:
                        candidate_final_metrics = _evaluate_final_suite(
                            candidate_summary_path=outcome.candidate_summary_path,
                            device=device,
                            output_dir=run_dir / "final_eval",
                        )
                    final_gate_result = evaluate_final_gate(
                        candidate_final_metrics,
                        champion_metrics=release_champion.final_metrics,
                        gate=config.final_gate,
                    )
                    _json_write(
                        run_dir / "final_gate.json",
                        {
                            **final_gate_result,
                            "status": "passed" if final_gate_result["accepted"] else "failed",
                        },
                    )
                    final_gate_status = "passed" if final_gate_result["accepted"] else "failed"
                    if final_gate_result["accepted"]:
                        champions["release_champion"] = replace(
                            new_dev_champion,
                            source=f"{family}:release",
                            final_metrics=candidate_final_metrics,
                        )
                else:
                    final_gate_status = "not_run"
                state["current_family"] = "eval_only"
                streaks = dict(state.get("consecutive_rejections_by_family") or {})
                streaks = {family_name: 0 for family_name in FAMILY_ORDER}
                state["consecutive_rejections_by_family"] = streaks
            else:
                rejection_reason = "dev_acceptance_failed"
        if rejection_reason is not None:
            streaks = dict(state.get("consecutive_rejections_by_family") or {})
            streaks[family] = int(streaks.get(family, 0)) + 1
            if streaks[family] >= config.rejection_streak_to_escalate:
                state["current_family"] = _next_family(family, config)
                streaks[family] = 0
            state["consecutive_rejections_by_family"] = streaks

        elapsed = time.monotonic() - started
        result_payload = {
            "run_id": run_id,
            "family": family,
            "tier": family,
            "parent_commit": dev_champion.commit_sha,
            "accepted": bool(dev_eval_result["accepted"]),
            "reason": rejection_reason,
            "dev_score": float(dev_eval_result["candidate_score"]),
            "score_gain": float(dev_eval_result["score_gain"]),
            "session22_accuracy": _session_metric(
                outcome.dev_metrics if outcome else dev_champion.dev_metrics, "Session22", "mean_accuracy"
            ),
            "session22_matched_accuracy": _session_metric(
                outcome.dev_metrics if outcome else dev_champion.dev_metrics,
                "Session22",
                "mean_matched_accuracy",
            ),
            "session61_accuracy": _session_metric(
                outcome.dev_metrics if outcome else dev_champion.dev_metrics, "Session61", "mean_accuracy"
            ),
            "session61_matched_accuracy": _session_metric(
                outcome.dev_metrics if outcome else dev_champion.dev_metrics,
                "Session61",
                "mean_matched_accuracy",
            ),
            "short_slice_matched_accuracy": _session_metric(
                outcome.dev_metrics if outcome else dev_champion.dev_metrics,
                "short_segment_slice",
                "mean_matched_accuracy",
            ),
            "final_gate_status": final_gate_status,
            "runtime_seconds": round(elapsed, 3),
            "commit_sha": commit_sha,
        }
        _json_write(run_dir / "result.json", result_payload)
        _write_result_artifacts(output_root=config.output_root, run_entry=result_payload)
        state["runs_completed"] = int(state.get("runs_completed") or 0) + 1
        state["last_run_id"] = run_id
        _save_state(config.output_root, state)
        _save_champions(config.output_root, champions)
        return result_payload
    finally:
        _remove_worktree(
            workspace_root=config.workspace_root,
            branch_name=branch_name,
            worktree_root=worktree_root,
            keep_branch=keep_branch,
        )


def run_autoresearch_loop(
    *,
    config_path: Path,
    once: bool = False,
    max_runs: Optional[int] = None,
    resume: bool = False,
    mode: Optional[str] = None,
    device: Optional[str] = None,
    mutation_backend: Optional[MutationBackend] = None,
    experiment_executor: Optional[ExperimentExecutor] = None,
) -> Dict[str, object]:
    config = _load_autoresearch_config(config_path.expanduser().resolve())
    if mode:
        config = replace(config, mode=str(mode))
    if device:
        config = replace(config, device=str(device))
    config.output_root.mkdir(parents=True, exist_ok=True)
    state = _load_state(config.output_root, config=config) if resume else _load_state(config.output_root, config=config)
    champions = _load_champions(config.output_root, config=config)
    backend = mutation_backend or CodexExecMutationBackend()
    executor = experiment_executor or AutoresearchExperimentExecutor()
    run_limit = 1 if once else (max_runs if max_runs is not None else -1)
    results: List[Dict[str, object]] = []
    while run_limit == -1 or len(results) < run_limit:
        result = _run_autoresearch_once(
            config=config,
            state=state,
            champions=champions,
            mutation_backend=backend,
            experiment_executor=executor,
            device=str(device or config.device),
        )
        results.append(result)
        if once:
            break
    return {
        "mode": config.mode,
        "runs": results,
        "output_root": str(config.output_root),
        "state_path": str(config.output_root / "state.json"),
        "champions_path": str(config.output_root / "champions.json"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an autoresearch-style autonomous optimization loop for speaker ID."
    )
    parser.add_argument("--config", type=Path, default=Path("config/autoresearch_speaker_id.yaml"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = run_autoresearch_loop(
        config_path=args.config,
        once=bool(args.once),
        max_runs=args.max_runs,
        resume=bool(args.resume),
        mode=args.mode,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


__all__ = [
    "AutoresearchConfig",
    "ChampionState",
    "CodexExecMutationBackend",
    "run_autoresearch_loop",
    "_load_autoresearch_config",
    "_next_family",
    "_reject_if_protected_paths_touched",
    "_render_prompt",
]
