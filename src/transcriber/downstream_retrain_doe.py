from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .cli import _load_yaml_or_json
from .hard_negatives import build_hard_negative_dataset
from .multitrack_eval import evaluate_multitrack_session
from .prep_artifacts import append_dataset
from .segment_classifier import (
    load_classifier_dataset,
    save_classifier_dataset,
    train_segment_classifier_from_dataset,
)


DEFAULT_ACCEPTANCE = {
    "session61_matched_accuracy_gain": 0.01,
    "session22_accuracy_regression_max": 0.01,
}


@dataclass(frozen=True)
class EvalSpec:
    name: str
    session_zip: Path
    transcript: Path


@dataclass(frozen=True)
class DownstreamContext:
    baseline_summary_path: Path
    recipe_path: Path
    recipe: Mapping[str, object]
    narrow_doe_recipe: Mapping[str, object]
    base_config: Mapping[str, object]
    output_root: Path
    hf_cache_root: Path
    speaker_mapping_path: Path
    diarization_model: str
    local_files_only: bool
    eval_specs: List[EvalSpec]
    eval_final_specs: List[EvalSpec]
    mining_eval_specs: List[EvalSpec]
    short_slice_session: str
    short_slice_window: Mapping[str, object]
    prepared_eval_root: Path
    bank_profile_dir: Path
    baseline_profile_dir: Path
    base_training_dataset_dir: Path
    final_training_dataset_dir: Path
    hard_negative_dataset_dir: Optional[Path]
    variant_dataset_dirs: Dict[str, Path]
    classifier: Dict[str, object]
    speaker_bank_overrides: Dict[str, object]
    baseline_metrics: Dict[str, Dict[str, float]]


def _json_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


def _jsonl_write(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record)) + "\n")
    return path


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _parse_eval_specs(raw_specs: Sequence[Mapping[str, object]]) -> List[EvalSpec]:
    specs: List[EvalSpec] = []
    for raw in raw_specs:
        specs.append(
            EvalSpec(
                name=str(raw["name"]),
                session_zip=Path(str(raw["session_zip"])).expanduser().resolve(),
                transcript=Path(str(raw["transcript"])).expanduser().resolve(),
            )
        )
    return specs


def _normalize_session_name(value: object) -> str:
    return str(value or "").strip().lower()


def _merge_nested_dict(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> Dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_nested_dict(
                dict(merged.get(key) or {}),
                dict(value),
            )
        else:
            merged[key] = value
    return merged


def _parse_pair_caps(raw_items: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], int]:
    parsed: Dict[Tuple[str, str], int] = {}
    for item in raw_items:
        pair = list(item.get("pair") or [])
        if len(pair) != 2:
            continue
        left, right = sorted([str(pair[0]).strip(), str(pair[1]).strip()])
        if not left or not right:
            continue
        parsed[(left, right)] = int(item["value"])
    return parsed


def _pair_caps_to_recipe_items(pair_caps: Mapping[Tuple[str, str], int]) -> List[Dict[str, object]]:
    return [
        {"pair": [left, right], "value": int(value)}
        for (left, right), value in sorted(pair_caps.items())
    ]


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower() or "experiment"


def _profile_name(prefix: str, phase: str, experiment_name: str, output_dir: Path) -> str:
    digest = sha1(f"{output_dir.resolve()}::{phase}::{experiment_name}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{_slugify(phase)}_{_slugify(experiment_name)}_{digest}"


def _write_training_summary(profile_dir: Path, summary: Mapping[str, object]) -> Path:
    return _json_write(profile_dir / "classifier_training_summary.json", summary)


def _write_eval_config(
    output_path: Path,
    *,
    base_config: Mapping[str, object],
    hf_cache_root: Path,
    profile_name: str,
    diarization_model: str,
    speaker_mapping_path: Path,
    classifier_min_margin: float,
    threshold: float,
    match_aggregation: str,
    min_segments_per_label: int,
    speaker_bank_overrides: Mapping[str, object],
) -> Path:
    payload = dict(base_config)
    payload["hf_cache_root"] = str(hf_cache_root)
    payload["speaker_bank_root"] = str(hf_cache_root)
    payload["speaker_mapping"] = str(speaker_mapping_path)
    payload["diarization_model"] = diarization_model

    speaker_bank_cfg = dict(payload.get("speaker_bank") or {})
    speaker_bank_cfg["enabled"] = True
    speaker_bank_cfg["path"] = profile_name
    speaker_bank_cfg["threshold"] = float(threshold)
    speaker_bank_cfg["match_per_segment"] = True
    speaker_bank_cfg["match_aggregation"] = str(match_aggregation or "mean")
    speaker_bank_cfg["min_segments_per_label"] = int(min_segments_per_label)

    classifier_cfg = dict(speaker_bank_cfg.get("classifier") or {})
    classifier_cfg["min_confidence"] = 0.0
    classifier_cfg["min_margin"] = float(classifier_min_margin)
    speaker_bank_cfg["classifier"] = classifier_cfg

    for key, value in dict(speaker_bank_overrides or {}).items():
        if key == "classifier":
            merged_classifier = dict(speaker_bank_cfg.get("classifier") or {})
            merged_classifier.update(dict(value or {}))
            speaker_bank_cfg["classifier"] = merged_classifier
            continue
        if key in {"repair", "session_graph"} and isinstance(value, Mapping):
            merged_nested = dict(speaker_bank_cfg.get(key) or {})
            merged_nested.update(dict(value))
            speaker_bank_cfg[key] = merged_nested
            continue
        speaker_bank_cfg[key] = value

    payload["speaker_bank"] = speaker_bank_cfg
    return _json_write(output_path, payload)


def _aggregate_eval_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    results = list(summary.get("results") or [])
    return {
        "mean_accuracy": _mean(
            float(item.get("metrics", {}).get("accuracy") or 0.0) for item in results
        ),
        "mean_coverage": _mean(
            float(item.get("metrics", {}).get("coverage") or 0.0) for item in results
        ),
        "mean_matched_accuracy": _mean(
            float(item.get("metrics", {}).get("matched_accuracy") or 0.0) for item in results
        ),
        "mean_accuracy_pre_graph": (
            _mean(
                float(item.get("metrics_pre_graph", {}).get("accuracy") or 0.0)
                for item in results
                if item.get("metrics_pre_graph") is not None
            )
            if any(item.get("metrics_pre_graph") is not None for item in results)
            else None
        ),
        "mean_matched_accuracy_pre_graph": (
            _mean(
                float(item.get("metrics_pre_graph", {}).get("matched_accuracy") or 0.0)
                for item in results
                if item.get("metrics_pre_graph") is not None
            )
            if any(item.get("metrics_pre_graph") is not None for item in results)
            else None
        ),
    }


def _baseline_metrics_from_summary(summary: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    raw = dict(summary.get("dev_eval") or summary.get("canonical_eval") or {})
    parsed: Dict[str, Dict[str, float]] = {}
    for session_name, metrics in raw.items():
        parsed[str(session_name)] = {
            key: float(value)
            for key, value in dict(metrics or {}).items()
            if isinstance(value, (int, float))
        }
    return parsed


def _load_experiment_spec(spec_path: Path) -> Dict[str, object]:
    raw = _load_yaml_or_json(str(spec_path.expanduser().resolve())) or {}
    acceptance = {
        **DEFAULT_ACCEPTANCE,
        **dict(raw.get("acceptance") or {}),
    }
    phase_a = dict(raw.get("phase_a") or {})
    phase_b = dict(raw.get("phase_b") or {})
    return {
        "dev_only": bool(raw.get("dev_only", True)),
        "acceptance": {
            "session61_matched_accuracy_gain": float(
                acceptance["session61_matched_accuracy_gain"]
            ),
            "session22_accuracy_regression_max": float(
                acceptance["session22_accuracy_regression_max"]
            ),
        },
        "phase_a": {
            "family_sweep": list(phase_a.get("family_sweep") or []),
            "calibration": {
                "top_n": int(dict(phase_a.get("calibration") or {}).get("top_n", 2)),
                "thresholds": [
                    float(value)
                    for value in list(dict(phase_a.get("calibration") or {}).get("thresholds") or [0.34, 0.36, 0.38])
                ],
                "classifier_min_margins": [
                    float(value)
                    for value in list(
                        dict(phase_a.get("calibration") or {}).get("classifier_min_margins")
                        or [0.04, 0.06, 0.08]
                    )
                ],
            },
        },
        "phase_b": {
            "candidate_variants": list(phase_b.get("candidate_variants") or ["mixed_raw"]),
            "experiments": list(phase_b.get("experiments") or []),
        },
    }


def _load_context(
    *,
    baseline_summary_path: Path,
) -> DownstreamContext:
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    recipe_path = Path(str(baseline_summary["recipe_path"])).expanduser().resolve()
    recipe = _load_yaml_or_json(str(recipe_path)) or {}
    narrow_doe_recipe = json.loads(
        Path(str(baseline_summary["narrow_doe_recipe_path"])).expanduser().read_text(
            encoding="utf-8"
        )
    )
    base_config_path = Path(str(recipe["base_config"])).expanduser().resolve()
    base_config = _load_yaml_or_json(str(base_config_path)) or {}
    speaker_mapping_path = Path(str(recipe["speaker_mapping"])).expanduser().resolve()
    output_root = Path(str(baseline_summary["output_root"])).expanduser().resolve()
    eval_manifest_path = Path(
        str(baseline_summary.get("dev_eval_manifest_path") or baseline_summary["eval_manifest_path"])
    ).expanduser().resolve()
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    canonical_suite = dict(eval_manifest.get("canonical_suite") or {})
    short_slice = dict(canonical_suite.get("short_segment_slice") or {})

    eval_specs = _parse_eval_specs(recipe.get("eval_dev") or recipe.get("eval") or [])
    eval_final_specs = _parse_eval_specs(recipe.get("eval_final") or [])
    mining_eval_specs = _parse_eval_specs(
        recipe.get("mining_heuristic_eval") or recipe.get("eval_dev") or recipe.get("eval") or []
    )
    if not eval_specs:
        raise RuntimeError("Downstream DOE requires at least one dev eval session")

    variant_dataset_dirs = {
        str(name): Path(str(path)).expanduser().resolve()
        for name, path in dict(narrow_doe_recipe.get("variant_dataset_dirs") or {}).items()
    }
    return DownstreamContext(
        baseline_summary_path=baseline_summary_path,
        recipe_path=recipe_path,
        recipe=recipe,
        narrow_doe_recipe=narrow_doe_recipe,
        base_config=base_config,
        output_root=output_root,
        hf_cache_root=Path(str(recipe.get("hf_cache_root") or output_root / "hf_cache"))
        .expanduser()
        .resolve(),
        speaker_mapping_path=speaker_mapping_path,
        diarization_model=str(
            recipe.get("diarization_model") or "pyannote/speaker-diarization-community-1"
        ),
        local_files_only=bool(recipe.get("local_files_only")),
        eval_specs=eval_specs,
        eval_final_specs=eval_final_specs,
        mining_eval_specs=mining_eval_specs,
        short_slice_session=str(short_slice.get("source_session") or ""),
        short_slice_window=dict(short_slice.get("window") or {}),
        prepared_eval_root=output_root / "prepared_eval",
        bank_profile_dir=Path(str(narrow_doe_recipe["bank_profile_dir"])).expanduser().resolve(),
        baseline_profile_dir=Path(str(narrow_doe_recipe["baseline_profile_dir"])).expanduser().resolve(),
        base_training_dataset_dir=Path(str(narrow_doe_recipe["base_training_dataset_dir"]))
        .expanduser()
        .resolve(),
        final_training_dataset_dir=Path(str(narrow_doe_recipe["training_dataset_dir"]))
        .expanduser()
        .resolve(),
        hard_negative_dataset_dir=(
            Path(str(narrow_doe_recipe["hard_negative_dataset_dir"])).expanduser().resolve()
            if narrow_doe_recipe.get("hard_negative_dataset_dir")
            else None
        ),
        variant_dataset_dirs=variant_dataset_dirs,
        classifier=dict(narrow_doe_recipe.get("classifier") or {}),
        speaker_bank_overrides=dict(narrow_doe_recipe.get("speaker_bank_overrides") or {}),
        baseline_metrics=_baseline_metrics_from_summary(baseline_summary),
    )


def _baseline_metric(context: DownstreamContext, session_name: str, metric_name: str) -> float:
    session_metrics = dict(context.baseline_metrics.get(session_name) or {})
    return float(session_metrics.get(metric_name) or 0.0)


def _accepted(
    result: Mapping[str, object],
    *,
    context: DownstreamContext,
    acceptance: Mapping[str, float],
) -> bool:
    session_results = dict(result.get("session_results") or {})
    session22_accuracy = float(
        dict(session_results.get("Session22") or {}).get("mean_accuracy") or 0.0
    )
    session61_matched = float(
        dict(session_results.get("Session61") or {}).get("mean_matched_accuracy") or 0.0
    )
    return bool(
        session61_matched
        >= _baseline_metric(context, "Session61", "mean_matched_accuracy")
        + float(acceptance["session61_matched_accuracy_gain"])
        and session22_accuracy
        >= _baseline_metric(context, "Session22", "mean_accuracy")
        - float(acceptance["session22_accuracy_regression_max"])
    )


def _ranking_key(result: Mapping[str, object]) -> Tuple[float, float, float, float]:
    session_results = dict(result.get("session_results") or {})
    session61 = dict(session_results.get("Session61") or {})
    short_slice = dict(session_results.get("short_segment_slice") or {})
    session22 = dict(session_results.get("Session22") or {})
    return (
        float(session61.get("mean_matched_accuracy") or 0.0),
        float(short_slice.get("mean_matched_accuracy") or 0.0),
        float(session22.get("mean_accuracy") or 0.0),
        float(session22.get("mean_matched_accuracy") or 0.0),
    )


def _sort_results(results: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        [dict(item) for item in results],
        key=_ranking_key,
        reverse=True,
    )


def _first_accepted(
    results: Sequence[Mapping[str, object]],
    *,
    context: DownstreamContext,
    acceptance: Mapping[str, float],
) -> Optional[Dict[str, object]]:
    for result in _sort_results(results):
        if _accepted(result, context=context, acceptance=acceptance):
            return result
    return None


def _deltas_for_result(
    *,
    context: DownstreamContext,
    session_results: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, float]]:
    deltas: Dict[str, Dict[str, float]] = {}
    for session_name, metrics in session_results.items():
        deltas[session_name] = {}
        baseline_metrics = dict(context.baseline_metrics.get(session_name) or {})
        for key, value in dict(metrics).items():
            if not isinstance(value, (int, float)):
                continue
            baseline_value = baseline_metrics.get(key)
            if baseline_value is None:
                continue
            deltas[session_name][key] = float(value) - float(baseline_value)
    return deltas


def _copy_profile(source_profile_dir: Path, target_profile_dir: Path) -> None:
    if target_profile_dir.exists():
        shutil.rmtree(target_profile_dir)
    shutil.copytree(source_profile_dir, target_profile_dir)


def _evaluate_dev_suite(
    *,
    context: DownstreamContext,
    experiment_dir: Path,
    profile_name: str,
    classifier_config: Mapping[str, object],
    speaker_bank_overrides: Mapping[str, object],
    device: str,
) -> Tuple[Dict[str, object], Path]:
    config_path = _write_eval_config(
        experiment_dir / "eval_config.json",
        base_config=context.base_config,
        hf_cache_root=context.hf_cache_root,
        profile_name=profile_name,
        diarization_model=context.diarization_model,
        speaker_mapping_path=context.speaker_mapping_path,
        classifier_min_margin=float(classifier_config["classifier_min_margin"]),
        threshold=float(classifier_config["threshold"]),
        match_aggregation=str(classifier_config["match_aggregation"]),
        min_segments_per_label=int(classifier_config["min_segments_per_label"]),
        speaker_bank_overrides=speaker_bank_overrides,
    )

    session_results: Dict[str, object] = {}
    for spec in context.eval_specs:
        summary = evaluate_multitrack_session(
            session_zip=spec.session_zip,
            session_jsonl=spec.transcript,
            output_dir=experiment_dir / "eval" / spec.name,
            cache_root=context.prepared_eval_root / spec.name,
            speaker_mapping_path=context.speaker_mapping_path,
            config_path=config_path,
            window_seconds=float(context.recipe.get("eval_window_seconds") or 300.0),
            hop_seconds=float(context.recipe.get("eval_hop_seconds") or 60.0),
            top_k=int(context.recipe.get("eval_top_k") or 3),
            min_speakers=int(context.recipe.get("eval_min_speakers") or 3),
            device_override=device,
            local_files_only_override=True if context.local_files_only else None,
        )
        session_results[spec.name] = _aggregate_eval_summary(summary)

        if (
            context.short_slice_window
            and _normalize_session_name(spec.name) == _normalize_session_name(context.short_slice_session)
        ):
            short_summary = evaluate_multitrack_session(
                session_zip=spec.session_zip,
                session_jsonl=spec.transcript,
                output_dir=experiment_dir / "eval" / "short_segment_slice",
                cache_root=context.prepared_eval_root / "short_segment_slice",
                speaker_mapping_path=context.speaker_mapping_path,
                config_path=config_path,
                window_seconds=float(context.recipe.get("eval_window_seconds") or 300.0),
                hop_seconds=float(context.recipe.get("eval_hop_seconds") or 60.0),
                top_k=1,
                min_speakers=int(context.recipe.get("eval_min_speakers") or 3),
                device_override=device,
                local_files_only_override=True if context.local_files_only else None,
                windows_override=[dict(context.short_slice_window)],
            )
            session_results["short_segment_slice"] = {
                **_aggregate_eval_summary(short_summary),
                "source_session": context.short_slice_session,
                "window": dict(context.short_slice_window),
            }

    return session_results, config_path


def _classifier_config_from_baseline(context: DownstreamContext) -> Dict[str, object]:
    return {
        "model_name": str(context.classifier["model_name"]),
        "classifier_c": float(context.classifier.get("classifier_c", 1.0)),
        "classifier_n_neighbors": int(context.classifier.get("classifier_n_neighbors", 13)),
        "classifier_min_margin": float(context.classifier["classifier_min_margin"]),
        "threshold": float(context.classifier["threshold"]),
        "match_aggregation": str(context.classifier["match_aggregation"]),
        "min_segments_per_label": int(context.classifier["min_segments_per_label"]),
    }


def _resolve_phase_b_hard_negative_settings(
    *,
    context: DownstreamContext,
    experiment: Mapping[str, object],
) -> Dict[str, object]:
    hard_negative = dict(experiment.get("hard_negative") or {})
    base_pair_caps = _parse_pair_caps(list(context.recipe.get("hard_negative_pair_caps") or []))
    override_pair_caps = _parse_pair_caps(list(hard_negative.get("pair_caps") or []))
    merged_pair_caps = {**base_pair_caps, **override_pair_caps}
    return {
        "seed_confusion_pairs": list(
            hard_negative.get("seed_confusion_pairs")
            or context.recipe.get("seed_confusion_pairs")
            or [["Cyrus Schwert", "Cletus Cobbington"]]
        ),
        "top_confusion_pairs": int(
            hard_negative.get("top_confusion_pairs")
            if hard_negative.get("top_confusion_pairs") is not None
            else context.recipe.get("top_confusion_pairs", 0)
        ),
        "hard_negative_max_margin": float(
            hard_negative.get("hard_negative_max_margin")
            if hard_negative.get("hard_negative_max_margin") is not None
            else context.recipe.get("hard_negative_max_margin", 0.12)
        ),
        "hard_negative_min_dominant_share": float(
            hard_negative.get("hard_negative_min_dominant_share")
            if hard_negative.get("hard_negative_min_dominant_share") is not None
            else context.recipe.get("hard_negative_min_dominant_share", 0.55)
        ),
        "hard_negative_per_pair_cap": int(
            hard_negative.get("hard_negative_per_pair_cap")
            if hard_negative.get("hard_negative_per_pair_cap") is not None
            else context.recipe.get("hard_negative_per_pair_cap", 60)
        ),
        "hard_negative_pair_caps": _pair_caps_to_recipe_items(merged_pair_caps),
        "hard_negative_per_speaker_cap": (
            int(hard_negative["hard_negative_per_speaker_cap"])
            if hard_negative.get("hard_negative_per_speaker_cap") is not None
            else (
                int(context.recipe["hard_negative_per_speaker_cap"])
                if context.recipe.get("hard_negative_per_speaker_cap") is not None
                else None
            )
        ),
        "hard_negative_max_fraction": float(
            hard_negative.get("hard_negative_max_fraction")
            if hard_negative.get("hard_negative_max_fraction") is not None
            else context.recipe.get("hard_negative_max_fraction", 0.20)
        ),
        "hard_negative_style_profile_name": str(
            hard_negative.get("hard_negative_style_profile_name")
            or context.recipe.get("hard_negative_style_profile_name")
            or "session61_like"
        ),
        "hard_negative_style_score_threshold": float(
            hard_negative.get("hard_negative_style_score_threshold")
            if hard_negative.get("hard_negative_style_score_threshold") is not None
            else context.recipe.get("hard_negative_style_score_threshold", 0.45)
        ),
        "hard_negative_min_style_samples_per_pair": int(
            hard_negative.get("hard_negative_min_style_samples_per_pair")
            if hard_negative.get("hard_negative_min_style_samples_per_pair") is not None
            else context.recipe.get("hard_negative_min_style_samples_per_pair", 10)
        ),
    }


def _forbidden_eval_sessions(context: DownstreamContext) -> List[str]:
    all_eval_specs = context.eval_specs + context.eval_final_specs + context.mining_eval_specs
    return sorted(
        {
            _normalize_session_name(spec.name)
            for spec in all_eval_specs
        }
        | {
            _normalize_session_name(spec.session_zip.stem)
            for spec in all_eval_specs
        }
    )


def _assert_records_exclude_eval_sessions(
    *,
    records: Sequence[Mapping[str, object]],
    context: DownstreamContext,
) -> None:
    forbidden = set(_forbidden_eval_sessions(context))
    leaked = sorted(
        {
            str(record.get("source_session") or "")
            for record in records
            if _normalize_session_name(record.get("source_session")) in forbidden
        }
    )
    if leaked:
        raise RuntimeError(
            "Hard-negative refresh attempted to train on eval sessions: " + ", ".join(leaked)
        )


def _run_classifier_only_experiment(
    *,
    context: DownstreamContext,
    output_dir: Path,
    phase: str,
    experiment_name: str,
    classifier_config: Mapping[str, object],
    notes: Optional[str],
    dev_only: bool,
    device: str,
) -> Dict[str, object]:
    experiment_dir = output_dir / phase / experiment_name
    profile_name = _profile_name("downstream_retrain", phase, experiment_name, output_dir)
    profile_dir = context.hf_cache_root / "speaker_bank" / profile_name
    _copy_profile(context.bank_profile_dir, profile_dir)

    dataset, dataset_summary = load_classifier_dataset(context.final_training_dataset_dir)
    training_summary = train_segment_classifier_from_dataset(
        dataset=dataset,
        profile_dir=profile_dir,
        model_name=str(classifier_config["model_name"]),
        classifier_c=float(classifier_config.get("classifier_c", 1.0)),
        classifier_n_neighbors=int(classifier_config.get("classifier_n_neighbors", 13)),
        base_summary={
            "artifact_id": str(dataset_summary.get("artifact_id") or ""),
            "parent_artifacts": list(dataset_summary.get("parent_artifacts") or []),
            "dev_only": bool(dev_only),
            "experiment_name": experiment_name,
        },
    )
    training_summary_path = _write_training_summary(profile_dir, training_summary)

    session_results, config_path = _evaluate_dev_suite(
        context=context,
        experiment_dir=experiment_dir,
        profile_name=profile_name,
        classifier_config=classifier_config,
        speaker_bank_overrides=context.speaker_bank_overrides,
        device=device,
    )
    result = {
        "name": experiment_name,
        "phase": phase,
        "mode": "classifier_only",
        "notes": notes,
        "dev_only": bool(dev_only),
        "classifier": dict(classifier_config),
        "hard_negative": None,
        "profile_dir": str(profile_dir),
        "artifact_paths": {
            "config_path": str(config_path),
            "training_summary_path": str(training_summary_path),
            "eval_root": str(experiment_dir / "eval"),
        },
        "session_results": session_results,
        "deltas": _deltas_for_result(context=context, session_results=session_results),
    }
    _json_write(experiment_dir / "experiment_result.json", result)
    return result


def _run_hard_negative_refresh_experiment(
    *,
    context: DownstreamContext,
    output_dir: Path,
    experiment_name: str,
    classifier_config: Mapping[str, object],
    hard_negative_settings: Mapping[str, object],
    candidate_variants: Sequence[str],
    notes: Optional[str],
    dev_only: bool,
    device: str,
) -> Dict[str, object]:
    experiment_dir = output_dir / "phase_b" / experiment_name
    base_training_dataset, base_training_summary = load_classifier_dataset(
        context.base_training_dataset_dir
    )
    candidate_pool_dirs: List[Path] = []
    for variant_name in candidate_variants:
        candidate_dir = context.variant_dataset_dirs.get(str(variant_name))
        if candidate_dir is None:
            raise RuntimeError(f"Missing candidate-pool dataset for variant: {variant_name}")
        candidate_pool_dirs.append(candidate_dir)

    mining_profile_name = _profile_name("downstream_mining", "phase_b", experiment_name, output_dir)
    mining_profile_dir = context.hf_cache_root / "speaker_bank" / mining_profile_name
    _copy_profile(context.bank_profile_dir, mining_profile_dir)
    mining_training_summary = train_segment_classifier_from_dataset(
        dataset=base_training_dataset,
        profile_dir=mining_profile_dir,
        model_name=str(classifier_config["model_name"]),
        classifier_c=float(classifier_config.get("classifier_c", 1.0)),
        classifier_n_neighbors=int(classifier_config.get("classifier_n_neighbors", 13)),
        base_summary={
            "artifact_id": str(base_training_summary.get("artifact_id") or ""),
            "parent_artifacts": list(base_training_summary.get("parent_artifacts") or []),
            "dev_only": bool(dev_only),
            "experiment_name": experiment_name,
            "purpose": "hard_negative_mining",
        },
    )
    mining_training_summary_path = _write_training_summary(mining_profile_dir, mining_training_summary)
    mining_eval_dir = experiment_dir / "mining_eval"
    mining_eval_results, mining_config_path = _evaluate_dev_suite(
        context=DownstreamContext(
            **{
                **context.__dict__,
                "eval_specs": list(context.mining_eval_specs),
            }
        ),
        experiment_dir=mining_eval_dir,
        profile_name=mining_profile_name,
        classifier_config=classifier_config,
        speaker_bank_overrides=context.speaker_bank_overrides,
        device=device,
    )

    eval_summaries: List[Mapping[str, object]] = []
    for spec in context.mining_eval_specs:
        summary_path = mining_eval_dir / "eval" / spec.name / "summary.json"
        eval_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    short_summary_path = mining_eval_dir / "eval" / "short_segment_slice" / "summary.json"
    if short_summary_path.exists():
        eval_summaries.append(json.loads(short_summary_path.read_text(encoding="utf-8")))

    hard_negative_dataset, hard_negative_records, hard_negative_summary = build_hard_negative_dataset(
        eval_summaries=eval_summaries,
        candidate_pool_dirs=candidate_pool_dirs,
        base_dataset_samples=base_training_dataset.samples,
        seed_pairs=list(hard_negative_settings["seed_confusion_pairs"]),
        top_confusion_pairs=int(hard_negative_settings["top_confusion_pairs"]),
        max_eval_margin=float(hard_negative_settings["hard_negative_max_margin"]),
        min_mixed_dominant_share=float(
            hard_negative_settings["hard_negative_min_dominant_share"]
        ),
        per_pair_cap=int(hard_negative_settings["hard_negative_per_pair_cap"]),
        pair_caps=_parse_pair_caps(
            list(hard_negative_settings["hard_negative_pair_caps"])
        ),
        per_speaker_cap=hard_negative_settings["hard_negative_per_speaker_cap"],
        max_fraction=float(hard_negative_settings["hard_negative_max_fraction"]),
        style_profile_name=str(hard_negative_settings["hard_negative_style_profile_name"]),
        style_score_threshold=float(
            hard_negative_settings["hard_negative_style_score_threshold"]
        ),
        min_style_samples_per_pair=int(
            hard_negative_settings["hard_negative_min_style_samples_per_pair"]
        ),
    )
    _assert_records_exclude_eval_sessions(records=hard_negative_records, context=context)

    _json_write(experiment_dir / "hard_negative_summary.json", hard_negative_summary)
    _jsonl_write(experiment_dir / "hard_negative_records.jsonl", hard_negative_records)
    hard_negative_dataset_dir = experiment_dir / "hard_negative_dataset"
    save_classifier_dataset(
        hard_negative_dataset_dir,
        hard_negative_dataset,
        summary={
            "artifact_id": f"{experiment_name}_hard_negative_refresh",
            "parent_artifacts": [str(base_training_summary.get("artifact_id") or "")],
            "dev_only": bool(dev_only),
            "experiment_name": experiment_name,
            "selected": int(hard_negative_summary.get("selected") or 0),
        },
    )

    final_training_dataset = append_dataset(base_training_dataset, hard_negative_dataset)
    final_training_dataset_dir = experiment_dir / "final_training_dataset"
    save_classifier_dataset(
        final_training_dataset_dir,
        final_training_dataset,
        summary={
            "artifact_id": f"{experiment_name}_final_training_refresh",
            "parent_artifacts": [
                str(base_training_summary.get("artifact_id") or ""),
                f"{experiment_name}_hard_negative_refresh",
            ],
            "dev_only": bool(dev_only),
            "experiment_name": experiment_name,
            "samples": int(final_training_dataset.samples),
        },
    )
    profile_name = _profile_name("downstream_retrain", "phase_b", experiment_name, output_dir)
    profile_dir = context.hf_cache_root / "speaker_bank" / profile_name
    _copy_profile(context.bank_profile_dir, profile_dir)
    training_summary = train_segment_classifier_from_dataset(
        dataset=final_training_dataset,
        profile_dir=profile_dir,
        model_name=str(classifier_config["model_name"]),
        classifier_c=float(classifier_config.get("classifier_c", 1.0)),
        classifier_n_neighbors=int(classifier_config.get("classifier_n_neighbors", 13)),
        base_summary={
            "artifact_id": str(base_training_summary.get("artifact_id") or ""),
            "parent_artifacts": list(base_training_summary.get("parent_artifacts") or []),
            "dev_only": bool(dev_only),
            "experiment_name": experiment_name,
            "hard_negative_selected": int(hard_negative_summary.get("selected") or 0),
        },
    )
    training_summary_path = _write_training_summary(profile_dir, training_summary)

    session_results, config_path = _evaluate_dev_suite(
        context=context,
        experiment_dir=experiment_dir,
        profile_name=profile_name,
        classifier_config=classifier_config,
        speaker_bank_overrides=context.speaker_bank_overrides,
        device=device,
    )
    result = {
        "name": experiment_name,
        "phase": "phase_b",
        "mode": "hard_negative_refresh",
        "notes": notes,
        "dev_only": bool(dev_only),
        "classifier": dict(classifier_config),
        "hard_negative": dict(hard_negative_settings),
        "profile_dir": str(profile_dir),
        "artifact_paths": {
            "config_path": str(config_path),
            "mining_config_path": str(mining_config_path),
            "training_summary_path": str(training_summary_path),
            "mining_training_summary_path": str(mining_training_summary_path),
            "eval_root": str(experiment_dir / "eval"),
            "mining_eval_root": str(mining_eval_dir / "eval"),
            "hard_negative_summary_path": str(experiment_dir / "hard_negative_summary.json"),
            "hard_negative_records_path": str(experiment_dir / "hard_negative_records.jsonl"),
            "hard_negative_dataset_dir": str(hard_negative_dataset_dir),
            "final_training_dataset_dir": str(final_training_dataset_dir),
        },
        "session_results": session_results,
        "mining_eval_results": mining_eval_results,
        "deltas": _deltas_for_result(context=context, session_results=session_results),
    }
    _json_write(experiment_dir / "experiment_result.json", result)
    return result


def _expand_phase_a_calibration(
    *,
    family_results: Sequence[Mapping[str, object]],
    calibration_spec: Mapping[str, object],
) -> List[Dict[str, object]]:
    ranked = _sort_results(family_results)
    top_n_raw = calibration_spec.get("top_n")
    top_n = 2 if top_n_raw is None else int(top_n_raw)
    thresholds = list(calibration_spec.get("thresholds") or [0.34, 0.36, 0.38])
    margins = list(calibration_spec.get("classifier_min_margins") or [0.04, 0.06, 0.08])
    expanded: List[Dict[str, object]] = []
    for result in ranked[:top_n]:
        classifier = dict(result.get("classifier") or {})
        base_name = str(result["name"])
        for threshold in thresholds:
            for margin in margins:
                expanded.append(
                    {
                        "name": (
                            f"{base_name}_thr_{str(float(threshold)).replace('.', '_')}"
                            f"_margin_{str(float(margin)).replace('.', '_')}"
                        ),
                        "notes": f"Calibration sweep derived from {base_name}",
                        "classifier": {
                            **classifier,
                            "threshold": float(threshold),
                            "classifier_min_margin": float(margin),
                        },
                    }
                )
    return expanded


def run_downstream_retrain_doe(
    *,
    baseline_summary_path: Path,
    spec_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, object]:
    context = _load_context(baseline_summary_path=baseline_summary_path.expanduser().resolve())
    spec = _load_experiment_spec(spec_path.expanduser().resolve())
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_classifier = _classifier_config_from_baseline(context)
    dev_only = bool(spec["dev_only"])
    acceptance = dict(spec["acceptance"])

    phase_a_family_results: List[Dict[str, object]] = []
    for experiment in list(dict(spec["phase_a"]).get("family_sweep") or []):
        classifier_config = {
            **base_classifier,
            **dict(experiment.get("classifier") or {}),
        }
        phase_a_family_results.append(
            _run_classifier_only_experiment(
                context=context,
                output_dir=output_dir,
                phase="phase_a_family",
                experiment_name=str(experiment["name"]),
                classifier_config=classifier_config,
                notes=str(experiment.get("notes") or "") or None,
                dev_only=dev_only,
                device=device,
            )
        )

    phase_a_calibration_results: List[Dict[str, object]] = []
    for calibration_experiment in _expand_phase_a_calibration(
        family_results=phase_a_family_results,
        calibration_spec=dict(dict(spec["phase_a"]).get("calibration") or {}),
    ):
        phase_a_calibration_results.append(
            _run_classifier_only_experiment(
                context=context,
                output_dir=output_dir,
                phase="phase_a_calibration",
                experiment_name=str(calibration_experiment["name"]),
                classifier_config=dict(calibration_experiment["classifier"]),
                notes=str(calibration_experiment.get("notes") or "") or None,
                dev_only=dev_only,
                device=device,
            )
        )

    phase_a_results = _sort_results(phase_a_family_results + phase_a_calibration_results)
    for result in phase_a_results:
        result["accepted"] = _accepted(result, context=context, acceptance=acceptance)

    promoted_phase_a = _first_accepted(phase_a_results, context=context, acceptance=acceptance)

    phase_b_results: List[Dict[str, object]] = []
    phase_b_classifier_seed = dict(promoted_phase_a.get("classifier") if promoted_phase_a else (phase_a_results[0].get("classifier") if phase_a_results else base_classifier))  # type: ignore[union-attr]
    if promoted_phase_a is None:
        candidate_variants = list(dict(spec["phase_b"]).get("candidate_variants") or ["mixed_raw"])
        for experiment in list(dict(spec["phase_b"]).get("experiments") or []):
            hard_negative_settings = _resolve_phase_b_hard_negative_settings(
                context=context,
                experiment=dict(experiment),
            )
            result = _run_hard_negative_refresh_experiment(
                context=context,
                output_dir=output_dir,
                experiment_name=str(experiment["name"]),
                classifier_config=phase_b_classifier_seed,
                hard_negative_settings=hard_negative_settings,
                candidate_variants=candidate_variants,
                notes=str(experiment.get("notes") or "") or None,
                dev_only=dev_only,
                device=device,
            )
            result["accepted"] = _accepted(result, context=context, acceptance=acceptance)
            phase_b_results.append(result)
        phase_b_results = _sort_results(phase_b_results)

    all_results = _sort_results(phase_a_results + phase_b_results)
    promoted_result = _first_accepted(all_results, context=context, acceptance=acceptance)

    report = {
        "dev_only": dev_only,
        "baseline_summary_path": str(context.baseline_summary_path),
        "recipe_path": str(context.recipe_path),
        "acceptance": acceptance,
        "baseline_metrics": context.baseline_metrics,
        "phase_a": {
            "family_sweep": phase_a_family_results,
            "calibration": phase_a_calibration_results,
            "results": phase_a_results,
            "promoted_result": promoted_phase_a,
        },
        "phase_b": {
            "skipped": promoted_phase_a is not None,
            "classifier_seed": phase_b_classifier_seed,
            "results": phase_b_results,
            "candidate_variants": list(dict(spec["phase_b"]).get("candidate_variants") or ["mixed_raw"]),
        },
        "best_result": all_results[0] if all_results else None,
        "promoted_result": promoted_result,
    }
    _json_write(output_dir / "downstream_retrain_doe_report.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run downstream-only speaker-ID retrain DOE without rebuilding bank or mixed-base datasets."
    )
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = run_downstream_retrain_doe(
        baseline_summary_path=args.baseline_summary,
        spec_path=args.spec,
        output_dir=args.output_dir,
        device=str(args.device),
    )
    print(json.dumps(report, indent=2))
    return 0


__all__ = [
    "run_downstream_retrain_doe",
    "_accepted",
    "_load_experiment_spec",
    "_ranking_key",
    "_sort_results",
]
