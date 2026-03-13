from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.cli import _load_yaml_or_json  # noqa: E402
from transcriber.multitrack_eval import evaluate_multitrack_session  # noqa: E402


def _json_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
    speaker_bank_overrides: Optional[Mapping[str, object]] = None,
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
            [float(item.get("metrics", {}).get("accuracy") or 0.0) for item in results]
        ),
        "mean_coverage": _mean(
            [float(item.get("metrics", {}).get("coverage") or 0.0) for item in results]
        ),
        "mean_matched_accuracy": _mean(
            [float(item.get("metrics", {}).get("matched_accuracy") or 0.0) for item in results]
        ),
        "mean_accuracy_pre_graph": (
            _mean(
                [
                    float(item.get("metrics_pre_graph", {}).get("accuracy") or 0.0)
                    for item in results
                    if item.get("metrics_pre_graph") is not None
                ]
            )
            if any(item.get("metrics_pre_graph") is not None for item in results)
            else None
        ),
        "mean_matched_accuracy_pre_graph": (
            _mean(
                [
                    float(item.get("metrics_pre_graph", {}).get("matched_accuracy") or 0.0)
                    for item in results
                    if item.get("metrics_pre_graph") is not None
                ]
            )
            if any(item.get("metrics_pre_graph") is not None for item in results)
            else None
        ),
    }


def _merge_session_graph_overrides(
    base_overrides: Mapping[str, object],
    experiment_override: Mapping[str, object],
) -> Dict[str, object]:
    merged = dict(base_overrides)
    merged["repair"] = {
        **dict(base_overrides.get("repair") or {}),
        "enabled": True,
    }
    base_graph = dict(base_overrides.get("session_graph") or {})
    experiment_graph = dict(experiment_override.get("session_graph") or {})
    base_pairs = dict(base_graph.get("pair_overrides") or {})
    experiment_pairs = dict(experiment_graph.get("pair_overrides") or {})
    merged_graph = {
        **base_graph,
        **{key: value for key, value in experiment_graph.items() if key != "pair_overrides"},
        "enabled": True,
        "pair_overrides": {
            **base_pairs,
            **experiment_pairs,
        },
    }
    merged["session_graph"] = merged_graph
    return merged


def _baseline_metric(
    baseline_summary: Mapping[str, object],
    session_name: str,
    metric_name: str,
) -> Optional[float]:
    dev_eval = dict(baseline_summary.get("dev_eval") or baseline_summary.get("canonical_eval") or {})
    session_metrics = dict(dev_eval.get(session_name) or {})
    value = session_metrics.get(metric_name)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pair-specific session-graph DOE against the current trained profile."
    )
    parser.add_argument("--baseline-summary", required=True, help="Path to baseline_summary.json")
    parser.add_argument("--pair-spec", required=True, help="YAML/JSON experiment spec file")
    parser.add_argument("--output-dir", required=True, help="Directory for configs and reports")
    parser.add_argument("--device", default="cuda", help="Eval device override")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force local-files-only model loading during eval",
    )
    args = parser.parse_args()

    baseline_summary_path = Path(args.baseline_summary).expanduser().resolve()
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    recipe = _load_yaml_or_json(str(Path(baseline_summary["recipe_path"]).expanduser().resolve())) or {}
    dev_eval_manifest_path = Path(
        str(baseline_summary.get("dev_eval_manifest_path") or baseline_summary["eval_manifest_path"])
    ).expanduser()
    dev_eval_manifest = json.loads(dev_eval_manifest_path.read_text(encoding="utf-8"))
    narrow_doe_recipe = json.loads(
        Path(str(baseline_summary["narrow_doe_recipe_path"])).expanduser().read_text(encoding="utf-8")
    )
    pair_spec = _load_yaml_or_json(str(Path(args.pair_spec).expanduser().resolve())) or {}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = Path(str(narrow_doe_recipe["baseline_profile_dir"])).expanduser().resolve()
    profile_name = profile_dir.name
    hf_cache_root = profile_dir.parent.parent
    base_config = _load_yaml_or_json(str(Path(recipe["base_config"]).expanduser().resolve())) or {}
    speaker_mapping_path = Path(str(recipe["speaker_mapping"])).expanduser().resolve()
    diarization_model = str(recipe.get("diarization_model") or "pyannote/speaker-diarization-community-1")
    classifier = dict(narrow_doe_recipe.get("classifier") or {})
    base_overrides = dict(narrow_doe_recipe.get("speaker_bank_overrides") or {})
    prepared_eval_root = Path(str(baseline_summary["output_root"])).expanduser().resolve() / "prepared_eval"

    eval_specs = list(recipe.get("eval_dev") or recipe.get("eval") or [])
    canonical_suite = dict(dev_eval_manifest.get("canonical_suite") or {})
    short_slice = dict(canonical_suite.get("short_segment_slice") or {})
    short_session_name = str(short_slice.get("source_session") or "")
    short_window = dict(short_slice.get("window") or {})
    experiments = list(pair_spec.get("experiments") or [])
    if not experiments:
        raise RuntimeError("Pair DOE spec must define at least one experiment")

    baseline_session22 = _baseline_metric(baseline_summary, "Session22", "mean_accuracy")
    baseline_session61 = _baseline_metric(baseline_summary, "Session61", "mean_matched_accuracy")

    results: List[Dict[str, object]] = []
    for experiment in experiments:
        name = str(experiment.get("name") or "").strip()
        if not name:
            continue
        notes = str(experiment.get("notes") or "").strip() or None
        speaker_bank_overrides = _merge_session_graph_overrides(base_overrides, dict(experiment))

        config_path = _write_eval_config(
            output_dir / "configs" / f"{name}.json",
            base_config=base_config,
            hf_cache_root=hf_cache_root,
            profile_name=profile_name,
            diarization_model=diarization_model,
            speaker_mapping_path=speaker_mapping_path,
            classifier_min_margin=float(classifier["classifier_min_margin"]),
            threshold=float(classifier["threshold"]),
            match_aggregation=str(classifier["match_aggregation"]),
            min_segments_per_label=int(classifier["min_segments_per_label"]),
            speaker_bank_overrides=speaker_bank_overrides,
        )

        session_results: Dict[str, object] = {}
        for spec in eval_specs:
            session_name = str(spec["name"])
            summary = evaluate_multitrack_session(
                session_zip=Path(str(spec["session_zip"])).expanduser().resolve(),
                session_jsonl=Path(str(spec["transcript"])).expanduser().resolve(),
                output_dir=output_dir / "eval" / name / session_name,
                cache_root=prepared_eval_root / session_name,
                speaker_mapping_path=speaker_mapping_path,
                config_path=config_path,
                window_seconds=float(recipe.get("eval_window_seconds") or 300.0),
                hop_seconds=float(recipe.get("eval_hop_seconds") or 60.0),
                top_k=int(recipe.get("eval_top_k") or 3),
                min_speakers=int(recipe.get("eval_min_speakers") or 3),
                device_override=args.device,
                local_files_only_override=True if args.local_files_only else None,
            )
            session_results[session_name] = _aggregate_eval_summary(summary)

            if short_window and session_name == short_session_name:
                short_summary = evaluate_multitrack_session(
                    session_zip=Path(str(spec["session_zip"])).expanduser().resolve(),
                    session_jsonl=Path(str(spec["transcript"])).expanduser().resolve(),
                    output_dir=output_dir / "eval" / name / "short_segment_slice",
                    cache_root=prepared_eval_root / "short_segment_slice",
                    speaker_mapping_path=speaker_mapping_path,
                    config_path=config_path,
                    window_seconds=float(recipe.get("eval_window_seconds") or 300.0),
                    hop_seconds=float(recipe.get("eval_hop_seconds") or 60.0),
                    top_k=1,
                    min_speakers=int(recipe.get("eval_min_speakers") or 3),
                    device_override=args.device,
                    local_files_only_override=True if args.local_files_only else None,
                    windows_override=[short_window],
                )
                session_results["short_segment_slice"] = {
                    **_aggregate_eval_summary(short_summary),
                    "source_session": short_session_name,
                    "window": short_window,
                }

        session22_accuracy = float(
            dict(session_results.get("Session22") or {}).get("mean_accuracy") or 0.0
        )
        session61_matched = float(
            dict(session_results.get("Session61") or {}).get("mean_matched_accuracy") or 0.0
        )
        short_slice_matched = float(
            dict(session_results.get("short_segment_slice") or {}).get("mean_matched_accuracy") or 0.0
        )
        dev_acceptance = (
            (baseline_session22 is None or session22_accuracy >= baseline_session22 - 0.01)
            and (baseline_session61 is None or session61_matched >= baseline_session61 + 0.01)
        )

        results.append(
            {
                "name": name,
                "notes": notes,
                "config_path": str(config_path),
                "session_results": session_results,
                "speaker_bank_overrides": speaker_bank_overrides,
                "dev_acceptance": dev_acceptance,
                "score_tuple": [session61_matched, short_slice_matched, session22_accuracy],
            }
        )

    results.sort(
        key=lambda item: (
            bool(item["dev_acceptance"]),
            float(item["score_tuple"][0]),
            float(item["score_tuple"][1]),
            float(item["score_tuple"][2]),
        ),
        reverse=True,
    )

    report = {
        "baseline_summary_path": str(baseline_summary_path),
        "pair_spec_path": str(Path(args.pair_spec).expanduser().resolve()),
        "baseline_metrics": {
            "Session22_mean_accuracy": baseline_session22,
            "Session61_mean_matched_accuracy": baseline_session61,
        },
        "results": results,
        "best_result": results[0] if results else None,
    }
    _json_write(output_dir / "session_graph_pair_doe_report.json", report)


if __name__ == "__main__":
    main()
