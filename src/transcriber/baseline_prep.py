from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .cli import _load_yaml_or_json
from .hard_negatives import build_hard_negative_dataset
from .multitrack_eval import evaluate_multitrack_session, select_candidate_windows
from .prep_artifacts import (
    append_dataset,
    build_artifact_manifest,
    build_coverage_report,
    build_stage_manifest,
    build_source_session_speaker_breakdown,
    collect_input_file_identities,
    current_git_commit,
    load_manifest,
    stage_manifest_is_reusable,
    StageMetricsLogger,
    save_manifest,
)
from .segment_classifier import (
    _find_session_sources,
    _safe_path_identity,
    balance_classifier_dataset,
    build_classifier_dataset_from_bank,
    build_classifier_dataset_from_multitrack,
    materialize_classifier_dataset_from_mixed_base,
    load_labeled_records,
    load_classifier_dataset,
    merge_classifier_datasets,
    relabel_classifier_dataset_sources,
    save_classifier_dataset,
    train_segment_classifier_from_dataset,
)
from .speaker_bank import SpeakerBank


DEFAULT_BASELINE_PACK = ("mixed_raw", "light_x1", "discord_x1")
DEFAULT_BUILD_VARIANTS = ("mixed_raw", "light_x1", "discord_x1")
DEFAULT_HARD_NEGATIVE_CANDIDATE_VARIANTS = ("mixed_raw",)
DEFAULT_STAGE_ORDER = ("bank", "mixed_base", "variants", "hard_negatives", "train", "eval")
DEFAULT_QUALITY_FILTERS = {
    "clipping_fraction_max": 0.005,
    "silence_fraction_max": 0.80,
}
DEFAULT_VARIANTS: Dict[str, Dict[str, object]] = {
    "mixed_raw": {
        "augmentation_profile": "none",
        "augmentation_copies": 0,
        "include_base_samples": True,
    },
    "light_x1": {
        "augmentation_profile": "light",
        "augmentation_copies": 1,
        "include_base_samples": False,
    },
    "discord_x1": {
        "augmentation_profile": "discord",
        "augmentation_copies": 1,
        "include_base_samples": False,
    },
    "discord_x2": {
        "augmentation_profile": "discord",
        "augmentation_copies": 2,
        "include_base_samples": False,
    },
}
DEFAULT_CURRENT_WINNER = {
    "model_name": "knn",
    "classifier_c": 1.0,
    "classifier_n_neighbors": 7,
    "classifier_min_margin": 0.03,
    "threshold": 0.40,
    "min_segments_per_label": 2,
}


@dataclass(frozen=True)
class EvalSpec:
    name: str
    session_zip: Path
    transcript: Path


def _json_write(path: Path, payload: Mapping[str, object]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return output_path


def _jsonl_write(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record)) + "\n")
    return output_path


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


def _collect_training_sources(input_roots: Sequence[Path], excluded_stems: Sequence[str]) -> List[Path]:
    excluded = {str(item).strip().lower() for item in excluded_stems if str(item).strip()}
    sources: List[Path] = []
    seen: set[str] = set()
    for root in input_roots:
        for source in _find_session_sources(root):
            if source.stem.strip().lower() in excluded:
                continue
            identity = _safe_path_identity(source)
            if identity in seen:
                continue
            seen.add(identity)
            sources.append(source)
    return sorted(sources, key=lambda item: item.name.lower())


def _materialize_training_dir(session_sources: Sequence[Path], target_dir: Path) -> Path:
    output_dir = Path(target_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in session_sources:
        link_path = output_dir / source.name
        if link_path.exists():
            continue
        link_path.symlink_to(source)
    return output_dir


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
    min_segments_per_label: int,
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
    speaker_bank_cfg["match_aggregation"] = "mean"
    speaker_bank_cfg["min_segments_per_label"] = int(min_segments_per_label)
    classifier_cfg = dict(speaker_bank_cfg.get("classifier") or {})
    classifier_cfg["min_confidence"] = 0.0
    classifier_cfg["min_margin"] = float(classifier_min_margin)
    speaker_bank_cfg["classifier"] = classifier_cfg
    payload["speaker_bank"] = speaker_bank_cfg
    return _json_write(output_path, payload)


def _window_records(records: Sequence[dict], start: float, end: float) -> List[dict]:
    return [
        record
        for record in records
        if float(record.get("end") or 0.0) > start and float(record.get("start") or 0.0) < end
    ]


def _median_segment_duration(records: Sequence[dict]) -> float:
    durations = [
        max(float(record.get("end") or 0.0) - float(record.get("start") or 0.0), 0.0)
        for record in records
    ]
    if not durations:
        return float("inf")
    ordered = sorted(durations)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _derive_short_segment_slice(
    eval_specs: Sequence[EvalSpec],
    *,
    speaker_mapping: Mapping[str, object],
    window_seconds: float,
    hop_seconds: float,
    top_k: int,
    min_speakers: int,
) -> Tuple[EvalSpec, dict]:
    best_spec = eval_specs[0]
    best_window = None
    best_score = float("inf")
    for spec in eval_specs:
        records = load_labeled_records(
            spec.transcript,
            speaker_aliases={"zariel torgan": "David Tanglethorn"},
            speaker_mapping=speaker_mapping,
        )
        windows = select_candidate_windows(
            records,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            top_k=top_k,
            min_speakers=min_speakers,
        )
        for window in windows:
            score = _median_segment_duration(
                _window_records(records, float(window["start"]), float(window["end"]))
            )
            if score < best_score:
                best_score = score
                best_spec = spec
                best_window = dict(window)
    if best_window is None:
        raise RuntimeError("Unable to derive a short-segment canonical eval slice")
    return best_spec, best_window


def _summarize_eval(summary: Mapping[str, object]) -> Dict[str, object]:
    results = list(summary.get("results") or [])
    accuracy_values = [float(item.get("metrics", {}).get("accuracy") or 0.0) for item in results]
    coverage_values = [float(item.get("metrics", {}).get("coverage") or 0.0) for item in results]
    confusion: Dict[str, Dict[str, int]] = {}
    for item in results:
        for speaker, counts in dict(item.get("metrics", {}).get("confusion") or {}).items():
            target = confusion.setdefault(str(speaker), {})
            for predicted, value in dict(counts).items():
                target[str(predicted)] = int(target.get(str(predicted), 0)) + int(value or 0)
    return {
        "mean_accuracy": (sum(accuracy_values) / len(accuracy_values)) if accuracy_values else 0.0,
        "mean_coverage": (sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0,
        "confusion": confusion,
    }


def _write_augmented_dataset_summary(
    dataset_dir: Path,
    *,
    dataset_summary: Mapping[str, object],
    artifact_id: str,
    parent_artifacts: Sequence[str],
    quality_filters: Optional[Mapping[str, object]] = None,
    source_groups: Optional[Mapping[str, object]] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Path:
    summary_path = Path(dataset_dir).expanduser() / "dataset_summary.json"
    payload = dict(dataset_summary)
    payload["artifact_id"] = artifact_id
    payload["parent_artifacts"] = list(parent_artifacts)
    payload.setdefault("quality_filters", dict(quality_filters or {}))
    payload.setdefault("source_groups", dict(source_groups or {}))
    if extra:
        payload.update(dict(extra))
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _bank_profile_name(prefix: str, artifact_id: str) -> str:
    return f"{prefix}_{artifact_id[:12]}"


def _training_summary_path(profile_dir: Path) -> Path:
    return Path(profile_dir).expanduser() / "classifier_training_summary.json"


def _write_training_summary(profile_dir: Path, summary: Mapping[str, object]) -> Path:
    return _json_write(_training_summary_path(profile_dir), summary)


def _stage_manifest_path(base_output_root: Path, stage: str) -> Path:
    return Path(base_output_root).expanduser() / "stage_manifests" / f"{stage}.json"


def _selected_stage_names(
    recipe_stages: Sequence[str],
    *,
    from_stage: Optional[str],
    to_stage: Optional[str],
) -> List[str]:
    requested = [stage for stage in DEFAULT_STAGE_ORDER if stage in set(recipe_stages)]
    if not requested:
        requested = list(DEFAULT_STAGE_ORDER)
    if from_stage:
        if from_stage not in DEFAULT_STAGE_ORDER:
            raise RuntimeError(f"Unknown from-stage: {from_stage}")
        requested = requested[requested.index(from_stage) :] if from_stage in requested else []
    if to_stage:
        if to_stage not in DEFAULT_STAGE_ORDER:
            raise RuntimeError(f"Unknown to-stage: {to_stage}")
        requested = requested[: requested.index(to_stage) + 1] if to_stage in requested else []
    return requested


def _load_stage_outputs(manifest_path: Path) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise RuntimeError(f"Missing stage manifest: {manifest_path}")
    return dict(manifest.get("outputs") or {})


def _save_stage(
    *,
    base_output_root: Path,
    stage: str,
    stage_signature: Mapping[str, object],
    outputs: Mapping[str, object],
    required_paths: Sequence[Path],
    parent_stages: Sequence[str],
    git_commit: Optional[str],
) -> Path:
    manifest = build_stage_manifest(
        stage=stage,
        stage_signature=stage_signature,
        outputs=outputs,
        required_paths=[str(Path(path).expanduser()) for path in required_paths],
        parent_stages=parent_stages,
        git_commit=git_commit,
    )
    manifest_path = _stage_manifest_path(base_output_root, stage)
    save_manifest(manifest_path, manifest)
    completion_marker = manifest_path.with_suffix(".done")
    completion_marker.parent.mkdir(parents=True, exist_ok=True)
    completion_marker.write_text(str(manifest["artifact_id"]), encoding="utf-8")
    return manifest_path


def prepare_baseline(
    *,
    recipe_path: Path,
    output_root: Optional[Path] = None,
    allow_legacy_reuse: bool = False,
    resume: Optional[bool] = None,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
) -> Dict[str, object]:
    recipe = _load_yaml_or_json(str(Path(recipe_path).expanduser().resolve())) or {}
    base_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else Path(str(recipe.get("output_root") or "baseline_prep")).expanduser().resolve()
    )
    base_output_root.mkdir(parents=True, exist_ok=True)
    resume_requested = bool(recipe.get("resume", True)) if resume is None else bool(resume)
    selected_stage_names = _selected_stage_names(
        list(recipe.get("stages") or list(DEFAULT_STAGE_ORDER)),
        from_stage=from_stage,
        to_stage=to_stage,
    )

    hf_cache_root = Path(
        str(recipe.get("hf_cache_root") or (base_output_root / "hf_cache"))
    ).expanduser().resolve()
    hf_cache_root.mkdir(parents=True, exist_ok=True)
    extracted_session_cache_root = Path(
        str(recipe.get("extracted_session_cache_root") or "data/.cache/extracted_sessions")
    ).expanduser().resolve()
    mixed_base_cache_root = Path(
        str(recipe.get("mixed_base_cache_root") or "data/.cache/mixed_base")
    ).expanduser().resolve()
    metrics_log_path = Path(
        str(recipe.get("metrics_log_path") or (base_output_root / "stage_metrics.jsonl"))
    ).expanduser().resolve()
    stage_logger = StageMetricsLogger(metrics_log_path)
    diarization_model = str(
        recipe.get("diarization_model") or "pyannote/speaker-diarization-community-1"
    )
    speaker_mapping_path = Path(str(recipe["speaker_mapping"])).expanduser().resolve()
    speaker_mapping = _load_yaml_or_json(str(speaker_mapping_path)) or {}
    base_config_path = (
        Path(str(recipe["base_config"])).expanduser().resolve()
        if recipe.get("base_config")
        else None
    )
    base_config = _load_yaml_or_json(str(base_config_path)) if base_config_path else {}
    eval_specs = _parse_eval_specs(recipe.get("eval") or [])
    if not eval_specs:
        raise RuntimeError("Baseline prep requires at least one eval session")

    excluded_stems = [spec.session_zip.stem for spec in eval_specs]
    training_sources = _collect_training_sources(
        [Path(str(path)).expanduser().resolve() for path in (recipe.get("training_inputs") or [])],
        excluded_stems=excluded_stems,
    )
    if not training_sources:
        raise RuntimeError("No training session sources were found for baseline prep")
    transcript_roots = [
        Path(str(path)).expanduser().resolve() for path in (recipe.get("transcript_roots") or [])
    ]
    recipe_root = base_output_root / "materialized_inputs"
    training_input_dir = _materialize_training_dir(training_sources, recipe_root / "train_inputs")
    input_identities = collect_input_file_identities(training_sources + transcript_roots)
    eval_input_identities = collect_input_file_identities(
        [spec.session_zip for spec in eval_specs] + [spec.transcript for spec in eval_specs],
        hash_contents=False,
    )
    core_speakers = list(recipe.get("core_speakers") or [])
    excluded_speakers = list(recipe.get("excluded_speakers") or [])
    git_commit = current_git_commit(cwd=Path(__file__).resolve().parents[2])
    device = str(recipe.get("device") or "cuda")
    quiet = bool(recipe.get("quiet", True))
    batch_size = int(recipe.get("batch_size") or 64)
    workers = int(recipe.get("workers") or 4)
    stage_manifests: Dict[str, str] = {}

    def _maybe_reuse_stage(
        stage: str,
        signature: Mapping[str, object],
    ) -> Tuple[bool, Dict[str, object]]:
        manifest_path = _stage_manifest_path(base_output_root, stage)
        if not resume_requested:
            return False, {}
        reusable, manifest, reason = stage_manifest_is_reusable(
            manifest_path,
            stage=stage,
            stage_signature=signature,
        )
        if not reusable:
            if stage not in selected_stage_names:
                raise RuntimeError(
                    f"Stage {stage} is required but cannot be resumed ({reason}) from {manifest_path}"
                )
            return False, {}
        stage_manifests[stage] = str(manifest_path)
        stage_logger.log(stage=stage, status="stage_reused", cache_hit=True, extra={"reason": reason})
        return True, dict((manifest or {}).get("outputs") or {})

    bank_params = {
        "training_mode": "clean",
        "window_size": 15.0,
        "window_stride": 15.0,
        "min_segment_dur": 6.0,
        "max_segment_dur": 20.0,
        "clean_max_records_per_speaker_per_session": 24,
        "augmentation_profile": "none",
        "augmentation_copies": 0,
        "include_base_samples": True,
    }
    bank_signature = {
        "diarization_model": diarization_model,
        "input_identities": input_identities,
        "core_speakers": core_speakers,
        "excluded_speakers": excluded_speakers,
        "params": bank_params,
    }
    bank_reused, bank_outputs = _maybe_reuse_stage("bank", bank_signature)
    if not bank_reused:
        if "bank" not in selected_stage_names:
            raise RuntimeError("Bank stage outputs are unavailable and bank is outside the requested stage range")
        stage_logger.log(stage="bank", status="stage_started", cache_hit=False)
        bank_manifest = build_artifact_manifest(
            artifact_type="bank",
            diarization_model=diarization_model,
            source_sessions=[path.stem for path in training_sources],
            input_file_identities=input_identities,
            build_params=bank_params,
            parent_artifacts=[],
            git_commit=git_commit,
        )
        bank_artifact_dir = base_output_root / "artifacts" / "bank" / str(bank_manifest["artifact_id"])
        bank_manifest_path = bank_artifact_dir / "bank_manifest.json"
        bank_profile_name = _bank_profile_name(
            str(recipe.get("bank_profile_prefix") or "baseline_bank"),
            str(bank_manifest["artifact_id"]),
        )
        bank_profile_dir = hf_cache_root / "speaker_bank" / bank_profile_name
        bank_dataset_dir = bank_artifact_dir / "dataset"
        if bank_profile_dir.exists():
            shutil.rmtree(bank_profile_dir)
        built = build_classifier_dataset_from_multitrack(
            input_path=str(training_input_dir),
            dataset_cache_dir=bank_dataset_dir,
            speaker_mapping=speaker_mapping,
            hf_token=None,
            force_device=device,
            quiet=quiet,
            top_k=0,
            hop_seconds=120.0,
            min_speakers=4,
            min_share=0.80,
            min_power=2e-4,
            min_segment_dur=6.0,
            max_segment_dur=20.0,
            max_samples_per_speaker=0,
            batch_size=batch_size,
            workers=workers,
            window_seconds=300.0,
            transcript_search_roots=transcript_roots,
            speaker_aliases={"zariel torgan": "David Tanglethorn"},
            extra_input_paths=[],
            allowed_speakers=core_speakers or None,
            excluded_speakers=excluded_speakers or None,
            training_mode="clean",
            clean_max_records_per_speaker_per_session=24,
            clean_window_size=15.0,
            clean_window_stride=15.0,
            augmentation_profile="none",
            augmentation_copies=0,
            include_base_samples=True,
            diarization_model_name=diarization_model,
            reuse_cached_dataset=False,
            extracted_session_cache_root=extracted_session_cache_root,
            progress_callback=stage_logger.bind(stage="bank"),
        )
        if built is None:
            raise RuntimeError("Failed to build clean bank dataset")
        bank_dataset, bank_dataset_summary = built
        bank_root = hf_cache_root / "speaker_bank"
        bank = SpeakerBank(bank_root, profile=bank_profile_name)
        bank.extend(
            [
                (
                    speaker,
                    bank_dataset.embeddings[index],
                    bank_dataset.sessions[index],
                    {
                        "mode": "baseline_clean_bank",
                        "source": bank_dataset.sources[index],
                        "duration": float(bank_dataset.durations[index]),
                    },
                )
                for index, speaker in enumerate(bank_dataset.labels)
            ]
        )
        bank.save()
        coverage_report_path = _json_write(
            bank_artifact_dir / "coverage_report.json",
            build_coverage_report(bank_dataset),
        )
        coverage_table_path = _json_write(
            bank_artifact_dir / "coverage_table.json",
            build_source_session_speaker_breakdown(bank_dataset),
        )
        pca_path = bank.render_pca(bank_artifact_dir / "bank_pca.png")
        dataset_summary_path = _write_augmented_dataset_summary(
            bank_dataset_dir,
            dataset_summary=bank_dataset_summary,
            artifact_id=str(bank_manifest["artifact_id"]),
            parent_artifacts=[],
            quality_filters=bank_dataset_summary.get("quality_filters"),  # type: ignore[arg-type]
            source_groups=bank_dataset_summary.get("source_groups"),  # type: ignore[arg-type]
            extra={
                "base_artifact_id": None,
                "materialization_mode": "clean_build",
                "cache_hits": dict(bank_dataset_summary.get("cache_hits") or {}),
                "stage_dependencies": [],
            },
        )
        bank_manifest = {
            **bank_manifest,
            "artifact_dir": str(bank_artifact_dir),
            "dataset_dir": str(bank_dataset_dir),
            "profile_dir": str(bank_profile_dir),
            "coverage_report_path": str(coverage_report_path),
            "coverage_table_path": str(coverage_table_path),
            "dataset_summary_path": str(dataset_summary_path),
            "quality_report_path": str(bank_dataset_dir / "quality_report.json"),
            "pca_artifact_path": str(pca_path) if pca_path else None,
        }
        save_manifest(bank_manifest_path, bank_manifest)
        bank_outputs = {
            "bank_manifest_path": str(bank_manifest_path),
            "bank_profile_dir": str(bank_profile_dir),
            "bank_profile_name": bank_profile_name,
            "bank_artifact_dir": str(bank_artifact_dir),
            "bank_dataset_dir": str(bank_dataset_dir),
        }
        stage_manifests["bank"] = str(
            _save_stage(
                base_output_root=base_output_root,
                stage="bank",
                stage_signature=bank_signature,
                outputs=bank_outputs,
                required_paths=[
                    bank_manifest_path,
                    bank_dataset_dir / "dataset.npz",
                    bank_profile_dir / "bank.json",
                    bank_profile_dir / "embeddings.npy",
                ],
                parent_stages=[],
                git_commit=git_commit,
            )
        )
        stage_logger.log(
            stage="bank",
            status="stage_completed",
            cache_hit=False,
            extra={"artifact_id": bank_manifest["artifact_id"]},
        )

    bank_manifest_path = Path(str(bank_outputs["bank_manifest_path"]))
    bank_manifest = load_manifest(bank_manifest_path)
    if bank_manifest is None:
        raise RuntimeError(f"Missing bank manifest: {bank_manifest_path}")
    bank_profile_dir = Path(str(bank_outputs["bank_profile_dir"]))
    bank_profile_name = str(bank_outputs["bank_profile_name"])
    bank_artifact_dir = Path(str(bank_outputs["bank_artifact_dir"]))
    bank_dataset_dir = Path(str(bank_outputs["bank_dataset_dir"]))
    bank_dataset_built = build_classifier_dataset_from_bank(
        profile_dir=bank_profile_dir,
        speaker_mapping=speaker_mapping,
        speaker_aliases={"zariel torgan": "David Tanglethorn"},
        allowed_speakers=core_speakers or None,
        excluded_speakers=excluded_speakers or None,
    )
    if bank_dataset_built is None:
        raise RuntimeError("Failed to load rebuilt bank dataset")
    bank_dataset, bank_dataset_summary = bank_dataset_built

    selected_pack = list(dict.fromkeys(recipe.get("baseline_pack") or list(DEFAULT_BASELINE_PACK)))
    build_variants = list(
        dict.fromkeys(
            recipe.get("build_variants")
            or [
                "mixed_raw",
                *selected_pack,
            ]
        )
    )
    for variant_name in build_variants:
        if variant_name not in DEFAULT_VARIANTS:
            raise RuntimeError(f"Unknown dataset variant requested: {variant_name}")
    candidate_variant_names = list(
        dict.fromkeys(
            recipe.get("hard_negative_candidate_variants")
            or list(DEFAULT_HARD_NEGATIVE_CANDIDATE_VARIANTS)
        )
    )

    mixed_base_params = {
        "training_mode": "mixed",
        "top_k": 8,
        "min_speakers": 4,
        "min_share": 0.80,
        "min_segment_dur": 0.8,
        "max_segment_dur": 15.0,
        "window_seconds": 300.0,
        "hop_seconds": 120.0,
        "augmentation_profile": "none",
        "augmentation_copies": 0,
        "include_base_samples": True,
    }
    mixed_base_signature = {
        "diarization_model": diarization_model,
        "input_identities": input_identities,
        "core_speakers": core_speakers,
        "excluded_speakers": excluded_speakers,
        "params": mixed_base_params,
    }
    mixed_base_reused, mixed_base_outputs = _maybe_reuse_stage("mixed_base", mixed_base_signature)
    if not mixed_base_reused:
        if "mixed_base" not in selected_stage_names:
            raise RuntimeError(
                "Mixed-base outputs are unavailable and mixed_base is outside the requested stage range"
            )
        stage_logger.log(stage="mixed_base", status="stage_started", cache_hit=False)
        mixed_base_manifest = build_artifact_manifest(
            artifact_type="dataset",
            diarization_model=diarization_model,
            source_sessions=[path.stem for path in training_sources],
            input_file_identities=input_identities,
            build_params={"variant": "mixed_base", **mixed_base_params},
            parent_artifacts=[],
            git_commit=git_commit,
        )
        mixed_base_dir = (
            base_output_root
            / "artifacts"
            / "datasets"
            / "mixed_base"
            / str(mixed_base_manifest["artifact_id"])
        )
        built = build_classifier_dataset_from_multitrack(
            input_path=str(training_input_dir),
            dataset_cache_dir=mixed_base_dir,
            speaker_mapping=speaker_mapping,
            hf_token=None,
            force_device=device,
            quiet=quiet,
            top_k=8,
            hop_seconds=120.0,
            min_speakers=4,
            min_share=0.80,
            min_power=2e-4,
            min_segment_dur=0.8,
            max_segment_dur=15.0,
            max_samples_per_speaker=0,
            batch_size=batch_size,
            workers=workers,
            window_seconds=300.0,
            transcript_search_roots=transcript_roots,
            speaker_aliases={"zariel torgan": "David Tanglethorn"},
            extra_input_paths=[],
            allowed_speakers=core_speakers or None,
            excluded_speakers=excluded_speakers or None,
            training_mode="mixed",
            augmentation_profile="none",
            augmentation_copies=0,
            augmentation_seed=13,
            include_base_samples=True,
            diarization_model_name=diarization_model,
            reuse_cached_dataset=False,
            extracted_session_cache_root=extracted_session_cache_root,
            window_cache_root=mixed_base_cache_root / "window_cache",
            progress_callback=stage_logger.bind(stage="mixed_base"),
        )
        if built is None:
            raise RuntimeError("Failed to build mixed-base dataset")
        mixed_base_dataset, mixed_base_summary = built
        mixed_base_coverage_report_path = _json_write(
            mixed_base_dir / "coverage_report.json",
            build_coverage_report(mixed_base_dataset),
        )
        mixed_base_summary_path = _write_augmented_dataset_summary(
            mixed_base_dir,
            dataset_summary=mixed_base_summary,
            artifact_id=str(mixed_base_manifest["artifact_id"]),
            parent_artifacts=[],
            quality_filters=mixed_base_summary.get("quality_filters"),  # type: ignore[arg-type]
            source_groups=mixed_base_summary.get("source_groups"),  # type: ignore[arg-type]
            extra={
                "base_artifact_id": None,
                "materialization_mode": "mixed_base_build",
                "cache_hits": dict(mixed_base_summary.get("cache_hits") or {}),
                "stage_dependencies": [],
            },
        )
        mixed_base_manifest = {
            **mixed_base_manifest,
            "artifact_dir": str(mixed_base_dir),
            "dataset_dir": str(mixed_base_dir),
            "coverage_report_path": str(mixed_base_coverage_report_path),
            "dataset_summary_path": str(mixed_base_summary_path),
            "quality_report_path": str(mixed_base_dir / "quality_report.json"),
            "mixed_base_cache_root": str(mixed_base_cache_root),
        }
        mixed_base_manifest_path = mixed_base_dir / "dataset_manifest.json"
        save_manifest(mixed_base_manifest_path, mixed_base_manifest)
        mixed_base_outputs = {
            "mixed_base_manifest_path": str(mixed_base_manifest_path),
            "mixed_base_dir": str(mixed_base_dir),
        }
        stage_manifests["mixed_base"] = str(
            _save_stage(
                base_output_root=base_output_root,
                stage="mixed_base",
                stage_signature=mixed_base_signature,
                outputs=mixed_base_outputs,
                required_paths=[
                    mixed_base_manifest_path,
                    mixed_base_dir / "dataset.npz",
                    mixed_base_dir / "prepared_windows.jsonl",
                ],
                parent_stages=[],
                git_commit=git_commit,
            )
        )
        stage_logger.log(
            stage="mixed_base",
            status="stage_completed",
            cache_hit=False,
            extra={"artifact_id": mixed_base_manifest["artifact_id"]},
        )

    mixed_base_manifest_path = Path(str(mixed_base_outputs["mixed_base_manifest_path"]))
    mixed_base_manifest = load_manifest(mixed_base_manifest_path)
    if mixed_base_manifest is None:
        raise RuntimeError(f"Missing mixed-base manifest: {mixed_base_manifest_path}")
    mixed_base_dir = Path(str(mixed_base_outputs["mixed_base_dir"]))
    mixed_base_dataset, mixed_base_summary = load_classifier_dataset(mixed_base_dir)

    variants_signature = {
        "bank_artifact_id": bank_manifest["artifact_id"],
        "mixed_base_artifact_id": mixed_base_manifest["artifact_id"],
        "selected_pack": selected_pack,
        "build_variants": build_variants,
        "candidate_variants": candidate_variant_names,
    }
    variants_reused, variants_outputs = _maybe_reuse_stage("variants", variants_signature)
    variant_manifests: Dict[str, Dict[str, object]] = {}
    variant_datasets: Dict[str, Any] = {}
    variant_summaries: Dict[str, Dict[str, object]] = {}
    if not variants_reused:
        if "variants" not in selected_stage_names:
            raise RuntimeError(
                "Variant outputs are unavailable and variants is outside the requested stage range"
            )
        stage_logger.log(stage="variants", status="stage_started", cache_hit=False)
        variant_manifest_paths: Dict[str, str] = {
            "mixed_raw": str(mixed_base_manifest_path),
        }
        variant_dataset_dirs: Dict[str, str] = {
            "mixed_raw": str(mixed_base_dir),
        }
        variant_manifests["mixed_raw"] = mixed_base_manifest
        variant_datasets["mixed_raw"] = mixed_base_dataset
        variant_summaries["mixed_raw"] = dict(mixed_base_summary)
        for variant_name in build_variants:
            if variant_name == "mixed_raw":
                continue
            variant_config = DEFAULT_VARIANTS[variant_name]
            build_params = {
                "training_mode": "mixed",
                "top_k": 8,
                "min_speakers": 4,
                "min_share": 0.80,
                "min_segment_dur": 0.8,
                "max_segment_dur": 15.0,
                "window_seconds": 300.0,
                "hop_seconds": 120.0,
                **variant_config,
            }
            manifest = build_artifact_manifest(
                artifact_type="dataset",
                diarization_model=diarization_model,
                source_sessions=[path.stem for path in training_sources],
                input_file_identities=input_identities,
                build_params={"variant": variant_name, **build_params},
                parent_artifacts=[str(mixed_base_manifest["artifact_id"])],
                git_commit=git_commit,
            )
            variant_dir = (
                base_output_root / "artifacts" / "datasets" / variant_name / str(manifest["artifact_id"])
            )
            dataset, dataset_summary = materialize_classifier_dataset_from_mixed_base(
                mixed_base_dir=mixed_base_dir,
                dataset_cache_dir=variant_dir,
                hf_token=None,
                force_device=device,
                quiet=quiet,
                batch_size=batch_size,
                workers=workers,
                augmentation_profile=str(variant_config["augmentation_profile"]),
                augmentation_copies=int(variant_config["augmentation_copies"]),
                augmentation_seed=13,
                include_base_samples=bool(variant_config["include_base_samples"]),
                max_samples_per_speaker=0,
                diarization_model_name=diarization_model,
                reuse_cached_dataset=resume_requested,
                progress_callback=stage_logger.bind(stage="variants", variant=variant_name),
            )
            coverage_report_path = _json_write(
                variant_dir / "coverage_report.json",
                build_coverage_report(dataset),
            )
            summary_path = _write_augmented_dataset_summary(
                variant_dir,
                dataset_summary=dataset_summary,
                artifact_id=str(manifest["artifact_id"]),
                parent_artifacts=[str(mixed_base_manifest["artifact_id"])],
                quality_filters=dataset_summary.get("quality_filters"),  # type: ignore[arg-type]
                source_groups=dataset_summary.get("source_groups"),  # type: ignore[arg-type]
                extra={
                    "base_artifact_id": str(mixed_base_manifest["artifact_id"]),
                    "materialization_mode": str(dataset_summary.get("materialization_mode") or "mixed_base_derived"),
                    "cache_hits": dict(dataset_summary.get("cache_hits") or {}),
                    "stage_dependencies": ["mixed_base"],
                },
            )
            manifest = {
                **manifest,
                "artifact_dir": str(variant_dir),
                "dataset_dir": str(variant_dir),
                "coverage_report_path": str(coverage_report_path),
                "dataset_summary_path": str(summary_path),
                "quality_report_path": str(variant_dir / "quality_report.json"),
                "base_artifact_id": str(mixed_base_manifest["artifact_id"]),
            }
            manifest_path = variant_dir / "dataset_manifest.json"
            save_manifest(manifest_path, manifest)
            variant_manifest_paths[variant_name] = str(manifest_path)
            variant_dataset_dirs[variant_name] = str(variant_dir)
            variant_manifests[variant_name] = manifest
            variant_datasets[variant_name] = dataset
            variant_summaries[variant_name] = dict(dataset_summary)

        if "mixed_raw" not in variant_datasets:
            raise RuntimeError("Baseline prep requires the mixed_raw dataset variant")
        missing_pack_variants = [name for name in selected_pack if name not in variant_datasets]
        if missing_pack_variants:
            raise RuntimeError(
                "Missing dataset variants required by baseline pack: "
                + ", ".join(sorted(missing_pack_variants))
            )
        missing_candidate_variants = [name for name in candidate_variant_names if name not in variant_datasets]
        if missing_candidate_variants:
            raise RuntimeError(
                "Missing dataset variants required for hard-negative mining: "
                + ", ".join(sorted(missing_candidate_variants))
            )
        merged_datasets = [
            relabel_classifier_dataset_sources(bank_dataset, "bank"),
            variant_datasets["mixed_raw"],
        ]
        for variant_name in selected_pack:
            if variant_name == "mixed_raw":
                continue
            merged_datasets.append(
                relabel_classifier_dataset_sources(variant_datasets[variant_name], "mixed_aug_total")
            )
        merged_dataset = merge_classifier_datasets(merged_datasets)
        base_training_dataset, base_balance_summary = balance_classifier_dataset(
            merged_dataset,
            target_speakers=core_speakers or sorted(set(merged_dataset.labels)),
            max_samples_per_cell=500,
        )
        base_quality_filters = dict(
            variant_summaries.get("mixed_raw", {}).get("quality_filters") or DEFAULT_QUALITY_FILTERS
        )
        base_training_manifest = build_artifact_manifest(
            artifact_type="dataset",
            diarization_model=diarization_model,
            source_sessions=[path.stem for path in training_sources],
            input_file_identities=input_identities,
            build_params={
                "variant": "baseline_pack",
                "pack": selected_pack,
                "source_groups": base_balance_summary.get("source_groups") or {},
            },
            parent_artifacts=[str(bank_manifest["artifact_id"])]
            + [str(variant_manifests[name]["artifact_id"]) for name in selected_pack if name in variant_manifests],
            git_commit=git_commit,
        )
        base_training_dir = (
            base_output_root
            / "artifacts"
            / "datasets"
            / "baseline_pack"
            / str(base_training_manifest["artifact_id"])
        )
        save_classifier_dataset(
            base_training_dir,
            base_training_dataset,
            summary={
                "artifact_id": str(base_training_manifest["artifact_id"]),
                "parent_artifacts": base_training_manifest["parent_artifacts"],
                "balance": base_balance_summary,
                "quality_filters": base_quality_filters,
                "source_groups": base_balance_summary.get("source_groups") or {},
                "breakdown": build_source_session_speaker_breakdown(base_training_dataset),
                "base_artifact_id": str(mixed_base_manifest["artifact_id"]),
                "materialization_mode": "balanced_pack",
                "cache_hits": {},
                "stage_dependencies": ["bank", "mixed_base", "variants"],
            },
        )
        base_coverage_report_path = _json_write(
            base_training_dir / "coverage_report.json",
            build_coverage_report(base_training_dataset),
        )
        base_training_manifest = {
            **base_training_manifest,
            "artifact_dir": str(base_training_dir),
            "dataset_dir": str(base_training_dir),
            "coverage_report_path": str(base_coverage_report_path),
            "dataset_summary_path": str(base_training_dir / "dataset_summary.json"),
            "quality_report_path": None,
        }
        base_training_manifest_path = base_training_dir / "dataset_manifest.json"
        save_manifest(base_training_manifest_path, base_training_manifest)
        variants_outputs = {
            "variant_manifest_paths": variant_manifest_paths,
            "variant_dataset_dirs": variant_dataset_dirs,
            "base_training_manifest_path": str(base_training_manifest_path),
            "base_training_dir": str(base_training_dir),
        }
        stage_manifests["variants"] = str(
            _save_stage(
                base_output_root=base_output_root,
                stage="variants",
                stage_signature=variants_signature,
                outputs=variants_outputs,
                required_paths=[
                    base_training_manifest_path,
                    *[Path(path) for path in variant_manifest_paths.values()],
                ],
                parent_stages=["bank", "mixed_base"],
                git_commit=git_commit,
            )
        )
        stage_logger.log(
            stage="variants",
            status="stage_completed",
            cache_hit=False,
            extra={"variants": sorted(variant_manifest_paths)},
        )
    else:
        for name, manifest_path_str in dict(variants_outputs.get("variant_manifest_paths") or {}).items():
            manifest = load_manifest(Path(str(manifest_path_str)))
            if manifest is None:
                raise RuntimeError(f"Missing variant manifest: {manifest_path_str}")
            dataset_dir = Path(str((variants_outputs.get("variant_dataset_dirs") or {}).get(name) or manifest["dataset_dir"]))
            dataset, summary = load_classifier_dataset(dataset_dir)
            variant_manifests[name] = manifest
            variant_datasets[name] = dataset
            variant_summaries[name] = summary
        base_training_manifest_path = Path(str(variants_outputs["base_training_manifest_path"]))
        base_training_manifest = load_manifest(base_training_manifest_path)
        if base_training_manifest is None:
            raise RuntimeError(f"Missing base training manifest: {base_training_manifest_path}")
        base_training_dir = Path(str(variants_outputs["base_training_dir"]))
        base_training_dataset, base_training_summary = load_classifier_dataset(base_training_dir)
        base_quality_filters = dict(base_training_summary.get("quality_filters") or DEFAULT_QUALITY_FILTERS)
        base_balance_summary = dict(base_training_summary.get("balance") or {})

    if not variants_reused:
        base_training_manifest_path = Path(str(variants_outputs["base_training_manifest_path"]))
        base_training_manifest = load_manifest(base_training_manifest_path)
        if base_training_manifest is None:
            raise RuntimeError(f"Missing base training manifest: {base_training_manifest_path}")
        base_training_dir = Path(str(variants_outputs["base_training_dir"]))
    base_training_dataset, base_training_summary = load_classifier_dataset(base_training_dir)
    base_quality_filters = dict(base_training_summary.get("quality_filters") or DEFAULT_QUALITY_FILTERS)
    base_balance_summary = dict(base_training_summary.get("balance") or {})

    current_winner = {**DEFAULT_CURRENT_WINNER, **dict(recipe.get("current_winner") or {})}
    eval_window_seconds = float(recipe.get("eval_window_seconds") or 300.0)
    eval_hop_seconds = float(recipe.get("eval_hop_seconds") or 60.0)
    eval_top_k = int(recipe.get("eval_top_k") or 3)
    eval_min_speakers = int(recipe.get("eval_min_speakers") or 3)
    eval_params = {
        "window_seconds": eval_window_seconds,
        "hop_seconds": eval_hop_seconds,
        "top_k": eval_top_k,
        "min_speakers": eval_min_speakers,
        "threshold": float(current_winner["threshold"]),
        "classifier_min_margin": float(current_winner["classifier_min_margin"]),
        "min_segments_per_label": int(current_winner["min_segments_per_label"]),
    }
    short_spec, short_window = _derive_short_segment_slice(
        eval_specs,
        speaker_mapping=speaker_mapping,
        window_seconds=eval_window_seconds,
        hop_seconds=eval_hop_seconds,
        top_k=eval_top_k,
        min_speakers=eval_min_speakers,
    )
    canonical_suite = {
        "sessions": [spec.name for spec in eval_specs],
        "short_segment_slice": {
            "source_session": short_spec.name,
            "window": short_window,
        },
    }
    prepared_eval_root = base_output_root / "prepared_eval"

    hard_negative_signature = {
        "base_training_artifact_id": base_training_manifest["artifact_id"],
        "candidate_variant_ids": [
            str(variant_manifests[name]["artifact_id"]) for name in candidate_variant_names if name in variant_manifests
        ],
        "current_winner": current_winner,
        "eval_params": eval_params,
        "canonical_suite": canonical_suite,
        "seed_pairs": list(recipe.get("seed_confusion_pairs") or [["Cyrus Schwert", "Cletus Cobbington"]]),
        "top_confusion_pairs": int(recipe.get("top_confusion_pairs") or 5),
        "hard_negative_max_margin": float(recipe.get("hard_negative_max_margin") or 0.12),
        "hard_negative_min_dominant_share": float(recipe.get("hard_negative_min_dominant_share") or 0.55),
        "hard_negative_per_pair_cap": int(recipe.get("hard_negative_per_pair_cap") or 75),
        "hard_negative_max_fraction": float(recipe.get("hard_negative_max_fraction") or 0.20),
    }
    hard_negative_reused, hard_negative_outputs = _maybe_reuse_stage("hard_negatives", hard_negative_signature)
    if not hard_negative_reused:
        if "hard_negatives" not in selected_stage_names:
            raise RuntimeError(
                "Hard-negative outputs are unavailable and hard_negatives is outside the requested stage range"
            )
        stage_logger.log(stage="hard_negatives", status="stage_started", cache_hit=False)
        mining_profile_name = _bank_profile_name(
            str(recipe.get("mining_profile_prefix") or "baseline_mining_profile"),
            str(base_training_manifest["artifact_id"]),
        )
        mining_profile_dir = hf_cache_root / "speaker_bank" / mining_profile_name
        if mining_profile_dir.exists():
            shutil.rmtree(mining_profile_dir)
        shutil.copytree(bank_profile_dir, mining_profile_dir)
        mining_training_summary = train_segment_classifier_from_dataset(
            dataset=base_training_dataset,
            profile_dir=mining_profile_dir,
            model_name=str(current_winner["model_name"]),
            classifier_c=float(current_winner["classifier_c"]),
            classifier_n_neighbors=int(current_winner["classifier_n_neighbors"]),
            base_summary={
                "artifact_id": str(base_training_manifest["artifact_id"]),
                "parent_artifacts": base_training_manifest["parent_artifacts"],
            },
        )
        mining_training_summary_path = _write_training_summary(mining_profile_dir, mining_training_summary)
        mining_eval_manifest = build_artifact_manifest(
            artifact_type="eval",
            diarization_model=diarization_model,
            source_sessions=[spec.name for spec in eval_specs],
            input_file_identities=eval_input_identities,
            build_params={
                **eval_params,
                "stage": "hard_negative_mining",
                "canonical_suite": canonical_suite,
            },
            parent_artifacts=[str(base_training_manifest["artifact_id"])],
            git_commit=git_commit,
        )
        mining_eval_artifact_dir = (
            base_output_root
            / "artifacts"
            / "eval"
            / "hard_negative_mining"
            / str(mining_eval_manifest["artifact_id"])
        )
        mining_eval_config_path = _write_eval_config(
            mining_eval_artifact_dir / "baseline_eval_config.json",
            base_config=base_config,
            hf_cache_root=hf_cache_root,
            profile_name=mining_profile_name,
            diarization_model=diarization_model,
            speaker_mapping_path=speaker_mapping_path,
            classifier_min_margin=float(current_winner["classifier_min_margin"]),
            threshold=float(current_winner["threshold"]),
            min_segments_per_label=int(current_winner["min_segments_per_label"]),
        )
        mining_eval: Dict[str, object] = {}
        mining_eval_summaries: List[Dict[str, object]] = []
        for spec in eval_specs:
            summary = evaluate_multitrack_session(
                session_zip=spec.session_zip,
                session_jsonl=spec.transcript,
                output_dir=mining_eval_artifact_dir / spec.name,
                cache_root=prepared_eval_root / spec.name,
                speaker_mapping_path=speaker_mapping_path,
                config_path=mining_eval_config_path,
                window_seconds=eval_window_seconds,
                hop_seconds=eval_hop_seconds,
                top_k=eval_top_k,
                min_speakers=eval_min_speakers,
                device_override=device,
                local_files_only_override=True if recipe.get("local_files_only") else None,
            )
            mining_eval[spec.name] = _summarize_eval(summary)
            mining_eval_summaries.append(summary)
        short_summary = evaluate_multitrack_session(
            session_zip=short_spec.session_zip,
            session_jsonl=short_spec.transcript,
            output_dir=mining_eval_artifact_dir / "short_segment_slice",
            cache_root=prepared_eval_root / "short_segment_slice",
            speaker_mapping_path=speaker_mapping_path,
            config_path=mining_eval_config_path,
            window_seconds=eval_window_seconds,
            hop_seconds=eval_hop_seconds,
            top_k=1,
            min_speakers=eval_min_speakers,
            device_override=device,
            local_files_only_override=True if recipe.get("local_files_only") else None,
            windows_override=[short_window],
        )
        mining_eval["short_segment_slice"] = {
            **_summarize_eval(short_summary),
            "source_session": short_spec.name,
            "window": short_window,
        }
        mining_eval_summaries.append(short_summary)
        mining_eval_manifest = {
            **mining_eval_manifest,
            "artifact_dir": str(mining_eval_artifact_dir),
            "config_path": str(mining_eval_config_path),
            "canonical_suite": canonical_suite,
            "canonical_eval": mining_eval,
            "summary_paths": [
                str(summary.get("summary_path") or "") for summary in mining_eval_summaries
            ],
        }
        mining_eval_manifest_path = mining_eval_artifact_dir / "eval_manifest.json"
        save_manifest(mining_eval_manifest_path, mining_eval_manifest)
        hard_negative_dataset, hard_negative_records, hard_negative_summary = build_hard_negative_dataset(
            eval_summaries=mining_eval_summaries,
            candidate_pool_dirs=[
                Path(str(variant_manifests[name]["dataset_dir"])) for name in candidate_variant_names
            ],
            base_dataset_samples=base_training_dataset.samples,
            seed_pairs=list(
                recipe.get("seed_confusion_pairs") or [["Cyrus Schwert", "Cletus Cobbington"]]
            ),
            top_confusion_pairs=int(recipe.get("top_confusion_pairs") or 5),
            max_eval_margin=float(recipe.get("hard_negative_max_margin") or 0.12),
            min_mixed_dominant_share=float(recipe.get("hard_negative_min_dominant_share") or 0.55),
            per_pair_cap=int(recipe.get("hard_negative_per_pair_cap") or 75),
            max_fraction=float(recipe.get("hard_negative_max_fraction") or 0.20),
        )
        hard_negative_manifest = None
        hard_negative_dir = base_output_root / "artifacts" / "datasets" / "hard_negative"
        if hard_negative_dataset is not None:
            hard_negative_manifest = build_artifact_manifest(
                artifact_type="dataset",
                diarization_model=diarization_model,
                source_sessions=sorted({str(item["source_session"]) for item in hard_negative_records}),
                input_file_identities=collect_input_file_identities(
                    [mining_eval_artifact_dir]
                    + [Path(str(variant_manifests[name]["dataset_dir"])) for name in candidate_variant_names],
                    hash_contents=False,
                ),
                build_params={
                    "variant": "hard_negative",
                    "tracked_pairs": hard_negative_summary.get("tracked_pairs"),
                    "candidate_variants": candidate_variant_names,
                },
                parent_artifacts=[
                    str(mining_eval_manifest["artifact_id"]),
                    *[
                        str(variant_manifests[name]["artifact_id"])
                        for name in candidate_variant_names
                        if name in variant_manifests
                    ],
                ],
                git_commit=git_commit,
            )
            hard_negative_dir = hard_negative_dir / str(hard_negative_manifest["artifact_id"])
            save_classifier_dataset(
                hard_negative_dir,
                hard_negative_dataset,
                summary={
                    "artifact_id": str(hard_negative_manifest["artifact_id"]),
                    "parent_artifacts": hard_negative_manifest["parent_artifacts"],
                    "hard_negative": hard_negative_summary,
                    "quality_filters": base_quality_filters,
                    "breakdown": build_source_session_speaker_breakdown(hard_negative_dataset),
                    "source_groups": {},
                    "base_artifact_id": str(base_training_manifest["artifact_id"]),
                    "materialization_mode": "hard_negative_supplement",
                    "cache_hits": {},
                    "stage_dependencies": ["hard_negatives"],
                },
            )
            _jsonl_write(hard_negative_dir / "hard_negative_records.jsonl", hard_negative_records)
            _json_write(
                hard_negative_dir / "coverage_report.json",
                build_coverage_report(hard_negative_dataset),
            )
            hard_negative_manifest = {
                **hard_negative_manifest,
                "artifact_dir": str(hard_negative_dir),
                "dataset_dir": str(hard_negative_dir),
                "records_path": str(hard_negative_dir / "hard_negative_records.jsonl"),
                "coverage_report_path": str(hard_negative_dir / "coverage_report.json"),
                "dataset_summary_path": str(hard_negative_dir / "dataset_summary.json"),
            }
            save_manifest(hard_negative_dir / "hard_negative_manifest.json", hard_negative_manifest)
        hard_negative_outputs = {
            "mining_profile_dir": str(mining_profile_dir),
            "mining_profile_name": mining_profile_name,
            "mining_training_summary_path": str(mining_training_summary_path),
            "mining_eval_manifest_path": str(mining_eval_manifest_path),
            "hard_negative_manifest_path": (
                str(hard_negative_dir / "hard_negative_manifest.json") if hard_negative_manifest else ""
            ),
            "hard_negative_dir": str(hard_negative_dir) if hard_negative_manifest else "",
        }
        stage_manifests["hard_negatives"] = str(
            _save_stage(
                base_output_root=base_output_root,
                stage="hard_negatives",
                stage_signature=hard_negative_signature,
                outputs=hard_negative_outputs,
                required_paths=[
                    mining_training_summary_path,
                    mining_eval_manifest_path,
                    *(
                        [hard_negative_dir / "hard_negative_manifest.json"]
                        if hard_negative_manifest
                        else []
                    ),
                ],
                parent_stages=["variants"],
                git_commit=git_commit,
            )
        )
        stage_logger.log(
            stage="hard_negatives",
            status="stage_completed",
            cache_hit=False,
            extra={"hard_negative_dataset": bool(hard_negative_outputs["hard_negative_manifest_path"])},
        )

    mining_profile_dir = Path(str(hard_negative_outputs["mining_profile_dir"]))
    mining_profile_name = str(hard_negative_outputs["mining_profile_name"])
    mining_training_summary_path = Path(str(hard_negative_outputs["mining_training_summary_path"]))
    mining_eval_manifest_path = Path(str(hard_negative_outputs["mining_eval_manifest_path"]))
    mining_eval_manifest = load_manifest(mining_eval_manifest_path)
    if mining_eval_manifest is None:
        raise RuntimeError(f"Missing mining eval manifest: {mining_eval_manifest_path}")
    mining_eval = dict(mining_eval_manifest.get("canonical_eval") or {})
    hard_negative_manifest_path_raw = str(hard_negative_outputs.get("hard_negative_manifest_path") or "")
    hard_negative_manifest = (
        load_manifest(Path(hard_negative_manifest_path_raw))
        if hard_negative_manifest_path_raw
        else None
    )
    hard_negative_dir = Path(str(hard_negative_outputs.get("hard_negative_dir") or "")) if hard_negative_manifest else base_output_root / "artifacts" / "datasets" / "hard_negative"
    if hard_negative_manifest is not None:
        hard_negative_dataset, hard_negative_dataset_summary = load_classifier_dataset(hard_negative_dir)
        hard_negative_summary = dict(hard_negative_dataset_summary.get("hard_negative") or {})
    else:
        hard_negative_dataset = None
        hard_negative_summary = {"tracked_pairs": [], "selected": 0}

    train_signature = {
        "bank_artifact_id": bank_manifest["artifact_id"],
        "base_training_artifact_id": base_training_manifest["artifact_id"],
        "hard_negative_artifact_id": hard_negative_manifest["artifact_id"] if hard_negative_manifest else None,
        "current_winner": current_winner,
    }
    train_reused, train_outputs = _maybe_reuse_stage("train", train_signature)
    if not train_reused:
        if "train" not in selected_stage_names:
            raise RuntimeError("Train outputs are unavailable and train is outside the requested stage range")
        stage_logger.log(stage="train", status="stage_started", cache_hit=False)
        final_training_dataset = append_dataset(base_training_dataset, hard_negative_dataset)
        final_training_manifest = build_artifact_manifest(
            artifact_type="dataset",
            diarization_model=diarization_model,
            source_sessions=[path.stem for path in training_sources],
            input_file_identities=input_identities,
            build_params={"variant": "final_training_dataset", "pack": selected_pack},
            parent_artifacts=[
                str(base_training_manifest["artifact_id"]),
                str(hard_negative_manifest["artifact_id"]) if hard_negative_manifest else "",
            ],
            git_commit=git_commit,
        )
        final_training_dir = (
            base_output_root
            / "artifacts"
            / "datasets"
            / "final_training"
            / str(final_training_manifest["artifact_id"])
        )
        save_classifier_dataset(
            final_training_dir,
            final_training_dataset,
            summary={
                "artifact_id": str(final_training_manifest["artifact_id"]),
                "parent_artifacts": [item for item in final_training_manifest["parent_artifacts"] if item],
                "quality_filters": base_quality_filters,
                "source_groups": base_balance_summary.get("source_groups") or {},
                "hard_negative": hard_negative_summary,
                "breakdown": build_source_session_speaker_breakdown(final_training_dataset),
                "base_artifact_id": str(base_training_manifest["artifact_id"]),
                "materialization_mode": "final_training_dataset",
                "cache_hits": {},
                "stage_dependencies": ["variants", "hard_negatives"],
            },
        )
        final_coverage_report_path = _json_write(
            final_training_dir / "coverage_report.json",
            build_coverage_report(final_training_dataset),
        )
        final_training_manifest = {
            **final_training_manifest,
            "artifact_dir": str(final_training_dir),
            "dataset_dir": str(final_training_dir),
            "coverage_report_path": str(final_coverage_report_path),
            "dataset_summary_path": str(final_training_dir / "dataset_summary.json"),
            "quality_report_path": None,
        }
        final_training_manifest_path = final_training_dir / "dataset_manifest.json"
        save_manifest(final_training_manifest_path, final_training_manifest)

        baseline_profile_name = _bank_profile_name(
            str(recipe.get("baseline_profile_prefix") or "baseline_profile"),
            str(final_training_manifest["artifact_id"]),
        )
        baseline_profile_dir = hf_cache_root / "speaker_bank" / baseline_profile_name
        if baseline_profile_dir.exists():
            shutil.rmtree(baseline_profile_dir)
        shutil.copytree(bank_profile_dir, baseline_profile_dir)
        final_training_summary = train_segment_classifier_from_dataset(
            dataset=final_training_dataset,
            profile_dir=baseline_profile_dir,
            model_name=str(current_winner["model_name"]),
            classifier_c=float(current_winner["classifier_c"]),
            classifier_n_neighbors=int(current_winner["classifier_n_neighbors"]),
            base_summary={
                "artifact_id": str(final_training_manifest["artifact_id"]),
                "parent_artifacts": final_training_manifest["parent_artifacts"],
            },
        )
        final_training_summary_path = _write_training_summary(baseline_profile_dir, final_training_summary)
        train_outputs = {
            "final_training_manifest_path": str(final_training_manifest_path),
            "final_training_dir": str(final_training_dir),
            "baseline_profile_dir": str(baseline_profile_dir),
            "baseline_profile_name": baseline_profile_name,
            "final_training_summary_path": str(final_training_summary_path),
        }
        stage_manifests["train"] = str(
            _save_stage(
                base_output_root=base_output_root,
                stage="train",
                stage_signature=train_signature,
                outputs=train_outputs,
                required_paths=[
                    final_training_manifest_path,
                    final_training_summary_path,
                    baseline_profile_dir / "segment_classifier.meta.json",
                ],
                parent_stages=["variants", "hard_negatives"],
                git_commit=git_commit,
            )
        )
        stage_logger.log(
            stage="train",
            status="stage_completed",
            cache_hit=False,
            extra={"artifact_id": final_training_manifest["artifact_id"]},
        )

    final_training_manifest_path = Path(str(train_outputs["final_training_manifest_path"]))
    final_training_manifest = load_manifest(final_training_manifest_path)
    if final_training_manifest is None:
        raise RuntimeError(f"Missing final training manifest: {final_training_manifest_path}")
    final_training_dir = Path(str(train_outputs["final_training_dir"]))
    baseline_profile_dir = Path(str(train_outputs["baseline_profile_dir"]))
    baseline_profile_name = str(train_outputs["baseline_profile_name"])
    final_training_summary_path = Path(str(train_outputs["final_training_summary_path"]))
    final_training_dataset, _final_training_dataset_summary = load_classifier_dataset(final_training_dir)
    final_coverage_report_path = final_training_dir / "coverage_report.json"

    eval_signature = {
        "final_training_artifact_id": final_training_manifest["artifact_id"],
        "eval_input_identities": eval_input_identities,
        "eval_params": eval_params,
        "canonical_suite": canonical_suite,
        "baseline_profile_name": baseline_profile_name,
    }
    eval_reused, eval_outputs = _maybe_reuse_stage("eval", eval_signature)
    if not eval_reused:
        if "eval" not in selected_stage_names:
            raise RuntimeError("Eval outputs are unavailable and eval is outside the requested stage range")
        stage_logger.log(stage="eval", status="stage_started", cache_hit=False)
        eval_manifest = build_artifact_manifest(
            artifact_type="eval",
            diarization_model=diarization_model,
            source_sessions=[spec.name for spec in eval_specs],
            input_file_identities=eval_input_identities,
            build_params={
                **eval_params,
                "stage": "canonical_baseline",
                "canonical_suite": canonical_suite,
            },
            parent_artifacts=[str(final_training_manifest["artifact_id"])],
            git_commit=git_commit,
        )
        eval_artifact_dir = (
            base_output_root
            / "artifacts"
            / "eval"
            / "canonical_baseline"
            / str(eval_manifest["artifact_id"])
        )
        eval_config_path = _write_eval_config(
            eval_artifact_dir / "baseline_eval_config.json",
            base_config=base_config,
            hf_cache_root=hf_cache_root,
            profile_name=baseline_profile_name,
            diarization_model=diarization_model,
            speaker_mapping_path=speaker_mapping_path,
            classifier_min_margin=float(current_winner["classifier_min_margin"]),
            threshold=float(current_winner["threshold"]),
            min_segments_per_label=int(current_winner["min_segments_per_label"]),
        )
        canonical_eval: Dict[str, object] = {}
        eval_summaries: List[Dict[str, object]] = []
        for spec in eval_specs:
            summary = evaluate_multitrack_session(
                session_zip=spec.session_zip,
                session_jsonl=spec.transcript,
                output_dir=eval_artifact_dir / spec.name,
                cache_root=prepared_eval_root / spec.name,
                speaker_mapping_path=speaker_mapping_path,
                config_path=eval_config_path,
                window_seconds=eval_window_seconds,
                hop_seconds=eval_hop_seconds,
                top_k=eval_top_k,
                min_speakers=eval_min_speakers,
                device_override=device,
                local_files_only_override=True if recipe.get("local_files_only") else None,
            )
            canonical_eval[spec.name] = _summarize_eval(summary)
            eval_summaries.append(summary)
        short_summary = evaluate_multitrack_session(
            session_zip=short_spec.session_zip,
            session_jsonl=short_spec.transcript,
            output_dir=eval_artifact_dir / "short_segment_slice",
            cache_root=prepared_eval_root / "short_segment_slice",
            speaker_mapping_path=speaker_mapping_path,
            config_path=eval_config_path,
            window_seconds=eval_window_seconds,
            hop_seconds=eval_hop_seconds,
            top_k=1,
            min_speakers=eval_min_speakers,
            device_override=device,
            local_files_only_override=True if recipe.get("local_files_only") else None,
            windows_override=[short_window],
        )
        canonical_eval["short_segment_slice"] = {
            **_summarize_eval(short_summary),
            "source_session": short_spec.name,
            "window": short_window,
        }
        eval_summaries.append(short_summary)
        eval_manifest = {
            **eval_manifest,
            "artifact_dir": str(eval_artifact_dir),
            "config_path": str(eval_config_path),
            "canonical_suite": canonical_suite,
            "canonical_eval": canonical_eval,
            "summary_paths": [str(summary.get("summary_path") or "") for summary in eval_summaries],
        }
        eval_manifest_path = eval_artifact_dir / "eval_manifest.json"
        save_manifest(eval_manifest_path, eval_manifest)
        narrow_doe_recipe = {
            "bank_profile_dir": str(bank_profile_dir),
            "mining_profile_dir": str(mining_profile_dir),
            "baseline_profile_dir": str(baseline_profile_dir),
            "training_dataset_dir": str(final_training_dir),
            "base_training_dataset_dir": str(base_training_dir),
            "hard_negative_dataset_dir": str(hard_negative_dir) if hard_negative_manifest else None,
            "model_family": str(current_winner["model_name"]),
            "classifier": current_winner,
            "selected_pack": selected_pack,
            "variant_dataset_dirs": {
                name: str(manifest["dataset_dir"]) for name, manifest in variant_manifests.items()
            },
        }
        narrow_doe_recipe_path = _json_write(
            base_output_root / "narrow_doe_recipe.json",
            narrow_doe_recipe,
        )
        baseline_summary = {
            "recipe_path": str(Path(recipe_path).expanduser().resolve()),
            "output_root": str(base_output_root),
            "resume": resume_requested,
            "selected_stages": selected_stage_names,
            "stage_manifests": dict(stage_manifests),
            "stage_metrics_path": str(metrics_log_path),
            "bank_manifest_path": str(bank_manifest_path),
            "mixed_base_manifest_path": str(mixed_base_manifest_path),
            "variant_manifests": {
                name: str(path)
                for name, path in dict(variants_outputs.get("variant_manifest_paths") or {}).items()
            },
            "base_training_manifest_path": str(base_training_manifest_path),
            "hard_negative_manifest_path": (
                str(hard_negative_dir / "hard_negative_manifest.json") if hard_negative_manifest else None
            ),
            "final_training_manifest_path": str(final_training_manifest_path),
            "mining_eval_manifest_path": str(mining_eval_manifest_path),
            "eval_manifest_path": str(eval_manifest_path),
            "mining_training_summary_path": str(mining_training_summary_path),
            "final_training_summary_path": str(final_training_summary_path),
            "mining_eval": mining_eval,
            "canonical_eval": canonical_eval,
            "quality_reports": {
                "bank": str(bank_artifact_dir / "dataset" / "quality_report.json"),
                "mixed_base": str(mixed_base_dir / "quality_report.json"),
                **{
                    name: str(Path(str(manifest["dataset_dir"])) / "quality_report.json")
                    for name, manifest in variant_manifests.items()
                    if name != "mixed_raw"
                },
            },
            "coverage_reports": {
                "bank": str(bank_artifact_dir / "coverage_report.json"),
                "mixed_base": str(mixed_base_dir / "coverage_report.json"),
                "base_training": str(base_training_dir / "coverage_report.json"),
                "final_training": str(final_coverage_report_path),
                **(
                    {"hard_negative": str(hard_negative_dir / "coverage_report.json")}
                    if hard_negative_manifest
                    else {}
                ),
                **{
                    name: str(Path(str(manifest["dataset_dir"])) / "coverage_report.json")
                    for name, manifest in variant_manifests.items()
                    if name != "mixed_raw"
                },
            },
            "narrow_doe_recipe_path": str(narrow_doe_recipe_path),
        }
        baseline_summary_path = _json_write(base_output_root / "baseline_summary.json", baseline_summary)
        eval_outputs = {
            "eval_manifest_path": str(eval_manifest_path),
            "narrow_doe_recipe_path": str(narrow_doe_recipe_path),
            "baseline_summary_path": str(baseline_summary_path),
        }
        stage_manifests["eval"] = str(
            _save_stage(
                base_output_root=base_output_root,
                stage="eval",
                stage_signature=eval_signature,
                outputs=eval_outputs,
                required_paths=[eval_manifest_path, narrow_doe_recipe_path, baseline_summary_path],
                parent_stages=["train"],
                git_commit=git_commit,
            )
        )
        stage_logger.log(
            stage="eval",
            status="stage_completed",
            cache_hit=False,
            extra={"eval_manifest_path": str(eval_manifest_path)},
        )

    baseline_summary_path = Path(str(eval_outputs["baseline_summary_path"]))
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_summary["stage_manifests"] = dict(stage_manifests)
    baseline_summary["stage_metrics_path"] = str(metrics_log_path)
    baseline_summary["smoke_profile_name"] = recipe.get("smoke_profile_name")
    _json_write(baseline_summary_path, baseline_summary)
    return baseline_summary
