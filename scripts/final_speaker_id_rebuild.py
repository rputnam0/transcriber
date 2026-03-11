from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.cli import _load_yaml_or_json  # noqa: E402
from transcriber.multitrack_eval import evaluate_multitrack_session  # noqa: E402
from transcriber.segment_classifier import (  # noqa: E402
    ClassifierDataset,
    _find_session_sources,
    balance_classifier_dataset,
    build_classifier_dataset_from_bank,
    build_classifier_dataset_from_multitrack,
    load_classifier_dataset,
    merge_classifier_datasets,
    relabel_classifier_dataset_sources,
    save_classifier_dataset,
    train_segment_classifier_from_dataset,
)
from transcriber.speaker_bank import SpeakerBank  # noqa: E402


CORE_SPEAKERS = (
    "Dungeon Master",
    "David Tanglethorn",
    "Leopold Magnus",
    "Kaladen Shash",
    "Cyrus Schwert",
    "Cletus Cobbington",
)
DEFAULT_EXCLUDED_SPEAKERS = ("B. Ver",)
DEFAULT_ALIASES = {"zariel torgan": "David Tanglethorn"}
DEFAULT_AUGMENTATION_PACKS: Dict[str, Tuple[str, ...]] = {
    "raw": (),
    "raw_plus_light": ("light_x1",),
    "raw_plus_discord_x1": ("discord_x1",),
    "raw_plus_discord_x2": ("discord_x2",),
    "raw_plus_all_aug": ("light_x1", "discord_x1", "discord_x2"),
}


def _parse_eval_spec(raw: str) -> Tuple[str, Path, Path]:
    parts = raw.split("::", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Eval specs must look like Session22::/path/to/session.zip::/path/to/transcript.txt"
        )
    return parts[0], Path(parts[1]).expanduser().resolve(), Path(parts[2]).expanduser().resolve()


def _collect_training_sources(input_roots: Sequence[Path], excluded_stems: Sequence[str]) -> List[Path]:
    excluded = {stem.strip().lower() for stem in excluded_stems if stem.strip()}
    seen: set[str] = set()
    results: List[Path] = []
    for root in input_roots:
        for session_source in _find_session_sources(root):
            if session_source.stem.strip().lower() in excluded:
                continue
            resolved = str(session_source.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(session_source)
    return sorted(results, key=lambda item: item.name.lower())


def _chunked(items: Sequence[Path], size: int) -> List[List[Path]]:
    return [list(items[index : index + size]) for index in range(0, len(items), max(size, 1))]


def _materialize_training_dir(session_sources: Sequence[Path], target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for session_source in session_sources:
        link_path = target_dir / session_source.name
        if link_path.exists():
            continue
        link_path.symlink_to(session_source)
    return target_dir


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _dataset_summary(dataset: ClassifierDataset) -> Dict[str, object]:
    dominant_values = [float(value) for value in dataset.dominant_shares if value == value and value >= 0.0]
    return {
        "samples": int(dataset.samples),
        "dimensions": int(dataset.embeddings.shape[1]) if dataset.samples else 0,
        "speakers": {label: int(count) for label, count in Counter(dataset.labels).items()},
        "sources": {label: int(count) for label, count in Counter(dataset.sources).items()},
        "domains": {label: int(count) for label, count in Counter(dataset.domains).items()},
        "sessions": {label: int(count) for label, count in Counter(dataset.sessions).items()},
        "dominant_share_mean": _mean(dominant_values),
    }


def _aggregate_eval_summary(summary: Dict[str, object]) -> Dict[str, object]:
    results = list(summary.get("results") or [])
    per_speaker_total: Counter[str] = Counter()
    per_speaker_correct: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    session_accuracies: List[float] = []
    session_coverages: List[float] = []
    matched_accuracies: List[float] = []

    for result in results:
        metrics = dict(result.get("metrics") or {})
        session_accuracies.append(float(metrics.get("accuracy") or 0.0))
        session_coverages.append(float(metrics.get("coverage") or 0.0))
        matched_accuracies.append(float(metrics.get("matched_accuracy") or 0.0))
        for speaker, payload in (metrics.get("per_speaker_accuracy") or {}).items():
            per_speaker_total[str(speaker)] += int(payload.get("total") or 0)
            per_speaker_correct[str(speaker)] += int(payload.get("correct") or 0)
        for speaker, counts in (metrics.get("confusion") or {}).items():
            for predicted, value in dict(counts).items():
                confusion[str(speaker)][str(predicted)] += int(value)

    per_speaker_accuracy = {
        speaker: {
            "total": int(per_speaker_total[speaker]),
            "correct": int(per_speaker_correct[speaker]),
            "accuracy": (
                per_speaker_correct[speaker] / per_speaker_total[speaker]
                if per_speaker_total[speaker]
                else 0.0
            ),
        }
        for speaker in sorted(per_speaker_total)
    }
    macro_accuracy = _mean(payload["accuracy"] for payload in per_speaker_accuracy.values())
    return {
        "mean_accuracy": _mean(session_accuracies),
        "mean_coverage": _mean(session_coverages),
        "mean_matched_accuracy": _mean(matched_accuracies),
        "macro_per_speaker_accuracy": macro_accuracy,
        "per_speaker_accuracy": per_speaker_accuracy,
        "confusion": {speaker: dict(counts) for speaker, counts in sorted(confusion.items())},
        "summary_path": str(summary.get("summary_path") or ""),
    }


def _write_eval_config(
    output_path: Path,
    *,
    base_config: Dict[str, object],
    hf_cache_root: Path,
    profile_name: str,
    diarization_model: str,
    speaker_mapping_path: Path,
    classifier_min_margin: float,
    threshold: float,
    min_segments_per_label: int,
    scoring_margin: float = 0.0,
    scoring_whiten: bool = False,
    scoring_as_norm_enabled: bool = False,
    scoring_as_norm_cohort_size: int = 50,
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
    speaker_bank_cfg["scoring_margin"] = float(scoring_margin)
    speaker_bank_cfg["match_per_segment"] = True
    speaker_bank_cfg["match_aggregation"] = "mean"
    speaker_bank_cfg["min_segments_per_label"] = int(min_segments_per_label)
    speaker_bank_cfg["diarization_model"] = diarization_model
    classifier_cfg = dict(speaker_bank_cfg.get("classifier") or {})
    classifier_cfg["min_confidence"] = 0.0
    classifier_cfg["min_margin"] = float(classifier_min_margin)
    speaker_bank_cfg["classifier"] = classifier_cfg
    scoring_cfg = dict(speaker_bank_cfg.get("scoring") or {})
    scoring_cfg["whiten"] = bool(scoring_whiten)
    scoring_cfg["margin"] = float(scoring_margin)
    scoring_cfg["as_norm"] = {
        "enabled": bool(scoring_as_norm_enabled),
        "cohort_size": int(scoring_as_norm_cohort_size),
    }
    speaker_bank_cfg["scoring"] = scoring_cfg
    payload["speaker_bank"] = speaker_bank_cfg
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _evaluate_profile(
    *,
    profile_name: str,
    eval_specs: Sequence[Tuple[str, Path, Path]],
    output_root: Path,
    prepared_root: Path,
    base_config: Dict[str, object],
    hf_cache_root: Path,
    diarization_model: str,
    speaker_mapping_path: Path,
    classifier_min_margin: float,
    threshold: float,
    min_segments_per_label: int,
    device: str,
    local_files_only: bool,
    scoring_margin: float = 0.0,
    scoring_whiten: bool = False,
    scoring_as_norm_enabled: bool = False,
    scoring_as_norm_cohort_size: int = 50,
) -> Dict[str, object]:
    config_path = _write_eval_config(
        output_root / "configs" / f"{profile_name}.json",
        base_config=base_config,
        hf_cache_root=hf_cache_root,
        profile_name=profile_name,
        diarization_model=diarization_model,
        speaker_mapping_path=speaker_mapping_path,
        classifier_min_margin=classifier_min_margin,
        threshold=threshold,
        min_segments_per_label=min_segments_per_label,
        scoring_margin=scoring_margin,
        scoring_whiten=scoring_whiten,
        scoring_as_norm_enabled=scoring_as_norm_enabled,
        scoring_as_norm_cohort_size=scoring_as_norm_cohort_size,
    )
    sessions: Dict[str, object] = {}
    mean_accuracy_values: List[float] = []
    mean_coverage_values: List[float] = []
    for session_name, session_zip, transcript in eval_specs:
        eval_output_dir = output_root / "eval" / profile_name / session_name
        summary = evaluate_multitrack_session(
            session_zip=session_zip,
            session_jsonl=transcript,
            output_dir=eval_output_dir,
            cache_root=prepared_root / session_name,
            speaker_mapping_path=speaker_mapping_path,
            config_path=config_path,
            window_seconds=300.0,
            hop_seconds=60.0,
            top_k=3,
            min_speakers=3,
            device_override=device,
            local_files_only_override=True if local_files_only else None,
        )
        summary["summary_path"] = str(eval_output_dir / "summary.json")
        aggregated = _aggregate_eval_summary(summary)
        sessions[session_name] = aggregated
        mean_accuracy_values.append(float(aggregated["mean_accuracy"]))
        mean_coverage_values.append(float(aggregated["mean_coverage"]))

    return {
        "profile_name": profile_name,
        "mean_accuracy": _mean(mean_accuracy_values),
        "mean_coverage": _mean(mean_coverage_values),
        "sessions": sessions,
        "config_path": str(config_path),
    }


def _build_widened_bank(
    *,
    training_sources: Sequence[Path],
    transcript_roots: Sequence[Path],
    output_root: Path,
    hf_cache_root: Path,
    speaker_mapping: Dict[str, object],
    diarization_model: str,
    device: str,
    quiet: bool,
    bank_profile_name: str,
) -> Path:
    print(
        json.dumps(
            {
                "event": "bank_build_start",
                "sources": len(training_sources),
                "profile": bank_profile_name,
            }
        ),
        flush=True,
    )
    source_root = output_root / "bank_train_inputs"
    source_root.mkdir(parents=True, exist_ok=True)
    for source in training_sources:
        link_path = source_root / source.name
        if link_path.exists():
            continue
        link_path.symlink_to(source)

    built = build_classifier_dataset_from_multitrack(
        input_path=str(source_root),
        dataset_cache_dir=output_root / "datasets" / "bank_source",
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
        batch_size=64,
        workers=4,
        window_seconds=300.0,
        transcript_search_roots=transcript_roots,
        speaker_aliases=DEFAULT_ALIASES,
        extra_input_paths=[],
        allowed_speakers=list(CORE_SPEAKERS),
        excluded_speakers=list(DEFAULT_EXCLUDED_SPEAKERS),
        training_mode="clean",
        clean_max_records_per_speaker_per_session=24,
        clean_window_size=15.0,
        clean_window_stride=15.0,
        augmentation_profile="none",
        augmentation_copies=0,
        include_base_samples=True,
        diarization_model_name=diarization_model,
    )
    if built is None:
        raise RuntimeError("Failed to build the widened bank dataset")
    bank_dataset, bank_dataset_summary = built
    bank_root = hf_cache_root / "speaker_bank"
    bank = SpeakerBank(bank_root, profile=bank_profile_name)
    bank_entries = []
    for index, speaker in enumerate(bank_dataset.labels):
        bank_entries.append(
            (
                speaker,
                bank_dataset.embeddings[index],
                bank_dataset.sessions[index],
                {
                    "mode": "final_widened_clean_dataset",
                    "source": bank_dataset.sources[index],
                    "duration": float(bank_dataset.durations[index]),
                },
            )
        )
    bank.extend(bank_entries)
    bank.save()
    save_classifier_dataset(
        output_root / "datasets" / "bank_source",
        bank_dataset,
        summary={"variant": "bank_source", **bank_dataset_summary},
    )
    print(
        json.dumps(
            {
                "event": "bank_build_complete",
                "samples": int(bank_dataset.samples),
                "speakers": bank_dataset_summary.get("speakers"),
                "profile_dir": str(hf_cache_root / "speaker_bank" / bank_profile_name),
            }
        ),
        flush=True,
    )
    return hf_cache_root / "speaker_bank" / bank_profile_name


def _build_dataset_variant(
    *,
    variant_name: str,
    training_batches: Sequence[Sequence[Path]],
    transcript_roots: Sequence[Path],
    output_root: Path,
    speaker_mapping: Dict[str, object],
    diarization_model: str,
    device: str,
    quiet: bool,
    augmentation_profile: str,
    augmentation_copies: int,
    include_base_samples: bool,
) -> Path:
    print(
        json.dumps(
            {
                "event": "dataset_variant_start",
                "variant": variant_name,
                "batches": len(training_batches),
                "augmentation_profile": augmentation_profile,
                "augmentation_copies": augmentation_copies,
                "include_base_samples": include_base_samples,
            }
        ),
        flush=True,
    )
    batch_datasets: List[ClassifierDataset] = []
    batch_summaries: List[Dict[str, object]] = []
    for batch_index, batch in enumerate(training_batches, start=1):
        print(
            json.dumps(
                {
                    "event": "dataset_batch_start",
                    "variant": variant_name,
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                }
            ),
            flush=True,
        )
        batch_input_dir = _materialize_training_dir(
            batch,
            output_root / "batch_inputs" / variant_name / f"batch_{batch_index:02d}",
        )
        batch_cache_dir = output_root / "datasets" / variant_name / f"batch_{batch_index:02d}"
        built = build_classifier_dataset_from_multitrack(
            input_path=str(batch_input_dir),
            dataset_cache_dir=batch_cache_dir,
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
            batch_size=64,
            workers=4,
            window_seconds=300.0,
            transcript_search_roots=transcript_roots,
            speaker_aliases=DEFAULT_ALIASES,
            extra_input_paths=[],
            allowed_speakers=list(CORE_SPEAKERS),
            excluded_speakers=list(DEFAULT_EXCLUDED_SPEAKERS),
            training_mode="mixed",
            augmentation_profile=augmentation_profile,
            augmentation_copies=augmentation_copies,
            augmentation_seed=13,
            include_base_samples=include_base_samples,
            diarization_model_name=diarization_model,
        )
        if built is None:
            continue
        dataset, summary = built
        relabeled_source = "mixed_raw" if include_base_samples else variant_name
        batch_datasets.append(relabel_classifier_dataset_sources(dataset, relabeled_source))
        batch_summaries.append(summary)
        print(
            json.dumps(
                {
                    "event": "dataset_batch_complete",
                    "variant": variant_name,
                    "batch_index": batch_index,
                    "samples": int(dataset.samples),
                    "speakers": summary.get("speakers"),
                }
            ),
            flush=True,
        )

    if not batch_datasets:
        raise RuntimeError(f"No dataset batches were produced for variant {variant_name}")

    merged = merge_classifier_datasets(batch_datasets)
    merged_dir = output_root / "datasets" / variant_name / "merged"
    save_classifier_dataset(
        merged_dir,
        merged,
        summary={
            "variant": variant_name,
            "batches": batch_summaries,
            **_dataset_summary(merged),
        },
    )
    print(
        json.dumps(
            {
                "event": "dataset_variant_complete",
                "variant": variant_name,
                "samples": int(merged.samples),
                "speakers": _dataset_summary(merged).get("speakers"),
            }
        ),
        flush=True,
    )
    return merged_dir


def _copy_bank_profile(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full widened mixed-speaker rebuild pipeline.")
    parser.add_argument("--train-input", action="append", default=[], help="Training ZIP root. Repeatable.")
    parser.add_argument(
        "--transcript-root",
        action="append",
        default=[],
        help="WSL-local transcript roots used for training/eval discovery. Repeatable.",
    )
    parser.add_argument("--speaker-mapping", required=True, help="Speaker mapping YAML/JSON.")
    parser.add_argument("--output-dir", required=True, help="Artifact directory for the rebuild.")
    parser.add_argument("--base-config", help="Optional transcriber config JSON/YAML.")
    parser.add_argument("--eval", action="append", default=[], type=_parse_eval_spec)
    parser.add_argument("--baseline-profile-dir", help="Optional existing profile used for regression checks.")
    parser.add_argument("--hf-cache-root", help="HF cache root for rebuilt bank/profiles.")
    parser.add_argument("--bank-profile-name", default="final_expanded_bank_v3")
    parser.add_argument(
        "--skip-bank-build",
        action="store_true",
        help="Reuse an existing widened bank profile instead of rebuilding the bank from stems.",
    )
    parser.add_argument(
        "--existing-bank-profile-dir",
        help="Existing widened bank profile directory to reuse when --skip-bank-build is set.",
    )
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of training sessions per mixed batch.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hf_cache_root = (
        Path(args.hf_cache_root).expanduser().resolve()
        if args.hf_cache_root
        else (output_root / "hf_cache")
    )
    hf_cache_root.mkdir(parents=True, exist_ok=True)
    speaker_mapping_path = Path(args.speaker_mapping).expanduser().resolve()
    speaker_mapping = _load_yaml_or_json(str(speaker_mapping_path)) or {}
    base_config = _load_yaml_or_json(str(Path(args.base_config).expanduser().resolve())) if args.base_config else {}
    eval_specs = list(args.eval)
    excluded_session_stems = [session_zip.stem for _, session_zip, _ in eval_specs]
    transcript_roots = [Path(path).expanduser().resolve() for path in args.transcript_root]
    training_sources = _collect_training_sources(
        [Path(path).expanduser().resolve() for path in args.train_input],
        excluded_stems=excluded_session_stems,
    )
    if not training_sources:
        raise SystemExit("No training multitrack sources were found for the widened rebuild.")
    training_batches = _chunked(training_sources, args.batch_size)

    if args.skip_bank_build:
        if not args.existing_bank_profile_dir:
            raise SystemExit("--existing-bank-profile-dir is required when --skip-bank-build is set.")
        bank_profile_dir = Path(args.existing_bank_profile_dir).expanduser().resolve()
        if not bank_profile_dir.exists():
            raise SystemExit(f"Existing bank profile dir does not exist: {bank_profile_dir}")
        print(
            json.dumps(
                {
                    "event": "bank_build_skipped",
                    "profile_dir": str(bank_profile_dir),
                }
            ),
            flush=True,
        )
    else:
        bank_profile_dir = _build_widened_bank(
            training_sources=training_sources,
            transcript_roots=transcript_roots,
            output_root=output_root,
            hf_cache_root=hf_cache_root,
            speaker_mapping=speaker_mapping,
            diarization_model=args.diarization_model,
            device=args.device,
            quiet=bool(args.quiet),
            bank_profile_name=args.bank_profile_name,
        )

    bank_dataset_built = build_classifier_dataset_from_bank(
        profile_dir=bank_profile_dir,
        speaker_mapping=speaker_mapping,
        speaker_aliases=DEFAULT_ALIASES,
        allowed_speakers=CORE_SPEAKERS,
        excluded_speakers=DEFAULT_EXCLUDED_SPEAKERS,
    )
    if bank_dataset_built is None:
        raise RuntimeError("Failed to build classifier dataset from the widened speaker bank")
    bank_dataset, bank_summary = bank_dataset_built
    save_classifier_dataset(
        output_root / "datasets" / "bank",
        bank_dataset,
        summary={"variant": "bank", **bank_summary},
    )

    variant_dirs = {
        "mixed_raw": _build_dataset_variant(
            variant_name="mixed_raw",
            training_batches=training_batches,
            transcript_roots=transcript_roots,
            output_root=output_root,
            speaker_mapping=speaker_mapping,
            diarization_model=args.diarization_model,
            device=args.device,
            quiet=bool(args.quiet),
            augmentation_profile="none",
            augmentation_copies=0,
            include_base_samples=True,
        ),
        "light_x1": _build_dataset_variant(
            variant_name="light_x1",
            training_batches=training_batches,
            transcript_roots=transcript_roots,
            output_root=output_root,
            speaker_mapping=speaker_mapping,
            diarization_model=args.diarization_model,
            device=args.device,
            quiet=bool(args.quiet),
            augmentation_profile="light",
            augmentation_copies=1,
            include_base_samples=False,
        ),
        "discord_x1": _build_dataset_variant(
            variant_name="discord_x1",
            training_batches=training_batches,
            transcript_roots=transcript_roots,
            output_root=output_root,
            speaker_mapping=speaker_mapping,
            diarization_model=args.diarization_model,
            device=args.device,
            quiet=bool(args.quiet),
            augmentation_profile="discord",
            augmentation_copies=1,
            include_base_samples=False,
        ),
        "discord_x2": _build_dataset_variant(
            variant_name="discord_x2",
            training_batches=training_batches,
            transcript_roots=transcript_roots,
            output_root=output_root,
            speaker_mapping=speaker_mapping,
            diarization_model=args.diarization_model,
            device=args.device,
            quiet=bool(args.quiet),
            augmentation_profile="discord",
            augmentation_copies=2,
            include_base_samples=False,
        ),
    }

    loaded_variants = {
        name: load_classifier_dataset(path)[0]
        for name, path in variant_dirs.items()
    }
    prepared_root = output_root / "prepared_eval"
    stage1_results: List[Dict[str, object]] = []

    for pack_name, aug_variants in DEFAULT_AUGMENTATION_PACKS.items():
        merged_datasets = [relabel_classifier_dataset_sources(bank_dataset, "bank"), loaded_variants["mixed_raw"]]
        for aug_name in aug_variants:
            merged_datasets.append(relabel_classifier_dataset_sources(loaded_variants[aug_name], "mixed_aug_total"))
        merged_dataset = merge_classifier_datasets(merged_datasets)
        balanced_dataset, balance_summary = balance_classifier_dataset(
            merged_dataset,
            target_speakers=CORE_SPEAKERS,
            max_samples_per_cell=500,
        )

        model_grid: List[Tuple[str, str, float | int]] = []
        for value in (5, 7, 9, 13, 17):
            model_grid.append(("knn", "n_neighbors", value))
        for value in (1.0, 4.0):
            model_grid.append(("logreg", "c", value))
            model_grid.append(("logreg_unbalanced", "c", value))

        for model_name, hyper_name, hyper_value in model_grid:
            for classifier_margin in (0.0, 0.03, 0.05, 0.08, 0.12):
                profile_name = (
                    f"{pack_name}_{model_name}_{hyper_name}_{str(hyper_value).replace('.', '_')}"
                    f"_margin_{str(classifier_margin).replace('.', '_')}"
                )
                print(
                    json.dumps(
                        {
                            "event": "stage1_profile_start",
                            "pack": pack_name,
                            "profile_name": profile_name,
                        }
                    ),
                    flush=True,
                )
                profile_dir = hf_cache_root / "speaker_bank" / profile_name
                _copy_bank_profile(bank_profile_dir, profile_dir)
                save_classifier_dataset(
                    profile_dir / "training_dataset",
                    balanced_dataset,
                    summary={
                        "pack": pack_name,
                        "balance": balance_summary,
                        **_dataset_summary(balanced_dataset),
                    },
                )
                training_summary = train_segment_classifier_from_dataset(
                    dataset=balanced_dataset,
                    profile_dir=profile_dir,
                    model_name=model_name,
                    classifier_c=float(hyper_value) if hyper_name == "c" else 1.0,
                    classifier_n_neighbors=int(hyper_value) if hyper_name == "n_neighbors" else 7,
                    base_summary={
                        "pack": pack_name,
                        "balance": balance_summary,
                    },
                )
                evaluation = _evaluate_profile(
                    profile_name=profile_name,
                    eval_specs=eval_specs,
                    output_root=output_root,
                    prepared_root=prepared_root,
                    base_config=base_config,
                    hf_cache_root=hf_cache_root,
                    diarization_model=args.diarization_model,
                    speaker_mapping_path=speaker_mapping_path,
                    classifier_min_margin=classifier_margin,
                    threshold=0.40,
                    min_segments_per_label=2,
                    device=args.device,
                    local_files_only=bool(args.local_files_only),
                )
                stage1_results.append(
                    {
                        "stage": "classifier",
                        "profile_name": profile_name,
                        "pack": pack_name,
                        "model_name": model_name,
                        hyper_name: hyper_value,
                        "classifier_min_margin": classifier_margin,
                        "training_summary": training_summary,
                        "evaluation": evaluation,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "stage1_profile_complete",
                            "profile_name": profile_name,
                            "mean_accuracy": evaluation["mean_accuracy"],
                            "mean_coverage": evaluation["mean_coverage"],
                        }
                    ),
                    flush=True,
                )

    stage1_results.sort(
        key=lambda item: (
            float(item["evaluation"]["mean_accuracy"]),
            float(item["evaluation"]["mean_coverage"]),
        ),
        reverse=True,
    )
    top3 = stage1_results[:3]

    stage2_results: List[Dict[str, object]] = []
    for winner in top3:
        for threshold, min_segments in product((0.35, 0.40, 0.45), (1, 2, 3)):
            print(
                json.dumps(
                    {
                        "event": "stage2_profile_start",
                        "base_profile": winner["profile_name"],
                        "threshold": threshold,
                        "min_segments_per_label": min_segments,
                    }
                ),
                flush=True,
            )
            evaluation = _evaluate_profile(
                profile_name=str(winner["profile_name"]),
                eval_specs=eval_specs,
                output_root=output_root / "stage2",
                prepared_root=prepared_root,
                base_config=base_config,
                hf_cache_root=hf_cache_root,
                diarization_model=args.diarization_model,
                speaker_mapping_path=speaker_mapping_path,
                classifier_min_margin=float(winner["classifier_min_margin"]),
                threshold=float(threshold),
                min_segments_per_label=int(min_segments),
                device=args.device,
                local_files_only=bool(args.local_files_only),
            )
            stage2_results.append(
                {
                    "stage": "calibration",
                    "base_profile": winner["profile_name"],
                    "classifier_min_margin": winner["classifier_min_margin"],
                    "threshold": threshold,
                    "min_segments_per_label": min_segments,
                    "evaluation": evaluation,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "stage2_profile_complete",
                        "base_profile": winner["profile_name"],
                        "threshold": threshold,
                        "min_segments_per_label": min_segments,
                        "mean_accuracy": evaluation["mean_accuracy"],
                    }
                ),
                flush=True,
            )

    stage2_results.sort(
        key=lambda item: (
            float(item["evaluation"]["mean_accuracy"]),
            float(item["evaluation"]["mean_coverage"]),
        ),
        reverse=True,
    )
    top2 = stage2_results[:2]

    stage3_results: List[Dict[str, object]] = []
    for winner in top2:
        for scoring_whiten, as_norm_enabled in product((False, True), (False, True)):
            print(
                json.dumps(
                    {
                        "event": "stage3_profile_start",
                        "base_profile": winner["base_profile"],
                        "threshold": winner["threshold"],
                        "min_segments_per_label": winner["min_segments_per_label"],
                        "scoring_whiten": scoring_whiten,
                        "scoring_as_norm_enabled": as_norm_enabled,
                    }
                ),
                flush=True,
            )
            evaluation = _evaluate_profile(
                profile_name=str(winner["base_profile"]),
                eval_specs=eval_specs,
                output_root=output_root / "stage3",
                prepared_root=prepared_root,
                base_config=base_config,
                hf_cache_root=hf_cache_root,
                diarization_model=args.diarization_model,
                speaker_mapping_path=speaker_mapping_path,
                classifier_min_margin=float(winner["classifier_min_margin"]),
                threshold=float(winner["threshold"]),
                min_segments_per_label=int(winner["min_segments_per_label"]),
                device=args.device,
                local_files_only=bool(args.local_files_only),
                scoring_whiten=scoring_whiten,
                scoring_as_norm_enabled=as_norm_enabled,
                scoring_as_norm_cohort_size=50,
            )
            stage3_results.append(
                {
                    "stage": "normalization",
                    "base_profile": winner["base_profile"],
                    "threshold": winner["threshold"],
                    "min_segments_per_label": winner["min_segments_per_label"],
                    "scoring_whiten": scoring_whiten,
                    "scoring_as_norm_enabled": as_norm_enabled,
                    "evaluation": evaluation,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "stage3_profile_complete",
                        "base_profile": winner["base_profile"],
                        "scoring_whiten": scoring_whiten,
                        "scoring_as_norm_enabled": as_norm_enabled,
                        "mean_accuracy": evaluation["mean_accuracy"],
                    }
                ),
                flush=True,
            )

    all_results = stage1_results + stage2_results + stage3_results
    all_results.sort(
        key=lambda item: (
            float(item["evaluation"]["mean_accuracy"]),
            float(item["evaluation"]["mean_coverage"]),
        ),
        reverse=True,
    )

    baseline = None
    if args.baseline_profile_dir:
        baseline_profile_dir = Path(args.baseline_profile_dir).expanduser().resolve()
        if baseline_profile_dir.exists():
            baseline_name = baseline_profile_dir.name
            baseline_root = baseline_profile_dir.parent.parent
            baseline = _evaluate_profile(
                profile_name=baseline_name,
                eval_specs=eval_specs,
                output_root=output_root / "baseline",
                prepared_root=prepared_root,
                base_config=base_config,
                hf_cache_root=baseline_root,
                diarization_model=args.diarization_model,
                speaker_mapping_path=speaker_mapping_path,
                classifier_min_margin=0.05,
                threshold=0.40,
                min_segments_per_label=2,
                device=args.device,
                local_files_only=bool(args.local_files_only),
            )

    best = all_results[0] if all_results else None
    session22_accuracy = (
        float(best["evaluation"]["sessions"].get("Session22", {}).get("mean_accuracy", 0.0))
        if best
        else 0.0
    )
    session61_accuracy = (
        float(best["evaluation"]["sessions"].get("Session61", {}).get("mean_accuracy", 0.0))
        if best
        else 0.0
    )
    baseline_session61_accuracy = (
        float(baseline["sessions"].get("Session61", {}).get("mean_accuracy", 0.0))
        if baseline
        else None
    )
    promotion = {
        "beats_session22_floor": session22_accuracy > 0.4533,
        "session61_regression_ok": (
            baseline_session61_accuracy is None
            or session61_accuracy >= (baseline_session61_accuracy - 0.01)
        ),
        "meets_target_mean_accuracy": (
            float(best["evaluation"]["mean_accuracy"]) >= 0.50 if best else False
        ),
        "meets_target_mean_coverage": (
            float(best["evaluation"]["mean_coverage"]) >= 0.75 if best else False
        ),
    }
    promotion["promote"] = bool(
        promotion["beats_session22_floor"] and promotion["session61_regression_ok"]
    )
    final_summary = {
        "bank_profile_dir": str(bank_profile_dir),
        "training_sessions": [str(path) for path in training_sources],
        "eval_sessions": [
            {"name": name, "session_zip": str(session_zip), "session_jsonl": str(transcript)}
            for name, session_zip, transcript in eval_specs
        ],
        "baseline": baseline,
        "best": best,
        "promotion": promotion,
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_results,
        "success_criteria": {
            "target_mean_accuracy": 0.50,
            "target_mean_coverage": 0.75,
            "session22_floor": 0.4533,
        },
    }
    summary_path = output_root / "final_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "rebuild_complete",
                "summary_path": str(summary_path),
                "best_profile": best["profile_name"] if best else None,
                "best_mean_accuracy": best["evaluation"]["mean_accuracy"] if best else None,
                "promotion": promotion,
            }
        ),
        flush=True,
    )
    print(json.dumps({"final_summary": str(summary_path), "best_profile": best}, indent=2))


if __name__ == "__main__":
    main()
