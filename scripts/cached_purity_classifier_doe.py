from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.cli import _load_yaml_or_json  # noqa: E402
from transcriber.multitrack_eval import evaluate_multitrack_session  # noqa: E402
from transcriber.segment_classifier import (  # noqa: E402
    balance_classifier_dataset,
    build_classifier_dataset_from_bank,
    filter_classifier_dataset,
    load_classifier_dataset,
    merge_classifier_datasets,
    relabel_classifier_dataset_sources,
    save_classifier_dataset,
    train_segment_classifier_from_dataset,
)


CORE_SPEAKERS = (
    "Dungeon Master",
    "David Tanglethorn",
    "Leopold Magnus",
    "Kaladen Shash",
    "Cyrus Schwert",
    "Cletus Cobbington",
)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _copy_bank_profile(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _parse_eval_spec(raw: str) -> Tuple[str, Path, Path, Path]:
    parts = raw.split("::", maxsplit=3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Eval spec must look like Session22::/path/to/session.zip::/path/to/transcript.txt::/path/to/cache"
        )
    return (
        parts[0],
        Path(parts[1]).expanduser().resolve(),
        Path(parts[2]).expanduser().resolve(),
        Path(parts[3]).expanduser().resolve(),
    )


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
) -> Path:
    payload = dict(base_config)
    payload["hf_cache_root"] = str(hf_cache_root)
    payload["speaker_bank_root"] = str(hf_cache_root)
    payload["speaker_mapping"] = str(speaker_mapping_path)
    payload["diarization_model"] = diarization_model
    speaker_bank_cfg = dict(payload.get("speaker_bank") or {})
    classifier_cfg = dict(speaker_bank_cfg.get("classifier") or {})
    speaker_bank_cfg["enabled"] = True
    speaker_bank_cfg["path"] = profile_name
    speaker_bank_cfg["threshold"] = float(threshold)
    speaker_bank_cfg["match_per_segment"] = True
    speaker_bank_cfg["match_aggregation"] = "mean"
    speaker_bank_cfg["min_segments_per_label"] = int(min_segments_per_label)
    classifier_cfg["min_confidence"] = 0.0
    classifier_cfg["min_margin"] = float(classifier_min_margin)
    speaker_bank_cfg["classifier"] = classifier_cfg
    payload["speaker_bank"] = speaker_bank_cfg
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _aggregate_eval_summary(summary: Dict[str, object]) -> Dict[str, object]:
    results = list(summary.get("results") or [])
    accuracy_values = [float(item["metrics"]["accuracy"]) for item in results]
    coverage_values = [float(item["metrics"]["coverage"]) for item in results]
    per_speaker_correct: Dict[str, int] = {}
    per_speaker_total: Dict[str, int] = {}
    for item in results:
        for speaker, payload in (item.get("metrics", {}).get("per_speaker_accuracy") or {}).items():
            per_speaker_correct[speaker] = per_speaker_correct.get(speaker, 0) + int(payload.get("correct") or 0)
            per_speaker_total[speaker] = per_speaker_total.get(speaker, 0) + int(payload.get("total") or 0)
    return {
        "mean_accuracy": _mean(accuracy_values),
        "mean_coverage": _mean(coverage_values),
        "per_speaker_accuracy": {
            speaker: (
                per_speaker_correct[speaker] / per_speaker_total[speaker]
                if per_speaker_total[speaker]
                else 0.0
            )
            for speaker in sorted(per_speaker_total)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached purity/balance DOE against existing eval caches.")
    parser.add_argument("--bank-profile-dir", required=True)
    parser.add_argument("--mixed-dataset-dir", required=True)
    parser.add_argument("--speaker-mapping", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval", action="append", default=[], type=_parse_eval_spec)
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-segments-per-label", type=int, default=2)
    parser.add_argument("--max-samples-per-cell", type=int, default=500)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bank_profile_dir = Path(args.bank_profile_dir).expanduser().resolve()
    hf_cache_root = bank_profile_dir.parent.parent
    speaker_mapping_path = Path(args.speaker_mapping).expanduser().resolve()
    speaker_mapping = _load_yaml_or_json(str(speaker_mapping_path)) or {}
    base_config = _load_yaml_or_json(str(Path(args.base_config).expanduser().resolve())) or {}

    bank_built = build_classifier_dataset_from_bank(
        profile_dir=bank_profile_dir,
        speaker_mapping=speaker_mapping,
        speaker_aliases={"zariel torgan": "David Tanglethorn"},
        allowed_speakers=CORE_SPEAKERS,
        excluded_speakers=["B. Ver"],
    )
    if bank_built is None:
        raise SystemExit("Failed to load bank dataset")
    bank_dataset, bank_summary = bank_built
    mixed_dataset, mixed_summary = load_classifier_dataset(Path(args.mixed_dataset_dir).expanduser().resolve())

    bank_dataset = relabel_classifier_dataset_sources(bank_dataset, "bank")
    mixed_dataset = relabel_classifier_dataset_sources(mixed_dataset, "mixed_raw")

    results: List[Dict[str, object]] = []
    grid = product(
        (None, 0.80, 0.85, 0.90),
        (None, 2, 3),
        (7, 9, 13),
        (0.03, 0.05, 0.08),
    )
    for min_share, max_active_speakers, n_neighbors, min_margin in grid:
        filtered_mixed, filter_summary = filter_classifier_dataset(
            mixed_dataset,
            allowed_speakers=CORE_SPEAKERS,
            allowed_sources=["mixed_raw"],
            min_dominant_share=min_share,
            max_active_speakers=max_active_speakers,
            min_duration=0.8,
            max_duration=15.0,
        )
        if filtered_mixed.samples <= 0:
            continue
        merged_dataset = merge_classifier_datasets([bank_dataset, filtered_mixed])
        balanced_dataset, balance_summary = balance_classifier_dataset(
            merged_dataset,
            target_speakers=CORE_SPEAKERS,
            max_samples_per_cell=int(args.max_samples_per_cell),
        )
        if balanced_dataset.samples <= 0:
            continue

        profile_name = (
            f"purity_share_{'none' if min_share is None else str(min_share).replace('.', '_')}"
            f"_active_{'none' if max_active_speakers is None else max_active_speakers}"
            f"_knn_{n_neighbors}_margin_{str(min_margin).replace('.', '_')}"
        )
        profile_dir = hf_cache_root / "speaker_bank" / profile_name
        _copy_bank_profile(bank_profile_dir, profile_dir)
        save_classifier_dataset(
            profile_dir / "training_dataset",
            balanced_dataset,
            summary={
                "bank": bank_summary,
                "mixed": mixed_summary,
                "filter": filter_summary,
                "balance": balance_summary,
            },
        )
        training_summary = train_segment_classifier_from_dataset(
            dataset=balanced_dataset,
            profile_dir=profile_dir,
            model_name="knn",
            classifier_n_neighbors=int(n_neighbors),
            classifier_c=1.0,
            base_summary={
                "bank": bank_summary,
                "mixed": mixed_summary,
                "filter": filter_summary,
                "balance": balance_summary,
            },
        )
        config_path = _write_eval_config(
            output_root / "configs" / f"{profile_name}.json",
            base_config=base_config,
            hf_cache_root=hf_cache_root,
            profile_name=profile_name,
            diarization_model=str(args.diarization_model),
            speaker_mapping_path=speaker_mapping_path,
            classifier_min_margin=float(min_margin),
            threshold=float(args.threshold),
            min_segments_per_label=int(args.min_segments_per_label),
        )
        session_results: Dict[str, object] = {}
        accuracy_values: List[float] = []
        coverage_values: List[float] = []
        for session_name, session_zip, transcript_path, cache_root in args.eval:
            summary = evaluate_multitrack_session(
                session_zip=session_zip,
                session_jsonl=transcript_path,
                output_dir=output_root / "eval" / profile_name / session_name,
                cache_root=cache_root,
                speaker_mapping_path=speaker_mapping_path,
                config_path=config_path,
                window_seconds=300.0,
                hop_seconds=60.0,
                top_k=3,
                min_speakers=3,
                device_override=args.device,
                local_files_only_override=True,
            )
            aggregated = _aggregate_eval_summary(summary)
            session_results[session_name] = aggregated
            accuracy_values.append(float(aggregated["mean_accuracy"]))
            coverage_values.append(float(aggregated["mean_coverage"]))

        result = {
            "profile_name": profile_name,
            "min_dominant_share": min_share,
            "max_active_speakers": max_active_speakers,
            "n_neighbors": int(n_neighbors),
            "classifier_min_margin": float(min_margin),
            "mean_accuracy": _mean(accuracy_values),
            "mean_coverage": _mean(coverage_values),
            "sessions": session_results,
            "training_summary": training_summary,
            "filter_summary": filter_summary,
            "balance_summary": balance_summary,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "profile_name": profile_name,
                    "mean_accuracy": result["mean_accuracy"],
                    "mean_coverage": result["mean_coverage"],
                }
            ),
            flush=True,
        )

    results.sort(key=lambda item: (float(item["mean_accuracy"]), float(item["mean_coverage"])), reverse=True)
    payload = {"results": results}
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
