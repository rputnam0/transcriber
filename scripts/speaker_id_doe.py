from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.cli import _load_yaml_or_json  # noqa: E402
from transcriber.multitrack_eval import evaluate_multitrack_session  # noqa: E402
from transcriber.segment_classifier import (  # noqa: E402
    _find_session_zips,
    train_segment_classifier_from_multitrack,
)
from transcriber.diarization import release_runtime_caches  # noqa: E402


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_eval_spec(raw: str) -> Tuple[Path, Path]:
    parts = raw.split("::", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Evaluation specs must look like /path/to/session.zip::/path/to/transcript.txt"
        )
    return Path(parts[0]).expanduser(), Path(parts[1]).expanduser()


def _collect_training_zips(input_roots: Sequence[Path], excluded_stems: set[str]) -> List[Path]:
    session_zips: List[Path] = []
    seen: set[str] = set()
    for root in input_roots:
        for zip_path in _find_session_zips(root):
            normalized_stem = zip_path.stem.strip().lower()
            if normalized_stem in excluded_stems:
                continue
            resolved = str(zip_path.expanduser().resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            session_zips.append(zip_path)
    return sorted(session_zips, key=lambda item: item.name)


def _materialize_training_dir(session_zips: Sequence[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in session_zips:
        link_path = target_dir / zip_path.name
        if link_path.exists():
            continue
        link_path.symlink_to(zip_path)


def _write_eval_config(
    base_config_path: Path | None,
    output_path: Path,
    *,
    speaker_bank_root: Path,
    profile_name: str,
    speaker_bank_threshold: float,
    speaker_bank_scoring_margin: float,
    speaker_bank_min_segments_per_label: int,
    classifier_min_confidence: float,
    classifier_min_margin: float,
    diarization_model: str | None,
) -> Path:
    payload: Dict[str, object] = {}
    if base_config_path and base_config_path.exists():
        payload = _load_yaml_or_json(str(base_config_path))

    payload["speaker_bank_root"] = str(speaker_bank_root.parent)
    speaker_bank_cfg = dict(payload.get("speaker_bank") or {})
    speaker_bank_cfg["enabled"] = True
    speaker_bank_cfg["path"] = profile_name
    speaker_bank_cfg["threshold"] = speaker_bank_threshold
    speaker_bank_cfg["scoring_margin"] = speaker_bank_scoring_margin
    speaker_bank_cfg["min_segments_per_label"] = speaker_bank_min_segments_per_label
    speaker_bank_cfg["match_per_segment"] = True
    if diarization_model:
        speaker_bank_cfg["diarization_model"] = diarization_model
        payload["diarization_model"] = diarization_model
    classifier_cfg = dict(speaker_bank_cfg.get("classifier") or {})
    classifier_cfg["min_confidence"] = classifier_min_confidence
    classifier_cfg["min_margin"] = classifier_min_margin
    speaker_bank_cfg["classifier"] = classifier_cfg
    payload["speaker_bank"] = speaker_bank_cfg
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speaker-ID DOE experiments.")
    parser.add_argument("--base-profile-dir", required=True, help="Existing speaker-bank profile directory.")
    parser.add_argument(
        "--train-input",
        action="append",
        default=[],
        help="Directory or ZIP root used to build classifier samples. Repeatable.",
    )
    parser.add_argument(
        "--transcript-root",
        action="append",
        default=[],
        help="Additional transcript search roots. Repeatable.",
    )
    parser.add_argument(
        "--exclude-session",
        action="append",
        default=[],
        help="Session stem to exclude from training, e.g. 'Session 61'. Repeatable.",
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        type=_parse_eval_spec,
        help="Held-out eval pair: /path/to/session.zip::/path/to/transcript.txt",
    )
    parser.add_argument("--speaker-mapping", required=True, help="Speaker mapping YAML/JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to store DOE artifacts.")
    parser.add_argument("--config", help="Optional base transcriber config JSON.")
    parser.add_argument("--hf-token", help="Optional HF token override.")
    parser.add_argument("--device", default="cuda", help="Training/eval device.")
    parser.add_argument(
        "--diarization-model",
        help="Optional diarization model override, e.g. pyannote/speaker-diarization-community-1.",
    )
    parser.add_argument(
        "--excluded-speaker",
        action="append",
        default=[],
        help="Canonical speaker labels to exclude from training.",
    )
    parser.add_argument(
        "--augmentation-profile",
        default="none",
        help="Waveform augmentation profile for classifier training (none, light, discord).",
    )
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=0,
        help="How many augmented copies to generate per training sample.",
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=13,
        help="Base RNG seed for waveform augmentation.",
    )
    parser.add_argument("--speaker-bank-threshold", type=float, default=0.40)
    parser.add_argument("--speaker-bank-scoring-margin", type=float, default=0.00)
    parser.add_argument("--speaker-bank-min-segments-per-label", type=int, default=2)
    parser.add_argument(
        "--models",
        default="logreg,logreg_unbalanced,knn",
        help="Comma-separated classifier models.",
    )
    parser.add_argument(
        "--training-modes",
        default="mixed,hybrid",
        help="Comma-separated training modes: mixed, clean, hybrid.",
    )
    parser.add_argument("--c-values", default="1.0,4.0", help="Comma-separated logistic C values.")
    parser.add_argument(
        "--knn-values",
        default="7,15",
        help="Comma-separated KNN neighbor counts.",
    )
    parser.add_argument(
        "--min-confidences",
        default="0.0",
        help="Comma-separated classifier minimum confidence values.",
    )
    parser.add_argument(
        "--min-margins",
        default="0.08,0.15",
        help="Comma-separated classifier minimum margin values.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Training windows per session.")
    parser.add_argument("--hop-seconds", type=float, default=120.0, help="Training window hop.")
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=5,
        help="Minimum speakers required for a training window.",
    )
    parser.add_argument("--max-samples-per-speaker", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_bank_root = output_dir / "speaker_bank"
    speaker_bank_root.mkdir(parents=True, exist_ok=True)

    base_profile_dir = Path(args.base_profile_dir).expanduser().resolve()
    if not base_profile_dir.exists():
        raise SystemExit(f"Base profile directory does not exist: {base_profile_dir}")

    train_input_roots = [Path(path).expanduser().resolve() for path in args.train_input]
    transcript_roots = [Path(path).expanduser().resolve() for path in args.transcript_root]
    excluded_stems = {item.strip().lower() for item in args.exclude_session if item.strip()}
    session_zips = _collect_training_zips(train_input_roots, excluded_stems)
    if not session_zips:
        raise SystemExit("No training ZIP sessions were found.")
    speaker_mapping = _load_yaml_or_json(str(Path(args.speaker_mapping).expanduser().resolve()))

    experiments: List[Dict[str, object]] = []
    model_names = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    training_modes = [item.strip().lower() for item in args.training_modes.split(",") if item.strip()]
    c_values = _parse_csv_floats(args.c_values)
    knn_values = _parse_csv_ints(args.knn_values)
    min_confidences = _parse_csv_floats(args.min_confidences)
    min_margins = _parse_csv_floats(args.min_margins)

    base_config_path = Path(args.config).expanduser().resolve() if args.config else None
    eval_specs: List[Tuple[Path, Path]] = list(args.eval)

    with tempfile.TemporaryDirectory(prefix="speaker_id_doe_train_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        training_input_dir = temp_dir / "train_inputs"
        _materialize_training_dir(session_zips, training_input_dir)

        for training_mode in training_modes:
            dataset_cache_dir = output_dir / "datasets" / training_mode
            for model_name in model_names:
                if model_name in {"logreg", "logreg_unbalanced", "torch_linear"}:
                    hyper_values = [("c", value) for value in c_values]
                elif model_name == "knn":
                    hyper_values = [("n_neighbors", value) for value in knn_values]
                else:
                    raise SystemExit(f"Unsupported model: {model_name}")

                for hyper_name, hyper_value in hyper_values:
                    for min_confidence in min_confidences:
                        for min_margin in min_margins:
                            profile_name = (
                                f"{training_mode}_{model_name}_{hyper_name}_{str(hyper_value).replace('.', '_')}"
                                f"_conf_{str(min_confidence).replace('.', '_')}"
                                f"_margin_{str(min_margin).replace('.', '_')}"
                            )
                            profile_dir = speaker_bank_root / profile_name
                            if profile_dir.exists():
                                shutil.rmtree(profile_dir)
                            shutil.copytree(base_profile_dir, profile_dir)
                            print(
                                json.dumps(
                                    {
                                        "event": "experiment_start",
                                        "profile_name": profile_name,
                                        "training_mode": training_mode,
                                        "model_name": model_name,
                                        hyper_name: hyper_value,
                                        "min_confidence": min_confidence,
                                        "min_margin": min_margin,
                                    }
                                ),
                                flush=True,
                            )

                            classifier_summary = train_segment_classifier_from_multitrack(
                                input_path=str(training_input_dir),
                                extra_input_paths=[],
                                profile_dir=profile_dir,
                                speaker_mapping=speaker_mapping,
                                hf_token=args.hf_token,
                                force_device=args.device,
                                quiet=False,
                                top_k=args.top_k,
                                hop_seconds=args.hop_seconds,
                                min_speakers=args.min_speakers,
                                max_samples_per_speaker=args.max_samples_per_speaker,
                                batch_size=args.batch_size,
                                workers=args.workers,
                                transcript_search_roots=transcript_roots or None,
                                speaker_aliases={"zariel torgan": "David Tanglethorn"},
                                excluded_speakers=args.excluded_speaker or None,
                                model_name=model_name,
                                classifier_c=float(hyper_value) if hyper_name == "c" else 1.0,
                                classifier_n_neighbors=int(hyper_value)
                                if hyper_name == "n_neighbors"
                                else 7,
                                training_mode=training_mode,
                                augmentation_profile=args.augmentation_profile,
                                augmentation_copies=args.augmentation_copies,
                                augmentation_seed=args.augmentation_seed,
                                dataset_cache_dir=dataset_cache_dir,
                                diarization_model_name=args.diarization_model,
                            )
                            release_runtime_caches()

                            eval_config_path = _write_eval_config(
                                base_config_path,
                                output_dir / f"{profile_name}.config.json",
                                speaker_bank_root=speaker_bank_root,
                                profile_name=profile_name,
                                speaker_bank_threshold=args.speaker_bank_threshold,
                                speaker_bank_scoring_margin=args.speaker_bank_scoring_margin,
                                speaker_bank_min_segments_per_label=args.speaker_bank_min_segments_per_label,
                                classifier_min_confidence=min_confidence,
                                classifier_min_margin=min_margin,
                                diarization_model=args.diarization_model,
                            )

                            eval_results: List[Dict[str, object]] = []
                            for session_zip, transcript_path in eval_specs:
                                eval_output_dir = output_dir / "eval" / profile_name / session_zip.stem
                                eval_cache_root = output_dir / "prepared" / session_zip.stem
                                summary = evaluate_multitrack_session(
                                    session_zip=session_zip,
                                    session_jsonl=transcript_path,
                                    output_dir=eval_output_dir,
                                    cache_root=eval_cache_root,
                                    speaker_mapping_path=Path(args.speaker_mapping).expanduser().resolve(),
                                    config_path=eval_config_path,
                                    window_seconds=300.0,
                                    hop_seconds=60.0,
                                    top_k=3,
                                    min_speakers=3,
                                    device_override=args.device,
                                    local_files_only_override=None,
                                )
                                accuracies = [
                                    float(item["metrics"]["accuracy"])
                                    for item in summary.get("results", [])
                                ]
                                coverages = [
                                    float(item["metrics"]["coverage"])
                                    for item in summary.get("results", [])
                                ]
                                eval_results.append(
                                    {
                                        "session_zip": str(session_zip),
                                        "transcript": str(transcript_path),
                                        "summary_path": str(eval_output_dir / "summary.json"),
                                        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
                                        "mean_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
                                    }
                                )
                                release_runtime_caches()

                            experiment = {
                                "profile_name": profile_name,
                                "training_mode": training_mode,
                                "model_name": model_name,
                                hyper_name: hyper_value,
                                "min_confidence": min_confidence,
                                "min_margin": min_margin,
                                "training_summary": classifier_summary,
                                "eval_results": eval_results,
                            }
                            experiment["mean_eval_accuracy"] = (
                                sum(float(item["mean_accuracy"]) for item in eval_results) / len(eval_results)
                                if eval_results
                                else 0.0
                            )
                            experiment["mean_eval_coverage"] = (
                                sum(float(item["mean_coverage"]) for item in eval_results) / len(eval_results)
                                if eval_results
                                else 0.0
                            )
                            experiments.append(experiment)

                            report_path = output_dir / "speaker_id_doe_report.json"
                            report_path.write_text(json.dumps(experiments, indent=2), encoding="utf-8")
                            release_runtime_caches()
                            print(
                                json.dumps(
                                    {
                                        "event": "experiment_complete",
                                        "profile_name": profile_name,
                                        "mean_eval_accuracy": experiment["mean_eval_accuracy"],
                                        "mean_eval_coverage": experiment["mean_eval_coverage"],
                                    }
                                )
                            )


if __name__ == "__main__":
    main()
