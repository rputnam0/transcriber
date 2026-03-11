from __future__ import annotations

import argparse
import json
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from transcriber.cli import _load_yaml_or_json
from transcriber.multitrack_eval import evaluate_multitrack_session
from transcriber.segment_classifier import train_segment_classifier_from_multitrack


def _parse_aliases(values: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Alias must use source=target syntax: {value}")
        source, target = value.split("=", 1)
        aliases[source.strip().lower()] = target.strip()
    return aliases


def _parse_eval_session(values: Sequence[str]) -> List[Dict[str, Path]]:
    sessions: List[Dict[str, Path]] = []
    for value in values:
        parts = value.split("::")
        if len(parts) != 3:
            raise ValueError(
                "Eval session must use name::/path/to/session.zip::/path/to/transcript syntax"
            )
        name, session_zip, transcript = parts
        sessions.append(
            {
                "name": Path(name).stem,
                "session_zip": Path(session_zip).expanduser().resolve(),
                "session_jsonl": Path(transcript).expanduser().resolve(),
            }
        )
    return sessions


def _copy_bank_profile(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("bank.json", "embeddings.npy"):
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required speaker bank artifact: {source_path}")
        shutil.copy2(source_path, target_dir / filename)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _build_eval_config(
    *,
    base_config: Optional[Dict[str, object]],
    speaker_mapping_path: Path,
    speaker_bank_root: Path,
    speaker_bank_profile: str,
    classifier_min_confidence: float,
    classifier_min_margin: float,
    hf_cache_root: Optional[str],
    device: Optional[str],
) -> Dict[str, object]:
    cfg = dict(base_config or {})
    speaker_bank_cfg = dict(cfg.get("speaker_bank") or {})
    classifier_cfg = dict(speaker_bank_cfg.get("classifier") or {})
    speaker_bank_cfg["enabled"] = True
    speaker_bank_cfg["path"] = speaker_bank_profile
    classifier_cfg["min_confidence"] = float(classifier_min_confidence)
    classifier_cfg["min_margin"] = float(classifier_min_margin)
    speaker_bank_cfg["classifier"] = classifier_cfg
    cfg["speaker_bank"] = speaker_bank_cfg
    cfg["speaker_bank_root"] = str(speaker_bank_root)
    cfg["speaker_mapping"] = str(speaker_mapping_path)
    if hf_cache_root:
        cfg["hf_cache_root"] = str(hf_cache_root)
    if device:
        cfg["device"] = device
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate segment-classifier variants on held-out multitrack sessions.",
    )
    parser.add_argument("--input-path", required=True, help="Training ZIP root or specific ZIP file.")
    parser.add_argument(
        "--extra-input-path",
        action="append",
        default=[],
        help="Additional training ZIP roots.",
    )
    parser.add_argument("--speaker-mapping", required=True, help="Speaker mapping YAML/JSON path.")
    parser.add_argument(
        "--base-profile-dir",
        required=True,
        help="Existing speaker-bank profile directory that already contains bank.json and embeddings.npy.",
    )
    parser.add_argument("--output-root", required=True, help="Directory for variant profiles and reports.")
    parser.add_argument(
        "--eval-session",
        action="append",
        default=[],
        help="Held-out eval target as name::/path/to/session.zip::/path/to/transcript",
    )
    parser.add_argument(
        "--allowed-speaker",
        action="append",
        default=[],
        help="Restrict training to these canonical speakers.",
    )
    parser.add_argument(
        "--speaker-alias",
        action="append",
        default=[],
        help="Extra speaker alias as source=target.",
    )
    parser.add_argument("--config", help="Optional base transcriber config used for eval runs.")
    parser.add_argument("--device", default="cuda", help="Force device for training/eval.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hop-seconds", type=float, default=120.0)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--min-share", type=float, default=0.72)
    parser.add_argument("--min-power", type=float, default=2e-4)
    parser.add_argument("--min-segment-dur", type=float, default=0.6)
    parser.add_argument("--max-segment-dur", type=float, default=20.0)
    parser.add_argument("--max-samples-per-speaker", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-speakers", type=int, action="append", default=[4])
    parser.add_argument(
        "--model-name",
        action="append",
        default=["logreg", "logreg_unbalanced", "knn"],
        help="Classifier model name to sweep.",
    )
    parser.add_argument("--classifier-c", type=float, action="append", default=[1.0, 4.0])
    parser.add_argument("--classifier-n-neighbors", type=int, action="append", default=[7, 15])
    parser.add_argument("--classifier-min-confidence", type=float, action="append", default=[0.0, 0.2])
    parser.add_argument("--classifier-min-margin", type=float, action="append", default=[0.08, 0.15])
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    base_profile_dir = Path(args.base_profile_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(args.speaker_mapping).expanduser().resolve()
    mapping = _load_yaml_or_json(str(mapping_path)) or {}
    base_config = _load_yaml_or_json(str(Path(args.config).expanduser().resolve())) if args.config else {}
    eval_sessions = _parse_eval_session(args.eval_session)
    speaker_aliases = _parse_aliases(args.speaker_alias)

    report: Dict[str, object] = {
        "input_path": str(Path(args.input_path).expanduser()),
        "extra_input_paths": [str(Path(path).expanduser()) for path in args.extra_input_path],
        "base_profile_dir": str(base_profile_dir),
        "allowed_speakers": list(args.allowed_speaker),
        "eval_sessions": [
            {
                "name": entry["name"],
                "session_zip": str(entry["session_zip"]),
                "session_jsonl": str(entry["session_jsonl"]),
            }
            for entry in eval_sessions
        ],
        "experiments": [],
    }

    hf_token = (
        os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    hf_cache_root = base_config.get("hf_cache_root")

    experiment_index = 0
    for (
        model_name,
        min_speakers,
        classifier_c,
        classifier_n_neighbors,
        classifier_min_confidence,
        classifier_min_margin,
    ) in product(
        dict.fromkeys(args.model_name),
        dict.fromkeys(args.min_speakers),
        dict.fromkeys(args.classifier_c),
        dict.fromkeys(args.classifier_n_neighbors),
        dict.fromkeys(args.classifier_min_confidence),
        dict.fromkeys(args.classifier_min_margin),
    ):
        if model_name != "knn" and classifier_n_neighbors != args.classifier_n_neighbors[0]:
            continue
        if model_name == "knn" and classifier_c != args.classifier_c[0]:
            continue

        experiment_index += 1
        variant_name = (
            f"{experiment_index:02d}_{model_name}_mins{int(min_speakers)}"
            f"_c{str(classifier_c).replace('.', 'p')}_k{int(classifier_n_neighbors)}"
            f"_conf{str(classifier_min_confidence).replace('.', 'p')}"
            f"_margin{str(classifier_min_margin).replace('.', 'p')}"
        )
        variant_profile_dir = output_root / "profiles" / variant_name
        _copy_bank_profile(base_profile_dir, variant_profile_dir)

        training_summary = train_segment_classifier_from_multitrack(
            input_path=args.input_path,
            extra_input_paths=args.extra_input_path,
            profile_dir=variant_profile_dir,
            speaker_mapping=mapping,
            hf_token=hf_token,
            force_device=args.device,
            quiet=bool(args.quiet),
            top_k=args.top_k,
            hop_seconds=args.hop_seconds,
            min_speakers=min_speakers,
            min_share=args.min_share,
            min_power=args.min_power,
            min_segment_dur=args.min_segment_dur,
            max_segment_dur=args.max_segment_dur,
            max_samples_per_speaker=args.max_samples_per_speaker,
            batch_size=args.batch_size,
            workers=args.workers,
            window_seconds=args.window_seconds,
            speaker_aliases=speaker_aliases,
            allowed_speakers=args.allowed_speaker,
            model_name=model_name,
            classifier_c=classifier_c,
            classifier_n_neighbors=classifier_n_neighbors,
        )

        eval_config = _build_eval_config(
            base_config=base_config,
            speaker_mapping_path=mapping_path,
            speaker_bank_root=variant_profile_dir.parent,
            speaker_bank_profile=variant_profile_dir.name,
            classifier_min_confidence=classifier_min_confidence,
            classifier_min_margin=classifier_min_margin,
            hf_cache_root=str(hf_cache_root) if hf_cache_root else None,
            device=args.device,
        )
        eval_config_path = output_root / "profiles" / variant_name / "eval_config.json"
        eval_config_path.write_text(json.dumps(eval_config, indent=2), encoding="utf-8")

        eval_results: List[Dict[str, object]] = []
        for session in eval_sessions:
            session_output_dir = output_root / "eval" / variant_name / str(session["name"])
            summary = evaluate_multitrack_session(
                session_zip=session["session_zip"],
                session_jsonl=session["session_jsonl"],
                output_dir=session_output_dir,
                speaker_mapping_path=mapping_path,
                config_path=eval_config_path,
                window_seconds=args.window_seconds,
                hop_seconds=60.0,
                top_k=3,
                min_speakers=3,
                device_override=args.device,
                local_files_only_override=None,
            )
            eval_results.append(summary)

        accuracy_values = [
            float(window["metrics"]["accuracy"])
            for session in eval_results
            for window in session.get("results", [])
        ]
        coverage_values = [
            float(window["metrics"]["coverage"])
            for session in eval_results
            for window in session.get("results", [])
        ]
        entry = {
            "variant_name": variant_name,
            "model_name": model_name,
            "min_speakers": min_speakers,
            "classifier_c": classifier_c,
            "classifier_n_neighbors": classifier_n_neighbors,
            "classifier_min_confidence": classifier_min_confidence,
            "classifier_min_margin": classifier_min_margin,
            "training_summary": training_summary,
            "mean_accuracy": _mean(accuracy_values),
            "mean_coverage": _mean(coverage_values),
            "eval_results": eval_results,
        }
        report["experiments"].append(entry)
        print(
            json.dumps(
                {
                    "variant_name": variant_name,
                    "mean_accuracy": entry["mean_accuracy"],
                    "mean_coverage": entry["mean_coverage"],
                }
            ),
            flush=True,
        )

    report["experiments"] = sorted(
        report["experiments"],
        key=lambda item: (float(item["mean_accuracy"]), float(item["mean_coverage"])),
        reverse=True,
    )
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
