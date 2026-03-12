from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.hard_negatives import build_hard_negative_dataset  # noqa: E402
from transcriber.prep_artifacts import (  # noqa: E402
    build_artifact_manifest,
    build_coverage_report,
    build_source_session_speaker_breakdown,
    collect_input_file_identities,
    current_git_commit,
    save_manifest,
)
from transcriber.segment_classifier import load_classifier_dataset, save_classifier_dataset  # noqa: E402


def _parse_pair(value: str) -> Tuple[str, str]:
    left, sep, right = value.partition("::")
    if not sep or not left.strip() or not right.strip():
        raise argparse.ArgumentTypeError(
            "Seed pairs must look like 'Speaker A::Speaker B'."
        )
    return left.strip(), right.strip()


def _write_json(path: Path, payload: object) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _write_jsonl(path: Path, records: List[dict]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine hard-negative supplement examples from eval and mixed-dataset artifacts."
    )
    parser.add_argument(
        "--eval-summary",
        dest="eval_summaries",
        type=Path,
        action="append",
        required=True,
        help="Path to an evaluate_multitrack_session summary.json file. Repeat for multiple sessions.",
    )
    parser.add_argument(
        "--candidate-pool-dir",
        dest="candidate_pool_dirs",
        type=Path,
        action="append",
        required=True,
        help="Dataset directory containing candidate_pool.jsonl and candidate_pool_embeddings.npz.",
    )
    parser.add_argument(
        "--base-dataset-dir",
        type=Path,
        required=True,
        help="Balanced base training dataset directory used to set the supplement cap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the mined hard-negative dataset and reports should be written.",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="Diarization model name to record in the hard-negative manifest.",
    )
    parser.add_argument(
        "--seed-pair",
        dest="seed_pairs",
        type=_parse_pair,
        action="append",
        default=[],
        help="Explicit confusion pair to track, written as 'Speaker A::Speaker B'. Repeat as needed.",
    )
    parser.add_argument("--top-confusion-pairs", type=int, default=5)
    parser.add_argument("--max-eval-margin", type=float, default=0.12)
    parser.add_argument("--min-mixed-dominant-share", type=float, default=0.55)
    parser.add_argument("--per-pair-cap", type=int, default=75)
    parser.add_argument("--max-fraction", type=float, default=0.20)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    eval_summaries = [
        json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        for path in args.eval_summaries
    ]
    base_dataset, base_summary = load_classifier_dataset(args.base_dataset_dir)
    hard_negative_dataset, metadata_records, mining_summary = build_hard_negative_dataset(
        eval_summaries=eval_summaries,
        candidate_pool_dirs=[Path(path).expanduser() for path in args.candidate_pool_dirs],
        base_dataset_samples=base_dataset.samples,
        seed_pairs=[list(pair) for pair in args.seed_pairs],
        top_confusion_pairs=int(args.top_confusion_pairs),
        max_eval_margin=float(args.max_eval_margin),
        min_mixed_dominant_share=float(args.min_mixed_dominant_share),
        per_pair_cap=int(args.per_pair_cap),
        max_fraction=float(args.max_fraction),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_json(output_dir / "hard_negative_summary.json", mining_summary)
    if hard_negative_dataset is None:
        print(
            json.dumps(
                {
                    "dataset_created": False,
                    "summary_path": str(summary_path),
                },
                indent=2,
            )
        )
        return 0

    base_artifact_id = str(base_summary.get("artifact_id") or "").strip()
    manifest = build_artifact_manifest(
        artifact_type="dataset",
        diarization_model=str(args.diarization_model),
        source_sessions=sorted({str(item["source_session"]) for item in metadata_records}),
        input_file_identities=collect_input_file_identities(
            [Path(path).expanduser() for path in args.eval_summaries]
            + [Path(path).expanduser() for path in args.candidate_pool_dirs]
            + [Path(args.base_dataset_dir).expanduser()],
            hash_contents=False,
        ),
        build_params={
            "variant": "hard_negative",
            "seed_pairs": [list(pair) for pair in args.seed_pairs],
            "top_confusion_pairs": int(args.top_confusion_pairs),
            "max_eval_margin": float(args.max_eval_margin),
            "min_mixed_dominant_share": float(args.min_mixed_dominant_share),
            "per_pair_cap": int(args.per_pair_cap),
            "max_fraction": float(args.max_fraction),
            "tracked_pairs": mining_summary.get("tracked_pairs"),
        },
        parent_artifacts=[base_artifact_id] if base_artifact_id else [],
        git_commit=current_git_commit(cwd=REPO_ROOT),
    )
    save_classifier_dataset(
        output_dir,
        hard_negative_dataset,
        summary={
            "artifact_id": str(manifest["artifact_id"]),
            "parent_artifacts": manifest["parent_artifacts"],
            "hard_negative": mining_summary,
            "quality_filters": {},
            "source_groups": {},
            "breakdown": build_source_session_speaker_breakdown(hard_negative_dataset),
        },
    )
    records_path = _write_jsonl(output_dir / "hard_negative_records.jsonl", metadata_records)
    coverage_report_path = _write_json(
        output_dir / "coverage_report.json",
        build_coverage_report(hard_negative_dataset),
    )
    manifest = {
        **manifest,
        "artifact_dir": str(output_dir),
        "dataset_dir": str(output_dir),
        "dataset_summary_path": str(output_dir / "dataset_summary.json"),
        "records_path": str(records_path),
        "coverage_report_path": str(coverage_report_path),
        "summary_path": str(summary_path),
    }
    save_manifest(output_dir / "hard_negative_manifest.json", manifest)
    print(
        json.dumps(
            {
                "dataset_created": True,
                "artifact_id": manifest["artifact_id"],
                "manifest_path": str(output_dir / "hard_negative_manifest.json"),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
