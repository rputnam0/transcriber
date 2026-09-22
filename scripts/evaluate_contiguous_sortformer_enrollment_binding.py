from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_contiguous_sortformer_mask_cutset import group_contiguous_cuts
from evaluate_sortformer_enrollment_binding import (
    _embed_waveforms,
    _load_titanet,
    _load_wave,
    _read_jsonl,
    _resolve_path,
    _score_mapping,
    _session_name,
    assignment_maps,
    enrollment_leave_one_out_score,
    enrollment_paths,
    enrollment_provenance,
    slot_waveforms,
    trim_clean_enrollment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind persistent contiguous Sortformer slots using full-run mono evidence."
    )
    parser.add_argument("--activity-cutset", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--exclusivity-margin", type=float, default=0.10)
    parser.add_argument("--min-slot-seconds", type=float, default=0.5)
    parser.add_argument("--max-enrollment-clips", type=int, default=4)
    args = parser.parse_args()

    class CutRecord:
        def __init__(self, row: dict):
            self.id = str(row["id"]).removesuffix("-mask-sortformer")
            self.row = row
            self.duration = float(row.get("duration") or 0.0)

    records_by_id = {
        str(row["id"]).removesuffix("-mask-sortformer"): row
        for row in _read_jsonl(args.activity_cutset)
    }
    runs = group_contiguous_cuts([CutRecord(row) for row in records_by_id.values()])
    model = _load_titanet(args.device)
    enrollment_cache = {}
    output_records = []
    for run_index, run in enumerate(runs):
        session = _session_name(run[0].id)
        if session not in enrollment_cache:
            paths_by_speaker = enrollment_paths(
                args.enrollment_manifest,
                session=session,
                max_clips_per_speaker=args.max_enrollment_clips,
            )
            speakers = sorted(paths_by_speaker)
            labels = [speaker for speaker in speakers for _path in paths_by_speaker[speaker]]
            waves = [
                trim_clean_enrollment(_load_wave(path), sample_rate=16_000)
                for speaker in speakers
                for path in paths_by_speaker[speaker]
            ]
            vectors = _embed_waveforms(
                model,
                waves,
                sample_rate=16_000,
                batch_size=args.batch_size,
                device=args.device,
            )
            centroids = np.stack(
                [
                    _normalize_rows(
                        vectors[[label == speaker for label in labels]].mean(axis=0, keepdims=True)
                    )[0]
                    for speaker in speakers
                ]
            )
            enrollment_cache[session] = {
                "speakers": speakers,
                "centroids": centroids,
                "sources": {
                    speaker: [str(path) for path in paths_by_speaker[speaker]]
                    for speaker in speakers
                },
                "diagnostic": enrollment_leave_one_out_score(vectors, labels),
                "provenance": enrollment_provenance(
                    args.enrollment_manifest,
                    session=session,
                ),
            }
        enrollment = enrollment_cache[session]
        mono_parts = []
        probability_parts = []
        cut_details = []
        for cut in run:
            row = records_by_id[cut.id]
            source = list(dict(row["recording"])["sources"])[0]
            audio_path = _resolve_path(
                source["source"],
                relative_to=args.activity_cutset.resolve().parent,
            )
            custom = dict(row.get("custom") or {})
            probability_path = _resolve_path(
                custom.get("sortformer_probabilities_path"),
                relative_to=args.activity_cutset.resolve().parent,
            )
            mono_parts.append(_load_wave(audio_path))
            probability_parts.append(np.load(probability_path, allow_pickle=False))
            cut_details.append((cut.id, row, audio_path, probability_path))
        mixture = np.concatenate(mono_parts)
        probabilities = np.concatenate(probability_parts, axis=0)
        slots, waves, slot_metadata = slot_waveforms(
            mixture,
            probabilities,
            sample_rate=16_000,
            activity_threshold=args.activity_threshold,
            exclusivity_margin=args.exclusivity_margin,
            min_slot_seconds=args.min_slot_seconds,
        )
        slot_vectors = _embed_waveforms(
            model,
            waves,
            sample_rate=16_000,
            batch_size=args.batch_size,
            device=args.device,
        )
        similarity = _normalize_rows(slot_vectors) @ _normalize_rows(enrollment["centroids"]).T
        independent, one_to_one, evidence = assignment_maps(
            similarity,
            slots,
            enrollment["speakers"],
        )
        for cut_id, row, audio_path, probability_path in cut_details:
            expected = dict(dict(row.get("custom") or {}).get("sortformer_slot_to_speaker") or {})
            output_records.append(
                {
                    "cut_id": cut_id,
                    "session": session,
                    "contiguous_run": run_index,
                    "mono_audio_path": str(audio_path),
                    "probability_path": str(probability_path),
                    "inference_uses_isolated_target_audio": False,
                    "enrollment_audio_source": enrollment["provenance"]["source"],
                    "enrollment_provenance": enrollment["provenance"],
                    "enrollment_sources": enrollment["sources"],
                    "enrollment_leave_one_out": enrollment["diagnostic"],
                    "slots": slot_metadata,
                    "independent_mapping": independent,
                    "one_to_one_mapping": one_to_one,
                    "binding_evidence": evidence,
                    "oracle_mapping_diagnostic_only": expected,
                    "independent_score": _score_mapping(independent, expected),
                    "one_to_one_score": _score_mapping(one_to_one, expected),
                }
            )
        print(f"bound contiguous run {run_index + 1}/{len(runs)}", flush=True)

    summary = {}
    for mode in ("independent", "one_to_one"):
        scores = [record[f"{mode}_score"] for record in output_records]
        reference = sum(int(score["reference_slots"]) for score in scores)
        correct = sum(int(score["correct_slots"]) for score in scores)
        summary[mode] = {
            "reference_slots": reference,
            "mapped_reference_slots": sum(int(score["mapped_reference_slots"]) for score in scores),
            "correct_slots": correct,
            "accuracy": correct / max(1, reference),
        }
    result = {
        "activity_cutset": str(args.activity_cutset),
        "enrollment_manifest": str(args.enrollment_manifest),
        "activity_source": "mono-sortformer-contiguous-soft",
        "binding_scope": "full-contiguous-run",
        "binding_model": "titanet_small",
        "cut_count": len(output_records),
        "contiguous_run_count": len(runs),
        "summary": summary,
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


if __name__ == "__main__":
    main()
