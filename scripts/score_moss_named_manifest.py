from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from score_moss_manifest import (
    _slice_counts,
    manifest_reference_segments,
    reference_texts,
    temporal_attributed_score,
)
from score_moss_speaker_attributed import prediction_texts, score_mapping


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def named_reference_segments(row: Mapping[str, object]) -> list[dict]:
    metadata = dict(row.get("metadata") or {})
    name_to_stream = {
        str(name): str(stream)
        for name, stream in dict(metadata.get("stable_session_speaker_ids") or {}).items()
    }
    stream_to_name = {stream: name for name, stream in name_to_stream.items()}
    if not stream_to_name:
        raise ValueError(
            f"No stable_session_speaker_ids for {metadata.get('cut_id', '<unknown-cut>')}"
        )
    segments = manifest_reference_segments(row)
    for segment in segments:
        stream = str(segment["speaker"])
        if stream not in stream_to_name:
            raise ValueError(f"No named reference mapping for {stream}")
        segment["speaker"] = stream_to_name[stream]
    return segments


def score_named_outputs(
    manifest_rows: Iterable[Mapping[str, object]],
    output_records: Iterable[Mapping[str, object]],
    binding_records: Iterable[Mapping[str, object]],
    *,
    binding_mode: str,
    brief_turn_seconds: float,
    temporal_tolerance_seconds: float = 1.0,
) -> dict:
    references = {}
    for row in manifest_rows:
        cut_id = str(dict(row.get("metadata") or {}).get("cut_id") or "")
        references[cut_id] = named_reference_segments(row)
    bindings = {str(record.get("cut_id") or ""): record for record in binding_records}

    totals = defaultdict(int)
    slice_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    temporal_totals = defaultdict(int)
    temporal_slice_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    records = []
    for output in output_records:
        cut_id = str(output.get("cut_id") or "")
        reference_segments = references.get(cut_id)
        binding = bindings.get(cut_id)
        if reference_segments is None or binding is None:
            raise ValueError(f"Missing named reference or binding for {cut_id}")
        mapping = {
            str(stream): str(speaker)
            for stream, speaker in dict(binding.get(f"{binding_mode}_mapping") or {}).items()
        }
        named_segments = [
            {
                **dict(segment),
                "speaker": mapping.get(str(segment.get("speaker") or ""), "unknown"),
            }
            for segment in list(output.get("segments") or [])
        ]
        refs = reference_texts(reference_segments)
        preds = prediction_texts(named_segments)
        identity_mapping = {speaker: speaker for speaker in preds if speaker in refs}
        score = score_mapping(preds, refs, identity_mapping)
        for key in (
            "reference_words",
            "predicted_words",
            "bag_matches",
            "sequence_matches",
            "sequence_prediction_matches",
        ):
            totals[key] += int(score[key])
        slices = _slice_counts(
            reference_segments,
            preds,
            identity_mapping,
            brief_turn_seconds=brief_turn_seconds,
        )
        for category, values in slices.items():
            for key, value in values.items():
                slice_totals[category][key] += value
        temporal = temporal_attributed_score(
            reference_segments,
            named_segments,
            {speaker: speaker for speaker in preds if speaker in refs},
            brief_turn_seconds=brief_turn_seconds,
            tolerance_seconds=temporal_tolerance_seconds,
        )
        for key in ("reference_words", "predicted_words", "matched_words"):
            temporal_totals[key] += int(temporal[key])
        for category, values in temporal["categories"].items():
            for key in ("reference_words", "matched_words"):
                temporal_slice_totals[category][key] += int(values[key])
        records.append(
            {
                "cut_id": cut_id,
                "binding_mapping": mapping,
                **{key: value for key, value in score.items() if key != "records"},
            }
        )
    for values in slice_totals.values():
        values["recall"] = values["matched_words"] / max(1, values["reference_words"])
    for values in temporal_slice_totals.values():
        values["recall"] = values["matched_words"] / max(1, values["reference_words"])
    return {
        "record_count": len(records),
        **dict(totals),
        "attributed_bag_recall": totals["bag_matches"] / max(1, totals["reference_words"]),
        "attributed_bag_precision": totals["bag_matches"] / max(1, totals["predicted_words"]),
        "attributed_sequence_recall": totals["sequence_matches"]
        / max(1, totals["reference_words"]),
        "attributed_sequence_precision": totals["sequence_prediction_matches"]
        / max(1, totals["predicted_words"]),
        "attributed_temporal_recall": temporal_totals["matched_words"]
        / max(1, temporal_totals["reference_words"]),
        "attributed_temporal_precision": temporal_totals["matched_words"]
        / max(1, temporal_totals["predicted_words"]),
        "temporal_matches": temporal_totals["matched_words"],
        "temporal_tolerance_seconds": temporal_tolerance_seconds,
        "brief_turn_seconds": brief_turn_seconds,
        "categories": {key: dict(value) for key, value in slice_totals.items()},
        "temporal_categories": {key: dict(value) for key, value in temporal_slice_totals.items()},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score time-local named MOSS output using historical enrollment binding."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--binding-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binding-mode", choices=("independent", "one_to_one"), default="one_to_one"
    )
    parser.add_argument("--brief-turn-seconds", type=float, default=2.0)
    parser.add_argument("--temporal-tolerance-seconds", type=float, default=1.0)
    args = parser.parse_args()

    model_output = json.loads(args.model_output.read_text(encoding="utf-8"))
    binding_output = json.loads(args.binding_output.read_text(encoding="utf-8"))
    if bool(binding_output.get("inference_uses_isolated_target_audio")):
        raise ValueError("Binding output used isolated evaluation audio")
    provenance = dict(binding_output.get("enrollment_provenance") or {})
    if bool(provenance.get("uses_evaluation_session_audio", True)):
        raise ValueError("Binding output used evaluation-session enrollment")
    score = score_named_outputs(
        _read_jsonl(args.manifest),
        list(model_output.get("records") or []),
        list(binding_output.get("record_bindings") or []),
        binding_mode=args.binding_mode,
        brief_turn_seconds=args.brief_turn_seconds,
        temporal_tolerance_seconds=args.temporal_tolerance_seconds,
    )
    payload = {
        "manifest": str(args.manifest),
        "model_output": str(args.model_output),
        "binding_output": str(args.binding_output),
        "historical_cross_session_enrollment_only": True,
        "mono_only_inference": True,
        **score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
