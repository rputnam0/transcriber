from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evaluate_se_dicow_oracle_cut import normalized_words
from score_moss_speaker_attributed import (
    _optimal_unique_mapping,
    prediction_texts,
    score_mapping,
)
from score_se_dicow_recovery_slices import _lcs_reference_indices


TARGET_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\]\s*"
    r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_target(text: str) -> list[dict]:
    return [
        {
            "start": float(match.group("start")),
            "end": float(match.group("end")),
            "speaker": match.group("speaker"),
            "text": match.group("text").strip(),
        }
        for match in TARGET_RE.finditer(text)
        if float(match.group("end")) > float(match.group("start")) and match.group("text").strip()
    ]


def manifest_reference_segments(row: Mapping[str, object]) -> list[dict]:
    conversation = list(row.get("conversation") or [])
    if len(conversation) != 3:
        raise ValueError("Malformed MOSS manifest conversation")
    segments = parse_target(str(conversation[2].get("content") or ""))
    authored = list(dict(row.get("metadata") or {}).get("reference_segments") or [])
    if authored:
        if len(authored) != len(segments):
            raise ValueError("Authored reference-segment metadata does not match target")
        for segment, metadata in zip(segments, authored):
            if (
                str(segment["speaker"]) != str(metadata.get("speaker") or "")
                or abs(float(segment["start"]) - float(metadata.get("start") or 0.0)) > 0.011
                or abs(float(segment["end"]) - float(metadata.get("end") or 0.0)) > 0.011
            ):
                raise ValueError("Authored reference-segment metadata is out of sync")
            segment["overlap"] = bool(metadata.get("overlap"))
            segment["brief"] = bool(metadata.get("brief"))
            segment["word_spans"] = list(metadata.get("word_spans") or [])
    return segments


def reference_texts(segments: Iterable[Mapping[str, object]]) -> dict[str, str]:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for segment in segments:
        grouped[str(segment["speaker"])].append((float(segment["start"]), str(segment["text"])))
    return {
        speaker: " ".join(text for _start, text in sorted(parts))
        for speaker, parts in sorted(grouped.items())
    }


def _slice_counts(
    reference_segments: Sequence[Mapping[str, object]],
    predictions: Mapping[str, str],
    mapping: Mapping[str, str],
    *,
    brief_turn_seconds: float,
) -> dict:
    named_predictions = {
        reference: predictions[predicted]
        for predicted, reference in mapping.items()
        if predicted in predictions
    }
    categories = {
        name: {"reference_words": 0, "matched_words": 0}
        for name in ("all", "ordinary_nonoverlap", "overlap", "brief_turn", "brief_overlap")
    }
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for segment in reference_segments:
        speaker = str(segment["speaker"])
        start = float(segment["start"])
        end = float(segment["end"])
        overlap = any(
            str(other["speaker"]) != speaker
            and float(other["start"]) < end
            and float(other["end"]) > start
            for other in reference_segments
        )
        brief = end - start <= brief_turn_seconds
        for token in normalized_words(str(segment["text"])):
            by_speaker[speaker].append({"token": token, "overlap": overlap, "brief": brief})
    for speaker, items in by_speaker.items():
        matched = _lcs_reference_indices(
            [item["token"] for item in items],
            normalized_words(named_predictions.get(speaker, "")),
        )
        for index, item in enumerate(items):
            selected = {
                "all": True,
                "ordinary_nonoverlap": not item["overlap"],
                "overlap": item["overlap"],
                "brief_turn": item["brief"],
                "brief_overlap": item["brief"] and item["overlap"],
            }
            for category, enabled in selected.items():
                if enabled:
                    categories[category]["reference_words"] += 1
                    categories[category]["matched_words"] += int(index in matched)
    return categories


def _timed_tokens(
    segments: Sequence[Mapping[str, object]],
    *,
    include_categories: bool,
    brief_turn_seconds: float,
) -> dict[str, list[dict]]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    ordered = sorted(
        segments,
        key=lambda segment: (
            float(segment.get("start") or 0.0),
            float(segment.get("end") or 0.0),
            str(segment.get("speaker") or ""),
        ),
    )
    for segment in ordered:
        speaker = str(segment.get("speaker") or "").strip()
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if not speaker or end <= start:
            continue
        word_spans = list(segment.get("word_spans") or []) if include_categories else []
        if word_spans:
            for word in word_spans:
                word_start = float(word.get("start") or start)
                word_end = float(word.get("end") or word_start)
                for token in normalized_words(str(word.get("text") or "")):
                    by_speaker[speaker].append(
                        {
                            "token": token,
                            "start": word_start,
                            "end": word_end,
                            "overlap": bool(word.get("overlap")),
                            "brief": bool(word.get("brief")),
                        }
                    )
            continue
        overlap = False
        brief = False
        if include_categories:
            overlap = (
                bool(segment.get("overlap"))
                if "overlap" in segment
                else any(
                    str(other.get("speaker") or "") != speaker
                    and float(other.get("start") or 0.0) < end
                    and float(other.get("end") or 0.0) > start
                    for other in ordered
                )
            )
            brief = (
                bool(segment.get("brief"))
                if "brief" in segment
                else end - start <= brief_turn_seconds
            )
        for token in normalized_words(str(segment.get("text") or "")):
            by_speaker[speaker].append(
                {
                    "token": token,
                    "start": start,
                    "end": end,
                    "overlap": overlap,
                    "brief": brief,
                }
            )
    return by_speaker


def _temporal_lcs_indices(
    reference: Sequence[Mapping[str, object]],
    prediction: Sequence[Mapping[str, object]],
    *,
    tolerance_seconds: float,
) -> tuple[set[int], set[int]]:
    """Match equal tokens in sequence only when their timestamp spans are nearby."""

    row_count = len(reference)
    column_count = len(prediction)
    scores = [[0] * (column_count + 1) for _ in range(row_count + 1)]

    def compatible(row: int, column: int) -> bool:
        left = reference[row]
        right = prediction[column]
        return (
            left["token"] == right["token"]
            and float(left["start"]) <= float(right["end"]) + tolerance_seconds
            and float(right["start"]) <= float(left["end"]) + tolerance_seconds
        )

    for row in range(1, row_count + 1):
        for column in range(1, column_count + 1):
            match_score = scores[row - 1][column - 1] + int(compatible(row - 1, column - 1))
            scores[row][column] = max(
                scores[row - 1][column],
                scores[row][column - 1],
                match_score,
            )

    reference_matches: set[int] = set()
    prediction_matches: set[int] = set()
    row = row_count
    column = column_count
    while row and column:
        if (
            compatible(row - 1, column - 1)
            and scores[row][column] == scores[row - 1][column - 1] + 1
        ):
            reference_matches.add(row - 1)
            prediction_matches.add(column - 1)
            row -= 1
            column -= 1
        elif scores[row - 1][column] >= scores[row][column - 1]:
            row -= 1
        else:
            column -= 1
    return reference_matches, prediction_matches


def temporal_attributed_score(
    reference_segments: Sequence[Mapping[str, object]],
    prediction_segments: Sequence[Mapping[str, object]],
    mapping: Mapping[str, str],
    *,
    brief_turn_seconds: float,
    tolerance_seconds: float,
) -> dict:
    """Score token, speaker, and local time jointly to avoid distant repeated-word credit."""

    references = _timed_tokens(
        reference_segments,
        include_categories=True,
        brief_turn_seconds=brief_turn_seconds,
    )
    raw_predictions = _timed_tokens(
        prediction_segments,
        include_categories=False,
        brief_turn_seconds=brief_turn_seconds,
    )
    predictions: dict[str, list[dict]] = defaultdict(list)
    for predicted_speaker, items in raw_predictions.items():
        named_speaker = str(mapping.get(predicted_speaker) or "").strip()
        if named_speaker:
            predictions[named_speaker].extend(items)

    categories = {
        name: {"reference_words": 0, "matched_words": 0}
        for name in ("all", "ordinary_nonoverlap", "overlap", "brief_turn", "brief_overlap")
    }
    matched_words = 0
    for speaker, reference_items in references.items():
        prediction_items = sorted(
            predictions.get(speaker, []),
            key=lambda item: (float(item["start"]), float(item["end"])),
        )
        reference_matches, _prediction_matches = _temporal_lcs_indices(
            reference_items,
            prediction_items,
            tolerance_seconds=tolerance_seconds,
        )
        matched_words += len(reference_matches)
        for index, item in enumerate(reference_items):
            selected = {
                "all": True,
                "ordinary_nonoverlap": not item["overlap"],
                "overlap": item["overlap"],
                "brief_turn": item["brief"],
                "brief_overlap": item["brief"] and item["overlap"],
            }
            for category, enabled in selected.items():
                if enabled:
                    categories[category]["reference_words"] += 1
                    categories[category]["matched_words"] += int(index in reference_matches)

    for values in categories.values():
        values["recall"] = values["matched_words"] / max(1, values["reference_words"])
    reference_words = sum(len(items) for items in references.values())
    predicted_words = sum(len(items) for items in raw_predictions.values())
    return {
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "matched_words": matched_words,
        "recall": matched_words / max(1, reference_words),
        "precision": matched_words / max(1, predicted_words),
        "tolerance_seconds": tolerance_seconds,
        "categories": categories,
    }


def score_outputs(
    manifest_rows: Iterable[Mapping[str, object]],
    output_records: Iterable[Mapping[str, object]],
    *,
    brief_turn_seconds: float = 2.0,
    temporal_tolerance_seconds: float = 1.0,
) -> dict:
    references = {}
    for row in manifest_rows:
        metadata = dict(row.get("metadata") or {})
        cut_id = str(metadata.get("cut_id") or "")
        if cut_id in references:
            continue
        references[cut_id] = manifest_reference_segments(row)

    totals = defaultdict(int)
    slice_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    temporal_totals = defaultdict(int)
    temporal_slice_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    records = []
    for output in output_records:
        cut_id = str(output.get("cut_id") or "")
        reference_segments = references.get(cut_id)
        if reference_segments is None:
            raise ValueError(f"No manifest reference for {cut_id}")
        refs = reference_texts(reference_segments)
        preds = prediction_texts(list(output.get("segments") or []))
        mapping = _optimal_unique_mapping(preds, refs)
        score = score_mapping(preds, refs, mapping)
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
            mapping,
            brief_turn_seconds=brief_turn_seconds,
        )
        for category, values in slices.items():
            for key, value in values.items():
                slice_totals[category][key] += value
        temporal = temporal_attributed_score(
            reference_segments,
            list(output.get("segments") or []),
            mapping,
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
                "oracle_mapping": mapping,
                "reference_speakers": len(refs),
                "predicted_speakers": len(preds),
                **{key: value for key, value in score.items() if key != "records"},
            }
        )
    for values in slice_totals.values():
        values["recall"] = values["matched_words"] / max(1, values["reference_words"])
    for values in temporal_slice_totals.values():
        values["recall"] = values["matched_words"] / max(1, values["reference_words"])
    return {
        "record_count": len(records),
        "oracle_mapping_diagnostic_only": True,
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
        description="Score MOSS manifest decoding with a per-cut oracle speaker permutation."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brief-turn-seconds", type=float, default=2.0)
    parser.add_argument("--temporal-tolerance-seconds", type=float, default=1.0)
    args = parser.parse_args()
    model_output = json.loads(args.model_output.read_text(encoding="utf-8"))
    score = score_outputs(
        _read_jsonl(args.manifest),
        list(model_output.get("records") or []),
        brief_turn_seconds=args.brief_turn_seconds,
        temporal_tolerance_seconds=args.temporal_tolerance_seconds,
    )
    payload = {
        "manifest": str(args.manifest),
        "model_output": str(args.model_output),
        **score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
