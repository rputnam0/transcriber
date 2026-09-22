from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evaluate_se_dicow_oracle_cut import attributed_bag_metrics, attributed_sequence_metrics


CUT_ID_RE = re.compile(r"session_(\d+)_w(\d+)_c(\d+)")


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _cut_order(cut: Mapping[str, object]) -> tuple[int, int, int]:
    match = CUT_ID_RE.match(str(cut.get("id") or ""))
    if not match:
        raise ValueError(f"Cannot order cut id {cut.get('id')!r}")
    return tuple(int(value) for value in match.groups())


def _reference_spans(cut: Mapping[str, object]) -> list[dict]:
    custom = dict(cut.get("custom") or {})
    spans = list(custom.get("transcript_spans") or [])
    if spans:
        return [dict(span) for span in spans]
    return [
        {
            "speaker": supervision.get("speaker"),
            "text": supervision.get("text"),
        }
        for supervision in list(cut.get("supervisions") or [])
    ]


def reference_texts(cuts: Sequence[Mapping[str, object]]) -> dict[str, str]:
    texts: dict[str, list[str]] = defaultdict(list)
    for cut in sorted(cuts, key=_cut_order):
        for span in _reference_spans(cut):
            speaker = str(span.get("speaker") or "").strip()
            text = str(span.get("text") or "").strip()
            if speaker and text:
                texts[speaker].append(text)
    return {speaker: " ".join(parts) for speaker, parts in sorted(texts.items())}


def prediction_texts(segments: Iterable[Mapping[str, object]]) -> dict[str, str]:
    texts: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for segment in segments:
        speaker = str(segment.get("speaker") or "").strip()
        text = str(segment.get("text") or "").strip()
        if speaker and text:
            texts[speaker].append((float(segment.get("start") or 0.0), text))
    return {
        speaker: " ".join(text for _start, text in sorted(parts))
        for speaker, parts in sorted(texts.items())
    }


def _optimal_unique_mapping(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
) -> dict[str, str]:
    predicted_speakers = sorted(predictions)
    reference_speakers = sorted(references)
    if not predicted_speakers or not reference_speakers:
        return {}

    # Add zero-value dummy references when the model emits more streams than the roster.
    columns: list[str | None] = list(reference_speakers)
    columns.extend([None] * max(0, len(predicted_speakers) - len(reference_speakers)))
    weights = [
        [
            (
                int(attributed_bag_metrics(references[name], predictions[speaker])["bag_matches"])
                if name is not None
                else 0
            )
            for name in columns
        ]
        for speaker in predicted_speakers
    ]

    @lru_cache(maxsize=None)
    def solve(row: int, used: int) -> tuple[int, tuple[int, ...]]:
        if row == len(predicted_speakers):
            return 0, ()
        best_score = -1
        best_assignment: tuple[int, ...] = ()
        for column in range(len(columns)):
            if used & (1 << column):
                continue
            remainder, assignment = solve(row + 1, used | (1 << column))
            score = weights[row][column] + remainder
            candidate = (column,) + assignment
            if score > best_score or (score == best_score and candidate < best_assignment):
                best_score = score
                best_assignment = candidate
        return best_score, best_assignment

    _score, assignment = solve(0, 0)
    return {
        speaker: columns[column]
        for speaker, column in zip(predicted_speakers, assignment)
        if columns[column] is not None
    }


def score_mapping(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
    mapping: Mapping[str, str],
) -> dict:
    mapped_predictions: dict[str, list[str]] = defaultdict(list)
    for predicted_speaker, text in predictions.items():
        named_speaker = str(mapping.get(predicted_speaker) or "").strip()
        if named_speaker:
            mapped_predictions[named_speaker].append(text)

    records = []
    for speaker, reference in references.items():
        prediction = " ".join(mapped_predictions.pop(speaker, [])).strip()
        records.append(
            {
                "speaker": speaker,
                "reference": reference,
                "prediction": prediction,
                **attributed_bag_metrics(reference, prediction),
                **attributed_sequence_metrics(reference, prediction),
            }
        )

    # Wrong-roster names and unassigned streams must still count against precision.
    accounted = set(mapping)
    extras = {f"mapped:{speaker}": " ".join(parts) for speaker, parts in mapped_predictions.items()}
    extras.update(
        {
            f"unmapped:{speaker}": text
            for speaker, text in predictions.items()
            if speaker not in accounted
        }
    )
    for speaker, prediction in sorted(extras.items()):
        records.append(
            {
                "speaker": speaker,
                "reference": "",
                "prediction": prediction,
                **attributed_bag_metrics("", prediction),
                **attributed_sequence_metrics("", prediction),
            }
        )

    totals = {
        key: sum(int(record[key]) for record in records)
        for key in (
            "reference_words",
            "predicted_words",
            "bag_matches",
            "sequence_matches",
            "sequence_prediction_matches",
        )
    }
    return {
        "records": records,
        **totals,
        "attributed_bag_recall": totals["bag_matches"] / max(1, totals["reference_words"]),
        "attributed_bag_precision": totals["bag_matches"] / max(1, totals["predicted_words"]),
        "attributed_sequence_recall": totals["sequence_matches"]
        / max(1, totals["reference_words"]),
        "attributed_sequence_precision": totals["sequence_prediction_matches"]
        / max(1, totals["predicted_words"]),
    }


def materialize_cut_results(
    cuts: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
    mapping: Mapping[str, str],
    output_dir: Path,
    *,
    mapping_is_oracle: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_cuts = sorted(cuts, key=_cut_order)
    for cut in ordered_cuts:
        session, _window, cut_milliseconds = _cut_order(cut)
        cut_start = cut_milliseconds / 1000.0
        cut_end = cut_start + float(cut.get("duration") or 0.0)
        references: dict[str, list[str]] = defaultdict(list)
        for span in _reference_spans(cut):
            speaker = str(span.get("speaker") or "").strip()
            text = str(span.get("text") or "").strip()
            if speaker and text:
                references[speaker].append(text)
        predictions: dict[str, list[str]] = defaultdict(list)
        for segment in segments:
            midpoint = (float(segment.get("start") or 0.0) + float(segment.get("end") or 0.0)) / 2
            if not cut_start <= midpoint < cut_end:
                continue
            speaker = str(mapping.get(str(segment.get("speaker") or "")) or "").strip()
            text = str(segment.get("text") or "").strip()
            if speaker and text:
                predictions[speaker].append(text)
        speakers = sorted(set(references) | set(predictions))
        payload = {
            "cut_id": str(cut.get("id") or "").removesuffix("-mask-sortformer"),
            "session": f"Session {session}",
            "records": [
                {
                    "speaker": speaker,
                    "reference": " ".join(references.get(speaker, [])),
                    "prediction": " ".join(predictions.get(speaker, [])),
                }
                for speaker in speakers
            ],
            "mapping": dict(mapping),
            "mapping_is_oracle": mapping_is_oracle,
        }
        (output_dir / f"{payload['cut_id']}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score MOSS joint ASR/diarization with an oracle anonymous-speaker permutation."
    )
    parser.add_argument("--cutset", type=Path, required=True)
    parser.add_argument("--moss-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialized-results-dir", type=Path)
    parser.add_argument(
        "--binding-output",
        type=Path,
        help="Optional mono-stream enrollment binding JSON for a deployable named score.",
    )
    parser.add_argument("--named-results-dir", type=Path)
    args = parser.parse_args()

    cuts = list(_read_jsonl(args.cutset))
    moss = json.loads(args.moss_output.read_text(encoding="utf-8"))
    segments = list(moss.get("segments") or [])
    references = reference_texts(cuts)
    predictions = prediction_texts(segments)
    mapping = _optimal_unique_mapping(predictions, references)
    score = score_mapping(predictions, references, mapping)
    payload = {
        "cutset": str(args.cutset),
        "moss_output": str(args.moss_output),
        "scoring_scope": "contiguous-session-speaker-aggregate",
        "oracle_mapping_diagnostic_only": True,
        "oracle_mapping": mapping,
        "reference_speaker_count": len(references),
        "predicted_speaker_count": len(predictions),
        **score,
    }
    if args.binding_output:
        binding = json.loads(args.binding_output.read_text(encoding="utf-8"))
        if bool(binding.get("inference_uses_isolated_target_audio")):
            raise ValueError("Binding output used isolated evaluation audio")
        provenance = dict(binding.get("enrollment_provenance") or {})
        if bool(provenance.get("uses_evaluation_session_audio", True)):
            raise ValueError("Binding output used evaluation-session enrollment")
        enrollment_mapping = dict(binding.get("one_to_one_mapping") or {})
        if not enrollment_mapping:
            raise ValueError("Binding output has no one_to_one_mapping")
        payload["enrollment_mapping"] = enrollment_mapping
        payload["enrollment_mapping_matches_oracle"] = enrollment_mapping == mapping
        payload["enrollment_score"] = score_mapping(
            predictions,
            references,
            enrollment_mapping,
        )
        if args.named_results_dir:
            materialize_cut_results(
                cuts,
                segments,
                enrollment_mapping,
                args.named_results_dir,
                mapping_is_oracle=False,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.materialized_results_dir:
        materialize_cut_results(cuts, segments, mapping, args.materialized_results_dir)
    printed = {key: value for key, value in payload.items() if key != "records"}
    if "enrollment_score" in printed:
        printed["enrollment_score"] = {
            key: value
            for key, value in dict(printed["enrollment_score"]).items()
            if key != "records"
        }
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
