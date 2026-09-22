from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from train_tsvad_word_owner_baseline import (
    _group_manifest_rows,
    _load_reference_groups,
    _materialized_path,
    _read_jsonl,
    _split_values,
    _word_has_overlap,
    _write_jsonl,
)


def _choose_turn_label(
    start: float,
    end: float,
    turns: Sequence[Mapping[str, object]],
) -> str | None:
    midpoint = (start + end) / 2.0
    best_label = None
    best_overlap = 0.0
    midpoint_label = None
    midpoint_distance = float("inf")
    for turn in turns:
        turn_start = float(turn.get("start") or 0.0)
        turn_end = float(turn.get("end") or turn_start)
        label = str(turn.get("speaker") or "")
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
        if turn_start <= midpoint <= turn_end:
            distance = 0.0
        else:
            distance = min(abs(midpoint - turn_start), abs(midpoint - turn_end))
        if distance < midpoint_distance:
            midpoint_distance = distance
            midpoint_label = label
    if best_label is not None and best_overlap > 0.0:
        return best_label
    return midpoint_label


def _one_to_one_mapping(
    confusion: Mapping[str, Mapping[str, int]],
) -> dict[str, str]:
    speakers = sorted(confusion)
    clusters = sorted({cluster for counts in confusion.values() for cluster in counts})
    if not speakers or not clusters:
        return {}
    if len(clusters) <= len(speakers):
        best_score = -1
        best = {}
        for assigned_speakers in itertools.permutations(speakers, len(clusters)):
            mapping = dict(zip(clusters, assigned_speakers))
            score = sum(
                int(counts.get(cluster, 0))
                for speaker, counts in confusion.items()
                for cluster, mapped in mapping.items()
                if mapped == speaker
            )
            if score > best_score:
                best_score = score
                best = mapping
        return best

    best_score = -1
    best = {}
    for assigned_clusters in itertools.permutations(clusters, len(speakers)):
        mapping = {cluster: speaker for speaker, cluster in zip(speakers, assigned_clusters)}
        score = sum(
            int(counts.get(cluster, 0))
            for speaker, counts in confusion.items()
            for cluster, mapped in mapping.items()
            if mapped == speaker
        )
        if score > best_score:
            best_score = score
            best = mapping
    return best


def _many_to_one_mapping(
    confusion: Mapping[str, Mapping[str, int]],
) -> dict[str, str]:
    clusters = sorted({cluster for counts in confusion.values() for cluster in counts})
    mapping = {}
    for cluster in clusters:
        best_speaker = None
        best_count = -1
        for speaker, counts in confusion.items():
            count = int(counts.get(cluster, 0))
            if count > best_count:
                best_count = count
                best_speaker = speaker
        if best_speaker is not None:
            mapping[cluster] = best_speaker
    return mapping


def _mapped_correct(confusion: Mapping[str, Mapping[str, int]], mapping: Mapping[str, str]) -> int:
    return sum(
        int(count)
        for speaker, counts in confusion.items()
        for cluster, count in counts.items()
        if mapping.get(cluster) == speaker
    )


def _score_group(
    *,
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    words: Sequence[Mapping[str, object]],
    turns: Sequence[Mapping[str, object]],
) -> dict:
    reference_words = 0
    assigned_words = 0
    overlap_words = 0
    assigned_overlap_words = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    by_cluster: Counter[str] = Counter()
    unassigned_words = 0
    records = []
    for index, word in enumerate(words):
        reference = str(word.get("speaker") or "")
        if not reference:
            continue
        reference_words += 1
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        has_overlap = _word_has_overlap(word, words)
        overlap_words += int(has_overlap)
        cluster = _choose_turn_label(start, end, turns)
        if cluster is None:
            unassigned_words += 1
            continue
        assigned_words += 1
        assigned_overlap_words += int(has_overlap)
        confusion[reference][cluster] += 1
        by_speaker[reference]["words"] += 1
        by_cluster[cluster] += 1
        records.append(
            {
                "word_index": index,
                "speaker": reference,
                "cluster": cluster,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "text": word.get("text"),
                "score": word.get("score"),
                "source_span_start": word.get("source_span_start"),
                "source_span_end": word.get("source_span_end"),
                "overlap": has_overlap,
            }
        )

    confusion_dict = {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}
    many_mapping = _many_to_one_mapping(confusion_dict)
    one_mapping = _one_to_one_mapping(confusion_dict)
    many_correct = _mapped_correct(confusion_dict, many_mapping)
    one_correct = _mapped_correct(confusion_dict, one_mapping)
    for reference, counts in confusion_dict.items():
        by_speaker[reference]["many_to_one_correct"] = sum(
            count for cluster, count in counts.items() if many_mapping.get(cluster) == reference
        )
        by_speaker[reference]["one_to_one_correct"] = sum(
            count for cluster, count in counts.items() if one_mapping.get(cluster) == reference
        )

    def _quality_slice(name: str, selected: Sequence[Mapping[str, object]]) -> tuple[str, dict]:
        words = len(selected)
        many = sum(
            1
            for record in selected
            if many_mapping.get(str(record.get("cluster") or ""))
            == str(record.get("speaker") or "")
        )
        one = sum(
            1
            for record in selected
            if one_mapping.get(str(record.get("cluster") or "")) == str(record.get("speaker") or "")
        )
        return name, {
            "words": words,
            "many_to_one_correct_words": many,
            "many_to_one_accuracy": many / words if words else 0.0,
            "one_to_one_correct_words": one,
            "one_to_one_accuracy": one / words if words else 0.0,
            "overlap_words": sum(1 for record in selected if record.get("overlap")),
            "word_share": words / len(records) if records else 0.0,
        }

    quality_filters = dict(
        [
            _quality_slice("all", records),
            _quality_slice(
                "score_ge_0_05",
                [record for record in records if float(record.get("score") or 0.0) >= 0.05],
            ),
            _quality_slice(
                "duration_ge_80ms",
                [record for record in records if float(record.get("duration") or 0.0) >= 0.08],
            ),
            _quality_slice(
                "score_ge_0_05_and_duration_ge_80ms",
                [
                    record
                    for record in records
                    if float(record.get("score") or 0.0) >= 0.05
                    and float(record.get("duration") or 0.0) >= 0.08
                ],
            ),
            _quality_slice(
                "non_overlap", [record for record in records if not record.get("overlap")]
            ),
            _quality_slice(
                "non_overlap_score_ge_0_05",
                [
                    record
                    for record in records
                    if not record.get("overlap") and float(record.get("score") or 0.0) >= 0.05
                ],
            ),
            _quality_slice(
                "non_overlap_score_ge_0_05_duration_ge_80ms",
                [
                    record
                    for record in records
                    if not record.get("overlap")
                    and float(record.get("score") or 0.0) >= 0.05
                    and float(record.get("duration") or 0.0) >= 0.08
                ],
            ),
        ]
    )
    return {
        "session": key[0],
        "window_start": key[1],
        "window_end": key[2],
        "split_id": ",".join(sorted({str(row.get("split_id") or "") for row in rows})),
        "row_count": len(rows),
        "turn_count": len(turns),
        "cluster_count": len(by_cluster),
        "clusters": dict(sorted(by_cluster.items())),
        "reference_words": reference_words,
        "assigned_words": assigned_words,
        "unassigned_words": unassigned_words,
        "coverage": assigned_words / reference_words if reference_words else 0.0,
        "overlap_words": overlap_words,
        "assigned_overlap_words": assigned_overlap_words,
        "confusion": confusion_dict,
        "many_to_one_mapping": many_mapping,
        "one_to_one_mapping": one_mapping,
        "quality_filters": quality_filters,
        "many_to_one_correct_words": many_correct,
        "many_to_one_accuracy": many_correct / reference_words if reference_words else 0.0,
        "one_to_one_correct_words": one_correct,
        "one_to_one_accuracy": one_correct / reference_words if reference_words else 0.0,
        "by_speaker": {
            speaker: {
                "words": int(counts["words"]),
                "many_to_one_correct": int(counts["many_to_one_correct"]),
                "many_to_one_accuracy": (
                    counts["many_to_one_correct"] / counts["words"] if counts["words"] else 0.0
                ),
                "one_to_one_correct": int(counts["one_to_one_correct"]),
                "one_to_one_accuracy": (
                    counts["one_to_one_correct"] / counts["words"] if counts["words"] else 0.0
                ),
            }
            for speaker, counts in sorted(by_speaker.items())
        },
        "word_records": records,
    }


def _sum_confusion(groups: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for speaker, counts in dict(group.get("confusion") or {}).items():
            for cluster, count in dict(counts).items():
                confusion[str(speaker)][str(cluster)] += int(count)
    return {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}


def _summarize(groups: Sequence[Mapping[str, object]]) -> dict:
    valid = [group for group in groups if not group.get("error")]
    reference_words = sum(int(group.get("reference_words") or 0) for group in valid)
    assigned_words = sum(int(group.get("assigned_words") or 0) for group in valid)
    many_correct = sum(int(group.get("many_to_one_correct_words") or 0) for group in valid)
    one_correct = sum(int(group.get("one_to_one_correct_words") or 0) for group in valid)
    overlap_words = sum(int(group.get("overlap_words") or 0) for group in valid)
    by_session: dict[str, Counter[str]] = defaultdict(Counter)
    quality: dict[str, Counter[str]] = defaultdict(Counter)
    for group in valid:
        session = str(group.get("session") or "")
        by_session[session]["reference_words"] += int(group.get("reference_words") or 0)
        by_session[session]["assigned_words"] += int(group.get("assigned_words") or 0)
        by_session[session]["many_to_one_correct_words"] += int(
            group.get("many_to_one_correct_words") or 0
        )
        by_session[session]["one_to_one_correct_words"] += int(
            group.get("one_to_one_correct_words") or 0
        )
        for name, item in dict(group.get("quality_filters") or {}).items():
            item = dict(item)
            quality[str(name)]["words"] += int(item.get("words") or 0)
            quality[str(name)]["many_to_one_correct_words"] += int(
                item.get("many_to_one_correct_words") or 0
            )
            quality[str(name)]["one_to_one_correct_words"] += int(
                item.get("one_to_one_correct_words") or 0
            )
            quality[str(name)]["overlap_words"] += int(item.get("overlap_words") or 0)
    return {
        "group_count": len(groups),
        "valid_groups": len(valid),
        "reference_words": reference_words,
        "assigned_words": assigned_words,
        "coverage": assigned_words / reference_words if reference_words else 0.0,
        "overlap_words": overlap_words,
        "many_to_one_correct_words": many_correct,
        "many_to_one_accuracy": many_correct / reference_words if reference_words else 0.0,
        "one_to_one_correct_words": one_correct,
        "one_to_one_accuracy": one_correct / reference_words if reference_words else 0.0,
        "confusion": _sum_confusion(valid),
        "quality_filters": {
            name: {
                "words": int(counts["words"]),
                "many_to_one_correct_words": int(counts["many_to_one_correct_words"]),
                "many_to_one_accuracy": (
                    counts["many_to_one_correct_words"] / counts["words"]
                    if counts["words"]
                    else 0.0
                ),
                "one_to_one_correct_words": int(counts["one_to_one_correct_words"]),
                "one_to_one_accuracy": (
                    counts["one_to_one_correct_words"] / counts["words"] if counts["words"] else 0.0
                ),
                "overlap_words": int(counts["overlap_words"]),
                "word_share": counts["words"] / reference_words if reference_words else 0.0,
            }
            for name, counts in sorted(quality.items())
        },
        "by_session": {
            session: {
                "reference_words": int(counts["reference_words"]),
                "assigned_words": int(counts["assigned_words"]),
                "many_to_one_correct_words": int(counts["many_to_one_correct_words"]),
                "many_to_one_accuracy": (
                    counts["many_to_one_correct_words"] / counts["reference_words"]
                    if counts["reference_words"]
                    else 0.0
                ),
                "one_to_one_correct_words": int(counts["one_to_one_correct_words"]),
                "one_to_one_accuracy": (
                    counts["one_to_one_correct_words"] / counts["reference_words"]
                    if counts["reference_words"]
                    else 0.0
                ),
            }
            for session, counts in sorted(by_session.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pyannote diarization with oracle cluster-to-speaker mapping."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-splits", default="test")
    parser.add_argument("--eval-sessions")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--num-speakers-from-candidates", action="store_true")
    parser.add_argument("--use-exclusive", action="store_true")
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    from transcriber.diarization import diarize_audio

    manifest_dir = args.manifest.resolve().parent
    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped_rows = _group_manifest_rows(_read_jsonl(args.manifest))
    eval_splits = _split_values(args.eval_splits)
    eval_sessions = _split_values(args.eval_sessions or "")
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    group_results = []
    word_records = []
    for key, all_rows in sorted(grouped_rows.items()):
        rows = [
            row
            for row in all_rows
            if str(row.get("split_id") or "") in eval_splits
            and (not eval_sessions or str(row.get("session") or "") in eval_sessions)
            and row.get("materialized")
        ]
        if not rows:
            continue
        if key not in reference_groups:
            group_results.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_reference_group",
                }
            )
            continue
        mixture_path = _materialized_path(rows[0], "mixture_path", manifest_dir=manifest_dir)
        if mixture_path is None or not mixture_path.exists():
            group_results.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_mixture_path",
                }
            )
            continue
        min_speakers = args.min_speakers
        max_speakers = args.max_speakers
        if args.num_speakers_from_candidates:
            min_speakers = len(rows)
            max_speakers = len(rows)
        diarization = diarize_audio(
            str(mixture_path),
            model_name=args.model_name,
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            device=args.device,
        )
        turn_source = diarization.exclusive_segments if args.use_exclusive else diarization.segments
        turns = [turn.to_dict() for turn in turn_source]
        result = _score_group(
            key=key,
            rows=rows,
            words=reference_groups[key],
            turns=turns,
        )
        result["mixture_path"] = str(mixture_path)
        result["diarization_metadata"] = diarization.metadata
        group_records = result.pop("word_records")
        group_results.append(result)
        if args.save_word_records:
            for record in group_records:
                record.update(
                    {
                        "session": key[0],
                        "window_start": key[1],
                        "window_end": key[2],
                    }
                )
                word_records.append(record)

    summary = _summarize(group_results)
    summary.update(
        {
            "manifest": str(args.manifest),
            "reference_jsonl": str(args.reference_jsonl),
            "eval_splits": sorted(eval_splits),
            "eval_sessions": sorted(eval_sessions),
            "model_name": args.model_name,
            "device": args.device,
            "num_speakers_from_candidates": bool(args.num_speakers_from_candidates),
            "use_exclusive": bool(args.use_exclusive),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "pyannote_oracle_diarization_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "pyannote_oracle_diarization_words.jsonl", word_records)
    (args.output_dir / "pyannote_oracle_diarization_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
