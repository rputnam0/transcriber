from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from score_speaker_tagged_asr import _lcs_token_pairs, _normalize_token  # noqa: E402
from transcriber.asr import transcribe_with_faster_whisper  # noqa: E402


MANIFEST_STEM_RE = re.compile(
    r"session_(?P<session>\d+)_(?P<start_ms>\d+)_(?P<end_ms>\d+)$",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _word_group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _manifest_group_key(row: Mapping[str, object]) -> tuple[str, float, float] | None:
    path = Path(str(row.get("audio_filepath") or ""))
    match = MANIFEST_STEM_RE.search(path.stem)
    if not match:
        return None
    start = int(match.group("start_ms")) / 1000.0
    end = int(match.group("end_ms")) / 1000.0
    return (f"Session {int(match.group('session'))}", round(start, 3), round(end, 3))


def _load_diarization_words(path: Path) -> dict[tuple[str, float, float], list[dict]]:
    groups: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in _read_jsonl(path):
        groups[_word_group_key(row)].append(dict(row))
    return {
        key: sorted(words, key=lambda item: int(item.get("word_index") or 0))
        for key, words in groups.items()
    }


def _reference_tokens(words: Sequence[Mapping[str, object]]) -> list[dict]:
    tokens = []
    for local_index, word in enumerate(words):
        token = _normalize_token(word.get("normalized") or word.get("text"))
        if not token:
            continue
        item = dict(word)
        item["token"] = token
        item["local_index"] = local_index
        item["overlap"] = bool(item.get("overlap"))
        tokens.append(item)
    return tokens


def _asr_tokens(words: Sequence[Mapping[str, object]]) -> list[dict]:
    tokens = []
    for index, word in enumerate(words):
        token = _normalize_token(word.get("word") or word.get("text"))
        if not token:
            continue
        item = dict(word)
        item["token"] = token
        item["index"] = index
        tokens.append(item)
    return tokens


def _one_to_one_mapping(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    speakers = sorted(confusion)
    clusters = sorted({cluster for counts in confusion.values() for cluster in counts})
    if not speakers or not clusters:
        return {}
    if len(clusters) <= len(speakers):
        best_score = -1
        best = {}
        for assigned_speakers in itertools.permutations(speakers, len(clusters)):
            mapping = dict(zip(clusters, assigned_speakers, strict=False))
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
        mapping = {
            cluster: speaker for speaker, cluster in zip(speakers, assigned_clusters, strict=False)
        }
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


def _many_to_one_mapping(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
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


def _cluster_confusion(words: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for word in words:
        speaker = str(word.get("speaker") or "")
        cluster = str(word.get("cluster") or "")
        if speaker and cluster:
            confusion[speaker][cluster] += 1
    return {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}


def _mapped_correct_count(
    records: Sequence[Mapping[str, object]],
    mapping: Mapping[str, str],
) -> int:
    return sum(
        1
        for record in records
        if mapping.get(str(record.get("cluster") or "")) == str(record.get("speaker") or "")
    )


def _quality_slice(
    name: str,
    selected: Sequence[Mapping[str, object]],
    matched_indices: set[int],
    many_mapping: Mapping[str, str],
    one_mapping: Mapping[str, str],
) -> tuple[str, dict]:
    matched = [
        record for record in selected if int(record.get("local_index") or -1) in matched_indices
    ]
    words = len(selected)
    predicted_words = len(matched)
    many = _mapped_correct_count(matched, many_mapping)
    one = _mapped_correct_count(matched, one_mapping)
    return name, {
        "words": words,
        "lexical_matched_words": predicted_words,
        "lexical_coverage": predicted_words / words if words else 0.0,
        "many_to_one_correct_words": many,
        "many_to_one_accuracy": many / words if words else 0.0,
        "many_to_one_matched_accuracy": many / predicted_words if predicted_words else 0.0,
        "one_to_one_correct_words": one,
        "one_to_one_accuracy": one / words if words else 0.0,
        "one_to_one_matched_accuracy": one / predicted_words if predicted_words else 0.0,
        "overlap_words": sum(1 for record in selected if record.get("overlap")),
    }


def _quality_filters(
    reference: Sequence[Mapping[str, object]],
    matched_indices: set[int],
    many_mapping: Mapping[str, str],
    one_mapping: Mapping[str, str],
) -> dict[str, dict]:
    return dict(
        [
            _quality_slice("all", reference, matched_indices, many_mapping, one_mapping),
            _quality_slice(
                "score_ge_0_05",
                [record for record in reference if float(record.get("score") or 0.0) >= 0.05],
                matched_indices,
                many_mapping,
                one_mapping,
            ),
            _quality_slice(
                "duration_ge_80ms",
                [record for record in reference if float(record.get("duration") or 0.0) >= 0.08],
                matched_indices,
                many_mapping,
                one_mapping,
            ),
            _quality_slice(
                "score_ge_0_05_and_duration_ge_80ms",
                [
                    record
                    for record in reference
                    if float(record.get("score") or 0.0) >= 0.05
                    and float(record.get("duration") or 0.0) >= 0.08
                ],
                matched_indices,
                many_mapping,
                one_mapping,
            ),
            _quality_slice(
                "non_overlap",
                [record for record in reference if not record.get("overlap")],
                matched_indices,
                many_mapping,
                one_mapping,
            ),
            _quality_slice(
                "non_overlap_score_ge_0_05",
                [
                    record
                    for record in reference
                    if not record.get("overlap") and float(record.get("score") or 0.0) >= 0.05
                ],
                matched_indices,
                many_mapping,
                one_mapping,
            ),
            _quality_slice(
                "non_overlap_score_ge_0_05_duration_ge_80ms",
                [
                    record
                    for record in reference
                    if not record.get("overlap")
                    and float(record.get("score") or 0.0) >= 0.05
                    and float(record.get("duration") or 0.0) >= 0.08
                ],
                matched_indices,
                many_mapping,
                one_mapping,
            ),
        ]
    )


def _score_group(
    *,
    key: tuple[str, float, float],
    diarization_words: Sequence[Mapping[str, object]],
    asr_words: Sequence[Mapping[str, object]],
) -> dict:
    reference = _reference_tokens(diarization_words)
    predicted = _asr_tokens(asr_words)
    pairs = _lcs_token_pairs(reference, predicted)
    confusion = _cluster_confusion(reference)
    many_mapping = _many_to_one_mapping(confusion)
    one_mapping = _one_to_one_mapping(confusion)
    matched_indices = {int(reference[ref_index].get("local_index") or -1) for ref_index, _ in pairs}

    matched_records = [reference[ref_index] for ref_index, _ in pairs]
    many_correct = _mapped_correct_count(matched_records, many_mapping)
    one_correct = _mapped_correct_count(matched_records, one_mapping)
    overlap_records = [record for record in reference if record.get("overlap")]
    non_overlap_records = [record for record in reference if not record.get("overlap")]
    matched_overlap = [record for record in matched_records if record.get("overlap")]
    matched_non_overlap = [record for record in matched_records if not record.get("overlap")]
    many_overlap = _mapped_correct_count(matched_overlap, many_mapping)
    many_non_overlap = _mapped_correct_count(matched_non_overlap, many_mapping)
    one_overlap = _mapped_correct_count(matched_overlap, one_mapping)
    one_non_overlap = _mapped_correct_count(matched_non_overlap, one_mapping)

    reference_count = len(reference)
    predicted_count = len(predicted)
    matched_count = len(pairs)
    many_oracle_correct = _mapped_correct_count(reference, many_mapping)
    one_oracle_correct = _mapped_correct_count(reference, one_mapping)
    return {
        "session": key[0],
        "window_start": key[1],
        "window_end": key[2],
        "reference_words": reference_count,
        "predicted_words": predicted_count,
        "lexical_matched_words": matched_count,
        "lexical_coverage": matched_count / reference_count if reference_count else 0.0,
        "overlap_words": len(overlap_records),
        "non_overlap_words": len(non_overlap_records),
        "many_to_one_mapping": dict(sorted(many_mapping.items())),
        "one_to_one_mapping": dict(sorted(one_mapping.items())),
        "many_to_one_oracle_diarization_correct_words": many_oracle_correct,
        "many_to_one_oracle_diarization_accuracy": (
            many_oracle_correct / reference_count if reference_count else 0.0
        ),
        "one_to_one_oracle_diarization_correct_words": one_oracle_correct,
        "one_to_one_oracle_diarization_accuracy": (
            one_oracle_correct / reference_count if reference_count else 0.0
        ),
        "many_to_one_correct_words": many_correct,
        "many_to_one_accuracy": many_correct / reference_count if reference_count else 0.0,
        "many_to_one_matched_accuracy": many_correct / matched_count if matched_count else 0.0,
        "many_to_one_prediction_precision_proxy": (
            many_correct / predicted_count if predicted_count else 0.0
        ),
        "many_to_one_overlap_correct_words": many_overlap,
        "many_to_one_overlap_accuracy": (
            many_overlap / len(overlap_records) if overlap_records else 0.0
        ),
        "many_to_one_non_overlap_correct_words": many_non_overlap,
        "many_to_one_non_overlap_accuracy": (
            many_non_overlap / len(non_overlap_records) if non_overlap_records else 0.0
        ),
        "one_to_one_correct_words": one_correct,
        "one_to_one_accuracy": one_correct / reference_count if reference_count else 0.0,
        "one_to_one_matched_accuracy": one_correct / matched_count if matched_count else 0.0,
        "one_to_one_prediction_precision_proxy": (
            one_correct / predicted_count if predicted_count else 0.0
        ),
        "one_to_one_overlap_correct_words": one_overlap,
        "one_to_one_overlap_accuracy": (
            one_overlap / len(overlap_records) if overlap_records else 0.0
        ),
        "one_to_one_non_overlap_correct_words": one_non_overlap,
        "one_to_one_non_overlap_accuracy": (
            one_non_overlap / len(non_overlap_records) if non_overlap_records else 0.0
        ),
        "quality_filters": _quality_filters(
            reference,
            matched_indices,
            many_mapping,
            one_mapping,
        ),
    }


def _aggregate(groups: Sequence[Mapping[str, object]]) -> dict:
    totals = Counter()
    quality: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for key in [
            "reference_words",
            "predicted_words",
            "lexical_matched_words",
            "overlap_words",
            "non_overlap_words",
            "many_to_one_oracle_diarization_correct_words",
            "one_to_one_oracle_diarization_correct_words",
            "many_to_one_correct_words",
            "many_to_one_overlap_correct_words",
            "many_to_one_non_overlap_correct_words",
            "one_to_one_correct_words",
            "one_to_one_overlap_correct_words",
            "one_to_one_non_overlap_correct_words",
        ]:
            totals[key] += int(group.get(key) or 0)
        for name, item in dict(group.get("quality_filters") or {}).items():
            item = dict(item)
            for key in [
                "words",
                "lexical_matched_words",
                "many_to_one_correct_words",
                "one_to_one_correct_words",
                "overlap_words",
            ]:
                quality[str(name)][key] += int(item.get(key) or 0)
    reference_words = totals["reference_words"]
    predicted_words = totals["predicted_words"]
    lexical_matched = totals["lexical_matched_words"]
    overlap_words = totals["overlap_words"]
    non_overlap_words = totals["non_overlap_words"]
    return {
        "group_count": len(groups),
        "primary_metric": "many_to_one_accuracy",
        **dict(totals),
        "lexical_coverage": lexical_matched / reference_words if reference_words else 0.0,
        "many_to_one_oracle_diarization_accuracy": (
            totals["many_to_one_oracle_diarization_correct_words"] / reference_words
            if reference_words
            else 0.0
        ),
        "one_to_one_oracle_diarization_accuracy": (
            totals["one_to_one_oracle_diarization_correct_words"] / reference_words
            if reference_words
            else 0.0
        ),
        "many_to_one_accuracy": (
            totals["many_to_one_correct_words"] / reference_words if reference_words else 0.0
        ),
        "many_to_one_matched_accuracy": (
            totals["many_to_one_correct_words"] / lexical_matched if lexical_matched else 0.0
        ),
        "many_to_one_prediction_precision_proxy": (
            totals["many_to_one_correct_words"] / predicted_words if predicted_words else 0.0
        ),
        "many_to_one_overlap_accuracy": (
            totals["many_to_one_overlap_correct_words"] / overlap_words if overlap_words else 0.0
        ),
        "many_to_one_non_overlap_accuracy": (
            totals["many_to_one_non_overlap_correct_words"] / non_overlap_words
            if non_overlap_words
            else 0.0
        ),
        "one_to_one_accuracy": (
            totals["one_to_one_correct_words"] / reference_words if reference_words else 0.0
        ),
        "one_to_one_matched_accuracy": (
            totals["one_to_one_correct_words"] / lexical_matched if lexical_matched else 0.0
        ),
        "one_to_one_prediction_precision_proxy": (
            totals["one_to_one_correct_words"] / predicted_words if predicted_words else 0.0
        ),
        "one_to_one_overlap_accuracy": (
            totals["one_to_one_overlap_correct_words"] / overlap_words if overlap_words else 0.0
        ),
        "one_to_one_non_overlap_accuracy": (
            totals["one_to_one_non_overlap_correct_words"] / non_overlap_words
            if non_overlap_words
            else 0.0
        ),
        "quality_filters": {
            name: {
                "words": int(counts["words"]),
                "lexical_matched_words": int(counts["lexical_matched_words"]),
                "lexical_coverage": (
                    counts["lexical_matched_words"] / counts["words"] if counts["words"] else 0.0
                ),
                "many_to_one_correct_words": int(counts["many_to_one_correct_words"]),
                "many_to_one_accuracy": (
                    counts["many_to_one_correct_words"] / counts["words"]
                    if counts["words"]
                    else 0.0
                ),
                "many_to_one_matched_accuracy": (
                    counts["many_to_one_correct_words"] / counts["lexical_matched_words"]
                    if counts["lexical_matched_words"]
                    else 0.0
                ),
                "one_to_one_correct_words": int(counts["one_to_one_correct_words"]),
                "one_to_one_accuracy": (
                    counts["one_to_one_correct_words"] / counts["words"] if counts["words"] else 0.0
                ),
                "one_to_one_matched_accuracy": (
                    counts["one_to_one_correct_words"] / counts["lexical_matched_words"]
                    if counts["lexical_matched_words"]
                    else 0.0
                ),
                "overlap_words": int(counts["overlap_words"]),
                "word_share": counts["words"] / reference_words if reference_words else 0.0,
            }
            for name, counts in sorted(quality.items())
        },
    }


def _cache_identity(
    *,
    key: tuple[str, float, float],
    audio_path: Path,
    model_name: str,
    compute_type: str,
    device: str,
    batch_size: int,
) -> dict:
    stat = audio_path.stat()
    try:
        faster_whisper_version = importlib.metadata.version("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        faster_whisper_version = "unknown"
    return {
        "group_key": key,
        "audio_path": str(audio_path),
        "audio_mtime_ns": stat.st_mtime_ns,
        "audio_size": stat.st_size,
        "asr_model": model_name,
        "asr_device": device,
        "compute_type": compute_type,
        "batch_size": int(batch_size),
        "decode_options": {
            "vad_filter": True,
            "word_timestamps": True,
        },
        "faster_whisper_version": faster_whisper_version,
    }


def _cache_path(cache_dir: Path, identity: Mapping[str, object]) -> Path:
    payload = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"


def _transcribe_mixture(
    *,
    key: tuple[str, float, float],
    audio_path: Path,
    cache_dir: Path,
    model_name: str,
    compute_type: str,
    device: str,
    batch_size: int,
) -> list[dict]:
    identity = _cache_identity(
        key=key,
        audio_path=audio_path,
        model_name=model_name,
        compute_type=compute_type,
        device=device,
        batch_size=batch_size,
    )
    cache_file = _cache_path(cache_dir, identity)
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        result = transcribe_with_faster_whisper(
            str(audio_path),
            model_name=model_name,
            compute_type=compute_type,
            device=device,
            batch_size=batch_size,
        )
        payload = {
            "audio_path": str(audio_path),
            "segments": [segment.to_dict() for segment in result.segments],
            "metadata": result.metadata,
            "cache_identity": identity,
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    words = []
    for segment in payload.get("segments") or []:
        for word in dict(segment).get("words") or []:
            item = dict(word)
            token = item.get("word") or item.get("text")
            start = item.get("start")
            end = item.get("end")
            if token and start is not None and end is not None:
                words.append(
                    {
                        "word": str(token).strip(),
                        "start": float(start),
                        "end": float(end),
                        "score": item.get("score"),
                    }
                )
    return words


def _selected_manifest_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    session: str | None,
    window_start: float | None,
    max_groups: int | None,
) -> list[tuple[tuple[str, float, float], dict]]:
    selected = []
    for row in rows:
        key = _manifest_group_key(row)
        if key is None:
            continue
        if session and key[0] != session:
            continue
        if window_start is not None and abs(key[1] - float(window_start)) > 1e-3:
            continue
        selected.append((key, dict(row)))
        if max_groups is not None and len(selected) >= max_groups:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score mixture-level ASR words assigned through oracle-mapped diarization word records."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diarization-words-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--session")
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()

    manifest_rows = list(_read_jsonl(args.manifest))
    selected_rows = _selected_manifest_rows(
        manifest_rows,
        session=args.session,
        window_start=args.window_start,
        max_groups=args.max_groups,
    )
    diarization_groups = _load_diarization_words(args.diarization_words_jsonl)
    cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "asr_cache"
    group_rows = []
    for key, row in selected_rows:
        words = diarization_groups.get(key)
        if not words:
            group_rows.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_diarization_words",
                }
            )
            continue
        audio_path = Path(str(row.get("audio_filepath") or ""))
        if not audio_path.exists():
            group_rows.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_audio",
                    "audio_path": str(audio_path),
                }
            )
            continue
        asr_words = _transcribe_mixture(
            key=key,
            audio_path=audio_path,
            cache_dir=cache_dir,
            model_name=str(args.model),
            compute_type=str(args.compute_type),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        score = _score_group(key=key, diarization_words=words, asr_words=asr_words)
        score["audio_path"] = str(audio_path)
        group_rows.append(score)

    valid = [row for row in group_rows if not row.get("error")]
    summary = _aggregate(valid)
    summary.update(
        {
            "manifest": str(args.manifest),
            "diarization_words_jsonl": str(args.diarization_words_jsonl),
            "cache_dir": str(cache_dir),
            "model": str(args.model),
            "compute_type": str(args.compute_type),
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "selected_groups": len(selected_rows),
            "error_count": len(group_rows) - len(valid),
            "errors": dict(
                Counter(str(row.get("error")) for row in group_rows if row.get("error"))
            ),
            "evaluation_note": (
                "Upper-bound diagnostic: diarization clusters are oracle-mapped to speakers from "
                "the reference word records, then scored only on lexically matched mixture-ASR words."
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "mixture_asr_diarization_groups.jsonl", group_rows)
    (args.output_dir / "mixture_asr_diarization_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
