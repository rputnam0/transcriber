from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
from zipfile import ZipFile

from transcriber.cli import _load_yaml_or_json
from transcriber.consolidate import choose_speaker
from transcriber.multitrack_eval import clip_audio, extract_session_stems, mix_audio_files
from transcriber.segment_classifier import load_labeled_records


SPEAKER_ALIASES = {"zariel torgan": "David Tanglethorn"}
TARGET_SHARE_BUCKETS = (
    ("lt_010", 0.0, 0.10),
    ("010_020", 0.10, 0.20),
    ("020_035", 0.20, 0.35),
    ("035_050", 0.35, 0.50),
    ("050_075", 0.50, 0.75),
    ("075_100", 0.75, 1.01),
)


@dataclass(frozen=True)
class SessionInput:
    session: str
    audio_zip: Path
    transcript: Path
    split: str


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _safe_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _parse_session_numbers(values: Sequence[str]) -> List[str]:
    sessions: List[str] = []
    for value in values:
        for raw in str(value).split(","):
            item = raw.strip()
            if not item:
                continue
            if item.lower().startswith("session"):
                digits = re.sub(r"[^0-9]+", "", item)
                sessions.append(f"Session {digits}" if digits else item)
            else:
                sessions.append(f"Session {item}")
    return list(dict.fromkeys(sessions))


def _share_bucket(value: float) -> str:
    for name, low, high in TARGET_SHARE_BUCKETS:
        if low <= value < high:
            return name
    return "unknown"


def _record_words(record: Mapping[str, object]) -> int:
    return len(str(record.get("text") or "").split())


def _duration(record: Mapping[str, object]) -> float:
    return max(float(record.get("end") or 0.0) - float(record.get("start") or 0.0), 0.0)


def _window_records(
    records: Sequence[Mapping[str, object]], start: float, end: float
) -> List[dict]:
    rows = []
    for record in records:
        record_start = float(record.get("start") or 0.0)
        record_end = float(record.get("end") or 0.0)
        if record_end > start and record_start < end:
            rows.append(dict(record))
    return sorted(
        rows, key=lambda item: (float(item.get("start") or 0.0), str(item.get("speaker")))
    )


def _select_windows(
    records: Sequence[Mapping[str, object]],
    *,
    window_seconds: float,
    hop_seconds: float,
    top_k: int,
    min_speakers: int,
) -> List[dict]:
    if not records:
        return []
    max_end = max(float(record.get("end") or 0.0) for record in records)
    candidates: List[dict] = []
    start = 0.0
    while start + window_seconds <= max_end + 1e-6:
        end = start + window_seconds
        rows = _window_records(records, start, end)
        words_by_speaker: Counter[str] = Counter()
        turns = 0
        previous = None
        for row in rows:
            speaker = str(row.get("speaker") or "").strip()
            if not speaker:
                continue
            words_by_speaker[speaker] += _record_words(row)
            if previous is not None and previous != speaker:
                turns += 1
            previous = speaker
        total_words = sum(words_by_speaker.values())
        if len(words_by_speaker) >= min_speakers and total_words:
            balance = 1.0 - (max(words_by_speaker.values()) / total_words)
            score = (
                (len(words_by_speaker) * 1000.0) + (turns * 10.0) + total_words + (balance * 100.0)
            )
            candidates.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(window_seconds, 3),
                    "score": score,
                    "speaker_count": len(words_by_speaker),
                    "turn_count": turns,
                    "total_words": total_words,
                    "words_by_speaker": dict(sorted(words_by_speaker.items())),
                    "speakers": sorted(words_by_speaker),
                }
            )
        start += hop_seconds
    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["speaker_count"]),
            int(item["turn_count"]),
        ),
        reverse=True,
    )
    selected: List[dict] = []
    for candidate in candidates:
        if any(
            not (
                float(candidate["end"]) <= float(chosen["start"])
                or float(candidate["start"]) >= float(chosen["end"])
            )
            for chosen in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _audio_members(zip_path: Path) -> List[str]:
    audio_suffixes = {".ogg", ".wav", ".flac", ".mp3", ".m4a"}
    with ZipFile(zip_path) as archive:
        return sorted(
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in audio_suffixes
        )


def _speaker_members(zip_path: Path, mapping: Mapping[str, object]) -> Dict[str, str]:
    members: Dict[str, str] = {}
    for member in _audio_members(zip_path):
        speaker, matched = choose_speaker(Path(member).name, dict(mapping), return_match=True)
        if matched:
            members.setdefault(str(speaker), member)
    return dict(sorted(members.items()))


def _clip_span(record: Mapping[str, object], start: float, end: float) -> dict:
    clipped_start = max(float(record.get("start") or 0.0), start)
    clipped_end = min(float(record.get("end") or 0.0), end)
    return {
        "speaker": str(record.get("speaker") or "unknown"),
        "start": round(clipped_start - start, 3),
        "end": round(max(clipped_end - start, clipped_start - start), 3),
        "session_start": round(clipped_start, 3),
        "session_end": round(clipped_end, 3),
        "text": str(record.get("text") or ""),
        "word_count": _record_words(record),
    }


def _enrollment_candidates(
    records: Sequence[Mapping[str, object]],
    *,
    target: str,
    window_start: float,
    window_end: float,
    positive: bool,
    guard_seconds: float,
    count: int,
    min_seconds: float,
) -> List[dict]:
    candidates = []
    for record in records:
        speaker = str(record.get("speaker") or "")
        is_positive = speaker == target
        if positive != is_positive:
            continue
        start = float(record.get("start") or 0.0)
        end = float(record.get("end") or 0.0)
        if end > window_start - guard_seconds and start < window_end + guard_seconds:
            continue
        seconds = _duration(record)
        if seconds < min_seconds:
            continue
        candidates.append(
            {
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(seconds, 3),
                "text": str(record.get("text") or ""),
                "word_count": _record_words(record),
            }
        )
    candidates.sort(
        key=lambda item: (float(item["duration"]), int(item["word_count"])), reverse=True
    )
    return candidates[:count]


def _resolve_session_inputs(args: argparse.Namespace) -> List[SessionInput]:
    sessions = _parse_session_numbers(args.sessions)
    dev_sessions = set(_parse_session_numbers(args.dev_sessions or []))
    test_sessions = set(_parse_session_numbers(args.test_sessions or []))
    resolved = []
    for session in sessions:
        audio_zip = args.audio_root / f"{session}.zip"
        transcript_dir = args.transcript_root / session
        transcript = transcript_dir / f"{session}.txt"
        if not transcript.exists():
            matches = sorted(transcript_dir.glob("*.txt"))
            if matches:
                transcript = matches[0]
        if not audio_zip.exists() or not transcript.exists():
            continue
        split = (
            "test" if session in test_sessions else "dev" if session in dev_sessions else "train"
        )
        resolved.append(
            SessionInput(session=session, audio_zip=audio_zip, transcript=transcript, split=split)
        )
    return resolved


def _build_rows_for_session(
    session_input: SessionInput,
    *,
    mapping: Mapping[str, object],
    timed_end_mode: str,
    window_seconds: float,
    hop_seconds: float,
    top_k: int,
    min_speakers: int,
    min_target_words: int,
    enrollment_count: int,
    min_enrollment_seconds: float,
    enrollment_guard_seconds: float,
) -> List[dict]:
    records = load_labeled_records(
        session_input.transcript,
        speaker_aliases=SPEAKER_ALIASES,
        speaker_mapping=dict(mapping),
        timed_end_mode=timed_end_mode,
    )
    windows = _select_windows(
        records,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        top_k=top_k,
        min_speakers=min_speakers,
    )
    speaker_members = _speaker_members(session_input.audio_zip, mapping)
    rows: List[dict] = []
    for window_index, window in enumerate(windows, start=1):
        start = float(window["start"])
        end = float(window["end"])
        spans = [_clip_span(record, start, end) for record in _window_records(records, start, end)]
        words_by_speaker = Counter(
            {
                str(speaker): int(count)
                for speaker, count in dict(window.get("words_by_speaker") or {}).items()
            }
        )
        total_words = sum(words_by_speaker.values())
        for target in sorted(words_by_speaker):
            target_words = int(words_by_speaker[target])
            if target_words < min_target_words:
                continue
            if target not in speaker_members:
                continue
            positive_enrollment = _enrollment_candidates(
                records,
                target=target,
                window_start=start,
                window_end=end,
                positive=True,
                guard_seconds=enrollment_guard_seconds,
                count=enrollment_count,
                min_seconds=min_enrollment_seconds,
            )
            negative_enrollment = _enrollment_candidates(
                records,
                target=target,
                window_start=start,
                window_end=end,
                positive=False,
                guard_seconds=enrollment_guard_seconds,
                count=enrollment_count,
                min_seconds=min_enrollment_seconds,
            )
            target_share = target_words / total_words if total_words else 0.0
            row_id = (
                f"{_safe_id(session_input.session)}_w{window_index:02d}_"
                f"{int(start):05d}_{int(end):05d}_{_safe_id(target)}"
            )
            rows.append(
                {
                    "manifest_version": 1,
                    "row_id": row_id,
                    "split_id": session_input.split,
                    "session": session_input.session,
                    "source_zip": str(session_input.audio_zip),
                    "transcript_path": str(session_input.transcript),
                    "window": window,
                    "window_start": start,
                    "window_end": end,
                    "duration": float(window_seconds),
                    "speaker_id": target,
                    "mixture_members": [
                        speaker_members[speaker] for speaker in sorted(speaker_members)
                    ],
                    "target_member": speaker_members[target],
                    "interferer_members": [
                        member
                        for speaker, member in sorted(speaker_members.items())
                        if speaker != target
                    ],
                    "positive_enrollment_spans": positive_enrollment,
                    "negative_enrollment_spans": negative_enrollment,
                    "word_spans": spans,
                    "target_word_count": target_words,
                    "total_word_count": total_words,
                    "target_share": round(target_share, 6),
                    "overlap_bucket": _share_bucket(target_share),
                    "active_speaker_bucket": str(int(window.get("speaker_count") or 0)),
                    "materialized": {},
                }
            )
    return rows


def _run_ffmpeg(command: List[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _stem_path_for_member(stems: Sequence[Path], member: str) -> Path:
    target_name = Path(member).name
    for stem in stems:
        if stem.name == target_name:
            return stem
    raise FileNotFoundError(f"Could not locate extracted member {member}")


def _materialize_row(
    row: Mapping[str, object],
    *,
    mapping: Mapping[str, object],
    output_root: Path,
    stems_cache_root: Path,
) -> dict:
    materialized = dict(row)
    session_id = _safe_id(row["session"])
    row_id = str(row["row_id"])
    row_dir = output_root / str(row["split_id"]) / row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    stems = extract_session_stems(Path(str(row["source_zip"])), stems_cache_root / session_id)
    member_paths = {Path(path).name: _stem_path_for_member(stems, str(path)) for path in row["mixture_members"]}  # type: ignore[index]
    start = float(row["window_start"])
    duration = float(row["duration"])

    clip_paths: List[Path] = []
    for member in row["mixture_members"]:  # type: ignore[index]
        member_name = Path(str(member)).name
        speaker = choose_speaker(member_name, dict(mapping))
        clip_path = row_dir / "sources" / f"{_safe_id(speaker)}.wav"
        if not clip_path.exists():
            clip_audio(member_paths[member_name], clip_path, start=start, duration=duration)
        clip_paths.append(clip_path)

    mixture_path = row_dir / "mixture.wav"
    if not mixture_path.exists():
        mix_audio_files(clip_paths, mixture_path)

    target_name = Path(str(row["target_member"])).name
    target_path = row_dir / "target.wav"
    if not target_path.exists():
        clip_audio(member_paths[target_name], target_path, start=start, duration=duration)

    positive_paths = []
    for index, span in enumerate(row.get("positive_enrollment_spans") or [], start=1):
        span_map = dict(span)
        path = row_dir / "enrollment" / f"positive_{index:02d}.wav"
        if not path.exists():
            clip_audio(
                member_paths[target_name],
                path,
                start=float(span_map["start"]),
                duration=float(span_map["duration"]),
            )
        positive_paths.append(str(path))

    negative_paths = []
    for index, span in enumerate(row.get("negative_enrollment_spans") or [], start=1):
        span_map = dict(span)
        speaker = str(span_map.get("speaker") or "")
        member = None
        for candidate_speaker, candidate_member in _speaker_members(
            Path(str(row["source_zip"])), mapping
        ).items():
            if candidate_speaker == speaker:
                member = candidate_member
                break
        if member is None:
            continue
        path = row_dir / "enrollment" / f"negative_{index:02d}_{_safe_id(speaker)}.wav"
        if not path.exists():
            clip_audio(
                member_paths[Path(member).name],
                path,
                start=float(span_map["start"]),
                duration=float(span_map["duration"]),
            )
        negative_paths.append(str(path))

    materialized["materialized"] = {
        "row_dir": str(row_dir),
        "mixture_path": str(mixture_path),
        "target_source_path": str(target_path),
        "source_paths": [str(path) for path in clip_paths],
        "positive_enrollment_paths": positive_paths,
        "negative_enrollment_paths": negative_paths,
    }
    return materialized


def _build_summary(
    rows: Sequence[Mapping[str, object]], *, materialized_rows: int
) -> Dict[str, object]:
    by_split = Counter(str(row.get("split_id") or "unknown") for row in rows)
    by_speaker = Counter(str(row.get("speaker_id") or "unknown") for row in rows)
    by_session = Counter(str(row.get("session") or "unknown") for row in rows)
    by_bucket = Counter(str(row.get("overlap_bucket") or "unknown") for row in rows)
    words_by_split = Counter()
    words_by_speaker = Counter()
    missing_positive = 0
    missing_negative = 0
    leakage_violations = []
    for row in rows:
        split = str(row.get("split_id") or "unknown")
        speaker = str(row.get("speaker_id") or "unknown")
        words = int(row.get("target_word_count") or 0)
        words_by_split[split] += words
        words_by_speaker[speaker] += words
        positives = list(row.get("positive_enrollment_spans") or [])
        negatives = list(row.get("negative_enrollment_spans") or [])
        if not positives:
            missing_positive += 1
        if not negatives:
            missing_negative += 1
        start = float(row.get("window_start") or 0.0)
        end = float(row.get("window_end") or 0.0)
        for kind, spans in (("positive", positives), ("negative", negatives)):
            for span in spans:
                span_map = dict(span)
                if (
                    float(span_map.get("end") or 0.0) > start
                    and float(span_map.get("start") or 0.0) < end
                ):
                    leakage_violations.append(
                        {
                            "row_id": row.get("row_id"),
                            "kind": kind,
                            "span": span_map,
                        }
                    )
    test_sessions = {str(row.get("session")) for row in rows if row.get("split_id") == "test"}
    train_sessions = {str(row.get("session")) for row in rows if row.get("split_id") == "train"}
    heldout_session_leakage = sorted(test_sessions & train_sessions)
    return {
        "manifest_version": 1,
        "row_count": len(rows),
        "materialized_rows": materialized_rows,
        "by_split": dict(sorted(by_split.items())),
        "by_session": dict(sorted(by_session.items())),
        "by_speaker": dict(sorted(by_speaker.items())),
        "by_overlap_bucket": dict(sorted(by_bucket.items())),
        "target_words_by_split": dict(sorted(words_by_split.items())),
        "target_words_by_speaker": dict(sorted(words_by_speaker.items())),
        "missing_positive_enrollment_rows": missing_positive,
        "missing_negative_enrollment_rows": missing_negative,
        "leakage_audit": {
            "window_enrollment_overlap_violations": leakage_violations,
            "heldout_session_leakage": heldout_session_leakage,
            "passes": not leakage_violations and not heldout_session_leakage,
        },
    }


def _write_markdown(summary: Mapping[str, object], path: Path, *, manifest_path: Path) -> None:
    lines = [
        "# Speaker TSE Manifest Summary",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Rows: {summary.get('row_count', 0)}",
        f"- Materialized rows: {summary.get('materialized_rows', 0)}",
        f"- Leakage audit passes: {dict(summary.get('leakage_audit') or {}).get('passes')}",
        "",
        "## Rows By Split",
        "",
        "| split | rows | target words |",
        "| --- | ---: | ---: |",
    ]
    by_split = dict(summary.get("by_split") or {})
    words_by_split = dict(summary.get("target_words_by_split") or {})
    for split, count in sorted(by_split.items()):
        lines.append(f"| {split} | {count} | {words_by_split.get(split, 0)} |")
    lines.extend(
        ["", "## Rows By Speaker", "", "| speaker | rows | target words |", "| --- | ---: | ---: |"]
    )
    by_speaker = dict(summary.get("by_speaker") or {})
    words_by_speaker = dict(summary.get("target_words_by_speaker") or {})
    for speaker, count in sorted(by_speaker.items()):
        lines.append(f"| {speaker} | {count} | {words_by_speaker.get(speaker, 0)} |")
    lines.extend(["", "## Rows By Target Share Bucket", "", "| bucket | rows |", "| --- | ---: |"])
    for bucket, count in sorted(dict(summary.get("by_overlap_bucket") or {}).items()):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            "## Enrollment Gaps",
            "",
            f"- Missing positive enrollment rows: {summary.get('missing_positive_enrollment_rows', 0)}",
            f"- Missing negative enrollment rows: {summary.get('missing_negative_enrollment_rows', 0)}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    mapping = _load_yaml_or_json(str(args.speaker_mapping))
    session_inputs = _resolve_session_inputs(args)
    rows: List[dict] = []
    for session_input in session_inputs:
        rows.extend(
            _build_rows_for_session(
                session_input,
                mapping=mapping,
                timed_end_mode=args.timed_end_mode,
                window_seconds=args.window_seconds,
                hop_seconds=args.hop_seconds,
                top_k=args.windows_per_session,
                min_speakers=args.min_speakers,
                min_target_words=args.min_target_words,
                enrollment_count=args.enrollment_count,
                min_enrollment_seconds=args.min_enrollment_seconds,
                enrollment_guard_seconds=args.enrollment_guard_seconds,
            )
        )
    split_order = {"train": 0, "dev": 1, "test": 2}
    rows.sort(
        key=lambda row: (
            split_order.get(str(row["split_id"]), 99),
            str(row["session"]),
            str(row["row_id"]),
        )
    )
    materialized_count = 0
    materialize_splits = set(args.materialize_splits or [])
    if args.materialize_limit > 0:
        materialized_rows: List[dict] = []
        materialized_seen = 0
        for index, row in enumerate(rows):
            split = str(row.get("split_id") or "")
            split_selected = not materialize_splits or split in materialize_splits
            if split_selected and materialized_seen < args.materialize_limit:
                materialized_rows.append(
                    _materialize_row(
                        row,
                        mapping=mapping,
                        output_root=args.output_dir / "materialized",
                        stems_cache_root=args.output_dir / "_stems",
                    )
                )
                materialized_count += 1
                materialized_seen += 1
            else:
                materialized_rows.append(row)
        rows = materialized_rows

    manifest_path = args.output_dir / "speaker_tse_manifest.jsonl"
    summary_path = args.output_dir / "speaker_tse_manifest_summary.json"
    markdown_path = args.output_dir / "speaker_tse_manifest_summary.md"
    _write_jsonl(manifest_path, rows)
    summary = _build_summary(rows, materialized_rows=materialized_count)
    summary["sessions_requested"] = _parse_session_numbers(args.sessions)
    summary["sessions_resolved"] = [
        {
            **asdict(item),
            "audio_zip": str(item.audio_zip),
            "transcript": str(item.transcript),
        }
        for item in session_inputs
    ]
    summary["manifest_path"] = str(manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, markdown_path, manifest_path=manifest_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a target-speaker extraction manifest from multitrack session zips."
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/prod/Audio"))
    parser.add_argument("--transcript-root", type=Path, default=Path("data/prod/Transcripts"))
    parser.add_argument("--speaker-mapping", type=Path, default=Path("config/speaker_mapping.yaml"))
    parser.add_argument(
        "--timed-end-mode",
        choices=("next_line", "speaker_estimate"),
        default="next_line",
    )
    parser.add_argument("--sessions", action="append", required=True)
    parser.add_argument("--dev-sessions", action="append", default=[])
    parser.add_argument("--test-sessions", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--hop-seconds", type=float, default=60.0)
    parser.add_argument("--windows-per-session", type=int, default=2)
    parser.add_argument("--min-speakers", type=int, default=5)
    parser.add_argument("--min-target-words", type=int, default=20)
    parser.add_argument("--enrollment-count", type=int, default=2)
    parser.add_argument("--min-enrollment-seconds", type=float, default=3.0)
    parser.add_argument("--enrollment-guard-seconds", type=float, default=300.0)
    parser.add_argument(
        "--materialize-limit",
        type=int,
        default=0,
        help="Number of rows to materialize as mixture/target/enrollment wav files.",
    )
    parser.add_argument(
        "--materialize-splits",
        action="append",
        default=[],
        choices=["train", "dev", "test"],
        help="Only materialize rows from these split IDs. Defaults to all splits.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_manifest(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
