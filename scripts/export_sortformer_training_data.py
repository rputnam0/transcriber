from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from transcriber.multitrack_eval import clip_audio, extract_session_stems, mix_audio_files


GroupKey = tuple[str, float, float]


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


def _safe_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _resolve_path(path_value: object, *, manifest_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _group_key(row: Mapping[str, object]) -> GroupKey:
    return (
        str(row.get("session") or "unknown"),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _group_rows(rows: Iterable[Mapping[str, object]]) -> dict[GroupKey, list[dict]]:
    grouped: dict[GroupKey, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(dict(row))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _load_reference_groups(path: Path | None) -> dict[GroupKey, dict]:
    if path is None:
        return {}
    groups = {}
    for row in _read_jsonl(path):
        groups[_group_key(row)] = row
    return groups


def _intervals_from_reference(group: Mapping[str, object]) -> list[dict]:
    intervals = []
    for word in group.get("words") or []:
        item = dict(word)
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        speaker = str(item.get("speaker") or "").strip()
        if speaker and end > start:
            intervals.append({"speaker": speaker, "start": start, "end": end, "source": "forced"})
    return intervals


def _intervals_from_manifest(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    if not rows:
        return []
    intervals = []
    for span in rows[0].get("word_spans") or []:
        item = dict(span)
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        speaker = str(item.get("speaker") or "").strip()
        if speaker and end > start:
            intervals.append({"speaker": speaker, "start": start, "end": end, "source": "manifest"})
    return intervals


def _clamp_and_merge_intervals(
    intervals: Sequence[Mapping[str, object]],
    *,
    duration: float,
    merge_gap_seconds: float,
) -> list[dict]:
    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for interval in intervals:
        start = max(0.0, min(float(interval.get("start") or 0.0), duration))
        end = max(0.0, min(float(interval.get("end") or start), duration))
        if end > start:
            by_speaker[str(interval.get("speaker") or "unknown")].append((start, end))

    merged = []
    for speaker, spans in sorted(by_speaker.items()):
        current_start: float | None = None
        current_end: float | None = None
        for start, end in sorted(spans):
            if current_start is None or current_end is None:
                current_start, current_end = start, end
                continue
            if start <= current_end + merge_gap_seconds:
                current_end = max(current_end, end)
            else:
                merged.append({"speaker": speaker, "start": current_start, "end": current_end})
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            merged.append({"speaker": speaker, "start": current_start, "end": current_end})
    merged.sort(key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])))
    return merged


def _write_rttm(
    path: Path, *, file_id: str, intervals: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    speaker_map = {
        speaker: _safe_id(speaker)
        for speaker in sorted({str(item["speaker"]) for item in intervals})
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for interval in intervals:
        speaker = str(interval["speaker"])
        start = float(interval["start"])
        duration = max(0.0, float(interval["end"]) - start)
        if duration <= 0.0:
            continue
        lines.append(
            "SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>".format(
                file_id=file_id,
                start=start,
                duration=duration,
                speaker=speaker_map[speaker],
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return speaker_map


def _materialized_mixture_path(row: Mapping[str, object], *, manifest_dir: Path) -> Path | None:
    materialized = dict(row.get("materialized") or {})
    value = materialized.get("mixture_path")
    if not value:
        return None
    path = _resolve_path(value, manifest_dir=manifest_dir)
    return path if path.exists() else None


def _stem_path_for_member(stems: Sequence[Path], member: str) -> Path:
    target_name = Path(str(member)).name
    for stem in stems:
        if stem.name == target_name:
            return stem
    raise FileNotFoundError(f"Could not locate extracted member {member}")


def _audio_for_group(
    rows: Sequence[Mapping[str, object]],
    *,
    key: GroupKey,
    manifest_dir: Path,
    output_dir: Path,
    stems_cache_root: Path,
    force: bool,
) -> tuple[Path, str]:
    row = rows[0]
    file_id = _group_file_id(key)
    audio_path = output_dir / "audio" / f"{file_id}.wav"
    if audio_path.exists() and not force:
        return audio_path, "existing_export"

    materialized = _materialized_mixture_path(row, manifest_dir=manifest_dir)
    if materialized is not None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(materialized, audio_path)
        return audio_path, "copied_materialized"

    source_zip = row.get("source_zip")
    members = list(row.get("mixture_members") or [])
    if not source_zip or not members:
        raise ValueError(f"Group {file_id} has no materialized mixture and no source zip members")

    source_zip_path = _resolve_path(source_zip, manifest_dir=manifest_dir)
    stems = extract_session_stems(source_zip_path, stems_cache_root / _safe_id(key[0]))
    start = float(row.get("window_start") or 0.0)
    duration = max(float(row.get("duration") or 0.0), key[2] - key[1])
    source_dir = output_dir / "sources" / file_id
    clip_paths = []
    for member in members:
        stem_path = _stem_path_for_member(stems, str(member))
        clip_path = source_dir / f"{_safe_id(Path(str(member)).stem)}.wav"
        if force or not clip_path.exists():
            clip_audio(stem_path, clip_path, start=start, duration=duration)
        clip_paths.append(clip_path)
    mix_audio_files(clip_paths, audio_path)
    return audio_path, "materialized_from_zip"


def _group_file_id(key: GroupKey) -> str:
    session, start, end = key
    return f"{_safe_id(session)}_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"


def export_sortformer_manifests(
    *,
    manifest_path: Path,
    output_dir: Path,
    reference_jsonl: Path | None = None,
    splits: set[str] | None = None,
    require_reference: bool = False,
    merge_gap_seconds: float = 0.0,
    force: bool = False,
    stems_cache_root: Path | None = None,
) -> dict:
    manifest_dir = manifest_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stems_cache_root = stems_cache_root or (output_dir / "stem_cache")
    reference_groups = _load_reference_groups(reference_jsonl)
    rows = [
        row
        for row in _read_jsonl(manifest_path)
        if splits is None or str(row.get("split_id") or "") in splits
    ]
    grouped = _group_rows(rows)
    split_manifests: dict[str, list[dict]] = defaultdict(list)
    metadata_rows = []
    skipped = Counter()

    for key, group_rows in grouped.items():
        split = str(group_rows[0].get("split_id") or "unknown")
        duration = max(float(group_rows[0].get("duration") or 0.0), key[2] - key[1])
        reference_group = reference_groups.get(key)
        if reference_group is not None:
            raw_intervals = _intervals_from_reference(reference_group)
            reference_source = "forced"
        elif require_reference:
            skipped["missing_reference"] += 1
            continue
        else:
            raw_intervals = _intervals_from_manifest(group_rows)
            reference_source = "manifest"
        intervals = _clamp_and_merge_intervals(
            raw_intervals,
            duration=duration,
            merge_gap_seconds=merge_gap_seconds,
        )
        if not intervals:
            skipped["empty_intervals"] += 1
            continue

        file_id = _group_file_id(key)
        audio_path, audio_source = _audio_for_group(
            group_rows,
            key=key,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            stems_cache_root=stems_cache_root,
            force=force,
        )
        rttm_path = output_dir / "rttm" / f"{file_id}.rttm"
        speaker_map = _write_rttm(rttm_path, file_id=file_id, intervals=intervals)
        manifest_row = {
            "audio_filepath": str(audio_path.resolve()),
            "offset": 0.0,
            "duration": round(duration, 3),
            "label": "infer",
            "text": "-",
            "num_speakers": len(speaker_map),
            "rttm_filepath": str(rttm_path.resolve()),
            "uem_filepath": None,
            "ctm_filepath": None,
        }
        split_manifests[split].append(manifest_row)
        metadata_rows.append(
            {
                "file_id": file_id,
                "split_id": split,
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "duration": round(duration, 3),
                "source_rows": len(group_rows),
                "reference_source": reference_source,
                "raw_interval_count": len(raw_intervals),
                "rttm_interval_count": len(intervals),
                "speaker_count": len(speaker_map),
                "speaker_map": speaker_map,
                "audio_source": audio_source,
                "audio_filepath": manifest_row["audio_filepath"],
                "rttm_filepath": manifest_row["rttm_filepath"],
            }
        )

    for split, split_rows in sorted(split_manifests.items()):
        _write_jsonl(output_dir / f"{split}_manifest.jsonl", split_rows)
    _write_jsonl(output_dir / "sortformer_export_groups.jsonl", metadata_rows)
    summary = {
        "manifest": str(manifest_path),
        "reference_jsonl": str(reference_jsonl) if reference_jsonl else None,
        "output_dir": str(output_dir),
        "splits": sorted(split_manifests),
        "groups": len(metadata_rows),
        "skipped": dict(sorted(skipped.items())),
        "by_split": {
            split: {
                "groups": len(rows),
                "total_duration_seconds": round(sum(float(row["duration"]) for row in rows), 3),
                "max_speakers": max((int(row["num_speakers"]) for row in rows), default=0),
            }
            for split, rows in sorted(split_manifests.items())
        },
        "reference_sources": dict(Counter(str(row["reference_source"]) for row in metadata_rows)),
        "audio_sources": dict(Counter(str(row["audio_source"]) for row in metadata_rows)),
    }
    (output_dir / "sortformer_export_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Drive manifest groups as NeMo Sortformer audio manifests and RTTM labels."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--merge-gap-seconds", type=float, default=0.0)
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    summary = export_sortformer_manifests(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        reference_jsonl=args.reference_jsonl,
        splits=splits or None,
        require_reference=bool(args.require_reference),
        merge_gap_seconds=float(args.merge_gap_seconds),
        force=bool(args.force),
        stems_cache_root=args.stems_cache_root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
