from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .srt import write_srt
import re


def _format_ts(seconds: float) -> str:
    # Prototype-style HH:MM:SS (zero-padded), drop milliseconds
    secs = int(max(seconds, 0.0))
    hours, remainder = divmod(secs, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def choose_speaker(
    filename: str,
    mapping: Dict[str, str],
    *,
    return_match: bool = False,
) -> Union[str, Tuple[str, bool]]:
    """Pick a friendly speaker label based on path / mapping tokens.

    Behavior mirrors the prototype's flexible matching:
      - Trim file extension for comparisons
      - Prefer the portion after the last hyphen (e.g., "2-foo_0" -> "foo_0")
      - Also strip any leading "<digits>-" prefix if present
      - Match case-insensitively using either exact stem or substring containment
    """
    stem = Path(filename).stem
    if not mapping:
        return (stem, False) if return_match else stem

    # Normalise mapping keys (drop extension, lower)
    normalized = {Path(key).stem.lower(): value for key, value in mapping.items()}
    key_simplified: Dict[str, str] = {}
    simplified_lookup: Dict[str, str] = {}
    for key, value in normalized.items():
        simplified = re.sub(r"[^a-z0-9]+", "", key)
        key_simplified[key] = simplified
        if simplified and simplified not in simplified_lookup:
            simplified_lookup[simplified] = value

    raw_lower = stem.lower()
    # Candidate tokens from filename to match against mapping keys
    candidates = [raw_lower]
    # After last hyphen (prototype behavior)
    if "-" in raw_lower:
        candidates.append(raw_lower.split("-")[-1])
    # Strip any leading digits + hyphen (e.g., "2-foo_0" -> "foo_0")
    candidates.append(re.sub(r"^\d+-", "", raw_lower))

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq_candidates.append(c)

    # 1) Exact match on any candidate token (original or simplified)
    for cand in uniq_candidates:
        simplified = re.sub(r"[^a-z0-9]+", "", cand)
        if cand in normalized:
            value = normalized[cand]
            return (value, True) if return_match else value
        if simplified and simplified in simplified_lookup:
            value = simplified_lookup[simplified]
            return (value, True) if return_match else value

    # 2) Substring containment either direction (like the prototype)
    for cand in uniq_candidates:
        simplified = re.sub(r"[^a-z0-9]+", "", cand)
        for key, value in normalized.items():
            if cand in key or key in cand:
                result = value
                return (result, True) if return_match else result
            key_simple = key_simplified.get(key)
            if simplified and key_simple and (simplified in key_simple or key_simple in simplified):
                result = value
                return (result, True) if return_match else result

    return (stem, False) if return_match else stem


def consolidate(all_segments: List[Tuple[str, List[dict]]]) -> List[Tuple[str, str, str]]:
    """Flatten per-file segments into a single timeline sorted by start time."""
    flattened: List[Tuple[float, str, str]] = []
    for filename, segments in all_segments:
        fallback_speaker = Path(filename).stem
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            start = float(segment.get("start") or 0.0)
            speaker = segment.get("speaker") or fallback_speaker
            flattened.append((start, speaker, text))

    flattened.sort(key=lambda item: item[0])
    return [(_format_ts(start), speaker, text) for start, speaker, text in flattened]


def save_outputs(
    base_stem: str,
    output_dir: str,
    per_file_segments: List[Tuple[str, List[dict]]],
    consolidated_pairs: List[Tuple[str, str, str]],
    diar_by_file: Dict[str, List[dict]] | None,
    exclusive_diar_by_file: Dict[str, List[dict]] | None = None,
    write_srt_file: bool = True,
    write_jsonl_file: bool = True,
) -> Path:
    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback: use local ./outputs if the requested path is not writable
        local_fallback = Path.cwd() / "outputs"
        local_fallback.mkdir(parents=True, exist_ok=True)
        out_dir = local_fallback

    # Prototype text format: "<speaker> <HH:MM:SS> <text>" one per line
    txt_path = out_dir / f"{base_stem}.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        for ts, speaker, text in consolidated_pairs:
            handle.write(f"{speaker} {ts} {text}\n")

    if write_jsonl_file:
        jsonl_path = out_dir / f"{base_stem}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for filename, segments in per_file_segments:
                file_label = Path(filename).name
                for segment in segments:
                    record = {
                        "file": file_label,
                        "start": float(segment.get("start") or 0.0),
                        "end": float(segment.get("end") or 0.0),
                        "speaker": segment.get("speaker"),
                        "text": segment.get("text", "").strip(),
                    }
                    for meta_key in (
                        "speaker_raw",
                        "speaker_match_score",
                        "speaker_match_distance",
                        "speaker_match_cluster",
                        "speaker_match_source",
                    ):
                        if meta_key in segment:
                            record[meta_key] = segment.get(meta_key)
                    if "speaker_match" in segment:
                        record["speaker_match"] = segment["speaker_match"]
                    if "words" in segment:
                        record["words"] = segment["words"]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if write_srt_file:
        srt_path = out_dir / f"{base_stem}.srt"
        ordered = []
        for filename, segments in per_file_segments:
            for segment in segments:
                ordered.append(
                    (
                        float(segment.get("start") or 0.0),
                        float(segment.get("end") or 0.0),
                        segment.get("text", "").strip(),
                    )
                )
        ordered.sort(key=lambda item: item[0])
        srt_payload = [
            (idx, start, end, text)
            for idx, (start, end, text) in enumerate(ordered, start=1)
            if text
        ]
        write_srt(srt_path, srt_payload)

    if diar_by_file:
        diar_path = out_dir / f"{base_stem}.diarization.json"
        serialisable = {
            Path(filename).name: [
                {
                    "start": float(entry.get("start") or 0.0),
                    "end": float(entry.get("end") or 0.0),
                    "speaker": entry.get("speaker"),
                }
                for entry in entries
            ]
            for filename, entries in diar_by_file.items()
        }
        diar_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    if exclusive_diar_by_file:
        exclusive_path = out_dir / f"{base_stem}.exclusive_diarization.json"
        serialisable = {
            Path(filename).name: [
                {
                    "start": float(entry.get("start") or 0.0),
                    "end": float(entry.get("end") or 0.0),
                    "speaker": entry.get("speaker"),
                }
                for entry in entries
            ]
            for filename, entries in exclusive_diar_by_file.items()
        }
        exclusive_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    return out_dir
