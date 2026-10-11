from __future__ import annotations

import json
import logging
import math
import tempfile
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

    Behavior mirrors the prototype's flexible matching, with one guardrail:
      - Trim file extension for comparisons
      - Prefer the portion after the last hyphen (e.g., "2-foo_0" -> "foo_0")
      - Also strip any leading "<digits>-" prefix if present
      - Match case-insensitively using either exact stem or substring containment
      - Avoid fuzzy substring matches for numeric-only stems like "1.flac"
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
        if not _allow_fuzzy_speaker_match(cand, simplified):
            continue
        for key, value in normalized.items():
            if cand in key or key in cand:
                result = value
                return (result, True) if return_match else result
            key_simple = key_simplified.get(key)
            if simplified and key_simple and (simplified in key_simple or key_simple in simplified):
                result = value
                return (result, True) if return_match else result

    return (stem, False) if return_match else stem


def _allow_fuzzy_speaker_match(candidate: str, simplified: str) -> bool:
    if len(simplified) < 3:
        return False
    return any(ch.isalpha() for ch in candidate)


def split_segments(segments: List[dict], pause_threshold: float = 0.8) -> List[dict]:
    """Split timed words without changing their order, timestamps, or speaker metadata.

    Untimed or incomplete word lists retain their original segment rather than losing text.
    Only punctuation-only fragments are attached to a neighbour; short replies stand alone.
    """
    if not math.isfinite(pause_threshold) or pause_threshold <= 0:
        raise ValueError("pause_threshold must be finite and greater than zero")
    result: List[dict] = []
    for segment in segments:
        words = segment.get("words") or []
        # Ignore whitespace differences, but require complete lexical coverage.
        word_text = "".join(str(word.get("word", "")) for word in words)
        original_text = str(segment.get("text", ""))
        if (
            not words
            or re.sub(r"\s+", "", word_text) != re.sub(r"\s+", "", original_text)
            or any(word.get("start") is None or word.get("end") is None for word in words)
        ):
            result.append(segment)
            continue
        groups: List[List[dict]] = []
        pending: List[dict] = []
        for index, word in enumerate(words):
            pending.append(word)
            text = str(word["word"]).strip()
            final = bool(re.search(r"[.!?…][\"'”’)}\]]*$", text))
            if text.lower() in {"mr.", "mrs.", "ms.", "dr.", "st.", "e.g.", "i.e."}:
                final = False
            pause = index + 1 < len(words) and (
                float(words[index + 1]["start"]) - float(word["end"]) >= pause_threshold
            )
            if final or pause:
                groups.append(pending)
                pending = []
        if pending:
            groups.append(pending)
        # Punctuation alone has no meaning; preserve it alongside adjacent words.
        merged: List[List[dict]] = []
        for group in groups:
            if not any(any(ch.isalnum() for ch in str(w["word"])) for w in group) and merged:
                merged[-1].extend(group)
            elif merged and not any(any(ch.isalnum() for ch in str(w["word"])) for w in merged[-1]):
                merged[-1].extend(group)
            else:
                merged.append(group)
        for group in merged:
            result.append(
                {
                    **segment,
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": " ".join(str(w["word"]).strip() for w in group).strip(),
                    "words": group,
                }
            )
    return result


def split_multitrack_segments(
    all_segments: List[Tuple[str, List[dict]]],
    *,
    sentence_level: bool = True,
    pause_threshold: float = 0.8,
) -> List[Tuple[str, List[dict]]]:
    """Leave single-file sessions alone; use the same split records for every export."""
    if not sentence_level or len(all_segments) <= 1:
        return all_segments
    return [
        (filename, split_segments(segments, pause_threshold)) for filename, segments in all_segments
    ]


def check_speaker_roster(pairs: List[Tuple[str, str, str]], roster: List[str] | None) -> List[str]:
    """Report unexpected display names without dropping their dialogue."""
    if roster is None:
        return []
    unexpected = sorted({speaker for _, speaker, _ in pairs} - set(roster))
    if unexpected:
        logging.getLogger("transcriber").warning(
            "Output speakers outside configured roster: %s", ", ".join(unexpected)
        )
    return unexpected


def consolidate(
    all_segments: List[Tuple[str, List[dict]]],
    *,
    sentence_level: bool = True,
    pause_threshold: float = 0.8,
) -> List[Tuple[str, str, str]]:
    """Flatten per-file segments into a single timeline sorted by start time."""
    all_segments = split_multitrack_segments(
        all_segments, sentence_level=sentence_level, pause_threshold=pause_threshold
    )
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
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prototype text format: "<speaker> <HH:MM:SS> <text>" one per line
    txt_path = out_dir / f"{base_stem}.txt"
    transcript_lines = [f"{speaker} {ts} {text}\n" for ts, speaker, text in consolidated_pairs]
    _atomic_write_text(txt_path, "".join(transcript_lines))

    if write_jsonl_file:
        jsonl_path = out_dir / f"{base_stem}.jsonl"
        jsonl_records: list[dict] = []
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
                jsonl_records.append(record)
        jsonl_records.sort(key=lambda record: record["start"])
        _atomic_write_text(
            jsonl_path,
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in jsonl_records),
        )

    if write_srt_file:
        srt_path = out_dir / f"{base_stem}.srt"
        ordered = []
        for filename, segments in per_file_segments:
            for segment in segments:
                text = segment.get("text", "").strip()
                if text and len(per_file_segments) > 1:
                    speaker = segment.get("speaker") or Path(filename).stem
                    text = f"{speaker}: {text}"
                ordered.append(
                    (
                        float(segment.get("start") or 0.0),
                        float(segment.get("end") or 0.0),
                        text,
                    )
                )
        ordered.sort(key=lambda item: item[0])
        srt_payload = [
            (idx, start, end, text)
            for idx, (start, end, text) in enumerate(ordered, start=1)
            if text
        ]
        _atomic_write_srt(srt_path, srt_payload)

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
        _atomic_write_text(diar_path, json.dumps(serialisable, indent=2))

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
        _atomic_write_text(exclusive_path, json.dumps(serialisable, indent=2))

    return out_dir


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file:
            tmp_file.write(text)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_srt(
    path: Path,
    items: List[Tuple[int, float, float, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    try:
        tmp_file.close()
        write_srt(tmp_path, items)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
