from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torchaudio

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from transcriber.multitrack_eval import extract_session_stems  # noqa: E402


ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


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


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _select_groups(
    rows: Sequence[dict],
    *,
    session: str | None,
    sessions: set[str],
    window_start: float | None,
    max_groups: int | None,
    include_nonmaterialized: bool,
) -> list[tuple[tuple[str, float, float], list[dict]]]:
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        if not include_nonmaterialized and not row.get("materialized"):
            continue
        key = _group_key(row)
        if session and key[0] != session:
            continue
        if sessions and key[0] not in sessions:
            continue
        if window_start is not None and abs(key[1] - window_start) > 1e-3:
            continue
        grouped[key].append(row)
    items = sorted(grouped.items())
    return items[:max_groups] if max_groups is not None else items


def _parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _safe_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _run_ffmpeg(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True, capture_output=True, text=True)


class StemCache:
    def __init__(self, *, root: Path, wav_root: Path) -> None:
        self.root = root
        self.wav_root = wav_root
        self._members_by_session: dict[str, dict[str, Path]] = {}

    def _session_members(self, row: Mapping[str, object]) -> dict[str, Path]:
        session = str(row.get("session") or "unknown")
        if session in self._members_by_session:
            return self._members_by_session[session]
        session_id = _safe_id(session)
        extracted = extract_session_stems(Path(str(row["source_zip"])), self.root / session_id)
        members = {path.name: path for path in extracted}
        self._members_by_session[session] = members
        return members

    def wav_for_member(self, row: Mapping[str, object], member: object) -> Path:
        members = self._session_members(row)
        member_name = Path(str(member)).name
        if member_name not in members:
            raise FileNotFoundError(f"Missing member {member_name} for {row.get('row_id')}")
        session_id = _safe_id(row.get("session"))
        output = self.wav_root / session_id / f"{Path(member_name).stem}.wav"
        if output.exists() and _valid_cached_wav(output):
            return output
        if output.exists():
            output.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"cache_convert session={row.get('session')} member={member_name}", flush=True)
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(members[member_name]),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ]
        )
        return output


def _valid_cached_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        metadata = torchaudio.info(str(path))
    except Exception:
        return False
    return int(metadata.num_frames or 0) > 0 and int(metadata.sample_rate or 0) == 16000


def _number_to_words(value: int) -> str:
    if value < 20:
        return ONES[value]
    if value < 100:
        tens = value // 10 * 10
        rest = value % 10
        return TENS[tens] if rest == 0 else f"{TENS[tens]} {ONES[rest]}"
    if value < 1000:
        hundreds = value // 100
        rest = value % 100
        prefix = f"{ONES[hundreds]} hundred"
        return prefix if rest == 0 else f"{prefix} {_number_to_words(rest)}"
    if value < 1_000_000:
        thousands = value // 1000
        rest = value % 1000
        prefix = f"{_number_to_words(thousands)} thousand"
        return prefix if rest == 0 else f"{prefix} {_number_to_words(rest)}"
    return " ".join(ONES[int(digit)] for digit in str(value))


def _normalize_token(token: str, allowed: set[str]) -> str:
    text = token.lower().strip()
    text = text.replace("`", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    if re.fullmatch(r"\d+", text):
        text = _number_to_words(int(text))
    text = re.sub(r"\d+", lambda match: " " + _number_to_words(int(match.group(0))) + " ", text)
    text = text.replace("-", " ")
    pieces = []
    for raw_piece in text.split():
        chars = [char for char in raw_piece if char in allowed]
        normalized = "".join(chars).strip("'")
        if normalized:
            pieces.append(normalized)
    return "".join(pieces)


def _token_units(text: str, allowed: set[str]) -> list[dict]:
    units = []
    for index, token in enumerate(str(text or "").split()):
        normalized = _normalize_token(token, allowed)
        if normalized:
            units.append(
                {
                    "index": index,
                    "text": token,
                    "normalized": normalized,
                }
            )
    return units


def _fallback_word_spans(
    *,
    span: Mapping[str, object],
    units: Sequence[Mapping[str, object]],
    reason: str,
) -> list[dict]:
    start = max(float(span.get("start") or 0.0), 0.0)
    end = max(float(span.get("end") or start), start)
    if not units or end <= start:
        return []
    step = (end - start) / len(units)
    words = []
    for index, unit in enumerate(units):
        word_start = start + index * step
        word_end = end if index == len(units) - 1 else start + (index + 1) * step
        words.append(
            {
                "speaker": str(span.get("speaker") or ""),
                "start": round(float(word_start), 3),
                "end": round(float(max(word_end, word_start + 0.001)), 3),
                "text": unit["text"],
                "normalized": unit["normalized"],
                "score": None,
                "source_span_start": float(span.get("start") or 0.0),
                "source_span_end": float(span.get("end") or 0.0),
                "alignment_source": "source_span_fallback",
                "fallback_reason": reason,
            }
        )
    return words


def _load_track(path: Path, sample_rate: int) -> torch.Tensor:
    wave, got_sample_rate = torchaudio.load(str(path))
    if wave.ndim != 2:
        raise ValueError(f"Expected 2D waveform from {path}, got {tuple(wave.shape)}")
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if got_sample_rate != sample_rate:
        wave = torchaudio.functional.resample(wave, got_sample_rate, sample_rate)
    return torch.nan_to_num(wave.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _load_track_window(
    path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> torch.Tensor:
    frame_offset = max(0, int(round(start_seconds * sample_rate)))
    num_frames = max(1, int(round(duration_seconds * sample_rate)))
    wave, got_sample_rate = torchaudio.load(
        str(path),
        frame_offset=frame_offset,
        num_frames=num_frames,
    )
    if wave.ndim != 2:
        raise ValueError(f"Expected 2D waveform from {path}, got {tuple(wave.shape)}")
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if got_sample_rate != sample_rate:
        wave = torchaudio.functional.resample(wave, got_sample_rate, sample_rate)
    if wave.shape[-1] < num_frames:
        wave = torch.nn.functional.pad(wave, (0, num_frames - wave.shape[-1]))
    return torch.nan_to_num(wave[:, :num_frames].to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _align_span(
    *,
    span: Mapping[str, object],
    wave: torch.Tensor,
    sample_rate: int,
    model: torch.nn.Module,
    tokenizer,
    aligner,
    allowed: set[str],
    device: torch.device,
    pad_seconds: float,
    fallback_to_source_spans: bool,
) -> tuple[list[dict], dict]:
    units = _token_units(str(span.get("text") or ""), allowed)
    if not units:
        return [], {"status": "skipped", "reason": "no_alignable_tokens"}

    start = max(float(span.get("start") or 0.0), 0.0)
    end = max(float(span.get("end") or start), start)
    duration = wave.shape[-1] / sample_rate
    crop_start = max(0.0, start - pad_seconds)
    crop_end = min(duration, end + pad_seconds)
    start_sample = int(round(crop_start * sample_rate))
    end_sample = int(round(crop_end * sample_rate))
    if end_sample <= start_sample:
        return [], {"status": "skipped", "reason": "empty_crop"}

    crop = wave[:, start_sample:end_sample].to(device)
    if crop.shape[-1] < int(0.05 * sample_rate):
        return [], {"status": "skipped", "reason": "too_short"}

    try:
        tokenized = tokenizer([unit["normalized"] for unit in units])
        with torch.inference_mode():
            emission, _ = model(crop)
        spans = aligner(emission[0], tokenized)
    except Exception as exc:  # noqa: BLE001
        if fallback_to_source_spans:
            fallback_words = _fallback_word_spans(
                span=span,
                units=units,
                reason=type(exc).__name__,
            )
            if fallback_words:
                return fallback_words, {
                    "status": "fallback",
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "input_tokens": len(units),
                    "aligned_tokens": 0,
                    "fallback_tokens": len(fallback_words),
                }
        return [], {"status": "failed", "reason": type(exc).__name__, "detail": str(exc)}

    aligned_words = []
    frame_count = max(int(emission.shape[1]), 1)
    seconds_per_frame = (crop_end - crop_start) / frame_count
    for unit, token_spans in zip(units, spans, strict=False):
        if not token_spans:
            continue
        word_start = crop_start + min(item.start for item in token_spans) * seconds_per_frame
        word_end = crop_start + max(item.end for item in token_spans) * seconds_per_frame
        scores = [float(item.score) for item in token_spans]
        aligned_words.append(
            {
                "speaker": str(span.get("speaker") or ""),
                "start": round(float(word_start), 3),
                "end": round(float(max(word_end, word_start + 0.02)), 3),
                "text": unit["text"],
                "normalized": unit["normalized"],
                "score": sum(scores) / len(scores) if scores else None,
                "source_span_start": float(span.get("start") or 0.0),
                "source_span_end": float(span.get("end") or 0.0),
                "alignment_source": "mms",
            }
        )

    if aligned_words:
        return aligned_words, {
            "status": "ok",
            "reason": None,
            "input_tokens": len(units),
            "aligned_tokens": len(aligned_words),
        }
    if fallback_to_source_spans:
        fallback_words = _fallback_word_spans(
            span=span,
            units=units,
            reason="no_token_spans",
        )
        if fallback_words:
            return fallback_words, {
                "status": "fallback",
                "reason": "no_token_spans",
                "input_tokens": len(units),
                "aligned_tokens": 0,
                "fallback_tokens": len(fallback_words),
            }
    return [], {
        "status": "failed",
        "reason": "no_token_spans",
        "input_tokens": len(units),
        "aligned_tokens": 0,
    }


def _speaker_units(
    spans: Sequence[Mapping[str, object]],
    *,
    speaker: str,
    allowed: set[str],
) -> list[dict]:
    units = []
    for span in sorted(
        spans,
        key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)),
    ):
        if str(span.get("speaker") or "") != speaker:
            continue
        for unit in _token_units(str(span.get("text") or ""), allowed):
            units.append(
                {
                    **unit,
                    "speaker": speaker,
                    "source_span_start": float(span.get("start") or 0.0),
                    "source_span_end": float(span.get("end") or 0.0),
                }
            )
    return units


def _chunked_emission(
    wave: torch.Tensor,
    *,
    model: torch.nn.Module,
    sample_rate: int,
    device: torch.device,
    chunk_seconds: float,
) -> torch.Tensor:
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    emissions = []
    with torch.inference_mode():
        for start in range(0, wave.shape[-1], chunk_samples):
            chunk = wave[:, start : start + chunk_samples].to(device)
            if chunk.shape[-1] < int(0.05 * sample_rate):
                continue
            emission, _ = model(chunk)
            emissions.append(emission.cpu())
    if not emissions:
        raise ValueError("No audio chunks produced MMS emissions")
    return torch.cat(emissions, dim=1)


def _chunked_emissions(
    waves: Mapping[str, torch.Tensor],
    *,
    model: torch.nn.Module,
    sample_rate: int,
    device: torch.device,
    chunk_seconds: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    speakers = sorted(waves)
    output: dict[str, list[torch.Tensor]] = {speaker: [] for speaker in speakers}
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    with torch.inference_mode():
        for batch_start in range(0, len(speakers), batch_size):
            batch_speakers = speakers[batch_start : batch_start + batch_size]
            lengths = {waves[speaker].shape[-1] for speaker in batch_speakers}
            if len(lengths) != 1:
                raise ValueError("Speaker-window tracks must have equal lengths")
            wave_length = lengths.pop()
            for start in range(0, wave_length, chunk_samples):
                chunks = [
                    waves[speaker][:, start : start + chunk_samples] for speaker in batch_speakers
                ]
                maximum = max(chunk.shape[-1] for chunk in chunks)
                if maximum < int(0.05 * sample_rate):
                    continue
                padded = [
                    torch.nn.functional.pad(chunk, (0, maximum - chunk.shape[-1]))
                    for chunk in chunks
                ]
                batch = torch.cat(padded, dim=0).to(device)
                if batch.shape[0] > 1 and bool(getattr(model, "append_star", False)):
                    if bool(getattr(model, "normalize_waveform", False)):
                        batch = torch.nn.functional.layer_norm(batch, (batch.shape[-1],))
                    emission, _ = model.model(batch)
                    if bool(getattr(model, "apply_log_softmax", False)):
                        emission = torch.nn.functional.log_softmax(emission, dim=-1)
                    star = torch.zeros(
                        (batch.shape[0], emission.shape[1], 1),
                        dtype=emission.dtype,
                        device=emission.device,
                    )
                    emission = torch.cat((emission, star), dim=-1)
                else:
                    emission, _ = model(batch)
                for index, speaker in enumerate(batch_speakers):
                    output[speaker].append(emission[index : index + 1].cpu())
    return {speaker: torch.cat(parts, dim=1) for speaker, parts in output.items() if parts}


def _align_speaker_window(
    *,
    spans: Sequence[Mapping[str, object]],
    speaker: str,
    wave: torch.Tensor,
    sample_rate: int,
    model: torch.nn.Module,
    tokenizer,
    aligner,
    allowed: set[str],
    device: torch.device,
    emission_chunk_seconds: float,
    emission: torch.Tensor | None = None,
) -> tuple[list[dict], dict]:
    units = _speaker_units(spans, speaker=speaker, allowed=allowed)
    if not units:
        return [], {"status": "skipped", "reason": "no_alignable_tokens"}
    try:
        if emission is None:
            emission = _chunked_emission(
                wave,
                model=model,
                sample_rate=sample_rate,
                device=device,
                chunk_seconds=emission_chunk_seconds,
            )
        tokenized = tokenizer([unit["normalized"] for unit in units])
        aligned_spans = aligner(emission[0].to(device), tokenized)
    except Exception as exc:  # noqa: BLE001
        return [], {"status": "failed", "reason": type(exc).__name__, "detail": str(exc)}

    duration = wave.shape[-1] / sample_rate
    seconds_per_frame = duration / max(int(emission.shape[1]), 1)
    words = []
    for unit, token_spans in zip(units, aligned_spans, strict=False):
        if not token_spans:
            continue
        word_start = min(item.start for item in token_spans) * seconds_per_frame
        word_end = max(item.end for item in token_spans) * seconds_per_frame
        scores = [float(item.score) for item in token_spans]
        words.append(
            {
                "speaker": speaker,
                "start": round(float(word_start), 3),
                "end": round(float(max(word_end, word_start + 0.02)), 3),
                "text": unit["text"],
                "normalized": unit["normalized"],
                "score": sum(scores) / len(scores) if scores else None,
                "source_span_start": unit["source_span_start"],
                "source_span_end": unit["source_span_end"],
                "alignment_source": "mms_speaker_window",
            }
        )
    status = "ok" if len(words) == len(units) else "partial"
    return words, {
        "status": status,
        "reason": None if status == "ok" else "missing_token_spans",
        "input_tokens": len(units),
        "aligned_tokens": len(words),
        "emission_frames": int(emission.shape[1]),
    }


def _target_path(row: Mapping[str, object]) -> Path | None:
    materialized = dict(row.get("materialized") or {})
    target_path = materialized.get("target_source_path")
    return Path(str(target_path)) if target_path else None


def _target_wave(
    row: Mapping[str, object],
    *,
    sample_rate: int,
    stem_cache: StemCache | None,
) -> torch.Tensor | None:
    materialized_path = _target_path(row)
    if materialized_path and materialized_path.exists():
        return _load_track(materialized_path, sample_rate)
    if stem_cache is None:
        return None
    target_member = row.get("target_member")
    if not target_member:
        return None
    return _load_track_window(
        stem_cache.wav_for_member(row, target_member),
        start_seconds=float(row.get("window_start") or 0.0),
        duration_seconds=float(row.get("duration") or 0.0),
        sample_rate=sample_rate,
    )


def _align_group(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    *,
    model: torch.nn.Module,
    tokenizer,
    aligner,
    allowed: set[str],
    sample_rate: int,
    device: torch.device,
    pad_seconds: float,
    stem_cache: StemCache | None,
    fallback_to_source_spans: bool,
    alignment_scope: str,
    emission_chunk_seconds: float,
    emission_batch_size: int,
) -> dict:
    all_spans = [dict(span) for span in rows[0].get("word_spans") or []]
    words = []
    errors = []
    track_cache: dict[Path, torch.Tensor] = {}
    aligned_by_speaker: Counter[str] = Counter()
    fallback_by_speaker: Counter[str] = Counter()
    reference_by_speaker: Counter[str] = Counter()

    for span in all_spans:
        speaker = str(span.get("speaker") or "")
        reference_by_speaker[speaker] += len(str(span.get("text") or "").split())

    window_emissions = {}
    if alignment_scope == "speaker-window":
        waves_by_speaker = {}
        for row in rows:
            speaker = str(row.get("speaker_id") or "")
            cache_key = Path(
                str(_target_path(row) or f"{row.get('session')}::{row.get('target_member')}")
            )
            if cache_key not in track_cache:
                wave = _target_wave(row, sample_rate=sample_rate, stem_cache=stem_cache)
                if wave is None:
                    errors.append(
                        {
                            "speaker": speaker,
                            "reason": "missing_target_source",
                            "path": str(_target_path(row)),
                        }
                    )
                    continue
                track_cache[cache_key] = wave
            waves_by_speaker.setdefault(speaker, track_cache[cache_key])
        window_emissions = _chunked_emissions(
            waves_by_speaker,
            model=model,
            sample_rate=sample_rate,
            device=device,
            chunk_seconds=emission_chunk_seconds,
            batch_size=emission_batch_size,
        )

    for row in rows:
        speaker = str(row.get("speaker_id") or "")
        cache_key = Path(
            str(_target_path(row) or f"{row.get('session')}::{row.get('target_member')}")
        )
        if cache_key not in track_cache:
            wave = _target_wave(row, sample_rate=sample_rate, stem_cache=stem_cache)
            if wave is None:
                errors.append(
                    {
                        "speaker": speaker,
                        "reason": "missing_target_source",
                        "path": str(_target_path(row)),
                    }
                )
                continue
            track_cache[cache_key] = wave
        wave = track_cache[cache_key]
        if alignment_scope == "speaker-window":
            aligned, status = _align_speaker_window(
                spans=all_spans,
                speaker=speaker,
                wave=wave,
                sample_rate=sample_rate,
                model=model,
                tokenizer=tokenizer,
                aligner=aligner,
                allowed=allowed,
                device=device,
                emission_chunk_seconds=emission_chunk_seconds,
                emission=window_emissions.get(speaker),
            )
            words.extend(aligned)
            aligned_by_speaker[speaker] += len(aligned)
            if status.get("status") != "ok":
                errors.append({"speaker": speaker, **status})
            continue
        for span in all_spans:
            if str(span.get("speaker") or "") != speaker:
                continue
            aligned, status = _align_span(
                span=span,
                wave=wave,
                sample_rate=sample_rate,
                model=model,
                tokenizer=tokenizer,
                aligner=aligner,
                allowed=allowed,
                device=device,
                pad_seconds=pad_seconds,
                fallback_to_source_spans=fallback_to_source_spans,
            )
            words.extend(aligned)
            fallback_count = int(status.get("fallback_tokens") or 0)
            fallback_by_speaker[speaker] += fallback_count
            aligned_by_speaker[speaker] += max(0, len(aligned) - fallback_count)
            if status.get("status") != "ok":
                errors.append(
                    {
                        "speaker": speaker,
                        "span_start": span.get("start"),
                        "span_end": span.get("end"),
                        "text": span.get("text"),
                        **status,
                    }
                )

    words.sort(key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])))
    return {
        "session": key[0],
        "window_start": key[1],
        "window_end": key[2],
        "word_count": len(words),
        "reference_token_count": sum(reference_by_speaker.values()),
        "forced_word_count": sum(aligned_by_speaker.values()),
        "fallback_word_count": sum(fallback_by_speaker.values()),
        "aligned_by_speaker": dict(sorted(aligned_by_speaker.items())),
        "fallback_by_speaker": dict(sorted(fallback_by_speaker.items())),
        "reference_by_speaker": dict(sorted(reference_by_speaker.items())),
        "words": words,
        "errors": errors,
    }


def _summarize(groups: Sequence[Mapping[str, object]]) -> dict:
    reference_words = sum(int(group.get("reference_token_count") or 0) for group in groups)
    output_words = sum(int(group.get("word_count") or 0) for group in groups)
    forced_words = sum(int(group.get("forced_word_count") or 0) for group in groups)
    fallback_words = sum(int(group.get("fallback_word_count") or 0) for group in groups)
    error_count = sum(len(group.get("errors") or []) for group in groups)
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for speaker, count in dict(group.get("reference_by_speaker") or {}).items():
            by_speaker[str(speaker)]["reference"] += int(count)
        for speaker, count in dict(group.get("aligned_by_speaker") or {}).items():
            by_speaker[str(speaker)]["forced"] += int(count)
        for speaker, count in dict(group.get("fallback_by_speaker") or {}).items():
            by_speaker[str(speaker)]["fallback"] += int(count)
    return {
        "group_count": len(groups),
        "reference_token_count": reference_words,
        "aligned_word_count": output_words,
        "forced_word_count": forced_words,
        "fallback_word_count": fallback_words,
        "alignment_coverage": output_words / reference_words if reference_words else 0.0,
        "forced_alignment_coverage": forced_words / reference_words if reference_words else 0.0,
        "fallback_coverage": fallback_words / reference_words if reference_words else 0.0,
        "error_count": error_count,
        "by_speaker": {
            speaker: {
                "reference": counts["reference"],
                "aligned": counts["forced"] + counts["fallback"],
                "forced": counts["forced"],
                "fallback": counts["fallback"],
                "coverage": (
                    (counts["forced"] + counts["fallback"]) / counts["reference"]
                    if counts["reference"]
                    else 0.0
                ),
                "forced_coverage": (
                    counts["forced"] / counts["reference"] if counts["reference"] else 0.0
                ),
            }
            for speaker, counts in sorted(by_speaker.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MMS forced-aligned word references for TSE manifest groups."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-nonmaterialized",
        action="store_true",
        help="Align non-materialized rows by extracting/caching source stems from source_zip.",
    )
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--stems-wav-root", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--sessions", help="Comma-separated session names to align.")
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pad-seconds", type=float, default=0.75)
    parser.add_argument(
        "--alignment-scope",
        choices=("span", "speaker-window"),
        default="span",
        help="Align each source span or each speaker's full ordered window transcript.",
    )
    parser.add_argument("--emission-chunk-seconds", type=float, default=30.0)
    parser.add_argument("--emission-batch-size", type=int, default=4)
    parser.add_argument(
        "--fallback-to-source-spans",
        action="store_true",
        help=(
            "When MMS forced alignment fails, emit word intervals distributed across the source "
            "span. Intended for training RTTMs, not eval references."
        ),
    )
    args = parser.parse_args()
    if args.emission_batch_size < 1:
        parser.error("--emission-batch-size must be positive")

    rows = list(_read_jsonl(args.manifest))
    groups = _select_groups(
        rows,
        session=args.session,
        sessions=_parse_csv_set(args.sessions),
        window_start=args.window_start,
        max_groups=args.max_groups,
        include_nonmaterialized=bool(args.include_nonmaterialized),
    )
    stem_cache = None
    if args.include_nonmaterialized:
        stem_cache = StemCache(
            root=args.stems_cache_root or args.output_dir / "_stems",
            wav_root=args.stems_wav_root or args.output_dir / "_stems16",
        )

    bundle = torchaudio.pipelines.MMS_FA
    sample_rate = int(bundle.sample_rate)
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = bundle.get_model().to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    allowed = set(bundle.get_dict(star=None).keys()) - {"-"}

    group_results = []
    for index, (key, group_rows) in enumerate(groups, 1):
        print(
            f"align_group {index}/{len(groups)} session={key[0]} window_start={key[1]:.3f}",
            flush=True,
        )
        group_results.append(
            _align_group(
                key,
                group_rows,
                model=model,
                tokenizer=tokenizer,
                aligner=aligner,
                allowed=allowed,
                sample_rate=sample_rate,
                device=device,
                pad_seconds=float(args.pad_seconds),
                stem_cache=stem_cache,
                fallback_to_source_spans=bool(args.fallback_to_source_spans),
                alignment_scope=args.alignment_scope,
                emission_chunk_seconds=float(args.emission_chunk_seconds),
                emission_batch_size=int(args.emission_batch_size),
            )
        )
    summary = _summarize(group_results)
    summary["manifest"] = str(args.manifest)
    summary["device"] = str(device)
    summary["pad_seconds"] = float(args.pad_seconds)
    summary["include_nonmaterialized"] = bool(args.include_nonmaterialized)
    summary["fallback_to_source_spans"] = bool(args.fallback_to_source_spans)
    summary["alignment_scope"] = args.alignment_scope
    summary["emission_chunk_seconds"] = float(args.emission_chunk_seconds)
    summary["emission_batch_size"] = int(args.emission_batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "forced_word_reference_groups.jsonl", group_results)
    (args.output_dir / "forced_word_reference_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
