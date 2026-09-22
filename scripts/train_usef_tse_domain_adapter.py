from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from run_usef_tse_manifest import (  # noqa: E402
    DEFAULT_HF_REPO,
    _load_usef_tfgridnet,
)
from transcriber.multitrack_eval import extract_session_stems  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _run_ffmpeg(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True, capture_output=True, text=True)


class StemCache:
    def __init__(self, *, root: Path, wav_root: Path, sample_rate: int = 16000) -> None:
        self.root = root
        self.wav_root = wav_root
        self.sample_rate = int(sample_rate)
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
        if output.exists() and _valid_cached_wav(output, sample_rate=self.sample_rate):
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
                str(self.sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ]
        )
        return output


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


class ForcedReferenceIndex:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._groups: dict[tuple[str, float, float], list[dict]] = {}
        if path is None:
            return
        for row in _read_jsonl(path):
            key = (
                str(row.get("session") or ""),
                round(float(row.get("window_start") or 0.0), 3),
                round(float(row.get("window_end") or 0.0), 3),
            )
            self._groups[key] = [dict(word) for word in row.get("words") or []]

    def words_for(self, row: Mapping[str, object]) -> list[dict]:
        return list(self._groups.get(_group_key(row), []))

    def has_words_for(self, row: Mapping[str, object]) -> bool:
        return bool(self._groups.get(_group_key(row)))


def _row_target_word_count(row: Mapping[str, object]) -> int:
    explicit = row.get("target_word_count")
    if explicit is not None:
        return int(explicit or 0)
    target = str(row.get("speaker_id") or "")
    count = 0
    for span in row.get("word_spans") or []:
        item = dict(span)
        if str(item.get("speaker") or "") == target:
            count += len(str(item.get("text") or "").split())
    return count


def _forced_reference_coverage(
    rows: Sequence[Mapping[str, object]],
    forced_references: ForcedReferenceIndex | None,
) -> dict[str, object]:
    row_count = len(rows)
    rows_with_reference = 0
    manifest_target_words = 0
    forced_target_words = 0
    missing_row_ids: list[str] = []
    for row in rows:
        target = str(row.get("speaker_id") or "")
        expected_words = _row_target_word_count(row)
        manifest_target_words += expected_words
        forced_words = forced_references.words_for(row) if forced_references is not None else []
        target_words = [
            word for word in forced_words if str(dict(word).get("speaker") or "") == target
        ]
        forced_target_words += len(target_words)
        if target_words:
            rows_with_reference += 1
        else:
            missing_row_ids.append(str(row.get("row_id") or "unknown"))
    return {
        "row_count": row_count,
        "rows_with_forced_reference": rows_with_reference,
        "row_coverage": rows_with_reference / row_count if row_count else 0.0,
        "manifest_target_words": manifest_target_words,
        "forced_target_words": forced_target_words,
        "forced_target_word_coverage": (
            forced_target_words / manifest_target_words if manifest_target_words else 0.0
        ),
        "missing_row_ids": missing_row_ids[:25],
        "missing_row_count": len(missing_row_ids),
    }


def _validate_forced_reference_coverage(
    *,
    label: str,
    coverage: Mapping[str, object],
    require_forced_reference: bool,
    min_row_coverage: float,
    min_word_coverage: float,
) -> None:
    row_coverage = float(coverage.get("row_coverage") or 0.0)
    word_coverage = float(coverage.get("forced_target_word_coverage") or 0.0)
    should_gate = (
        bool(require_forced_reference)
        or float(min_row_coverage) > 0.0
        or float(min_word_coverage) > 0.0
    )
    if not should_gate:
        return
    failures = []
    if row_coverage < float(min_row_coverage):
        failures.append(f"row_coverage={row_coverage:.4f} < required {float(min_row_coverage):.4f}")
    if word_coverage < float(min_word_coverage):
        failures.append(
            "forced_target_word_coverage="
            f"{word_coverage:.4f} < required {float(min_word_coverage):.4f}"
        )
    if bool(require_forced_reference) and int(coverage.get("missing_row_count") or 0) > 0:
        failures.append(f"missing_row_count={coverage.get('missing_row_count')}")
    if failures:
        missing = ", ".join(str(item) for item in coverage.get("missing_row_ids") or [])
        raise ValueError(
            f"{label} forced-reference coverage gate failed: {'; '.join(failures)}"
            + (f"; first missing rows: {missing}" if missing else "")
        )


def _audio_metadata(path: Path) -> tuple[int, int]:
    info = getattr(torchaudio, "info", None)
    if callable(info):
        metadata = info(str(path))
        return int(metadata.sample_rate or 0), int(metadata.num_frames or 0)
    metadata = sf.info(str(path))
    return int(metadata.samplerate or 0), int(metadata.frames or 0)


def _valid_cached_wav(path: Path, *, sample_rate: int = 16000) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        got_sample_rate, frames = _audio_metadata(path)
    except Exception:
        return False
    return frames > 0 and got_sample_rate == int(sample_rate)


def _load_chunk(
    path: Path, *, start_seconds: float, duration_seconds: float, sample_rate: int
) -> torch.Tensor:
    try:
        source_sample_rate, source_frame_count = _audio_metadata(path)
    except Exception as exc:
        raise ValueError(f"Failed to inspect cached audio {path}") from exc
    if source_sample_rate <= 0:
        raise ValueError(f"Cached audio has no valid sample rate: {path}")
    frame_offset = max(0, int(round(start_seconds * source_sample_rate)))
    source_frames = int(round(duration_seconds * source_sample_rate))
    output_frames = int(round(duration_seconds * sample_rate))
    if frame_offset >= source_frame_count:
        return torch.zeros(output_frames, dtype=torch.float32)
    try:
        wave, got_sample_rate = torchaudio.load(
            str(path),
            frame_offset=frame_offset,
            num_frames=min(source_frames, source_frame_count - frame_offset),
        )
    except Exception as exc:
        raise ValueError(f"Failed to decode cached audio {path}") from exc
    if wave.ndim > 1:
        wave = wave.mean(dim=0)
    wave = wave.to(torch.float32).flatten()
    if int(got_sample_rate) != int(sample_rate):
        wave = torchaudio.functional.resample(
            wave.unsqueeze(0),
            orig_freq=int(got_sample_rate),
            new_freq=int(sample_rate),
        ).squeeze(0)
    if wave.numel() < output_frames:
        wave = F.pad(wave, (0, output_frames - wave.numel()))
    return torch.nan_to_num(wave[:output_frames], nan=0.0, posinf=0.0, neginf=0.0)


def _rms(wave: torch.Tensor) -> float:
    if wave.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(wave.to(torch.float32) ** 2)).detach().cpu())


def _resample_batch(waves: torch.Tensor, *, source_rate: int, target_rate: int) -> torch.Tensor:
    if int(source_rate) == int(target_rate):
        return waves
    return torchaudio.functional.resample(
        waves,
        orig_freq=int(source_rate),
        new_freq=int(target_rate),
    )


def _resample_mask_batch(
    masks: torch.Tensor, *, source_rate: int, target_rate: int
) -> torch.Tensor:
    if int(source_rate) == int(target_rate):
        return masks
    target_frames = max(1, int(round(masks.shape[-1] * float(target_rate) / float(source_rate))))
    return F.interpolate(masks.unsqueeze(1), size=target_frames, mode="nearest").squeeze(1)


def _target_spans(
    row: Mapping[str, object],
    *,
    forced_references: ForcedReferenceIndex | None = None,
) -> list[dict]:
    target = str(row.get("speaker_id") or "")
    if forced_references is not None:
        forced = [
            dict(word)
            for word in forced_references.words_for(row)
            if str(word.get("speaker") or "") == target
            and float(word.get("end") or 0.0) > float(word.get("start") or 0.0)
        ]
        if forced:
            return forced
    spans = []
    for span in row.get("word_spans") or []:
        item = dict(span)
        if str(item.get("speaker") or "") == target and float(item.get("end") or 0.0) > float(
            item.get("start") or 0.0
        ):
            spans.append(item)
    return spans


def _overlap_spans(
    row: Mapping[str, object],
    *,
    forced_references: ForcedReferenceIndex | None = None,
) -> list[dict]:
    target = str(row.get("speaker_id") or "")
    forced_words = forced_references.words_for(row) if forced_references is not None else []
    words = (
        [dict(word) for word in forced_words]
        if forced_words
        else [dict(span) for span in row.get("word_spans") or []]
    )
    target_spans = [word for word in words if str(word.get("speaker") or "") == target]
    non_owner_spans = [word for word in words if str(word.get("speaker") or "") != target]
    overlaps = []
    for target_span in target_spans:
        target_start = float(target_span.get("start") or 0.0)
        target_end = float(target_span.get("end") or target_start)
        for non_owner_span in non_owner_spans:
            non_owner_start = float(non_owner_span.get("start") or 0.0)
            non_owner_end = float(non_owner_span.get("end") or non_owner_start)
            start = max(target_start, non_owner_start)
            end = min(target_end, non_owner_end)
            if end > start:
                overlaps.append({"start": start, "end": end})
    return overlaps


def _sample_chunk_start(
    row: Mapping[str, object],
    *,
    rng: random.Random,
    chunk_seconds: float,
    active_probability: float,
    overlap_probability: float,
    forced_references: ForcedReferenceIndex | None,
) -> float:
    duration = float(row.get("duration") or 0.0)
    max_start = max(0.0, duration - chunk_seconds)
    overlaps = _overlap_spans(row, forced_references=forced_references)
    if overlaps and rng.random() < overlap_probability:
        span = rng.choice(overlaps)
        center = (float(span["start"]) + float(span["end"])) / 2.0
        jitter = rng.uniform(-chunk_seconds * 0.2, chunk_seconds * 0.2)
        return min(max(center + jitter - chunk_seconds / 2.0, 0.0), max_start)
    spans = _target_spans(row, forced_references=forced_references)
    if spans and rng.random() < active_probability:
        span = rng.choice(spans)
        center = (float(span["start"]) + float(span["end"])) / 2.0
        jitter = rng.uniform(-chunk_seconds * 0.35, chunk_seconds * 0.35)
        return min(max(center + jitter - chunk_seconds / 2.0, 0.0), max_start)
    return rng.uniform(0.0, max_start) if max_start > 0 else 0.0


def _paint_spans(
    spans: Sequence[Mapping[str, object]],
    *,
    chunk_start: float,
    chunk_seconds: float,
    sample_rate: int,
) -> torch.Tensor:
    samples = int(round(chunk_seconds * sample_rate))
    mask = torch.zeros(samples, dtype=torch.float32)
    chunk_end = chunk_start + chunk_seconds
    for span in spans:
        start = max(float(span.get("start") or 0.0), chunk_start)
        end = min(float(span.get("end") or start), chunk_end)
        if end <= start:
            continue
        start_sample = max(0, int(math.floor((start - chunk_start) * sample_rate)))
        end_sample = min(samples, int(math.ceil((end - chunk_start) * sample_rate)))
        mask[start_sample:end_sample] = 1.0
    return mask


def _word_masks(
    row: Mapping[str, object],
    *,
    chunk_start: float,
    chunk_seconds: float,
    sample_rate: int,
    forced_references: ForcedReferenceIndex | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = str(row.get("speaker_id") or "")
    forced_words = forced_references.words_for(row) if forced_references is not None else []
    if forced_words:
        target_spans = [word for word in forced_words if str(word.get("speaker") or "") == target]
        non_owner_spans = [
            word for word in forced_words if str(word.get("speaker") or "") != target
        ]
    else:
        target_spans = _target_spans(row)
        non_owner_spans = [
            item
            for span in row.get("word_spans") or []
            for item in [dict(span)]
            if str(item.get("speaker") or "") != target
        ]
    return (
        _paint_spans(
            target_spans,
            chunk_start=chunk_start,
            chunk_seconds=chunk_seconds,
            sample_rate=sample_rate,
        ),
        _paint_spans(
            non_owner_spans,
            chunk_start=chunk_start,
            chunk_seconds=chunk_seconds,
            sample_rate=sample_rate,
        ),
    )


def _load_training_example(
    row: Mapping[str, object],
    *,
    cache: StemCache,
    rng: random.Random,
    chunk_seconds: float,
    enrollment_seconds: float,
    sample_rate: int,
    active_probability: float,
    overlap_probability: float,
    min_mixture_rms: float,
    min_target_rms: float,
    min_enrollment_rms: float,
    forced_references: ForcedReferenceIndex | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_start = _sample_chunk_start(
        row,
        rng=rng,
        chunk_seconds=chunk_seconds,
        active_probability=active_probability,
        overlap_probability=overlap_probability,
        forced_references=forced_references,
    )
    absolute_start = float(row["window_start"]) + chunk_start
    source_chunks = [
        _load_chunk(
            cache.wav_for_member(row, member),
            start_seconds=absolute_start,
            duration_seconds=chunk_seconds,
            sample_rate=sample_rate,
        )
        for member in row.get("mixture_members") or []
    ]
    if not source_chunks:
        raise ValueError(f"No mixture members for {row.get('row_id')}")
    mixture = torch.stack(source_chunks).sum(dim=0)
    target = _load_chunk(
        cache.wav_for_member(row, row["target_member"]),
        start_seconds=absolute_start,
        duration_seconds=chunk_seconds,
        sample_rate=sample_rate,
    )
    if _rms(mixture) < min_mixture_rms or _rms(target) < min_target_rms:
        raise ValueError(f"low-rms training chunk for {row.get('row_id')}")
    peak = max(float(torch.max(torch.abs(mixture))), 1e-6)
    if peak > 0.95:
        scale = 0.95 / peak
        mixture = mixture * scale
        target = target * scale

    enrollments = list(row.get("positive_enrollment_spans") or [])
    if not enrollments:
        raise ValueError(f"No positive enrollments for {row.get('row_id')}")
    enrollment_span = dict(rng.choice(enrollments))
    enrollment_duration = float(enrollment_span.get("duration") or 0.0)
    max_offset = max(0.0, enrollment_duration - enrollment_seconds)
    enrollment_start = float(enrollment_span["start"]) + (
        rng.uniform(0.0, max_offset) if max_offset else 0.0
    )
    enrollment = _load_chunk(
        cache.wav_for_member(row, row["target_member"]),
        start_seconds=enrollment_start,
        duration_seconds=enrollment_seconds,
        sample_rate=sample_rate,
    )
    if _rms(enrollment) < min_enrollment_rms:
        raise ValueError(f"low-rms enrollment for {row.get('row_id')}")
    target_mask, non_owner_mask = _word_masks(
        row,
        chunk_start=chunk_start,
        chunk_seconds=chunk_seconds,
        sample_rate=sample_rate,
        forced_references=forced_references,
    )
    return mixture, target, enrollment, target_mask, non_owner_mask


def _si_snr_loss(
    estimate: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    target_energy = torch.sum(target * target, dim=-1, keepdim=True).clamp_min(eps)
    projection = torch.sum(estimate * target, dim=-1, keepdim=True) * target / target_energy
    noise = estimate - projection
    ratio = torch.sum(projection * projection, dim=-1).clamp_min(eps) / torch.sum(
        noise * noise,
        dim=-1,
    ).clamp_min(eps)
    return -10.0 * torch.log10(ratio).mean()


def _activity_loss_components(
    estimate: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    non_owner_mask: torch.Tensor,
    *,
    active_weight: float,
    silence_weight: float,
    non_owner_weight: float,
    overlap_recon_weight: float,
) -> dict[str, torch.Tensor]:
    active = target_mask
    inactive = (1.0 - active).clamp(0.0, 1.0)
    non_owner_only = _interferer_only_mask(active, non_owner_mask)
    overlap = _overlap_mask(active, non_owner_mask)
    active_weights = 1.0 + active_weight * active
    weighted_l1 = torch.abs(estimate - target) * active_weights
    weighted_l1_loss = weighted_l1.sum(dim=-1) / active_weights.sum(dim=-1).clamp_min(1.0)
    inactive_den = inactive.sum(dim=-1).clamp_min(1.0)
    silence_loss = (torch.abs(estimate) * inactive).sum(dim=-1) / inactive_den
    non_owner_den = non_owner_only.sum(dim=-1).clamp_min(1.0)
    non_owner_loss = (torch.abs(estimate) * non_owner_only).sum(dim=-1) / non_owner_den
    overlap_den = overlap.sum(dim=-1).clamp_min(1.0)
    overlap_recon_loss = (torch.abs(estimate - target) * overlap).sum(dim=-1) / overlap_den
    target_active = (torch.abs(target) * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
    estimate_active = (torch.abs(estimate) * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
    presence_loss = F.relu((0.25 * target_active.detach()) - estimate_active).mean()
    loss = (
        weighted_l1_loss.mean()
        + silence_weight * silence_loss.mean()
        + non_owner_weight * non_owner_loss.mean()
        + overlap_recon_weight * overlap_recon_loss.mean()
        + 0.1 * presence_loss
    )
    return {
        "loss": loss,
        "weighted_l1_loss": weighted_l1_loss.mean(),
        "silence_loss": silence_loss.mean(),
        "non_owner_loss": non_owner_loss.mean(),
        "overlap_recon_loss": overlap_recon_loss.mean(),
        "presence_loss": presence_loss,
        "target_active_fraction": active.mean(),
        "interferer_only_fraction": non_owner_only.mean(),
        "overlap_fraction": overlap.mean(),
    }


def _activity_losses(
    estimate: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    non_owner_mask: torch.Tensor,
    *,
    active_weight: float,
    silence_weight: float,
    non_owner_weight: float,
    overlap_recon_weight: float = 0.0,
) -> torch.Tensor:
    return _activity_loss_components(
        estimate,
        target,
        target_mask,
        non_owner_mask,
        active_weight=active_weight,
        silence_weight=silence_weight,
        non_owner_weight=non_owner_weight,
        overlap_recon_weight=overlap_recon_weight,
    )["loss"]


def _masked_energy_and_count(
    wave: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=-1)
    energy = (wave.to(torch.float32).square() * mask).sum(dim=-1) / count.clamp_min(1.0)
    return energy.clamp_min(eps), count


def _mean_or_negative_inf(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("-inf")
    return float(values.mean().detach().cpu())


def _tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _masked_energy(wave: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    energy, _ = _masked_energy_and_count(wave, mask, eps=eps)
    return energy.clamp_min(eps)


def _db_ratio(
    numerator: torch.Tensor, denominator: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    return 10.0 * torch.log10(numerator.clamp_min(eps) / denominator.clamp_min(eps))


def _interferer_only_mask(target_mask: torch.Tensor, non_owner_mask: torch.Tensor) -> torch.Tensor:
    inactive = (1.0 - target_mask).clamp(0.0, 1.0)
    return (non_owner_mask * inactive).clamp(0.0, 1.0)


def _overlap_mask(target_mask: torch.Tensor, non_owner_mask: torch.Tensor) -> torch.Tensor:
    return (target_mask * non_owner_mask).clamp(0.0, 1.0)


def _suppression_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    non_owner_mask: torch.Tensor,
) -> dict[str, float]:
    active = target_mask
    non_owner_only = _interferer_only_mask(active, non_owner_mask)
    target_energy, target_count = _masked_energy_and_count(estimate, active)
    target_reference_energy, _ = _masked_energy_and_count(target, active)
    non_owner_energy, non_owner_count = _masked_energy_and_count(estimate, non_owner_only)
    valid_target = target_count > 1.0
    valid_suppression = valid_target & (non_owner_count > 1.0)
    target_to_non_owner = _db_ratio(
        target_energy[valid_suppression],
        non_owner_energy[valid_suppression],
    )
    target_retention = _db_ratio(
        target_energy[valid_suppression],
        target_reference_energy[valid_suppression],
    )
    # Reward leakage reduction, but penalize disappearing target speech.
    suppression_score = target_to_non_owner + torch.minimum(
        target_retention,
        torch.zeros_like(target_retention),
    )
    target_retention_all = _db_ratio(
        target_energy[valid_target], target_reference_energy[valid_target]
    )
    return {
        "target_to_non_owner_db": _mean_or_negative_inf(target_to_non_owner),
        "target_retention_db": _mean_or_negative_inf(target_retention_all),
        "suppression_score": _mean_or_negative_inf(suppression_score),
        "suppression_valid_examples": int(valid_suppression.sum().detach().cpu()),
    }


def _build_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    cache: StemCache,
    rng: random.Random,
    batch_size: int,
    chunk_seconds: float,
    enrollment_seconds: float,
    source_sample_rate: int,
    model_sample_rate: int,
    active_probability: float,
    overlap_probability: float,
    min_mixture_rms: float,
    min_target_rms: float,
    min_enrollment_rms: float,
    max_retries: int,
    forced_references: ForcedReferenceIndex | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixtures = []
    targets = []
    enrollments = []
    target_masks = []
    non_owner_masks = []
    attempts = 0
    while len(mixtures) < batch_size and attempts < max_retries:
        attempts += 1
        row = rows[rng.randrange(0, len(rows))]
        try:
            mixture, target, enrollment, target_mask, non_owner_mask = _load_training_example(
                row,
                cache=cache,
                rng=rng,
                chunk_seconds=chunk_seconds,
                enrollment_seconds=enrollment_seconds,
                sample_rate=source_sample_rate,
                active_probability=active_probability,
                overlap_probability=overlap_probability,
                min_mixture_rms=min_mixture_rms,
                min_target_rms=min_target_rms,
                min_enrollment_rms=min_enrollment_rms,
                forced_references=forced_references,
            )
        except (OSError, RuntimeError, ValueError):
            continue
        mixtures.append(mixture)
        targets.append(target)
        enrollments.append(enrollment)
        target_masks.append(target_mask)
        non_owner_masks.append(non_owner_mask)
    if len(mixtures) < batch_size:
        raise RuntimeError(f"Could only build {len(mixtures)}/{batch_size} examples")
    mixture_batch = _resample_batch(
        torch.stack(mixtures), source_rate=source_sample_rate, target_rate=model_sample_rate
    )
    target_batch = _resample_batch(
        torch.stack(targets), source_rate=source_sample_rate, target_rate=model_sample_rate
    )
    enrollment_batch = _resample_batch(
        torch.stack(enrollments),
        source_rate=source_sample_rate,
        target_rate=model_sample_rate,
    )
    target_mask_batch = _resample_mask_batch(
        torch.stack(target_masks),
        source_rate=source_sample_rate,
        target_rate=model_sample_rate,
    )
    non_owner_mask_batch = _resample_mask_batch(
        torch.stack(non_owner_masks),
        source_rate=source_sample_rate,
        target_rate=model_sample_rate,
    )
    return (
        mixture_batch,
        target_batch,
        enrollment_batch,
        target_mask_batch.clamp(0.0, 1.0),
        non_owner_mask_batch.clamp(0.0, 1.0),
    )


def _evaluate_chunks(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, object]],
    *,
    cache: StemCache,
    args: argparse.Namespace,
    rng: random.Random,
    device: torch.device,
    forced_references: ForcedReferenceIndex | None,
) -> dict[str, float]:
    model.eval()
    losses = []
    sisnrs = []
    target_to_non_owner = []
    target_retention = []
    suppression_scores = []
    suppression_valid_examples = 0
    activity_component_values: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for _ in range(max(1, int(args.dev_batches))):
            mixture, target, enrollment, target_mask, non_owner_mask = _build_batch(
                rows,
                cache=cache,
                rng=rng,
                batch_size=int(args.batch_size),
                chunk_seconds=float(args.chunk_seconds),
                enrollment_seconds=float(args.enrollment_seconds),
                source_sample_rate=int(args.source_sample_rate),
                model_sample_rate=int(args.model_sample_rate),
                active_probability=float(args.activity_probability),
                overlap_probability=float(args.dev_overlap_probability),
                min_mixture_rms=float(args.min_mixture_rms),
                min_target_rms=float(args.min_target_rms),
                min_enrollment_rms=float(args.min_enrollment_rms),
                max_retries=int(args.max_batch_retries),
                forced_references=forced_references,
            )
            mixture = mixture.to(device)
            target = target.to(device)
            enrollment = enrollment.to(device)
            target_mask = target_mask.to(device)
            non_owner_mask = non_owner_mask.to(device)
            estimate = model(mixture, enrollment).reshape_as(target)
            if not torch.isfinite(estimate).all():
                losses.append(float("inf"))
                sisnrs.append(float("-inf"))
                continue
            sisnr_loss = _si_snr_loss(estimate, target)
            activity_components = _activity_loss_components(
                estimate,
                target,
                target_mask,
                non_owner_mask,
                active_weight=float(args.active_weight),
                silence_weight=float(args.silence_weight),
                non_owner_weight=float(args.non_owner_weight),
                overlap_recon_weight=float(args.overlap_recon_weight),
            )
            loss = sisnr_loss + float(args.activity_weight) * activity_components["loss"]
            if not torch.isfinite(loss):
                losses.append(float("inf"))
                sisnrs.append(float("-inf"))
                continue
            losses.append(float(loss.detach().cpu()))
            sisnrs.append(float((-sisnr_loss).detach().cpu()))
            for name, value in activity_components.items():
                activity_component_values[name].append(_tensor_float(value))
            suppression = _suppression_metrics(estimate, target, target_mask, non_owner_mask)
            target_retention.append(suppression["target_retention_db"])
            suppression_valid_examples += int(suppression["suppression_valid_examples"])
            if int(suppression["suppression_valid_examples"]) > 0:
                target_to_non_owner.append(suppression["target_to_non_owner_db"])
                suppression_scores.append(suppression["suppression_score"])
    model.train()
    result = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "si_snr": float(np.mean(sisnrs)) if sisnrs else 0.0,
        "target_to_non_owner_db": (
            float(np.mean(target_to_non_owner)) if target_to_non_owner else float("-inf")
        ),
        "target_retention_db": (
            float(np.mean(target_retention)) if target_retention else float("-inf")
        ),
        "suppression_score": (
            float(np.mean(suppression_scores)) if suppression_scores else float("-inf")
        ),
        "suppression_valid_examples": suppression_valid_examples,
    }
    for name, values in sorted(activity_component_values.items()):
        result[f"activity_{name}"] = float(np.mean(values)) if values else 0.0
    return result


def _metric_value(record: Mapping[str, object], metric: str) -> float:
    value = record.get(metric)
    return float(value) if value is not None else float("-inf")


def _metric_is_better(value: float, best_value: float, metric: str) -> bool:
    if metric == "dev_loss":
        return value < best_value
    return value > best_value


def _initial_best_value(metric: str) -> float:
    return float("inf") if metric == "dev_loss" else float("-inf")


def _parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_rows(
    path: Path,
    splits: set[str],
    *,
    sessions: set[str],
    max_rows: int | None,
) -> list[dict]:
    rows = []
    for row in _read_jsonl(path):
        if str(row.get("split_id") or "") not in splits:
            continue
        if sessions and str(row.get("session") or "") not in sessions:
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError(
            f"No rows found for splits {sorted(splits)} sessions {sorted(sessions)} in {path}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune USEF-TFGridNet on domain target-speaker extraction chunks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--stems-wav-root", type=Path)
    parser.add_argument(
        "--forced-reference-jsonl",
        type=Path,
        help="Optional forced-aligned word reference groups for precise owner/suppression masks.",
    )
    parser.add_argument(
        "--require-forced-reference",
        action="store_true",
        help="Fail if any selected train/dev row lacks forced target-word references.",
    )
    parser.add_argument(
        "--min-forced-reference-row-coverage",
        type=float,
        default=0.0,
        help="Minimum selected-row forced-reference coverage required before training.",
    )
    parser.add_argument(
        "--min-forced-reference-word-coverage",
        type=float,
        default=0.0,
        help="Minimum selected target-word forced-reference coverage required before training.",
    )
    parser.add_argument("--usef-repo", type=Path, required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--checkpoint-file", default="chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar"
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--train-sessions")
    parser.add_argument("--dev-sessions")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-dev-rows", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--dev-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--enrollment-seconds", type=float, default=10.0)
    parser.add_argument("--source-sample-rate", type=int, default=16000)
    parser.add_argument("--model-sample-rate", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--activity-probability", type=float, default=0.85)
    parser.add_argument(
        "--train-overlap-probability",
        type=float,
        default=0.0,
        help="Probability of centering a training chunk on forced-aligned speaker overlap.",
    )
    parser.add_argument(
        "--dev-overlap-probability",
        type=float,
        default=0.0,
        help="Probability of centering a deterministic dev chunk on speaker overlap.",
    )
    parser.add_argument("--activity-weight", type=float, default=0.25)
    parser.add_argument("--active-weight", type=float, default=4.0)
    parser.add_argument("--silence-weight", type=float, default=2.0)
    parser.add_argument("--non-owner-weight", type=float, default=3.0)
    parser.add_argument(
        "--overlap-recon-weight",
        type=float,
        default=0.0,
        help="Extra target reconstruction weight only on target/non-target overlap samples.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("dev_loss", "dev_si_snr", "dev_target_to_non_owner_db", "dev_suppression_score"),
        default="dev_loss",
        help="Metric used to write best_domain_adapter.pt.",
    )
    parser.add_argument("--min-mixture-rms", type=float, default=1e-4)
    parser.add_argument("--min-target-rms", type=float, default=1e-4)
    parser.add_argument("--min-enrollment-rms", type=float, default=1e-4)
    parser.add_argument("--max-batch-retries", type=int, default=80)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = StemCache(
        root=args.stems_cache_root or args.output_dir / "_stems",
        wav_root=args.stems_wav_root or args.output_dir / "_stems16",
    )
    forced_references = ForcedReferenceIndex(args.forced_reference_jsonl)
    train_rows = _load_rows(
        args.manifest,
        {str(args.train_split)},
        sessions=_parse_csv_set(args.train_sessions),
        max_rows=args.max_train_rows,
    )
    dev_rows = _load_rows(
        args.manifest,
        {str(args.dev_split)},
        sessions=_parse_csv_set(args.dev_sessions),
        max_rows=args.max_dev_rows,
    )
    train_forced_reference_coverage = _forced_reference_coverage(train_rows, forced_references)
    dev_forced_reference_coverage = _forced_reference_coverage(dev_rows, forced_references)
    _validate_forced_reference_coverage(
        label="train",
        coverage=train_forced_reference_coverage,
        require_forced_reference=bool(args.require_forced_reference),
        min_row_coverage=float(args.min_forced_reference_row_coverage),
        min_word_coverage=float(args.min_forced_reference_word_coverage),
    )
    _validate_forced_reference_coverage(
        label="dev",
        coverage=dev_forced_reference_coverage,
        require_forced_reference=bool(args.require_forced_reference),
        min_row_coverage=float(args.min_forced_reference_row_coverage),
        min_word_coverage=float(args.min_forced_reference_word_coverage),
    )
    metadata = {
        "manifest": str(args.manifest),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "forced_reference_coverage": {
            "train": train_forced_reference_coverage,
            "dev": dev_forced_reference_coverage,
        },
        "args": vars(args),
    }
    (args.output_dir / "domain_adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    model = _load_usef_tfgridnet(
        usef_repo=args.usef_repo,
        hf_repo=str(args.hf_repo),
        checkpoint_file=str(args.checkpoint_file),
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_rng = random.Random(int(args.seed))
    best_metric = str(args.selection_metric)
    best_value = _initial_best_value(best_metric)
    history = []

    for step in range(1, int(args.train_steps) + 1):
        mixture, target, enrollment, target_mask, non_owner_mask = _build_batch(
            train_rows,
            cache=cache,
            rng=train_rng,
            batch_size=int(args.batch_size),
            chunk_seconds=float(args.chunk_seconds),
            enrollment_seconds=float(args.enrollment_seconds),
            source_sample_rate=int(args.source_sample_rate),
            model_sample_rate=int(args.model_sample_rate),
            active_probability=float(args.activity_probability),
            overlap_probability=float(args.train_overlap_probability),
            min_mixture_rms=float(args.min_mixture_rms),
            min_target_rms=float(args.min_target_rms),
            min_enrollment_rms=float(args.min_enrollment_rms),
            max_retries=int(args.max_batch_retries),
            forced_references=forced_references,
        )
        mixture = mixture.to(device)
        target = target.to(device)
        enrollment = enrollment.to(device)
        target_mask = target_mask.to(device)
        non_owner_mask = non_owner_mask.to(device)
        estimate = model(mixture, enrollment).reshape_as(target)
        if not torch.isfinite(estimate).all():
            print(json.dumps({"step": step, "skip": "nonfinite_estimate"}), flush=True)
            continue
        sisnr_loss = _si_snr_loss(estimate, target)
        activity_components = _activity_loss_components(
            estimate,
            target,
            target_mask,
            non_owner_mask,
            active_weight=float(args.active_weight),
            silence_weight=float(args.silence_weight),
            non_owner_weight=float(args.non_owner_weight),
            overlap_recon_weight=float(args.overlap_recon_weight),
        )
        activity_loss = activity_components["loss"]
        loss = sisnr_loss + float(args.activity_weight) * activity_loss
        if not torch.isfinite(loss):
            print(json.dumps({"step": step, "skip": "nonfinite_loss"}), flush=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.train_steps):
            eval_rng = random.Random(int(args.seed) + 1)
            dev_metrics = _evaluate_chunks(
                model,
                dev_rows,
                cache=cache,
                args=args,
                rng=eval_rng,
                device=device,
                forced_references=forced_references,
            )
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_si_snr": float((-sisnr_loss).detach().cpu()),
                "train_activity_loss": float(activity_loss.detach().cpu()),
                "train_activity_weighted_l1_loss": _tensor_float(
                    activity_components["weighted_l1_loss"]
                ),
                "train_activity_silence_loss": _tensor_float(activity_components["silence_loss"]),
                "train_activity_non_owner_loss": _tensor_float(
                    activity_components["non_owner_loss"]
                ),
                "train_activity_overlap_recon_loss": _tensor_float(
                    activity_components["overlap_recon_loss"]
                ),
                "train_activity_presence_loss": _tensor_float(activity_components["presence_loss"]),
                "train_activity_target_active_fraction": _tensor_float(
                    activity_components["target_active_fraction"]
                ),
                "train_activity_interferer_only_fraction": _tensor_float(
                    activity_components["interferer_only_fraction"]
                ),
                "train_activity_overlap_fraction": _tensor_float(
                    activity_components["overlap_fraction"]
                ),
                "dev_loss": dev_metrics["loss"],
                "dev_si_snr": dev_metrics["si_snr"],
                "dev_target_to_non_owner_db": dev_metrics["target_to_non_owner_db"],
                "dev_target_retention_db": dev_metrics["target_retention_db"],
                "dev_suppression_score": dev_metrics["suppression_score"],
                "dev_suppression_valid_examples": dev_metrics["suppression_valid_examples"],
            }
            for name, value in sorted(dev_metrics.items()):
                if name.startswith("activity_"):
                    record[f"dev_{name}"] = value
            history.append(record)
            print(json.dumps(record), flush=True)
            metric_value = _metric_value(record, best_metric)
            if _metric_is_better(metric_value, best_value, best_metric):
                best_value = metric_value
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "metadata": metadata,
                        "history": history,
                        "best_step": step,
                        "best_metric": best_metric,
                        "best_metric_value": best_value,
                        "best_dev_loss": dev_metrics["loss"],
                    },
                    args.output_dir / "best_domain_adapter.pt",
                )
        if step % int(args.save_every) == 0 or step == int(args.train_steps):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "history": history,
                    "step": step,
                },
                args.output_dir / "last_domain_adapter.pt",
            )

    (args.output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_metric": best_metric,
                "best_metric_value": best_value,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
