from __future__ import annotations

import argparse
import json
import math
import re
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def _load_reference_groups(path: Path) -> dict[tuple[str, float, float], list[dict]]:
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in _read_jsonl(path):
        groups[_group_key(row)] = [dict(word) for word in row.get("words") or []]
    return groups


def _split_values(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


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


class StemCache:
    def __init__(self, *, root: Path) -> None:
        self.root = root
        self._members_by_session: dict[str, dict[str, Path]] = {}

    def _session_members(self, row: Mapping[str, object]) -> dict[str, Path]:
        session = str(row.get("session") or "unknown")
        if session in self._members_by_session:
            return self._members_by_session[session]
        session_id = _safe_id(session)
        output_dir = self.root / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".m4a"}
        )
        if not existing:
            with ZipFile(Path(str(row["source_zip"]))) as archive:
                archive.extractall(output_dir)
            existing = sorted(
                path
                for path in output_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".m4a"}
            )
        members = {path.name: path for path in existing}
        self._members_by_session[session] = members
        return members

    def path_for_member(self, row: Mapping[str, object], member: object) -> Path:
        members = self._session_members(row)
        member_name = Path(str(member)).name
        if member_name not in members:
            raise FileNotFoundError(f"Missing member {member_name} for {row.get('row_id')}")
        return members[member_name]


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0), int(sample_rate)


def _load_audio_window(
    path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    sample_rate = int(info.samplerate)
    frame_offset = max(0, int(round(start_seconds * sample_rate)))
    num_frames = max(1, int(round(duration_seconds * sample_rate)))
    audio, _ = sf.read(
        path,
        start=frame_offset,
        frames=num_frames,
        dtype="float32",
        always_2d=False,
    )
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if wave.shape[0] < num_frames:
        wave = np.pad(wave, (0, num_frames - wave.shape[0]))
    wave = np.nan_to_num(wave[:num_frames], nan=0.0, posinf=0.0, neginf=0.0)
    return wave.astype(np.float32, copy=False), sample_rate


def _resample_if_needed(wave: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return wave.astype(np.float32, copy=False)
    import torchaudio.functional as AF

    tensor = torch.from_numpy(wave.astype(np.float32, copy=False)).unsqueeze(0)
    resampled = AF.resample(tensor, orig_freq=int(source_rate), new_freq=int(target_rate))
    return resampled.squeeze(0).numpy().astype(np.float32, copy=False)


def _rms_normalize(wave: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if wave.size else 0.0
    if rms <= eps:
        return wave.astype(np.float32, copy=False)
    return (wave / max(rms, eps)).astype(np.float32, copy=False)


def _log_spectrum(
    wave: np.ndarray,
    *,
    sample_rate: int,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    wave = _resample_if_needed(wave, source_rate=sample_rate, target_rate=target_rate)
    wave = _rms_normalize(wave)
    tensor = torch.from_numpy(wave)
    if tensor.numel() < n_fft:
        tensor = torch.nn.functional.pad(tensor, (0, n_fft - tensor.numel()))
    window = torch.hann_window(n_fft, dtype=tensor.dtype)
    spectrum = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = torch.abs(spectrum).transpose(0, 1)
    if magnitude.shape[1] < feature_bins:
        magnitude = torch.nn.functional.pad(magnitude, (0, feature_bins - magnitude.shape[1]))
    features = torch.log1p(magnitude[:, :feature_bins]).numpy().astype(np.float32, copy=False)
    frame_centers = np.arange(features.shape[0], dtype=np.float32) * (hop_length / target_rate)
    return features, frame_centers


def _materialized_path(row: Mapping[str, object], key: str, *, manifest_dir: Path) -> Path | None:
    materialized = dict(row.get("materialized") or {})
    value = materialized.get(key)
    if not value:
        return None
    return _resolve_path(value, manifest_dir=manifest_dir)


def _positive_enrollment_paths(
    row: Mapping[str, object],
    *,
    manifest_dir: Path,
) -> list[Path]:
    materialized = dict(row.get("materialized") or {})
    return [
        _resolve_path(path, manifest_dir=manifest_dir)
        for path in materialized.get("positive_enrollment_paths") or []
    ]


def _speaker_profile_from_chunks(
    chunks: Sequence[tuple[np.ndarray, int]],
    *,
    max_seconds: float,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
) -> np.ndarray:
    feature_chunks = []
    remaining = max_seconds
    for wave, sample_rate in chunks:
        if remaining > 0:
            limit = int(round(remaining * sample_rate))
            wave = wave[:limit]
            remaining -= wave.shape[0] / sample_rate
        features, _ = _log_spectrum(
            wave,
            sample_rate=sample_rate,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
        )
        feature_chunks.append(features)
        if max_seconds > 0 and remaining <= 0:
            break
    if not feature_chunks:
        raise ValueError("No positive enrollment paths available")
    merged = np.concatenate(feature_chunks, axis=0)
    return np.concatenate(
        [
            merged.mean(axis=0),
            merged.std(axis=0) + 1e-4,
        ]
    ).astype(np.float32, copy=False)


def _speaker_profile_for_row(
    row: Mapping[str, object],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    max_seconds: float,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
) -> np.ndarray:
    paths = _positive_enrollment_paths(row, manifest_dir=manifest_dir)
    if paths:
        chunks = [_load_mono(path) for path in paths]
    else:
        if stem_cache is None or not row.get("target_member"):
            raise ValueError(f"No positive enrollment audio available for {row.get('row_id')}")
        member_path = stem_cache.path_for_member(row, row.get("target_member"))
        chunks = []
        for span in row.get("positive_enrollment_spans") or []:
            start = float(dict(span).get("start") or 0.0)
            end = float(dict(span).get("end") or start)
            if end <= start:
                continue
            chunks.append(
                _load_audio_window(
                    member_path,
                    start_seconds=start,
                    duration_seconds=end - start,
                )
            )
    return _speaker_profile_from_chunks(
        chunks,
        max_seconds=max_seconds,
        target_rate=target_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        feature_bins=feature_bins,
    )


def _conditioned_features(mixture_features: np.ndarray, profile: np.ndarray) -> np.ndarray:
    bins = mixture_features.shape[1]
    mean = profile[:bins][None, :]
    std = profile[bins : 2 * bins][None, :]
    return np.concatenate(
        [
            mixture_features,
            np.broadcast_to(mean, mixture_features.shape),
            np.broadcast_to(std, mixture_features.shape),
            np.abs(mixture_features - mean),
            mixture_features * mean,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _frame_labels(
    words: Sequence[Mapping[str, object]],
    speaker: str,
    frame_centers: np.ndarray,
    *,
    min_label_seconds: float = 0.04,
) -> np.ndarray:
    labels = np.zeros(frame_centers.shape[0], dtype=bool)
    for word in words:
        if str(word.get("speaker") or "") != speaker:
            continue
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        if end < start:
            start, end = end, start
        if end - start < min_label_seconds:
            midpoint = (start + end) / 2.0
            start = midpoint - min_label_seconds / 2.0
            end = midpoint + min_label_seconds / 2.0
        labels |= (frame_centers >= start) & (frame_centers < end)
    return labels


def _word_has_overlap(
    word: Mapping[str, object],
    words: Sequence[Mapping[str, object]],
    *,
    min_overlap_seconds: float = 0.02,
) -> bool:
    speaker = str(word.get("speaker") or "")
    start = float(word.get("start") or 0.0)
    end = float(word.get("end") or start)
    for other in words:
        if other is word or str(other.get("speaker") or "") == speaker:
            continue
        other_start = float(other.get("start") or 0.0)
        other_end = float(other.get("end") or other_start)
        if min(end, other_end) - max(start, other_start) >= min_overlap_seconds:
            return True
    return False


def _interval_indices(frame_centers: np.ndarray, start: float, end: float) -> np.ndarray:
    if end < start:
        start, end = end, start
    indices = np.flatnonzero((frame_centers >= start) & (frame_centers < end))
    if indices.size:
        return indices
    midpoint = (start + end) / 2.0
    nearest = int(np.argmin(np.abs(frame_centers - midpoint)))
    return np.asarray([nearest], dtype=np.int64)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _score_word_owners(
    *,
    words: Sequence[Mapping[str, object]],
    speakers: Sequence[str],
    frame_centers: np.ndarray,
    speaker_posteriors: Mapping[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    reference_words = 0
    scored_words = 0
    correct_words = 0
    skipped_missing_speaker_words = 0
    overlap_words = 0
    overlap_correct_words = 0
    non_overlap_words = 0
    non_overlap_correct_words = 0
    reference_margins = []
    top_margins = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    word_records = []

    speaker_set = set(speakers)
    for index, word in enumerate(words):
        reference = str(word.get("speaker") or "")
        if not reference:
            continue
        reference_words += 1
        if reference not in speaker_set:
            skipped_missing_speaker_words += 1
            continue
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        indices = _interval_indices(frame_centers, start, end)
        scores = {
            speaker: float(np.mean(np.asarray(speaker_posteriors[speaker])[indices]))
            for speaker in speakers
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predicted = ordered[0][0]
        top_margin = ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0)
        best_non_reference = max(
            (score for speaker, score in scores.items() if speaker != reference),
            default=0.0,
        )
        reference_margin = scores.get(reference, 0.0) - best_non_reference
        has_overlap = _word_has_overlap(word, words)
        is_correct = predicted == reference

        scored_words += 1
        correct_words += int(is_correct)
        top_margins.append(float(top_margin))
        reference_margins.append(float(reference_margin))
        confusion[reference][predicted] += 1
        by_speaker[reference]["words"] += 1
        by_speaker[reference]["correct"] += int(is_correct)
        if has_overlap:
            overlap_words += 1
            overlap_correct_words += int(is_correct)
        else:
            non_overlap_words += 1
            non_overlap_correct_words += int(is_correct)

        word_records.append(
            {
                "word_index": index,
                "text": word.get("text"),
                "speaker": reference,
                "predicted": predicted,
                "correct": is_correct,
                "start": start,
                "end": end,
                "overlap": has_overlap,
                "reference_score": scores.get(reference),
                "predicted_score": scores[predicted],
                "reference_margin": reference_margin,
                "top_margin": top_margin,
            }
        )

    result = {
        "speakers": list(speakers),
        "reference_words": reference_words,
        "scored_words": scored_words,
        "skipped_missing_speaker_words": skipped_missing_speaker_words,
        "correct_words": correct_words,
        "coverage": scored_words / reference_words if reference_words else 0.0,
        "accuracy": correct_words / reference_words if reference_words else 0.0,
        "scored_accuracy": correct_words / scored_words if scored_words else 0.0,
        "overlap_words": overlap_words,
        "overlap_correct_words": overlap_correct_words,
        "overlap_accuracy": overlap_correct_words / overlap_words if overlap_words else None,
        "non_overlap_words": non_overlap_words,
        "non_overlap_correct_words": non_overlap_correct_words,
        "non_overlap_accuracy": (
            non_overlap_correct_words / non_overlap_words if non_overlap_words else None
        ),
        "mean_reference_margin": _mean(reference_margins),
        "mean_top_margin": _mean(top_margins),
        "confusion": {speaker: dict(counts) for speaker, counts in sorted(confusion.items())},
        "by_speaker": {
            speaker: {
                "words": int(counts["words"]),
                "correct": int(counts["correct"]),
                "accuracy": counts["correct"] / counts["words"] if counts["words"] else 0.0,
            }
            for speaker, counts in sorted(by_speaker.items())
        },
    }
    return result, word_records


class _ActivityMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_indices(
    labels: np.ndarray,
    *,
    budget: int,
    negative_ratio: float,
    rng: random.Random,
) -> np.ndarray:
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if not positive.size:
        take = min(budget, negative.size)
        return np.asarray(rng.sample(list(negative), take), dtype=np.int64)
    positive_budget = max(1, int(round(budget / (1.0 + negative_ratio))))
    negative_budget = max(1, budget - positive_budget)
    positive_take = min(positive_budget, positive.size)
    negative_take = min(negative_budget, negative.size)
    chosen = []
    if positive_take:
        chosen.extend(rng.sample(list(positive), positive_take))
    if negative_take:
        chosen.extend(rng.sample(list(negative), negative_take))
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def _load_group_mixture_features(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = _materialized_path(rows[0], "mixture_path", manifest_dir=manifest_dir)
    if path is not None and path.exists():
        wave, sample_rate = _load_mono(path)
    else:
        if stem_cache is None:
            raise ValueError("Group has no materialized mixture path and no stem cache")
        row = rows[0]
        member_waves = []
        sample_rates = []
        for member in row.get("mixture_members") or []:
            member_path = stem_cache.path_for_member(row, member)
            wave_part, sample_rate_part = _load_audio_window(
                member_path,
                start_seconds=float(row.get("window_start") or 0.0),
                duration_seconds=float(row.get("duration") or 0.0),
            )
            member_waves.append(wave_part)
            sample_rates.append(sample_rate_part)
        if not member_waves:
            raise ValueError(f"Group {row.get('row_id')} has no mixture members")
        if len(set(sample_rates)) != 1:
            target_len = max(wave_part.shape[0] for wave_part in member_waves)
            member_waves = [
                _resample_if_needed(
                    wave_part,
                    source_rate=sample_rate_part,
                    target_rate=target_rate,
                )
                for wave_part, sample_rate_part in zip(member_waves, sample_rates)
            ]
            sample_rate = target_rate
            target_len = max(wave_part.shape[0] for wave_part in member_waves)
        else:
            sample_rate = sample_rates[0]
            target_len = max(wave_part.shape[0] for wave_part in member_waves)
        padded = [
            np.pad(wave_part, (0, max(0, target_len - wave_part.shape[0])))[:target_len]
            for wave_part in member_waves
        ]
        wave = np.sum(np.stack(padded, axis=0), axis=0).astype(np.float32, copy=False)
    return _log_spectrum(
        wave,
        sample_rate=sample_rate,
        target_rate=target_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        feature_bins=feature_bins,
    )


def _reference_words_for_training(
    key: tuple[str, float, float],
    rows: Sequence[Mapping[str, object]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    source: str,
) -> tuple[list[dict], str]:
    if source in {"auto", "forced"} and key in reference_groups:
        return [dict(word) for word in reference_groups[key]], "forced"
    if source in {"auto", "manifest"}:
        words = []
        for span in rows[0].get("word_spans") or []:
            item = dict(span)
            if item.get("speaker") and item.get("end") is not None:
                words.append(item)
        return words, "manifest_span"
    return [], "missing"


def _row_has_audio(row: Mapping[str, object], *, include_nonmaterialized: bool) -> bool:
    return bool(row.get("materialized")) or bool(include_nonmaterialized and row.get("source_zip"))


def _build_training_examples(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    include_nonmaterialized: bool,
    train_reference_source: str,
    train_splits: set[str],
    train_sessions: set[str],
    max_train_examples: int,
    negative_ratio: float,
    max_enrollment_seconds: float,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_rows = [
        row
        for key, rows in grouped_rows.items()
        for row in rows
        if str(row.get("split_id") or "") in train_splits
        and (not train_sessions or str(row.get("session") or "") in train_sessions)
        and _row_has_audio(row, include_nonmaterialized=include_nonmaterialized)
    ]
    if not train_rows:
        raise ValueError(f"No materialized training rows found for splits {sorted(train_splits)}")

    rows_by_group: dict[tuple[str, float, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in train_rows:
        rows_by_group[_group_key(row)].append(row)

    rng = random.Random(seed)
    per_row_budget = max(32, math.ceil(max_train_examples / max(1, len(train_rows))))
    feature_chunks = []
    label_chunks = []
    row_summaries = []

    for key, rows in sorted(rows_by_group.items()):
        words, reference_source = _reference_words_for_training(
            key,
            rows,
            reference_groups,
            source=train_reference_source,
        )
        if not words:
            continue
        mixture_features, frame_centers = _load_group_mixture_features(
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
        )
        for row in sorted(rows, key=lambda item: str(item.get("speaker_id") or "")):
            speaker = str(row.get("speaker_id") or "")
            labels = _frame_labels(words, speaker, frame_centers)
            sample_indices = _sample_indices(
                labels,
                budget=per_row_budget,
                negative_ratio=negative_ratio,
                rng=rng,
            )
            if not sample_indices.size:
                continue
            profile = _speaker_profile_for_row(
                row,
                manifest_dir=manifest_dir,
                stem_cache=stem_cache,
                max_seconds=max_enrollment_seconds,
                target_rate=target_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                feature_bins=feature_bins,
            )
            conditioned = _conditioned_features(mixture_features[sample_indices], profile)
            feature_chunks.append(conditioned)
            label_chunks.append(labels[sample_indices].astype(np.float32))
            row_summaries.append(
                {
                    "row_id": row.get("row_id"),
                    "session": row.get("session"),
                    "speaker": speaker,
                    "examples": int(sample_indices.size),
                    "positive_examples": int(labels[sample_indices].sum()),
                    "positive_frames": int(labels.sum()),
                    "total_frames": int(labels.shape[0]),
                    "reference_source": reference_source,
                }
            )

    if not feature_chunks:
        raise ValueError("Training example construction produced no examples")

    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    if features.shape[0] > max_train_examples:
        positive = np.flatnonzero(labels > 0.5)
        negative = np.flatnonzero(labels <= 0.5)
        positive_take = min(positive.size, max_train_examples // 2)
        negative_take = min(negative.size, max_train_examples - positive_take)
        chosen = []
        if positive_take:
            chosen.extend(rng.sample(list(positive), positive_take))
        if negative_take:
            chosen.extend(rng.sample(list(negative), negative_take))
        rng.shuffle(chosen)
        features = features[chosen]
        labels = labels[chosen]

    summary = {
        "train_splits": sorted(train_splits),
        "train_sessions": sorted(train_sessions),
        "train_rows": len(train_rows),
        "sampled_examples": int(features.shape[0]),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels <= 0.5).sum()),
        "train_reference_source": train_reference_source,
        "row_summaries": row_summaries,
    }
    return features, labels, summary


def _standardize_train(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    return ((features - mean) / std).astype(np.float32, copy=False), mean, std


def _fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[_ActivityMlp, np.ndarray, np.ndarray, list[dict]]:
    torch.manual_seed(seed)
    standardized, mean, std = _standardize_train(features)
    dataset = TensorDataset(
        torch.from_numpy(standardized),
        torch.from_numpy(labels.astype(np.float32, copy=False)),
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    model = _ActivityMlp(features.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positives = float(labels.sum())
    negatives = float(labels.shape[0] - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_labels.numel()
            total += int(batch_labels.numel())
            predictions = torch.sigmoid(logits) >= 0.5
            correct += int((predictions == (batch_labels >= 0.5)).sum().item())
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total if total else 0.0,
                "training_accuracy": correct / total if total else 0.0,
            }
        )
    model.eval()
    return model, mean, std, history


def _predict_posteriors(
    model: _ActivityMlp,
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    posteriors = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = ((features[start : start + batch_size] - mean) / std).astype(
                np.float32,
                copy=False,
            )
            logits = model(torch.from_numpy(batch).to(device))
            posteriors.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(posteriors).astype(np.float32, copy=False)


def _sum_confusion(groups: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for reference, counts in dict(group.get("confusion") or {}).items():
            for predicted, count in dict(counts).items():
                confusion[str(reference)][str(predicted)] += int(count)
    return {speaker: dict(counts) for speaker, counts in sorted(confusion.items())}


def _merge_by_speaker(groups: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    merged: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        for speaker, counts in dict(group.get("by_speaker") or {}).items():
            merged[str(speaker)]["words"] += int(dict(counts).get("words") or 0)
            merged[str(speaker)]["correct"] += int(dict(counts).get("correct") or 0)
    return {
        speaker: {
            "words": int(counts["words"]),
            "correct": int(counts["correct"]),
            "accuracy": counts["correct"] / counts["words"] if counts["words"] else 0.0,
        }
        for speaker, counts in sorted(merged.items())
    }


def _summarize_groups(groups: Sequence[Mapping[str, object]]) -> dict:
    valid = [group for group in groups if not group.get("error")]
    reference_words = sum(int(group.get("reference_words") or 0) for group in valid)
    scored_words = sum(int(group.get("scored_words") or 0) for group in valid)
    correct_words = sum(int(group.get("correct_words") or 0) for group in valid)
    overlap_words = sum(int(group.get("overlap_words") or 0) for group in valid)
    overlap_correct_words = sum(int(group.get("overlap_correct_words") or 0) for group in valid)
    non_overlap_words = sum(int(group.get("non_overlap_words") or 0) for group in valid)
    non_overlap_correct_words = sum(
        int(group.get("non_overlap_correct_words") or 0) for group in valid
    )
    reference_margins = [
        float(group["mean_reference_margin"])
        for group in valid
        if group.get("mean_reference_margin") is not None
    ]
    top_margins = [
        float(group["mean_top_margin"])
        for group in valid
        if group.get("mean_top_margin") is not None
    ]
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_session: dict[str, Counter[str]] = defaultdict(Counter)
    for group in valid:
        for bucket, key in (
            (by_split, str(group.get("split_id") or "")),
            (by_session, str(group.get("session") or "")),
        ):
            bucket[key]["reference_words"] += int(group.get("reference_words") or 0)
            bucket[key]["scored_words"] += int(group.get("scored_words") or 0)
            bucket[key]["correct_words"] += int(group.get("correct_words") or 0)
    return {
        "group_count": len(groups),
        "valid_groups": len(valid),
        "reference_words": reference_words,
        "scored_words": scored_words,
        "correct_words": correct_words,
        "coverage": scored_words / reference_words if reference_words else 0.0,
        "accuracy": correct_words / reference_words if reference_words else 0.0,
        "scored_accuracy": correct_words / scored_words if scored_words else 0.0,
        "overlap_words": overlap_words,
        "overlap_correct_words": overlap_correct_words,
        "overlap_accuracy": (overlap_correct_words / overlap_words if overlap_words else None),
        "non_overlap_words": non_overlap_words,
        "non_overlap_correct_words": non_overlap_correct_words,
        "non_overlap_accuracy": (
            non_overlap_correct_words / non_overlap_words if non_overlap_words else None
        ),
        "mean_reference_margin": _mean(reference_margins),
        "mean_top_margin": _mean(top_margins),
        "confusion": _sum_confusion(valid),
        "by_speaker": _merge_by_speaker(valid),
        "by_split": {
            key: {
                "reference_words": int(counts["reference_words"]),
                "scored_words": int(counts["scored_words"]),
                "correct_words": int(counts["correct_words"]),
                "accuracy": (
                    counts["correct_words"] / counts["reference_words"]
                    if counts["reference_words"]
                    else 0.0
                ),
                "scored_accuracy": (
                    counts["correct_words"] / counts["scored_words"]
                    if counts["scored_words"]
                    else 0.0
                ),
            }
            for key, counts in sorted(by_split.items())
        },
        "by_session": {
            key: {
                "reference_words": int(counts["reference_words"]),
                "scored_words": int(counts["scored_words"]),
                "correct_words": int(counts["correct_words"]),
                "accuracy": (
                    counts["correct_words"] / counts["reference_words"]
                    if counts["reference_words"]
                    else 0.0
                ),
                "scored_accuracy": (
                    counts["correct_words"] / counts["scored_words"]
                    if counts["scored_words"]
                    else 0.0
                ),
            }
            for key, counts in sorted(by_session.items())
        },
    }


def _evaluate(
    grouped_rows: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    reference_groups: Mapping[tuple[str, float, float], Sequence[Mapping[str, object]]],
    *,
    model: _ActivityMlp,
    mean: np.ndarray,
    std: np.ndarray,
    manifest_dir: Path,
    stem_cache: StemCache | None,
    include_nonmaterialized: bool,
    eval_splits: set[str],
    eval_sessions: set[str],
    max_enrollment_seconds: float,
    target_rate: int,
    n_fft: int,
    hop_length: int,
    feature_bins: int,
    batch_size: int,
    device: torch.device,
    save_word_records: bool,
) -> tuple[list[dict], list[dict]]:
    group_results = []
    word_records = []
    for key, all_rows in sorted(grouped_rows.items()):
        rows = [
            row
            for row in all_rows
            if str(row.get("split_id") or "") in eval_splits
            and (not eval_sessions or str(row.get("session") or "") in eval_sessions)
            and _row_has_audio(row, include_nonmaterialized=include_nonmaterialized)
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
        split_ids = sorted({str(row.get("split_id") or "") for row in rows})
        mixture_features, frame_centers = _load_group_mixture_features(
            rows,
            manifest_dir=manifest_dir,
            stem_cache=stem_cache,
            target_rate=target_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_bins=feature_bins,
        )
        speaker_posteriors = {}
        for row in sorted(rows, key=lambda item: str(item.get("speaker_id") or "")):
            speaker = str(row.get("speaker_id") or "")
            profile = _speaker_profile_for_row(
                row,
                manifest_dir=manifest_dir,
                stem_cache=stem_cache,
                max_seconds=max_enrollment_seconds,
                target_rate=target_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                feature_bins=feature_bins,
            )
            conditioned = _conditioned_features(mixture_features, profile)
            speaker_posteriors[speaker] = _predict_posteriors(
                model,
                conditioned,
                mean=mean,
                std=std,
                batch_size=batch_size,
                device=device,
            )

        result, group_word_records = _score_word_owners(
            words=reference_groups[key],
            speakers=sorted(speaker_posteriors),
            frame_centers=frame_centers,
            speaker_posteriors=speaker_posteriors,
        )
        result.update(
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "split_id": ",".join(split_ids),
                "row_count": len(rows),
            }
        )
        group_results.append(result)
        if save_word_records:
            for record in group_word_records:
                record.update(
                    {
                        "session": key[0],
                        "window_start": key[1],
                        "window_end": key[2],
                        "split_id": ",".join(split_ids),
                    }
                )
                word_records.append(record)
    return group_results, word_records


def _group_manifest_rows(rows: Iterable[dict]) -> dict[tuple[str, float, float], list[dict]]:
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight TS-VAD-style target-speaker activity baseline and score "
            "forced-word ownership from speaker-conditioned posteriors."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-splits", default="dev")
    parser.add_argument("--eval-splits", default="test")
    parser.add_argument("--train-sessions", help="Comma-separated session names to train on.")
    parser.add_argument("--eval-sessions", help="Comma-separated session names to evaluate on.")
    parser.add_argument(
        "--train-reference-source",
        choices=("auto", "forced", "manifest"),
        default="auto",
        help="Use forced word refs when available, or manifest spans for non-forced train groups.",
    )
    parser.add_argument(
        "--include-nonmaterialized",
        action="store_true",
        help="Read non-materialized rows directly from cached/extracted session stems.",
    )
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-seconds", type=float, default=0.02)
    parser.add_argument("--feature-bins", type=int, default=96)
    parser.add_argument("--max-enrollment-seconds", type=float, default=45.0)
    parser.add_argument("--max-train-examples", type=int, default=120_000)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    manifest_dir = args.manifest.resolve().parent
    hop_length = int(round(float(args.hop_seconds) * int(args.sample_rate)))
    if hop_length <= 0:
        raise ValueError("--hop-seconds must produce a positive hop length")
    if args.feature_bins > args.n_fft // 2 + 1:
        raise ValueError("--feature-bins cannot exceed n_fft / 2 + 1")

    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped_rows = _group_manifest_rows(_read_jsonl(args.manifest))
    train_splits = _split_values(args.train_splits)
    eval_splits = _split_values(args.eval_splits)
    train_sessions = _split_values(args.train_sessions or "")
    eval_sessions = _split_values(args.eval_sessions or "")
    device = _select_device(args.device)
    stem_cache = (
        StemCache(
            root=args.stems_cache_root or args.output_dir / "_stems",
        )
        if args.include_nonmaterialized
        else None
    )

    features, labels, training_data_summary = _build_training_examples(
        grouped_rows,
        reference_groups,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        include_nonmaterialized=bool(args.include_nonmaterialized),
        train_reference_source=str(args.train_reference_source),
        train_splits=train_splits,
        train_sessions=train_sessions,
        max_train_examples=int(args.max_train_examples),
        negative_ratio=float(args.negative_ratio),
        max_enrollment_seconds=float(args.max_enrollment_seconds),
        target_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=hop_length,
        feature_bins=int(args.feature_bins),
        seed=int(args.seed),
    )
    model, mean, std, history = _fit_model(
        features,
        labels,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=device,
        seed=int(args.seed),
    )
    group_results, word_records = _evaluate(
        grouped_rows,
        reference_groups,
        model=model,
        mean=mean,
        std=std,
        manifest_dir=manifest_dir,
        stem_cache=stem_cache,
        include_nonmaterialized=bool(args.include_nonmaterialized),
        eval_splits=eval_splits,
        eval_sessions=eval_sessions,
        max_enrollment_seconds=float(args.max_enrollment_seconds),
        target_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=hop_length,
        feature_bins=int(args.feature_bins),
        batch_size=int(args.batch_size),
        device=device,
        save_word_records=bool(args.save_word_records),
    )
    summary = _summarize_groups(group_results)
    summary.update(
        {
            "model": "speaker_conditioned_frame_activity_mlp",
            "manifest": str(args.manifest),
            "reference_jsonl": str(args.reference_jsonl),
            "train_splits": sorted(train_splits),
            "eval_splits": sorted(eval_splits),
            "train_sessions": sorted(train_sessions),
            "eval_sessions": sorted(eval_sessions),
            "train_reference_source": str(args.train_reference_source),
            "include_nonmaterialized": bool(args.include_nonmaterialized),
            "stems_cache_root": (
                str(args.stems_cache_root or args.output_dir / "_stems")
                if args.include_nonmaterialized
                else None
            ),
            "sample_rate": int(args.sample_rate),
            "n_fft": int(args.n_fft),
            "hop_seconds": float(args.hop_seconds),
            "feature_bins": int(args.feature_bins),
            "max_enrollment_seconds": float(args.max_enrollment_seconds),
            "max_train_examples": int(args.max_train_examples),
            "negative_ratio": float(args.negative_ratio),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "device": str(device),
            "seed": int(args.seed),
            "elapsed_seconds": time.time() - start_time,
        }
    )
    training_summary = {
        **training_data_summary,
        "history": history,
        "feature_dim": int(features.shape[1]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "tsvad_word_owner_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "tsvad_word_owner_words.jsonl", word_records)
    (args.output_dir / "tsvad_word_owner_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "tsvad_training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
