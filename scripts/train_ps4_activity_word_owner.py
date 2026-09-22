from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_sequence_tsvad_word_owner_baseline import _summarize_groups  # noqa: E402
from run_usef_tse_manifest import (  # noqa: E402
    DEFAULT_LAURA_CHECKPOINT,
    DEFAULT_LAURA_HF_REPO,
    _load_usef_laura_front,
)
from train_tsvad_word_owner_baseline import (  # noqa: E402
    _score_word_owners,
    _word_has_overlap,
    _write_jsonl,
)
from train_usef_tse_domain_adapter import (  # noqa: E402
    ForcedReferenceIndex,
    StemCache,
    _forced_reference_coverage,
    _load_chunk,
    _load_rows,
    _parse_csv_set,
    _rms,
    _validate_forced_reference_coverage,
    _word_masks,
)

DEFAULT_PS4_REPO = "TaurenMountain/PS4"
DEFAULT_PS4_CHECKPOINT = "checkpoint_epoch037.pt"
PS4_MODEL_ARGS = {
    "feat_type": "consistent",
    "feature_dim": 128,
    "joint_training": True,
    "multi_fuse": False,
    "multi_task": False,
    "num_repeat": 6,
    "spk_args": {"embed_dim": 192, "feat_dim": 80, "pooling_func": "ASTP"},
    "spk_emb_dim": 192,
    "spk_feat": False,
    "spk_fuse_type": "multiply",
    "spk_model": "ECAPA_TDNN_GLOB_c512",
    "spk_model_freeze": True,
    "spk_model_init": None,
    "spksInTrain": 251,
    "sr": 16000,
    "stride": 128,
    "use_spk_transform": False,
    "win": 512,
}


def _group_key(row: Mapping[str, object]) -> tuple[str, float, float]:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _group_rows(rows: Sequence[Mapping[str, object]]) -> list[list[dict]]:
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(dict(row))
    return [
        sorted(group, key=lambda item: str(item.get("speaker_id") or ""))
        for _, group in sorted(grouped.items())
    ]


def _register_source_package(name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


def _ps4_model_class(*, wesep_repo: Path, wespeaker_repo: Path) -> type[nn.Module]:
    """Import only the ECAPA and BSRNN modules needed by PS4.

    Both upstream package initializers import optional diarization stacks. Registering namespace
    packages here keeps inference pinned to the audited model code without requiring those unrelated
    dependencies.
    """

    _register_source_package("wespeaker", wespeaker_repo / "wespeaker")
    _register_source_package("wespeaker.models", wespeaker_repo / "wespeaker" / "models")
    ecapa = importlib.import_module("wespeaker.models.ecapa_tdnn")
    speaker_model = types.ModuleType("wespeaker.models.speaker_model")

    def get_speaker_model(name: str) -> type[nn.Module]:
        return getattr(ecapa, name)

    speaker_model.get_speaker_model = get_speaker_model
    sys.modules["wespeaker.models.speaker_model"] = speaker_model
    _register_source_package("wesep", wesep_repo / "wesep")
    _register_source_package("wesep.models", wesep_repo / "wesep" / "models")
    return importlib.import_module("wesep.models.bsrnn").BSRNN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(
    *,
    checkpoint_path: Path | None,
    hf_repo: str,
    checkpoint_file: str,
) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path.resolve()
    return Path(hf_hub_download(hf_repo, checkpoint_file)).resolve()


def load_ps4_model(
    *,
    wesep_repo: Path,
    wespeaker_repo: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    model_class = _ps4_model_class(wesep_repo=wesep_repo, wespeaker_repo=wespeaker_repo)
    model = model_class(**PS4_MODEL_ARGS)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class PS4FeatureExtractor(nn.Module):
    """Run PS4 through its enrolled-speaker separator and return its hidden feature map."""

    def __init__(self, model: nn.Module, *, trainable_separator_blocks: int = 0) -> None:
        super().__init__()
        self.model = model
        self.frame_hop_seconds = PS4_MODEL_ARGS["stride"] / PS4_MODEL_ARGS["sr"]
        blocks = list(model.separator.separation[1:])
        if not 0 <= trainable_separator_blocks <= len(blocks):
            raise ValueError(f"trainable_separator_blocks must be between 0 and {len(blocks)}")
        self.trainable_separator_blocks = int(trainable_separator_blocks)
        self._separator_blocks = blocks
        if self.trainable_separator_blocks:
            for block in blocks[-self.trainable_separator_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def set_trainable_mode(self, training: bool) -> None:
        self.model.eval()
        if self.trainable_separator_blocks:
            for block in self._separator_blocks[-self.trainable_separator_blocks :]:
                block.train(training)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if name in trainable_names
        }

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        current = self.model.state_dict()
        unexpected = sorted(set(state) - set(current))
        if unexpected:
            raise ValueError(f"Unexpected PS4 adapter tensors: {unexpected[:5]}")
        current.update(state)
        self.model.load_state_dict(current, strict=True)

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        model = self.model
        # The backbone is frozen, but the returned tensor must remain usable by the trainable head.
        with torch.no_grad():
            spectrum = torch.stft(
                mixture,
                n_fft=model.win,
                hop_length=model.stride,
                window=torch.hann_window(
                    model.win,
                    device=mixture.device,
                    dtype=mixture.dtype,
                ),
                return_complex=True,
            )
            spectrum_ri = torch.stack([spectrum.real, spectrum.imag], dim=1)
            subband_features = []
            band_index = 0
            for band_width, normalizer in zip(model.band_width, model.BN):
                band = spectrum_ri[:, :, band_index : band_index + band_width].contiguous()
                subband_features.append(normalizer(band.view(mixture.shape[0], band_width * 2, -1)))
                band_index += band_width
            mixture_features = torch.stack(subband_features, dim=1)

            speaker_features = model.preEmphasis(enrollment)
            speaker_features = model.spk_encoder(speaker_features).add(1e-8).log()
            speaker_features = speaker_features - speaker_features.mean(dim=-1, keepdim=True)
            speaker_features = speaker_features.permute(0, 2, 1)
            speaker_embedding = model.spk_model(speaker_features)
            if isinstance(speaker_embedding, tuple):
                speaker_embedding = speaker_embedding[-1]
            speaker_embedding = model.spk_transform(speaker_embedding)
            speaker_embedding = speaker_embedding.unsqueeze(1).unsqueeze(3)
            fused = model.separator.separation[0](mixture_features, speaker_embedding)
            separated = fused.view(mixture.shape[0], model.nband * model.feature_dim, -1)

        blocks = list(model.separator.separation[1:])
        frozen_blocks = len(blocks) - self.trainable_separator_blocks
        with torch.no_grad():
            for block in blocks[:frozen_blocks]:
                separated = block(separated, speaker_embedding)
        separated = separated.detach()
        for block in blocks[frozen_blocks:]:
            separated = block(separated, speaker_embedding)
        return separated.view(mixture.shape[0], model.nband, model.feature_dim, -1)


class LauraFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, *, trainable_separator_blocks: int = 0) -> None:
        super().__init__()
        self.model = model
        self.frame_hop_seconds = 0.01
        blocks = list(model.dual_mdl)
        if not 0 <= trainable_separator_blocks <= len(blocks):
            raise ValueError(f"trainable_separator_blocks must be between 0 and {len(blocks)}")
        self.trainable_separator_blocks = int(trainable_separator_blocks)
        self._separator_blocks = blocks
        if self.trainable_separator_blocks:
            for block in blocks[-self.trainable_separator_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if name in trainable_names
        }

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        current = self.model.state_dict()
        unexpected = sorted(set(state) - set(current))
        if unexpected:
            raise ValueError(f"Unexpected Laura adapter tensors: {unexpected[:5]}")
        current.update(state)
        self.model.load_state_dict(current, strict=True)

    def set_trainable_mode(self, training: bool) -> None:
        self.model.eval()
        if self.trainable_separator_blocks:
            for block in self._separator_blocks[-self.trainable_separator_blocks :]:
                block.train(training)

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        model = self.model
        with torch.no_grad():
            mixture = mixture.unsqueeze(1)
            enrollment = enrollment.unsqueeze(1)
            mixture = mixture / mixture.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            enrollment = enrollment / enrollment.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            mixture_complex = model.stft(mixture)[-1]
            enrollment_complex = model.stft(enrollment)[-1]
            mixture_ri = torch.cat([mixture_complex.real, mixture_complex.imag], dim=1)
            enrollment_ri = torch.cat([enrollment_complex.real, enrollment_complex.imag], dim=1)
            mixture_ri = mixture_ri.permute(0, 1, 3, 2).contiguous().clamp(-1e4, 1e4)
            enrollment_ri = enrollment_ri.permute(0, 1, 3, 2).contiguous().clamp(-1e4, 1e4)
            mixture_features = model.conv(mixture_ri)
            enrollment_features = model.conv(enrollment_ri)
            attended = model.att(mixture_features, enrollment_features)
            separated = torch.cat([mixture_features, attended], dim=1)

        frozen_blocks = len(self._separator_blocks) - self.trainable_separator_blocks
        with torch.no_grad():
            for block in self._separator_blocks[:frozen_blocks]:
                separated = block(separated)
        separated = separated.detach()
        for block in self._separator_blocks[frozen_blocks:]:
            separated = block(separated)
        return separated


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(features)
        hidden = self.pointwise(F.gelu(hidden))
        hidden = self.dropout(F.gelu(self.norm(hidden)))
        return features + hidden


class PS4ActivityHead(nn.Module):
    def __init__(
        self,
        *,
        bands: int = 32,
        feature_dim: int = 128,
        channels: int = 256,
        layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = bands * feature_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _TemporalBlock(channels, dilation=2 ** (index % 6), dropout=dropout)
                for index in range(layers)
            ]
        )
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, bands, feature_dim, frames = features.shape
        flattened = features.reshape(batch, bands * feature_dim, frames).transpose(1, 2)
        hidden = self.input_norm(flattened).transpose(1, 2)
        hidden = self.input_projection(hidden)
        hidden = self.blocks(hidden)
        return self.output(hidden).squeeze(1)


class LauraActivityHead(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 256,
        channels: int = 256,
        layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = feature_dim * 3
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _TemporalBlock(channels, dilation=2 ** (index % 6), dropout=dropout)
                for index in range(layers)
            ]
        )
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=-1)
        std = features.std(dim=-1)
        maximum = features.amax(dim=-1)
        flattened = torch.cat([mean, std, maximum], dim=1).transpose(1, 2)
        hidden = self.input_norm(flattened).transpose(1, 2)
        hidden = self.input_projection(hidden)
        hidden = self.blocks(hidden)
        return self.output(hidden).squeeze(1)


def _sample_group_chunk_start(
    words: Sequence[Mapping[str, object]],
    *,
    duration: float,
    chunk_seconds: float,
    overlap_probability: float,
    active_probability: float,
    rng: random.Random,
) -> float:
    max_start = max(0.0, duration - chunk_seconds)
    overlaps = []
    for index, word in enumerate(words):
        speaker = str(word.get("speaker") or "")
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        for other in words[index + 1 :]:
            if str(other.get("speaker") or "") == speaker:
                continue
            overlap_start = max(start, float(other.get("start") or 0.0))
            overlap_end = min(end, float(other.get("end") or 0.0))
            if overlap_end > overlap_start:
                overlaps.append((overlap_start, overlap_end))
    if overlaps and rng.random() < overlap_probability:
        start, end = rng.choice(overlaps)
        center = (start + end) / 2.0 + rng.uniform(-0.15, 0.15) * chunk_seconds
        return min(max(center - chunk_seconds / 2.0, 0.0), max_start)
    active_words = [word for word in words if float(word.get("end") or 0.0) > 0.0]
    if active_words and rng.random() < active_probability:
        word = rng.choice(active_words)
        center = (float(word.get("start") or 0.0) + float(word.get("end") or 0.0)) / 2.0
        center += rng.uniform(-0.3, 0.3) * chunk_seconds
        return min(max(center - chunk_seconds / 2.0, 0.0), max_start)
    return rng.uniform(0.0, max_start) if max_start else 0.0


class EnrollmentSelector:
    def __init__(
        self,
        *,
        seconds: float,
        sample_rate: int,
        min_rms: float,
        scan_hop_seconds: float | None = None,
    ) -> None:
        self.seconds = float(seconds)
        self.sample_rate = int(sample_rate)
        self.min_rms = float(min_rms)
        self.scan_hop_seconds = float(scan_hop_seconds or max(1.0, seconds / 2.0))
        self._selection: dict[str, tuple[Path, float]] = {}

    def load(self, row: Mapping[str, object], *, cache: StemCache) -> torch.Tensor:
        row_id = str(row.get("row_id") or "")
        if row_id in self._selection:
            path, start = self._selection[row_id]
            return _load_chunk(
                path,
                start_seconds=start,
                duration_seconds=self.seconds,
                sample_rate=self.sample_rate,
            )

        path = cache.wav_for_member(row, row["target_member"])
        best_wave = None
        best_start = 0.0
        best_rms = -1.0
        window_samples = int(round(self.seconds * self.sample_rate))
        hop_samples = max(1, int(round(self.scan_hop_seconds * self.sample_rate)))
        for raw_span in row.get("positive_enrollment_spans") or []:
            span = dict(raw_span)
            start = float(span.get("start") or 0.0)
            end = float(span.get("end") or start)
            if end <= start:
                continue
            span_wave = _load_chunk(
                path,
                start_seconds=start,
                duration_seconds=max(self.seconds, end - start),
                sample_rate=self.sample_rate,
            )
            latest_offset = max(0, span_wave.numel() - window_samples)
            offsets = torch.arange(0, latest_offset + 1, hop_samples, dtype=torch.long)
            if offsets.numel() == 0 or int(offsets[-1]) != latest_offset:
                offsets = torch.cat([offsets, torch.tensor([latest_offset], dtype=torch.long)])
            cumulative = F.pad(span_wave.square().cumsum(dim=0), (1, 0))
            energies = (cumulative[offsets + window_samples] - cumulative[offsets]) / window_samples
            best_index = int(torch.argmax(energies))
            offset = int(offsets[best_index])
            wave = span_wave[offset : offset + window_samples]
            rms = float(torch.sqrt(energies[best_index]))
            if rms > best_rms:
                best_wave = wave
                best_start = start + offset / self.sample_rate
                best_rms = rms
        if best_wave is None or best_rms < self.min_rms:
            raise ValueError(f"Low-RMS enrollment for {row_id}: {best_rms:.6g}")
        self._selection[row_id] = (path, best_start)
        return best_wave


def _load_group_chunk(
    rows: Sequence[Mapping[str, object]],
    *,
    cache: StemCache,
    enrollment_selector: EnrollmentSelector,
    references: ForcedReferenceIndex,
    rng: random.Random,
    chunk_seconds: float,
    sample_rate: int,
    overlap_probability: float,
    active_probability: float,
    min_mixture_rms: float,
    chunk_start: float | None = None,
) -> dict[str, object]:
    rows = sorted(rows, key=lambda row: str(row.get("speaker_id") or ""))
    first = rows[0]
    words = references.words_for(first)
    if chunk_start is None:
        chunk_start = _sample_group_chunk_start(
            words,
            duration=float(first.get("duration") or 0.0),
            chunk_seconds=chunk_seconds,
            overlap_probability=overlap_probability,
            active_probability=active_probability,
            rng=rng,
        )
    absolute_start = float(first.get("window_start") or 0.0) + float(chunk_start)
    source_by_member = {
        str(member): _load_chunk(
            cache.wav_for_member(first, member),
            start_seconds=absolute_start,
            duration_seconds=chunk_seconds,
            sample_rate=sample_rate,
        )
        for member in first.get("mixture_members") or []
    }
    sources = list(source_by_member.values())
    if not sources:
        raise ValueError(f"No mixture sources for {_group_key(first)}")
    mixture = torch.stack(sources).sum(dim=0)
    if _rms(mixture) < min_mixture_rms:
        raise ValueError(f"Low-RMS mixture for {_group_key(first)}")
    peak = max(float(mixture.abs().max()), 1e-6)
    scale = 1.0
    if peak > 0.95:
        scale = 0.95 / peak
        mixture = mixture * scale

    enrollments = []
    target_sources = []
    target_masks = []
    speakers = []
    for row in rows:
        enrollment = enrollment_selector.load(row, cache=cache)
        target_member = str(row["target_member"])
        target_source = source_by_member.get(target_member)
        if target_source is None:
            target_source = _load_chunk(
                cache.wav_for_member(row, target_member),
                start_seconds=absolute_start,
                duration_seconds=chunk_seconds,
                sample_rate=sample_rate,
            )
        target_mask, _ = _word_masks(
            row,
            chunk_start=float(chunk_start),
            chunk_seconds=chunk_seconds,
            sample_rate=sample_rate,
            forced_references=references,
        )
        speakers.append(str(row.get("speaker_id") or ""))
        enrollments.append(enrollment)
        target_sources.append(target_source * scale)
        target_masks.append(target_mask)
    return {
        "mixture": mixture.unsqueeze(0).expand(len(rows), -1).contiguous(),
        "enrollments": torch.stack(enrollments),
        "target_sources": torch.stack(target_sources),
        "target_masks": torch.stack(target_masks),
        "speakers": speakers,
        "words": words,
        "chunk_start": float(chunk_start),
        "chunk_seconds": float(chunk_seconds),
        "key": _group_key(first),
    }


def _subset_batch_candidates(
    batch: Mapping[str, object],
    *,
    limit: int,
    rng: random.Random,
) -> dict[str, object]:
    speakers = list(batch["speakers"])
    if limit <= 0 or len(speakers) <= limit:
        return dict(batch)
    chunk_start = float(batch["chunk_start"])
    chunk_end = chunk_start + float(batch["chunk_seconds"])
    active = {
        str(word.get("speaker") or "")
        for word in list(batch["words"])
        if float(word.get("end") or 0.0) > chunk_start
        and float(word.get("start") or 0.0) < chunk_end
    }
    active_indices = [index for index, speaker in enumerate(speakers) if speaker in active]
    rng.shuffle(active_indices)
    selected = active_indices[:limit]
    remaining = [index for index in range(len(speakers)) if index not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: limit - len(selected)])
    selected = sorted(selected)
    selected_batch = {
        **batch,
        "mixture": torch.as_tensor(batch["mixture"])[selected],
        "enrollments": torch.as_tensor(batch["enrollments"])[selected],
        "target_masks": torch.as_tensor(batch["target_masks"])[selected],
        "speakers": [speakers[index] for index in selected],
    }
    if "target_sources" in batch:
        selected_batch["target_sources"] = torch.as_tensor(batch["target_sources"])[selected]
    return selected_batch


def _word_owner_loss(
    logits: torch.Tensor,
    *,
    words: Sequence[Mapping[str, object]],
    speakers: Sequence[str],
    chunk_start: float,
    chunk_seconds: float,
    frame_hop_seconds: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    speaker_indices = {speaker: index for index, speaker in enumerate(speakers)}
    scores = []
    labels = []
    overlap_flags = []
    chunk_end = chunk_start + chunk_seconds
    for word in words:
        speaker = str(word.get("speaker") or "")
        if speaker not in speaker_indices:
            continue
        start = max(float(word.get("start") or 0.0), chunk_start)
        end = min(float(word.get("end") or start), chunk_end)
        if end <= start:
            continue
        first_frame = max(0, int(math.floor((start - chunk_start) / frame_hop_seconds)))
        last_frame = min(
            logits.shape[-1],
            max(first_frame + 1, int(math.ceil((end - chunk_start) / frame_hop_seconds))),
        )
        scores.append(logits[:, first_frame:last_frame].mean(dim=-1))
        labels.append(speaker_indices[speaker])
        overlap_flags.append(_word_has_overlap(word, words))
    if not scores:
        return logits.sum() * 0.0, {
            "words": 0,
            "correct": 0,
            "overlap_words": 0,
            "overlap_correct": 0,
        }
    score_tensor = torch.stack(scores)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=logits.device)
    predictions = score_tensor.argmax(dim=-1)
    correct = predictions.eq(label_tensor)
    overlap = torch.tensor(overlap_flags, dtype=torch.bool, device=logits.device)
    return F.cross_entropy(score_tensor, label_tensor), {
        "words": len(labels),
        "correct": int(correct.sum().detach().cpu()),
        "overlap_words": int(overlap.sum().detach().cpu()),
        "overlap_correct": int((correct & overlap).sum().detach().cpu()),
    }


def _frame_loss_and_metrics(
    logits: torch.Tensor,
    sample_masks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    labels = F.interpolate(
        sample_masks.unsqueeze(1),
        size=logits.shape[-1],
        mode="nearest",
    ).squeeze(1)
    positives = labels.sum()
    negatives = labels.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 8.0)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)
    predicted = torch.sigmoid(logits) >= 0.5
    positive = labels >= 0.5
    return loss, {
        "tp": int((predicted & positive).sum().detach().cpu()),
        "fp": int((predicted & ~positive).sum().detach().cpu()),
        "fn": int((~predicted & positive).sum().detach().cpu()),
    }


def _run_chunk(
    *,
    extractor: PS4FeatureExtractor | LauraFeatureExtractor,
    head: PS4ActivityHead | LauraActivityHead,
    batch: Mapping[str, object],
    device: torch.device,
    word_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    mixture = torch.as_tensor(batch["mixture"]).to(device)
    enrollments = torch.as_tensor(batch["enrollments"]).to(device)
    masks = torch.as_tensor(batch["target_masks"]).to(device)
    extractor.set_trainable_mode(head.training)
    features = extractor(mixture, enrollments)
    logits = head(features)
    frame_loss, frame_metrics = _frame_loss_and_metrics(logits, masks)
    word_loss, word_metrics = _word_owner_loss(
        logits,
        words=list(batch["words"]),
        speakers=list(batch["speakers"]),
        chunk_start=float(batch["chunk_start"]),
        chunk_seconds=float(batch["chunk_seconds"]),
        frame_hop_seconds=extractor.frame_hop_seconds,
    )
    loss = frame_loss + word_loss_weight * word_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "frame_loss": float(frame_loss.detach().cpu()),
        "word_loss": float(word_loss.detach().cpu()),
        **frame_metrics,
        **word_metrics,
    }


def _metric_summary(records: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    totals = {
        key: sum(int(record.get(key) or 0) for record in records)
        for key in ("tp", "fp", "fn", "words", "correct", "overlap_words", "overlap_correct")
    }
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    summary: dict[str, float | int] = {
        "loss": float(np.mean([float(record["loss"]) for record in records])) if records else 0.0,
        "frame_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "word_accuracy": totals["correct"] / max(1, totals["words"]),
        "overlap_word_accuracy": totals["overlap_correct"] / max(1, totals["overlap_words"]),
        **totals,
    }
    for key in ("frame_loss", "word_loss", "extraction_loss"):
        values = [float(record[key]) for record in records if key in record]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def _evaluate_chunks(
    *,
    groups: Sequence[Sequence[Mapping[str, object]]],
    cache: StemCache,
    enrollment_selector: EnrollmentSelector,
    references: ForcedReferenceIndex,
    extractor: PS4FeatureExtractor | LauraFeatureExtractor,
    head: PS4ActivityHead | LauraActivityHead,
    device: torch.device,
    seed: int,
    batches: int,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    head.eval()
    records = []
    with torch.inference_mode():
        attempts = 0
        while len(records) < batches and attempts < batches * 20:
            attempts += 1
            try:
                batch = _load_group_chunk(
                    groups[attempts % len(groups)],
                    cache=cache,
                    enrollment_selector=enrollment_selector,
                    references=references,
                    rng=rng,
                    chunk_seconds=float(args.chunk_seconds),
                    sample_rate=int(args.sample_rate),
                    overlap_probability=float(args.dev_overlap_probability),
                    active_probability=float(args.active_probability),
                    min_mixture_rms=float(args.min_mixture_rms),
                )
            except (OSError, RuntimeError, ValueError):
                continue
            _, record = _run_chunk(
                extractor=extractor,
                head=head,
                batch=batch,
                device=device,
                word_loss_weight=float(args.word_loss_weight),
            )
            records.append(record)
    head.train()
    return _metric_summary(records)


def _resolve_audio_path(value: object, *, manifest_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, manifest_dir / path):
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _materialized_enrollment(
    row: Mapping[str, object],
    *,
    manifest_dir: Path,
    seconds: float,
    sample_rate: int,
) -> torch.Tensor:
    paths = list(dict(row.get("materialized") or {}).get("positive_enrollment_paths") or [])
    if not paths:
        raise ValueError(f"Missing materialized enrollment for {row.get('row_id')}")
    return _load_chunk(
        _resolve_audio_path(paths[0], manifest_dir=manifest_dir),
        start_seconds=0.0,
        duration_seconds=seconds,
        sample_rate=sample_rate,
    )


def _infer_group_posteriors(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_dir: Path,
    extractor: PS4FeatureExtractor | LauraFeatureExtractor,
    head: PS4ActivityHead | LauraActivityHead,
    device: torch.device,
    chunk_seconds: float,
    enrollment_seconds: float,
    sample_rate: int,
    candidate_batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = sorted(rows, key=lambda row: str(row.get("speaker_id") or ""))
    materialized = dict(rows[0].get("materialized") or {})
    mixture_path = _resolve_audio_path(materialized["mixture_path"], manifest_dir=manifest_dir)
    total_seconds = float(rows[0].get("duration") or 0.0)
    total_frames = int(math.ceil(total_seconds / extractor.frame_hop_seconds)) + 1
    frame_centers = np.arange(total_frames, dtype=np.float32) * extractor.frame_hop_seconds
    accumulators = {
        str(row.get("speaker_id") or ""): np.zeros(total_frames, dtype=np.float64) for row in rows
    }
    weights = {speaker: np.zeros(total_frames, dtype=np.float64) for speaker in accumulators}
    enrollments = [
        _materialized_enrollment(
            row,
            manifest_dir=manifest_dir,
            seconds=enrollment_seconds,
            sample_rate=sample_rate,
        )
        for row in rows
    ]
    chunk_samples = int(round(chunk_seconds * sample_rate))
    head.eval()
    with torch.inference_mode():
        for start_sample in range(0, int(round(total_seconds * sample_rate)), chunk_samples):
            mixture = _load_chunk(
                mixture_path,
                start_seconds=start_sample / sample_rate,
                duration_seconds=chunk_seconds,
                sample_rate=sample_rate,
            )
            for candidate_start in range(0, len(rows), candidate_batch_size):
                candidate_rows = rows[candidate_start : candidate_start + candidate_batch_size]
                enrollment_batch = torch.stack(
                    enrollments[candidate_start : candidate_start + candidate_batch_size]
                ).to(device)
                mixture_batch = mixture.unsqueeze(0).expand(len(candidate_rows), -1).to(device)
                logits = head(extractor(mixture_batch, enrollment_batch))
                posteriors = torch.sigmoid(logits).float().cpu().numpy()
                frame_start = int(round(start_sample / sample_rate / extractor.frame_hop_seconds))
                frame_stop = min(total_frames, frame_start + posteriors.shape[-1])
                valid = frame_stop - frame_start
                for index, row in enumerate(candidate_rows):
                    speaker = str(row.get("speaker_id") or "")
                    accumulators[speaker][frame_start:frame_stop] += posteriors[index, :valid]
                    weights[speaker][frame_start:frame_stop] += 1.0
    return frame_centers, {
        speaker: (values / np.maximum(weights[speaker], 1.0)).astype(np.float32)
        for speaker, values in accumulators.items()
    }


def _evaluate_full_groups(
    groups: Sequence[Sequence[Mapping[str, object]]],
    *,
    references: ForcedReferenceIndex,
    manifest_dir: Path,
    extractor: PS4FeatureExtractor | LauraFeatureExtractor,
    head: PS4ActivityHead | LauraActivityHead,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    results = []
    records = []
    selected = groups[: int(args.max_full_eval_groups)] if args.max_full_eval_groups > 0 else groups
    for group_index, rows in enumerate(selected, start=1):
        key = _group_key(rows[0])
        print(f"full_dev_group {group_index}/{len(selected)} key={key}", flush=True)
        frame_centers, posteriors = _infer_group_posteriors(
            rows,
            manifest_dir=manifest_dir,
            extractor=extractor,
            head=head,
            device=device,
            chunk_seconds=float(args.inference_chunk_seconds),
            enrollment_seconds=float(args.enrollment_seconds),
            sample_rate=int(args.sample_rate),
            candidate_batch_size=int(args.inference_candidate_batch_size),
        )
        words = references.words_for(rows[0])
        result, word_records = _score_word_owners(
            words=words,
            speakers=list(posteriors),
            frame_centers=frame_centers,
            speaker_posteriors=posteriors,
        )
        result.update(
            {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "split_id": str(rows[0].get("split_id") or ""),
                "row_count": len(rows),
            }
        )
        for record in word_records:
            record.update({"session": key[0], "window_start": key[1], "window_end": key[2]})
        results.append(result)
        records.extend(word_records)
    return results, records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a direct known-speaker activity and word-owner head on TSE features."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=("ps4", "laura"), default="ps4")
    parser.add_argument("--wesep-repo", type=Path)
    parser.add_argument("--wespeaker-repo", type=Path)
    parser.add_argument("--laura-repo", type=Path)
    parser.add_argument("--laura-checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--initial-head-checkpoint",
        type=Path,
        help="Optional prior activity-head checkpoint to continue from.",
    )
    parser.add_argument("--hf-repo", default=DEFAULT_PS4_REPO)
    parser.add_argument("--checkpoint-file", default=DEFAULT_PS4_CHECKPOINT)
    parser.add_argument("--stems-cache-root", type=Path)
    parser.add_argument("--stems-wav-root", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--train-sessions")
    parser.add_argument("--dev-sessions")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-dev-rows", type=int)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--dev-batches", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--active-probability", type=float, default=0.9)
    parser.add_argument("--train-overlap-probability", type=float, default=0.65)
    parser.add_argument("--dev-overlap-probability", type=float, default=0.65)
    parser.add_argument("--word-loss-weight", type=float, default=1.0)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--trainable-separator-blocks", type=int, default=0)
    parser.add_argument(
        "--train-candidates-per-step",
        type=int,
        default=0,
        help="Limit each training update to this many active candidates; dev always uses all.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--min-mixture-rms", type=float, default=1e-4)
    parser.add_argument("--min-enrollment-rms", type=float, default=1e-4)
    parser.add_argument("--max-batch-retries", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-dev-eval", action="store_true")
    parser.add_argument("--max-full-eval-groups", type=int, default=0)
    parser.add_argument("--inference-chunk-seconds", type=float, default=20.0)
    parser.add_argument("--inference-candidate-batch-size", type=int, default=2)
    args = parser.parse_args()

    if int(args.sample_rate) != 16000:
        raise ValueError("TSE activity training requires 16 kHz audio")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    references = ForcedReferenceIndex(args.forced_reference_jsonl)
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
    train_coverage = _forced_reference_coverage(train_rows, references)
    dev_coverage = _forced_reference_coverage(dev_rows, references)
    for label, coverage in (("train", train_coverage), ("dev", dev_coverage)):
        _validate_forced_reference_coverage(
            label=label,
            coverage=coverage,
            require_forced_reference=True,
            min_row_coverage=1.0,
            min_word_coverage=0.99,
        )
    train_groups = _group_rows(train_rows)
    dev_groups = _group_rows(dev_rows)
    if args.backbone == "ps4":
        if args.wesep_repo is None or args.wespeaker_repo is None:
            raise ValueError("PS4 requires --wesep-repo and --wespeaker-repo")
        checkpoint_path = _resolve_checkpoint(
            checkpoint_path=args.checkpoint_path,
            hf_repo=str(args.hf_repo),
            checkpoint_file=str(args.checkpoint_file),
        )
        backbone = load_ps4_model(
            wesep_repo=args.wesep_repo,
            wespeaker_repo=args.wespeaker_repo,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        extractor: PS4FeatureExtractor | LauraFeatureExtractor = PS4FeatureExtractor(
            backbone,
            trainable_separator_blocks=int(args.trainable_separator_blocks),
        )
        head: PS4ActivityHead | LauraActivityHead = PS4ActivityHead(
            channels=int(args.channels),
            layers=int(args.layers),
            dropout=float(args.dropout),
        ).to(device)
    else:
        if args.laura_repo is None:
            raise ValueError("Laura requires --laura-repo")
        checkpoint_path = args.laura_checkpoint_path or Path(
            hf_hub_download(DEFAULT_LAURA_HF_REPO, DEFAULT_LAURA_CHECKPOINT)
        )
        backbone = _load_usef_laura_front(
            usef_repo=args.laura_repo,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        extractor = LauraFeatureExtractor(
            backbone,
            trainable_separator_blocks=int(args.trainable_separator_blocks),
        )
        head = LauraActivityHead(
            channels=int(args.channels),
            layers=int(args.layers),
            dropout=float(args.dropout),
        ).to(device)
    initial_step = 0
    initial_dev: Mapping[str, object] | None = None
    if args.initial_head_checkpoint is not None:
        initial = torch.load(
            args.initial_head_checkpoint,
            map_location=device,
            weights_only=True,
        )
        saved_backbone = initial.get("backbone_type")
        if saved_backbone and str(saved_backbone) != str(args.backbone):
            raise ValueError(
                f"Head checkpoint backbone is {saved_backbone}, requested {args.backbone}"
            )
        head.load_state_dict(initial.get("head", initial), strict=True)
        if initial.get("backbone"):
            extractor.load_trainable_state_dict(initial["backbone"])
        initial_step = int(initial.get("step") or 0)
        initial_dev = dict(initial.get("dev") or {})
    optimized_parameters = list(head.parameters()) + extractor.trainable_parameters()
    parameter_groups = [{"params": list(head.parameters()), "lr": float(args.learning_rate)}]
    if extractor.trainable_parameters():
        parameter_groups.append(
            {
                "params": extractor.trainable_parameters(),
                "lr": float(args.backbone_learning_rate),
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))
    cache = StemCache(
        root=args.stems_cache_root or args.output_dir / "_stems",
        wav_root=args.stems_wav_root or args.output_dir / "_stems16",
    )
    enrollment_selector = EnrollmentSelector(
        seconds=float(args.enrollment_seconds),
        sample_rate=int(args.sample_rate),
        min_rms=float(args.min_enrollment_rms),
    )
    metadata = {
        "args": vars(args),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "forced_reference_coverage": {"train": train_coverage, "dev": dev_coverage},
        "backbone": str(args.backbone),
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_checkpoint_sha256": _sha256(checkpoint_path),
        "ps4_model_args": PS4_MODEL_ARGS if args.backbone == "ps4" else None,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "trainable_backbone_parameters": sum(
            parameter.numel() for parameter in extractor.trainable_parameters()
        ),
        "initial_head_checkpoint": (
            str(args.initial_head_checkpoint.resolve()) if args.initial_head_checkpoint else None
        ),
        "initial_step": initial_step,
    }
    (args.output_dir / "ps4_activity_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    rng = random.Random(int(args.seed))
    history = []
    best_accuracy = float(initial_dev.get("word_accuracy") or -1.0) if initial_dev else -1.0
    successful_steps = 0
    attempts = 0
    if args.initial_head_checkpoint is not None:
        torch.save(
            {
                "head": head.state_dict(),
                "backbone": extractor.trainable_state_dict(),
                "backbone_type": str(args.backbone),
                "step": initial_step,
                "dev": dict(initial_dev or {}),
                "head_args": {
                    "channels": int(args.channels),
                    "layers": int(args.layers),
                    "dropout": float(args.dropout),
                },
            },
            args.output_dir / "best_ps4_activity_head.pt",
        )
    while successful_steps < int(args.train_steps) and attempts < int(args.train_steps) * int(
        args.max_batch_retries
    ):
        attempts += 1
        group = train_groups[rng.randrange(len(train_groups))]
        try:
            batch = _load_group_chunk(
                group,
                cache=cache,
                enrollment_selector=enrollment_selector,
                references=references,
                rng=rng,
                chunk_seconds=float(args.chunk_seconds),
                sample_rate=int(args.sample_rate),
                overlap_probability=float(args.train_overlap_probability),
                active_probability=float(args.active_probability),
                min_mixture_rms=float(args.min_mixture_rms),
            )
        except (OSError, RuntimeError, ValueError):
            continue
        batch = _subset_batch_candidates(
            batch,
            limit=int(args.train_candidates_per_step),
            rng=rng,
        )
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, train_record = _run_chunk(
            extractor=extractor,
            head=head,
            batch=batch,
            device=device,
            word_loss_weight=float(args.word_loss_weight),
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized_parameters, float(args.clip_grad_norm))
        optimizer.step()
        successful_steps += 1
        global_step = initial_step + successful_steps
        if successful_steps == 1 or successful_steps % int(args.log_every) == 0:
            dev_metrics = _evaluate_chunks(
                groups=dev_groups,
                cache=cache,
                enrollment_selector=enrollment_selector,
                references=references,
                extractor=extractor,
                head=head,
                device=device,
                seed=int(args.seed) + 1,
                batches=int(args.dev_batches),
                args=args,
            )
            record = {"step": global_step, "train": train_record, "dev": dev_metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            accuracy = float(dev_metrics["word_accuracy"])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    {
                        "head": head.state_dict(),
                        "backbone": extractor.trainable_state_dict(),
                        "backbone_type": str(args.backbone),
                        "step": global_step,
                        "dev": dev_metrics,
                        "head_args": {
                            "channels": int(args.channels),
                            "layers": int(args.layers),
                            "dropout": float(args.dropout),
                        },
                    },
                    args.output_dir / "best_ps4_activity_head.pt",
                )
    if successful_steps < int(args.train_steps):
        raise RuntimeError(
            f"Only completed {successful_steps}/{args.train_steps} updates after {attempts} attempts"
        )
    best = torch.load(
        args.output_dir / "best_ps4_activity_head.pt",
        map_location=device,
        weights_only=True,
    )
    head.load_state_dict(best["head"], strict=True)
    if best.get("backbone"):
        extractor.load_trainable_state_dict(best["backbone"])
    summary = {
        "model": f"{args.backbone}_activity_word_owner",
        "backbone": str(args.backbone),
        "best_step": int(best["step"]),
        "best_chunk_dev": best["dev"],
        "completed_steps": successful_steps,
        "initial_step": initial_step,
        "final_step": initial_step + successful_steps,
        "attempts": attempts,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    if args.full_dev_eval:
        group_results, word_records = _evaluate_full_groups(
            dev_groups,
            references=references,
            manifest_dir=args.manifest.resolve().parent,
            extractor=extractor,
            head=head,
            device=device,
            args=args,
        )
        summary["full_dev"] = _summarize_groups(group_results)
        _write_jsonl(args.output_dir / "ps4_activity_dev_groups.jsonl", group_results)
        _write_jsonl(args.output_dir / "ps4_activity_dev_words.jsonl", word_records)
    (args.output_dir / "ps4_activity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
