from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AudioAugmentationConfig:
    profile: str = "none"
    copies: int = 0
    seed: int = 13

    @property
    def enabled(self) -> bool:
        profile = self.profile.strip().lower()
        return profile not in {"", "none", "off", "disabled"} and int(self.copies) > 0


WaveformAugmenter = Callable[[np.ndarray, int, str, int], np.ndarray]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def _peak_normalize(waveform: np.ndarray, *, peak: float = 0.98) -> np.ndarray:
    peak_value = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak_value <= 1e-8:
        return waveform.astype(np.float32, copy=False)
    scale = min(float(peak) / peak_value, 1.5)
    return (waveform * scale).astype(np.float32)


def _apply_gain(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    return waveform * float(10.0 ** (gain_db / 20.0))


def _add_noise(waveform: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    signal_rms = max(_rms(waveform), 1e-6)
    noise_rms = signal_rms / float(10.0 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, waveform.shape[0]).astype(np.float32)
    return waveform + noise


def _bandlimit_fft(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    low_cut_hz: float | None,
    high_cut_hz: float | None,
) -> np.ndarray:
    if waveform.size < 32:
        return waveform
    spectrum = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(waveform.shape[0], d=1.0 / float(sample_rate))
    mask = np.ones(freqs.shape[0], dtype=bool)
    if low_cut_hz is not None and low_cut_hz > 0.0:
        mask &= freqs >= float(low_cut_hz)
    if high_cut_hz is not None and high_cut_hz > 0.0:
        mask &= freqs <= float(high_cut_hz)
    filtered = np.fft.irfft(spectrum * mask, n=waveform.shape[0])
    return filtered.astype(np.float32)


def _resample_roundtrip(waveform: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if waveform.size < 8 or target_rate <= 0 or target_rate == sample_rate:
        return waveform
    duration = waveform.shape[0] / float(sample_rate)
    down_len = max(8, int(round(duration * float(target_rate))))
    if down_len == waveform.shape[0]:
        return waveform
    base_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=True)
    down_positions = np.linspace(0.0, 1.0, num=down_len, endpoint=True)
    downsampled = np.interp(down_positions, base_positions, waveform).astype(np.float32)
    restored = np.interp(base_positions, down_positions, downsampled).astype(np.float32)
    return restored


def _apply_reverb(waveform: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    if waveform.size < 64:
        return waveform
    ir_len = max(int(0.03 * sample_rate), 32)
    ir_len = min(ir_len, max(waveform.shape[0] // 2, 32))
    times = np.arange(ir_len, dtype=np.float32) / float(sample_rate)
    decay_rate = rng.uniform(18.0, 45.0)
    impulse = np.exp(-decay_rate * times).astype(np.float32)
    impulse[0] = 1.0
    impulse[1:] *= rng.uniform(0.08, 0.22)
    impulse[1:] *= rng.uniform(0.6, 1.4, size=impulse.shape[0] - 1).astype(np.float32)
    convolved = np.convolve(waveform, impulse, mode="full")[: waveform.shape[0]]
    dry_mix = rng.uniform(0.78, 0.92)
    wet_mix = 1.0 - dry_mix
    return (dry_mix * waveform) + (wet_mix * convolved.astype(np.float32))


def _soft_clip(waveform: np.ndarray, drive: float) -> np.ndarray:
    drive = max(float(drive), 1.0)
    denom = math.tanh(drive)
    if abs(denom) <= 1e-6:
        return waveform
    return np.tanh(waveform * drive).astype(np.float32) / denom


def _dropout_bursts(
    waveform: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    count: int,
) -> np.ndarray:
    if waveform.size < 128 or count <= 0:
        return waveform
    augmented = waveform.copy()
    for _ in range(count):
        burst_len = max(8, int(sample_rate * rng.uniform(0.006, 0.03)))
        if burst_len >= augmented.shape[0]:
            break
        start = int(rng.integers(0, max(1, augmented.shape[0] - burst_len)))
        augmented[start : start + burst_len] *= rng.uniform(0.0, 0.25)
    return augmented


def build_waveform_augmenter(
    config: AudioAugmentationConfig,
    *,
    domain: str,
    pass_index: int,
) -> WaveformAugmenter | None:
    profile = str(config.profile or "none").strip().lower()
    if profile in {"", "none", "off", "disabled"}:
        return None

    def _augment(waveform: np.ndarray, sample_rate: int, speaker: str, index: int) -> np.ndarray:
        if waveform.size == 0:
            return waveform.astype(np.float32, copy=False)
        rng = np.random.default_rng(
            _stable_seed(
                config.seed, profile, domain, pass_index, speaker, index, waveform.shape[0]
            )
        )
        augmented = waveform.astype(np.float32, copy=True)
        baseline_peak = max(float(np.max(np.abs(augmented))), 1e-3)

        if profile == "light":
            augmented = _apply_gain(augmented, rng.uniform(-2.5, 2.5))
            if rng.random() < 0.55:
                augmented = _add_noise(augmented, rng, snr_db=rng.uniform(18.0, 28.0))
            if rng.random() < 0.35:
                augmented = _bandlimit_fft(
                    augmented,
                    sample_rate,
                    low_cut_hz=rng.uniform(50.0, 120.0),
                    high_cut_hz=rng.uniform(3600.0, 7000.0),
                )
            if rng.random() < 0.15:
                augmented = _apply_reverb(augmented, sample_rate, rng)
        else:
            augmented = _apply_gain(augmented, rng.uniform(-5.0, 4.0))
            if rng.random() < 0.8:
                augmented = _add_noise(augmented, rng, snr_db=rng.uniform(10.0, 24.0))
            if rng.random() < 0.85:
                target_rate = int(rng.choice(np.array([8000, 12000], dtype=np.int32)))
                augmented = _resample_roundtrip(augmented, sample_rate, target_rate)
            if rng.random() < 0.7:
                augmented = _bandlimit_fft(
                    augmented,
                    sample_rate,
                    low_cut_hz=rng.uniform(70.0, 180.0),
                    high_cut_hz=rng.uniform(2600.0, 5200.0),
                )
            if rng.random() < 0.25:
                augmented = _apply_reverb(augmented, sample_rate, rng)
            if rng.random() < 0.45:
                augmented = _soft_clip(augmented, drive=rng.uniform(1.2, 2.4))
            if rng.random() < 0.3:
                augmented = _dropout_bursts(
                    augmented,
                    sample_rate,
                    rng,
                    count=int(rng.integers(1, 4)),
                )

        if not np.isfinite(augmented).all() or _rms(augmented) <= 1e-7:
            return waveform.astype(np.float32, copy=False)
        normalized = _peak_normalize(augmented, peak=min(0.98, baseline_peak * 1.05))
        return normalized.astype(np.float32, copy=False)

    return _augment
