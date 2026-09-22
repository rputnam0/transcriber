from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    CleanRecord,
    _limit_rows_by_group,
    _load_audio_segment,
    _load_clean_records,
    _normalize_wave,
    _session_stems,
    _slice_wave,
    _wave_rms,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _load_titanet_word_npz,
    evaluate_mask_embeddings,
)


class EnrollmentTargetMaskNet(nn.Module):
    def __init__(self, embedding_dim: int, cond_channels: int = 16) -> None:
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, cond_channels),
            nn.SiLU(),
        )
        channels = 1 + cond_channels
        layers: List[nn.Module] = [
            nn.Conv2d(channels, 48, kernel_size=5, padding=2),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
        ]
        for dilation in (1, 2, 4, 8, 16, 24):
            layers.extend(
                [
                    nn.Conv2d(
                        48,
                        48,
                        kernel_size=(3, 5),
                        padding=(1, 2 * dilation),
                        dilation=(1, dilation),
                    ),
                    nn.GroupNorm(6, 48),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(48, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, logmag: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        batch, _channel, freq, frames = logmag.shape
        cond = self.cond_proj(enrollment).view(batch, -1, 1, 1).expand(-1, -1, freq, frames)
        return self.net(torch.cat([logmag, cond], dim=1))


def _speaker_centroids(clean_bank_path: Path, speakers: Sequence[str]) -> Dict[str, np.ndarray]:
    payload = np.load(clean_bank_path, allow_pickle=False)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    centroids: Dict[str, np.ndarray] = {}
    for speaker in speakers:
        rows = embeddings[[index for index, label in enumerate(labels) if label == speaker]]
        if rows.size == 0:
            raise ValueError(f"No clean-bank embeddings for {speaker}")
        centroid = np.mean(rows, axis=0)
        norm = max(float(np.linalg.norm(centroid)), 1e-8)
        centroids[speaker] = (centroid / norm).astype(np.float32)
    return centroids


def _load_all_stem_segments(
    paths: Sequence[Path],
    *,
    start: float,
    duration: float,
    sample_rate: int,
    samples: int,
) -> List[Tuple[Path, np.ndarray]]:
    segments: List[Tuple[Path, np.ndarray]] = []
    for path in paths:
        try:
            wave = _load_audio_segment(path, start, duration, sample_rate)
        except Exception:
            continue
        if wave.shape[0] == samples and np.isfinite(wave).all():
            segments.append((path, np.asarray(wave, dtype=np.float32)))
    return segments


def _infer_speaker_stem(
    paths: Sequence[Path],
    records: Sequence[CleanRecord],
    *,
    sample_rate: int,
    duration: float,
    samples: int,
    max_records: int,
    rng: random.Random,
) -> Path | None:
    candidates = list(records)
    rng.shuffle(candidates)
    scores: Dict[Path, float] = defaultdict(float)
    for record in candidates[:max_records]:
        span = max(record.end - record.start - duration, 0.0)
        offset = rng.uniform(0.0, span) if span > 0.0 else 0.0
        for path, wave in _load_all_stem_segments(
            paths,
            start=record.start + offset,
            duration=duration,
            sample_rate=sample_rate,
            samples=samples,
        ):
            scores[path] += _wave_rms(wave)
    if not scores:
        return None
    return max(scores, key=scores.get)


def _make_real_stem_cache(
    *,
    records_by_speaker: Mapping[str, Sequence[CleanRecord]],
    audio_root: Path,
    cache_root: Path,
    output_path: Path,
    sample_rate: int,
    duration: float,
    max_per_speaker: int,
    min_interferer_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    targets: List[np.ndarray] = []
    interferers: List[np.ndarray] = []
    labels: List[str] = []
    target_shares: List[float] = []
    active_counts: List[int] = []
    sessions: List[str] = []
    stem_cache = cache_root / "stems"
    session_cache: Dict[str, Tuple[Mapping[str, Path], List[Path]]] = {}
    target_path_cache: Dict[Tuple[str, str], Path | None] = {}
    samples = int(round(duration * sample_rate))

    for speaker in CORE_SPEAKERS:
        candidates = list(records_by_speaker.get(speaker) or [])
        rng.shuffle(candidates)
        speaker_count = 0
        attempts = 0
        for record in candidates:
            if speaker_count >= max_per_speaker:
                break
            attempts += 1
            if record.session not in session_cache:
                session_cache[record.session] = _session_stems(
                    record.session,
                    audio_root=audio_root,
                    stem_cache=stem_cache,
                )
            key = (record.session, speaker)
            if key not in target_path_cache:
                _by_speaker, infer_paths = session_cache[record.session]
                target_path_cache[key] = _infer_speaker_stem(
                    infer_paths,
                    [
                        candidate
                        for candidate in records_by_speaker.get(speaker, [])
                        if candidate.session == record.session
                    ],
                    sample_rate=sample_rate,
                    duration=duration,
                    samples=samples,
                    max_records=8,
                    rng=rng,
                )
            target_path = target_path_cache[key]
            if target_path is None:
                continue
            span = max(record.end - record.start - duration, 0.0)
            offset = rng.uniform(0.0, span) if span > 0.0 else 0.0
            _by_speaker, paths = session_cache[record.session]
            source_items = _load_all_stem_segments(
                paths,
                start=record.start + offset,
                duration=duration,
                sample_rate=sample_rate,
                samples=samples,
            )
            source_paths = [path for path, _wave in source_items]
            source_waves = [wave for _path, wave in source_items]
            if len(source_waves) < 2:
                continue
            if target_path not in source_paths:
                continue
            rms_values = np.asarray([_wave_rms(wave) for wave in source_waves], dtype=np.float64)
            target_index = source_paths.index(target_path)
            target_rms = float(rms_values[target_index])
            if target_rms < 1e-4:
                continue
            other_indices = [index for index in range(len(source_waves)) if index != target_index]
            other_max = max(float(rms_values[index]) for index in other_indices)
            if other_max < target_rms * min_interferer_ratio:
                continue
            target = np.asarray(source_waves[target_index], dtype=np.float32)
            interferer = np.sum(
                np.stack([source_waves[index] for index in other_indices]), axis=0
            ).astype(np.float32)
            peak = max(float(np.max(np.abs(target + interferer))), 1e-6)
            if peak > 0.95:
                scale = 0.95 / peak
                target = target * scale
                interferer = interferer * scale
                rms_values = rms_values * scale
            energy = np.square(rms_values)
            target_share = float(energy[target_index] / max(float(energy.sum()), 1e-12))
            active = int(np.sum(rms_values >= max(target_rms * 0.05, 1e-4)))
            targets.append(target.astype(np.float32))
            interferers.append(interferer.astype(np.float32))
            labels.append(speaker)
            target_shares.append(target_share)
            active_counts.append(active)
            sessions.append(record.session)
            speaker_count += 1
        print(
            f"real_stem_crops {speaker}: {speaker_count}/{attempts} "
            f"min_interferer_ratio={min_interferer_ratio}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        targets=np.stack(targets).astype(np.float32),
        interferers=np.stack(interferers).astype(np.float32),
        labels=np.asarray(labels),
        target_shares=np.asarray(target_shares, dtype=np.float32),
        active_counts=np.asarray(active_counts, dtype=np.int16),
        sessions=np.asarray(sessions),
    )


def _load_or_create_real_stems(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cache_path = args.training_cache.expanduser()
    if not cache_path.exists():
        sessions = [item.strip() for item in str(args.train_sessions).split(",") if item.strip()]
        records_by_speaker = _load_clean_records(
            args.quality_records.expanduser(),
            sessions=sessions,
            speakers=CORE_SPEAKERS,
            min_duration=float(args.window_seconds),
        )
        _make_real_stem_cache(
            records_by_speaker=records_by_speaker,
            audio_root=args.audio_root.expanduser(),
            cache_root=args.cache_root.expanduser(),
            output_path=cache_path,
            sample_rate=int(args.sample_rate),
            duration=float(args.window_seconds),
            max_per_speaker=int(args.max_crops_per_speaker),
            min_interferer_ratio=float(args.min_interferer_ratio),
            seed=int(args.seed),
        )
    payload = np.load(cache_path, allow_pickle=False)
    return (
        np.asarray(payload["targets"], dtype=np.float32),
        np.asarray(payload["interferers"], dtype=np.float32),
        [str(item) for item in payload["labels"].tolist()],
    )


def _stft_mag(
    waves: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    return torch.abs(
        torch.stft(
            waves,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
    )


def _synth_real_batch(
    targets: np.ndarray,
    interferers: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    batch_size: int,
    rng: random.Random,
    min_target_snr_db: float,
    max_target_snr_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = [rng.randrange(targets.shape[0]) for _ in range(batch_size)]
    mixture_rows: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []
    interferer_rows: List[np.ndarray] = []
    enrollment_rows: List[np.ndarray] = []
    for index in indices:
        target = _normalize_wave(targets[index])
        interferer = _normalize_wave(interferers[index])
        snr_db = rng.uniform(min_target_snr_db, max_target_snr_db)
        interferer_gain = 10.0 ** (-snr_db / 20.0)
        target_aug = target.astype(np.float32)
        interferer_aug = (interferer * interferer_gain).astype(np.float32)
        mixture = target_aug + interferer_aug
        peak = max(float(np.max(np.abs(mixture))), 1e-6)
        if peak > 0.95:
            scale = 0.95 / peak
            target_aug = target_aug * scale
            interferer_aug = interferer_aug * scale
            mixture = mixture * scale
        mixture_rows.append(mixture.astype(np.float32))
        target_rows.append(target_aug.astype(np.float32))
        interferer_rows.append(interferer_aug.astype(np.float32))
        enrollment_rows.append(centroids[str(labels[index])])
    return (
        np.stack(mixture_rows).astype(np.float32),
        np.stack(target_rows).astype(np.float32),
        np.stack(interferer_rows).astype(np.float32),
        np.stack(enrollment_rows).astype(np.float32),
    )


def _train_model(
    targets: np.ndarray,
    interferers: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> EnrollmentTargetMaskNet:
    embedding_dim = next(iter(centroids.values())).shape[0]
    model = EnrollmentTargetMaskNet(embedding_dim=embedding_dim).to(device)
    model_path = args.model_output.expanduser()
    if model_path.exists():
        payload = torch.load(model_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4
    )
    window = torch.hann_window(int(args.n_fft), device=device)
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        mixture_np, target_np, interferer_np, enrollment_np = _synth_real_batch(
            targets,
            interferers,
            labels,
            centroids=centroids,
            batch_size=int(args.batch_size),
            rng=rng,
            min_target_snr_db=float(args.min_target_snr_db),
            max_target_snr_db=float(args.max_target_snr_db),
        )
        mixture = torch.from_numpy(mixture_np).to(device)
        target = torch.from_numpy(target_np).to(device)
        interferer = torch.from_numpy(interferer_np).to(device)
        enrollment = torch.from_numpy(enrollment_np).to(device)
        mix_mag = _stft_mag(
            mixture,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
        )
        target_mag = _stft_mag(
            target,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
        )
        interferer_mag = _stft_mag(
            interferer,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            window=window,
        )
        true_mask = (target_mag / (target_mag + interferer_mag).clamp_min(1e-5)).unsqueeze(1)
        predicted = model(torch.log1p(mix_mag).unsqueeze(1), enrollment)
        mask_loss = torch.nn.functional.mse_loss(predicted, true_mask)
        recon_loss = torch.nn.functional.mse_loss(
            torch.log1p(predicted.squeeze(1) * mix_mag),
            torch.log1p(target_mag),
        )
        loss = mask_loss + 0.25 * recon_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"mask={mask_loss.item():.5f} recon={recon_loss.item():.5f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "speakers": CORE_SPEAKERS,
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "conditioning": "clean_bank_centroid",
        },
        model_path,
    )
    return model


def _separate_batch(
    model: EnrollmentTargetMaskNet,
    mixtures: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    n_fft: int,
    hop_length: int,
    device: str,
) -> np.ndarray:
    window = torch.hann_window(n_fft, device=device)
    mixture = torch.from_numpy(np.asarray(mixtures, dtype=np.float32)).to(device)
    enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(device)
    with torch.inference_mode():
        stft = torch.stft(
            mixture,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mag = torch.abs(stft)
        mask = model(torch.log1p(mag).unsqueeze(1), enrollment).squeeze(1)
        separated = torch.istft(
            stft * mask,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            length=mixtures.shape[1],
        )
    return separated.detach().cpu().numpy().astype(np.float32)


def _write_embeddings(
    rows: Sequence[MaskRow],
    *,
    model: EnrollmentTargetMaskNet,
    centroids: Mapping[str, np.ndarray],
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
    n_fft: int,
    hop_length: int,
    device: str,
) -> None:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    titanet = _load_titanet(device)
    samples = int(round(window_seconds * sample_rate))

    all_embeddings: List[np.ndarray] = []
    all_windows: List[str] = []
    all_indices: List[int] = []
    all_truths: List[str] = []
    all_shares: List[float] = []
    all_active: List[int] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)
        waves: List[np.ndarray] = []
        labels: List[str] = []
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            waves.append(_slice_wave(mixed, start_sample, start_sample + samples))
            labels.append(row.truth)
        embeddings: List[np.ndarray] = []
        for offset in range(0, len(waves), batch_size):
            enhanced = _separate_batch(
                model,
                np.stack(waves[offset : offset + batch_size]),
                labels[offset : offset + batch_size],
                centroids=centroids,
                n_fft=n_fft,
                hop_length=hop_length,
                device=device,
            )
            embeddings.append(
                _embed_waveforms(
                    titanet,
                    list(enhanced),
                    sample_rate=sample_rate,
                    batch_size=batch_size,
                    device=device,
                )
            )
        all_embeddings.append(np.vstack(embeddings).astype(np.float32))
        all_windows.extend([window_name] * len(window_rows))
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        print(f"embedded {window_name}: {len(window_rows)} target-extractor words", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.vstack(all_embeddings).astype(np.float32),
        windows=np.asarray(all_windows),
        indices=np.asarray(all_indices, dtype=np.int32),
        truths=np.asarray(all_truths),
        target_shares=np.asarray(all_shares, dtype=np.float32),
        active_5pct=np.asarray(all_active, dtype=np.int16),
    )


def _load_embeddings(path: Path) -> Tuple[np.ndarray, List[MaskRow]]:
    payload = np.load(path, allow_pickle=False)
    rows = [
        MaskRow(
            window=str(window),
            index=int(index),
            truth=str(truth),
            target_file="",
            target_share=float(share),
            active_5pct=int(active),
        )
        for window, index, truth, share, active in zip(
            payload["windows"].tolist(),
            payload["indices"].tolist(),
            payload["truths"].tolist(),
            payload["target_shares"].tolist(),
            payload["active_5pct"].tolist(),
        )
    ]
    return np.asarray(payload["embeddings"], dtype=np.float32), rows


def _mixed_same_rows(
    rows_by_window: Mapping[str, Sequence[MaskRow]],
    *,
    titanet_cache_root: Path,
    window_seconds: float,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for window_name, rows in sorted(rows_by_window.items()):
        payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        for row in rows:
            vectors.append(embeddings[row.index])
    return np.stack(vectors).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an enrollment-conditioned target-speaker mask on real Discord stems."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--quality-records",
        type=Path,
        default=Path(
            ".outputs/speaker_id_baseline_prod_graph/artifacts/bank/"
            "1e2a7a014a23be63/dataset/quality_records.jsonl"
        ),
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/prod/Audio"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/tmp/codex_target_extractor_cache")
    )
    parser.add_argument(
        "--training-cache", type=Path, default=Path("/tmp/codex_target_extractor_real_stems.npz")
    )
    parser.add_argument("--model-output", type=Path, default=Path("/tmp/codex_target_extractor.pt"))
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_target_extractor_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_target_extractor_results.json")
    )
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument(
        "--train-sessions",
        default="Session 50,Session 51,Session 52,Session 53,Session 54,Session 55,"
        "Session 56,Session 57,Session 58,Session 59,Session 60",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--max-crops-per-speaker", type=int, default=120)
    parser.add_argument("--min-interferer-ratio", type=float, default=0.08)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-target-snr-db", type=float, default=-10.0)
    parser.add_argument("--max-target-snr-db", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    centroids = _speaker_centroids(args.clean_bank.expanduser(), CORE_SPEAKERS)
    targets, interferers, labels = _load_or_create_real_stems(args)
    counts = Counter(labels)
    missing = [speaker for speaker in CORE_SPEAKERS if counts.get(speaker, 0) < 2]
    if missing:
        raise RuntimeError(f"Not enough real-stem crops for {missing}: {dict(counts)}")
    print(f"loaded_real_stem_crops {dict(counts)}", flush=True)

    model = _train_model(
        targets,
        interferers,
        labels,
        centroids=centroids,
        args=args,
        device=device,
    )

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    if not args.embedding_output.expanduser().exists():
        _write_embeddings(
            rows,
            model=model,
            centroids=centroids,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.embedding_output.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            batch_size=int(args.batch_size),
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            device=device,
        )

    embeddings, embedding_rows = _load_embeddings(args.embedding_output.expanduser())
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in embedding_rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    result = evaluate_mask_embeddings(
        embeddings,
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
    )
    result.pop("predictions", None)
    mixed_result = evaluate_mask_embeddings(
        _mixed_same_rows(
            rows_by_window,
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
        ),
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name="lda_shrinkage",
    )
    mixed_result.pop("predictions", None)
    payload = {
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "training_counts": dict(counts),
        "train_steps": int(args.train_steps),
        "conditioning": "clean_bank_centroid",
        "true_target_conditioned": True,
        "target_extractor": result,
        "mixed_same_rows": mixed_result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    print(
        ",".join(
            [
                "target_extractor_true_target/lda_shrinkage",
                str(result["direct"]["examples"]),
                f"{float(result['direct']['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )
    print(
        ",".join(
            [
                "mixed_same_rows/lda_shrinkage",
                str(mixed_result["direct"]["examples"]),
                f"{float(mixed_result['direct']['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
