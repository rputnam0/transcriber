from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _load_titanet_word_npz,
    _safe_name,
    _window_group,
    evaluate_mask_embeddings,
)


CORE_SPEAKERS = (
    "Dungeon Master",
    "David Tanglethorn",
    "Leopold Magnus",
    "Kaladen Shash",
    "Cyrus Schwert",
    "Cletus Cobbington",
)

SPEAKER_HANDLES = {
    "Dungeon Master": "joeeeenathan",
    "David Tanglethorn": "jessev567890",
    "Leopold Magnus": "traceritops",
    "Kaladen Shash": "kinglizard7958",
    "Cyrus Schwert": "travisaurus6985",
    "Cletus Cobbington": "bfschmity",
}


@dataclass(frozen=True)
class CleanRecord:
    session: str
    speaker: str
    start: float
    end: float


class TinyTargetMaskNet(nn.Module):
    def __init__(self, speakers: Sequence[str]) -> None:
        super().__init__()
        self.speakers = tuple(speakers)
        channels = 1 + len(self.speakers)
        layers: List[nn.Module] = [
            nn.Conv2d(channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        ]
        for dilation in (1, 2, 4, 8, 16):
            layers.extend(
                [
                    nn.Conv2d(
                        32,
                        32,
                        kernel_size=(3, 5),
                        padding=(1, 2 * dilation),
                        dilation=(1, dilation),
                    ),
                    nn.GroupNorm(4, 32),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(32, 24, kernel_size=3, padding=1),
                nn.GroupNorm(4, 24),
                nn.SiLU(),
                nn.Conv2d(24, 1, kernel_size=1),
                nn.Sigmoid(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, logmag: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        batch, _channel, freq, frames = logmag.shape
        one_hot = torch.zeros(
            (batch, len(self.speakers), freq, frames),
            dtype=logmag.dtype,
            device=logmag.device,
        )
        one_hot.scatter_(1, speaker_ids.view(batch, 1, 1, 1).expand(-1, 1, freq, frames), 1.0)
        return self.net(torch.cat([logmag, one_hot], dim=1))


def _load_clean_records(
    path: Path,
    *,
    sessions: Sequence[str],
    speakers: Sequence[str],
    min_duration: float,
) -> Dict[str, List[CleanRecord]]:
    session_set = {str(item).strip() for item in sessions if str(item).strip()}
    speaker_set = set(speakers)
    records: Dict[str, List[CleanRecord]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            session = str(raw.get("session") or "").strip()
            speaker = str(raw.get("speaker") or "").strip()
            if session not in session_set or speaker not in speaker_set:
                continue
            if raw.get("qa_rejection") is not None or not bool(raw.get("decode_ok", True)):
                continue
            start = float(raw.get("start") or 0.0)
            end = float(raw.get("end") or 0.0)
            if end - start < min_duration:
                continue
            records[speaker].append(
                CleanRecord(session=session, speaker=speaker, start=start, end=end)
            )
    return records


def _extract_session(zip_path: Path, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path for path in output_dir.rglob("*") if path.suffix.lower() in {".ogg", ".wav"}
    )
    if existing:
        return existing
    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    return sorted(path for path in output_dir.rglob("*") if path.suffix.lower() in {".ogg", ".wav"})


def _session_stems(
    session: str,
    *,
    audio_root: Path,
    stem_cache: Path,
) -> Tuple[Mapping[str, Path], List[Path]]:
    zip_path = audio_root / f"{session}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    paths = _extract_session(zip_path, stem_cache / _safe_name(session))
    by_speaker: Dict[str, Path] = {}
    for speaker, handle in SPEAKER_HANDLES.items():
        for path in paths:
            if handle in path.name:
                by_speaker[speaker] = path
                break
    return by_speaker, paths


def _load_audio_segment(path: Path, start: float, duration: float, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    wave = np.frombuffer(result.stdout, dtype=np.float32)
    samples = int(round(duration * sample_rate))
    if wave.shape[0] < samples:
        wave = np.pad(wave, (0, samples - wave.shape[0]))
    return wave[:samples].astype(np.float32)


def _wave_rms(wave: np.ndarray) -> float:
    if wave.size == 0:
        return 0.0
    values = np.asarray(wave, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def _best_training_segment(
    paths: Sequence[Path],
    *,
    preferred_path: Path | None,
    start: float,
    duration: float,
    sample_rate: int,
    samples: int,
) -> Tuple[np.ndarray | None, Path | None, float]:
    candidates: List[Path] = []
    if preferred_path is not None:
        candidates.append(preferred_path)
    candidates.extend(path for path in paths if path not in candidates)

    best_wave: np.ndarray | None = None
    best_path: Path | None = None
    best_rms = -1.0
    for path in candidates:
        try:
            wave = _load_audio_segment(path, start, duration, sample_rate)
        except subprocess.CalledProcessError:
            continue
        if wave.shape[0] != samples or not np.isfinite(wave).all():
            continue
        rms = _wave_rms(wave)
        if rms > best_rms:
            best_wave = wave
            best_path = path
            best_rms = rms
    return best_wave, best_path, best_rms


def _make_snippet_cache(
    *,
    records_by_speaker: Mapping[str, Sequence[CleanRecord]],
    audio_root: Path,
    cache_root: Path,
    output_path: Path,
    sample_rate: int,
    duration: float,
    max_per_speaker: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    waves: List[np.ndarray] = []
    labels: List[str] = []
    sessions: List[str] = []
    stem_cache = cache_root / "stems"
    session_cache: Dict[str, Tuple[Mapping[str, Path], List[Path]]] = {}
    samples = int(round(duration * sample_rate))

    for speaker in CORE_SPEAKERS:
        candidates = list(records_by_speaker.get(speaker) or [])
        rng.shuffle(candidates)
        speaker_count = 0
        for record in candidates:
            if speaker_count >= max_per_speaker:
                break
            if record.session not in session_cache:
                session_cache[record.session] = _session_stems(
                    record.session,
                    audio_root=audio_root,
                    stem_cache=stem_cache,
                )
            span = max(record.end - record.start - duration, 0.0)
            offset = rng.uniform(0.0, span) if span > 0.0 else 0.0
            by_speaker, paths = session_cache[record.session]
            wave, _chosen_path, rms = _best_training_segment(
                paths,
                preferred_path=by_speaker.get(speaker),
                start=record.start + offset,
                duration=duration,
                sample_rate=sample_rate,
                samples=samples,
            )
            if wave is None:
                continue
            if rms < 1e-4:
                continue
            waves.append(wave)
            labels.append(speaker)
            sessions.append(record.session)
            speaker_count += 1
        print(f"snippets {speaker}: {speaker_count}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        waves=np.stack(waves).astype(np.float32),
        labels=np.asarray(labels),
        sessions=np.asarray(sessions),
    )


def _load_or_create_snippets(args: argparse.Namespace) -> Tuple[np.ndarray, List[str]]:
    snippet_path = args.snippet_cache.expanduser()
    if not snippet_path.exists():
        sessions = [item.strip() for item in str(args.train_sessions).split(",") if item.strip()]
        records_by_speaker = _load_clean_records(
            args.quality_records.expanduser(),
            sessions=sessions,
            speakers=CORE_SPEAKERS,
            min_duration=float(args.snippet_seconds),
        )
        _make_snippet_cache(
            records_by_speaker=records_by_speaker,
            audio_root=args.audio_root.expanduser(),
            cache_root=args.cache_root.expanduser(),
            output_path=snippet_path,
            sample_rate=int(args.sample_rate),
            duration=float(args.snippet_seconds),
            max_per_speaker=int(args.max_snippets_per_speaker),
            seed=int(args.seed),
        )
    payload = np.load(snippet_path, allow_pickle=False)
    return np.asarray(payload["waves"], dtype=np.float32), [
        str(item) for item in payload["labels"].tolist()
    ]


def _normalize_wave(wave: np.ndarray) -> np.ndarray:
    rms = _wave_rms(wave)
    if rms <= 1e-6:
        return wave.astype(np.float32)
    return (wave / rms).astype(np.float32)


def _limit_rows_by_group(
    rows: Sequence[MaskRow],
    *,
    limit: int,
    seed: int,
) -> List[MaskRow]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)

    rng = random.Random(seed)
    by_group: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        by_group[_window_group(row.window)].append(row)

    selected: List[MaskRow] = []
    groups = sorted(by_group)
    quota = max(1, math.ceil(limit / max(len(groups), 1)))
    for group in groups:
        group_rows = list(by_group[group])
        rng.shuffle(group_rows)
        selected.extend(group_rows[:quota])

    if len(selected) > limit:
        rng.shuffle(selected)
        selected = selected[:limit]
    selected.sort(key=lambda row: (row.window, row.index, row.truth))
    return selected


def _synth_batch(
    waves: np.ndarray,
    labels: Sequence[str],
    *,
    batch_size: int,
    speaker_order: Sequence[str],
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    by_speaker: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_speaker[str(label)].append(index)

    mixtures: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    target_labels: List[str] = []
    source_sets: List[np.ndarray] = []
    source_counts: List[int] = []
    max_sources = 3
    for _ in range(batch_size):
        target_speaker = rng.choice(list(speaker_order))
        target_index = rng.choice(by_speaker[target_speaker])
        target = _normalize_wave(waves[target_index])
        sources = [target]
        interferer_count = 1 if rng.random() < 0.75 else 2
        interferer_speakers = [speaker for speaker in speaker_order if speaker != target_speaker]
        rng.shuffle(interferer_speakers)
        for speaker in interferer_speakers[:interferer_count]:
            interferer = _normalize_wave(waves[rng.choice(by_speaker[speaker])])
            snr_db = rng.uniform(-8.0, 8.0)
            sources.append(interferer * (10.0 ** (-snr_db / 20.0)))
        mixture = np.sum(np.stack(sources), axis=0)
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.95:
            mixture = mixture * (0.95 / peak)
            sources = [source * (0.95 / peak) for source in sources]
            target = sources[0]
        padded_sources = np.zeros((max_sources, waves.shape[1]), dtype=np.float32)
        for index, source in enumerate(sources):
            padded_sources[index] = source.astype(np.float32)
        mixtures.append(mixture.astype(np.float32))
        targets.append(target.astype(np.float32))
        target_labels.append(target_speaker)
        source_sets.append(padded_sources)
        source_counts.append(len(sources))
    return (
        np.stack(mixtures),
        np.stack(targets),
        target_labels,
        np.stack(source_sets),
        np.asarray(source_counts, dtype=np.int64),
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


def _train_model(
    waves: np.ndarray,
    labels: Sequence[str],
    *,
    model_path: Path,
    steps: int,
    batch_size: int,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    seed: int,
    device: str,
) -> TinyTargetMaskNet:
    del sample_rate
    model = TinyTargetMaskNet(CORE_SPEAKERS).to(device)
    if model_path.exists():
        payload = torch.load(model_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    window = torch.hann_window(n_fft, device=device)
    speaker_to_id = {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}
    model.train()
    for step in range(1, steps + 1):
        mixture_np, _target_np, target_labels, sources_np, counts_np = _synth_batch(
            waves,
            labels,
            batch_size=batch_size,
            speaker_order=CORE_SPEAKERS,
            rng=rng,
        )
        mixture = torch.from_numpy(mixture_np).to(device)
        sources = torch.from_numpy(sources_np).to(device)
        counts = torch.from_numpy(counts_np).to(device)
        speaker_ids = torch.tensor(
            [speaker_to_id[label] for label in target_labels],
            dtype=torch.long,
            device=device,
        )
        mix_mag = _stft_mag(mixture, n_fft=n_fft, hop_length=hop_length, window=window)
        batch, max_sources, samples = sources.shape
        source_mag = _stft_mag(
            sources.reshape(batch * max_sources, samples),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        ).reshape(batch, max_sources, mix_mag.shape[-2], mix_mag.shape[-1])
        active_sources = torch.arange(max_sources, device=device).view(
            1, max_sources, 1, 1
        ) < counts.view(batch, 1, 1, 1)
        source_mag_sum = (source_mag * active_sources).sum(dim=1).clamp_min(1e-5)
        target_mag = source_mag[:, 0]
        true_mask = (target_mag / source_mag_sum).clamp(0.0, 1.0).unsqueeze(1)
        logmag = torch.log1p(mix_mag).unsqueeze(1)
        predicted = model(logmag, speaker_ids)
        mask_loss = torch.nn.functional.mse_loss(predicted, true_mask)
        recon_loss = torch.nn.functional.mse_loss(
            torch.log1p(predicted.squeeze(1) * mix_mag),
            torch.log1p(target_mag),
        )
        loss = mask_loss + 0.25 * recon_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            print(
                "train_step "
                f"{step}/{steps} loss={float(loss.item()):.5f} "
                f"mask={float(mask_loss.item()):.5f} recon={float(recon_loss.item()):.5f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "speakers": CORE_SPEAKERS,
            "n_fft": n_fft,
            "hop_length": hop_length,
        },
        model_path,
    )
    return model


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _separate_true_target_batch(
    model: TinyTargetMaskNet,
    mixtures: np.ndarray,
    target_labels: Sequence[str],
    *,
    n_fft: int,
    hop_length: int,
    device: str,
) -> np.ndarray:
    speaker_to_id = {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}
    window = torch.hann_window(n_fft, device=device)
    mixture = torch.from_numpy(np.asarray(mixtures, dtype=np.float32)).to(device)
    speaker_ids = torch.tensor(
        [speaker_to_id[label] for label in target_labels],
        dtype=torch.long,
        device=device,
    )
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
        mask = model(torch.log1p(mag).unsqueeze(1), speaker_ids).squeeze(1)
        separated = torch.istft(
            stft * mask,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            length=mixtures.shape[1],
        )
    return separated.detach().cpu().numpy().astype(np.float32)


def _write_learned_embeddings(
    rows: Sequence[MaskRow],
    *,
    model: TinyTargetMaskNet,
    prepared_root: Path,
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

    all_embeddings: List[np.ndarray] = []
    all_windows: List[str] = []
    all_indices: List[int] = []
    all_truths: List[str] = []
    all_shares: List[float] = []
    all_active: List[int] = []
    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = _load_titanet_word_npz(
            Path("/tmp/codex_titanet_word_window"),
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)
        separated_waves: List[np.ndarray] = []
        separated_labels: List[str] = []
        samples = int(round(window_seconds * sample_rate))
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            separated_waves.append(_slice_wave(mixed, start_sample, start_sample + samples))
            separated_labels.append(row.truth)

        embeddings: List[np.ndarray] = []
        for offset in range(0, len(separated_waves), batch_size):
            batch_waves = np.stack(separated_waves[offset : offset + batch_size])
            batch_labels = separated_labels[offset : offset + batch_size]
            enhanced = _separate_true_target_batch(
                model,
                batch_waves,
                batch_labels,
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
        window_embeddings = np.vstack(embeddings).astype(np.float32)
        all_embeddings.append(window_embeddings)
        all_windows.extend([window_name] * len(window_rows))
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        print(f"embedded {window_name}: {len(window_rows)} learned-mask words", flush=True)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tiny target-conditioned mask model for overlap speaker-ID smokes."
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
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/codex_learned_mask_cache"))
    parser.add_argument(
        "--snippet-cache", type=Path, default=Path("/tmp/codex_learned_mask_snippets.npz")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("/tmp/codex_learned_mask_model.pt")
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_learned_mask_embeddings_true_target.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/codex_learned_mask_results.json"))
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
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--train-sessions",
        default="Session 50,Session 51,Session 52,Session 53,Session 54,Session 55,"
        "Session 56,Session 57,Session 58,Session 59,Session 60",
    )
    parser.add_argument("--snippet-seconds", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--max-snippets-per-speaker", type=int, default=120)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    waves, labels = _load_or_create_snippets(args)
    counts = Counter(labels)
    missing = [speaker for speaker in CORE_SPEAKERS if counts.get(speaker, 0) < 2]
    if missing:
        raise RuntimeError(f"Not enough snippets for {missing}: {dict(counts)}")
    print(f"loaded_snippets {dict(counts)}", flush=True)

    model = _train_model(
        waves,
        labels,
        model_path=args.model_output.expanduser(),
        steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        sample_rate=int(args.sample_rate),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        seed=int(args.seed),
        device=device,
    )

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    if int(args.eval_limit) > 0:
        rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    if not args.embedding_output.expanduser().exists():
        _write_learned_embeddings(
            rows,
            model=model,
            prepared_root=args.prepared_root.expanduser(),
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
    payload = {
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "snippet_counts": dict(counts),
        "train_sessions": [
            item.strip() for item in str(args.train_sessions).split(",") if item.strip()
        ],
        "train_steps": int(args.train_steps),
        "true_target_conditioned": True,
        "result": result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "name,examples,direct_accuracy",
        flush=True,
    )
    print(
        ",".join(
            [
                "learned_mask_true_target/lda_shrinkage",
                str(result["direct"]["examples"]),
                f"{float(result['direct']['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )
    print(json.dumps(result["share_slices"], indent=2), flush=True)


if __name__ == "__main__":
    main()
