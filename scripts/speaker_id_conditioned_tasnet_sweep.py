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
import torch.nn.functional as F
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
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
from speaker_id_target_extractor_sweep import (  # noqa: E402
    _load_or_create_real_stems,
    _mixed_same_rows,
    _speaker_centroids,
    _synth_real_batch,
)


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _si_snr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        torch.sum(estimate * target, dim=-1, keepdim=True)
        / torch.sum(target * target, dim=-1, keepdim=True).clamp_min(eps)
    ) * target
    noise = estimate - projection
    ratio = torch.sum(projection * projection, dim=-1).clamp_min(eps) / torch.sum(
        noise * noise, dim=-1
    ).clamp_min(eps)
    return 10.0 * torch.log10(ratio)


def _match_length(wave: torch.Tensor, length: int) -> torch.Tensor:
    if wave.shape[-1] == length:
        return wave
    if wave.shape[-1] > length:
        return wave[..., :length]
    return torch.nn.functional.pad(wave, (0, length - wave.shape[-1]))


class FilmTcnBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.PReLU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.GroupNorm(8, channels),
        )
        self.film = nn.Linear(cond_dim, channels * 2)
        self.activation = nn.PReLU()
        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(self.pre(value))
        scale, bias = self.film(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + torch.tanh(scale).unsqueeze(-1)) + bias.unsqueeze(-1)
        hidden = self.out(self.activation(hidden))
        return value + hidden


class ConditionedTasNetExtractor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        enc_feats: int = 128,
        bottleneck: int = 128,
        cond_dim: int = 64,
        enc_kernel: int = 16,
        layers: int = 6,
        stacks: int = 2,
        mask_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.enc_kernel = int(enc_kernel)
        self.enc_stride = int(enc_kernel) // 2
        self.mask_activation = str(mask_activation)
        self.encoder = nn.Conv1d(1, enc_feats, kernel_size=enc_kernel, stride=self.enc_stride)
        self.cond = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cond_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(enc_feats, bottleneck, kernel_size=1),
            nn.GroupNorm(8, bottleneck),
            nn.PReLU(),
        )
        blocks: List[FilmTcnBlock] = []
        for _stack in range(stacks):
            for layer in range(layers):
                blocks.append(FilmTcnBlock(bottleneck, cond_dim, dilation=2**layer))
        self.blocks = nn.ModuleList(blocks)
        mask_layers: List[nn.Module] = [nn.PReLU(), nn.Conv1d(bottleneck, enc_feats, kernel_size=1)]
        if self.mask_activation == "sigmoid":
            mask_layers.append(nn.Sigmoid())
        elif self.mask_activation not in {"relu", "softplus"}:
            raise ValueError(f"Unknown mask activation: {self.mask_activation}")
        self.mask = nn.Sequential(*mask_layers)
        self.decoder = nn.ConvTranspose1d(
            enc_feats,
            1,
            kernel_size=enc_kernel,
            stride=self.enc_stride,
        )

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        length = mixture.shape[-1]
        encoded = torch.relu(self.encoder(mixture.unsqueeze(1)))
        condition = self.cond(enrollment)
        hidden = self.input_proj(encoded)
        for block in self.blocks:
            hidden = block(hidden, condition)
        mask = self.mask(hidden)
        if self.mask_activation == "relu":
            mask = torch.relu(mask)
        elif self.mask_activation == "softplus":
            mask = F.softplus(mask)
        masked = encoded * mask
        decoded = self.decoder(masked).squeeze(1)
        return _match_length(decoded, length)


def _normalize_batch(waves: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rms = torch.sqrt(torch.mean(waves * waves, dim=-1, keepdim=True).clamp_min(1e-8))
    normalized = waves / rms
    peak = torch.amax(torch.abs(normalized), dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.maximum(peak, torch.ones_like(peak))
    return normalized / scale, rms * scale


def _train_model(
    targets: np.ndarray,
    interferers: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> ConditionedTasNetExtractor:
    embedding_dim = next(iter(centroids.values())).shape[0]
    model = ConditionedTasNetExtractor(
        embedding_dim=embedding_dim,
        enc_feats=int(args.enc_feats),
        bottleneck=int(args.bottleneck),
        cond_dim=int(args.cond_dim),
        enc_kernel=int(args.enc_kernel),
        layers=int(args.layers),
        stacks=int(args.stacks),
        mask_activation=str(args.mask_activation),
    ).to(device)
    model_path = args.model_output.expanduser()
    if model_path.exists():
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    model.train()
    for step in range(1, int(args.train_steps) + 1):
        mixture_np, target_np, _interferer_np, enrollment_np = _synth_real_batch(
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
        enrollment = torch.from_numpy(enrollment_np).to(device)
        mixture_norm, scale = _normalize_batch(mixture)
        target_norm = target / scale
        estimate = model(mixture_norm, enrollment)
        si_loss = -_si_snr(estimate, target_norm).mean()
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        loss = si_loss + float(args.wave_loss_weight) * wav_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "speakers": CORE_SPEAKERS,
            "conditioning": "clean_bank_centroid",
        },
        model_path,
    )
    return model


def _conditioning_vectors(
    *,
    clean_bank_path: Path,
    mode: str,
) -> Dict[str, np.ndarray]:
    if mode == "clean_bank_centroid":
        return _speaker_centroids(clean_bank_path, CORE_SPEAKERS)
    if mode == "one_hot":
        return {
            speaker: np.eye(len(CORE_SPEAKERS), dtype=np.float32)[index]
            for index, speaker in enumerate(CORE_SPEAKERS)
        }
    raise ValueError(f"Unknown conditioning mode: {mode}")


def _extract_batch(
    model: ConditionedTasNetExtractor,
    mixtures: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    device: str,
) -> np.ndarray:
    mixture = torch.from_numpy(np.asarray(mixtures, dtype=np.float32)).to(device)
    enrollment = torch.from_numpy(np.stack([centroids[label] for label in labels])).to(device)
    with torch.inference_mode():
        mixture_norm, scale = _normalize_batch(mixture)
        estimate = model(mixture_norm, enrollment) * scale
    return estimate.detach().cpu().numpy().astype(np.float32)


def _write_embeddings(
    rows: Sequence[MaskRow],
    *,
    model: ConditionedTasNetExtractor,
    centroids: Mapping[str, np.ndarray],
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
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
            enhanced = _extract_batch(
                model,
                np.stack(waves[offset : offset + batch_size]),
                labels[offset : offset + batch_size],
                centroids=centroids,
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
        print(f"embedded {window_name}: {len(window_rows)} conditioned-tasnet words", flush=True)

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
        description="Train a time-domain enrollment-conditioned target extractor on real stems."
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
        "--cache-root", type=Path, default=Path("/tmp/codex_conditioned_tasnet_cache")
    )
    parser.add_argument(
        "--training-cache", type=Path, default=Path("/tmp/codex_conditioned_tasnet_real_stems.npz")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("/tmp/codex_conditioned_tasnet.pt")
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_conditioned_tasnet_results.json")
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
    parser.add_argument("--min-interferer-ratio", type=float, default=0.03)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--conditioning",
        choices=("clean_bank_centroid", "one_hot"),
        default="clean_bank_centroid",
    )
    parser.add_argument("--min-target-snr-db", type=float, default=-10.0)
    parser.add_argument("--max-target-snr-db", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
    parser.add_argument(
        "--mask-activation",
        choices=("sigmoid", "relu", "softplus"),
        default="sigmoid",
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    centroids = _conditioning_vectors(
        clean_bank_path=args.clean_bank.expanduser(),
        mode=str(args.conditioning),
    )
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
            batch_size=int(args.eval_batch_size),
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
        "model": "conditioned_tasnet_target_extractor",
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "training_counts": dict(counts),
        "train_steps": int(args.train_steps),
        "conditioning": str(args.conditioning),
        "true_target_conditioned": True,
        "target_extractor": result,
        "mixed_same_rows": mixed_result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    print(
        ",".join(
            [
                "conditioned_tasnet_true_target/lda_shrinkage",
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
