from __future__ import annotations

import argparse
import json
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

from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    ConditionedTasNetExtractor,
    _conditioning_vectors,
    _load_embeddings,
    _normalize_batch,
    _si_snr,
    _write_embeddings,
)
from speaker_id_learned_mask_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    _limit_rows_by_group,
    _normalize_wave,
)
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    evaluate_mask_embeddings,
)
from speaker_id_target_extractor_sweep import (  # noqa: E402
    _load_or_create_real_stems,
    _mixed_same_rows,
)


class LogMagSpeakerClassifier(nn.Module):
    def __init__(self, num_speakers: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=(2, 1)),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=(2, 2)),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=(2, 2)),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = nn.Linear(96, num_speakers)

    def forward(self, waves: torch.Tensor, *, n_fft: int, hop_length: int) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=waves.device)
        stft = torch.stft(
            waves,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        logmag = torch.log1p(torch.abs(stft)).unsqueeze(1)
        hidden = self.net(logmag).flatten(1)
        return self.out(hidden)


def _speaker_to_id() -> Dict[str, int]:
    return {speaker: index for index, speaker in enumerate(CORE_SPEAKERS)}


def _titanet_centroid_matrix(clean_bank_path: Path, *, device: str) -> torch.Tensor:
    payload = np.load(clean_bank_path, allow_pickle=False)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    rows: List[np.ndarray] = []
    for speaker in CORE_SPEAKERS:
        speaker_rows = embeddings[[index for index, label in enumerate(labels) if label == speaker]]
        if speaker_rows.size == 0:
            raise ValueError(f"No clean Titanet rows for {speaker}")
        centroid = speaker_rows.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        rows.append(centroid.astype(np.float32))
    return torch.from_numpy(np.vstack(rows).astype(np.float32)).to(device)


def _titanet_identity_loss(
    titanet_model,
    estimate: torch.Tensor,
    label_ids: torch.Tensor,
    centroid_matrix: torch.Tensor,
    *,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.full(
        (estimate.shape[0],),
        int(estimate.shape[1]),
        dtype=torch.long,
        device=estimate.device,
    )
    output = titanet_model(input_signal=estimate, input_signal_length=lengths)
    embedding = output[-1] if isinstance(output, tuple) else output
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    centroids = torch.nn.functional.normalize(centroid_matrix, p=2, dim=1)
    logits = (embedding @ centroids.T) / max(float(temperature), 1e-3)
    return torch.nn.functional.cross_entropy(logits, label_ids), logits


def _stft_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=estimate.device)
    estimate_mag = torch.abs(
        torch.stft(
            estimate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
    )
    target_mag = torch.abs(
        torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
    )
    return torch.nn.functional.l1_loss(torch.log1p(estimate_mag), torch.log1p(target_mag))


def _synth_real_batch_with_labels(
    targets: np.ndarray,
    interferers: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    batch_size: int,
    rng: random.Random,
    min_target_snr_db: float,
    max_target_snr_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    indices = [rng.randrange(targets.shape[0]) for _ in range(batch_size)]
    mixture_rows: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []
    interferer_rows: List[np.ndarray] = []
    enrollment_rows: List[np.ndarray] = []
    batch_labels: List[str] = []
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
        label = str(labels[index])
        mixture_rows.append(mixture.astype(np.float32))
        target_rows.append(target_aug.astype(np.float32))
        interferer_rows.append(interferer_aug.astype(np.float32))
        enrollment_rows.append(centroids[label])
        batch_labels.append(label)
    return (
        np.stack(mixture_rows).astype(np.float32),
        np.stack(target_rows).astype(np.float32),
        np.stack(interferer_rows).astype(np.float32),
        np.stack(enrollment_rows).astype(np.float32),
        batch_labels,
    )


def _train_speaker_classifier(
    targets: np.ndarray,
    labels: Sequence[str],
    *,
    args: argparse.Namespace,
    device: str,
) -> LogMagSpeakerClassifier:
    speaker_to_id = _speaker_to_id()
    model = LogMagSpeakerClassifier(len(CORE_SPEAKERS)).to(device)
    classifier_path = args.classifier_output.expanduser()
    if classifier_path.exists():
        payload = torch.load(classifier_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    label_ids = np.asarray([speaker_to_id[str(label)] for label in labels], dtype=np.int64)
    rng = random.Random(int(args.seed) + 101)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.classifier_lr), weight_decay=1e-4
    )
    model.train()
    for step in range(1, int(args.classifier_steps) + 1):
        indices = [rng.randrange(targets.shape[0]) for _ in range(int(args.batch_size))]
        waves = torch.from_numpy(np.asarray(targets[indices], dtype=np.float32)).to(device)
        label_tensor = torch.from_numpy(label_ids[indices]).to(device)
        waves_norm, _scale = _normalize_batch(waves)
        logits = model(
            waves_norm,
            n_fft=int(args.aux_n_fft),
            hop_length=int(args.aux_hop_length),
        )
        loss = torch.nn.functional.cross_entropy(logits, label_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.classifier_steps) // 5) == 0:
            with torch.inference_mode():
                acc = (torch.argmax(logits, dim=1) == label_tensor).float().mean()
            print(
                f"classifier_step {step}/{args.classifier_steps} "
                f"loss={loss.item():.5f} batch_acc={acc.item():.4f}",
                flush=True,
            )

    model.eval()
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "speakers": CORE_SPEAKERS}, classifier_path)
    return model


def _train_aux_extractor(
    targets: np.ndarray,
    interferers: np.ndarray,
    labels: Sequence[str],
    *,
    centroids: Mapping[str, np.ndarray],
    speaker_classifier: LogMagSpeakerClassifier | None,
    titanet_model,
    titanet_centroids: torch.Tensor | None,
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
    ).to(device)
    model_path = args.model_output.expanduser()
    if model_path.exists():
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    speaker_to_id = _speaker_to_id()
    rng = random.Random(int(args.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    if speaker_classifier is not None:
        for param in speaker_classifier.parameters():
            param.requires_grad_(False)
        speaker_classifier.eval()
    if titanet_model is not None:
        for param in titanet_model.parameters():
            param.requires_grad_(False)
        titanet_model.eval()
    model.train()

    for step in range(1, int(args.train_steps) + 1):
        (
            mixture_np,
            target_np,
            _interferer_np,
            enrollment_np,
            batch_labels,
        ) = _synth_real_batch_with_labels(
            targets,
            interferers,
            labels,
            centroids=centroids,
            batch_size=int(args.batch_size),
            rng=rng,
            min_target_snr_db=float(args.min_target_snr_db),
            max_target_snr_db=float(args.max_target_snr_db),
        )
        label_ids = torch.tensor(
            [speaker_to_id[label] for label in batch_labels],
            dtype=torch.long,
            device=device,
        )

        mixture = torch.from_numpy(mixture_np).to(device)
        target = torch.from_numpy(target_np).to(device)
        enrollment = torch.from_numpy(enrollment_np).to(device)
        mixture_norm, scale = _normalize_batch(mixture)
        target_norm = target / scale
        estimate = model(mixture_norm, enrollment)
        si_loss = -_si_snr(estimate, target_norm).mean()
        wav_loss = torch.nn.functional.l1_loss(estimate, target_norm)
        spectral_loss = _stft_loss(
            estimate,
            target_norm,
            n_fft=int(args.aux_n_fft),
            hop_length=int(args.aux_hop_length),
        )
        if str(args.speaker_loss_mode) == "small_cnn":
            if speaker_classifier is None:
                raise RuntimeError("small_cnn speaker loss requires speaker_classifier")
            logits = speaker_classifier(
                estimate,
                n_fft=int(args.aux_n_fft),
                hop_length=int(args.aux_hop_length),
            )
            speaker_loss = torch.nn.functional.cross_entropy(logits, label_ids)
        elif str(args.speaker_loss_mode) == "titanet":
            if titanet_model is None or titanet_centroids is None:
                raise RuntimeError("titanet speaker loss requires Titanet model and centroids")
            speaker_loss, logits = _titanet_identity_loss(
                titanet_model,
                estimate,
                label_ids,
                titanet_centroids,
                temperature=float(args.titanet_loss_temperature),
            )
        else:
            logits = torch.zeros(
                (estimate.shape[0], len(CORE_SPEAKERS)),
                dtype=estimate.dtype,
                device=estimate.device,
            )
            speaker_loss = torch.zeros((), dtype=estimate.dtype, device=estimate.device)
        loss = (
            si_loss
            + float(args.wave_loss_weight) * wav_loss
            + float(args.stft_loss_weight) * spectral_loss
            + float(args.speaker_loss_weight) * speaker_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, int(args.train_steps) // 10) == 0:
            with torch.inference_mode():
                batch_acc = (torch.argmax(logits, dim=1) == label_ids).float().mean()
            print(
                f"train_step {step}/{args.train_steps} loss={loss.item():.5f} "
                f"si={si_loss.item():.5f} wav={wav_loss.item():.5f} "
                f"stft={spectral_loss.item():.5f} spk={speaker_loss.item():.5f} "
                f"spk_acc={batch_acc.item():.4f}",
                flush=True,
            )

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "speakers": CORE_SPEAKERS,
            "conditioning": str(args.conditioning),
            "auxiliary": {
                "stft_loss_weight": float(args.stft_loss_weight),
                "speaker_loss_weight": float(args.speaker_loss_weight),
            },
        },
        model_path,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train conditioned TasNet with STFT and speaker-ID auxiliary losses."
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
        "--training-cache",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--classifier-output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_aux_speaker_classifier.pt"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_aux.pt"),
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_aux_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_conditioned_tasnet_aux_results.json")
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
    parser.add_argument("--min-interferer-ratio", type=float, default=0.0)
    parser.add_argument("--classifier-steps", type=int, default=600)
    parser.add_argument("--classifier-lr", type=float, default=8e-4)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--wave-loss-weight", type=float, default=0.05)
    parser.add_argument("--stft-loss-weight", type=float, default=0.30)
    parser.add_argument("--speaker-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--speaker-loss-mode",
        choices=("small_cnn", "titanet", "none"),
        default="small_cnn",
    )
    parser.add_argument("--titanet-loss-temperature", type=float, default=0.08)
    parser.add_argument("--aux-n-fft", type=int, default=512)
    parser.add_argument("--aux-hop-length", type=int, default=160)
    parser.add_argument("--conditioning", choices=("one_hot",), default="one_hot")
    parser.add_argument("--min-target-snr-db", type=float, default=-10.0)
    parser.add_argument("--max-target-snr-db", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
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

    classifier = (
        _train_speaker_classifier(targets, labels, args=args, device=device)
        if str(args.speaker_loss_mode) == "small_cnn"
        else None
    )
    titanet_model = _load_titanet(device) if str(args.speaker_loss_mode) == "titanet" else None
    titanet_centroids = (
        _titanet_centroid_matrix(args.clean_bank.expanduser(), device=device)
        if str(args.speaker_loss_mode) == "titanet"
        else None
    )
    model = _train_aux_extractor(
        targets,
        interferers,
        labels,
        centroids=centroids,
        speaker_classifier=classifier,
        titanet_model=titanet_model,
        titanet_centroids=titanet_centroids,
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
        "model": "conditioned_tasnet_aux_target_extractor",
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "training_counts": dict(counts),
        "train_steps": int(args.train_steps),
        "classifier_steps": int(args.classifier_steps),
        "conditioning": str(args.conditioning),
        "true_target_conditioned": True,
        "auxiliary": {
            "stft_loss_weight": float(args.stft_loss_weight),
            "speaker_loss_weight": float(args.speaker_loss_weight),
            "speaker_loss_mode": str(args.speaker_loss_mode),
        },
        "target_extractor": result,
        "mixed_same_rows": mixed_result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    print(
        ",".join(
            [
                "conditioned_tasnet_aux_true_target/lda_shrinkage",
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
