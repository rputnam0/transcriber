from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import _load_audio, _load_titanet_word_npz  # noqa: E402
from speaker_id_wavlm_embedding_sweep import (  # noqa: E402
    _evaluate,
    _load_clean,
    _load_embedding_rows,
    _load_rows,
    _normalize_rows,
    _normalize_wave_batch,
    _rows_by_window,
    _slice_wave,
)


def _load_wavlm_hidden(*, model_name: str, device: str):
    from transformers import WavLMForXVector

    model = WavLMForXVector.from_pretrained(model_name).to(device)
    model.eval()
    return model


def _pool_hidden(hidden_states: Sequence[torch.Tensor], *, pooling: str) -> torch.Tensor:
    if pooling == "last_mean":
        hidden = hidden_states[-1]
        return hidden.mean(dim=1)
    if pooling == "last_mean_std":
        hidden = hidden_states[-1]
    elif pooling == "last4_mean":
        hidden = torch.stack(list(hidden_states[-4:]), dim=0).mean(dim=0)
        return hidden.mean(dim=1)
    elif pooling == "last4_mean_std":
        hidden = torch.stack(list(hidden_states[-4:]), dim=0).mean(dim=0)
    else:
        raise ValueError(pooling)
    mean = hidden.mean(dim=1)
    std = hidden.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=1)


def _embed_waveforms_wavlm_hidden(
    model,
    waves: Sequence[np.ndarray],
    *,
    pooling: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for offset in range(0, len(waves), batch_size):
        batch = [
            np.asarray(wave, dtype=np.float32).flatten()
            for wave in waves[offset : offset + batch_size]
        ]
        max_len = max(wave.shape[0] for wave in batch)
        wave_batch = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
        for row, wave in enumerate(batch):
            wave_batch[row, : wave.shape[0]] = torch.from_numpy(wave).to(device)
        with torch.inference_mode():
            output = model(_normalize_wave_batch(wave_batch), output_hidden_states=True)
            pooled = _pool_hidden(output.hidden_states, pooling=pooling)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        del wave_batch
        del output
        del pooled
    return _normalize_rows(np.vstack(vectors).astype(np.float32))


def _write_mixed_embeddings(
    rows,
    *,
    model,
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
    pooling: str,
    device: str,
) -> None:
    samples = int(round(window_seconds * sample_rate))
    all_embeddings: List[np.ndarray] = []
    all_windows: List[str] = []
    all_indices: List[int] = []
    all_truths: List[str] = []
    all_shares: List[float] = []
    all_active: List[int] = []
    all_mixed_pred: List[str] = []

    for window_name, window_rows in sorted(_rows_by_window(rows).items()):
        word_payload = _load_titanet_word_npz(
            titanet_cache_root, "reference", window_name, window_seconds
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)
        waves: List[np.ndarray] = []
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            waves.append(_slice_wave(mixed, start_sample, start_sample + samples))
        all_embeddings.append(
            _embed_waveforms_wavlm_hidden(
                model, waves, pooling=pooling, batch_size=batch_size, device=device
            )
        )
        all_windows.extend([row.window for row in window_rows])
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        all_mixed_pred.extend([row.mixed_pred for row in window_rows])
        print(f"embedded_mixed_hidden {window_name}: {len(window_rows)} rows", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.vstack(all_embeddings).astype(np.float32),
        windows=np.asarray(all_windows),
        indices=np.asarray(all_indices, dtype=np.int32),
        truths=np.asarray(all_truths),
        target_shares=np.asarray(all_shares, dtype=np.float32),
        active_5pct=np.asarray(all_active, dtype=np.int16),
        mixed_pred=np.asarray(all_mixed_pred),
    )


def _write_clean_embeddings(
    *,
    model,
    training_cache: Path,
    output_path: Path,
    pooling: str,
    batch_size: int,
    device: str,
) -> None:
    payload = np.load(training_cache, allow_pickle=False)
    targets = np.asarray(payload["targets"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    embeddings = _embed_waveforms_wavlm_hidden(
        model, list(targets), pooling=pooling, batch_size=batch_size, device=device
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        labels=np.asarray(labels),
        source_training_cache=str(training_cache),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen WavLM hidden-state features for word speaker attribution."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--mixed-embedding-cache",
        type=Path,
        default=Path("/tmp/codex_wavlm_hidden_mixed.npz"),
    )
    parser.add_argument(
        "--clean-embedding-cache",
        type=Path,
        default=Path("/tmp/codex_wavlm_hidden_clean.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/codex_wavlm_hidden_sweep.json"))
    parser.add_argument(
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--pooling",
        choices=("last_mean", "last_mean_std", "last4_mean", "last4_mean_std"),
        default="last4_mean_std",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument(
        "--models",
        default="centroid_cosine,knn7_cosine,lda_shrinkage,linear_svc,logreg_balanced",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rows, _mixed_by_key = _load_rows(
        args.dominance_json.expanduser(), max_target_share=float(args.max_target_share)
    )
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    model = None
    if (
        not args.clean_embedding_cache.expanduser().exists()
        or not args.mixed_embedding_cache.expanduser().exists()
    ):
        model = _load_wavlm_hidden(model_name=str(args.model_name), device=device)
    if not args.clean_embedding_cache.expanduser().exists():
        _write_clean_embeddings(
            model=model,
            training_cache=args.training_cache.expanduser(),
            output_path=args.clean_embedding_cache.expanduser(),
            pooling=str(args.pooling),
            batch_size=int(args.batch_size),
            device=device,
        )
    if not args.mixed_embedding_cache.expanduser().exists():
        if model is None:
            model = _load_wavlm_hidden(model_name=str(args.model_name), device=device)
        _write_mixed_embeddings(
            rows,
            model=model,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.mixed_embedding_cache.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            batch_size=int(args.batch_size),
            pooling=str(args.pooling),
            device=device,
        )

    embeddings, embedding_rows = _load_embedding_rows(args.mixed_embedding_cache.expanduser())
    clean_embeddings, clean_labels = _load_clean(args.clean_embedding_cache.expanduser())
    model_names = [item.strip() for item in str(args.models).split(",") if item.strip()]
    scores = _evaluate(
        embedding_rows,
        embeddings,
        clean_embeddings=clean_embeddings,
        clean_labels=clean_labels,
        model_names=model_names,
    )
    payload = {
        "model": "wavlm_hidden_state_sweep",
        "model_name": str(args.model_name),
        "pooling": str(args.pooling),
        "mixed_embedding_cache": str(args.mixed_embedding_cache.expanduser()),
        "clean_embedding_cache": str(args.clean_embedding_cache.expanduser()),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "models": model_names,
        "scores": scores,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("slice,name,examples,accuracy", flush=True)
    for slice_name, slice_scores in scores.items():
        for name, score in sorted(slice_scores.items()):
            direct = score["direct"]
            print(
                ",".join(
                    [
                        slice_name,
                        name,
                        str(direct["examples"]),
                        f"{float(direct['accuracy']):.4f}",
                    ]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
