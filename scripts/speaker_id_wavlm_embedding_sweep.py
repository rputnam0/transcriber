from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_architecture_sweep import _fit_predict  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    _load_audio,
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
)
from speaker_id_word_window_sweep import _window_group  # noqa: E402


@dataclass(frozen=True)
class SpeakerRow:
    window: str
    index: int
    truth: str
    target_share: float
    active_5pct: int
    mixed_pred: str


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _load_wavlm(*, device: str):
    from transformers import AutoFeatureExtractor, WavLMForXVector

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
    model.eval()
    return feature_extractor, model


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _normalize_wave_batch(wave_batch: torch.Tensor) -> torch.Tensor:
    centered = wave_batch - wave_batch.mean(dim=1, keepdim=True)
    scale = torch.sqrt(torch.mean(centered * centered, dim=1, keepdim=True).clamp_min(1e-8))
    return centered / scale


def _embed_waveforms_wavlm(
    model,
    waves: Sequence[np.ndarray],
    *,
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
            output = model(_normalize_wave_batch(wave_batch))
        embeddings = output.embeddings.detach().cpu().numpy().astype(np.float32)
        vectors.append(_normalize_rows(embeddings))
        del wave_batch
        del output
    return np.vstack(vectors).astype(np.float32)


def _load_rows(
    path: Path, *, max_target_share: float
) -> Tuple[List[SpeakerRow], Dict[Tuple[str, int], str]]:
    raw_rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    rows: List[SpeakerRow] = []
    mixed_pred_by_key: Dict[Tuple[str, int], str] = {}
    for raw in raw_rows:
        window = str(raw["window"])
        index = int(raw["index"])
        mixed_pred = str(raw.get("mixed_pred") or "unknown")
        mixed_pred_by_key[(window, index)] = mixed_pred
        share = float(raw.get("target_share") or 0.0)
        if share > max_target_share:
            continue
        rows.append(
            SpeakerRow(
                window=window,
                index=index,
                truth=str(raw["truth"]),
                target_share=share,
                active_5pct=int(raw.get("active_5pct") or 0),
                mixed_pred=mixed_pred,
            )
        )
    return rows, mixed_pred_by_key


def _rows_by_window(rows: Sequence[SpeakerRow]) -> Dict[str, List[SpeakerRow]]:
    by_window: Dict[str, List[SpeakerRow]] = defaultdict(list)
    for row in rows:
        by_window[row.window].append(row)
    return by_window


def _write_mixed_embeddings(
    rows: Sequence[SpeakerRow],
    *,
    model,
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
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
            _embed_waveforms_wavlm(model, waves, batch_size=batch_size, device=device)
        )
        all_windows.extend([row.window for row in window_rows])
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        all_mixed_pred.extend([row.mixed_pred for row in window_rows])
        print(f"embedded_mixed {window_name}: {len(window_rows)} rows", flush=True)

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
    batch_size: int,
    device: str,
) -> None:
    payload = np.load(training_cache, allow_pickle=False)
    targets = np.asarray(payload["targets"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    embeddings = _embed_waveforms_wavlm(model, list(targets), batch_size=batch_size, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        labels=np.asarray(labels),
        source_training_cache=str(training_cache),
    )


def _load_embedding_rows(path: Path) -> Tuple[np.ndarray, List[SpeakerRow]]:
    payload = np.load(path, allow_pickle=False)
    rows = [
        SpeakerRow(
            window=str(window),
            index=int(index),
            truth=str(truth),
            target_share=float(share),
            active_5pct=int(active),
            mixed_pred=str(mixed_pred),
        )
        for window, index, truth, share, active, mixed_pred in zip(
            payload["windows"].tolist(),
            payload["indices"].tolist(),
            payload["truths"].tolist(),
            payload["target_shares"].tolist(),
            payload["active_5pct"].tolist(),
            payload["mixed_pred"].tolist(),
        )
    ]
    return np.asarray(payload["embeddings"], dtype=np.float32), rows


def _load_clean(path: Path) -> Tuple[np.ndarray, List[str]]:
    payload = np.load(path, allow_pickle=False)
    return (
        np.asarray(payload["embeddings"], dtype=np.float32),
        [str(item) for item in payload["labels"].tolist()],
    )


def _score_subset(
    name: str,
    rows: Sequence[SpeakerRow],
    predictions: Sequence[str],
) -> Dict[str, object]:
    return {
        "name": name,
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
    }


def _evaluate(
    rows: Sequence[SpeakerRow],
    embeddings: np.ndarray,
    *,
    clean_embeddings: np.ndarray,
    clean_labels: Sequence[str],
    model_names: Sequence[str],
) -> Dict[str, object]:
    groups = sorted({_window_group(row.window) for row in rows})
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group_index[_window_group(row.window)].append(index)
    predictions: Dict[str, List[str]] = {
        "titanet_mixed_baseline_from_dominance": [row.mixed_pred for row in rows],
    }
    for model_name in model_names:
        predictions[f"wavlm_clean_plus_mixed_lgo/{model_name}"] = ["unknown"] * len(rows)

    for group in groups:
        test_indices = by_group_index[group]
        train_indices = [
            index for index, row in enumerate(rows) if _window_group(row.window) != group
        ]
        train_x = np.vstack([clean_embeddings, embeddings[train_indices]]).astype(np.float32)
        train_y = list(clean_labels) + [rows[index].truth for index in train_indices]
        test_x = embeddings[test_indices]
        for model_name in model_names:
            predicted = _fit_predict(model_name, train_x, train_y, test_x)
            labels = predictions[f"wavlm_clean_plus_mixed_lgo/{model_name}"]
            for global_idx, label in zip(test_indices, predicted):
                labels[global_idx] = label

    scores = {
        name: _score_subset(name, rows, labels) for name, labels in sorted(predictions.items())
    }
    hard_indices = [index for index, row in enumerate(rows) if row.target_share <= 0.90]
    hard_rows = [rows[index] for index in hard_indices]
    hard_scores = {
        name: _score_subset(name, hard_rows, [labels[index] for index in hard_indices])
        for name, labels in sorted(predictions.items())
    }
    return {
        "all_rows": scores,
        "hard_rows": hard_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WavLM speaker-verification x-vector embeddings on flattened speaker-ID rows."
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
        default=Path("/tmp/codex_wavlm_xvector_mixed_all_rows.npz"),
    )
    parser.add_argument(
        "--clean-embedding-cache",
        type=Path,
        default=Path("/tmp/codex_wavlm_xvector_clean_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_wavlm_xvector_embedding_sweep.json"),
    )
    parser.add_argument(
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-target-share", type=float, default=1.01)
    parser.add_argument(
        "--models",
        default="centroid_cosine,knn7_cosine,lda_shrinkage,linear_svc,logreg_balanced",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rows, _mixed_by_key = _load_rows(
        args.dominance_json.expanduser(),
        max_target_share=float(args.max_target_share),
    )
    wavlm_model = None
    if (
        not args.clean_embedding_cache.expanduser().exists()
        or not args.mixed_embedding_cache.expanduser().exists()
    ):
        _feature_extractor, wavlm_model = _load_wavlm(device=device)
    if not args.clean_embedding_cache.expanduser().exists():
        _write_clean_embeddings(
            model=wavlm_model,
            training_cache=args.training_cache.expanduser(),
            output_path=args.clean_embedding_cache.expanduser(),
            batch_size=int(args.batch_size),
            device=device,
        )
    if not args.mixed_embedding_cache.expanduser().exists():
        if wavlm_model is None:
            _feature_extractor, wavlm_model = _load_wavlm(device=device)
        _write_mixed_embeddings(
            rows,
            model=wavlm_model,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.mixed_embedding_cache.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            batch_size=int(args.batch_size),
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
        "model": "wavlm_xvector_embedding_sweep",
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
