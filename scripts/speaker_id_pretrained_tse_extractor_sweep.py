from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import hf_hub_download

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_learned_mask_sweep import _limit_rows_by_group  # noqa: E402
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
from speaker_id_speechbrain_embedding_sweep import _load_ecapa  # noqa: E402
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402


def _normalize_row(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return (vector / max(float(np.linalg.norm(vector)), 1e-8)).astype(np.float32)


def _embed_ecapa_condition_vectors(
    model,
    waves: Sequence[np.ndarray],
    *,
    batch_size: int,
    device: str,
    normalize_embeddings: bool,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for offset in range(0, len(waves), batch_size):
        batch = [
            np.asarray(wave, dtype=np.float32).flatten()
            for wave in waves[offset : offset + batch_size]
        ]
        lengths = [wave.shape[0] for wave in batch]
        max_len = max(lengths)
        wave_batch = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
        for row, wave in enumerate(batch):
            wave_batch[row, : wave.shape[0]] = torch.from_numpy(wave).to(device)
        with torch.inference_mode():
            embeddings = model.encode_batch(wave_batch, normalize=normalize_embeddings)
        vectors.append(embeddings.detach().cpu().numpy().astype(np.float32).reshape(len(batch), -1))
        del wave_batch
        del embeddings
    return np.vstack(vectors).astype(np.float32)


def _load_or_create_ecapa_centroids(
    *,
    training_cache: Path,
    output_path: Path,
    savedir: Path,
    batch_size: int,
    device: str,
    normalize_embeddings: bool,
    normalize_centroid: bool,
) -> Dict[str, np.ndarray]:
    if output_path.exists():
        payload = np.load(output_path, allow_pickle=False)
        return {
            str(speaker): vector
            for speaker, vector in zip(payload["speakers"], payload["centroids"])
        }

    payload = np.load(training_cache, allow_pickle=False)
    targets = np.asarray(payload["targets"], dtype=np.float32)
    labels = [str(item) for item in payload["labels"].tolist()]
    model = _load_ecapa(device=device, savedir=savedir)
    embeddings = _embed_ecapa_condition_vectors(
        model,
        list(targets),
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize_embeddings,
    )

    by_speaker: Dict[str, List[np.ndarray]] = defaultdict(list)
    for label, embedding in zip(labels, embeddings):
        by_speaker[label].append(
            np.nan_to_num(np.asarray(embedding, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        )
    speakers = sorted(by_speaker)
    centroids = []
    for speaker in speakers:
        centroid = np.mean(np.stack(by_speaker[speaker]), axis=0).astype(np.float32)
        if normalize_centroid:
            centroid = _normalize_row(centroid)
        centroids.append(centroid)
    centroids_np = np.stack(centroids).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, speakers=np.asarray(speakers), centroids=centroids_np)
    return {speaker: centroids_np[index] for index, speaker in enumerate(speakers)}


def _resample_1d(
    wave: np.ndarray, *, source_rate: int, target_rate: int, device: str
) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32).flatten()
    if int(source_rate) == int(target_rate):
        return wave
    tensor = torch.from_numpy(wave).to(device).unsqueeze(0)
    with torch.inference_mode():
        resampled = torchaudio.functional.resample(
            tensor,
            orig_freq=int(source_rate),
            new_freq=int(target_rate),
        )
    return resampled.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _input_scale(wave: np.ndarray, *, mode: str) -> float:
    if mode == "none":
        return 1.0
    if mode == "peak":
        peak = max(float(np.max(np.abs(wave))), 1e-6)
        return max(peak / 0.95, 1e-6)
    if mode == "rms":
        rms = max(float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))), 1e-6)
        return max(rms / 0.05, 1e-6)
    raise ValueError(mode)


def _state_shape(shape: Sequence[object]) -> Tuple[int, ...]:
    dims: List[int] = []
    for dim in shape:
        dims.append(int(dim) if isinstance(dim, int) and dim > 0 else 1)
    return tuple(dims)


def _load_tse_session(
    *, repo_id: str, providers: Sequence[str]
) -> Tuple[ort.InferenceSession, List, str, str]:
    onnx_path = hf_hub_download(repo_id, "tse_prod_48k.onnx")
    hf_hub_download(repo_id, "tse_prod_48k.onnx.data")
    available = set(ort.get_available_providers())
    selected = [provider for provider in providers if provider in available]
    if not selected:
        selected = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=selected)
    state_inputs = sorted(
        [item for item in session.get_inputs() if item.name.startswith("state_in_")],
        key=lambda item: int(item.name.rsplit("_", 1)[-1]),
    )
    state_names = {item.name for item in state_inputs}
    non_state_inputs = [item for item in session.get_inputs() if item.name not in state_names]
    audio_candidates = [item.name for item in non_state_inputs if "audio" in item.name]
    cond_candidates = [
        item.name
        for item in non_state_inputs
        if "cond" in item.name or "embed" in item.name or "spk" in item.name
    ]
    if not audio_candidates or not cond_candidates:
        names = [item.name for item in session.get_inputs()]
        raise RuntimeError(f"Could not infer ONNX audio/condition input names from {names}")
    return session, state_inputs, audio_candidates[0], cond_candidates[0]


def _extract_with_tse(
    session: ort.InferenceSession,
    state_inputs: Sequence,
    wave_16k: np.ndarray,
    cond: np.ndarray,
    *,
    audio_input_name: str,
    cond_input_name: str,
    condition_normalize: bool,
    input_sample_rate: int,
    tse_sample_rate: int,
    chunk_samples: int,
    normalize: str,
    resample_device: str,
) -> np.ndarray:
    wave_48k = _resample_1d(
        wave_16k,
        source_rate=input_sample_rate,
        target_rate=tse_sample_rate,
        device=resample_device,
    )
    scale = _input_scale(wave_48k, mode=normalize)
    model_wave = (wave_48k / scale).astype(np.float32)
    original_len = model_wave.shape[0]
    pad = (-original_len) % chunk_samples
    if pad:
        model_wave = np.pad(model_wave, (0, pad)).astype(np.float32)

    state = {
        item.name: np.zeros(_state_shape(item.shape), dtype=np.float32) for item in state_inputs
    }
    cond_vector = (
        _normalize_row(cond) if condition_normalize else np.asarray(cond, dtype=np.float32)
    )
    cond_2d = (
        np.nan_to_num(cond_vector, nan=0.0, posinf=0.0, neginf=0.0)
        .reshape(1, -1)
        .astype(np.float32)
    )
    output_chunks: List[np.ndarray] = []
    for offset in range(0, model_wave.shape[0], chunk_samples):
        feed = {
            audio_input_name: model_wave[offset : offset + chunk_samples].reshape(1, chunk_samples),
            cond_input_name: cond_2d,
            **state,
        }
        outputs = session.run(None, feed)
        output_chunks.append(
            np.nan_to_num(
                np.asarray(outputs[0], dtype=np.float32).reshape(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        )
        state = {f"state_in_{index}": outputs[index + 1] for index in range(len(state_inputs))}

    extracted_48k = np.nan_to_num(
        np.concatenate(output_chunks)[:original_len].astype(np.float32) * scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    extracted_16k = _resample_1d(
        extracted_48k,
        source_rate=tse_sample_rate,
        target_rate=input_sample_rate,
        device=resample_device,
    )
    extracted_16k = np.nan_to_num(extracted_16k, nan=0.0, posinf=0.0, neginf=0.0)
    target_len = np.asarray(wave_16k).shape[0]
    if extracted_16k.shape[0] < target_len:
        extracted_16k = np.pad(extracted_16k, (0, target_len - extracted_16k.shape[0]))
    return np.clip(extracted_16k[:target_len], -1.0, 1.0).astype(np.float32)


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _write_tse_embeddings(
    rows: Sequence[MaskRow],
    *,
    session: ort.InferenceSession,
    state_inputs: Sequence,
    centroids: Mapping[str, np.ndarray],
    audio_input_name: str,
    cond_input_name: str,
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    input_sample_rate: int,
    tse_sample_rate: int,
    chunk_samples: int,
    batch_size: int,
    normalize: str,
    condition_normalize: bool,
    device: str,
    resample_device: str,
) -> None:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    titanet = _load_titanet(device)
    samples = int(round(window_seconds * input_sample_rate))

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
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", input_sample_rate)
        extracted: List[np.ndarray] = []
        for row_index, row in enumerate(window_rows, start=1):
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * input_sample_rate))
            crop = _slice_wave(mixed, start_sample, start_sample + samples)
            extracted.append(
                _extract_with_tse(
                    session,
                    state_inputs,
                    crop,
                    centroids[row.truth],
                    audio_input_name=audio_input_name,
                    cond_input_name=cond_input_name,
                    condition_normalize=condition_normalize,
                    input_sample_rate=input_sample_rate,
                    tse_sample_rate=tse_sample_rate,
                    chunk_samples=chunk_samples,
                    normalize=normalize,
                    resample_device=resample_device,
                )
            )
            if row_index % 25 == 0 or row_index == len(window_rows):
                print(f"extracted {window_name}: {row_index}/{len(window_rows)}", flush=True)
        all_embeddings.append(
            _embed_waveforms(
                titanet,
                extracted,
                sample_rate=input_sample_rate,
                batch_size=batch_size,
                device=device,
            )
        )
        all_windows.extend([window_name] * len(window_rows))
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        print(f"embedded {window_name}: {len(window_rows)} pretrained-tse words", flush=True)

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
        description="Evaluate a pretrained ECAPA-conditioned TSE Conv-TasNet target extractor."
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
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/tmp/codex_conditioned_tasnet_real_stems_s120_min0.npz"),
    )
    parser.add_argument(
        "--centroid-cache",
        type=Path,
        default=Path("/tmp/codex_pretrained_tse_ecapa_raw_centroids.npz"),
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_pretrained_tse_embeddings.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_pretrained_tse_results.json"),
    )
    parser.add_argument(
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument("--hf-repo-id", default="penta2himajin/tse-conv-tasnet-48k")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--input-sample-rate", type=int, default=16000)
    parser.add_argument("--tse-sample-rate", type=int, default=48000)
    parser.add_argument("--chunk-samples", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ecapa-batch-size", type=int, default=48)
    parser.add_argument("--normalize", choices=("none", "peak", "rms"), default="peak")
    parser.add_argument(
        "--ecapa-normalize",
        action="store_true",
        help="Ask SpeechBrain to normalize enrollment embeddings before averaging.",
    )
    parser.add_argument(
        "--condition-normalize",
        action="store_true",
        help="L2-normalize the centroid before feeding it to the TSE ONNX graph.",
    )
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resample-device", default="cuda")
    parser.add_argument("--ecapa-savedir", type=Path, default=Path("/tmp/codex_speechbrain_ecapa"))
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    resample_device = str(args.resample_device)
    if resample_device == "cuda" and not torch.cuda.is_available():
        resample_device = "cpu"
    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    centroids = _load_or_create_ecapa_centroids(
        training_cache=args.training_cache.expanduser(),
        output_path=args.centroid_cache.expanduser(),
        savedir=args.ecapa_savedir.expanduser(),
        batch_size=int(args.ecapa_batch_size),
        device=device,
        normalize_embeddings=bool(args.ecapa_normalize),
        normalize_centroid=bool(args.condition_normalize),
    )
    missing = sorted({row.truth for row in rows} - set(centroids))
    if missing:
        raise RuntimeError(f"Missing ECAPA centroids for {missing}")

    providers = [item.strip() for item in str(args.providers).split(",") if item.strip()]
    session, state_inputs, audio_input_name, cond_input_name = _load_tse_session(
        repo_id=str(args.hf_repo_id), providers=providers
    )
    if not args.embedding_output.expanduser().exists():
        _write_tse_embeddings(
            rows,
            session=session,
            state_inputs=state_inputs,
            centroids=centroids,
            audio_input_name=audio_input_name,
            cond_input_name=cond_input_name,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.embedding_output.expanduser(),
            window_seconds=float(args.window_seconds),
            input_sample_rate=int(args.input_sample_rate),
            tse_sample_rate=int(args.tse_sample_rate),
            chunk_samples=int(args.chunk_samples),
            batch_size=int(args.batch_size),
            normalize=str(args.normalize),
            condition_normalize=bool(args.condition_normalize),
            device=device,
            resample_device=resample_device,
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
        "model": "pretrained_ecapa_conditioned_tse_conv_tasnet",
        "hf_repo_id": str(args.hf_repo_id),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "training_cache": str(args.training_cache.expanduser()),
        "normalize": str(args.normalize),
        "ecapa_normalize": bool(args.ecapa_normalize),
        "condition_normalize": bool(args.condition_normalize),
        "onnx_audio_input": audio_input_name,
        "onnx_condition_input": cond_input_name,
        "target_extractor": result,
        "mixed_same_rows": mixed_result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    print(
        ",".join(
            [
                "pretrained_tse_true_target/lda_shrinkage",
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
