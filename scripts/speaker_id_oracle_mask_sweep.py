from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torchaudio

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_architecture_sweep import TrainingSet, _fit_predict  # noqa: E402
from speaker_id_word_window_sweep import (  # noqa: E402
    WordEmbeddingSet,
    WordWindow,
    _score_word_predictions,
    _window_group,
)
from transcriber.multitrack_eval import WordSpan, extract_words_from_jsonl  # noqa: E402


@dataclass(frozen=True)
class MaskRow:
    window: str
    index: int
    truth: str
    target_file: str
    target_share: float
    active_5pct: int


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    waveform, source_sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_sr) != sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(source_sr), sample_rate)
    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _slice_wave(wave: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end > wave.shape[0]:
        padded = np.zeros(max(end - start, 0), dtype=np.float32)
        left = max(start, 0)
        right = min(end, wave.shape[0])
        if right > left:
            padded[left - start : right - start] = wave[left:right]
        return padded
    return np.asarray(wave[start:end], dtype=np.float32)


def _rms(wave: np.ndarray) -> float:
    if wave.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))


def _ideal_ratio_mask_crop(
    mixed: np.ndarray,
    sources: Mapping[str, np.ndarray],
    target_file: str,
    start_sample: int,
    end_sample: int,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    power: float,
    normalize: str,
) -> np.ndarray:
    del sample_rate
    mixed_crop = _slice_wave(mixed, start_sample, end_sample)
    target_crop = _slice_wave(sources[target_file], start_sample, end_sample)
    source_crops = [
        _slice_wave(wave, start_sample, end_sample) for _name, wave in sorted(sources.items())
    ]
    if not source_crops or max(_rms(target_crop), _rms(mixed_crop)) == 0.0:
        return mixed_crop.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = torch.hann_window(n_fft, device=device)
    source_tensor = torch.from_numpy(np.stack(source_crops)).to(device)
    mixed_tensor = torch.from_numpy(mixed_crop).to(device)
    target_index = sorted(sources).index(target_file)

    with torch.inference_mode():
        source_stft = torch.stft(
            source_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mixed_stft = torch.stft(
            mixed_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        magnitudes = torch.abs(source_stft).clamp_min(1e-8).pow(power)
        mask = magnitudes[target_index] / magnitudes.sum(dim=0).clamp_min(1e-8)
        separated = torch.istft(
            mixed_stft * mask,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            length=mixed_crop.shape[0],
        )
    separated_np = separated.detach().cpu().numpy().astype(np.float32)

    if normalize == "target_rms":
        source_rms = _rms(target_crop)
        output_rms = _rms(separated_np)
        if source_rms > 0.0 and output_rms > 0.0:
            separated_np = separated_np * min(source_rms / output_rms, 10.0)
    elif normalize == "mixed_rms":
        source_rms = _rms(mixed_crop)
        output_rms = _rms(separated_np)
        if source_rms > 0.0 and output_rms > 0.0:
            separated_np = separated_np * min(source_rms / output_rms, 10.0)

    return np.clip(separated_np, -1.0, 1.0).astype(np.float32)


def _load_titanet(device: str):
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("titanet_small")
    model.eval().to(device)
    return model


def _embed_waveforms(
    model,
    waves: Sequence[np.ndarray],
    *,
    sample_rate: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    del sample_rate
    vectors: List[np.ndarray] = []
    for offset in range(0, len(waves), batch_size):
        batch = [
            np.asarray(wave, dtype=np.float32).flatten()
            for wave in waves[offset : offset + batch_size]
        ]
        lengths = torch.tensor([wave.shape[0] for wave in batch], dtype=torch.long, device=device)
        max_len = int(lengths.max().item())
        wave_batch = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
        for row, wave in enumerate(batch):
            wave_batch[row, : wave.shape[0]] = torch.from_numpy(wave).to(device)
        with torch.inference_mode():
            output = model(input_signal=wave_batch, input_signal_length=lengths)
        embedding = output[-1] if isinstance(output, tuple) else output
        embedding_np = embedding.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embedding_np, axis=1, keepdims=True)
        vectors.append(embedding_np / np.maximum(norms, 1e-8))
    return np.vstack(vectors).astype(np.float32)


def _load_rows(path: Path, max_target_share: float) -> List[MaskRow]:
    raw_rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    rows: List[MaskRow] = []
    for raw in raw_rows:
        share = float(raw.get("target_share") or 0.0)
        if share > max_target_share:
            continue
        rows.append(
            MaskRow(
                window=str(raw["window"]),
                index=int(raw["index"]),
                truth=str(raw["truth"]),
                target_file=str(raw["target_file"]),
                target_share=share,
                active_5pct=int(raw.get("active_5pct") or 0),
            )
        )
    return rows


def _load_titanet_word_npz(cache_root: Path, source: str, window: str, window_seconds: float):
    path = cache_root / source / _safe_name(window) / f"titanet_small_{window_seconds:.2f}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _load_clean_bank(path: Path) -> TrainingSet:
    payload = np.load(path, allow_pickle=False)
    return TrainingSet(
        name=f"clean_bank:{path.name}",
        embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
        labels=[str(item) for item in payload["labels"].tolist()],
    )


def _collect_training_items(
    rows_by_window: Mapping[str, Sequence[MaskRow]],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
) -> Dict[str, WordEmbeddingSet]:
    items: Dict[str, WordEmbeddingSet] = {}
    for window in sorted(rows_by_window):
        payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window,
            window_seconds,
        )
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        truths = [str(item) for item in payload["truths"].tolist()]
        starts = np.asarray(payload["word_starts"], dtype=np.float32)
        ends = np.asarray(payload["word_ends"], dtype=np.float32)
        texts = [str(item) for item in payload["texts"].tolist()]
        window_dir = prepared_root / window
        reference_words = extract_words_from_jsonl(
            window_dir / "reference" / "clips" / "clips.jsonl"
        )
        word_window = WordWindow(
            name=window,
            group=_window_group(window),
            source="reference",
            audio_path=window_dir / "mixed.wav",
            reference_words=reference_words,
            predicted_words=[
                WordSpan(speaker="unknown", start=float(start), end=float(end), text=text)
                for start, end, text in zip(starts, ends, texts)
            ],
            truths=truths,
        )
        items[window] = WordEmbeddingSet(
            window=word_window,
            embeddings=embeddings,
            truths=truths,
            predicted_words=word_window.predicted_words,
        )
    return items


def _score_direct(truths: Sequence[str], predictions: Sequence[str]) -> Dict[str, object]:
    total = len(truths)
    correct = sum(str(a) == str(b) for a, b in zip(truths, predictions))
    return {
        "examples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def _score_slices(
    rows: Sequence[MaskRow],
    predictions: Sequence[str],
    *,
    field: str,
) -> List[Dict[str, object]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if field == "share":
            if row.target_share >= 0.75:
                key = "0.75-0.90"
            elif row.target_share >= 0.60:
                key = "0.60-0.75"
            elif row.target_share >= 0.45:
                key = "0.45-0.60"
            elif row.target_share >= 0.30:
                key = "0.30-0.45"
            else:
                key = "<0.30"
        elif field == "active_5pct":
            key = f"active5={row.active_5pct}"
        else:
            raise ValueError(field)
        buckets[key].append(index)

    summary: List[Dict[str, object]] = []
    for key, indices in sorted(buckets.items()):
        truths = [rows[index].truth for index in indices]
        preds = [predictions[index] for index in indices]
        score = _score_direct(truths, preds)
        score["bucket"] = key
        score["mean_target_share"] = float(np.mean([rows[index].target_share for index in indices]))
        summary.append(score)
    return summary


def _write_mask_embeddings(
    rows: Sequence[MaskRow],
    *,
    prepared_root: Path,
    cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    batch_size: int,
    n_fft: int,
    hop_length: int,
    mask_power: float,
    normalize: str,
    device: str,
) -> None:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)

    model = _load_titanet(device)
    all_vectors: List[np.ndarray] = []
    all_keys: List[Tuple[str, int]] = []
    all_truths: List[str] = []
    all_shares: List[float] = []
    all_active: List[int] = []

    cache_root.mkdir(parents=True, exist_ok=True)
    for window_name, window_rows in sorted(rows_by_window.items()):
        window_cache = (
            cache_root
            / _safe_name(window_name)
            / f"oracle_mask_{window_seconds:.2f}_p{mask_power:.1f}_{normalize}.npz"
        )
        if window_cache.exists():
            payload = np.load(window_cache, allow_pickle=False)
            vectors = np.asarray(payload["embeddings"], dtype=np.float32)
            keys = [(window_name, int(index)) for index in payload["indices"].tolist()]
            truths = [str(item) for item in payload["truths"].tolist()]
            shares = [float(item) for item in payload["target_shares"].tolist()]
            active = [int(item) for item in payload["active_5pct"].tolist()]
        else:
            word_payload = _load_titanet_word_npz(
                Path("/tmp/codex_titanet_word_window"),
                "reference",
                window_name,
                window_seconds,
            )
            starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
            ends = np.asarray(word_payload["word_ends"], dtype=np.float32)

            window_dir = prepared_root / window_name
            mixed = _load_audio(window_dir / "mixed.wav", sample_rate)
            sources = {
                path.name: _load_audio(path, sample_rate)
                for path in sorted((window_dir / "clips").glob("*.wav"))
            }
            if not sources:
                raise FileNotFoundError(f"No source clips under {window_dir / 'clips'}")

            waves: List[np.ndarray] = []
            for row in window_rows:
                midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
                start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
                end_sample = start_sample + int(round(window_seconds * sample_rate))
                waves.append(
                    _ideal_ratio_mask_crop(
                        mixed,
                        sources,
                        row.target_file,
                        start_sample,
                        end_sample,
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        power=mask_power,
                        normalize=normalize,
                    )
                )
            vectors = _embed_waveforms(
                model,
                waves,
                sample_rate=sample_rate,
                batch_size=batch_size,
                device=device,
            )
            keys = [(window_name, row.index) for row in window_rows]
            truths = [row.truth for row in window_rows]
            shares = [row.target_share for row in window_rows]
            active = [row.active_5pct for row in window_rows]
            window_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                window_cache,
                embeddings=vectors,
                indices=np.asarray([row.index for row in window_rows], dtype=np.int32),
                truths=np.asarray(truths),
                target_shares=np.asarray(shares, dtype=np.float32),
                active_5pct=np.asarray(active, dtype=np.int16),
            )
        all_vectors.append(vectors)
        all_keys.extend(keys)
        all_truths.extend(truths)
        all_shares.extend(shares)
        all_active.extend(active)
        print(f"cached {window_name}: {len(truths)} masked words", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.vstack(all_vectors).astype(np.float32),
        windows=np.asarray([key[0] for key in all_keys]),
        indices=np.asarray([key[1] for key in all_keys], dtype=np.int32),
        truths=np.asarray(all_truths),
        target_shares=np.asarray(all_shares, dtype=np.float32),
        active_5pct=np.asarray(all_active, dtype=np.int16),
    )


def _load_mask_embeddings(path: Path) -> Tuple[np.ndarray, List[MaskRow]]:
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


def _rows_for_items(items: Iterable[WordEmbeddingSet]) -> Tuple[np.ndarray, List[str]]:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for item in items:
        vectors.extend(np.asarray(item.embeddings, dtype=np.float32))
        labels.extend(item.truths)
    return np.vstack(vectors), labels


def evaluate_mask_embeddings(
    mask_embeddings: np.ndarray,
    mask_rows: Sequence[MaskRow],
    *,
    clean_bank: TrainingSet,
    training_items: Mapping[str, WordEmbeddingSet],
    model_name: str,
) -> Dict[str, object]:
    groups = sorted({_window_group(row.window) for row in mask_rows})
    predictions = ["unknown"] * len(mask_rows)
    by_group_index: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(mask_rows):
        by_group_index[_window_group(row.window)].append(index)

    for group in groups:
        train_items = [item for item in training_items.values() if item.window.group != group]
        train_x, train_y = _rows_for_items(train_items)
        train_x = np.vstack([clean_bank.embeddings, train_x]).astype(np.float32)
        train_y = list(clean_bank.labels) + train_y
        test_indices = by_group_index[group]
        predicted = _fit_predict(model_name, train_x, train_y, mask_embeddings[test_indices])
        for index, label in zip(test_indices, predicted):
            predictions[index] = label

    direct = _score_direct([row.truth for row in mask_rows], predictions)
    return {
        "name": f"oracle_mask/{model_name}",
        "direct": direct,
        "share_slices": _score_slices(mask_rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(mask_rows, predictions, field="active_5pct"),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ideal ratio-mask source isolation for overlapped speaker ID."
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
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/codex_oracle_mask_cache"))
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_oracle_mask_embeddings.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_oracle_mask_results.json"),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--mask-power", type=float, default=2.0)
    parser.add_argument(
        "--normalize",
        choices=("none", "target_rms", "mixed_rms"),
        default="target_rms",
    )
    parser.add_argument("--model", default="lda_shrinkage")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    if not rows:
        raise RuntimeError("No rows selected")

    if not args.embedding_output.expanduser().exists():
        _write_mask_embeddings(
            rows,
            prepared_root=args.prepared_root.expanduser(),
            cache_root=args.cache_root.expanduser(),
            output_path=args.embedding_output.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            batch_size=int(args.batch_size),
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            mask_power=float(args.mask_power),
            normalize=str(args.normalize),
            device=str(args.device),
        )

    mask_embeddings, mask_rows = _load_mask_embeddings(args.embedding_output.expanduser())
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in mask_rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    result = evaluate_mask_embeddings(
        mask_embeddings,
        mask_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        model_name=str(args.model),
    )

    truth = [row.truth for row in mask_rows]
    by_window_predictions: Dict[str, List[str]] = defaultdict(list)
    by_window_rows: Dict[str, List[MaskRow]] = defaultdict(list)
    for row, prediction in zip(mask_rows, result["predictions"]):
        by_window_predictions[row.window].append(prediction)
        by_window_rows[row.window].append(row)

    subset_embedding_sets: List[WordEmbeddingSet] = []
    for window, selected_rows in by_window_rows.items():
        item = training_items[window]
        indices = [row.index for row in selected_rows]
        subset_embedding_sets.append(
            WordEmbeddingSet(
                window=item.window,
                embeddings=mask_embeddings[
                    [idx for idx, row in enumerate(mask_rows) if row.window == window]
                ],
                truths=[
                    truth[index]
                    for index in range(len(mask_rows))
                    if mask_rows[index].window == window
                ],
                predicted_words=[item.predicted_words[index] for index in indices],
            )
        )
    result["word"] = _score_word_predictions(subset_embedding_sets, by_window_predictions)

    del result["predictions"]
    payload = {
        "prepared_root": str(args.prepared_root.expanduser()),
        "dominance_json": str(args.dominance_json.expanduser()),
        "embedding_output": str(args.embedding_output.expanduser()),
        "window_seconds": float(args.window_seconds),
        "max_target_share": float(args.max_target_share),
        "mask_power": float(args.mask_power),
        "normalize": str(args.normalize),
        "selected_rows": len(mask_rows),
        "selected_speakers": dict(Counter(row.truth for row in mask_rows)),
        "result": result,
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    direct = result["direct"]
    print("name,examples,direct_accuracy", flush=True)
    print(
        ",".join(
            [
                result["name"],
                str(direct["examples"]),
                f"{float(direct['accuracy']):.4f}",
            ]
        ),
        flush=True,
    )
    print(json.dumps(result["share_slices"], indent=2), flush=True)


if __name__ == "__main__":
    main()
