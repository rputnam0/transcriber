from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _evaluate_candidate_embeddings,
    _load_candidate_embeddings,
)
from speaker_id_conditioned_tasnet_sweep import _conditioning_vectors, _extract_batch  # noqa: E402
from speaker_id_eval_stem_candidate_sweep import (  # noqa: E402
    _load_fold_model,
    _model_path_for_group,
    _split_group,
)
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet_word_npz,
    _score_direct,
    _score_slices,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "for",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "the",
    "to",
    "uh",
    "um",
    "was",
    "we",
    "yeah",
    "you",
}


def _normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value).lower()))


def _text_match_score(target: str, transcript: str) -> float:
    target_norm = _normalize_text(target)
    transcript_norm = _normalize_text(transcript)
    if not target_norm or not transcript_norm:
        return 0.0
    if f" {target_norm} " in f" {transcript_norm} ":
        return 1.0
    target_parts = [token for token in target_norm.split() if token not in _STOPWORDS]
    if not target_parts:
        target_parts = target_norm.split()
    transcript_parts = transcript_norm.split()
    scores: List[float] = []
    for target_token in target_parts:
        scores.append(
            max(
                (
                    SequenceMatcher(a=target_token, b=transcript_token).ratio()
                    for transcript_token in transcript_parts
                ),
                default=0.0,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def _load_reference_targets(
    rows: Sequence[MaskRow],
    *,
    titanet_cache_root: Path,
    window_seconds: float,
    context_radius: int,
) -> Tuple[List[str], List[str]]:
    payload_by_window: Dict[str, Mapping[str, np.ndarray]] = {}
    target_words: List[str] = []
    target_phrases: List[str] = []
    for row in rows:
        if row.window not in payload_by_window:
            payload_by_window[row.window] = _load_titanet_word_npz(
                titanet_cache_root,
                "reference",
                row.window,
                window_seconds,
            )
        window_payload = payload_by_window[row.window]
        word_texts = np.asarray(window_payload["texts"])
        word_truths = np.asarray(window_payload["truths"])
        selected = [int(row.index)]
        left = int(row.index) - 1
        left_count = 0
        while left >= 0 and left_count < context_radius:
            if str(word_truths[left]) != row.truth:
                break
            selected.insert(0, left)
            left_count += 1
            left -= 1
        right = int(row.index) + 1
        right_count = 0
        while right < len(word_texts) and right_count < context_radius:
            if str(word_truths[right]) != row.truth:
                break
            selected.append(right)
            right_count += 1
            right += 1
        target_words.append(str(word_texts[row.index]))
        target_phrases.append(" ".join(str(word_texts[index]) for index in selected))
    return target_words, target_phrases


def _load_word_crops(
    rows: Sequence[MaskRow],
    *,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    payload_by_window: Dict[str, Mapping[str, np.ndarray]] = {}
    audio_by_window: Dict[str, np.ndarray] = {}
    samples = int(round(window_seconds * sample_rate))
    waves: List[np.ndarray] = []
    for row in rows:
        if row.window not in payload_by_window:
            payload_by_window[row.window] = _load_titanet_word_npz(
                titanet_cache_root,
                "reference",
                row.window,
                window_seconds,
            )
        if row.window not in audio_by_window:
            audio_by_window[row.window] = _load_audio(
                prepared_root / row.window / "mixed.wav",
                sample_rate,
            )
        payload = payload_by_window[row.window]
        starts = np.asarray(payload["word_starts"], dtype=np.float32)
        ends = np.asarray(payload["word_ends"], dtype=np.float32)
        mixed = audio_by_window[row.window]
        midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
        start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
        end_sample = start_sample + samples
        if start_sample < 0 or end_sample > mixed.shape[0]:
            padded = np.zeros(samples, dtype=np.float32)
            left = max(start_sample, 0)
            right = min(end_sample, mixed.shape[0])
            if right > left:
                padded[left - start_sample : right - start_sample] = mixed[left:right]
            waves.append(padded)
        else:
            waves.append(np.asarray(mixed[start_sample:end_sample], dtype=np.float32))
    return np.stack(waves).astype(np.float32)


def _extract_candidate_waves(
    rows: Sequence[MaskRow],
    mixtures: np.ndarray,
    *,
    model_base_path: Path,
    split_mode: str,
    centroids: Mapping[str, np.ndarray],
    row_batch_size: int,
    device: str,
    args: argparse.Namespace,
) -> np.ndarray:
    candidates = tuple(CORE_SPEAKERS)
    embedding_dim = next(iter(centroids.values())).shape[0]
    model_cache = {}
    extracted_by_row = np.zeros(
        (len(rows), len(candidates), mixtures.shape[-1]),
        dtype=np.float32,
    )
    for offset in range(0, len(rows), row_batch_size):
        batch_rows = rows[offset : offset + row_batch_size]
        batch_waves = mixtures[offset : offset + row_batch_size]
        for group in sorted({_split_group(row, split_mode) for row in batch_rows}):
            local_indices = [
                index
                for index, row in enumerate(batch_rows)
                if _split_group(row, split_mode) == group
            ]
            if group not in model_cache:
                model_path = _model_path_for_group(model_base_path, group)
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                model_cache[group] = _load_fold_model(
                    model_path,
                    embedding_dim=embedding_dim,
                    device=device,
                    args=args,
                )
            group_waves = np.stack([batch_waves[index] for index in local_indices])
            expanded_waves = np.repeat(group_waves, len(candidates), axis=0)
            expanded_labels = [speaker for _ in local_indices for speaker in candidates]
            extracted = _extract_batch(
                model_cache[group],
                expanded_waves,
                expanded_labels,
                centroids=centroids,
                device=device,
            )
            extracted = extracted.reshape(len(local_indices), len(candidates), extracted.shape[-1])
            for local_output_index, batch_index in enumerate(local_indices):
                extracted_by_row[offset + batch_index] = extracted[local_output_index]
        print(
            f"candidate_asr_extracted {min(offset + row_batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
    return extracted_by_row


def _prepare_asr_wave(wave: np.ndarray, *, min_rms: float, peak: float) -> np.ndarray | None:
    wave = np.asarray(wave, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(wave))))
    if rms < min_rms:
        return None
    max_abs = float(np.max(np.abs(wave)))
    if max_abs <= 1e-8:
        return None
    return (wave * (peak / max_abs)).astype(np.float32)


def _transcribe_wave(
    model: WhisperModel,
    wave: np.ndarray,
    *,
    sample_rate: int,
    path: Path,
    min_rms: float,
    normalize_peak: float,
    language: str,
) -> Dict[str, object]:
    prepared = _prepare_asr_wave(wave, min_rms=min_rms, peak=normalize_peak)
    if prepared is None:
        return {"text": "", "segments": [], "rms_too_low": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, prepared, sample_rate)
    segments_iter, info = model.transcribe(
        str(path),
        language=language,
        beam_size=1,
        best_of=1,
        vad_filter=False,
        word_timestamps=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    segments = [
        {
            "start": float(getattr(segment, "start", 0.0) or 0.0),
            "end": float(getattr(segment, "end", 0.0) or 0.0),
            "text": str(getattr(segment, "text", "") or "").strip(),
        }
        for segment in segments_iter
    ]
    return {
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
        "language": str(getattr(info, "language", "") or ""),
        "rms_too_low": False,
    }


def _score_predictions(
    name: str,
    rows: Sequence[MaskRow],
    predictions: Sequence[str],
) -> Dict[str, object]:
    return {
        "name": name,
        "direct": _score_direct([row.truth for row in rows], predictions),
        "share_slices": _score_slices(rows, predictions, field="share"),
        "active_5pct_slices": _score_slices(rows, predictions, field="active_5pct"),
    }


def _best_threshold(
    values: np.ndarray, candidate_pred: Sequence[str], mixed_pred: Sequence[str], rows
):
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 41, dtype=np.float32),
                np.asarray(values, dtype=np.float32),
            ]
        )
    )
    best_threshold = 1.01
    best_correct = -1
    for threshold in thresholds:
        routed = [
            candidate if value >= threshold else mixed
            for value, candidate, mixed in zip(values, candidate_pred, mixed_pred)
        ]
        correct = sum(row.truth == prediction for row, prediction in zip(rows, routed))
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(threshold)
    return best_threshold


def _threshold_router(
    rows: Sequence[MaskRow],
    values: np.ndarray,
    candidate_pred: Sequence[str],
    mixed_pred: Sequence[str],
) -> Tuple[List[str], Dict[str, float]]:
    by_group: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[_window_group(row.window)].append(index)
    output = ["unknown"] * len(rows)
    thresholds: Dict[str, float] = {}
    for group, test_indices in sorted(by_group.items()):
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        threshold = _best_threshold(
            values[train_indices],
            [candidate_pred[index] for index in train_indices],
            [mixed_pred[index] for index in train_indices],
            [rows[index] for index in train_indices],
        )
        thresholds[group] = threshold
        for index in test_indices:
            output[index] = (
                candidate_pred[index] if float(values[index]) >= threshold else mixed_pred[index]
            )
    return output, thresholds


def _candidate_feature_matrix(
    match_scores: np.ndarray,
    candidates: Sequence[str],
    mixed_predictions: Sequence[str],
) -> np.ndarray:
    rows, candidate_count = match_scores.shape
    sorted_scores = np.sort(match_scores, axis=1)
    top_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2] if candidate_count > 1 else np.zeros(rows)
    margins = top_scores - second_scores
    ranks = np.argsort(np.argsort(-match_scores, axis=1), axis=1)
    features: List[List[float]] = []
    for row_index in range(rows):
        for candidate_index, candidate in enumerate(candidates):
            one_hot = [1.0 if index == candidate_index else 0.0 for index in range(candidate_count)]
            score = float(match_scores[row_index, candidate_index])
            features.append(
                [
                    score,
                    float(score - top_scores[row_index]),
                    float(score - second_scores[row_index]),
                    float(top_scores[row_index]),
                    float(margins[row_index]),
                    float(ranks[row_index, candidate_index]),
                    float(candidate == mixed_predictions[row_index]),
                    *one_hot,
                ]
            )
    return np.asarray(features, dtype=np.float32)


def _candidate_labels(rows: Sequence[MaskRow], candidates: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [1 if row.truth == candidate else 0 for row in rows for candidate in candidates],
        dtype=np.int64,
    )


def _fit_asr_selector(train_x: np.ndarray, train_y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selector = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
    )
    selector.fit(train_x, train_y)
    return selector


def _learned_asr_selector(
    rows: Sequence[MaskRow],
    match_scores: np.ndarray,
    candidates: Sequence[str],
    mixed_predictions: Sequence[str],
) -> Tuple[List[str], List[str], Dict[str, float]]:
    selector_predictions = ["unknown"] * len(rows)
    router_predictions = ["unknown"] * len(rows)
    thresholds: Dict[str, float] = {}
    by_group: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[_window_group(row.window)].append(index)

    for group, test_indices in sorted(by_group.items()):
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        train_x = _candidate_feature_matrix(
            match_scores[train_indices],
            candidates,
            [mixed_predictions[index] for index in train_indices],
        )
        train_y = _candidate_labels([rows[index] for index in train_indices], candidates)
        if len(np.unique(train_y)) < 2:
            for index in test_indices:
                selector_predictions[index] = mixed_predictions[index]
                router_predictions[index] = mixed_predictions[index]
            thresholds[group] = 1.01
            continue
        selector = _fit_asr_selector(train_x, train_y)
        train_scores = selector.predict_proba(train_x)[:, 1].reshape(
            len(train_indices),
            len(candidates),
        )
        train_best = np.argmax(train_scores, axis=1)
        train_best_scores = train_scores[np.arange(len(train_indices)), train_best]
        train_candidate_predictions = [candidates[int(index)] for index in train_best]
        threshold = _best_threshold(
            train_best_scores,
            train_candidate_predictions,
            [mixed_predictions[index] for index in train_indices],
            [rows[index] for index in train_indices],
        )
        thresholds[group] = threshold

        test_x = _candidate_feature_matrix(
            match_scores[test_indices],
            candidates,
            [mixed_predictions[index] for index in test_indices],
        )
        test_scores = selector.predict_proba(test_x)[:, 1].reshape(
            len(test_indices),
            len(candidates),
        )
        test_best = np.argmax(test_scores, axis=1)
        test_best_scores = test_scores[np.arange(len(test_indices)), test_best]
        for local_index, row_index in enumerate(test_indices):
            candidate = candidates[int(test_best[local_index])]
            selector_predictions[row_index] = candidate
            router_predictions[row_index] = (
                candidate
                if float(test_best_scores[local_index]) >= threshold
                else mixed_predictions[row_index]
            )
    return selector_predictions, router_predictions, thresholds


def _mixed_predictions_from_candidate_cache(
    rows: Sequence[MaskRow],
    *,
    candidate_embedding_path: Path,
    clean_bank_path: Path,
    prepared_root: Path,
    titanet_cache_root: Path,
    window_seconds: float,
) -> List[str]:
    candidate_embeddings, candidates, embedding_rows = _load_candidate_embeddings(
        candidate_embedding_path
    )
    if len(embedding_rows) < len(rows):
        raise RuntimeError("Candidate embedding cache has fewer rows than requested")
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in embedding_rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(clean_bank_path)
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=prepared_root,
        titanet_cache_root=titanet_cache_root,
        window_seconds=window_seconds,
    )
    mixed_embeddings = _mixed_same_rows(
        rows_by_window,
        titanet_cache_root=titanet_cache_root,
        window_seconds=window_seconds,
    )
    scored = _evaluate_candidate_embeddings(
        candidate_embeddings,
        candidates,
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        mixed_embeddings=mixed_embeddings,
    )
    diagnostics = scored["diagnostic_rows"]
    predictions = ["unknown"] * len(embedding_rows)
    for item in diagnostics:
        predictions[int(item["index"])] = str(item["mixed_pred"])
    for requested, cached in zip(rows, embedding_rows):
        if requested.window != cached.window or requested.index != cached.index:
            raise RuntimeError(
                "Candidate embedding cache row order does not match the requested ASR rows"
            )
    return predictions[: len(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use ASR word evidence to select among all-candidate target-extractor outputs."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--model-base-path",
        type=Path,
        default=Path("/tmp/codex_eval_stem_tasnet_lgo_s300_big_s1600.pt"),
    )
    parser.add_argument(
        "--candidate-embedding-cache",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidates_lgo_s300_big_s1600_embeddings.npz"),
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
        "--titanet-cache-root",
        type=Path,
        default=Path("/tmp/codex_titanet_word_window"),
    )
    parser.add_argument(
        "--transcript-cache",
        type=Path,
        default=Path("/tmp/codex_candidate_asr_attribution_s60_transcripts.json"),
    )
    parser.add_argument(
        "--audio-output-dir",
        type=Path,
        default=Path("/tmp/codex_candidate_asr_attribution_audio_s60"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/codex_candidate_asr_attribution_s60.json"),
    )
    parser.add_argument(
        "--split-mode", choices=("leave_group", "leave_session"), default="leave_group"
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--asr-limit", type=int, default=60)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--conditioning", choices=("one_hot",), default="one_hot")
    parser.add_argument("--enc-feats", type=int, default=192)
    parser.add_argument("--bottleneck", type=int, default=192)
    parser.add_argument("--cond-dim", type=int, default=96)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--stacks", type=int, default=3)
    parser.add_argument("--asr-model", default="tiny.en")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-rms", type=float, default=2e-4)
    parser.add_argument("--normalize-peak", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    ordered_rows: List[MaskRow] = []
    for window in sorted(rows_by_window):
        ordered_rows.extend(rows_by_window[window])
    selected_rows = ordered_rows[: int(args.asr_limit)]
    target_words, reference_texts = _load_reference_targets(
        selected_rows,
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
        context_radius=int(args.context_radius),
    )
    mixed_predictions = _mixed_predictions_from_candidate_cache(
        ordered_rows,
        candidate_embedding_path=args.candidate_embedding_cache.expanduser(),
        clean_bank_path=args.clean_bank.expanduser(),
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )[: len(selected_rows)]

    candidates = tuple(CORE_SPEAKERS)
    transcript_rows: List[Dict[str, object]]
    if args.transcript_cache.expanduser().exists():
        transcript_rows = json.loads(args.transcript_cache.expanduser().read_text(encoding="utf-8"))
        for row_payload, target_word, reference_text in zip(
            transcript_rows,
            target_words,
            reference_texts,
        ):
            row_payload["target_word"] = target_word
            row_payload["target_text"] = reference_text
            for candidate_payload in row_payload["candidates"]:
                candidate_payload["match_score"] = _text_match_score(
                    reference_text,
                    str(candidate_payload["text"]),
                )
    else:
        mixtures = _load_word_crops(
            selected_rows,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
        )
        centroids = _conditioning_vectors(
            clean_bank_path=args.clean_bank.expanduser(),
            mode=str(args.conditioning),
        )
        extracted = _extract_candidate_waves(
            selected_rows,
            mixtures,
            model_base_path=args.model_base_path.expanduser(),
            split_mode=str(args.split_mode),
            centroids=centroids,
            row_batch_size=int(args.row_batch_size),
            device=device,
            args=args,
        )
        asr_device = str(args.asr_device)
        if asr_device == "cuda" and not torch.cuda.is_available():
            asr_device = "cpu"
        model = WhisperModel(
            str(args.asr_model),
            device=asr_device,
            compute_type=str(args.compute_type),
        )
        transcript_rows = []
        for row_index, (row, target_word, target_text) in enumerate(
            zip(selected_rows, target_words, reference_texts)
        ):
            candidate_payloads: List[Dict[str, object]] = []
            for candidate_index, candidate in enumerate(candidates):
                audio_path = (
                    args.audio_output_dir.expanduser()
                    / f"row_{row_index:04d}_{candidate_index}_{candidate.replace(' ', '_')}.wav"
                )
                result = _transcribe_wave(
                    model,
                    extracted[row_index, candidate_index],
                    sample_rate=int(args.sample_rate),
                    path=audio_path,
                    min_rms=float(args.min_rms),
                    normalize_peak=float(args.normalize_peak),
                    language=str(args.language),
                )
                score = _text_match_score(target_text, str(result["text"]))
                candidate_payloads.append(
                    {
                        "candidate": candidate,
                        "text": result["text"],
                        "match_score": score,
                        "rms_too_low": bool(result["rms_too_low"]),
                    }
                )
            transcript_rows.append(
                {
                    "row_index": row_index,
                    "window": row.window,
                    "word_index": int(row.index),
                    "truth": row.truth,
                    "target_word": target_word,
                    "target_text": target_text,
                    "target_share": float(row.target_share),
                    "active_5pct": int(row.active_5pct),
                    "mixed_prediction": mixed_predictions[row_index],
                    "candidates": candidate_payloads,
                }
            )
            print(
                f"candidate_asr_transcribed {row_index + 1}/{len(selected_rows)} "
                f"target={target_text!r}",
                flush=True,
            )
        args.transcript_cache.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.transcript_cache.expanduser().write_text(
            json.dumps(transcript_rows, indent=2),
            encoding="utf-8",
        )

    match_scores = np.asarray(
        [[float(item["match_score"]) for item in row["candidates"]] for row in transcript_rows],
        dtype=np.float32,
    )
    best_indices = np.argmax(match_scores, axis=1)
    sorted_scores = np.sort(match_scores, axis=1)
    best_scores = sorted_scores[:, -1]
    margins = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
        if match_scores.shape[1] > 1
        else np.ones(match_scores.shape[0], dtype=np.float32)
    )
    asr_predictions = [candidates[int(index)] for index in best_indices]
    best_score_router, score_thresholds = _threshold_router(
        selected_rows,
        best_scores,
        asr_predictions,
        mixed_predictions,
    )
    margin_router, margin_thresholds = _threshold_router(
        selected_rows,
        margins,
        asr_predictions,
        mixed_predictions,
    )
    learned_selector, learned_router, learned_thresholds = _learned_asr_selector(
        selected_rows,
        match_scores,
        candidates,
        mixed_predictions,
    )
    oracle_asr_or_mixed = [
        row.truth if row.truth in {mixed, asr} else mixed
        for row, mixed, asr in zip(selected_rows, mixed_predictions, asr_predictions)
    ]
    truth_positions = {speaker: index for index, speaker in enumerate(candidates)}
    truth_scores = [
        float(match_scores[index, truth_positions[row.truth]])
        for index, row in enumerate(selected_rows)
    ]
    top2_hits = 0
    for index, row in enumerate(selected_rows):
        order = np.argsort(match_scores[index])[::-1]
        top2_hits += int(row.truth in {candidates[int(item)] for item in order[:2]})

    scores = {
        "mixed_same_rows": _score_predictions("mixed_same_rows", selected_rows, mixed_predictions),
        "asr_text_argmax": _score_predictions("asr_text_argmax", selected_rows, asr_predictions),
        "asr_score_threshold_router": _score_predictions(
            "asr_score_threshold_router",
            selected_rows,
            best_score_router,
        ),
        "asr_margin_threshold_router": _score_predictions(
            "asr_margin_threshold_router",
            selected_rows,
            margin_router,
        ),
        "learned_asr_selector": _score_predictions(
            "learned_asr_selector",
            selected_rows,
            learned_selector,
        ),
        "learned_asr_router": _score_predictions(
            "learned_asr_router",
            selected_rows,
            learned_router,
        ),
        "oracle_asr_or_mixed": _score_predictions(
            "oracle_asr_or_mixed",
            selected_rows,
            oracle_asr_or_mixed,
        ),
        "asr_top2_contains_truth": {
            "name": "asr_top2_contains_truth",
            "direct": {
                "examples": len(selected_rows),
                "correct": top2_hits,
                "accuracy": (top2_hits / len(selected_rows)) if selected_rows else 0.0,
            },
        },
    }
    payload = {
        "model": "candidate_asr_attribution",
        "source_model": str(args.model_base_path.expanduser()),
        "asr_model": str(args.asr_model),
        "selected_rows": len(selected_rows),
        "selected_speakers": dict(Counter(row.truth for row in selected_rows)),
        "thresholds": {
            "score": score_thresholds,
            "margin": margin_thresholds,
            "learned": learned_thresholds,
        },
        "mean_truth_match_score": float(np.mean(truth_scores)) if truth_scores else 0.0,
        "mean_best_match_score": float(np.mean(best_scores)) if len(best_scores) else 0.0,
        "scores": scores,
        "diagnostic_rows": transcript_rows[:100],
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    for name, score in scores.items():
        direct = score["direct"]
        print(
            ",".join(
                [
                    name,
                    str(direct["examples"]),
                    f"{float(direct['accuracy']):.4f}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
