from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_architecture_sweep import (  # noqa: E402
    CORE_SPEAKERS,
    TrainingSet,
    _fit_predict,
    _parse_named_path,
    load_bank_training_set,
    load_dataset_training_set,
)
from transcriber.diarization import extract_embeddings_for_segments  # noqa: E402
from transcriber.multitrack_eval import (  # noqa: E402
    WordSpan,
    extract_words_from_jsonl,
    load_jsonl_records,
    score_word_speaker_alignment,
)


@dataclass
class WordWindow:
    name: str
    group: str
    source: str
    audio_path: Path
    reference_words: List[WordSpan]
    predicted_words: List[WordSpan]
    truths: List[str]


@dataclass
class WordEmbeddingSet:
    window: WordWindow
    embeddings: np.ndarray
    truths: List[str]
    predicted_words: List[WordSpan]


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _window_group(name: str) -> str:
    if name == "short_segment_slice/window_01_00000_00300":
        return "Session61/window_01_00000_00300"
    return name


def _predicted_words_from_segments(segments: Sequence[Mapping[str, object]]) -> List[WordSpan]:
    words: List[WordSpan] = []
    for segment in segments:
        segment_speaker = str(segment.get("speaker") or "unknown")
        for raw_word in segment.get("words") or []:
            if not isinstance(raw_word, Mapping):
                continue
            start = raw_word.get("start")
            end = raw_word.get("end")
            if start is None or end is None:
                continue
            text = str(raw_word.get("word") or raw_word.get("text") or "").strip()
            if not text:
                continue
            words.append(
                WordSpan(
                    speaker=segment_speaker,
                    start=float(start),
                    end=float(end),
                    text=text,
                )
            )
    words.sort(key=lambda item: (item.start, item.end, item.text))
    return words


def _match_truths_for_predicted_words(
    predicted_words: Sequence[WordSpan],
    reference_words: Sequence[WordSpan],
    *,
    tolerance_seconds: float,
    speakers: Sequence[str],
) -> Tuple[List[WordSpan], List[str]]:
    allowed = set(speakers)
    reference = sorted(reference_words, key=lambda item: (item.start, item.end))
    matched_words: List[WordSpan] = []
    truths: List[str] = []
    cursor = 0
    for predicted in sorted(predicted_words, key=lambda item: (item.start, item.end)):
        midpoint = predicted.midpoint
        while cursor < len(reference) and reference[cursor].end < midpoint - tolerance_seconds:
            cursor += 1
        best_match: WordSpan | None = None
        best_distance: tuple[float, float] | None = None
        probe = max(cursor - 1, 0)
        while probe < len(reference):
            ref = reference[probe]
            if ref.start > midpoint + tolerance_seconds:
                break
            if ref.start <= midpoint <= ref.end:
                interval_distance = 0.0
            else:
                interval_distance = min(abs(midpoint - ref.start), abs(midpoint - ref.end))
            if interval_distance <= tolerance_seconds:
                candidate_distance = (interval_distance, abs(ref.midpoint - midpoint))
                if best_distance is None or candidate_distance < best_distance:
                    best_match = ref
                    best_distance = candidate_distance
            probe += 1
        if best_match is None or best_match.speaker not in allowed:
            continue
        matched_words.append(predicted)
        truths.append(best_match.speaker)
    return matched_words, truths


def load_word_windows(
    prepared_root: Path,
    *,
    tolerance_seconds: float,
    speakers: Sequence[str],
    word_source: str,
) -> List[WordWindow]:
    word_source = str(word_source or "predicted").strip().lower()
    if word_source not in {"predicted", "reference"}:
        raise ValueError("word_source must be 'predicted' or 'reference'")
    windows: List[WordWindow] = []
    for raw_jsonl in sorted(prepared_root.glob("*/window_*/raw_predicted/mixed/mixed.jsonl")):
        window_dir = raw_jsonl.parents[2]
        audio_path = window_dir / "mixed.wav"
        reference_jsonl = window_dir / "reference" / "clips" / "clips.jsonl"
        if not (audio_path.exists() and reference_jsonl.exists()):
            continue
        name = str(window_dir.relative_to(prepared_root))
        reference_words = extract_words_from_jsonl(reference_jsonl)
        if word_source == "reference":
            allowed = set(speakers)
            matched_predicted = [word for word in reference_words if word.speaker in allowed]
            truths = [word.speaker for word in matched_predicted]
        else:
            predicted_words = _predicted_words_from_segments(load_jsonl_records(raw_jsonl))
            matched_predicted, truths = _match_truths_for_predicted_words(
                predicted_words,
                reference_words,
                tolerance_seconds=tolerance_seconds,
                speakers=speakers,
            )
        windows.append(
            WordWindow(
                name=name,
                group=_window_group(name),
                source=word_source,
                audio_path=audio_path,
                reference_words=reference_words,
                predicted_words=matched_predicted,
                truths=truths,
            )
        )
    if not windows:
        raise FileNotFoundError(f"No cached word windows found under {prepared_root}")
    return windows


def _embedding_cache_path(cache_root: Path, window: WordWindow, window_seconds: float) -> Path:
    return (
        cache_root
        / str(window.source)
        / _safe_name(window.name)
        / f"word_embeddings_{window_seconds:.2f}.npz"
    )


def load_or_create_word_embeddings(
    windows: Sequence[WordWindow],
    *,
    cache_root: Path,
    window_seconds: float,
    hf_token: str | None,
    diarization_model_name: str | None,
    force_device: str,
    batch_size: int,
) -> List[WordEmbeddingSet]:
    cache_root.mkdir(parents=True, exist_ok=True)
    results: List[WordEmbeddingSet] = []
    for window in windows:
        cache_path = _embedding_cache_path(cache_root, window, window_seconds)
        if cache_path.exists():
            payload = np.load(cache_path, allow_pickle=False)
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            truths = [str(item) for item in payload["truths"].tolist()]
            starts = np.asarray(payload["word_starts"], dtype=np.float32)
            ends = np.asarray(payload["word_ends"], dtype=np.float32)
            texts = [str(item) for item in payload["texts"].tolist()]
            predicted_words = [
                WordSpan(speaker="unknown", start=float(start), end=float(end), text=text)
                for start, end, text in zip(starts, ends, texts)
            ]
            results.append(
                WordEmbeddingSet(
                    window=window,
                    embeddings=embeddings,
                    truths=truths,
                    predicted_words=predicted_words,
                )
            )
            continue

        payload_segments = []
        for word, truth in zip(window.predicted_words, window.truths):
            midpoint = word.midpoint
            start = midpoint - (window_seconds / 2.0)
            end = midpoint + (window_seconds / 2.0)
            payload_segments.append((start, end, truth))
        embed_results, summary = extract_embeddings_for_segments(
            str(window.audio_path),
            payload_segments,
            hf_token=hf_token,
            diarization_model_name=diarization_model_name,
            force_device=force_device,
            quiet=True,
            pre_pad=0.0,
            post_pad=0.0,
            batch_size=batch_size,
            workers=1,
        )
        vectors: List[np.ndarray] = []
        truths: List[str] = []
        predicted_words: List[WordSpan] = []
        for result in embed_results:
            if result.index >= len(window.predicted_words):
                continue
            vectors.append(np.asarray(result.embedding, dtype=np.float32))
            truths.append(window.truths[result.index])
            predicted_words.append(window.predicted_words[result.index])
        if not vectors:
            raise RuntimeError(f"No word embeddings extracted for {window.name}: {summary}")
        embeddings = np.vstack(vectors).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            truths=np.asarray(truths),
            word_starts=np.asarray([word.start for word in predicted_words], dtype=np.float32),
            word_ends=np.asarray([word.end for word in predicted_words], dtype=np.float32),
            texts=np.asarray([word.text for word in predicted_words]),
        )
        results.append(
            WordEmbeddingSet(
                window=window,
                embeddings=embeddings,
                truths=truths,
                predicted_words=predicted_words,
            )
        )
    return results


def _score_word_predictions(
    embedding_sets: Sequence[WordEmbeddingSet],
    predictions_by_window: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    reference_words = 0
    matched_words = 0
    correct_words = 0
    direct_total = 0
    direct_correct = 0
    for item in embedding_sets:
        predicted_labels = list(predictions_by_window.get(item.window.name) or [])
        predicted_words = [
            WordSpan(
                speaker=str(label),
                start=word.start,
                end=word.end,
                text=word.text,
            )
            for word, label in zip(item.predicted_words, predicted_labels)
        ]
        metrics = score_word_speaker_alignment(item.window.reference_words, predicted_words)
        reference_words += int(metrics.get("reference_words") or 0)
        matched_words += int(metrics.get("matched_words") or 0)
        correct_words += int(metrics.get("correct_words") or 0)
        for truth, predicted in zip(item.truths, predicted_labels):
            direct_total += 1
            direct_correct += int(str(predicted) == truth)
    return {
        "reference_words": reference_words,
        "matched_words": matched_words,
        "correct_words": correct_words,
        "coverage": (matched_words / reference_words) if reference_words else 0.0,
        "accuracy": (correct_words / reference_words) if reference_words else 0.0,
        "matched_accuracy": (correct_words / matched_words) if matched_words else 0.0,
        "direct_word_examples": direct_total,
        "direct_word_accuracy": (direct_correct / direct_total) if direct_total else 0.0,
    }


def _smooth_majority(labels: Sequence[str], radius: int) -> List[str]:
    if radius <= 0:
        return list(labels)
    smoothed: List[str] = []
    for index, current in enumerate(labels):
        start = max(0, index - radius)
        end = min(len(labels), index + radius + 1)
        counts = Counter(labels[start:end])
        if not counts:
            smoothed.append(str(current))
            continue
        best_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == best_count}
        smoothed.append(str(current) if current in tied else sorted(tied)[0])
    return smoothed


def _prediction_variants(
    predictions_by_window: Mapping[str, Sequence[str]],
    smoothing_radii: Sequence[int],
) -> Iterable[Tuple[str, Dict[str, List[str]]]]:
    yielded_raw = False
    for radius in smoothing_radii:
        if radius <= 0:
            if yielded_raw:
                continue
            yielded_raw = True
            yield "raw", {name: list(labels) for name, labels in predictions_by_window.items()}
            continue
        yield (
            f"majority_{(radius * 2) + 1}",
            {
                name: _smooth_majority(labels, radius)
                for name, labels in predictions_by_window.items()
            },
        )
    if not yielded_raw:
        yield "raw", {name: list(labels) for name, labels in predictions_by_window.items()}


def _rows_for_sets(
    items: Sequence[WordEmbeddingSet],
) -> Tuple[np.ndarray, List[str], List[Tuple[str, int]]]:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    keys: List[Tuple[str, int]] = []
    for item in items:
        for index, (vector, truth) in enumerate(zip(item.embeddings, item.truths)):
            vectors.append(np.asarray(vector, dtype=np.float32))
            labels.append(truth)
            keys.append((item.window.name, index))
    return np.vstack(vectors), labels, keys


def evaluate_oracle(embedding_sets: Sequence[WordEmbeddingSet]) -> Dict[str, object]:
    return {
        "name": "oracle_word_truth",
        "word": _score_word_predictions(
            embedding_sets,
            {item.window.name: item.truths for item in embedding_sets},
        ),
    }


def evaluate_leave_group_out(
    embedding_sets: Sequence[WordEmbeddingSet],
    model_names: Sequence[str],
    smoothing_radii: Sequence[int],
) -> List[Dict[str, object]]:
    groups = sorted({item.window.group for item in embedding_sets})
    results: List[Dict[str, object]] = []
    for model_name in model_names:
        predictions_by_window: Dict[str, List[str]] = {}
        for group in groups:
            train_items = [item for item in embedding_sets if item.window.group != group]
            test_items = [item for item in embedding_sets if item.window.group == group]
            train_x, train_y, _ = _rows_for_sets(train_items)
            test_x, _test_y, keys = _rows_for_sets(test_items)
            predicted = _fit_predict(model_name, train_x, train_y, test_x)
            by_window: Dict[str, Dict[int, str]] = defaultdict(dict)
            for (window_name, index), label in zip(keys, predicted):
                by_window[window_name][index] = label
            for item in test_items:
                predictions_by_window[item.window.name] = [
                    by_window[item.window.name].get(index, "unknown")
                    for index in range(len(item.truths))
                ]
        for variant_name, variant_predictions in _prediction_variants(
            predictions_by_window,
            smoothing_radii,
        ):
            results.append(
                {
                    "name": f"word_leave_group_out/{model_name}/{variant_name}",
                    "word": _score_word_predictions(embedding_sets, variant_predictions),
                }
            )
    return results


def evaluate_external(
    embedding_sets: Sequence[WordEmbeddingSet],
    training_sets: Sequence[TrainingSet],
    model_names: Sequence[str],
    smoothing_radii: Sequence[int],
) -> List[Dict[str, object]]:
    test_x, _test_y, keys = _rows_for_sets(embedding_sets)
    results: List[Dict[str, object]] = []
    for training_set in training_sets:
        for model_name in model_names:
            predicted = _fit_predict(
                model_name, training_set.embeddings, training_set.labels, test_x
            )
            by_window: Dict[str, Dict[int, str]] = defaultdict(dict)
            for (window_name, index), label in zip(keys, predicted):
                by_window[window_name][index] = label
            predictions_by_window = {
                item.window.name: [
                    by_window[item.window.name].get(index, "unknown")
                    for index in range(len(item.truths))
                ]
                for item in embedding_sets
            }
            for variant_name, variant_predictions in _prediction_variants(
                predictions_by_window,
                smoothing_radii,
            ):
                result = {
                    "name": f"{training_set.name}/word/{model_name}/{variant_name}",
                    "word": _score_word_predictions(embedding_sets, variant_predictions),
                    "training": {
                        "name": training_set.name,
                        "samples": int(training_set.embeddings.shape[0]),
                        "speakers": dict(Counter(training_set.labels)),
                    },
                }
                results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate word-centered speaker embeddings on cached flat-audio windows."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path(".outputs/word_window_speaker_id"))
    parser.add_argument("--bank-root", type=Path, default=Path.home() / "hf_cache" / "speaker_bank")
    parser.add_argument("--bank-profile", action="append", default=[])
    parser.add_argument("--classifier-dataset", action="append", default=[])
    parser.add_argument("--window-seconds", default="1.5")
    parser.add_argument("--word-source", choices=("predicted", "reference"), default="predicted")
    parser.add_argument("--models", default="centroid_cosine,knn7_cosine,lda_shrinkage,linear_svc")
    parser.add_argument("--smoothing-radii", default="0")
    parser.add_argument("--skip-leave-group-out", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--min-train-duration", type=float, default=0.0)
    parser.add_argument("--min-train-dominant-share", type=float, default=0.0)
    parser.add_argument("--max-train-active-speakers", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--speakers", default=",".join(CORE_SPEAKERS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import os

    speakers = tuple(item.strip() for item in args.speakers.split(",") if item.strip())
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    window_seconds_values = [
        float(item.strip()) for item in str(args.window_seconds).split(",") if item.strip()
    ]
    smoothing_radii = [
        int(item.strip()) for item in str(args.smoothing_radii).split(",") if item.strip()
    ]
    word_windows = load_word_windows(
        args.prepared_root.expanduser(),
        tolerance_seconds=float(args.tolerance),
        speakers=speakers,
        word_source=str(args.word_source),
    )

    training_sets: List[TrainingSet] = []
    if not args.skip_external:
        for raw in args.bank_profile:
            profile_name = str(raw).strip()
            if profile_name:
                training_sets.append(
                    load_bank_training_set(args.bank_root.expanduser(), profile_name, speakers)
                )
        for raw in args.classifier_dataset:
            name, path = _parse_named_path(raw)
            training_sets.append(
                load_dataset_training_set(
                    name,
                    path,
                    speakers,
                    min_duration=float(args.min_train_duration),
                    min_dominant_share=float(args.min_train_dominant_share),
                    max_active_speakers=int(args.max_train_active_speakers),
                )
            )

    all_results: List[Dict[str, object]] = []
    for window_seconds in window_seconds_values:
        embedding_sets = load_or_create_word_embeddings(
            word_windows,
            cache_root=args.cache_root.expanduser(),
            window_seconds=window_seconds,
            hf_token=os.getenv(args.hf_token_env) if args.hf_token_env else None,
            diarization_model_name=args.diarization_model,
            force_device=str(args.device),
            batch_size=int(args.batch_size),
        )
        sweep_results = [evaluate_oracle(embedding_sets)]
        if not args.skip_leave_group_out:
            sweep_results.extend(
                evaluate_leave_group_out(embedding_sets, model_names, smoothing_radii)
            )
        if training_sets:
            sweep_results.extend(
                evaluate_external(embedding_sets, training_sets, model_names, smoothing_radii)
            )
        for result in sweep_results:
            result["window_seconds"] = float(window_seconds)
        all_results.extend(sweep_results)

    all_results.sort(
        key=lambda item: (
            float(item["word"]["accuracy"]),
            float(item["word"]["direct_word_accuracy"]),
        ),
        reverse=True,
    )
    payload = {
        "prepared_root": str(args.prepared_root.expanduser()),
        "cache_root": str(args.cache_root.expanduser()),
        "window_seconds": window_seconds_values,
        "word_source": str(args.word_source),
        "speakers": speakers,
        "models": model_names,
        "smoothing_radii": smoothing_radii,
        "results": all_results,
    }
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "name,window_seconds,word_accuracy,matched_accuracy,coverage,direct_word_accuracy,examples"
    )
    for result in all_results[:30]:
        word = result["word"]
        print(
            ",".join(
                [
                    str(result["name"]),
                    f"{float(result['window_seconds']):.2f}",
                    f"{float(word['accuracy']):.4f}",
                    f"{float(word['matched_accuracy']):.4f}",
                    f"{float(word['coverage']):.4f}",
                    f"{float(word['direct_word_accuracy']):.4f}",
                    str(word["direct_word_examples"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
