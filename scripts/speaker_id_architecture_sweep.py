from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.multitrack_eval import (  # noqa: E402
    WordSpan,
    _extract_words_from_segment_record,
    extract_words_from_jsonl,
    load_jsonl_records,
    score_word_speaker_alignment,
)
from transcriber.segment_classifier import load_classifier_dataset  # noqa: E402


CORE_SPEAKERS = (
    "Dungeon Master",
    "David Tanglethorn",
    "Leopold Magnus",
    "Kaladen Shash",
    "Cyrus Schwert",
    "Cletus Cobbington",
)


@dataclass
class EvalWindow:
    name: str
    group: str
    raw_segments: List[dict]
    reference_words: List[WordSpan]
    embeddings_by_index: Dict[int, np.ndarray]
    truth_by_index: Dict[int, str]
    duration_by_index: Dict[int, float]
    raw_label_by_index: Dict[int, str]


@dataclass
class TrainingSet:
    name: str
    embeddings: np.ndarray
    labels: List[str]


def _parse_named_path(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", maxsplit=1)
        return name.strip(), Path(path).expanduser()
    path = Path(raw).expanduser()
    return path.name, path


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    rows = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-8)


def _words_from_segments(segments: Sequence[dict]) -> List[WordSpan]:
    words: List[WordSpan] = []
    for segment in segments:
        words.extend(_extract_words_from_segment_record(segment))
    words.sort(key=lambda item: (item.start, item.end, item.text))
    return words


def _set_segment_speaker(segment: dict, speaker: str) -> None:
    segment["speaker"] = speaker
    for word in segment.get("words") or []:
        if isinstance(word, dict):
            word["speaker"] = speaker


def _score_window_predictions(
    window: EvalWindow,
    predictions_by_index: Mapping[int, str],
) -> Dict[str, object]:
    segments = copy.deepcopy(window.raw_segments)
    for index, segment in enumerate(segments):
        _set_segment_speaker(segment, str(predictions_by_index.get(index) or "unknown"))
    return score_word_speaker_alignment(window.reference_words, _words_from_segments(segments))


def _aggregate_word_scores(scores: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    reference_words = 0
    matched_words = 0
    correct_words = 0
    for score in scores:
        reference_words += int(score.get("reference_words") or 0)
        matched_words += int(score.get("matched_words") or 0)
        correct_words += int(score.get("correct_words") or 0)
    return {
        "reference_words": reference_words,
        "matched_words": matched_words,
        "correct_words": correct_words,
        "coverage": (matched_words / reference_words) if reference_words else 0.0,
        "accuracy": (correct_words / reference_words) if reference_words else 0.0,
        "matched_accuracy": (correct_words / matched_words) if matched_words else 0.0,
    }


def _segment_scores(
    windows: Sequence[EvalWindow],
    predictions_by_window: Mapping[str, Mapping[int, str]],
) -> Dict[str, object]:
    total = 0
    correct = 0
    duration_total = 0.0
    duration_correct = 0.0
    by_duration: Dict[str, Counter[str]] = defaultdict(Counter)
    buckets = (
        ("lt_0_75", 0.0, 0.75),
        ("0_75_to_1_25", 0.75, 1.25),
        ("1_25_to_2", 1.25, 2.0),
        ("gte_2", 2.0, float("inf")),
    )
    for window in windows:
        predictions = predictions_by_window.get(window.name) or {}
        for index, truth in window.truth_by_index.items():
            if index not in window.embeddings_by_index:
                continue
            predicted = str(predictions.get(index) or "unknown")
            duration = float(window.duration_by_index.get(index) or 0.0)
            is_correct = predicted == truth
            total += 1
            correct += int(is_correct)
            duration_total += duration
            duration_correct += duration if is_correct else 0.0
            for name, start, end in buckets:
                if start <= duration < end:
                    by_duration[name]["total"] += 1
                    by_duration[name]["correct"] += int(is_correct)
                    break
    return {
        "segments": total,
        "correct_segments": correct,
        "segment_accuracy": (correct / total) if total else 0.0,
        "duration_weighted_segment_accuracy": (
            duration_correct / duration_total if duration_total else 0.0
        ),
        "by_duration": {
            name: {
                "segments": int(counts["total"]),
                "correct_segments": int(counts["correct"]),
                "accuracy": (counts["correct"] / counts["total"] if counts["total"] else 0.0),
            }
            for name, counts in sorted(by_duration.items())
        },
    }


def _window_group(name: str) -> str:
    if name == "short_segment_slice/window_01_00000_00300":
        return "Session61/window_01_00000_00300"
    return name


def _load_npz_embeddings(path: Path) -> Dict[int, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {
        int(index): np.asarray(vector, dtype=np.float32)
        for index, vector in zip(payload["segment_indices"], payload["embeddings"])
    }


def load_eval_windows(prepared_root: Path, speakers: Sequence[str]) -> List[EvalWindow]:
    allowed = set(speakers)
    windows: List[EvalWindow] = []
    for raw_jsonl in sorted(prepared_root.glob("*/window_*/raw_predicted/mixed/mixed.jsonl")):
        window_dir = raw_jsonl.parents[2]
        reference_jsonl = window_dir / "reference" / "clips" / "clips.jsonl"
        purity_json = window_dir / "diarization_purity.json"
        segment_npz = window_dir / "raw_predicted" / "segment_embeddings.npz"
        if not (reference_jsonl.exists() and purity_json.exists() and segment_npz.exists()):
            continue
        raw_segments = load_jsonl_records(raw_jsonl)
        purity_records = json.loads(purity_json.read_text(encoding="utf-8")).get("records") or []
        if len(purity_records) != len(raw_segments):
            raise ValueError(
                f"Purity/segment count mismatch for {window_dir}: "
                f"{len(purity_records)} != {len(raw_segments)}"
            )
        truth_by_index: Dict[int, str] = {}
        duration_by_index: Dict[int, float] = {}
        raw_label_by_index: Dict[int, str] = {}
        for index, record in enumerate(purity_records):
            speaker = str(record.get("speaker") or "").strip()
            if speaker and speaker in allowed:
                truth_by_index[index] = speaker
            duration_by_index[index] = float(record.get("duration") or 0.0)
            raw_label_by_index[index] = str(record.get("raw_label") or "").strip()
        name = str(window_dir.relative_to(prepared_root))
        windows.append(
            EvalWindow(
                name=name,
                group=_window_group(name),
                raw_segments=raw_segments,
                reference_words=extract_words_from_jsonl(reference_jsonl),
                embeddings_by_index=_load_npz_embeddings(segment_npz),
                truth_by_index=truth_by_index,
                duration_by_index=duration_by_index,
                raw_label_by_index=raw_label_by_index,
            )
        )
    if not windows:
        raise FileNotFoundError(f"No cached eval windows found under {prepared_root}")
    return windows


def _oracle_segment_predictions(window: EvalWindow) -> Dict[int, str]:
    return dict(window.truth_by_index)


def _oracle_label_majority_predictions(window: EvalWindow) -> Dict[int, str]:
    durations_by_raw_label: Dict[str, Counter[str]] = defaultdict(Counter)
    for index, truth in window.truth_by_index.items():
        raw_label = window.raw_label_by_index.get(index)
        if not raw_label:
            continue
        durations_by_raw_label[raw_label][truth] += float(
            window.duration_by_index.get(index) or 0.0
        )
    label_map = {
        raw_label: counts.most_common(1)[0][0]
        for raw_label, counts in durations_by_raw_label.items()
        if counts
    }
    return {
        index: label_map.get(window.raw_label_by_index.get(index) or "", "unknown")
        for index in range(len(window.raw_segments))
    }


def evaluate_prediction_set(
    name: str,
    windows: Sequence[EvalWindow],
    predictions_by_window: Mapping[str, Mapping[int, str]],
) -> Dict[str, object]:
    word_summary = _aggregate_word_scores(
        _score_window_predictions(window, predictions_by_window.get(window.name) or {})
        for window in windows
    )
    segment_summary = _segment_scores(windows, predictions_by_window)
    return {
        "name": name,
        "word": word_summary,
        "segment": segment_summary,
    }


def evaluate_oracles(windows: Sequence[EvalWindow]) -> List[Dict[str, object]]:
    return [
        evaluate_prediction_set(
            "oracle_segment_dominant",
            windows,
            {window.name: _oracle_segment_predictions(window) for window in windows},
        ),
        evaluate_prediction_set(
            "oracle_local_label_majority",
            windows,
            {window.name: _oracle_label_majority_predictions(window) for window in windows},
        ),
    ]


def load_bank_training_set(bank_root: Path, profile: str, speakers: Sequence[str]) -> TrainingSet:
    profile_dir = bank_root / profile
    manifest = json.loads((profile_dir / "bank.json").read_text(encoding="utf-8"))
    matrix = np.asarray(
        np.load(profile_dir / "embeddings.npy", allow_pickle=False), dtype=np.float32
    )
    labels: List[str] = []
    vectors: List[np.ndarray] = []
    allowed = set(speakers)
    for index, entry in enumerate(manifest.get("entries") or []):
        speaker = str(entry.get("speaker") or "").strip()
        if speaker not in allowed:
            continue
        if index >= matrix.shape[0]:
            continue
        labels.append(speaker)
        vectors.append(np.asarray(matrix[index], dtype=np.float32))
    if not vectors:
        raise ValueError(f"No matching speaker embeddings found in bank profile {profile}")
    return TrainingSet(name=f"bank:{profile}", embeddings=np.vstack(vectors), labels=labels)


def load_dataset_training_set(
    name: str,
    dataset_dir: Path,
    speakers: Sequence[str],
    *,
    min_duration: float,
    min_dominant_share: float,
    max_active_speakers: int,
) -> TrainingSet:
    dataset, _summary = load_classifier_dataset(dataset_dir)
    allowed = set(speakers)
    selected: List[int] = []
    for index, label in enumerate(dataset.labels):
        if str(label) not in allowed:
            continue
        duration = float(dataset.durations[index])
        dominant_share = float(dataset.dominant_shares[index])
        active_speakers = int(dataset.active_speakers[index])
        if np.isfinite(duration) and duration < min_duration:
            continue
        if (
            np.isfinite(dominant_share)
            and dominant_share >= 0
            and dominant_share < min_dominant_share
        ):
            continue
        if (
            max_active_speakers > 0
            and active_speakers >= 0
            and active_speakers > max_active_speakers
        ):
            continue
        selected.append(index)
    if not selected:
        raise ValueError(f"No rows survived filters for dataset {dataset_dir}")
    subset = dataset.subset(selected)
    return TrainingSet(name=f"dataset:{name}", embeddings=subset.embeddings, labels=subset.labels)


def _fit_predict(
    model_name: str, train_x: np.ndarray, train_y: Sequence[str], test_x: np.ndarray
) -> List[str]:
    if test_x.shape[0] == 0:
        return []

    if model_name == "centroid_cosine":
        normalized_train = _normalize_rows(train_x)
        normalized_test = _normalize_rows(test_x)
        centroids: Dict[str, np.ndarray] = {}
        for speaker in sorted(set(train_y)):
            rows = normalized_train[
                [index for index, label in enumerate(train_y) if label == speaker]
            ]
            centroid = np.mean(rows, axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
            centroids[speaker] = centroid.astype(np.float32)
        speakers = sorted(centroids)
        centroid_matrix = np.vstack([centroids[speaker] for speaker in speakers])
        scores = normalized_test @ centroid_matrix.T
        return [speakers[int(index)] for index in np.argmax(scores, axis=1)]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if model_name == "knn1_cosine":
        model = KNeighborsClassifier(
            n_neighbors=1, weights="distance", metric="cosine", algorithm="brute"
        )
    elif model_name == "knn7_cosine":
        model = KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="cosine", algorithm="brute"
        )
    elif model_name == "knn21_cosine":
        model = KNeighborsClassifier(
            n_neighbors=21, weights="distance", metric="cosine", algorithm="brute"
        )
    elif model_name == "logreg_balanced":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
        )
    elif model_name == "logreg_c10":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000),
        )
    elif model_name == "linear_svc":
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, dual="auto"),
        )
    elif model_name == "lda_shrinkage":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=13,
            n_jobs=-1,
            max_features="sqrt",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(np.asarray(train_x, dtype=np.float32), list(train_y))
    return [str(label) for label in model.predict(np.asarray(test_x, dtype=np.float32))]


def _eval_rows_for_windows(
    windows: Sequence[EvalWindow],
) -> Tuple[np.ndarray, List[str], List[Tuple[str, int]]]:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    keys: List[Tuple[str, int]] = []
    for window in windows:
        for index, vector in sorted(window.embeddings_by_index.items()):
            truth = window.truth_by_index.get(index)
            if not truth:
                continue
            vectors.append(np.asarray(vector, dtype=np.float32))
            labels.append(truth)
            keys.append((window.name, index))
    return np.vstack(vectors), labels, keys


def evaluate_external_models(
    windows: Sequence[EvalWindow],
    training_sets: Sequence[TrainingSet],
    model_names: Sequence[str],
) -> List[Dict[str, object]]:
    test_x, _test_y, keys = _eval_rows_for_windows(windows)
    results: List[Dict[str, object]] = []
    for training_set in training_sets:
        for model_name in model_names:
            predicted = _fit_predict(
                model_name, training_set.embeddings, training_set.labels, test_x
            )
            predictions_by_window: Dict[str, Dict[int, str]] = defaultdict(dict)
            for (window_name, index), speaker in zip(keys, predicted):
                predictions_by_window[window_name][index] = speaker
            result = evaluate_prediction_set(
                f"{training_set.name}/{model_name}",
                windows,
                predictions_by_window,
            )
            result["training"] = {
                "name": training_set.name,
                "samples": int(training_set.embeddings.shape[0]),
                "speakers": dict(Counter(training_set.labels)),
            }
            results.append(result)
    return results


def evaluate_leave_group_out_models(
    windows: Sequence[EvalWindow],
    model_names: Sequence[str],
) -> List[Dict[str, object]]:
    groups = sorted({window.group for window in windows})
    results: List[Dict[str, object]] = []
    for model_name in model_names:
        predictions_by_window: Dict[str, Dict[int, str]] = defaultdict(dict)
        for group in groups:
            train_windows = [window for window in windows if window.group != group]
            test_windows = [window for window in windows if window.group == group]
            train_x, train_y, _train_keys = _eval_rows_for_windows(train_windows)
            test_x, _test_y, test_keys = _eval_rows_for_windows(test_windows)
            predicted = _fit_predict(model_name, train_x, train_y, test_x)
            for (window_name, index), speaker in zip(test_keys, predicted):
                predictions_by_window[window_name][index] = speaker
        results.append(
            evaluate_prediction_set(
                f"eval_leave_group_out/{model_name}",
                windows,
                predictions_by_window,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep closed-set speaker-ID architectures against cached flat-audio windows."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument("--bank-root", type=Path, default=Path.home() / "hf_cache" / "speaker_bank")
    parser.add_argument("--bank-profile", action="append", default=[])
    parser.add_argument("--classifier-dataset", action="append", default=[])
    parser.add_argument(
        "--models",
        default="centroid_cosine,knn1_cosine,knn7_cosine,knn21_cosine,logreg_balanced,logreg_c10,linear_svc,lda_shrinkage",
    )
    parser.add_argument("--include-extra-trees", action="store_true")
    parser.add_argument("--skip-leave-group-out", action="store_true")
    parser.add_argument("--min-train-duration", type=float, default=0.0)
    parser.add_argument("--min-train-dominant-share", type=float, default=0.0)
    parser.add_argument("--max-train-active-speakers", type=int, default=0)
    parser.add_argument("--speakers", default=",".join(CORE_SPEAKERS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    speakers = tuple(item.strip() for item in args.speakers.split(",") if item.strip())
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.include_extra_trees and "extra_trees" not in model_names:
        model_names.append("extra_trees")

    prepared_root = args.prepared_root.expanduser()
    windows = load_eval_windows(prepared_root, speakers)
    training_sets: List[TrainingSet] = []
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

    results = evaluate_oracles(windows)
    if training_sets:
        results.extend(evaluate_external_models(windows, training_sets, model_names))
    if not args.skip_leave_group_out:
        results.extend(evaluate_leave_group_out_models(windows, model_names))

    results.sort(
        key=lambda item: (
            float(item["word"]["accuracy"]),
            float(item["segment"]["segment_accuracy"]),
        ),
        reverse=True,
    )
    payload = {
        "prepared_root": str(prepared_root),
        "windows": [window.name for window in windows],
        "speakers": speakers,
        "models": model_names,
        "results": results,
    }
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("name,word_accuracy,matched_accuracy,coverage,segment_accuracy,segments")
    for result in results[:20]:
        print(
            ",".join(
                [
                    str(result["name"]),
                    f"{float(result['word']['accuracy']):.4f}",
                    f"{float(result['word']['matched_accuracy']):.4f}",
                    f"{float(result['word']['coverage']):.4f}",
                    f"{float(result['segment']['segment_accuracy']):.4f}",
                    str(result["segment"]["segments"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
