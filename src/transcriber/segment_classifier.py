from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from .audio_augment import AudioAugmentationConfig, build_waveform_augmenter
from .consolidate import choose_speaker
from .prep_artifacts import (
    artifact_id_for_payload,
    build_path_identity,
    build_audio_quality_metrics,
    build_source_session_speaker_breakdown,
    load_jsonl_records,
    quality_rejection_reason,
    save_candidate_pool,
    save_jsonl_records,
    summarize_quality_records,
)


DEFAULT_CLASSIFIER_MARGIN = 0.08
DEFAULT_CLASSIFIER_CONFIDENCE = 0.0
DEFAULT_CLASSIFIER_MODEL = "logreg"
DEFAULT_CLASSIFIER_TRAINING_MODE = "mixed"
MIN_FUZZY_TRANSCRIPT_MATCH_SCORE = 80.0
DEFAULT_SPEAKER_ALIASES: Dict[str, str] = {
    "zariel torgan": "David Tanglethorn",
    "dm": "Dungeon Master",
    "kaladin shash": "Kaladen Shash",
    "kaladen": "Kaladen Shash",
}

T = TypeVar("T")


class TorchLinearSoftmaxModel:
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        classes: np.ndarray,
        *,
        training_device: str,
        epochs: int,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.classes_ = np.asarray(classes, dtype=np.int64)
        self.training_device = str(training_device)
        self.epochs = int(epochs)

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        encoded: np.ndarray,
        *,
        c: float,
        max_iter: int,
        class_weight: str | None,
    ) -> "TorchLinearSoftmaxModel":
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "torch_linear requires PyTorch to be installed in the active environment"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        targets = torch.as_tensor(encoded, dtype=torch.int64, device=device)
        num_classes = int(encoded.max()) + 1
        layer = torch.nn.Linear(matrix.shape[1], num_classes, bias=True, device=device)
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

        class_weights = None
        if class_weight == "balanced":
            counts = np.bincount(encoded, minlength=num_classes).astype(np.float32)
            counts[counts <= 0.0] = 1.0
            weights = counts.sum() / counts
            weights /= max(float(weights.mean()), 1e-6)
            class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        effective_c = max(float(c), 1e-4)
        weight_decay = 1.0 / (effective_c * max(float(matrix.shape[0]), 1.0))
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.05, weight_decay=weight_decay)

        epochs_run = 0
        patience = 25
        best_loss = float("inf")
        stagnant = 0
        for epoch in range(max(50, min(int(max_iter), 400))):
            optimizer.zero_grad(set_to_none=True)
            logits = layer(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epochs_run = epoch + 1
            current_loss = float(loss.detach().cpu())
            if current_loss + 1e-5 < best_loss:
                best_loss = current_loss
                stagnant = 0
            else:
                stagnant += 1
            predictions = logits.argmax(dim=1)
            if bool(torch.equal(predictions, targets)) and stagnant >= 5:
                break
            if stagnant >= patience:
                break

        return cls(
            layer.weight.detach().cpu().numpy(),
            layer.bias.detach().cpu().numpy(),
            np.arange(num_classes, dtype=np.int64),
            training_device=device,
            epochs=epochs_run,
        )

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        vectors = np.asarray(matrix, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        logits = (vectors @ self.weights.T) + self.bias
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        denom = np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)
        return exp_logits / denom

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(matrix)
        return self.classes_[np.argmax(probabilities, axis=1)]


class ProjectedClassifierModel:
    def __init__(
        self,
        projector: Any,
        classifier: Any,
        *,
        model_name: str,
    ) -> None:
        self.projector = projector
        self.classifier = classifier
        self.model_name = str(model_name)
        self.classes_ = np.asarray(getattr(classifier, "classes_", []), dtype=np.int64)

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        encoded: np.ndarray,
        *,
        model_name: str,
        c: float,
        n_neighbors: int,
        max_iter: int,
    ) -> "ProjectedClassifierModel":
        projector = LinearDiscriminantAnalysis(
            n_components=max(
                1, min(int(encoded.max()) if encoded.size else 0, matrix.shape[1] - 1)
            ),
            solver="svd",
        )
        projected = projector.fit_transform(matrix, encoded)
        normalized = str(model_name).strip().lower()
        if normalized == "lda_knn":
            classifier = KNeighborsClassifier(
                n_neighbors=max(int(n_neighbors), 1),
                weights="distance",
                metric="cosine",
                algorithm="brute",
            )
        elif normalized == "lda_logreg":
            classifier = LogisticRegression(
                C=float(c),
                max_iter=max_iter,
                class_weight="balanced",
                solver="lbfgs",
            )
        else:
            raise ValueError(f"Unsupported projected classifier model: {model_name}")
        classifier.fit(projected, encoded)
        return cls(projector, classifier, model_name=normalized)

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        vectors = np.asarray(matrix, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return np.asarray(self.projector.transform(vectors), dtype=np.float32)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(self._transform(matrix)), dtype=np.float32)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict(self._transform(matrix)), dtype=np.int64)


@dataclass(frozen=True)
class SegmentClassifierPrediction:
    speaker: str
    score: float
    margin: float
    second_best: float
    candidates: List[Dict[str, object]]


@dataclass(frozen=True)
class ClassifierDataset:
    embeddings: np.ndarray
    labels: List[str]
    domains: List[str]
    sources: List[str]
    sessions: List[str]
    durations: np.ndarray
    dominant_shares: np.ndarray
    top1_powers: np.ndarray
    top2_powers: np.ndarray
    active_speakers: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("ClassifierDataset embeddings must be a 2D matrix")
        expected = int(matrix.shape[0])
        counts = {
            "labels": len(self.labels),
            "domains": len(self.domains),
            "sources": len(self.sources),
            "sessions": len(self.sessions),
            "durations": int(np.asarray(self.durations).shape[0]),
            "dominant_shares": int(np.asarray(self.dominant_shares).shape[0]),
            "top1_powers": int(np.asarray(self.top1_powers).shape[0]),
            "top2_powers": int(np.asarray(self.top2_powers).shape[0]),
            "active_speakers": int(np.asarray(self.active_speakers).shape[0]),
        }
        mismatched = {name: count for name, count in counts.items() if count != expected}
        if mismatched:
            raise ValueError(
                "ClassifierDataset field lengths must match embedding rows: "
                + ", ".join(f"{name}={count}" for name, count in sorted(mismatched.items()))
            )

    @property
    def samples(self) -> int:
        return int(np.asarray(self.embeddings).shape[0])

    def subset(self, indices: Sequence[int]) -> "ClassifierDataset":
        selected = np.asarray(list(indices), dtype=np.int64)
        return ClassifierDataset(
            embeddings=np.asarray(self.embeddings[selected], dtype=np.float32),
            labels=[self.labels[int(index)] for index in selected],
            domains=[self.domains[int(index)] for index in selected],
            sources=[self.sources[int(index)] for index in selected],
            sessions=[self.sessions[int(index)] for index in selected],
            durations=np.asarray(self.durations[selected], dtype=np.float32),
            dominant_shares=np.asarray(self.dominant_shares[selected], dtype=np.float32),
            top1_powers=np.asarray(self.top1_powers[selected], dtype=np.float32),
            top2_powers=np.asarray(self.top2_powers[selected], dtype=np.float32),
            active_speakers=np.asarray(self.active_speakers[selected], dtype=np.int32),
        )


class SegmentClassifier:
    MODEL_FILENAME = "segment_classifier.joblib"
    META_FILENAME = "segment_classifier.meta.json"

    def __init__(
        self,
        model: Any,
        label_encoder: LabelEncoder,
        *,
        model_name: str = DEFAULT_CLASSIFIER_MODEL,
        training_summary: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model = model
        self.label_encoder = label_encoder
        self.model_name = str(model_name or DEFAULT_CLASSIFIER_MODEL)
        self.training_summary = dict(training_summary or {})

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: Sequence[str],
        *,
        model_name: str = DEFAULT_CLASSIFIER_MODEL,
        c: float = 1.0,
        n_neighbors: int = 7,
        max_iter: int = 5000,
    ) -> "SegmentClassifier":
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Classifier training requires a 2D embedding matrix")
        if len(labels) != matrix.shape[0]:
            raise ValueError("Embedding and label counts must match")

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(list(labels))
        model_name_normalized = str(model_name or DEFAULT_CLASSIFIER_MODEL).strip().lower()
        if model_name_normalized == "logreg":
            model = LogisticRegression(
                C=float(c),
                max_iter=max_iter,
                class_weight="balanced",
                solver="lbfgs",
            )
        elif model_name_normalized == "logreg_unbalanced":
            model = LogisticRegression(
                C=float(c),
                max_iter=max_iter,
                solver="lbfgs",
            )
        elif model_name_normalized == "torch_linear":
            model = TorchLinearSoftmaxModel.fit(
                matrix,
                encoded,
                c=float(c),
                max_iter=max_iter,
                class_weight="balanced",
            )
        elif model_name_normalized == "knn":
            model = KNeighborsClassifier(
                n_neighbors=max(int(n_neighbors), 1),
                weights="distance",
                metric="cosine",
                algorithm="brute",
            )
        elif model_name_normalized in {"lda_knn", "lda_logreg"}:
            model = ProjectedClassifierModel.fit(
                matrix,
                encoded,
                model_name=model_name_normalized,
                c=float(c),
                n_neighbors=int(n_neighbors),
                max_iter=max_iter,
            )
        else:
            raise ValueError(f"Unsupported classifier model: {model_name}")
        if model_name_normalized not in {"torch_linear", "lda_knn", "lda_logreg"}:
            model.fit(matrix, encoded)

        train_accuracy = float((model.predict(matrix) == encoded).mean())
        summary = {
            "model_name": model_name_normalized,
            "model_params": {
                "c": float(c),
                "n_neighbors": int(n_neighbors),
                "max_iter": int(max_iter),
            },
            "samples": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "train_accuracy": train_accuracy,
            "speakers": {label: int(count) for label, count in Counter(labels).items()},
        }
        if model_name_normalized == "torch_linear":
            summary["model_params"]["training_device"] = model.training_device
            summary["model_params"]["epochs"] = model.epochs
        if model_name_normalized in {"lda_knn", "lda_logreg"}:
            summary["model_params"]["projected_dimensions"] = int(
                getattr(model.projector, "scalings_", np.zeros((matrix.shape[1], 1))).shape[1]
            )
        return cls(
            model,
            encoder,
            model_name=model_name_normalized,
            training_summary=summary,
        )

    def score_candidates(self, embedding: np.ndarray) -> List[Dict[str, object]]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        probabilities = self.model.predict_proba(vector)[0]
        class_ids = np.asarray(getattr(self.model, "classes_", np.arange(probabilities.shape[0])))
        order = np.argsort(probabilities)[::-1]

        candidates: List[Dict[str, object]] = []
        for rank in order:
            encoded_label = int(class_ids[int(rank)])
            speaker = str(self.label_encoder.inverse_transform([encoded_label])[0])
            candidates.append(
                {
                    "speaker": speaker,
                    "score": float(probabilities[int(rank)]),
                    "source": "segment_classifier",
                    "cluster_id": None,
                    "distance": None,
                }
            )
        return candidates

    def predict(
        self,
        embedding: np.ndarray,
        *,
        min_confidence: float = DEFAULT_CLASSIFIER_CONFIDENCE,
        min_margin: float = DEFAULT_CLASSIFIER_MARGIN,
    ) -> Optional[SegmentClassifierPrediction]:
        candidates = self.score_candidates(embedding)
        if not candidates:
            return None
        top1 = candidates[0]
        top2 = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = float(top1["score"]) - float(top2)
        if float(top1["score"]) < float(min_confidence) or margin < float(min_margin):
            return None
        return SegmentClassifierPrediction(
            speaker=str(top1["speaker"]),
            score=float(top1["score"]),
            margin=margin,
            second_best=float(top2),
            candidates=candidates,
        )

    def save(self, profile_dir: Path) -> Dict[str, str]:
        profile_dir.mkdir(parents=True, exist_ok=True)
        model_path = profile_dir / self.MODEL_FILENAME
        meta_path = profile_dir / self.META_FILENAME
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "model_name": self.model_name,
            },
            model_path,
        )
        meta_path.write_text(json.dumps(self.training_summary, indent=2), encoding="utf-8")
        return {
            "model": str(model_path),
            "meta": str(meta_path),
        }

    @classmethod
    def load(cls, profile_dir: Path) -> Optional["SegmentClassifier"]:
        model_path = profile_dir / cls.MODEL_FILENAME
        if not model_path.exists():
            return None
        payload = joblib.load(model_path)
        meta_path = profile_dir / cls.META_FILENAME
        training_summary: Dict[str, object] = {}
        if meta_path.exists():
            try:
                training_summary = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                training_summary = {}
        return cls(
            payload["model"],
            payload["label_encoder"],
            model_name=str(
                payload.get("model_name")
                or training_summary.get("model_name")
                or DEFAULT_CLASSIFIER_MODEL
            ),
            training_summary=training_summary,
        )

    def summary(self) -> Dict[str, object]:
        return dict(self.training_summary)


def load_segment_classifier(profile_dir: Path) -> Optional[SegmentClassifier]:
    return SegmentClassifier.load(profile_dir)


def _normalize_speaker_name(
    speaker: Optional[str],
    speaker_aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if not speaker:
        return speaker
    aliases = {**DEFAULT_SPEAKER_ALIASES, **(speaker_aliases or {})}
    return aliases.get(str(speaker).strip().lower(), str(speaker).strip())


def _canonicalize_speaker_label(
    speaker: Optional[str],
    *,
    speaker_aliases: Optional[Dict[str, str]] = None,
    speaker_mapping: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    normalized = _normalize_speaker_name(speaker, speaker_aliases=speaker_aliases)
    if not normalized:
        return normalized
    if speaker_mapping:
        mapped, matched = choose_speaker(str(normalized), speaker_mapping, return_match=True)
        if matched:
            return mapped
    return normalized


def _load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_TIMED_TRANSCRIPT_SPEAKER_FIRST_RE = re.compile(
    r"^(?P<speaker>.+?)\s+(?P<timestamp>\d{1,2}:\d{2}:\d{2})\s+(?P<text>.+?)\s*$"
)
_TIMED_TRANSCRIPT_TIMESTAMP_FIRST_RE = re.compile(
    r"^(?P<timestamp>\d{1,2}:\d{2}:\d{2})\s*:\s*(?P<speaker>[^:]+?)\s*:\s*(?P<text>.+?)\s*$"
)
_TIMED_TRANSCRIPT_PATTERNS = (
    _TIMED_TRANSCRIPT_SPEAKER_FIRST_RE,
    _TIMED_TRANSCRIPT_TIMESTAMP_FIRST_RE,
)


def _parse_timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds)


def _load_timed_transcript_records(
    path: Path,
    *,
    speaker_aliases: Optional[Dict[str, str]] = None,
    speaker_mapping: Optional[Dict[str, object]] = None,
) -> List[dict]:
    entries: List[Tuple[str, float, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = None
            for pattern in _TIMED_TRANSCRIPT_PATTERNS:
                match = pattern.match(line)
                if match:
                    break
            if not match:
                continue
            entries.append(
                (
                    _canonicalize_speaker_label(
                        match.group("speaker"),
                        speaker_aliases=speaker_aliases,
                        speaker_mapping=speaker_mapping,
                    )
                    or "",
                    _parse_timestamp_to_seconds(match.group("timestamp")),
                    match.group("text").strip(),
                )
            )

    records: List[dict] = []
    for index, (speaker, start, text) in enumerate(entries):
        next_start = entries[index + 1][1] if index + 1 < len(entries) else None
        if next_start is not None and next_start > start:
            end = next_start
        else:
            estimated_duration = max(1.0, min(15.0, len(text.split()) * 0.35))
            end = start + estimated_duration
        records.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "file": path.name,
            }
        )
    return records


def load_labeled_records(
    path: Path,
    *,
    speaker_aliases: Optional[Dict[str, str]] = None,
    speaker_mapping: Optional[Dict[str, object]] = None,
) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        records = _load_jsonl_records(path)
        normalized: List[dict] = []
        for record in records:
            item = dict(record)
            item["speaker"] = _canonicalize_speaker_label(
                item.get("speaker"),
                speaker_aliases=speaker_aliases,
                speaker_mapping=speaker_mapping,
            )
            normalized.append(item)
        return normalized
    return _load_timed_transcript_records(
        path,
        speaker_aliases=speaker_aliases,
        speaker_mapping=speaker_mapping,
    )


def _window_records(records: Sequence[dict], start: float, end: float) -> List[dict]:
    return [
        record
        for record in records
        if float(record.get("end") or 0.0) > start and float(record.get("start") or 0.0) < end
    ]


def _select_candidate_windows(
    records: Sequence[dict],
    *,
    window_seconds: float = 300.0,
    hop_seconds: float = 120.0,
    top_k: int = 15,
    min_speakers: int = 5,
) -> List[dict]:
    if not records:
        return []

    max_end = max(float(record.get("end") or 0.0) for record in records)
    candidates: List[dict] = []
    start = 0.0
    while start + window_seconds <= max_end + 1e-6:
        end = start + window_seconds
        window_rows = sorted(
            _window_records(records, start, end), key=lambda item: item.get("start") or 0.0
        )
        speaker_counts: Counter[str] = Counter()
        total_words = 0
        turns = 0
        previous_speaker: Optional[str] = None
        for row in window_rows:
            speaker = row.get("speaker")
            if not speaker:
                continue
            speaker = str(speaker)
            word_count = len(str(row.get("text") or "").split())
            speaker_counts[speaker] += word_count
            total_words += word_count
            if previous_speaker is not None and speaker != previous_speaker:
                turns += 1
            previous_speaker = speaker
        unique_speakers = len(speaker_counts)
        if unique_speakers >= min_speakers and total_words:
            dominant_words = max(speaker_counts.values())
            balance = 1.0 - (dominant_words / total_words)
            score = (unique_speakers * 1000.0) + (turns * 10.0) + total_words + (balance * 100.0)
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "score": score,
                    "speaker_count": unique_speakers,
                }
            )
        start += hop_seconds

    candidates.sort(key=lambda item: (item["score"], item["speaker_count"]), reverse=True)
    selected: List[dict] = []
    for candidate in candidates:
        overlaps = False
        for chosen in selected:
            if not (candidate["end"] <= chosen["start"] or candidate["start"] >= chosen["end"]):
                overlaps = True
                break
        if overlaps:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _run_ffmpeg(command: List[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _clip_audio(input_path: Path, output_path: Path, start: float, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def _mix_audio_files(inputs: Sequence[Path], output_path: Path) -> None:
    if not inputs:
        raise ValueError("No audio inputs were provided for mixing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            f"amix=inputs={len(inputs)}:normalize=0:dropout_transition=0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    _run_ffmpeg(command)


def _discover_session_transcript(zip_path: Path, search_roots: Sequence[Path]) -> Optional[Path]:
    stem = zip_path.stem
    normalized_stems = {stem, stem.replace("_", " "), stem.replace(" ", "_")}
    has_session_like_name = any("session" in item.lower() for item in normalized_stems)
    target_session_number = _extract_session_number(stem) if has_session_like_name else None
    candidate_names = []
    for candidate_root in normalized_stems:
        candidate_names.extend(
            [
                Path(candidate_root) / f"{candidate_root}.jsonl",
                Path(candidate_root) / f"{candidate_root}.txt",
                Path(f"{candidate_root}.jsonl"),
                Path(f"{candidate_root}.txt"),
            ]
        )
    for root in search_roots:
        for relative in candidate_names:
            candidate = root / relative
            if _safe_path_is_file(candidate):
                return candidate
    if target_session_number is None and not has_session_like_name:
        return None
    return _search_fuzzy_session_transcript(
        zip_path,
        search_roots,
        normalized_stems,
        target_session_number=target_session_number,
    )


def _extract_session_number(value: str) -> Optional[int]:
    match = re.search(r"session[\s_]*(\d+)", value, re.IGNORECASE)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", value)
    if digits:
        return int(digits[0])
    return None


def _score_transcript_candidate(
    candidate: Path,
    *,
    normalized_stems: Sequence[str],
    target_session_number: Optional[int],
) -> float:
    candidate_name = candidate.name.lower()
    candidate_stem = candidate.stem.lower()
    parent_name = candidate.parent.name.lower()
    score = 0.0

    if candidate.suffix.lower() == ".jsonl":
        score += 40.0
    elif candidate.suffix.lower() == ".txt":
        score += 20.0
    else:
        return float("-inf")

    if candidate_stem in normalized_stems:
        score += 200.0
    if parent_name in normalized_stems:
        score += 120.0

    candidate_session_number = _extract_session_number(candidate_stem)
    parent_session_number = _extract_session_number(parent_name)
    if target_session_number is not None:
        if candidate_session_number == target_session_number:
            score += 100.0
        if parent_session_number == target_session_number:
            score += 60.0

    for token in normalized_stems:
        lowered = token.lower()
        if lowered and lowered in candidate_name:
            score += 30.0
        if lowered and lowered in parent_name:
            score += 20.0

    penalties = {
        "copy of": 40.0,
        "overview": 30.0,
        "test": 20.0,
        "(1)": 15.0,
    }
    for needle, penalty in penalties.items():
        if needle in candidate_name:
            score -= penalty

    return score


def _search_fuzzy_session_transcript(
    zip_path: Path,
    search_roots: Sequence[Path],
    normalized_stems: Sequence[str],
    *,
    target_session_number: Optional[int] = None,
) -> Optional[Path]:
    if target_session_number is None:
        has_session_like_name = any("session" in item.lower() for item in normalized_stems)
        if has_session_like_name:
            target_session_number = _extract_session_number(zip_path.stem)
    best_path: Optional[Path] = None
    best_score = float("-inf")

    for root in search_roots:
        root = Path(root).expanduser()
        if not _safe_path_is_dir(root):
            continue

        candidate_files: List[Path] = []
        candidate_dirs = [root]
        for child in _safe_path_iterdir(root):
            if not _safe_path_is_dir(child):
                continue
            child_number = _extract_session_number(child.name)
            if child.name in normalized_stems or (
                target_session_number is not None and child_number == target_session_number
            ):
                candidate_dirs.append(child)

        for directory in candidate_dirs:
            for suffix in ("*.jsonl", "*.txt"):
                candidate_files.extend(
                    path for path in _safe_path_glob(directory, suffix) if _safe_path_is_file(path)
                )

        for candidate in candidate_files:
            score = _score_transcript_candidate(
                candidate,
                normalized_stems=[item.lower() for item in normalized_stems],
                target_session_number=target_session_number,
            )
            if score > best_score:
                best_score = score
                best_path = candidate

    if best_score < MIN_FUZZY_TRANSCRIPT_MATCH_SCORE:
        return None
    return best_path


def _normalize_session_records(
    records: Sequence[dict], speaker_mapping: Dict[str, object]
) -> List[dict]:
    normalized: List[dict] = []
    for record in records:
        item = dict(record)
        speaker = _canonicalize_speaker_label(item.get("speaker"), speaker_mapping=speaker_mapping)
        if not speaker:
            speaker = choose_speaker(str(item.get("file") or ""), speaker_mapping)
            if speaker == "unknown":
                speaker = None
        item["speaker"] = speaker
        normalized.append(item)
    return normalized


def _safe_path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_path_iterdir(path: Path) -> List[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_path_rglob(path: Path, pattern: str) -> List[Path]:
    try:
        return list(path.rglob(pattern))
    except OSError:
        return []


def _safe_path_glob(path: Path, pattern: str) -> List[Path]:
    try:
        return list(path.glob(pattern))
    except OSError:
        return []


def _safe_path_identity(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser().absolute())


def _directory_contains_audio(path: Path) -> bool:
    for candidate in _safe_path_rglob(path, "*"):
        if _safe_path_is_file(candidate) and candidate.suffix.lower() in {
            ".ogg",
            ".wav",
            ".flac",
            ".mp3",
            ".m4a",
        }:
            return True
    return False


def _find_session_zips(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.zip") if path.is_file())
    return []


def _looks_like_session_source_name(name: str) -> bool:
    candidate = str(name or "").strip()
    if not candidate:
        return False
    return re.search(r"session[\s_\-]*\d+", candidate, re.IGNORECASE) is not None


def _find_session_sources(input_path: Path) -> List[Path]:
    if _safe_path_is_file(input_path) and input_path.suffix.lower() == ".zip":
        return [input_path]
    if not _safe_path_is_dir(input_path):
        return []

    sources: List[Path] = []
    seen: set[str] = set()

    for zip_path in _safe_path_rglob(input_path, "*.zip"):
        if not _safe_path_is_file(zip_path):
            continue
        if not _looks_like_session_source_name(zip_path.stem):
            continue
        identity = _safe_path_identity(zip_path)
        if identity in seen:
            continue
        seen.add(identity)
        sources.append(zip_path)

    for child in sorted(_safe_path_iterdir(input_path), key=lambda item: item.name.lower()):
        if not _safe_path_is_dir(child):
            continue
        if not _looks_like_session_source_name(child.name):
            continue
        if not _directory_contains_audio(child):
            continue
        identity = _safe_path_identity(child)
        if identity in seen:
            continue
        seen.add(identity)
        sources.append(child)

    if not sources and _directory_contains_audio(input_path):
        sources.append(input_path)

    return sources


def _candidate_transcript_roots(input_path: Path, session_sources: Sequence[Path]) -> List[Path]:
    candidates: List[Path] = [Path("outputs"), Path(".outputs")]
    expanded_input = input_path.expanduser()
    if expanded_input.is_dir():
        candidates.append(expanded_input)
        candidates.append(expanded_input.parent / "Transcripts")
        candidates.append(expanded_input.parent.parent / "Transcripts")
    for session_source in session_sources:
        candidates.append(session_source.parent)
        candidates.append(session_source.parent / "Transcripts")
        candidates.append(session_source.parent.parent / "Transcripts")
        candidates.append(session_source.parent.parent.parent / "Transcripts")
    deduped: List[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _sample_evenly(items: Sequence[T], limit: int) -> List[T]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=limit)
    return [items[int(round(position))] for position in positions]


def _load_audio_array(path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    audio_data, sample_rate = sf.read(path)
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = audio_data.mean(axis=1)
    return np.asarray(audio_data, dtype=np.float32), int(sample_rate)


def _emit_progress(
    progress_callback: Optional[Any],
    *,
    status: str,
    session: Optional[str] = None,
    cache_hit: Optional[bool] = None,
    elapsed_seconds: Optional[float] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        status=status,
        session=session,
        cache_hit=cache_hit,
        elapsed_seconds=elapsed_seconds,
        extra=extra or {},
    )


def _build_mixed_base_signature(
    *,
    session_sources: Sequence[Path],
    transcript_search_roots: Sequence[Path],
    top_k: int,
    hop_seconds: float,
    min_speakers: int,
    min_share: float,
    min_power: float,
    min_segment_dur: float,
    max_segment_dur: float,
    window_seconds: float,
    allowed_speakers: Sequence[str],
    excluded_speakers: Sequence[str],
    diarization_model_name: Optional[str],
) -> Dict[str, object]:
    return {
        "version": 1,
        "session_sources": [build_path_identity(path, hash_contents=False) for path in session_sources],
        "transcript_search_roots": [
            build_path_identity(path, hash_contents=False) for path in transcript_search_roots
        ],
        "top_k": int(top_k),
        "hop_seconds": float(hop_seconds),
        "min_speakers": int(min_speakers),
        "min_share": float(min_share),
        "min_power": float(min_power),
        "min_segment_dur": float(min_segment_dur),
        "max_segment_dur": float(max_segment_dur),
        "window_seconds": float(window_seconds),
        "allowed_speakers": sorted(str(item) for item in allowed_speakers if str(item).strip()),
        "excluded_speakers": sorted(str(item) for item in excluded_speakers if str(item).strip()),
        "diarization_model_name": str(diarization_model_name or ""),
    }


def _prepare_extracted_session_source(
    session_source: Path,
    *,
    cache_root: Optional[Path],
    progress_callback: Optional[Any] = None,
) -> Path:
    if _safe_path_is_dir(session_source) or cache_root is None:
        return session_source

    source_identity = build_path_identity(session_source, hash_contents=False)
    cache_key = artifact_id_for_payload({"source_identity": source_identity}, length=24)
    cache_root = Path(cache_root).expanduser()
    extracted_dir = cache_root / cache_key
    manifest_path = extracted_dir / "extraction_manifest.json"
    started_at = time.monotonic()

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if manifest.get("source_identity") == source_identity:
            stem_inventory = list(manifest.get("stem_inventory") or [])
            if stem_inventory and all((extracted_dir / str(item)).exists() for item in stem_inventory):
                _emit_progress(
                    progress_callback,
                    status="extraction_cache_hit",
                    session=session_source.stem,
                    cache_hit=True,
                    elapsed_seconds=time.monotonic() - started_at,
                    extra={"cache_key": cache_key},
                )
                return extracted_dir

    cache_root.mkdir(parents=True, exist_ok=True)
    temp_dir = cache_root / f".{cache_key}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(session_source) as archive:
        archive.extractall(temp_dir)
    stem_inventory = [
        str(path.relative_to(temp_dir))
        for path in sorted(temp_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".m4a"}
    ]
    manifest_path_tmp = temp_dir / "extraction_manifest.json"
    manifest_path_tmp.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "source_identity": source_identity,
                "source_path": str(session_source),
                "extracted_dir": str(extracted_dir),
                "stem_inventory": stem_inventory,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    temp_dir.rename(extracted_dir)
    _emit_progress(
        progress_callback,
        status="extraction_cache_miss",
        session=session_source.stem,
        cache_hit=False,
        elapsed_seconds=time.monotonic() - started_at,
        extra={"cache_key": cache_key, "stem_files": len(stem_inventory)},
    )
    return extracted_dir


def _collect_labeled_stems(
    extract_dir: Path,
    *,
    speaker_mapping: Dict[str, object],
    allowed_speaker_set: set[str],
    excluded_speaker_set: set[str],
) -> List[Tuple[Path, str]]:
    stems: List[Tuple[Path, str]] = []
    for path in sorted(extract_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".ogg", ".wav", ".flac", ".mp3", ".m4a"}:
            continue
        label = choose_speaker(path.name, speaker_mapping)
        if label and label != "unknown":
            if label in excluded_speaker_set:
                continue
            if allowed_speaker_set and label not in allowed_speaker_set:
                continue
            stems.append((path, label))
    return stems


def _prepare_cached_window_inputs(
    *,
    session_source: Path,
    session_name: str,
    stems: Sequence[Tuple[Path, str]],
    window: Dict[str, object],
    window_index: int,
    cache_root: Optional[Path],
    progress_callback: Optional[Any] = None,
) -> Tuple[Path, List[Path]]:
    if cache_root is None:
        raise ValueError("window cache root is required for cached window prep")
    session_identity = build_path_identity(session_source, hash_contents=False)
    window_key = artifact_id_for_payload(
        {
            "session_identity": session_identity,
            "window": {
                "start": float(window["start"]),
                "end": float(window["end"]),
                "speaker_count": int(window["speaker_count"]),
                "window_index": int(window_index),
            },
        },
        length=24,
    )
    window_dir = Path(cache_root).expanduser() / session_name / window_key
    clips_dir = window_dir / "clips"
    mixed_path = window_dir / "mixed.wav"
    manifest_path = window_dir / "window_manifest.json"
    clip_paths = [clips_dir / f"{stem_path.stem}.wav" for stem_path, _label in stems]
    started_at = time.monotonic()

    if manifest_path.exists() and mixed_path.exists() and all(path.exists() for path in clip_paths):
        _emit_progress(
            progress_callback,
            status="window_cache_hit",
            session=session_name,
            cache_hit=True,
            elapsed_seconds=time.monotonic() - started_at,
            extra={"window_index": int(window_index), "window_key": window_key},
        )
        return mixed_path, clip_paths

    window_dir.mkdir(parents=True, exist_ok=True)
    for (stem_path, _label), clip_path in zip(stems, clip_paths):
        if not clip_path.exists():
            _clip_audio(
                stem_path,
                clip_path,
                start=float(window["start"]),
                duration=float(window["end"]) - float(window["start"]),
            )
    if not mixed_path.exists():
        _mix_audio_files(clip_paths, mixed_path)
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "session_source": str(session_source),
                "session_identity": session_identity,
                "session_name": session_name,
                "window_index": int(window_index),
                "window": {
                    "start": float(window["start"]),
                    "end": float(window["end"]),
                    "speaker_count": int(window["speaker_count"]),
                },
                "clips_dir": str(clips_dir),
                "clip_paths": [str(path) for path in clip_paths],
                "mixed_path": str(mixed_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _emit_progress(
        progress_callback,
        status="window_cache_miss",
        session=session_name,
        cache_hit=False,
        elapsed_seconds=time.monotonic() - started_at,
        extra={"window_index": int(window_index), "window_key": window_key},
    )
    return mixed_path, clip_paths


def _prepared_windows_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir).expanduser() / "prepared_windows.jsonl"


def _load_prepared_window_records(dataset_dir: Path) -> List[Dict[str, object]]:
    return load_jsonl_records(_prepared_windows_path(dataset_dir))


def _build_classifier_dataset_from_rows(
    rows: Sequence[Tuple[np.ndarray, str, str, str, str, float, float, float, float, int]],
) -> ClassifierDataset:
    if not rows:
        return ClassifierDataset(
            embeddings=np.zeros((0, 0), dtype=np.float32),
            labels=[],
            domains=[],
            sources=[],
            sessions=[],
            durations=np.zeros((0,), dtype=np.float32),
            dominant_shares=np.zeros((0,), dtype=np.float32),
            top1_powers=np.zeros((0,), dtype=np.float32),
            top2_powers=np.zeros((0,), dtype=np.float32),
            active_speakers=np.zeros((0,), dtype=np.int32),
        )
    return ClassifierDataset(
        embeddings=np.vstack([np.asarray(row[0], dtype=np.float32) for row in rows]).astype(np.float32),
        labels=[str(row[1]) for row in rows],
        domains=[str(row[2]) for row in rows],
        sources=[str(row[3]) for row in rows],
        sessions=[str(row[4]) for row in rows],
        durations=np.asarray([float(row[5]) for row in rows], dtype=np.float32),
        dominant_shares=np.asarray([float(row[6]) for row in rows], dtype=np.float32),
        top1_powers=np.asarray([float(row[7]) for row in rows], dtype=np.float32),
        top2_powers=np.asarray([float(row[8]) for row in rows], dtype=np.float32),
        active_speakers=np.asarray([int(row[9]) for row in rows], dtype=np.int32),
    )


def _dataset_from_candidate_pool(
    candidate_records: Sequence[Dict[str, object]],
    candidate_embeddings: np.ndarray,
) -> ClassifierDataset:
    selected_records: List[Dict[str, object]] = []
    selected_embeddings: List[np.ndarray] = []
    for record in candidate_records:
        if not bool(record.get("accepted")):
            continue
        raw_index = record.get("embedding_index")
        if raw_index is None:
            continue
        index = int(raw_index)
        if index < 0 or index >= candidate_embeddings.shape[0]:
            continue
        selected_records.append(dict(record))
        selected_embeddings.append(np.asarray(candidate_embeddings[index], dtype=np.float32))
    if not selected_records:
        return ClassifierDataset(
            embeddings=np.zeros((0, 0), dtype=np.float32),
            labels=[],
            domains=[],
            sources=[],
            sessions=[],
            durations=np.zeros((0,), dtype=np.float32),
            dominant_shares=np.zeros((0,), dtype=np.float32),
            top1_powers=np.zeros((0,), dtype=np.float32),
            top2_powers=np.zeros((0,), dtype=np.float32),
            active_speakers=np.zeros((0,), dtype=np.int32),
        )
    return ClassifierDataset(
        embeddings=np.vstack(selected_embeddings).astype(np.float32),
        labels=[str(record["speaker"]) for record in selected_records],
        domains=["mixed"] * len(selected_records),
        sources=["mixed_raw"] * len(selected_records),
        sessions=[str(record.get("session") or "unknown") for record in selected_records],
        durations=np.asarray(
            [float(record.get("duration") or 0.0) for record in selected_records],
            dtype=np.float32,
        ),
        dominant_shares=np.asarray(
            [float(record.get("dominant_share") or 0.0) for record in selected_records],
            dtype=np.float32,
        ),
        top1_powers=np.asarray(
            [float(record.get("top1_power") or 0.0) for record in selected_records],
            dtype=np.float32,
        ),
        top2_powers=np.asarray(
            [float(record.get("top2_power") or 0.0) for record in selected_records],
            dtype=np.float32,
        ),
        active_speakers=np.asarray(
            [int(record.get("active_speakers") or 0) for record in selected_records],
            dtype=np.int32,
        ),
    )


def materialize_classifier_dataset_from_mixed_base(
    *,
    mixed_base_dir: Path,
    dataset_cache_dir: Path,
    hf_token: Optional[str],
    force_device: str,
    quiet: bool,
    batch_size: int = 64,
    workers: int = 4,
    augmentation_profile: str = "none",
    augmentation_copies: int = 0,
    augmentation_seed: int = 13,
    include_base_samples: bool = False,
    max_samples_per_speaker: int = 0,
    diarization_model_name: Optional[str] = None,
    reuse_cached_dataset: bool = True,
    progress_callback: Optional[Any] = None,
) -> Tuple[ClassifierDataset, Dict[str, object]]:
    from .diarization import extract_embeddings_for_segments

    dataset_cache_dir = Path(dataset_cache_dir).expanduser()
    mixed_base_dir = Path(mixed_base_dir).expanduser()
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    prepared_windows_path = _prepared_windows_path(mixed_base_dir)
    mixed_base_summary_path = mixed_base_dir / "dataset_summary.json"
    if not prepared_windows_path.exists():
        raise FileNotFoundError(f"Missing prepared mixed-base windows: {prepared_windows_path}")
    if not mixed_base_summary_path.exists():
        raise FileNotFoundError(f"Missing mixed-base dataset summary: {mixed_base_summary_path}")

    augmentation_config = AudioAugmentationConfig(
        profile=str(augmentation_profile or "none"),
        copies=max(int(augmentation_copies or 0), 0),
        seed=int(augmentation_seed),
    )
    base_summary = json.loads(mixed_base_summary_path.read_text(encoding="utf-8"))
    materialization_signature = {
        "version": 1,
        "prepared_windows": build_path_identity(prepared_windows_path, hash_contents=False),
        "mixed_base_summary": build_path_identity(mixed_base_summary_path, hash_contents=False),
        "augmentation": {
            "profile": augmentation_config.profile,
            "copies": int(augmentation_config.copies),
            "seed": int(augmentation_config.seed),
        },
        "include_base_samples": bool(include_base_samples),
        "max_samples_per_speaker": int(max_samples_per_speaker),
        "diarization_model_name": str(diarization_model_name or ""),
    }
    cache_matrix_path = dataset_cache_dir / "dataset.npz"
    cache_meta_path = dataset_cache_dir / "dataset_summary.json"
    if reuse_cached_dataset and cache_matrix_path.exists() and cache_meta_path.exists():
        cached_summary = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if cached_summary.get("materialization_signature") == materialization_signature:
            return load_classifier_dataset(dataset_cache_dir)

    collected_rows: List[Tuple[np.ndarray, str, str, str, str, float, float, float, float, int]] = []
    started_at = time.monotonic()
    for window_record in _load_prepared_window_records(mixed_base_dir):
        session_name = str(window_record.get("session") or "unknown")
        mixed_path = Path(str(window_record.get("mixed_path") or "")).expanduser()
        accepted_segments = list(window_record.get("accepted_segments") or [])
        if not mixed_path.exists() or not accepted_segments:
            continue
        payload = [
            (
                float(segment["start"]),
                float(segment["end"]),
                str(segment.get("raw_label") or segment.get("speaker") or "segment"),
            )
            for segment in accepted_segments
        ]
        mixed_audio, mixed_sample_rate = _load_audio_array(mixed_path)
        if include_base_samples:
            embed_results, _ = extract_embeddings_for_segments(
                str(mixed_path),
                payload,
                hf_token=hf_token,
                diarization_model_name=diarization_model_name,
                force_device=force_device,
                quiet=quiet,
                batch_size=batch_size,
                workers=workers,
                audio_waveform=mixed_audio,
                audio_sample_rate=mixed_sample_rate,
            )
            for result in embed_results:
                if result.index >= len(accepted_segments):
                    continue
                segment = accepted_segments[result.index]
                collected_rows.append(
                    (
                        np.asarray(result.embedding, dtype=np.float32),
                        str(segment["speaker"]),
                        "mixed",
                        "mixed_raw",
                        session_name,
                        float(segment["duration"]),
                        float(segment["dominant_share"]),
                        float(segment["top1_power"]),
                        float(segment["top2_power"]),
                        int(segment["active_speakers"]),
                    )
                )
        if augmentation_config.enabled:
            for pass_index in range(augmentation_config.copies):
                waveform_augmenter = build_waveform_augmenter(
                    augmentation_config,
                    domain="mixed",
                    pass_index=pass_index,
                )
                if waveform_augmenter is None:
                    continue
                aug_results, _ = extract_embeddings_for_segments(
                    str(mixed_path),
                    payload,
                    hf_token=hf_token,
                    diarization_model_name=diarization_model_name,
                    force_device=force_device,
                    quiet=quiet,
                    batch_size=batch_size,
                    workers=workers,
                    waveform_transform=waveform_augmenter,
                    audio_waveform=mixed_audio,
                    audio_sample_rate=mixed_sample_rate,
                )
                for result in aug_results:
                    if result.index >= len(accepted_segments):
                        continue
                    segment = accepted_segments[result.index]
                    collected_rows.append(
                        (
                            np.asarray(result.embedding, dtype=np.float32),
                            str(segment["speaker"]),
                            "mixed_aug",
                            "mixed_aug",
                            session_name,
                            float(segment["duration"]),
                            float(segment["dominant_share"]),
                            float(segment["top1_power"]),
                            float(segment["top2_power"]),
                            int(segment["active_speakers"]),
                        )
                    )
        del mixed_audio
        _emit_progress(
            progress_callback,
            status="materialized_window",
            session=session_name,
            cache_hit=False,
            elapsed_seconds=time.monotonic() - started_at,
            extra={
                "accepted_segments": len(accepted_segments),
                "mixed_path": str(mixed_path),
            },
        )

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(collected_rows):
        grouped_indices[str(row[1])].append(index)
    selected_indices: List[int] = []
    for label, indices in grouped_indices.items():
        if max_samples_per_speaker > 0 and len(indices) > max_samples_per_speaker:
            indices = _sample_evenly(indices, max_samples_per_speaker)
        selected_indices.extend(indices)
    selected_indices.sort()
    selected_rows = [collected_rows[index] for index in selected_indices]
    dataset = _build_classifier_dataset_from_rows(selected_rows)
    summary: Dict[str, object] = {
        "materialization_mode": "mixed_base_derived",
        "base_artifact_id": base_summary.get("artifact_id"),
        "parent_artifacts": [base_summary.get("artifact_id")] if base_summary.get("artifact_id") else [],
        "materialization_signature": materialization_signature,
        "quality_filters": dict(base_summary.get("quality_filters") or {}),
        "source_groups": dict(base_summary.get("source_groups") or {}),
        "cache_hits": {"materialized_variant": False},
        "stage_dependencies": ["mixed_base"],
    }
    summary.update(_summarize_classifier_dataset(dataset))
    summary["breakdown"] = build_source_session_speaker_breakdown(dataset)
    artifacts = save_classifier_dataset(dataset_cache_dir, dataset, summary=summary)
    summary.setdefault("artifacts", {})
    summary["artifacts"].update(artifacts)
    for artifact_name in [
        "purity.jsonl",
        "quality_records.jsonl",
        "quality_report.json",
    ]:
        source_path = mixed_base_dir / artifact_name
        if source_path.exists():
            shutil.copy2(source_path, dataset_cache_dir / artifact_name)
    cache_meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return dataset, summary


def _purity_bucket(dominant_share: float) -> str:
    if dominant_share >= 0.80:
        return "high"
    if dominant_share >= 0.65:
        return "medium"
    return "low"


def _summarize_classifier_dataset(dataset: ClassifierDataset) -> Dict[str, object]:
    dominant_shares = np.asarray(dataset.dominant_shares, dtype=np.float32)
    durations = np.asarray(dataset.durations, dtype=np.float32)
    active_speakers = np.asarray(dataset.active_speakers, dtype=np.int32)
    finite_durations = durations[np.isfinite(durations)]
    finite_active_speakers = active_speakers[active_speakers >= 0]
    purity_buckets = Counter(
        _purity_bucket(float(value))
        for value in dominant_shares
        if math.isfinite(float(value)) and float(value) >= 0.0
    )
    return {
        "training_samples": int(dataset.samples),
        "dimensions": int(dataset.embeddings.shape[1]) if dataset.samples else 0,
        "speakers": {label: int(count) for label, count in Counter(dataset.labels).items()},
        "domains": {label: int(count) for label, count in Counter(dataset.domains).items()},
        "sources": {label: int(count) for label, count in Counter(dataset.sources).items()},
        "sessions": {label: int(count) for label, count in Counter(dataset.sessions).items()},
        "purity_buckets": dict(purity_buckets),
        "duration_stats": {
            "min": float(np.min(finite_durations)) if finite_durations.size else 0.0,
            "mean": float(np.mean(finite_durations)) if finite_durations.size else 0.0,
            "max": float(np.max(finite_durations)) if finite_durations.size else 0.0,
        },
        "active_speakers_stats": {
            "min": int(np.min(finite_active_speakers)) if finite_active_speakers.size else 0,
            "mean": float(np.mean(finite_active_speakers)) if finite_active_speakers.size else 0.0,
            "max": int(np.max(finite_active_speakers)) if finite_active_speakers.size else 0,
        },
    }


def save_classifier_dataset(
    dataset_dir: Path,
    dataset: ClassifierDataset,
    *,
    summary: Dict[str, object],
) -> Dict[str, str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.npz"
    summary_path = dataset_dir / "dataset_summary.json"
    np.savez_compressed(
        dataset_path,
        embeddings=np.asarray(dataset.embeddings, dtype=np.float32),
        labels=np.asarray(dataset.labels),
        domains=np.asarray(dataset.domains),
        sources=np.asarray(dataset.sources),
        sessions=np.asarray(dataset.sessions),
        durations=np.asarray(dataset.durations, dtype=np.float32),
        dominant_shares=np.asarray(dataset.dominant_shares, dtype=np.float32),
        top1_powers=np.asarray(dataset.top1_powers, dtype=np.float32),
        top2_powers=np.asarray(dataset.top2_powers, dtype=np.float32),
        active_speakers=np.asarray(dataset.active_speakers, dtype=np.int32),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "dataset": str(dataset_path),
        "summary": str(summary_path),
    }


def load_classifier_dataset(dataset_dir: Path) -> Tuple[ClassifierDataset, Dict[str, object]]:
    dataset_dir = Path(dataset_dir).expanduser()
    dataset_path = dataset_dir / "dataset.npz"
    summary_path = dataset_dir / "dataset_summary.json"
    payload = np.load(dataset_path, allow_pickle=False)
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset = ClassifierDataset(
        embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
        labels=payload["labels"].tolist(),
        domains=(
            payload["domains"].tolist()
            if "domains" in payload
            else ["mixed"] * int(payload["embeddings"].shape[0])
        ),
        sources=(
            payload["sources"].tolist()
            if "sources" in payload
            else ["mixed_raw"] * int(payload["embeddings"].shape[0])
        ),
        sessions=(
            payload["sessions"].tolist()
            if "sessions" in payload
            else ["unknown"] * int(payload["embeddings"].shape[0])
        ),
        durations=(
            np.asarray(payload["durations"], dtype=np.float32)
            if "durations" in payload
            else np.full(int(payload["embeddings"].shape[0]), np.nan, dtype=np.float32)
        ),
        dominant_shares=(
            np.asarray(payload["dominant_shares"], dtype=np.float32)
            if "dominant_shares" in payload
            else np.full(int(payload["embeddings"].shape[0]), np.nan, dtype=np.float32)
        ),
        top1_powers=(
            np.asarray(payload["top1_powers"], dtype=np.float32)
            if "top1_powers" in payload
            else np.full(int(payload["embeddings"].shape[0]), np.nan, dtype=np.float32)
        ),
        top2_powers=(
            np.asarray(payload["top2_powers"], dtype=np.float32)
            if "top2_powers" in payload
            else np.full(int(payload["embeddings"].shape[0]), np.nan, dtype=np.float32)
        ),
        active_speakers=(
            np.asarray(payload["active_speakers"], dtype=np.int32)
            if "active_speakers" in payload
            else np.full(int(payload["embeddings"].shape[0]), -1, dtype=np.int32)
        ),
    )
    return dataset, summary


def relabel_classifier_dataset_sources(
    dataset: ClassifierDataset, source: str
) -> ClassifierDataset:
    return ClassifierDataset(
        embeddings=np.asarray(dataset.embeddings, dtype=np.float32),
        labels=list(dataset.labels),
        domains=list(dataset.domains),
        sources=[str(source)] * dataset.samples,
        sessions=list(dataset.sessions),
        durations=np.asarray(dataset.durations, dtype=np.float32),
        dominant_shares=np.asarray(dataset.dominant_shares, dtype=np.float32),
        top1_powers=np.asarray(dataset.top1_powers, dtype=np.float32),
        top2_powers=np.asarray(dataset.top2_powers, dtype=np.float32),
        active_speakers=np.asarray(dataset.active_speakers, dtype=np.int32),
    )


def merge_classifier_datasets(datasets: Sequence[ClassifierDataset]) -> ClassifierDataset:
    merged = [dataset for dataset in datasets if dataset.samples]
    if not merged:
        raise ValueError("At least one non-empty classifier dataset is required for merging")
    return ClassifierDataset(
        embeddings=np.vstack(
            [np.asarray(dataset.embeddings, dtype=np.float32) for dataset in merged]
        ),
        labels=[label for dataset in merged for label in dataset.labels],
        domains=[label for dataset in merged for label in dataset.domains],
        sources=[label for dataset in merged for label in dataset.sources],
        sessions=[label for dataset in merged for label in dataset.sessions],
        durations=np.concatenate(
            [np.asarray(dataset.durations, dtype=np.float32) for dataset in merged]
        ).astype(np.float32),
        dominant_shares=np.concatenate(
            [np.asarray(dataset.dominant_shares, dtype=np.float32) for dataset in merged]
        ).astype(np.float32),
        top1_powers=np.concatenate(
            [np.asarray(dataset.top1_powers, dtype=np.float32) for dataset in merged]
        ).astype(np.float32),
        top2_powers=np.concatenate(
            [np.asarray(dataset.top2_powers, dtype=np.float32) for dataset in merged]
        ).astype(np.float32),
        active_speakers=np.concatenate(
            [np.asarray(dataset.active_speakers, dtype=np.int32) for dataset in merged]
        ).astype(np.int32),
    )


def balance_classifier_dataset(
    dataset: ClassifierDataset,
    *,
    target_speakers: Sequence[str],
    source_aliases: Optional[Dict[str, str]] = None,
    max_samples_per_cell: int = 500,
    cap_augmented_to_raw: bool = True,
) -> Tuple[ClassifierDataset, Dict[str, object]]:
    source_alias_map = {
        str(key): str(value)
        for key, value in (source_aliases or {}).items()
        if str(key).strip() and str(value).strip()
    }
    grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, (source, speaker) in enumerate(zip(dataset.sources, dataset.labels)):
        source_group = source_alias_map.get(str(source), str(source))
        grouped_indices[(source_group, str(speaker))].append(index)

    selected_indices: List[int] = []
    summary: Dict[str, object] = {"source_groups": {}}
    target_list = [str(speaker) for speaker in target_speakers if str(speaker).strip()]
    raw_counts = {
        speaker: len(grouped_indices.get(("mixed_raw", speaker), [])) for speaker in target_list
    }

    for source_group in sorted({key[0] for key in grouped_indices}):
        speaker_counts = {
            speaker: len(grouped_indices.get((source_group, speaker), []))
            for speaker in target_list
        }
        if any(count <= 0 for count in speaker_counts.values()):
            summary["source_groups"][source_group] = {
                "available": speaker_counts,
                "selected": {speaker: 0 for speaker in target_list},
                "cell_limit": 0,
                "skipped": True,
            }
            continue
        cell_limit = min(min(speaker_counts.values()), max(int(max_samples_per_cell), 1))
        selected_counts: Dict[str, int] = {}
        for speaker in target_list:
            limit = cell_limit
            if cap_augmented_to_raw and source_group == "mixed_aug_total":
                limit = min(limit, raw_counts.get(speaker, 0))
            chosen = _sample_evenly(grouped_indices[(source_group, speaker)], limit)
            selected_indices.extend(chosen)
            selected_counts[speaker] = len(chosen)
        summary["source_groups"][source_group] = {
            "available": speaker_counts,
            "selected": selected_counts,
            "cell_limit": cell_limit,
            "skipped": False,
        }

    selected_indices.sort()
    balanced = dataset.subset(selected_indices) if selected_indices else dataset.subset([])
    summary["before"] = _summarize_classifier_dataset(dataset)
    summary["after"] = _summarize_classifier_dataset(balanced)
    return balanced, summary


def filter_classifier_dataset(
    dataset: ClassifierDataset,
    *,
    allowed_speakers: Optional[Sequence[str]] = None,
    allowed_sources: Optional[Sequence[str]] = None,
    min_dominant_share: Optional[float] = None,
    max_active_speakers: Optional[int] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> Tuple[ClassifierDataset, Dict[str, object]]:
    allowed_speaker_set = {str(value) for value in (allowed_speakers or []) if str(value).strip()}
    allowed_source_set = {str(value) for value in (allowed_sources or []) if str(value).strip()}
    selected_indices: List[int] = []
    rejection_counts: Counter[str] = Counter()

    for index in range(dataset.samples):
        speaker = str(dataset.labels[index])
        source = str(dataset.sources[index])
        duration = float(dataset.durations[index])
        dominant_share = float(dataset.dominant_shares[index])
        active_speakers = int(dataset.active_speakers[index])

        if allowed_speaker_set and speaker not in allowed_speaker_set:
            rejection_counts["speaker"] += 1
            continue
        if allowed_source_set and source not in allowed_source_set:
            rejection_counts["source"] += 1
            continue
        if min_dominant_share is not None:
            if not math.isfinite(dominant_share) or dominant_share < float(min_dominant_share):
                rejection_counts["dominant_share"] += 1
                continue
        if max_active_speakers is not None:
            if active_speakers < 0 or active_speakers > int(max_active_speakers):
                rejection_counts["active_speakers"] += 1
                continue
        if min_duration is not None and duration < float(min_duration):
            rejection_counts["min_duration"] += 1
            continue
        if max_duration is not None and duration > float(max_duration):
            rejection_counts["max_duration"] += 1
            continue
        selected_indices.append(index)

    filtered = dataset.subset(selected_indices) if selected_indices else dataset.subset([])
    return filtered, {
        "before": _summarize_classifier_dataset(dataset),
        "after": _summarize_classifier_dataset(filtered),
        "rejections": dict(rejection_counts),
        "criteria": {
            "allowed_speakers": sorted(allowed_speaker_set),
            "allowed_sources": sorted(allowed_source_set),
            "min_dominant_share": min_dominant_share,
            "max_active_speakers": max_active_speakers,
            "min_duration": min_duration,
            "max_duration": max_duration,
        },
    }


def _build_dataset_cache_signature(
    *,
    session_sources: Sequence[Path],
    transcript_search_roots: Sequence[Path],
    training_mode: str,
    top_k: int,
    hop_seconds: float,
    min_speakers: int,
    min_share: float,
    min_power: float,
    min_segment_dur: float,
    max_segment_dur: float,
    max_samples_per_speaker: int,
    window_seconds: float,
    clean_max_records_per_speaker_per_session: int,
    clean_window_size: float,
    clean_window_stride: float,
    allowed_speakers: Sequence[str],
    excluded_speakers: Sequence[str],
    augmentation: AudioAugmentationConfig,
    diarization_model_name: Optional[str],
    include_base_samples: bool,
) -> Dict[str, object]:
    return {
        "version": 4,
        "session_sources": [_safe_path_identity(path) for path in session_sources],
        "transcript_search_roots": [_safe_path_identity(path) for path in transcript_search_roots],
        "training_mode": training_mode,
        "top_k": int(top_k),
        "hop_seconds": float(hop_seconds),
        "min_speakers": int(min_speakers),
        "min_share": float(min_share),
        "min_power": float(min_power),
        "min_segment_dur": float(min_segment_dur),
        "max_segment_dur": float(max_segment_dur),
        "max_samples_per_speaker": int(max_samples_per_speaker),
        "window_seconds": float(window_seconds),
        "clean_max_records_per_speaker_per_session": int(clean_max_records_per_speaker_per_session),
        "clean_window_size": float(clean_window_size),
        "clean_window_stride": float(clean_window_stride),
        "allowed_speakers": sorted(str(item) for item in allowed_speakers if str(item).strip()),
        "excluded_speakers": sorted(str(item) for item in excluded_speakers if str(item).strip()),
        "augmentation": {
            "profile": augmentation.profile,
            "copies": int(augmentation.copies),
            "seed": int(augmentation.seed),
        },
        "include_base_samples": bool(include_base_samples),
        "diarization_model_name": diarization_model_name or "",
    }


def _collect_clean_stem_segments(
    *,
    records: Sequence[dict],
    stem_paths_by_speaker: Dict[str, Path],
    min_segment_dur: float,
    max_segment_dur: float,
    max_records_per_speaker: int,
    window_size: float = 0.0,
    window_stride: float = 0.0,
) -> Dict[Path, List[Tuple[float, float, str]]]:
    from .segments import TrainingSegment, generate_windows_for_segments

    selected_by_speaker: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for record in records:
        speaker = str(record.get("speaker") or "").strip()
        if not speaker or speaker not in stem_paths_by_speaker:
            continue
        start = float(record.get("start") or 0.0)
        end = float(record.get("end") or 0.0)
        if end <= start:
            continue
        duration = end - start
        if duration < min_segment_dur:
            continue
        clipped_end = min(end, start + max_segment_dur)
        if clipped_end - start < min_segment_dur:
            continue
        selected_by_speaker[speaker].append((start, clipped_end, speaker))

    segments_by_path: Dict[Path, List[Tuple[float, float, str]]] = defaultdict(list)
    for speaker, items in selected_by_speaker.items():
        sampled_items = _sample_evenly(items, max_records_per_speaker)
        if window_size > 0.0:
            windows = generate_windows_for_segments(
                [
                    TrainingSegment(
                        audio_file=stem_paths_by_speaker[speaker].name,
                        start=start,
                        end=end,
                        speaker=label,
                        speaker_raw=label,
                    )
                    for start, end, label in sampled_items
                ],
                min_duration=min_segment_dur,
                max_duration=max_segment_dur,
                window_size=window_size,
                window_stride=window_stride or window_size,
            )
            sampled_items = [(window.start, window.end, window.speaker) for window in windows]
        for start, end, label in sampled_items:
            segments_by_path[stem_paths_by_speaker[speaker]].append((start, end, label))
    return segments_by_path


def train_segment_classifier_from_multitrack(
    *,
    input_path: str,
    profile_dir: Path,
    speaker_mapping: Dict[str, object],
    hf_token: Optional[str],
    force_device: str,
    quiet: bool,
    top_k: int = 15,
    hop_seconds: float = 120.0,
    min_speakers: int = 4,
    min_share: float = 0.72,
    min_power: float = 2e-4,
    min_segment_dur: float = 0.6,
    max_segment_dur: float = 20.0,
    max_samples_per_speaker: int = 700,
    batch_size: int = 64,
    workers: int = 4,
    window_seconds: float = 300.0,
    transcript_search_roots: Optional[Sequence[Path]] = None,
    speaker_aliases: Optional[Dict[str, str]] = None,
    extra_input_paths: Optional[Sequence[str]] = None,
    allowed_speakers: Optional[Sequence[str]] = None,
    excluded_speakers: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_CLASSIFIER_MODEL,
    classifier_c: float = 1.0,
    classifier_n_neighbors: int = 13,
    training_mode: str = DEFAULT_CLASSIFIER_TRAINING_MODE,
    augmentation_profile: str = "none",
    augmentation_copies: int = 0,
    augmentation_seed: int = 13,
    clean_max_records_per_speaker_per_session: int = 80,
    clean_window_size: float = 0.0,
    clean_window_stride: float = 0.0,
    dataset_cache_dir: Optional[Path] = None,
    reuse_cached_dataset: bool = True,
    diarization_model_name: Optional[str] = None,
    include_base_samples: bool = True,
    extracted_session_cache_root: Optional[Path] = None,
    window_cache_root: Optional[Path] = None,
    progress_callback: Optional[Any] = None,
    train_model: bool = True,
) -> Optional[Dict[str, object]]:
    from .diarization import diarize_audio, extract_embeddings_for_segments

    import soundfile as sf

    input_roots = [Path(input_path).expanduser()]
    for extra_path in extra_input_paths or []:
        if extra_path:
            input_roots.append(Path(extra_path).expanduser())
    session_sources: List[Path] = []
    seen_session_sources = set()
    for input_root in input_roots:
        for session_source in _find_session_sources(input_root):
            resolved = _safe_path_identity(session_source)
            if resolved in seen_session_sources:
                continue
            seen_session_sources.add(resolved)
            session_sources.append(session_source)
    if not session_sources:
        return None

    search_roots = list(transcript_search_roots or [])
    for input_root in input_roots:
        for candidate in _candidate_transcript_roots(input_root, session_sources):
            if candidate not in search_roots:
                search_roots.append(candidate)
    training_mode_normalized = (
        str(training_mode or DEFAULT_CLASSIFIER_TRAINING_MODE).strip().lower()
    )
    if training_mode_normalized not in {"mixed", "clean", "hybrid"}:
        raise ValueError(f"Unsupported classifier training mode: {training_mode}")
    use_clean_segments = training_mode_normalized in {"clean", "hybrid"}
    use_mixed_segments = training_mode_normalized in {"mixed", "hybrid"}
    augmentation_config = AudioAugmentationConfig(
        profile=str(augmentation_profile or "none"),
        copies=max(int(augmentation_copies or 0), 0),
        seed=int(augmentation_seed),
    )

    summary: Dict[str, object] = {
        "sessions": [],
        "training_mode": training_mode_normalized,
        "augmentation": {
            "profile": augmentation_config.profile,
            "copies": augmentation_config.copies,
            "seed": augmentation_config.seed,
        },
        "purity": {
            "segments": 0,
            "accepted": 0,
            "buckets": {"high": 0, "medium": 0, "low": 0},
            "rejections": {},
        },
        "quality_filters": {
            "clipping_fraction_max": 0.005,
            "silence_fraction_max": 0.80,
        },
        "source_groups": {},
    }
    collected_embeddings: List[np.ndarray] = []
    collected_labels: List[str] = []
    collected_domains: List[str] = []
    collected_sources: List[str] = []
    collected_sessions: List[str] = []
    collected_durations: List[float] = []
    collected_dominant_shares: List[float] = []
    collected_top1_powers: List[float] = []
    collected_top2_powers: List[float] = []
    collected_active_speakers: List[int] = []
    purity_records: List[Dict[str, object]] = []
    quality_records: List[Dict[str, object]] = []
    candidate_pool_records: List[Dict[str, object]] = []
    candidate_pool_embeddings: List[np.ndarray] = []
    prepared_window_records: List[Dict[str, object]] = []
    allowed_speaker_set = {
        speaker
        for speaker in (
            _canonicalize_speaker_label(
                value,
                speaker_aliases=speaker_aliases,
                speaker_mapping=speaker_mapping,
            )
            for value in (allowed_speakers or [])
        )
        if speaker
    }
    excluded_speaker_set = {
        speaker
        for speaker in (
            _canonicalize_speaker_label(
                value,
                speaker_aliases=speaker_aliases,
                speaker_mapping=speaker_mapping,
            )
            for value in (excluded_speakers or [])
        )
        if speaker
    }
    if allowed_speaker_set:
        allowed_speaker_set = {
            speaker for speaker in allowed_speaker_set if speaker not in excluded_speaker_set
        }
    cache_matrix_path = None
    cache_meta_path = None
    dataset: Optional[ClassifierDataset] = None
    cache_signature = _build_dataset_cache_signature(
        session_sources=session_sources,
        transcript_search_roots=search_roots,
        training_mode=training_mode_normalized,
        top_k=top_k,
        hop_seconds=hop_seconds,
        min_speakers=min_speakers,
        min_share=min_share,
        min_power=min_power,
        min_segment_dur=min_segment_dur,
        max_segment_dur=max_segment_dur,
        max_samples_per_speaker=max_samples_per_speaker,
        window_seconds=window_seconds,
        clean_max_records_per_speaker_per_session=clean_max_records_per_speaker_per_session,
        clean_window_size=clean_window_size,
        clean_window_stride=clean_window_stride,
        allowed_speakers=sorted(allowed_speaker_set),
        excluded_speakers=sorted(excluded_speaker_set),
        augmentation=augmentation_config,
        diarization_model_name=diarization_model_name,
        include_base_samples=include_base_samples,
    )
    if dataset_cache_dir is not None:
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_matrix_path = dataset_cache_dir / "dataset.npz"
        cache_meta_path = dataset_cache_dir / "dataset_summary.json"
        if reuse_cached_dataset and cache_matrix_path.exists() and cache_meta_path.exists():
            cached_summary = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            if cached_summary.get("dataset_cache_signature") == cache_signature:
                dataset, summary = load_classifier_dataset(dataset_cache_dir)
                summary.setdefault("cache_hits", {})
                summary["cache_hits"]["dataset"] = True
                _emit_progress(
                    progress_callback,
                    status="dataset_cache_hit",
                    elapsed_seconds=0.0,
                    cache_hit=True,
                    extra={"dataset_cache_dir": str(dataset_cache_dir)},
                )

    if dataset is None:
        with tempfile.TemporaryDirectory(prefix="segment_classifier_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for session_source in session_sources:
                session_started_at = time.monotonic()
                session_transcript = _discover_session_transcript(session_source, search_roots)
                if session_transcript is None:
                    continue
                _emit_progress(
                    progress_callback,
                    status="session_started",
                    session=session_source.stem,
                    elapsed_seconds=0.0,
                )

                records = _normalize_session_records(
                    load_labeled_records(
                        session_transcript,
                        speaker_aliases=speaker_aliases,
                        speaker_mapping=speaker_mapping,
                    ),
                    speaker_mapping,
                )
                if allowed_speaker_set:
                    records = [
                        record
                        for record in records
                        if str(record.get("speaker") or "").strip() in allowed_speaker_set
                    ]
                if excluded_speaker_set:
                    records = [
                        record
                        for record in records
                        if str(record.get("speaker") or "").strip() not in excluded_speaker_set
                    ]
                if not records:
                    continue
                windows = []
                if use_mixed_segments:
                    windows = _select_candidate_windows(
                        records,
                        window_seconds=window_seconds,
                        hop_seconds=hop_seconds,
                        top_k=top_k,
                        min_speakers=min_speakers,
                    )
                if not windows and not use_clean_segments:
                    continue
                extract_dir = _prepare_extracted_session_source(
                    session_source,
                    cache_root=extracted_session_cache_root,
                    progress_callback=progress_callback,
                )
                if not _safe_path_is_dir(extract_dir):
                    extract_dir = tmp_dir / f"stems_{session_source.stem.replace(' ', '_')}"
                    if not extract_dir.exists():
                        with zipfile.ZipFile(session_source) as archive:
                            archive.extractall(extract_dir)
                stems = _collect_labeled_stems(
                    extract_dir,
                    speaker_mapping=speaker_mapping,
                    allowed_speaker_set=allowed_speaker_set,
                    excluded_speaker_set=excluded_speaker_set,
                )
                if not stems:
                    continue

                session_stats: Dict[str, object] = {
                    "session_source": str(session_source),
                    "session_transcript": str(session_transcript),
                    "clean_segments": {
                        "used_segments": 0,
                        "speaker_counts": {},
                    },
                    "mixed_segments": {
                        "used_segments": 0,
                        "speaker_counts": {},
                        "rejections": {},
                        "purity_buckets": {"high": 0, "medium": 0, "low": 0},
                    },
                    "windows": [],
                }
                session_name = session_source.stem
                stem_paths_by_speaker = {
                    label: path for path, label in stems if label and label != "unknown"
                }

                if use_clean_segments and stem_paths_by_speaker:
                    clean_segments_by_path = _collect_clean_stem_segments(
                        records=records,
                        stem_paths_by_speaker=stem_paths_by_speaker,
                        min_segment_dur=min_segment_dur,
                        max_segment_dur=max_segment_dur,
                        max_records_per_speaker=clean_max_records_per_speaker_per_session,
                        window_size=clean_window_size,
                        window_stride=clean_window_stride,
                    )
                    clean_counts: Counter[str] = Counter()
                    for stem_path, payload in clean_segments_by_path.items():
                        waveform, sample_rate = _load_audio_array(stem_path)
                        accepted_payload: List[Tuple[float, float, str]] = []
                        for start, end, label in payload:
                            start_index = max(0, int(math.floor(float(start) * sample_rate)))
                            end_index = max(start_index + 1, int(math.ceil(float(end) * sample_rate)))
                            metrics = build_audio_quality_metrics(
                                waveform[start_index : min(end_index, waveform.shape[0])],
                                sample_rate,
                            )
                            qa_rejection = quality_rejection_reason(
                                metrics,
                                min_duration=min_segment_dur,
                                max_duration=max_segment_dur,
                            )
                            quality_records.append(
                                {
                                    "session": session_name,
                                    "window_index": None,
                                    "start": float(start),
                                    "end": float(end),
                                    "speaker": str(label),
                                    "raw_label": str(label),
                                    "source": "clean_raw",
                                    **metrics,
                                    "qa_rejection": qa_rejection,
                                }
                            )
                            if qa_rejection is not None:
                                continue
                            accepted_payload.append((start, end, label))
                        if include_base_samples and accepted_payload:
                            embed_results, _ = extract_embeddings_for_segments(
                                str(stem_path),
                                accepted_payload,
                                hf_token=hf_token,
                                diarization_model_name=diarization_model_name,
                                force_device=force_device,
                                quiet=quiet,
                                batch_size=batch_size,
                                workers=workers,
                                audio_waveform=waveform,
                                audio_sample_rate=sample_rate,
                            )
                            for result in embed_results:
                                if result.index >= len(accepted_payload):
                                    continue
                                start, end, label = accepted_payload[result.index]
                                collected_embeddings.append(
                                    np.asarray(result.embedding, dtype=np.float32)
                                )
                                collected_labels.append(label)
                                collected_domains.append("clean")
                                collected_sources.append("clean_raw")
                                collected_sessions.append(session_name)
                                collected_durations.append(float(end - start))
                                collected_dominant_shares.append(float("nan"))
                                collected_top1_powers.append(float("nan"))
                                collected_top2_powers.append(float("nan"))
                                collected_active_speakers.append(-1)
                                clean_counts[label] += 1
                        if augmentation_config.enabled:
                            for pass_index in range(augmentation_config.copies):
                                waveform_augmenter = build_waveform_augmenter(
                                    augmentation_config,
                                    domain="clean",
                                    pass_index=pass_index,
                                )
                                if waveform_augmenter is None or not accepted_payload:
                                    continue
                                aug_results, _ = extract_embeddings_for_segments(
                                    str(stem_path),
                                    accepted_payload,
                                    hf_token=hf_token,
                                    diarization_model_name=diarization_model_name,
                                    force_device=force_device,
                                    quiet=quiet,
                                    batch_size=batch_size,
                                    workers=workers,
                                    waveform_transform=waveform_augmenter,
                                    audio_waveform=waveform,
                                    audio_sample_rate=sample_rate,
                                )
                                for result in aug_results:
                                    if result.index >= len(accepted_payload):
                                        continue
                                    start, end, label = accepted_payload[result.index]
                                    collected_embeddings.append(
                                        np.asarray(result.embedding, dtype=np.float32)
                                    )
                                    collected_labels.append(label)
                                    collected_domains.append("clean_aug")
                                    collected_sources.append("clean_aug")
                                    collected_sessions.append(session_name)
                                    collected_durations.append(float(end - start))
                                    collected_dominant_shares.append(float("nan"))
                                    collected_top1_powers.append(float("nan"))
                                    collected_top2_powers.append(float("nan"))
                                    collected_active_speakers.append(-1)
                                    clean_counts[label] += 1
                        del waveform
                    session_stats["clean_segments"] = {
                        "used_segments": int(sum(clean_counts.values())),
                        "speaker_counts": {
                            label: int(count) for label, count in clean_counts.items()
                        },
                    }

                if not use_mixed_segments:
                    summary["sessions"].append(session_stats)
                    _emit_progress(
                        progress_callback,
                        status="session_completed",
                        session=session_name,
                        elapsed_seconds=time.monotonic() - session_started_at,
                        extra={"windows": 0, "mode": training_mode_normalized},
                    )
                    continue

                for index, window in enumerate(windows, start=1):
                    window_cache_base = (
                        Path(window_cache_root).expanduser()
                        if window_cache_root is not None
                        else tmp_dir / "window_cache"
                    )
                    mixed_path, clip_paths = _prepare_cached_window_inputs(
                        session_source=session_source,
                        session_name=session_name,
                        stems=stems,
                        window=window,
                        window_index=index,
                        cache_root=window_cache_base,
                        progress_callback=progress_callback,
                    )
                    stem_arrays: List[Tuple[str, np.ndarray]] = []
                    for (stem_path, label), clip_path in zip(stems, clip_paths):
                        audio_data, sample_rate = sf.read(clip_path)
                        if getattr(audio_data, "ndim", 1) > 1:
                            audio_data = audio_data.mean(axis=1)
                        if sample_rate != 16000:
                            raise RuntimeError(
                                f"Expected 16 kHz clips, found {sample_rate} for {clip_path}"
                            )
                        stem_arrays.append((label, np.asarray(audio_data, dtype=np.float32)))
                    mixed_audio, mixed_sample_rate = _load_audio_array(mixed_path)
                    if mixed_sample_rate != 16000:
                        raise RuntimeError(
                            f"Expected 16 kHz mixed audio, found {mixed_sample_rate} for {mixed_path}"
                        )
                    diarization = diarize_audio(
                        str(mixed_path),
                        model_name=diarization_model_name,
                        hf_token=hf_token,
                        min_speakers=int(window["speaker_count"]),
                        max_speakers=int(window["speaker_count"]),
                        device=force_device,
                    )

                    window_candidates: List[Dict[str, object]] = []
                    per_window_rejections: Counter[str] = Counter()
                    per_window_buckets: Counter[str] = Counter()
                    diar_turns = diarization.exclusive_segments or diarization.segments
                    for turn in diar_turns:
                        start = float(turn.start)
                        end = float(turn.end)
                        duration = end - start
                        raw_label = str(turn.speaker)
                        start_index = max(0, int(math.floor(start * 16000)))
                        end_index = max(start_index + 1, int(math.ceil(end * 16000)))

                        powers: Counter[str] = Counter()
                        for label, array in stem_arrays:
                            segment = array[start_index : min(end_index, array.shape[0])]
                            if segment.size:
                                powers[label] += float(np.mean(segment * segment))

                        total_power = float(sum(powers.values()))
                        top_label = None
                        second_label = None
                        top_power = 0.0
                        second_power = 0.0
                        dominance = 0.0
                        active_speakers = sum(1 for value in powers.values() if value >= min_power)
                        if total_power > 0.0 and powers:
                            power_ranking = powers.most_common(2)
                            top_label, top_power = power_ranking[0]
                            if len(power_ranking) > 1:
                                second_label, second_power = power_ranking[1]
                            dominance = top_power / total_power
                        purity_bucket = _purity_bucket(dominance)
                        per_window_buckets[purity_bucket] += 1
                        summary["purity"]["segments"] = int(summary["purity"]["segments"]) + 1
                        summary["purity"]["buckets"][purity_bucket] = (
                            int(summary["purity"]["buckets"][purity_bucket]) + 1
                        )

                        purity_rejection = None
                        if duration < min_segment_dur:
                            purity_rejection = "too_short"
                        elif duration > max_segment_dur:
                            purity_rejection = "too_long"
                        elif total_power <= 0.0:
                            purity_rejection = "no_power"
                        elif top_power < min_power:
                            purity_rejection = "low_power"
                        elif dominance < min_share:
                            purity_rejection = "low_share"
                        elif top_power <= second_power:
                            purity_rejection = "not_dominant"

                        metrics = build_audio_quality_metrics(
                            mixed_audio[start_index : min(end_index, mixed_audio.shape[0])],
                            mixed_sample_rate,
                        )
                        qa_rejection = quality_rejection_reason(
                            metrics,
                            min_duration=min_segment_dur,
                            max_duration=max_segment_dur,
                        )
                        rejection = qa_rejection or purity_rejection

                        diagnostic = {
                            "session": session_name,
                            "window_index": index,
                            "start": start,
                            "end": end,
                            "duration": duration,
                            "raw_label": raw_label,
                            "speaker": top_label,
                            "second_speaker": second_label,
                            "dominant_share": dominance,
                            "top1_power": top_power,
                            "top2_power": second_power,
                            "power_gap": top_power - second_power,
                            "active_speakers": active_speakers,
                            "bucket": purity_bucket,
                            "source": "mixed_raw",
                            **metrics,
                            "qa_rejection": qa_rejection,
                            "purity_rejection": purity_rejection,
                            "accepted": rejection is None,
                            "rejection": rejection,
                        }
                        purity_records.append(diagnostic)
                        quality_records.append(diagnostic)
                        window_candidates.append(diagnostic)
                        if purity_rejection is not None:
                            summary["purity"]["rejections"][purity_rejection] = (
                                int(summary["purity"]["rejections"].get(purity_rejection, 0)) + 1
                            )
                        else:
                            summary["purity"]["accepted"] = int(summary["purity"]["accepted"]) + 1
                        if rejection is not None:
                            per_window_rejections[rejection] += 1
                            continue

                    candidate_payload: List[Tuple[float, float, str]] = []
                    candidate_meta: List[Dict[str, object]] = []
                    for candidate in window_candidates:
                        if not candidate.get("speaker"):
                            continue
                        if candidate.get("qa_rejection") is not None:
                            continue
                        if float(candidate.get("top1_power") or 0.0) < min_power:
                            continue
                        candidate_payload.append(
                            (
                                float(candidate["start"]),
                                float(candidate["end"]),
                                str(candidate["raw_label"]),
                            )
                        )
                        candidate_meta.append(candidate)

                    if not candidate_meta:
                        session_stats["windows"].append(
                            {
                                "index": index,
                                "used_segments": 0,
                                "speaker_counts": {},
                                "rejections": dict(per_window_rejections),
                                "purity_buckets": dict(per_window_buckets),
                            }
                        )
                        continue

                    per_window_counts: Counter[str] = Counter()
                    embed_results, _ = extract_embeddings_for_segments(
                        str(mixed_path),
                        candidate_payload,
                        hf_token=hf_token,
                        diarization_model_name=diarization_model_name,
                        force_device=force_device,
                        quiet=quiet,
                        batch_size=batch_size,
                        workers=workers,
                        audio_waveform=mixed_audio,
                        audio_sample_rate=mixed_sample_rate,
                    )
                    accepted_payload: List[Tuple[float, float, str]] = []
                    accepted_meta: List[Dict[str, object]] = []
                    for result in embed_results:
                        if result.index >= len(candidate_meta):
                            continue
                        candidate = candidate_meta[result.index]
                        embedding = np.asarray(result.embedding, dtype=np.float32)
                        candidate_pool_records.append(dict(candidate))
                        candidate_pool_embeddings.append(embedding)
                        if not bool(candidate.get("accepted")):
                            continue
                        accepted_payload.append(candidate_payload[result.index])
                        accepted_meta.append(candidate)
                        if not include_base_samples:
                            continue
                        label = str(candidate["speaker"])
                        collected_embeddings.append(embedding)
                        collected_labels.append(label)
                        collected_domains.append("mixed")
                        collected_sources.append("mixed_raw")
                        collected_sessions.append(session_name)
                        collected_durations.append(float(candidate["duration"]))
                        collected_dominant_shares.append(float(candidate["dominant_share"]))
                        collected_top1_powers.append(float(candidate["top1_power"]))
                        collected_top2_powers.append(float(candidate["top2_power"]))
                        collected_active_speakers.append(int(candidate["active_speakers"]))
                        per_window_counts[label] += 1

                    if augmentation_config.enabled and accepted_payload:
                        for pass_index in range(augmentation_config.copies):
                            waveform_augmenter = build_waveform_augmenter(
                                augmentation_config,
                                domain="mixed",
                                pass_index=pass_index,
                            )
                            if waveform_augmenter is None:
                                continue
                            aug_results, _ = extract_embeddings_for_segments(
                                str(mixed_path),
                                accepted_payload,
                                hf_token=hf_token,
                                diarization_model_name=diarization_model_name,
                                force_device=force_device,
                                quiet=quiet,
                                batch_size=batch_size,
                                workers=workers,
                                waveform_transform=waveform_augmenter,
                                audio_waveform=mixed_audio,
                                audio_sample_rate=mixed_sample_rate,
                            )
                            for result in aug_results:
                                if result.index >= len(accepted_meta):
                                    continue
                                candidate = accepted_meta[result.index]
                                collected_embeddings.append(
                                    np.asarray(result.embedding, dtype=np.float32)
                                )
                                collected_labels.append(str(candidate["speaker"]))
                                collected_domains.append("mixed_aug")
                                collected_sources.append("mixed_aug")
                                collected_sessions.append(session_name)
                                collected_durations.append(float(candidate["duration"]))
                                collected_dominant_shares.append(float(candidate["dominant_share"]))
                                collected_top1_powers.append(float(candidate["top1_power"]))
                                collected_top2_powers.append(float(candidate["top2_power"]))
                                collected_active_speakers.append(int(candidate["active_speakers"]))
                                per_window_counts[str(candidate["speaker"])] += 1

                    if accepted_meta:
                        prepared_window_records.append(
                            {
                                "session": session_name,
                                "session_source": str(session_source),
                                "window_index": int(index),
                                "window": {
                                    "start": float(window["start"]),
                                    "end": float(window["end"]),
                                    "speaker_count": int(window["speaker_count"]),
                                },
                                "mixed_path": str(mixed_path),
                                "clip_paths": [str(path) for path in clip_paths],
                                "accepted_segments": [
                                    {
                                        "start": float(candidate["start"]),
                                        "end": float(candidate["end"]),
                                        "duration": float(candidate["duration"]),
                                        "speaker": str(candidate["speaker"]),
                                        "raw_label": str(candidate["raw_label"]),
                                        "dominant_share": float(candidate["dominant_share"]),
                                        "top1_power": float(candidate["top1_power"]),
                                        "top2_power": float(candidate["top2_power"]),
                                        "active_speakers": int(candidate["active_speakers"]),
                                    }
                                    for candidate in accepted_meta
                                ],
                            }
                        )

                    session_stats["mixed_segments"]["used_segments"] = int(
                        session_stats["mixed_segments"]["used_segments"]
                    ) + int(sum(per_window_counts.values()))
                    session_stats["mixed_segments"]["speaker_counts"] = {
                        label: int(count)
                        for label, count in (
                            Counter(session_stats["mixed_segments"]["speaker_counts"])
                            + per_window_counts
                        ).items()
                    }
                    session_stats["mixed_segments"]["rejections"] = {
                        label: int(count)
                        for label, count in (
                            Counter(session_stats["mixed_segments"]["rejections"])
                            + per_window_rejections
                        ).items()
                    }
                    session_stats["mixed_segments"]["purity_buckets"] = {
                        label: int(count)
                        for label, count in (
                            Counter(session_stats["mixed_segments"]["purity_buckets"])
                            + per_window_buckets
                        ).items()
                    }
                    session_stats["windows"].append(
                        {
                            "index": index,
                            "used_segments": int(sum(per_window_counts.values())),
                            "speaker_counts": {
                                label: int(count) for label, count in per_window_counts.items()
                            },
                            "rejections": dict(per_window_rejections),
                            "purity_buckets": dict(per_window_buckets),
                        }
                    )
                    del mixed_audio

                summary["sessions"].append(session_stats)
                _emit_progress(
                    progress_callback,
                    status="session_completed",
                    session=session_name,
                    elapsed_seconds=time.monotonic() - session_started_at,
                    extra={"windows": len(windows), "mode": training_mode_normalized},
                )

        if not collected_embeddings:
            return None

        grouped_indices: Dict[str, List[int]] = defaultdict(list)
        for index, label in enumerate(collected_labels):
            grouped_indices[label].append(index)

        selected_indices: List[int] = []
        for label, indices in grouped_indices.items():
            if max_samples_per_speaker > 0 and len(indices) > max_samples_per_speaker:
                indices = _sample_evenly(indices, max_samples_per_speaker)
            selected_indices.extend(indices)
        selected_indices.sort()

        dataset = ClassifierDataset(
            embeddings=np.vstack(
                [collected_embeddings[index] for index in selected_indices]
            ).astype(np.float32),
            labels=[collected_labels[index] for index in selected_indices],
            domains=[collected_domains[index] for index in selected_indices],
            sources=[collected_sources[index] for index in selected_indices],
            sessions=[collected_sessions[index] for index in selected_indices],
            durations=np.asarray(
                [collected_durations[index] for index in selected_indices], dtype=np.float32
            ),
            dominant_shares=np.asarray(
                [collected_dominant_shares[index] for index in selected_indices], dtype=np.float32
            ),
            top1_powers=np.asarray(
                [collected_top1_powers[index] for index in selected_indices], dtype=np.float32
            ),
            top2_powers=np.asarray(
                [collected_top2_powers[index] for index in selected_indices], dtype=np.float32
            ),
            active_speakers=np.asarray(
                [collected_active_speakers[index] for index in selected_indices], dtype=np.int32
            ),
        )
        summary["dataset_cache_signature"] = cache_signature
        summary["breakdown"] = build_source_session_speaker_breakdown(dataset)
        summary["quality"] = summarize_quality_records(
            quality_records,
            clipping_fraction_max=float(summary["quality_filters"]["clipping_fraction_max"]),
            silence_fraction_max=float(summary["quality_filters"]["silence_fraction_max"]),
        )
        summary.setdefault("cache_hits", {})
        summary["cache_hits"]["dataset"] = False
        if cache_matrix_path is not None and cache_meta_path is not None:
            summary.update(_summarize_classifier_dataset(dataset))
            artifacts = save_classifier_dataset(dataset_cache_dir, dataset, summary=summary)
            summary.setdefault("artifacts", {})
            summary["artifacts"].update(artifacts)
            if purity_records:
                purity_path = dataset_cache_dir / "purity.jsonl"
                with purity_path.open("w", encoding="utf-8") as handle:
                    for record in purity_records:
                        handle.write(json.dumps(record) + "\n")
                summary["artifacts"]["purity"] = str(purity_path)
            if quality_records:
                quality_records_path = dataset_cache_dir / "quality_records.jsonl"
                with quality_records_path.open("w", encoding="utf-8") as handle:
                    for record in quality_records:
                        handle.write(json.dumps(record) + "\n")
                quality_report_path = dataset_cache_dir / "quality_report.json"
                quality_report_path.write_text(
                    json.dumps(summary["quality"], indent=2),
                    encoding="utf-8",
                )
                summary["artifacts"]["quality_records"] = str(quality_records_path)
                summary["artifacts"]["quality_report"] = str(quality_report_path)
            if candidate_pool_records and candidate_pool_embeddings:
                summary["artifacts"].update(
                    save_candidate_pool(
                        dataset_cache_dir,
                        records=candidate_pool_records,
                        embeddings=candidate_pool_embeddings,
                    )
                )
            if prepared_window_records:
                prepared_windows_path = save_jsonl_records(
                    _prepared_windows_path(dataset_cache_dir),
                    prepared_window_records,
                )
                summary["artifacts"]["prepared_windows"] = str(prepared_windows_path)
            cache_meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if dataset is None or not dataset.labels:
        return None

    summary.update(_summarize_classifier_dataset(dataset))
    summary.setdefault("breakdown", build_source_session_speaker_breakdown(dataset))
    summary["model_name"] = model_name
    if not train_model:
        return summary

    classifier = SegmentClassifier.fit(
        np.asarray(dataset.embeddings, dtype=np.float32),
        dataset.labels,
        model_name=model_name,
        c=classifier_c,
        n_neighbors=classifier_n_neighbors,
    )
    artifacts = classifier.save(profile_dir)

    summary["artifacts"] = artifacts
    summary["classifier"] = classifier.summary()
    return summary


def build_classifier_dataset_from_multitrack(
    *,
    input_path: str,
    dataset_cache_dir: Path,
    **kwargs: object,
) -> Optional[Tuple[ClassifierDataset, Dict[str, object]]]:
    dataset_cache_dir = Path(dataset_cache_dir).expanduser()
    with tempfile.TemporaryDirectory(prefix="segment_classifier_profile_") as temp_dir_name:
        profile_dir = Path(temp_dir_name) / "profile"
        summary = train_segment_classifier_from_multitrack(
            input_path=input_path,
            profile_dir=profile_dir,
            dataset_cache_dir=dataset_cache_dir,
            train_model=False,
            **kwargs,
        )
    if summary is None:
        return None
    dataset, cached_summary = load_classifier_dataset(dataset_cache_dir)
    return dataset, cached_summary


def build_classifier_dataset_from_bank(
    *,
    profile_dir: Path,
    speaker_mapping: Optional[Dict[str, object]] = None,
    speaker_aliases: Optional[Dict[str, str]] = None,
    allowed_speakers: Optional[Sequence[str]] = None,
    excluded_speakers: Optional[Sequence[str]] = None,
    max_samples_per_speaker: Optional[int] = None,
) -> Optional[Tuple[ClassifierDataset, Dict[str, object]]]:
    profile_dir = Path(profile_dir).expanduser()
    bank_path = profile_dir / "bank.json"
    embeddings_path = profile_dir / "embeddings.npy"
    if not bank_path.exists() or not embeddings_path.exists():
        return None

    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    entries = list(payload.get("entries") or [])
    if not entries:
        return None

    matrix = np.asarray(np.load(embeddings_path), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(entries):
        raise ValueError(
            f"Speaker bank embedding count mismatch for {profile_dir}: "
            f"{matrix.shape[0]} vectors vs {len(entries)} entries"
        )

    allowed_speaker_set = {
        speaker
        for speaker in (
            _canonicalize_speaker_label(
                value,
                speaker_aliases=speaker_aliases,
                speaker_mapping=speaker_mapping,
            )
            for value in (allowed_speakers or [])
        )
        if speaker
    }
    excluded_speaker_set = {
        speaker
        for speaker in (
            _canonicalize_speaker_label(
                value,
                speaker_aliases=speaker_aliases,
                speaker_mapping=speaker_mapping,
            )
            for value in (excluded_speakers or [])
        )
        if speaker
    }
    if allowed_speaker_set:
        allowed_speaker_set = {
            speaker for speaker in allowed_speaker_set if speaker not in excluded_speaker_set
        }

    label_by_index: Dict[int, str] = {}
    for index, entry in enumerate(entries):
        speaker = _canonicalize_speaker_label(
            entry.get("speaker"),
            speaker_aliases=speaker_aliases,
            speaker_mapping=speaker_mapping,
        )
        if not speaker:
            continue
        if excluded_speaker_set and speaker in excluded_speaker_set:
            continue
        if allowed_speaker_set and speaker not in allowed_speaker_set:
            continue
        label_by_index[index] = speaker

    if not label_by_index:
        return None

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for matrix_index, label in label_by_index.items():
        grouped_indices[label].append(matrix_index)

    sampled_indices: List[int] = []
    for label, indices in grouped_indices.items():
        if (
            max_samples_per_speaker is not None
            and max_samples_per_speaker > 0
            and len(indices) > max_samples_per_speaker
        ):
            indices = _sample_evenly(indices, max_samples_per_speaker)
        sampled_indices.extend(indices)
    sampled_indices.sort()

    dataset = ClassifierDataset(
        embeddings=np.asarray(matrix[sampled_indices], dtype=np.float32),
        labels=[label_by_index[index] for index in sampled_indices],
        domains=["bank"] * len(sampled_indices),
        sources=["bank"] * len(sampled_indices),
        sessions=[
            str(
                entries[index].get("source")
                or entries[index].get("extra", {}).get("source")
                or "speaker_bank"
            )
            for index in sampled_indices
        ],
        durations=np.full(len(sampled_indices), np.nan, dtype=np.float32),
        dominant_shares=np.full(len(sampled_indices), np.nan, dtype=np.float32),
        top1_powers=np.full(len(sampled_indices), np.nan, dtype=np.float32),
        top2_powers=np.full(len(sampled_indices), np.nan, dtype=np.float32),
        active_speakers=np.full(len(sampled_indices), -1, dtype=np.int32),
    )
    summary = {
        "source": "speaker_bank",
        **_summarize_classifier_dataset(dataset),
    }
    return dataset, summary


def train_segment_classifier_from_bank(
    *,
    profile_dir: Path,
    speaker_mapping: Optional[Dict[str, object]] = None,
    speaker_aliases: Optional[Dict[str, str]] = None,
    allowed_speakers: Optional[Sequence[str]] = None,
    excluded_speakers: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_CLASSIFIER_MODEL,
    classifier_c: float = 1.0,
    classifier_n_neighbors: int = 13,
    max_samples_per_speaker: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    built = build_classifier_dataset_from_bank(
        profile_dir=profile_dir,
        speaker_mapping=speaker_mapping,
        speaker_aliases=speaker_aliases,
        allowed_speakers=allowed_speakers,
        excluded_speakers=excluded_speakers,
        max_samples_per_speaker=max_samples_per_speaker,
    )
    if built is None:
        return None
    dataset, summary = built
    classifier = SegmentClassifier.fit(
        np.asarray(dataset.embeddings, dtype=np.float32),
        dataset.labels,
        model_name=model_name,
        c=classifier_c,
        n_neighbors=classifier_n_neighbors,
    )
    artifacts = classifier.save(profile_dir)
    summary.update(
        {
            "model_name": model_name,
            "artifacts": artifacts,
            "classifier": classifier.summary(),
        }
    )
    return summary


def train_segment_classifier_from_dataset(
    *,
    dataset: ClassifierDataset,
    profile_dir: Path,
    model_name: str = DEFAULT_CLASSIFIER_MODEL,
    classifier_c: float = 1.0,
    classifier_n_neighbors: int = 13,
    base_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    classifier = SegmentClassifier.fit(
        np.asarray(dataset.embeddings, dtype=np.float32),
        dataset.labels,
        model_name=model_name,
        c=classifier_c,
        n_neighbors=classifier_n_neighbors,
    )
    artifacts = classifier.save(profile_dir)
    summary = dict(base_summary or {})
    summary.update(_summarize_classifier_dataset(dataset))
    summary.update(
        {
            "model_name": model_name,
            "artifacts": artifacts,
            "classifier": classifier.summary(),
        }
    )
    return summary
