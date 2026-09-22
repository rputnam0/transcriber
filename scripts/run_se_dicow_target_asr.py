from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_MODEL = "BUT-FIT/SE-DiCoW"
DEFAULT_MODEL_REVISION = "470fce9ffff844dd53a27751cdf6c6df9efecb39"
SAMPLE_RATE = 16_000
STNO_FRAME_RATE = 50
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
GroupKey = tuple[str, float, float]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _group_key(row: Mapping[str, object]) -> GroupKey:
    return (
        str(row.get("session") or ""),
        round(float(row.get("window_start") or 0.0), 3),
        round(float(row.get("window_end") or 0.0), 3),
    )


def _group_rows(rows: Iterable[Mapping[str, object]]) -> dict[GroupKey, list[dict]]:
    groups: dict[GroupKey, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(dict(row))
    return {
        key: sorted(group, key=lambda item: str(item.get("speaker_id") or ""))
        for key, group in sorted(groups.items())
    }


def _resolve_path(value: object, *, manifest_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path

    search_roots = (Path.cwd(), manifest_dir, *manifest_dir.parents)
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(value or "").lower()).strip("'")


def _predicted_tokens(text: str) -> list[str]:
    return [
        token for match in TOKEN_RE.finditer(text) if (token := _normalize_token(match.group(0)))
    ]


def _clip_words(
    words: Sequence[Mapping[str, object]],
    *,
    chunk_start: float,
    chunk_seconds: float,
) -> list[dict]:
    chunk_end = chunk_start + chunk_seconds
    clipped = []
    for word in words:
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        token = _normalize_token(word.get("normalized") or word.get("text"))
        midpoint = (start + end) / 2.0
        if token and chunk_start <= midpoint < chunk_end:
            item = dict(word)
            item["token"] = token
            clipped.append(item)
    return clipped


def _lcs_pairs(reference: Sequence[str], predicted: Sequence[str]) -> list[tuple[int, int]]:
    rows = len(reference)
    cols = len(predicted)
    dp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            if reference[row] == predicted[col]:
                dp[row, col] = dp[row + 1, col + 1] + 1
            else:
                dp[row, col] = max(dp[row + 1, col], dp[row, col + 1])
    pairs = []
    row = 0
    col = 0
    while row < rows and col < cols:
        if reference[row] == predicted[col]:
            pairs.append((row, col))
            row += 1
            col += 1
        elif dp[row + 1, col] >= dp[row, col + 1]:
            row += 1
        else:
            col += 1
    return pairs


def _grouped_lcs_pairs(
    reference: Sequence[str],
    predicted: Sequence[str],
    *,
    max_group: int = 3,
) -> list[tuple[list[int], list[int]]]:
    rows = len(reference)
    cols = len(predicted)
    dp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    actions: dict[tuple[int, int], tuple[str, int, int]] = {}
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            best = int(dp[row + 1, col])
            action = ("skip_ref", 1, 0)
            if int(dp[row, col + 1]) > best:
                best = int(dp[row, col + 1])
                action = ("skip_pred", 0, 1)
            for ref_count in range(1, min(max_group, rows - row) + 1):
                ref_text = "".join(reference[row : row + ref_count])
                for pred_count in range(1, min(max_group, cols - col) + 1):
                    if ref_text != "".join(predicted[col : col + pred_count]):
                        continue
                    score = ref_count + int(dp[row + ref_count, col + pred_count])
                    if score >= best:
                        best = score
                        action = ("match", ref_count, pred_count)
            dp[row, col] = best
            actions[row, col] = action
    groups = []
    row = 0
    col = 0
    while row < rows and col < cols:
        action, ref_count, pred_count = actions[row, col]
        if action == "match":
            groups.append(
                (
                    list(range(row, row + ref_count)),
                    list(range(col, col + pred_count)),
                )
            )
        row += ref_count
        col += pred_count
    return groups


def _overlap_flags(words: Sequence[Mapping[str, object]], *, target: str) -> list[bool]:
    target_words = [word for word in words if str(word.get("speaker") or "") == target]
    other_words = [word for word in words if str(word.get("speaker") or "") != target]
    return [
        any(
            float(other.get("start") or 0.0) < float(word.get("end") or 0.0)
            and float(other.get("end") or 0.0) > float(word.get("start") or 0.0)
            for other in other_words
        )
        for word in target_words
    ]


def score_candidate(
    *,
    reference_words: Sequence[Mapping[str, object]],
    speaker: str,
    predicted_text: str,
) -> dict:
    target_words = [word for word in reference_words if str(word.get("speaker") or "") == speaker]
    other_words = [word for word in reference_words if str(word.get("speaker") or "") != speaker]
    target_tokens = [str(word.get("token") or "") for word in target_words]
    predicted = _predicted_tokens(predicted_text)
    groups = _grouped_lcs_pairs(target_tokens, predicted)
    matched_reference = [index for references, _ in groups for index in references]
    matched_predicted = {index for _, predictions in groups for index in predictions}
    unmatched_predicted = [
        token for index, token in enumerate(predicted) if index not in matched_predicted
    ]
    wrong_pairs = _lcs_pairs(
        [str(word.get("token") or "") for word in other_words],
        unmatched_predicted,
    )
    overlap_flags = _overlap_flags(reference_words, target=speaker)
    overlap_words = sum(overlap_flags)
    overlap_matches = sum(overlap_flags[reference_index] for reference_index in matched_reference)
    matches = len(matched_reference)
    matched_prediction_units = len(matched_predicted)
    predicted_count = len(predicted)
    target_count = len(target_tokens)
    return {
        "speaker": speaker,
        "reference_words": target_count,
        "predicted_words": predicted_count,
        "matched_target_words": matches,
        "matched_prediction_units": matched_prediction_units,
        "target_recall": matches / target_count if target_count else 1.0,
        "target_precision": (
            matched_prediction_units / predicted_count
            if predicted_count
            else float(target_count == 0)
        ),
        "wrong_speaker_match_proxy": len(wrong_pairs),
        "wrong_speaker_leakage_proxy": (
            len(wrong_pairs) / predicted_count if predicted_count else 0.0
        ),
        "overlap_reference_words": overlap_words,
        "overlap_matched_words": overlap_matches,
        "overlap_recall": overlap_matches / overlap_words if overlap_words else 1.0,
        "no_speech_false_positive_words": predicted_count if target_count == 0 else 0,
    }


def build_stno_mask(
    words: Sequence[Mapping[str, object]],
    *,
    speaker: str,
    chunk_start: float,
    chunk_seconds: float = 30.0,
    frame_rate: int = STNO_FRAME_RATE,
    collar_seconds: float = 0.08,
) -> np.ndarray:
    frames = int(round(chunk_seconds * frame_rate))
    target = np.zeros(frames, dtype=bool)
    non_target = np.zeros(frames, dtype=bool)
    chunk_end = chunk_start + chunk_seconds
    for word in words:
        start = max(chunk_start, float(word.get("start") or 0.0) - collar_seconds)
        end = min(chunk_end, float(word.get("end") or 0.0) + collar_seconds)
        if end <= start:
            continue
        first = max(0, int(math.floor((start - chunk_start) * frame_rate)))
        last = min(frames, int(math.ceil((end - chunk_start) * frame_rate)))
        activity = target if str(word.get("speaker") or "") == speaker else non_target
        activity[first:last] = True
    silence = ~(target | non_target)
    target_only = target & ~non_target
    non_target_only = non_target & ~target
    overlap = target & non_target
    return np.stack((silence, target_only, non_target_only, overlap)).astype(np.float32)


def build_vad_stno_mask(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = 30.0,
    frame_rate: int = STNO_FRAME_RATE,
) -> np.ndarray:
    active = _vad_activity(
        samples,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        frame_rate=frame_rate,
    )
    silence = ~active
    zeros = np.zeros(len(active), dtype=bool)
    return np.stack((silence, active, zeros, zeros)).astype(np.float32)


def _vad_activity(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = 30.0,
    frame_rate: int = STNO_FRAME_RATE,
) -> np.ndarray:
    frames = int(round(chunk_seconds * frame_rate))
    frame_samples = max(1, int(round(sample_rate / frame_rate)))
    expected = frames * frame_samples
    padded = np.pad(
        np.asarray(samples, dtype=np.float32)[:expected], (0, max(0, expected - len(samples)))
    )
    rms = np.sqrt(np.mean(padded.reshape(frames, frame_samples) ** 2, axis=1) + 1e-12)
    positive = rms[rms > 1e-5]
    reference = float(np.quantile(positive, 0.9)) if positive.size else 0.0
    threshold = max(1e-5, reference * 0.08)
    active = rms >= threshold
    if active.any():
        active = np.convolve(active.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0
    return active


def build_stno_from_activity(target: np.ndarray, non_target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=bool)
    non_target = np.asarray(non_target, dtype=bool)
    if target.shape != non_target.shape:
        raise ValueError("Target and non-target activity must have the same shape")
    return np.stack(
        (
            ~(target | non_target),
            target & ~non_target,
            non_target & ~target,
            target & non_target,
        )
    ).astype(np.float32)


def build_stno_from_activity_probabilities(
    probabilities: np.ndarray,
    *,
    mixture_active: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    mixture_active = np.asarray(mixture_active, dtype=bool)
    if probabilities.shape != (len(mixture_active), 2):
        raise ValueError("Activity probabilities must have shape (frames, 2)")
    target = (probabilities[:, 0] >= threshold) & mixture_active
    non_target = (probabilities[:, 1] >= threshold) & mixture_active
    non_target |= mixture_active & ~(target | non_target)
    return build_stno_from_activity(target, non_target)


def score_activity_probabilities(
    probabilities: np.ndarray,
    oracle_stno: np.ndarray,
    *,
    threshold: float = 0.5,
    ambiguity_margin: float = 0.15,
) -> dict:
    labels = np.stack(
        (
            np.logical_or(oracle_stno[1], oracle_stno[3]),
            np.logical_or(oracle_stno[2], oracle_stno[3]),
        ),
        axis=-1,
    )
    predicted = np.asarray(probabilities) >= threshold
    true_positive = np.logical_and(predicted, labels).sum(axis=0)
    predicted_positive = predicted.sum(axis=0)
    actual_positive = labels.sum(axis=0)
    result = {}
    for index, name in enumerate(("target", "non_target")):
        result[f"{name}_frame_precision"] = (
            float(true_positive[index] / predicted_positive[index])
            if predicted_positive[index]
            else 0.0
        )
        result[f"{name}_frame_recall"] = (
            float(true_positive[index] / actual_positive[index]) if actual_positive[index] else 1.0
        )
    result["ambiguous_frame_fraction"] = float(
        (np.abs(np.asarray(probabilities)[:, 0] - threshold) <= ambiguity_margin).mean()
    )
    return result


def aggregate_scores(scores: Sequence[Mapping[str, object]]) -> dict:
    totals = {
        "reference_words": sum(int(score["reference_words"]) for score in scores),
        "predicted_words": sum(int(score["predicted_words"]) for score in scores),
        "matched_target_words": sum(int(score["matched_target_words"]) for score in scores),
        "matched_prediction_units": sum(
            int(score.get("matched_prediction_units", score["matched_target_words"]))
            for score in scores
        ),
        "wrong_speaker_match_proxy": sum(
            int(score["wrong_speaker_match_proxy"]) for score in scores
        ),
        "overlap_reference_words": sum(int(score["overlap_reference_words"]) for score in scores),
        "overlap_matched_words": sum(int(score["overlap_matched_words"]) for score in scores),
        "no_speech_false_positive_words": sum(
            int(score["no_speech_false_positive_words"]) for score in scores
        ),
    }
    reference = totals["reference_words"]
    predicted = totals["predicted_words"]
    overlap = totals["overlap_reference_words"]
    totals.update(
        {
            "speaker_attributed_word_recall": (
                totals["matched_target_words"] / reference if reference else 0.0
            ),
            "target_prediction_precision": (
                totals["matched_prediction_units"] / predicted if predicted else 0.0
            ),
            "wrong_speaker_leakage_proxy": (
                totals["wrong_speaker_match_proxy"] / predicted if predicted else 0.0
            ),
            "overlap_word_recall": (totals["overlap_matched_words"] / overlap if overlap else 0.0),
        }
    )
    return totals


def rescore_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    references: Mapping[GroupKey, Mapping[str, object]],
) -> list[dict]:
    rescored = []
    for source_row in rows:
        row = dict(source_row)
        key = _group_key(row)
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"Missing forced reference for {key}")
        chunk_start = float(row.get("chunk_start") or 0.0)
        chunk_end = float(row.get("chunk_end") or chunk_start + 30.0)
        clipped_words = _clip_words(
            list(reference.get("words") or []),
            chunk_start=chunk_start,
            chunk_seconds=chunk_end - chunk_start,
        )
        row["score"] = score_candidate(
            reference_words=clipped_words,
            speaker=str(row.get("speaker") or ""),
            predicted_text=str(row.get("predicted_text") or ""),
        )
        rescored.append(row)
    return rescored


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_rate)
        return resample_poly(
            samples,
            target_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32)
    except ImportError:
        target_frames = max(1, int(round(len(samples) * target_rate / source_rate)))
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.linspace(0, max(len(samples) - 1, 0), target_frames)
        return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _load_audio(path: Path, *, start: float = 0.0, seconds: float = 30.0) -> np.ndarray:
    import soundfile as sf

    info = sf.info(str(path))
    source_start = max(0, int(round(start * info.samplerate)))
    source_frames = max(1, int(round(seconds * info.samplerate)))
    samples, sample_rate = sf.read(
        str(path),
        start=source_start,
        frames=source_frames,
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1)
    mono = _resample(mono, int(sample_rate), SAMPLE_RATE)
    expected = int(round(seconds * SAMPLE_RATE))
    return np.pad(mono[:expected], (0, max(0, expected - len(mono)))).astype(np.float32)


def _load_full_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return _resample(samples.mean(axis=1), int(sample_rate), SAMPLE_RATE)


def select_max_energy_window(samples: np.ndarray, *, frames: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) <= frames:
        return np.pad(samples, (0, max(0, frames - len(samples)))).astype(np.float32)
    block = max(1, SAMPLE_RATE // 20)
    block_count = int(math.ceil(len(samples) / block))
    padded = np.pad(samples, (0, block_count * block - len(samples)))
    block_energy = np.mean(padded.reshape(block_count, block).astype(np.float64) ** 2, axis=1)
    window_blocks = max(1, int(math.ceil(frames / block)))
    prefix = np.concatenate(([0.0], np.cumsum(block_energy)))
    scores = prefix[window_blocks:] - prefix[:-window_blocks]
    start = int(np.argmax(scores)) * block
    return samples[start : start + frames].astype(np.float32)


def _load_enrollment(
    row: Mapping[str, object],
    *,
    manifest_dir: Path,
    seconds: float = 30.0,
) -> np.ndarray:
    materialized = dict(row.get("materialized") or {})
    paths = list(materialized.get("positive_enrollment_paths") or [])
    if not paths:
        raise ValueError(f"No materialized enrollment for {row.get('row_id')}")
    expected = int(round(seconds * SAMPLE_RATE))
    parts = [_load_full_audio(_resolve_path(value, manifest_dir=manifest_dir)) for value in paths]
    samples = np.concatenate(parts)
    return select_max_energy_window(samples, frames=expected)


def _add_uppercase_mapping(tokenizer: object) -> None:
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if not token:
            continue
        lowered = (
            token[0] + token[1].lower() + token[2:]
            if token.startswith(chr(0x120)) and len(token) > 1
            else token[0].lower() + token[1:]
        )
        lower_index = vocab.get(lowered)
        if lower_index is not None and lowered != token:
            tokenizer.upper_cased_tokens[lower_index] = index


def _decode_timestamped_text(tokenizer: object, sequences: object) -> str:
    raw = tokenizer.batch_decode(sequences, skip_special_tokens=False)[0]
    text = re.sub(r"<\|[^|]+\|>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


class SEDiCoWDecoder:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        device: str,
        dtype: str,
        adapter_dir: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

        self.torch = torch
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            revision=revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        _add_uppercase_mapping(self.tokenizer)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        if adapter_dir is not None:
            from peft import PeftModel
            from safetensors.torch import load_file

            extra_path = adapter_dir / "domain_extra.safetensors"
            if extra_path.exists():
                incompatible = model.load_state_dict(load_file(extra_path), strict=False)
                if incompatible.unexpected_keys:
                    raise ValueError(
                        f"Unexpected domain adapter keys: {incompatible.unexpected_keys[:5]}"
                    )
            model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
        self.model = model.eval().to(device)
        self.model.set_tokenizer(self.tokenizer)
        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.activity_head = None
        if adapter_dir is not None:
            from safetensors.torch import load_file

            activity_path = adapter_dir / "activity_head.safetensors"
            if activity_path.exists():
                state = load_file(activity_path)
                output_size = int(state["bias"].numel())
                self.activity_head = torch.nn.Linear(model.config.d_model, output_size)
                self.activity_head.load_state_dict(state)
                self.activity_head.eval().to(device=self.device, dtype=torch.float32)

    def _features(self, samples: np.ndarray):
        features = self.feature_extractor(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
        )
        return (
            features.input_features.to(device=self.device, dtype=self.dtype),
            features.attention_mask.to(device=self.device),
        )

    def decode(
        self,
        *,
        mixture: np.ndarray,
        stno_mask: np.ndarray,
        enrollment: np.ndarray,
        enrollment_stno: np.ndarray,
        language: str,
        max_new_tokens: int,
    ) -> str:
        mixture_features, attention_mask = self._features(mixture)
        enrollment_features, _ = self._features(enrollment)
        stno = (
            self.torch.from_numpy(stno_mask)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=self.dtype,
            )
        )
        enrollment_mask = (
            self.torch.from_numpy(enrollment_stno)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=self.dtype,
            )
        )
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_features=mixture_features,
                attention_mask=attention_mask,
                stno_mask=stno,
                enrollments={
                    "input_features": enrollment_features,
                    "stno_mask": enrollment_mask,
                },
                language=language,
                task="transcribe",
                return_timestamps=True,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        sequences = generated["sequences"] if isinstance(generated, dict) else generated
        return _decode_timestamped_text(self.tokenizer, sequences)

    def predict_activity(
        self,
        *,
        mixture: np.ndarray,
        stno_mask: np.ndarray,
        enrollment: np.ndarray,
        enrollment_stno: np.ndarray,
    ) -> np.ndarray:
        if self.activity_head is None:
            raise ValueError("Adapter does not contain an activity head")
        mixture_features, attention_mask = self._features(mixture)
        enrollment_features, _ = self._features(enrollment)
        stno = (
            self.torch.from_numpy(stno_mask)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=self.dtype,
            )
        )
        enrollment_mask = (
            self.torch.from_numpy(enrollment_stno)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=self.dtype,
            )
        )
        with self.torch.inference_mode():
            outputs = self.model.get_encoder()(
                input_features=mixture_features,
                attention_mask=attention_mask,
                stno_mask=stno,
                enrollments={
                    "input_features": enrollment_features,
                    "stno_mask": enrollment_mask,
                },
            )
            logits = self.activity_head(outputs.last_hidden_state.float())
        return self.torch.sigmoid(logits)[0].cpu().numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_overlap_count(
    words: Sequence[Mapping[str, object]], start: float, seconds: float
) -> int:
    clipped = _clip_words(words, chunk_start=start, chunk_seconds=seconds)
    return sum(
        _overlap_flags(clipped, target=speaker).count(True)
        for speaker in sorted({str(word.get("speaker") or "") for word in clipped})
    )


def _select_groups(
    groups: Mapping[GroupKey, Sequence[dict]],
    *,
    split: str,
    sessions: set[str],
    window_start: float | None,
    max_groups: int,
) -> list[tuple[GroupKey, list[dict]]]:
    selected = []
    for key, rows in groups.items():
        if str(rows[0].get("split_id") or "") != split:
            continue
        if sessions and key[0] not in sessions:
            continue
        if window_start is not None and not math.isclose(key[1], window_start, abs_tol=1e-3):
            continue
        selected.append((key, list(rows)))
    return selected[:max_groups] if max_groups > 0 else selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run enrollment-conditioned SE-DiCoW target-speaker ASR on mono mixtures."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--forced-reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rescore-predictions", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sessions", default="")
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--chunk-start", type=float)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument(
        "--chunk-selection", choices=("chronological", "overlap"), default="overlap"
    )
    parser.add_argument(
        "--conditioning",
        choices=("oracle-stems", "oracle-reference", "target-blind", "learned-activity"),
        default="oracle-stems",
    )
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not math.isclose(args.chunk_seconds, 30.0, abs_tol=1e-6):
        raise ValueError("SE-DiCoW expects 30-second Whisper windows")
    references = {_group_key(row): row for row in _read_jsonl(args.forced_reference_jsonl)}
    if args.rescore_predictions:
        rows = rescore_prediction_rows(
            list(_read_jsonl(args.rescore_predictions)),
            references,
        )
        summary = {
            "candidate_passes": len(rows),
            "rescore_source": str(args.rescore_predictions),
            "aggregate": aggregate_scores([row["score"] for row in rows]),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(args.output_dir / "se_dicow_predictions.jsonl", rows)
        (args.output_dir / "se_dicow_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --rescore-predictions is used")
    manifest = args.manifest.resolve()
    groups = _group_rows(_read_jsonl(manifest))
    sessions = {value.strip() for value in args.sessions.split(",") if value.strip()}
    selected = _select_groups(
        groups,
        split=args.split,
        sessions=sessions,
        window_start=args.window_start,
        max_groups=args.max_groups,
    )
    if not selected:
        raise ValueError("No manifest groups matched the requested selection")

    decoder = SEDiCoWDecoder(
        model_name=args.model,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        adapter_dir=args.adapter_dir,
    )
    output_rows = []
    all_scores = []
    manifest_dir = manifest.parent
    for key, rows in selected:
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"Missing forced reference for {key}")
        words = list(reference.get("words") or [])
        duration = float(rows[0].get("duration") or key[2] - key[1])
        if args.chunk_start is None:
            starts = [float(value) for value in np.arange(0.0, duration, args.chunk_seconds)]
            if args.chunk_selection == "overlap":
                starts.sort(
                    key=lambda start: (
                        _chunk_overlap_count(words, start, args.chunk_seconds),
                        -start,
                    ),
                    reverse=True,
                )
            starts = starts[: args.max_chunks] if args.max_chunks > 0 else starts
        else:
            starts = [args.chunk_start]

        mixture_value = dict(rows[0].get("materialized") or {}).get("mixture_path")
        if not mixture_value:
            raise ValueError(f"Missing materialized mixture for {key}")
        mixture_path = _resolve_path(mixture_value, manifest_dir=manifest_dir)
        for chunk_start in starts:
            mixture = _load_audio(mixture_path, start=chunk_start, seconds=args.chunk_seconds)
            clipped_words = _clip_words(
                words,
                chunk_start=chunk_start,
                chunk_seconds=args.chunk_seconds,
            )
            blind_stno = build_vad_stno_mask(mixture, chunk_seconds=args.chunk_seconds)
            stem_activity = {}
            if args.conditioning == "oracle-stems":
                for row in rows:
                    materialized = dict(row.get("materialized") or {})
                    source_value = materialized.get("target_source_path")
                    if not source_value:
                        raise ValueError(f"Missing target source for {row.get('row_id')}")
                    source = _load_audio(
                        _resolve_path(source_value, manifest_dir=manifest_dir),
                        start=chunk_start,
                        seconds=args.chunk_seconds,
                    )
                    stem_activity[str(row.get("speaker_id") or "")] = _vad_activity(source)
            for row in rows:
                speaker = str(row.get("speaker_id") or "")
                enrollment = _load_enrollment(row, manifest_dir=manifest_dir)
                enrollment_stno = build_vad_stno_mask(enrollment)
                if args.conditioning == "oracle-stems":
                    target_activity = stem_activity[speaker]
                    other_activity = np.logical_or.reduce(
                        [activity for owner, activity in stem_activity.items() if owner != speaker]
                    )
                    stno = build_stno_from_activity(target_activity, other_activity)
                    activity_score = None
                elif args.conditioning == "oracle-reference":
                    stno = build_stno_mask(
                        words,
                        speaker=speaker,
                        chunk_start=chunk_start,
                        chunk_seconds=args.chunk_seconds,
                    )
                    activity_score = None
                elif args.conditioning == "learned-activity":
                    probabilities = decoder.predict_activity(
                        mixture=mixture,
                        stno_mask=blind_stno,
                        enrollment=enrollment,
                        enrollment_stno=enrollment_stno,
                    )
                    stno = build_stno_from_activity_probabilities(
                        probabilities,
                        mixture_active=blind_stno[1].astype(bool),
                        threshold=args.activity_threshold,
                    )
                    oracle_stno = build_stno_mask(
                        words,
                        speaker=speaker,
                        chunk_start=chunk_start,
                        chunk_seconds=args.chunk_seconds,
                    )
                    activity_score = score_activity_probabilities(
                        probabilities,
                        oracle_stno,
                        threshold=args.activity_threshold,
                    )
                else:
                    stno = blind_stno
                    activity_score = None
                text = decoder.decode(
                    mixture=mixture,
                    stno_mask=stno,
                    enrollment=enrollment,
                    enrollment_stno=enrollment_stno,
                    language=args.language,
                    max_new_tokens=args.max_new_tokens,
                )
                score = score_candidate(
                    reference_words=clipped_words,
                    speaker=speaker,
                    predicted_text=text,
                )
                all_scores.append(score)
                record = {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "chunk_start": chunk_start,
                    "chunk_end": min(chunk_start + args.chunk_seconds, duration),
                    "speaker": speaker,
                    "conditioning": args.conditioning,
                    "activity": activity_score,
                    "predicted_text": text,
                    "score": score,
                }
                output_rows.append(record)
                print(
                    f"{key[0]} {key[1]:.0f}+{chunk_start:.0f} {speaker}: "
                    f"recall={score['target_recall']:.3f} precision={score['target_precision']:.3f} "
                    f"words={score['predicted_words']} text={text[:120]!r}",
                    flush=True,
                )

    summary = {
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "conditioning": args.conditioning,
        "split": args.split,
        "group_count": len(selected),
        "candidate_passes": len(output_rows),
        "aggregate": aggregate_scores(all_scores),
    }
    model_path = getattr(decoder.model, "name_or_path", None)
    if model_path and Path(str(model_path)).is_dir():
        weights = Path(str(model_path)) / "model.safetensors"
        if weights.exists():
            summary["model_sha256"] = _sha256(weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "se_dicow_predictions.jsonl", output_rows)
    (args.output_dir / "se_dicow_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
