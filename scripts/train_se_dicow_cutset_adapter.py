from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from evaluate_sortformer_enrollment_binding import trim_clean_enrollment
from run_se_dicow_target_asr import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    SAMPLE_RATE,
    _add_uppercase_mapping,
    _load_audio,
    build_vad_stno_mask,
    select_max_energy_window,
)
from train_se_dicow_domain_adapter import (
    _save_adapter,
    _trainable_parameter_summary,
    promote_trainable_parameters_to_float32,
)


FRAME_RATE = 50
MODEL_SECONDS = 30.0
LORA_TARGET_PROFILES: dict[str, str | list[str]] = {
    "qv": ["q_proj", "v_proj"],
    "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "all-linear": "all-linear",
}


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _base_cut_id(value: object) -> str:
    return re.sub(r"-mask-[^-]+$", "", str(value or ""))


def _session_name(cut_id: object) -> str:
    match = re.match(r"session_(\d+)_", _base_cut_id(cut_id))
    if not match:
        raise ValueError(f"Cannot infer session from cut id {cut_id!r}")
    return f"Session {int(match.group(1))}"


def _spans(cut: Mapping[str, object]) -> list[dict]:
    output = []
    for supervision in list(cut.get("supervisions") or []):
        start = float(supervision.get("start") or 0.0)
        end = start + max(0.0, float(supervision.get("duration") or 0.0))
        speaker = str(supervision.get("speaker") or "").strip()
        if end > start and speaker:
            output.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": str(supervision.get("text") or "").strip(),
                }
            )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


def _transcript_spans(cut: Mapping[str, object], activity_spans: Sequence[dict]) -> list[dict]:
    custom = dict(cut.get("custom") or {})
    raw_spans = list(custom.get("transcript_spans") or [])
    if not raw_spans:
        return list(activity_spans)
    output = []
    for raw in raw_spans:
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end", start) or start)
        speaker = str(raw.get("speaker") or "").strip()
        text = str(raw.get("text") or "").strip()
        if speaker and text and end > start:
            output.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


def target_text(spans: Sequence[Mapping[str, object]], speaker: str) -> str:
    return " ".join(
        str(span.get("text") or "").strip()
        for span in spans
        if str(span.get("speaker") or "") == speaker and str(span.get("text") or "").strip()
    )


def _timestamp_token(seconds: float) -> str:
    frame = min(1500, max(0, int(round(float(seconds) * FRAME_RATE))))
    return f"<|{frame / FRAME_RATE:.2f}|>"


def target_timestamped_text(
    spans: Sequence[Mapping[str, object]],
    speaker: str,
) -> str:
    segments = []
    for span in spans:
        if str(span.get("speaker") or "") != speaker:
            continue
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        start = float(span.get("start") or 0.0)
        end = max(start, float(span.get("end", start) or start))
        segments.append(f"{_timestamp_token(start)}{text}{_timestamp_token(end)}")
    return "".join(segments)


def stno_from_spans(
    spans: Sequence[Mapping[str, object]],
    *,
    target_speaker: str,
    duration: float = MODEL_SECONDS,
    frame_rate: int = FRAME_RATE,
) -> np.ndarray:
    frame_count = int(round(duration * frame_rate))
    target = np.zeros(frame_count, dtype=bool)
    non_target = np.zeros(frame_count, dtype=bool)
    for span in spans:
        start = max(0, int(np.floor(float(span.get("start") or 0.0) * frame_rate)))
        end = min(
            frame_count,
            max(start + 1, int(np.ceil(float(span.get("end") or 0.0) * frame_rate))),
        )
        activity = target if str(span.get("speaker") or "") == target_speaker else non_target
        activity[start:end] = True
    return np.stack(
        (
            ~(target | non_target),
            target & ~non_target,
            non_target & ~target,
            target & non_target,
        )
    ).astype(np.float32)


def soft_stno_from_sortformer(
    probabilities: np.ndarray,
    slot_to_speaker: Mapping[str, str],
    *,
    target_speaker: str,
    duration: float = MODEL_SECONDS,
    frame_rate: int = FRAME_RATE,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected Sortformer [frames, speakers], got {values.shape}")
    output_frames = int(round(duration * frame_rate))
    if values.shape[0] != output_frames:
        source = (np.arange(values.shape[0], dtype=np.float64) + 0.5) / values.shape[0]
        target_axis = (np.arange(output_frames, dtype=np.float64) + 0.5) / output_frames
        values = np.stack(
            [
                np.interp(
                    target_axis,
                    source,
                    values[:, column],
                    left=float(values[0, column]),
                    right=float(values[-1, column]),
                )
                for column in range(values.shape[1])
            ],
            axis=1,
        )
    values = np.clip(values, 0.0, 1.0)
    speaker_to_slot = {
        str(speaker): int(str(slot).rsplit("_", 1)[-1]) for slot, speaker in slot_to_speaker.items()
    }
    target_slot = speaker_to_slot.get(target_speaker)
    target = (
        values[:, target_slot]
        if target_slot is not None and target_slot < values.shape[1]
        else np.zeros(output_frames, dtype=np.float32)
    )
    other_slots = [
        index for index in range(values.shape[1]) if target_slot is None or index != target_slot
    ]
    no_other = (
        np.prod(1.0 - values[:, other_slots], axis=1)
        if other_slots
        else np.ones(output_frames, dtype=np.float32)
    )
    return np.stack(
        (
            (1.0 - target) * no_other,
            target * no_other,
            (1.0 - target) * (1.0 - no_other),
            target * (1.0 - no_other),
        )
    ).astype(np.float32)


def broaden_stno_to_all_speech(stno: np.ndarray, *, alpha: float) -> np.ndarray:
    """Blend a target-specific prior with a generic all-speaker speech prior."""
    values = np.asarray(stno, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError(f"Expected a [4, frames] STNO mask, got {values.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("STNO broadening alpha must be between zero and one")
    generic = np.zeros_like(values)
    generic[0] = np.clip(values[0], 0.0, 1.0)
    generic[1] = 1.0 - generic[0]
    broadened = (1.0 - alpha) * values + alpha * generic
    return broadened / np.maximum(broadened.sum(axis=0, keepdims=True), 1e-8)


def build_cutset_examples(
    cuts: Iterable[Mapping[str, object]],
    *,
    allowed_speakers: set[str],
    activity_sources: set[str],
) -> list[dict]:
    examples = []
    for cut in cuts:
        custom = dict(cut.get("custom") or {})
        activity_source = str(custom.get("activity_mask_source") or "unknown")
        if activity_sources and activity_source not in activity_sources:
            continue
        duration = float(cut.get("duration") or 0.0)
        if not np.isclose(duration, MODEL_SECONDS):
            continue
        spans = _spans(cut)
        transcript_spans = _transcript_spans(cut, spans)
        active_speakers = sorted({str(span["speaker"]) for span in spans} & allowed_speakers)
        for speaker in active_speakers:
            text = target_text(transcript_spans, speaker)
            if not text:
                continue
            examples.append(
                {
                    "cut_id": str(cut.get("id") or ""),
                    "session": _session_name(cut.get("id")),
                    "audio_path": str(list(dict(cut["recording"])["sources"])[0]["source"]),
                    "duration": duration,
                    "spans": spans,
                    "transcript_spans": transcript_spans,
                    "active_speakers": active_speakers,
                    "target_speaker": speaker,
                    "target_text": text,
                    "target_word_count": len(text.split()),
                    "activity_source": activity_source,
                    "sortformer_probabilities_path": custom.get("sortformer_probabilities_path"),
                    "sortformer_slot_to_speaker": dict(
                        custom.get("sortformer_slot_to_speaker") or {}
                    ),
                }
            )
    return examples


def enrollment_index(
    rows: Iterable[Mapping[str, object]],
    *,
    split: str,
) -> dict[str, list[dict]]:
    by_speaker: dict[str, list[dict]] = collections.defaultdict(list)
    seen = set()
    for raw in rows:
        row = dict(raw)
        if str(row.get("split_id") or "") != split:
            continue
        speaker = str(row.get("speaker_id") or "").strip()
        spans = list(row.get("positive_enrollment_spans") or [])
        if not speaker or not row.get("target_member") or not spans:
            continue
        key = (
            str(row.get("session") or ""),
            speaker,
            str(row.get("target_member") or ""),
            tuple(
                (float(span.get("start") or 0.0), float(span.get("duration") or 0.0))
                for span in spans
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        by_speaker[speaker].append(row)
    return {speaker: rows for speaker, rows in sorted(by_speaker.items()) if rows}


def choose_enrollment_row(
    rows: Sequence[Mapping[str, object]],
    *,
    exclude_session: str,
    rng: random.Random,
) -> Mapping[str, object]:
    cross_session = [row for row in rows if str(row.get("session") or "") != exclude_session]
    return rng.choice(cross_session or list(rows))


def sample_training_assignment(
    example: Mapping[str, object],
    *,
    roster: set[str],
    positive_transcript: str,
    absent_probability: float,
    wrong_enrollment_probability: float,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    activity_speaker = str(example["target_speaker"])
    enrollment_speaker = activity_speaker
    transcript = positive_transcript
    negative_type = "positive"
    draw = rng.random()
    absent = sorted(roster - set(example["active_speakers"]))
    if draw < wrong_enrollment_probability:
        wrong_candidates = absent or sorted(roster - {activity_speaker})
        if wrong_candidates:
            enrollment_speaker = rng.choice(wrong_candidates)
            transcript = ""
            negative_type = "wrong-enrollment-active-mask"
    elif draw < wrong_enrollment_probability + absent_probability and absent:
        activity_speaker = rng.choice(absent)
        enrollment_speaker = activity_speaker
        transcript = ""
        negative_type = "absent-speaker-empty-mask"
    return activity_speaker, enrollment_speaker, transcript, negative_type


def choose_wrong_enrollment_speaker(
    example: Mapping[str, object],
    *,
    roster: set[str],
    rng: random.Random,
) -> str | None:
    target = str(example["target_speaker"])
    absent = sorted(roster - set(example["active_speakers"]))
    candidates = absent or sorted(roster - {target})
    return rng.choice(candidates) if candidates else None


def enrollment_ranking_loss(positive_nll: object, wrong_nll: object, *, margin: float):
    """Require the target transcript to be more likely with the correct enrollment."""
    import torch.nn.functional as functional

    return functional.softplus(positive_nll.detach() - wrong_nll + margin)


def enrollment_cross_gate_values(model: object) -> dict[str, float]:
    return {
        name: float(parameter.detach().float().mean().cpu())
        for name, parameter in model.named_parameters()
        if "ca_enrolls" in name and name.endswith("cross_gate.gate")
    }


def set_enrollment_cross_gates(model: object, value: float) -> dict[str, float]:
    import torch

    if not np.isfinite(value):
        raise ValueError("Enrollment gate initialization must be finite")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "ca_enrolls" in name and name.endswith("cross_gate.gate"):
                parameter.fill_(value)
    gates = enrollment_cross_gate_values(model)
    if not gates:
        raise ValueError("Model has no enrollment cross-attention gates")
    return gates


def scale_enrollment_cross_gates(model: object, factor: float) -> dict[str, float]:
    import torch

    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("Enrollment gate scale must be positive and finite")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "ca_enrolls" in name and name.endswith("cross_gate.gate"):
                parameter.mul_(factor)
    gates = enrollment_cross_gate_values(model)
    if not gates:
        raise ValueError("Model has no enrollment cross-attention gates")
    return gates


def _safe_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _stem_path(stem_root: Path, row: Mapping[str, object]) -> Path:
    session_dir = stem_root / _safe_id(row.get("session"))
    name = Path(str(row.get("target_member") or "")).name
    direct = session_dir / name
    if direct.exists():
        return direct
    matches = [path for path in session_dir.rglob(name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Missing enrollment stem {name} under {session_dir}")
    return matches[0]


@lru_cache(maxsize=256)
def _cached_enrollment(
    stem_path: str,
    spans_json: str,
    max_seconds: float,
) -> np.ndarray:
    spans = json.loads(spans_json)
    parts = [
        _load_audio(
            Path(stem_path),
            start=float(span.get("start") or 0.0),
            seconds=float(span.get("duration") or 0.0),
        )
        for span in spans
        if float(span.get("duration") or 0.0) > 0.0
    ]
    if not parts:
        raise ValueError(f"No usable enrollment spans for {stem_path}")
    packed = trim_clean_enrollment(
        np.concatenate(parts),
        sample_rate=SAMPLE_RATE,
    )
    return select_max_energy_window(
        packed,
        frames=int(round(max_seconds * SAMPLE_RATE)),
    )


def load_enrollment(
    row: Mapping[str, object],
    *,
    stem_root: Path,
    max_seconds: float = 30.0,
) -> np.ndarray:
    path = _stem_path(stem_root, row)
    spans_json = json.dumps(
        list(row.get("positive_enrollment_spans") or []),
        sort_keys=True,
    )
    return _cached_enrollment(str(path), spans_json, max_seconds).copy()


def _features(feature_extractor: object, samples: np.ndarray):
    return feature_extractor(
        samples,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
    )


def merge_domain_adapter(model: object, adapter_dir: Path) -> object:
    from peft import PeftModel
    from safetensors.torch import load_file

    extra_path = adapter_dir / "domain_extra.safetensors"
    if extra_path.exists():
        incompatible = model.load_state_dict(load_file(extra_path), strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected initial adapter keys: {incompatible.unexpected_keys[:5]}")
    return PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()


def prepare_example(
    example: Mapping[str, object],
    *,
    activity_speaker: str,
    enrollment_speaker: str,
    target_transcript: str,
    enrollment_row: Mapping[str, object],
    stem_root: Path,
    feature_extractor: object,
    tokenizer: object,
    decoder_start_token_id: int,
    target_mask_broadening_alpha: float,
    device: object,
    dtype: object,
) -> tuple[dict, dict]:
    import torch

    mixture = _load_audio(
        Path(str(example["audio_path"])),
        seconds=float(example["duration"]),
    )
    enrollment = load_enrollment(enrollment_row, stem_root=stem_root)
    mixture_features = _features(feature_extractor, mixture)
    enrollment_features = _features(feature_extractor, enrollment)
    probability_path = example.get("sortformer_probabilities_path")
    slot_mapping = dict(example.get("sortformer_slot_to_speaker") or {})
    if probability_path and slot_mapping:
        stno = soft_stno_from_sortformer(
            np.load(Path(str(probability_path)), allow_pickle=False),
            slot_mapping,
            target_speaker=activity_speaker,
            duration=float(example["duration"]),
        )
        effective_activity_source = "mono-sortformer-soft"
    else:
        stno = stno_from_spans(
            list(example["spans"]),
            target_speaker=activity_speaker,
            duration=float(example["duration"]),
        )
        effective_activity_source = str(example["activity_source"])
    if target_mask_broadening_alpha > 0.0:
        stno = broaden_stno_to_all_speech(
            stno,
            alpha=target_mask_broadening_alpha,
        )
    enrollment_stno = build_vad_stno_mask(enrollment)
    encoded = tokenizer(
        target_transcript,
        return_tensors="pt",
        truncation=True,
        max_length=448,
    )
    labels = encoded.input_ids
    if labels.shape[1] and int(labels[0, 0]) == decoder_start_token_id:
        labels = labels[:, 1:]
    batch = {
        "input_features": mixture_features.input_features.to(device=device, dtype=dtype),
        "attention_mask": mixture_features.attention_mask.to(device=device),
        "stno_mask": torch.from_numpy(stno).unsqueeze(0).to(device=device, dtype=dtype),
        "enrollments": {
            "input_features": enrollment_features.input_features.to(
                device=device,
                dtype=dtype,
            ),
            "stno_mask": torch.from_numpy(enrollment_stno)
            .unsqueeze(0)
            .to(device=device, dtype=dtype),
        },
        "labels": labels.to(device=device),
        "upp_labels": labels.to(device=device),
        "use_cache": False,
    }
    return batch, {
        "cut_id": example["cut_id"],
        "session": example["session"],
        "activity_speaker": activity_speaker,
        "enrollment_speaker": enrollment_speaker,
        "target_words": len(target_transcript.split()),
        "activity_source": effective_activity_source,
        "target_mask_broadening_alpha": target_mask_broadening_alpha,
        "negative": not target_transcript,
        "enrollment_session": enrollment_row.get("session"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt SE-DiCoW on corrected mono cutsets with direct target-speaker ASR."
    )
    parser.add_argument("--train-cuts", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--stem-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--initial-adapter-dir",
        type=Path,
        action="append",
        default=[],
        help="Adapter to merge before training; repeat in chronological order.",
    )
    parser.add_argument(
        "--activity-sources",
        default=(
            "isolated-track-teacher,synthetic-mild,synthetic-strong,"
            "mono-sortformer,mono-sortformer-soft"
        ),
    )
    parser.add_argument("--enrollment-split", default="train")
    parser.add_argument(
        "--label-mode",
        choices=("timestamped", "plain"),
        default="timestamped",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--negative-example-probability", type=float, default=0.25)
    parser.add_argument(
        "--wrong-enrollment-probability",
        type=float,
        default=0.0,
        help="Train active masks with a different speaker enrollment and an empty transcript.",
    )
    parser.add_argument(
        "--paired-enrollment-ranking-probability",
        type=float,
        default=0.0,
        help="For positive examples, rank the transcript above the same mask with a wrong enrollment.",
    )
    parser.add_argument("--paired-enrollment-ranking-weight", type=float, default=0.1)
    parser.add_argument("--paired-enrollment-ranking-margin", type=float, default=0.5)
    parser.add_argument(
        "--target-mask-broadening-probability",
        type=float,
        default=0.0,
        help="Train positive and absent-speaker examples with a generic all-speaker prior.",
    )
    parser.add_argument("--target-mask-broadening-alpha", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-target-profile",
        choices=tuple(LORA_TARGET_PROFILES),
        default="qv",
    )
    parser.add_argument("--train-full-scb", action="store_true")
    parser.add_argument(
        "--enrollment-gate-init",
        type=float,
        help="Initialize enrollment cross-attention gates after loading prior adapters.",
    )
    parser.add_argument(
        "--enrollment-gate-scale",
        type=float,
        help="Scale loaded enrollment gates while preserving their layer-wise pattern.",
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    if args.enrollment_gate_init is not None and args.enrollment_gate_scale is not None:
        parser.error("Use only one of --enrollment-gate-init and --enrollment-gate-scale")

    if not 0.0 <= args.negative_example_probability <= 1.0:
        raise ValueError("--negative-example-probability must be between zero and one")
    if not 0.0 <= args.wrong_enrollment_probability <= 1.0:
        raise ValueError("--wrong-enrollment-probability must be between zero and one")
    if args.negative_example_probability + args.wrong_enrollment_probability > 1.0:
        raise ValueError("negative example probabilities must sum to at most one")
    if not 0.0 <= args.paired_enrollment_ranking_probability <= 1.0:
        raise ValueError("--paired-enrollment-ranking-probability must be between zero and one")
    if args.paired_enrollment_ranking_weight < 0.0:
        raise ValueError("--paired-enrollment-ranking-weight must be non-negative")
    if not 0.0 <= args.target_mask_broadening_probability <= 1.0:
        raise ValueError("--target-mask-broadening-probability must be between zero and one")
    if not 0.0 <= args.target_mask_broadening_alpha <= 1.0:
        raise ValueError("--target-mask-broadening-alpha must be between zero and one")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    activity_sources = {
        value.strip() for value in args.activity_sources.split(",") if value.strip()
    }
    enrollments = enrollment_index(
        _read_jsonl(args.enrollment_manifest),
        split=args.enrollment_split,
    )
    roster = set(enrollments)
    examples = build_cutset_examples(
        _read_jsonl(args.train_cuts),
        allowed_speakers=roster,
        activity_sources=activity_sources,
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No target-speaker examples matched the selected cuts and roster")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.set_prefix_tokens(
        language="en",
        task="transcribe",
        predict_timestamps=args.label_mode == "timestamped",
    )
    _add_uppercase_mapping(tokenizer)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.set_tokenizer(tokenizer)
    base_model.config.ctc_weight = 0.0
    base_model.config.use_cache = False
    for initial_adapter_dir in args.initial_adapter_dir:
        base_model = merge_domain_adapter(base_model, initial_adapter_dir)
        base_model.set_tokenizer(tokenizer)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_PROFILES[args.lora_target_profile],
            bias="none",
        ),
    ).to(device)
    if args.enrollment_gate_init is not None:
        set_enrollment_cross_gates(model, args.enrollment_gate_init)
    elif args.enrollment_gate_scale is not None:
        scale_enrollment_cross_gates(model, args.enrollment_gate_scale)
    for name, parameter in model.named_parameters():
        if (
            "fddt" in name
            or "cross_gate.gate" in name
            or (args.train_full_scb and "ca_enrolls" in name and "lora_" not in name)
        ):
            parameter.requires_grad = True
    promoted = promote_trainable_parameters_to_float32(model)
    model.train()
    parameter_summary = _trainable_parameter_summary(model)
    parameter_summary["promoted_to_float32"] = promoted
    print(json.dumps(parameter_summary, sort_keys=True), flush=True)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    history = []
    micro_step = 0
    negative_examples = 0
    negative_types: collections.Counter[str] = collections.Counter()
    cross_session_enrollments = 0
    paired_ranking_examples = 0
    broadened_examples = 0
    for step in range(1, args.max_steps + 1):
        losses = []
        ranking_losses = []
        step_metadata = []
        for _ in range(args.gradient_accumulation_steps):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            example = examples[order[cursor]]
            cursor += 1
            positive_transcript = (
                target_timestamped_text(
                    list(example["transcript_spans"]),
                    str(example["target_speaker"]),
                )
                if args.label_mode == "timestamped"
                else str(example["target_text"])
            )
            activity_speaker, enrollment_speaker, transcript, negative_type = (
                sample_training_assignment(
                    example,
                    roster=roster,
                    positive_transcript=positive_transcript,
                    absent_probability=args.negative_example_probability,
                    wrong_enrollment_probability=args.wrong_enrollment_probability,
                    rng=rng,
                )
            )
            negative_examples += int(not transcript)
            negative_types[negative_type] += 1
            enrollment_row = choose_enrollment_row(
                enrollments[enrollment_speaker],
                exclude_session=str(example["session"]),
                rng=rng,
            )
            cross_session_enrollments += int(
                str(enrollment_row.get("session") or "") != str(example["session"])
            )
            target_mask_broadening_alpha = 0.0
            if (
                args.target_mask_broadening_probability > 0.0
                and rng.random() < args.target_mask_broadening_probability
            ):
                target_mask_broadening_alpha = args.target_mask_broadening_alpha
                broadened_examples += 1
            batch, metadata = prepare_example(
                example,
                activity_speaker=activity_speaker,
                enrollment_speaker=enrollment_speaker,
                target_transcript=transcript,
                enrollment_row=enrollment_row,
                stem_root=args.stem_root,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                decoder_start_token_id=base_model.config.decoder_start_token_id,
                target_mask_broadening_alpha=target_mask_broadening_alpha,
                device=device,
                dtype=dtype,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ):
                loss = model(**batch).loss
            (loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            if (
                transcript
                and args.paired_enrollment_ranking_weight > 0.0
                and rng.random() < args.paired_enrollment_ranking_probability
            ):
                wrong_speaker = choose_wrong_enrollment_speaker(
                    example,
                    roster=roster,
                    rng=rng,
                )
                if wrong_speaker is not None:
                    wrong_row = choose_enrollment_row(
                        enrollments[wrong_speaker],
                        exclude_session=str(example["session"]),
                        rng=rng,
                    )
                    wrong_batch, _wrong_metadata = prepare_example(
                        example,
                        activity_speaker=activity_speaker,
                        enrollment_speaker=wrong_speaker,
                        target_transcript=transcript,
                        enrollment_row=wrong_row,
                        stem_root=args.stem_root,
                        feature_extractor=feature_extractor,
                        tokenizer=tokenizer,
                        decoder_start_token_id=base_model.config.decoder_start_token_id,
                        target_mask_broadening_alpha=target_mask_broadening_alpha,
                        device=device,
                        dtype=dtype,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=dtype,
                        enabled=device.type == "cuda" and dtype != torch.float32,
                    ):
                        wrong_loss = model(**wrong_batch).loss
                        ranking_loss = enrollment_ranking_loss(
                            loss,
                            wrong_loss,
                            margin=args.paired_enrollment_ranking_margin,
                        )
                    (
                        args.paired_enrollment_ranking_weight
                        * ranking_loss
                        / args.gradient_accumulation_steps
                    ).backward()
                    ranking_losses.append(float(ranking_loss.detach().cpu()))
                    paired_ranking_examples += 1
            step_metadata.append(metadata)
            micro_step += 1
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if step <= args.warmup_steps:
            scale = step / max(1, args.warmup_steps)
        else:
            scale = max(
                0.0,
                (args.max_steps - step) / max(1, args.max_steps - args.warmup_steps),
            )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "negative_examples": sum(int(item["negative"]) for item in step_metadata),
            "paired_ranking_examples": len(ranking_losses),
            "paired_ranking_loss": (float(np.mean(ranking_losses)) if ranking_losses else None),
            "broadened_examples": sum(
                float(item["target_mask_broadening_alpha"]) > 0.0 for item in step_metadata
            ),
            "activity_sources": dict(
                collections.Counter(str(item["activity_source"]) for item in step_metadata)
            ),
        }
        history.append(record)
        if step % args.log_every == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            _save_adapter(
                model,
                args.output_dir / f"step_{step:05d}",
                {
                    "step": step,
                    "history": history,
                    "parameter_summary": parameter_summary,
                    "paired_enrollment_ranking_probability": (
                        args.paired_enrollment_ranking_probability
                    ),
                    "paired_enrollment_ranking_weight": args.paired_enrollment_ranking_weight,
                    "paired_enrollment_ranking_margin": args.paired_enrollment_ranking_margin,
                    "target_mask_broadening_probability": (args.target_mask_broadening_probability),
                    "target_mask_broadening_alpha": args.target_mask_broadening_alpha,
                    "enrollment_gate_values": enrollment_cross_gate_values(model),
                },
            )

    metadata = {
        "model": args.model,
        "model_revision": args.model_revision,
        "initial_adapter_dirs": [str(path) for path in args.initial_adapter_dir],
        "train_cuts": str(args.train_cuts),
        "enrollment_manifest": str(args.enrollment_manifest),
        "stem_root": str(args.stem_root),
        "activity_sources": sorted(activity_sources),
        "label_mode": args.label_mode,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "micro_steps": micro_step,
        "learning_rate": args.learning_rate,
        "negative_example_probability": args.negative_example_probability,
        "wrong_enrollment_probability": args.wrong_enrollment_probability,
        "paired_enrollment_ranking_probability": (args.paired_enrollment_ranking_probability),
        "paired_enrollment_ranking_weight": args.paired_enrollment_ranking_weight,
        "paired_enrollment_ranking_margin": args.paired_enrollment_ranking_margin,
        "paired_ranking_examples": paired_ranking_examples,
        "target_mask_broadening_probability": args.target_mask_broadening_probability,
        "target_mask_broadening_alpha": args.target_mask_broadening_alpha,
        "broadened_examples": broadened_examples,
        "negative_examples": negative_examples,
        "negative_types": dict(negative_types),
        "cross_session_enrollments": cross_session_enrollments,
        "example_count": len(examples),
        "roster": sorted(roster),
        "backbone_dtype": args.dtype,
        "optimizer_parameter_dtype": "float32",
        "train_full_scb": args.train_full_scb,
        "lora_target_profile": args.lora_target_profile,
        "enrollment_gate_init": args.enrollment_gate_init,
        "enrollment_gate_scale": args.enrollment_gate_scale,
        "enrollment_gate_values": enrollment_cross_gate_values(model),
        "parameter_summary": parameter_summary,
        "history": history,
    }
    _save_adapter(model, args.output_dir, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
