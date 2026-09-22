from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL = "BUT-FIT/SE-DiCoW"
DEFAULT_REVISION = "470fce9ffff844dd53a27751cdf6c6df9efecb39"
FRAME_HZ = 50
MODEL_SECONDS = 30
MODEL_FRAMES = FRAME_HZ * MODEL_SECONDS


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_path(value: object, *, relative_to: Path) -> Path:
    path = Path(str(value))
    for candidate in (path, Path.cwd() / path, relative_to / path):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _load_cut(path: Path, cut_id: str | None) -> dict:
    for cut in _read_jsonl(path):
        if cut_id is None or str(cut.get("id")) == cut_id:
            return cut
    raise ValueError(f"Cut {cut_id!r} was not found in {path}")


def load_cut_audio(path: Path, cut: Mapping[str, object]) -> tuple[np.ndarray, int]:
    """Load only the region represented by a cut from a possibly shared recording."""
    info = sf.info(path)
    start = max(0, int(round(float(cut.get("start") or 0.0) * info.samplerate)))
    frames = max(1, int(round(float(cut.get("duration") or 0.0) * info.samplerate)))
    mixture, sample_rate = sf.read(
        path,
        start=start,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    return np.asarray(mixture.mean(axis=1), dtype=np.float32), int(sample_rate)


def _session_name(cut: Mapping[str, object]) -> str:
    match = re.match(r"session_(\d+)_", str(cut.get("id") or ""))
    if not match:
        raise ValueError(f"Cannot infer session from cut id {cut.get('id')!r}")
    return f"Session {int(match.group(1))}"


def _active_speakers(cut: Mapping[str, object]) -> list[str]:
    return sorted(
        {
            str(supervision.get("speaker") or "").strip()
            for supervision in list(cut.get("supervisions") or [])
            if str(supervision.get("speaker") or "").strip()
        }
    )


def build_diarization_mask(
    supervisions: Sequence[Mapping[str, object]],
    speakers: Sequence[str],
    *,
    duration: float,
    frame_hz: int = FRAME_HZ,
) -> torch.Tensor:
    frame_count = int(round(duration * frame_hz))
    indices = {speaker: index for index, speaker in enumerate(speakers)}
    mask = torch.zeros(len(speakers), frame_count, dtype=torch.float32)
    for supervision in supervisions:
        speaker = str(supervision.get("speaker") or "")
        if speaker not in indices:
            continue
        start_seconds = max(0.0, float(supervision.get("start") or 0.0))
        end_seconds = min(
            duration,
            start_seconds + max(0.0, float(supervision.get("duration") or 0.0)),
        )
        start = max(0, int(round(start_seconds * frame_hz)))
        end = min(frame_count, int(round(end_seconds * frame_hz)))
        mask[indices[speaker], start:end] = 1.0
    return mask


def stno_mask(diarization: torch.Tensor, target_index: int) -> torch.Tensor:
    non_target = torch.ones(diarization.shape[0], dtype=torch.bool)
    non_target[target_index] = False
    silence = (1.0 - diarization).prod(dim=0)
    no_other = (1.0 - diarization[non_target]).prod(dim=0)
    target_only = diarization[target_index] * no_other
    non_target_only = (1.0 - diarization[target_index]) * (1.0 - no_other)
    overlap = diarization[target_index] - target_only
    return torch.stack((silence, target_only, non_target_only, overlap))


def broaden_stno_masks(masks: torch.Tensor, *, alpha: float) -> torch.Tensor:
    if masks.ndim != 3 or masks.shape[1] != 4:
        raise ValueError(f"Expected [speakers, 4, frames] STNO masks, got {masks.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("STNO broadening alpha must be between zero and one")
    generic = torch.zeros_like(masks)
    generic[:, 0] = masks[:, 0].clamp(0.0, 1.0)
    generic[:, 1] = 1.0 - generic[:, 0]
    broadened = (1.0 - alpha) * masks + alpha * generic
    return broadened / broadened.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _resample_probabilities(probabilities: np.ndarray, output_frames: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected [frames, speakers] probabilities, got {values.shape}")
    if values.shape[0] == output_frames:
        return values
    source = (np.arange(values.shape[0], dtype=np.float64) + 0.5) / values.shape[0]
    target = (np.arange(output_frames, dtype=np.float64) + 0.5) / output_frames
    return np.stack(
        [
            np.interp(
                target,
                source,
                values[:, column],
                left=float(values[0, column]),
                right=float(values[-1, column]),
            )
            for column in range(values.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def sortformer_stno_masks(
    probabilities: np.ndarray,
    slot_to_speaker: Mapping[str, str],
    speakers: Sequence[str],
    *,
    threshold: float,
    output_frames: int = MODEL_FRAMES,
    require_all_speakers: bool = True,
) -> torch.Tensor:
    values = _resample_probabilities(probabilities, output_frames)
    speaker_to_slot = {
        speaker: int(str(slot).rsplit("_", 1)[-1]) for slot, speaker in slot_to_speaker.items()
    }
    missing = sorted(set(speakers) - set(speaker_to_slot))
    if missing and require_all_speakers:
        raise ValueError(f"Sortformer mapping is missing speakers: {', '.join(missing)}")
    masks = []
    for speaker in speakers:
        slot = speaker_to_slot.get(speaker)
        if slot is not None and (slot < 0 or slot >= values.shape[1]):
            raise ValueError(f"Sortformer slot {slot} is outside {values.shape[1]} columns")
        target = (
            values[:, slot] >= threshold
            if slot is not None
            else np.zeros(output_frames, dtype=bool)
        )
        other_slots = [index for index in range(values.shape[1]) if slot is None or index != slot]
        non_target = (
            np.max(values[:, other_slots], axis=1) >= threshold
            if other_slots
            else np.zeros(output_frames, dtype=bool)
        )
        masks.append(
            np.stack(
                (
                    ~(target | non_target),
                    target & ~non_target,
                    non_target & ~target,
                    target & non_target,
                )
            ).astype(np.float32)
        )
    return torch.from_numpy(np.stack(masks))


def sortformer_soft_stno_masks(
    probabilities: np.ndarray,
    slot_to_speaker: Mapping[str, str],
    speakers: Sequence[str],
    *,
    output_frames: int = MODEL_FRAMES,
    require_all_speakers: bool = True,
) -> torch.Tensor:
    values = np.clip(
        _resample_probabilities(probabilities, output_frames),
        0.0,
        1.0,
    )
    speaker_to_slot = {
        speaker: int(str(slot).rsplit("_", 1)[-1]) for slot, speaker in slot_to_speaker.items()
    }
    missing = sorted(set(speakers) - set(speaker_to_slot))
    if missing and require_all_speakers:
        raise ValueError(f"Sortformer mapping is missing speakers: {', '.join(missing)}")
    masks = []
    for speaker in speakers:
        slot = speaker_to_slot.get(speaker)
        target = (
            values[:, slot]
            if slot is not None and 0 <= slot < values.shape[1]
            else np.zeros(output_frames, dtype=np.float32)
        )
        other_slots = [index for index in range(values.shape[1]) if slot is None or index != slot]
        no_other = (
            np.prod(1.0 - values[:, other_slots], axis=1)
            if other_slots
            else np.ones(output_frames, dtype=np.float32)
        )
        masks.append(
            np.stack(
                (
                    (1.0 - target) * no_other,
                    target * no_other,
                    (1.0 - target) * (1.0 - no_other),
                    target * (1.0 - no_other),
                )
            ).astype(np.float32)
        )
    return torch.from_numpy(np.stack(masks))


def enrollment_stno_mask(
    wave: np.ndarray,
    *,
    sample_rate: int,
    frame_hz: int = FRAME_HZ,
    output_frames: int = MODEL_FRAMES,
) -> torch.Tensor:
    frame_samples = int(round(sample_rate / frame_hz))
    padded = np.pad(wave.astype(np.float32), (0, (-len(wave)) % frame_samples))
    frames = padded.reshape(-1, frame_samples)
    energy = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    threshold = max(1e-4, float(np.percentile(energy, 70)) * 0.08)
    active = energy > threshold
    mask = torch.zeros(4, output_frames, dtype=torch.float32)
    mask[0] = 1.0
    valid = min(output_frames, len(active))
    mask[0, :valid] = torch.from_numpy((~active[:valid]).astype(np.float32))
    mask[1, :valid] = torch.from_numpy(active[:valid].astype(np.float32))
    return mask


def _enrollment_paths(
    manifest_path: Path,
    *,
    session: str,
    speakers: Sequence[str],
) -> dict[str, Path]:
    candidates: dict[str, list[tuple[float, Path]]] = collections.defaultdict(list)
    for row in _read_jsonl(manifest_path):
        speaker = str(row.get("speaker_id") or "")
        if str(row.get("session") or "") != session or speaker not in speakers:
            continue
        materialized = dict(row.get("materialized") or {})
        for value in list(materialized.get("positive_enrollment_paths") or []):
            try:
                path = _resolve_path(value, relative_to=manifest_path.resolve().parent)
            except FileNotFoundError:
                continue
            candidates[speaker].append((float(row.get("window_start") or 0.0), path))
    missing = sorted(set(speakers) - set(candidates))
    if missing:
        raise ValueError(f"No materialized enrollment found for: {', '.join(missing)}")
    return {
        speaker: sorted(candidates[speaker], key=lambda item: item[0])[0][1] for speaker in speakers
    }


def _enrollment_roster(manifest_path: Path, *, session: str) -> list[str]:
    speakers = set()
    for row in _read_jsonl(manifest_path):
        if str(row.get("session") or "") != session:
            continue
        materialized = dict(row.get("materialized") or {})
        if materialized.get("positive_enrollment_paths"):
            speaker = str(row.get("speaker_id") or "").strip()
            if speaker:
                speakers.add(speaker)
    if not speakers:
        raise ValueError(f"No enrollment roster found for {session}")
    return sorted(speakers)


def _create_uppercase_mapping(tokenizer) -> None:
    tokenizer.upper_cased_tokens = {}
    vocabulary = tokenizer.get_vocab()
    for token, index in vocabulary.items():
        if not token:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower = token[0] + token[1].lower() + token[2:]
        else:
            lower = token[0].lower() + token[1:]
        if lower != token and lower in vocabulary:
            tokenizer.upper_cased_tokens[vocabulary[lower]] = index


def load_domain_adapter(model: object, adapter_dir: Path) -> object:
    from peft import PeftModel
    from safetensors.torch import load_file

    extra_path = adapter_dir / "domain_extra.safetensors"
    if extra_path.exists():
        incompatible = model.load_state_dict(load_file(extra_path), strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected domain adapter keys: {incompatible.unexpected_keys[:5]}")
    return PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def attributed_bag_metrics(reference: str, prediction: str) -> dict[str, float | int]:
    reference_counts = collections.Counter(normalized_words(reference))
    prediction_counts = collections.Counter(normalized_words(prediction))
    matched = sum((reference_counts & prediction_counts).values())
    reference_words = sum(reference_counts.values())
    predicted_words = sum(prediction_counts.values())
    return {
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "bag_matches": matched,
        "recall": matched / max(1, reference_words),
        "precision": matched / max(1, predicted_words),
    }


def _grouped_lcs_matches(
    reference: Sequence[str],
    prediction: Sequence[str],
    *,
    max_group: int = 3,
) -> tuple[int, int]:
    """Match ordered words while tolerating short tokenizer boundary differences."""
    rows = len(reference)
    columns = len(prediction)
    scores = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    prediction_matches = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            candidates = [
                (int(scores[row + 1, column]), int(prediction_matches[row + 1, column])),
                (int(scores[row, column + 1]), int(prediction_matches[row, column + 1])),
            ]
            for reference_count in range(1, min(max_group, rows - row) + 1):
                reference_text = "".join(reference[row : row + reference_count])
                for prediction_count in range(1, min(max_group, columns - column) + 1):
                    prediction_text = "".join(prediction[column : column + prediction_count])
                    if reference_text != prediction_text:
                        continue
                    candidates.append(
                        (
                            reference_count
                            + int(scores[row + reference_count, column + prediction_count]),
                            prediction_count
                            + int(
                                prediction_matches[row + reference_count, column + prediction_count]
                            ),
                        )
                    )
            scores[row, column], prediction_matches[row, column] = max(candidates)
    return int(scores[0, 0]), int(prediction_matches[0, 0])


def attributed_sequence_metrics(reference: str, prediction: str) -> dict[str, float | int]:
    reference_words = normalized_words(reference)
    predicted_words = normalized_words(prediction)
    matched_reference, matched_prediction = _grouped_lcs_matches(
        reference_words,
        predicted_words,
    )
    return {
        "sequence_matches": matched_reference,
        "sequence_prediction_matches": matched_prediction,
        "sequence_recall": matched_reference / max(1, len(reference_words)),
        "sequence_precision": matched_prediction / max(1, len(predicted_words)),
    }


def load_enrollment_slot_mapping(
    path: Path,
    *,
    cut_id: str,
    mode: str,
) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = next(
        (
            item
            for item in list(payload.get("records") or [])
            if str(item.get("cut_id") or "") == cut_id
        ),
        None,
    )
    if record is None:
        raise ValueError(f"No enrollment binding found for cut {cut_id}")
    mapping = dict(record.get(f"{mode}_mapping") or {})
    if not mapping:
        raise ValueError(f"Enrollment binding for {cut_id} has no {mode} mapping")
    return {str(slot): str(speaker) for slot, speaker in mapping.items()}


def _remove_timestamps(text: str) -> str:
    return re.sub(r"<\|[0-9.]+\|>", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic SE-DiCoW target-ASR evaluation with oracle activity on mono audio."
    )
    parser.add_argument("--cutset", type=Path, required=True)
    parser.add_argument("--cut-id")
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--initial-adapter-dir",
        type=Path,
        action="append",
        default=[],
        help="Adapter to merge before --adapter-dir; repeat in chronological order.",
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument(
        "--conditioning",
        choices=(
            "oracle-supervisions",
            "sortformer-oracle",
            "sortformer-enrollment",
        ),
        default="oracle-supervisions",
    )
    parser.add_argument("--activity-cutset", type=Path)
    parser.add_argument("--slot-mapping-json", type=Path)
    parser.add_argument(
        "--slot-mapping-mode",
        choices=("independent", "one_to_one"),
        default="one_to_one",
    )
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument(
        "--activity-mask-mode",
        choices=("hard", "soft"),
        default="hard",
    )
    parser.add_argument("--target-mask-broadening-alpha", type=float, default=0.0)
    parser.add_argument("--enrollment-gate-override", type=float)
    parser.add_argument("--enrollment-gate-scale", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument(
        "--speaker-set",
        choices=("active-reference", "enrollment-roster"),
        default="active-reference",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.enrollment_gate_override is not None and args.enrollment_gate_scale is not None:
        parser.error("Use only one enrollment gate override")

    started = time.time()
    cut = _load_cut(args.cutset, args.cut_id)
    if float(cut.get("duration") or 0.0) != MODEL_SECONDS:
        raise ValueError("This diagnostic currently requires an exact 30-second cut")
    session = _session_name(cut)
    speakers = (
        _active_speakers(cut)
        if args.speaker_set == "active-reference"
        else _enrollment_roster(args.enrollment_manifest, session=session)
    )
    enrollment_paths = _enrollment_paths(
        args.enrollment_manifest,
        session=session,
        speakers=speakers,
    )
    recording = dict(cut["recording"])
    source = list(recording["sources"])[0]
    audio_path = _resolve_path(
        source["source"],
        relative_to=args.cutset.resolve().parent,
    )
    mixture, sample_rate = load_cut_audio(audio_path, cut)

    from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    for initial_adapter_dir in args.initial_adapter_dir:
        model = load_domain_adapter(model, initial_adapter_dir)
    if args.adapter_dir is not None:
        model = load_domain_adapter(model, args.adapter_dir)
    from train_se_dicow_cutset_adapter import (
        enrollment_cross_gate_values,
        scale_enrollment_cross_gates,
        set_enrollment_cross_gates,
    )

    if args.enrollment_gate_override is not None:
        set_enrollment_cross_gates(model, args.enrollment_gate_override)
    elif args.enrollment_gate_scale is not None:
        scale_enrollment_cross_gates(model, args.enrollment_gate_scale)
    gate_values = enrollment_cross_gate_values(model)
    model = model.eval().to(args.device)
    _create_uppercase_mapping(tokenizer)
    model.set_tokenizer(tokenizer)

    mixture_features = feature_extractor(
        mixture,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if args.conditioning == "oracle-supervisions":
        diarization = build_diarization_mask(
            list(cut.get("supervisions") or []),
            speakers,
            duration=MODEL_SECONDS,
        )
        mixture_stno_masks = torch.stack(
            [stno_mask(diarization, index) for index in range(len(speakers))]
        )
        activity_source = "oracle-supervisions-diagnostic-only"
        uses_reference_activity = True
        slot_mapping = None
    else:
        if args.activity_cutset is None:
            parser.error("--activity-cutset is required for Sortformer conditioning")
        activity_cut = _load_cut(args.activity_cutset, f"{cut['id']}-mask-sortformer")
        custom = dict(activity_cut.get("custom") or {})
        if args.conditioning == "sortformer-oracle":
            slot_mapping = dict(custom.get("sortformer_slot_to_speaker") or {})
            activity_source = "mono-sortformer-oracle-slot-map-diagnostic-only"
            uses_reference_slot_mapping = True
        else:
            if args.slot_mapping_json is None:
                parser.error(
                    "--slot-mapping-json is required for sortformer-enrollment conditioning"
                )
            slot_mapping = load_enrollment_slot_mapping(
                args.slot_mapping_json,
                cut_id=str(cut["id"]),
                mode=args.slot_mapping_mode,
            )
            activity_source = "mono-sortformer-enrollment-binding"
            uses_reference_slot_mapping = False
        probability_path = _resolve_path(
            custom.get("sortformer_probabilities_path"),
            relative_to=args.activity_cutset.resolve().parent,
        )
        probability_values = np.load(probability_path, allow_pickle=False)
        mask_kwargs = {
            "require_all_speakers": (
                args.conditioning == "sortformer-oracle" and args.speaker_set == "active-reference"
            )
        }
        if args.activity_mask_mode == "soft":
            mixture_stno_masks = sortformer_soft_stno_masks(
                probability_values,
                slot_mapping,
                speakers,
                **mask_kwargs,
            )
        else:
            mixture_stno_masks = sortformer_stno_masks(
                probability_values,
                slot_mapping,
                speakers,
                threshold=args.activity_threshold,
                **mask_kwargs,
            )
        uses_reference_activity = False
    if args.target_mask_broadening_alpha > 0.0:
        mixture_stno_masks = broaden_stno_masks(
            mixture_stno_masks,
            alpha=args.target_mask_broadening_alpha,
        )
    enrollment_features = []
    enrollment_masks = []
    for speaker in speakers:
        wave, enrollment_rate = sf.read(enrollment_paths[speaker], dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if int(enrollment_rate) != int(sample_rate):
            raise ValueError(f"Enrollment rate mismatch for {speaker}: {enrollment_rate}")
        enrollment_features.append(
            feature_extractor(
                wave,
                sampling_rate=enrollment_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
        )
        enrollment_masks.append(enrollment_stno_mask(wave, sample_rate=int(enrollment_rate)))

    batch_size = len(speakers)
    model_inputs = {
        "input_features": mixture_features["input_features"]
        .repeat(batch_size, 1, 1)
        .to(args.device, dtype=dtype),
        "attention_mask": mixture_features["attention_mask"].repeat(batch_size, 1).to(args.device),
        "stno_mask": torch.stack([mixture_stno_masks[index] for index in range(batch_size)]).to(
            args.device, dtype=dtype
        ),
        "enrollments": {
            "input_features": torch.cat(
                [features["input_features"] for features in enrollment_features]
            ).to(args.device, dtype=dtype),
            "attention_mask": torch.cat(
                [features["attention_mask"] for features in enrollment_features]
            ).to(args.device),
            "stno_mask": torch.stack(enrollment_masks).to(args.device, dtype=dtype),
        },
    }
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            language="en",
            task="transcribe",
            return_timestamps=True,
            max_new_tokens=args.max_new_tokens,
        )
    sequences = generated["sequences"] if isinstance(generated, dict) else generated
    decoded = tokenizer.batch_decode(
        sequences,
        decode_with_timestamps=True,
        skip_special_tokens=True,
    )

    records = []
    for speaker, raw_prediction in zip(speakers, decoded, strict=True):
        reference = " ".join(
            str(supervision.get("text") or "")
            for supervision in list(cut.get("supervisions") or [])
            if str(supervision.get("speaker") or "") == speaker
            and str(supervision.get("text") or "").strip()
        )
        prediction = _remove_timestamps(raw_prediction)
        records.append(
            {
                "speaker": speaker,
                "reference": reference,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "enrollment_path": str(enrollment_paths[speaker]),
                **attributed_bag_metrics(reference, prediction),
                **attributed_sequence_metrics(reference, prediction),
            }
        )
    totals = {
        key: sum(int(record[key]) for record in records)
        for key in (
            "reference_words",
            "predicted_words",
            "bag_matches",
            "sequence_matches",
            "sequence_prediction_matches",
        )
    }
    result = {
        "model": args.model,
        "revision": args.revision,
        "initial_adapter_dirs": [str(path) for path in args.initial_adapter_dir],
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "cut_id": cut["id"],
        "session": session,
        "mono_audio_path": str(audio_path),
        "mono_audio_offset_seconds": float(cut.get("start") or 0.0),
        "activity_source": activity_source,
        "conditioning": args.conditioning,
        "speaker_set": args.speaker_set,
        "activity_threshold": args.activity_threshold,
        "activity_mask_mode": args.activity_mask_mode,
        "target_mask_broadening_alpha": args.target_mask_broadening_alpha,
        "enrollment_gate_override": args.enrollment_gate_override,
        "enrollment_gate_scale": args.enrollment_gate_scale,
        "enrollment_gate_values": gate_values,
        "uses_reference_activity": uses_reference_activity,
        "uses_reference_slot_mapping": (
            False if args.conditioning == "oracle-supervisions" else uses_reference_slot_mapping
        ),
        "slot_mapping": slot_mapping,
        "inference_uses_isolated_target_audio": False,
        "speakers": speakers,
        "records": records,
        **totals,
        "attributed_bag_recall": totals["bag_matches"] / max(1, totals["reference_words"]),
        "attributed_bag_precision": totals["bag_matches"] / max(1, totals["predicted_words"]),
        "attributed_sequence_recall": totals["sequence_matches"]
        / max(1, totals["reference_words"]),
        "attributed_sequence_precision": totals["sequence_prediction_matches"]
        / max(1, totals["predicted_words"]),
        "gpu_peak_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if args.device.startswith("cuda") else 0.0
        ),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
