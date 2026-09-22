from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from run_se_dicow_target_asr import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    SAMPLE_RATE,
    _add_uppercase_mapping,
    _clip_words,
    _group_key,
    _group_rows,
    _load_audio,
    _load_enrollment,
    _read_jsonl,
    _resolve_path,
    build_stno_mask,
    build_vad_stno_mask,
    select_max_energy_window,
)


PEFT_PREFIX = "base_model.model."


def _timestamp(value: float) -> float:
    return min(30.0, max(0.0, round(float(value) * 50.0) / 50.0))


def build_target_transcript(
    words: Sequence[Mapping[str, object]],
    *,
    speaker: str,
    chunk_start: float,
    merge_gap_seconds: float = 0.5,
) -> str:
    target = sorted(
        (word for word in words if str(word.get("speaker") or "") == speaker),
        key=lambda word: (float(word.get("start") or 0.0), float(word.get("end") or 0.0)),
    )
    segments: list[dict] = []
    for word in target:
        token = str(word.get("token") or "").strip()
        if not token:
            continue
        start = _timestamp(float(word.get("start") or 0.0) - chunk_start)
        end = _timestamp(float(word.get("end") or 0.0) - chunk_start)
        end = max(start + 0.02, end)
        end = min(30.0, end)
        if segments and start - float(segments[-1]["end"]) <= merge_gap_seconds:
            segments[-1]["end"] = max(float(segments[-1]["end"]), end)
            segments[-1]["tokens"].append(token)
        else:
            segments.append({"start": start, "end": end, "tokens": [token]})
    return "".join(
        f"<|{segment['start']:.2f}|>{' '.join(segment['tokens'])}" f"<|{segment['end']:.2f}|>"
        for segment in segments
    )


def build_training_examples(
    manifest_rows: Iterable[Mapping[str, object]],
    references: Mapping[tuple[str, float, float], Mapping[str, object]],
    *,
    split: str,
    chunk_seconds: float = 30.0,
    max_groups: int = 0,
) -> list[dict]:
    groups = [
        (key, rows)
        for key, rows in _group_rows(manifest_rows).items()
        if str(rows[0].get("split_id") or "") == split
    ]
    if max_groups > 0:
        groups = groups[:max_groups]
    examples = []
    for key, rows in groups:
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"Missing forced reference for {key}")
        duration = float(rows[0].get("duration") or key[2] - key[1])
        for chunk_start in np.arange(0.0, duration, chunk_seconds):
            clipped = _clip_words(
                list(reference.get("words") or []),
                chunk_start=float(chunk_start),
                chunk_seconds=chunk_seconds,
            )
            active = Counter(str(word.get("speaker") or "") for word in clipped)
            for row in rows:
                speaker = str(row.get("speaker_id") or "")
                examples.append(
                    {
                        "key": key,
                        "row": dict(row),
                        "all_words": list(reference.get("words") or []),
                        "clipped_words": clipped,
                        "chunk_start": float(chunk_start),
                        "chunk_seconds": chunk_seconds,
                        "target_word_count": active[speaker],
                        "speaker": speaker,
                    }
                )
    return examples


def has_positive_enrollment(example: Mapping[str, object]) -> bool:
    row = dict(example.get("row") or {})
    materialized = dict(row.get("materialized") or {})
    return bool(
        materialized.get("positive_enrollment_paths") or row.get("positive_enrollment_spans")
    )


def strip_peft_prefix(name: str) -> str:
    return name.removeprefix(PEFT_PREFIX).replace(".base_layer.", ".")


def _features(feature_extractor: object, samples: np.ndarray):
    return feature_extractor(
        samples,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
    )


def activity_labels_from_stno(stno: np.ndarray) -> np.ndarray:
    if stno.ndim != 2 or stno.shape[0] != 4:
        raise ValueError("STNO mask must have shape (4, frames)")
    return np.stack(
        (
            np.logical_or(stno[1], stno[3]),
            np.logical_or(stno[2], stno[3]),
        ),
        axis=-1,
    ).astype(np.float32)


def _session_stem_dir(manifest_dir: Path, session: object) -> Path:
    session_id = re.sub(r"[^a-z0-9]+", "_", str(session or "").lower()).strip("_")
    return manifest_dir / "_stems" / session_id


def _stem_path(stem_dir: Path, member: object) -> Path:
    name = Path(str(member)).name
    direct = stem_dir / name
    if direct.exists():
        return direct
    matches = [path for path in stem_dir.rglob(name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Missing extracted stem {name} under {stem_dir}")
    return matches[0]


def load_example_audio(
    example: Mapping[str, object],
    *,
    manifest_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    row = dict(example["row"])
    materialized = dict(row.get("materialized") or {})
    chunk_start = float(example["chunk_start"])
    chunk_seconds = float(example["chunk_seconds"])
    mixture_value = materialized.get("mixture_path")
    enrollment_values = list(materialized.get("positive_enrollment_paths") or [])
    if mixture_value and enrollment_values:
        mixture = _load_audio(
            _resolve_path(mixture_value, manifest_dir=manifest_dir),
            start=chunk_start,
            seconds=chunk_seconds,
        )
        enrollment = _load_enrollment(row, manifest_dir=manifest_dir)
        return mixture, enrollment

    stem_dir = _session_stem_dir(manifest_dir, row.get("session"))
    session_start = float(row.get("window_start") or 0.0) + chunk_start
    source_parts = [
        _load_audio(
            _stem_path(stem_dir, member),
            start=session_start,
            seconds=chunk_seconds,
        )
        for member in list(row.get("mixture_members") or [])
    ]
    if not source_parts:
        raise ValueError(f"No mixture members for {row.get('row_id')}")
    mixture = np.clip(np.sum(source_parts, axis=0), -1.0, 1.0).astype(np.float32)
    target_stem = _stem_path(stem_dir, row.get("target_member"))
    enrollment_parts = [
        _load_audio(
            target_stem,
            start=float(span.get("start") or 0.0),
            seconds=float(span.get("duration") or 0.0),
        )
        for span in list(row.get("positive_enrollment_spans") or [])
        if float(span.get("duration") or 0.0) > 0.0
    ]
    if not enrollment_parts:
        raise ValueError(f"No positive enrollment spans for {row.get('row_id')}")
    enrollment = select_max_energy_window(
        np.concatenate(enrollment_parts),
        frames=30 * SAMPLE_RATE,
    )
    return mixture, enrollment


def prepare_example(
    example: Mapping[str, object],
    *,
    manifest_dir: Path,
    feature_extractor: object,
    tokenizer: object,
    decoder_start_token_id: int,
    device: object,
    dtype: object,
    use_oracle_mask: bool,
) -> tuple[dict, dict]:
    import torch

    chunk_start = float(example["chunk_start"])
    chunk_seconds = float(example["chunk_seconds"])
    mixture, enrollment = load_example_audio(example, manifest_dir=manifest_dir)
    mixture_features = _features(feature_extractor, mixture)
    enrollment_features = _features(feature_extractor, enrollment)
    oracle_stno = build_stno_mask(
        list(example["all_words"]),
        speaker=str(example["speaker"]),
        chunk_start=chunk_start,
        chunk_seconds=chunk_seconds,
    )
    if use_oracle_mask:
        stno = oracle_stno
    else:
        stno = build_vad_stno_mask(mixture, chunk_seconds=chunk_seconds)
    enrollment_stno = build_vad_stno_mask(enrollment)
    transcript = build_target_transcript(
        list(example["clipped_words"]),
        speaker=str(example["speaker"]),
        chunk_start=chunk_start,
    )
    encoded = tokenizer(
        transcript,
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
            "input_features": enrollment_features.input_features.to(device=device, dtype=dtype),
            "stno_mask": torch.from_numpy(enrollment_stno)
            .unsqueeze(0)
            .to(device=device, dtype=dtype),
        },
        "labels": labels.to(device=device),
        "upp_labels": labels.to(device=device),
        "use_cache": False,
        "activity_labels": torch.from_numpy(activity_labels_from_stno(oracle_stno))
        .unsqueeze(0)
        .to(device=device),
    }
    metadata = {
        "session": example["key"][0],
        "window_start": example["key"][1],
        "chunk_start": chunk_start,
        "speaker": example["speaker"],
        "target_word_count": example["target_word_count"],
        "transcript": transcript,
        "mask": "oracle-reference" if use_oracle_mask else "target-blind",
    }
    return batch, metadata


def _save_adapter(
    model: object,
    output_dir: Path,
    metadata: Mapping[str, object],
    *,
    activity_head: object | None = None,
) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    extras = {
        strip_peft_prefix(name): parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    }
    save_file(extras, output_dir / "domain_extra.safetensors")
    if activity_head is not None:
        save_file(
            {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in activity_head.state_dict().items()
            },
            output_dir / "activity_head.safetensors",
        )
    (output_dir / "training_metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trainable_parameter_summary(model: object) -> dict:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for parameter in trainable_parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "fraction": trainable / total,
        "trainable_dtypes": sorted({str(parameter.dtype) for parameter in trainable_parameters}),
    }


def promote_trainable_parameters_to_float32(model: object) -> int:
    """Keep optimizer-owned weights out of low-precision quantization dead zones."""
    promoted = 0
    for parameter in model.parameters():
        if parameter.requires_grad and str(parameter.dtype) != "torch.float32":
            parameter.data = parameter.data.float()
            promoted += parameter.numel()
    return promoted


def multilabel_activity_loss(logits: object, labels: object, *, max_pos_weight: float = 20.0):
    import torch
    import torch.nn.functional as functional

    positive = labels.sum(dim=(0, 1))
    negative = labels.numel() / labels.shape[-1] - positive
    pos_weight = torch.where(
        positive > 0,
        (negative / positive.clamp_min(1.0)).clamp(1.0, max_pos_weight),
        torch.ones_like(positive),
    )
    return functional.binary_cross_entropy_with_logits(
        logits.float(),
        labels.float(),
        pos_weight=pos_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain-adapt official SE-DiCoW with direct target-speaker ASR supervision."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--oracle-mask-probability", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--train-full-scb", action="store_true")
    parser.add_argument("--activity-loss-weight", type=float, default=0.0)
    parser.add_argument("--activity-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    if not 0.0 <= args.oracle_mask_probability <= 1.0:
        raise ValueError("--oracle-mask-probability must be between zero and one")
    if args.activity_loss_weight < 0.0:
        raise ValueError("--activity-loss-weight cannot be negative")

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

    manifest = args.manifest.resolve()
    references = {_group_key(row): row for row in _read_jsonl(args.forced_reference_jsonl)}
    examples = build_training_examples(
        _read_jsonl(manifest),
        references,
        split=args.split,
        max_groups=args.max_groups,
    )
    unfiltered_example_count = len(examples)
    examples = [example for example in examples if has_positive_enrollment(example)]
    skipped_missing_enrollment = unfiltered_example_count - len(examples)
    if not examples:
        raise ValueError("No training examples matched the requested split")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=True)
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
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    for name, parameter in model.named_parameters():
        if (
            "fddt" in name
            or "cross_gate.gate" in name
            or (args.train_full_scb and "ca_enrolls" in name and "lora_" not in name)
        ):
            parameter.requires_grad = True
    promoted_parameters = promote_trainable_parameters_to_float32(model)
    activity_head = None
    if args.activity_loss_weight > 0.0:
        activity_head = torch.nn.Linear(base_model.config.d_model, 2).to(
            device=device,
            dtype=torch.float32,
        )
    model.train()
    if activity_head is not None:
        activity_head.train()
    parameter_summary = _trainable_parameter_summary(model)
    parameter_summary["promoted_to_float32"] = promoted_parameters
    parameter_summary["activity_head_parameters"] = (
        sum(parameter.numel() for parameter in activity_head.parameters())
        if activity_head is not None
        else 0
    )
    print(json.dumps(parameter_summary, sort_keys=True), flush=True)

    optimizer_groups = [
        {
            "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
            "lr": args.learning_rate,
            "base_lr": args.learning_rate,
        }
    ]
    if activity_head is not None:
        optimizer_groups.append(
            {
                "params": list(activity_head.parameters()),
                "lr": args.activity_learning_rate,
                "base_lr": args.activity_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    history = []
    micro_step = 0
    for step in range(1, args.max_steps + 1):
        step_losses = []
        step_asr_losses = []
        step_activity_losses = []
        step_examples = []
        for _ in range(args.gradient_accumulation_steps):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            example = examples[order[cursor]]
            cursor += 1
            use_oracle = rng.random() < args.oracle_mask_probability
            batch, example_metadata = prepare_example(
                example,
                manifest_dir=manifest.parent,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                decoder_start_token_id=base_model.config.decoder_start_token_id,
                device=device,
                dtype=dtype,
                use_oracle_mask=use_oracle,
            )
            activity_labels = batch.pop("activity_labels")
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ):
                outputs = model(**batch)
                asr_loss = outputs.loss
                if activity_head is not None:
                    activity_logits = activity_head(outputs.encoder_last_hidden_state)
                    activity_loss = multilabel_activity_loss(
                        activity_logits,
                        activity_labels,
                    )
                else:
                    activity_loss = asr_loss.new_zeros(())
                combined_loss = asr_loss + args.activity_loss_weight * activity_loss
            loss = combined_loss / args.gradient_accumulation_steps
            loss.backward()
            step_losses.append(float(combined_loss.detach().cpu()))
            step_asr_losses.append(float(asr_loss.detach().cpu()))
            step_activity_losses.append(float(activity_loss.detach().cpu()))
            step_examples.append(example_metadata)
            micro_step += 1
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            1.0,
        )
        if step <= args.warmup_steps:
            scale = step / max(1, args.warmup_steps)
        else:
            scale = max(
                0.0,
                (args.max_steps - step) / max(1, args.max_steps - args.warmup_steps),
            )
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": float(np.mean(step_losses)),
            "asr_loss": float(np.mean(step_asr_losses)),
            "activity_loss": float(np.mean(step_activity_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "activity_learning_rate": (
                optimizer.param_groups[1]["lr"] if activity_head is not None else 0.0
            ),
            "active_examples": sum(int(item["target_word_count"] > 0) for item in step_examples),
            "oracle_examples": sum(
                int(item["mask"] == "oracle-reference") for item in step_examples
            ),
        }
        history.append(record)
        if step % args.log_every == 0:
            print(json.dumps(record, sort_keys=True), flush=True)

    metadata = {
        "model": args.model,
        "model_revision": args.model_revision,
        "split": args.split,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "micro_steps": micro_step,
        "learning_rate": args.learning_rate,
        "activity_loss_weight": args.activity_loss_weight,
        "activity_learning_rate": args.activity_learning_rate,
        "backbone_dtype": args.dtype,
        "optimizer_parameter_dtype": "float32",
        "oracle_mask_probability": args.oracle_mask_probability,
        "train_full_scb": args.train_full_scb,
        "example_count": len(examples),
        "skipped_missing_enrollment_examples": skipped_missing_enrollment,
        "empty_target_examples": sum(
            int(example["target_word_count"] == 0) for example in examples
        ),
        "parameter_summary": parameter_summary,
        "history": history,
    }
    _save_adapter(model, args.output_dir, metadata, activity_head=activity_head)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
