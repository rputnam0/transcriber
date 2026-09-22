from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from run_se_dicow_target_asr import SAMPLE_RATE, _group_key, _group_rows, _read_jsonl
from train_se_dicow_domain_adapter import build_training_examples, load_example_audio


DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
SPEAKER_TAGS = {
    "B. Ver": "<|0.00|>",
    "Cletus Cobbington": "<|0.02|>",
    "Cyrus Schwert": "<|0.04|>",
    "David Tanglethorn": "<|0.06|>",
    "Dungeon Master": "<|0.08|>",
    "Kaladen Shash": "<|0.10|>",
    "Leopold Magnus": "<|0.12|>",
}
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def build_uniform_turn_references(
    manifest_rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, float, float], dict]:
    references = {}
    for key, rows in _group_rows(manifest_rows).items():
        words = []
        for span in list(rows[0].get("word_spans") or []):
            speaker = str(span.get("speaker") or "")
            start = float(span.get("start") or 0.0)
            end = max(start, float(span.get("end") or start))
            tokens = [
                match.group(0).lower() for match in WORD_RE.finditer(str(span.get("text") or ""))
            ]
            if not speaker or not tokens:
                continue
            step = (end - start) / len(tokens) if end > start else 0.02
            for index, token in enumerate(tokens):
                token_start = start + index * step
                token_end = end if index == len(tokens) - 1 else start + (index + 1) * step
                words.append(
                    {
                        "speaker": speaker,
                        "start": token_start,
                        "end": max(token_start + 0.02, token_end),
                        "token": token,
                        "normalized": token,
                        "text": token,
                        "source_span_start": start,
                        "source_span_end": end,
                        "alignment": "uniform_turn_fallback",
                    }
                )
        references[key] = {
            "session": key[0],
            "window_start": key[1],
            "window_end": key[2],
            "words": words,
            "reference_source": "uniform_turn_fallback",
        }
    return references


def fifo_order_words(words: Sequence[Mapping[str, object]]) -> list[dict]:
    turns: dict[tuple, list[dict]] = {}
    for index, word in enumerate(words):
        speaker = str(word.get("speaker") or "")
        source_start = word.get("source_span_start")
        source_end = word.get("source_span_end")
        if source_start is None or source_end is None:
            key = (speaker, "word", index)
        else:
            key = (speaker, float(source_start), float(source_end))
        turns.setdefault(key, []).append(dict(word))

    ordered_turns = sorted(
        turns.values(),
        key=lambda turn: (
            min(float(word.get("start") or 0.0) for word in turn),
            min(float(word.get("end") or 0.0) for word in turn),
            str(turn[0].get("speaker") or ""),
        ),
    )
    return [
        word
        for turn in ordered_turns
        for word in sorted(
            turn,
            key=lambda item: (
                float(item.get("start") or 0.0),
                float(item.get("end") or 0.0),
            ),
        )
    ]


def build_serialized_transcript(words: Sequence[Mapping[str, object]]) -> str:
    parts = []
    previous = None
    for word in fifo_order_words(words):
        speaker = str(word.get("speaker") or "")
        token = str(word.get("token") or "").strip()
        tag = SPEAKER_TAGS.get(speaker)
        if not tag or not token:
            continue
        if tag != previous:
            parts.append(tag)
            previous = tag
        parts.append(token)
    return " ".join(parts)


def unique_chunk_examples(examples: Iterable[Mapping[str, object]]) -> list[dict]:
    unique = {}
    for example in examples:
        key = (*example["key"], round(float(example["chunk_start"]), 3))
        unique.setdefault(key, dict(example))
    return [unique[key] for key in sorted(unique)]


def speaker_turn_counts(examples: Sequence[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in examples:
        previous = ""
        for word in fifo_order_words(list(example["clipped_words"])):
            speaker = str(word.get("speaker") or "")
            if speaker in SPEAKER_TAGS and speaker != previous:
                counts[speaker] += 1
                previous = speaker
    return counts


def balanced_speaker_weights(
    counts: Mapping[str, int],
    *,
    minimum: float = 0.5,
    maximum: float = 3.0,
) -> dict[str, float]:
    positive = [int(count) for count in counts.values() if int(count) > 0]
    if not positive:
        return {}
    mean = sum(positive) / len(positive)
    return {
        speaker: min(maximum, max(minimum, math.sqrt(mean / int(count))))
        for speaker, count in counts.items()
        if int(count) > 0
    }


def build_speaker_token_mask(
    token_ids: Sequence[int],
    speaker_token_ids: Sequence[int],
) -> list[bool]:
    speakers = set(speaker_token_ids)
    return [token_id in speakers for token_id in token_ids]


def _features(feature_extractor: object, samples: np.ndarray):
    return feature_extractor(
        samples,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
    )


def prepare_example(
    example: Mapping[str, object],
    *,
    manifest_dir: Path,
    feature_extractor: object,
    tokenizer: object,
    decoder_start_token_id: int,
    device: object,
    dtype: object,
) -> tuple[dict, dict]:
    mixture, _ = load_example_audio(example, manifest_dir=manifest_dir)
    features = _features(feature_extractor, mixture)
    transcript = build_serialized_transcript(list(example["clipped_words"]))
    encoded = tokenizer(
        transcript,
        return_tensors="pt",
        truncation=True,
        max_length=448,
    )
    labels = encoded.input_ids
    speaker_token_ids = [tokenizer.convert_tokens_to_ids(tag) for tag in SPEAKER_TAGS.values()]
    speaker_token_mask = build_speaker_token_mask(labels[0].tolist(), speaker_token_ids)
    if labels.shape[1] and int(labels[0, 0]) == decoder_start_token_id:
        labels = labels[:, 1:]
        speaker_token_mask = speaker_token_mask[1:]
    batch = {
        "input_features": features.input_features.to(device=device, dtype=dtype),
        "attention_mask": features.attention_mask.to(device=device),
        "labels": labels.to(device=device),
        "use_cache": False,
    }
    metadata = {
        "session": example["key"][0],
        "window_start": example["key"][1],
        "chunk_start": example["chunk_start"],
        "word_count": len(example["clipped_words"]),
        "transcript": transcript,
        "label_tokens": int(labels.shape[1]),
        "speaker_tag_tokens": sum(speaker_token_mask),
    }
    return batch, {**metadata, "speaker_token_mask": speaker_token_mask}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Whisper to jointly emit known-speaker tags and words from mono audio."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-reference-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--speaker-tag-loss-weight", type=float, default=8.0)
    parser.add_argument(
        "--balance-speaker-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--init-adapter-dir", type=Path)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
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
    manifest_rows = list(_read_jsonl(manifest))
    references = build_uniform_turn_references(manifest_rows)
    for reference_path in args.forced_reference_jsonl:
        references.update({_group_key(row): row for row in _read_jsonl(reference_path)})
    candidate_examples = build_training_examples(
        manifest_rows,
        references,
        split=args.split,
        max_groups=args.max_groups,
    )
    examples = unique_chunk_examples(candidate_examples)
    if not examples:
        raise ValueError("No serialized-output training examples matched")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, predict_timestamps=False)
    tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if args.init_adapter_dir:
        model = PeftModel.from_pretrained(
            base_model,
            str(args.init_adapter_dir),
            is_trainable=True,
        ).to(device)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
        )
        model = get_peft_model(base_model, lora_config).to(device)
    model.train()
    turn_counts = speaker_turn_counts(examples)
    class_weights = balanced_speaker_weights(turn_counts) if args.balance_speaker_tags else {}
    speaker_token_weights = {
        tokenizer.convert_tokens_to_ids(SPEAKER_TAGS[speaker]): weight
        for speaker, weight in class_weights.items()
    }
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "examples": len(examples),
                "trainable_parameters": trainable,
                "total_parameters": total,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        losses = []
        unweighted_losses = []
        label_tokens = 0
        speaker_tag_tokens = 0
        words = 0
        for _ in range(args.gradient_accumulation_steps):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            example = examples[order[cursor]]
            cursor += 1
            batch, metadata = prepare_example(
                example,
                manifest_dir=manifest.parent,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                decoder_start_token_id=base_model.config.decoder_start_token_id,
                device=device,
                dtype=dtype,
            )
            speaker_token_mask = torch.tensor(
                metadata.pop("speaker_token_mask"),
                device=device,
                dtype=torch.bool,
            ).unsqueeze(0)
            output = model(**batch)
            labels = batch["labels"]
            token_losses = torch.nn.functional.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(labels)
            valid = labels.ne(-100)
            weights = torch.ones_like(token_losses)
            weights = weights.masked_fill(speaker_token_mask, args.speaker_tag_loss_weight)
            for token_id, class_weight in speaker_token_weights.items():
                weights = weights.masked_fill(
                    labels.eq(token_id),
                    args.speaker_tag_loss_weight * class_weight,
                )
            loss = (token_losses * weights * valid).sum() / (weights * valid).sum()
            (loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            unweighted_losses.append(float(output.loss.detach().cpu()))
            label_tokens += int(metadata["label_tokens"])
            speaker_tag_tokens += int(metadata["speaker_tag_tokens"])
            words += int(metadata["word_count"])
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
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
            group["lr"] = args.learning_rate * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": float(np.mean(losses)),
            "unweighted_loss": float(np.mean(unweighted_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "label_tokens": label_tokens,
            "speaker_tag_tokens": speaker_tag_tokens,
            "words": words,
        }
        history.append(record)
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(record), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    metadata = {
        "model": args.model,
        "split": args.split,
        "examples": len(examples),
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "speaker_tag_loss_weight": args.speaker_tag_loss_weight,
        "balance_speaker_tags": args.balance_speaker_tags,
        "speaker_turn_counts": dict(sorted(turn_counts.items())),
        "speaker_class_weights": dict(sorted(class_weights.items())),
        "init_adapter_dir": str(args.init_adapter_dir) if args.init_adapter_dir else None,
        "seed": args.seed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "speaker_tags": SPEAKER_TAGS,
        "forced_reference_jsonl": [str(path) for path in args.forced_reference_jsonl],
        "history": history,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
