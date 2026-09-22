from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from run_se_dicow_target_asr import (
    SAMPLE_RATE,
    _clip_words,
    _group_key,
    _group_rows,
    _lcs_pairs,
    _load_audio,
    _read_jsonl,
    _resolve_path,
    _write_jsonl,
)
from train_se_dicow_domain_adapter import load_example_audio
from train_whisper_speaker_sot import (
    DEFAULT_MODEL,
    SPEAKER_TAGS,
    build_uniform_turn_references,
    fifo_order_words,
)


TAG_TO_SPEAKER = {tag: speaker for speaker, tag in SPEAKER_TAGS.items()}
TAG_RE = re.compile("(" + "|".join(re.escape(tag) for tag in SPEAKER_TAGS.values()) + ")")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
WHISPER_PREFIX_TOKENS = 4


def parse_tagged_words(text: str) -> list[dict]:
    current = ""
    words = []
    for index, part in enumerate(TAG_RE.split(str(text or ""))):
        if index % 2:
            current = TAG_TO_SPEAKER.get(part, "")
            continue
        lexical_part = SPECIAL_TOKEN_RE.sub(" ", part)
        for match in TOKEN_RE.finditer(lexical_part):
            token = re.sub(r"[^a-z0-9']+", "", match.group(0).lower()).strip("'")
            if token:
                words.append({"token": token, "speaker": current})
    return words


def render_tagged_text(text: str) -> str:
    rendered = str(text or "")
    for tag, speaker in TAG_TO_SPEAKER.items():
        rendered = rendered.replace(tag, f"[{speaker}]")
    return SPECIAL_TOKEN_RE.sub("", rendered).strip()


def score_tagged_transcript(
    reference_words: Sequence[Mapping[str, object]],
    predicted_text: str,
) -> dict:
    reference = fifo_order_words(
        [
            {
                **dict(word),
                "token": str(word.get("token") or ""),
            }
            for word in reference_words
            if str(word.get("token") or "")
        ]
    )
    predicted = parse_tagged_words(predicted_text)
    pairs = _lcs_pairs(
        [str(word["token"]) for word in reference],
        [str(word["token"]) for word in predicted],
    )
    lexical = len(pairs)
    correct = sum(
        str(reference[ref]["speaker"]) == str(predicted[pred]["speaker"]) for ref, pred in pairs
    )
    overlap_flags = []
    for left in reference:
        overlap_flags.append(
            any(
                right is not left
                and str(right.get("speaker") or "") != str(left.get("speaker") or "")
                and float(right.get("start") or 0.0) < float(left.get("end") or 0.0)
                and float(right.get("end") or 0.0) > float(left.get("start") or 0.0)
                for right in reference
            )
        )
    overlap_reference = sum(overlap_flags)
    overlap_correct = sum(
        overlap_flags[ref] and str(reference[ref]["speaker"]) == str(predicted[pred]["speaker"])
        for ref, pred in pairs
    )
    reference_count = len(reference)
    predicted_count = len(predicted)
    return {
        "reference_words": reference_count,
        "predicted_words": predicted_count,
        "lexical_matched_words": lexical,
        "correct_speaker_words": correct,
        "overlap_reference_words": overlap_reference,
        "overlap_correct_speaker_words": overlap_correct,
        "lexical_recall": lexical / reference_count if reference_count else 0.0,
        "speaker_attributed_word_accuracy": correct / reference_count if reference_count else 0.0,
        "matched_word_speaker_accuracy": correct / lexical if lexical else 0.0,
        "prediction_precision": correct / predicted_count if predicted_count else 0.0,
        "overlap_speaker_attributed_accuracy": (
            overlap_correct / overlap_reference if overlap_reference else 0.0
        ),
    }


def aggregate_scores(scores: Sequence[Mapping[str, object]]) -> dict:
    keys = (
        "reference_words",
        "predicted_words",
        "lexical_matched_words",
        "correct_speaker_words",
        "overlap_reference_words",
        "overlap_correct_speaker_words",
    )
    totals = Counter({key: sum(int(score[key]) for score in scores) for key in keys})

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        **dict(totals),
        "lexical_recall": ratio(totals["lexical_matched_words"], totals["reference_words"]),
        "speaker_attributed_word_accuracy": ratio(
            totals["correct_speaker_words"], totals["reference_words"]
        ),
        "matched_word_speaker_accuracy": ratio(
            totals["correct_speaker_words"], totals["lexical_matched_words"]
        ),
        "prediction_precision": ratio(totals["correct_speaker_words"], totals["predicted_words"]),
        "overlap_speaker_attributed_accuracy": ratio(
            totals["overlap_correct_speaker_words"], totals["overlap_reference_words"]
        ),
    }


def generation_token_limit(
    requested: int,
    *,
    max_target_positions: int,
    reserved_tokens: int = WHISPER_PREFIX_TOKENS,
) -> int:
    return max(1, min(requested, max_target_positions - reserved_tokens))


def unsuppress_control_tokens(
    suppressed: Sequence[int] | None,
    control_token_ids: Sequence[int],
) -> list[int]:
    control = set(control_token_ids)
    return [int(token_id) for token_id in suppressed or [] if token_id not in control]


class FirstSpeakerTokenLogitsProcessor:
    def __init__(
        self,
        *,
        speaker_token_ids: Sequence[int],
        torch: object,
    ):
        self.speaker_token_ids = sorted(set(speaker_token_ids))
        self.torch = torch
        self.trace: list[dict[str, object]] = []

    def reset(self) -> None:
        self.trace = []

    def __call__(self, input_ids: object, scores: object) -> object:
        sequence = input_ids[0].tolist()
        if any(token_id in self.speaker_token_ids for token_id in sequence):
            return scores
        if len(self.trace) < 12:
            self.trace.append(
                {
                    "sequence": sequence,
                    "allowed": self.speaker_token_ids,
                }
            )
        filtered = self.torch.full_like(scores, float("-inf"))
        filtered[:, self.speaker_token_ids] = scores[:, self.speaker_token_ids]
        return filtered


class WhisperSOTDecoder:
    def __init__(
        self,
        *,
        model_name: str,
        adapter_dir: Path,
        device: str,
        dtype: str,
        force_opening_tag: bool,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForSpeechSeq2Seq,
            AutoTokenizer,
            LogitsProcessorList,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, predict_timestamps=False)
        self.tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
        base = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        self.model.eval().to(self.device)
        self.last_generated_ids: list[int] = []
        tag_token_ids = []
        for tag in SPEAKER_TAGS.values():
            token_ids = self.tokenizer.encode(tag, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Expected one token for speaker tag {tag}, got {token_ids}")
            tag_token_ids.append(token_ids[0])
        self.suppress_tokens = unsuppress_control_tokens(
            self.model.generation_config.suppress_tokens,
            tag_token_ids,
        )
        self.decoder_input_ids = None
        self.logits_processor = None
        if force_opening_tag:
            self.decoder_input_ids = self.torch.tensor(
                [self.tokenizer.prefix_tokens],
                device=self.device,
            )
            processor = FirstSpeakerTokenLogitsProcessor(
                speaker_token_ids=tag_token_ids,
                torch=self.torch,
            )
            self.logits_processor = LogitsProcessorList([processor])

    def decode(self, samples: np.ndarray, *, max_new_tokens: int) -> str:
        features = self.feature_extractor(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
        )
        token_limit = generation_token_limit(
            max_new_tokens,
            max_target_positions=int(self.model.config.max_target_positions),
            reserved_tokens=(
                int(self.decoder_input_ids.shape[1])
                if self.decoder_input_ids is not None
                else WHISPER_PREFIX_TOKENS
            ),
        )
        with self.torch.inference_mode():
            generation_inputs = {
                "input_features": features.input_features.to(self.device, dtype=self.dtype),
                "attention_mask": features.attention_mask.to(self.device),
                "return_timestamps": False,
                "max_new_tokens": token_limit,
                "num_beams": 1,
                "suppress_tokens": self.suppress_tokens,
            }
            if self.decoder_input_ids is not None:
                self.logits_processor[0].reset()
                generation_inputs["decoder_input_ids"] = self.decoder_input_ids
                generation_inputs["logits_processor"] = self.logits_processor
            else:
                generation_inputs.update(language="en", task="transcribe")
            generated = self.model.generate(
                **generation_inputs,
            )
        self.last_generated_ids = generated[0].tolist()
        return self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=False,
            decode_with_timestamps=True,
        )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-speaker serialized Whisper ASR.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-reference-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sessions", default="")
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--chunk-start", type=float)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument(
        "--force-opening-tag",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--debug-token-ids", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    manifest_rows = list(_read_jsonl(manifest))
    references = build_uniform_turn_references(manifest_rows)
    for reference_path in args.forced_reference_jsonl:
        references.update({_group_key(row): row for row in _read_jsonl(reference_path)})
    groups = _group_rows(manifest_rows)
    sessions = {value.strip() for value in args.sessions.split(",") if value.strip()}
    selected = []
    for key, rows in groups.items():
        if str(rows[0].get("split_id") or "") != args.split:
            continue
        if sessions and key[0] not in sessions:
            continue
        if args.window_start is not None and not np.isclose(key[1], args.window_start):
            continue
        selected.append((key, rows))
    if args.max_groups > 0:
        selected = selected[: args.max_groups]
    decoder = WhisperSOTDecoder(
        model_name=args.model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        dtype=args.dtype,
        force_opening_tag=args.force_opening_tag,
    )
    output_rows = []
    scores = []
    for key, rows in selected:
        reference = references[key]
        words = list(reference.get("words") or [])
        duration = float(rows[0].get("duration") or key[2] - key[1])
        starts = (
            [args.chunk_start]
            if args.chunk_start is not None
            else [float(value) for value in np.arange(0.0, duration, 30.0)]
        )
        if args.max_chunks > 0:
            starts = starts[: args.max_chunks]
        mixture_value = dict(rows[0].get("materialized") or {}).get("mixture_path")
        mixture_path = (
            _resolve_path(mixture_value, manifest_dir=manifest.parent) if mixture_value else None
        )
        for start in starts:
            if mixture_path is not None:
                samples = _load_audio(mixture_path, start=start, seconds=30.0)
            else:
                samples, _ = load_example_audio(
                    {
                        "row": rows[0],
                        "chunk_start": start,
                        "chunk_seconds": 30.0,
                    },
                    manifest_dir=manifest.parent,
                )
            text = decoder.decode(samples, max_new_tokens=args.max_new_tokens)
            rendered_text = render_tagged_text(text)
            if args.debug_token_ids:
                print(
                    json.dumps(
                        {
                            "generated_ids": decoder.last_generated_ids[:64],
                            "generated_tokens": decoder.tokenizer.convert_ids_to_tokens(
                                decoder.last_generated_ids[:64]
                            ),
                            "tag_processor_trace": (
                                decoder.logits_processor[0].trace
                                if decoder.logits_processor is not None
                                else []
                            ),
                        }
                    ),
                    flush=True,
                )
            clipped = _clip_words(words, chunk_start=start, chunk_seconds=30.0)
            score = score_tagged_transcript(clipped, text)
            scores.append(score)
            row = {
                "session": key[0],
                "window_start": key[1],
                "window_end": key[2],
                "chunk_start": start,
                "chunk_end": min(start + 30.0, duration),
                "text": rendered_text,
                "serialized_text": text,
                "score": score,
            }
            output_rows.append(row)
            print(
                f"{key[0]} {key[1]:.0f}+{start:.0f}: "
                f"accuracy={score['speaker_attributed_word_accuracy']:.3f} "
                f"lexical={score['lexical_recall']:.3f} text={rendered_text[:180]!r}",
                flush=True,
            )
    summary = {
        "model": args.model,
        "adapter_dir": str(args.adapter_dir),
        "split": args.split,
        "group_count": len(selected),
        "chunk_count": len(output_rows),
        "aggregate": aggregate_scores(scores),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "whisper_sot_predictions.jsonl", output_rows)
    (args.output_dir / "whisper_sot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
