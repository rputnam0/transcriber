from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as functional

from evaluate_se_dicow_oracle_cut import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    _create_uppercase_mapping,
    _enrollment_paths,
    _enrollment_roster,
    _read_jsonl,
    _resolve_path,
    _session_name,
    enrollment_stno_mask,
    load_cut_audio,
    load_domain_adapter,
    sortformer_soft_stno_masks,
)
from train_se_dicow_cutset_adapter import (
    enrollment_cross_gate_values,
    scale_enrollment_cross_gates,
    set_enrollment_cross_gates,
    target_timestamped_text,
)


def per_sequence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Logit/label shape mismatch: {logits.shape} versus {labels.shape}")
    losses = functional.cross_entropy(
        logits.transpose(1, 2).float(),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def rank_enrollments(
    losses: Sequence[float],
    speakers: Sequence[str],
    *,
    truth: str,
) -> dict:
    order = sorted(range(len(losses)), key=lambda index: float(losses[index]))
    truth_index = speakers.index(truth)
    rank = order.index(truth_index) + 1
    best = order[0]
    alternatives = [float(losses[index]) for index in range(len(losses)) if index != truth_index]
    return {
        "predicted_speaker": speakers[best],
        "correct": best == truth_index,
        "truth_rank": rank,
        "truth_nll": float(losses[truth_index]),
        "truth_margin": min(alternatives) - float(losses[truth_index]),
        "nll_by_speaker": {
            speaker: float(loss) for speaker, loss in zip(speakers, losses, strict=True)
        },
    }


def _transcript_spans(cut: Mapping[str, object]) -> list[dict]:
    custom = dict(cut.get("custom") or {})
    spans = list(custom.get("transcript_spans") or [])
    if spans:
        return [dict(span) for span in spans]
    return [
        {
            "speaker": supervision.get("speaker"),
            "start": supervision.get("start"),
            "end": float(supervision.get("start") or 0.0)
            + float(supervision.get("duration") or 0.0),
            "text": supervision.get("text"),
        }
        for supervision in list(cut.get("supervisions") or [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure enrollment identity encoded by target-ASR sequence likelihood."
    )
    parser.add_argument("--cutset", type=Path, required=True)
    parser.add_argument("--activity-cutset", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--initial-adapter-dir", type=Path, action="append", default=[])
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--enrollment-gate-override", type=float)
    parser.add_argument("--enrollment-gate-scale", type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.enrollment_gate_override is not None and args.enrollment_gate_scale is not None:
        parser.error("Use only one enrollment gate override")

    from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
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
    tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    for adapter_dir in args.initial_adapter_dir:
        model = load_domain_adapter(model, adapter_dir)
    if args.adapter_dir is not None:
        model = load_domain_adapter(model, args.adapter_dir)
    if args.enrollment_gate_override is not None:
        set_enrollment_cross_gates(model, args.enrollment_gate_override)
    elif args.enrollment_gate_scale is not None:
        scale_enrollment_cross_gates(model, args.enrollment_gate_scale)
    gate_values = enrollment_cross_gate_values(model)
    model = model.eval().to(args.device)
    model.config.ctc_weight = 0.0
    model.config.use_cache = False
    _create_uppercase_mapping(tokenizer)
    model.set_tokenizer(tokenizer)

    activity_by_id = {str(cut["id"]): cut for cut in _read_jsonl(args.activity_cutset)}
    records = []
    cuts = list(_read_jsonl(args.cutset))
    if args.max_cuts > 0:
        cuts = cuts[: args.max_cuts]
    enrollment_cache = {}
    for cut_index, cut in enumerate(cuts):
        cut_id = str(cut["id"])
        session = _session_name(cut)
        speakers = _enrollment_roster(args.enrollment_manifest, session=session)
        enrollment_paths = _enrollment_paths(
            args.enrollment_manifest,
            session=session,
            speakers=speakers,
        )
        if session not in enrollment_cache:
            features = []
            masks = []
            for speaker in speakers:
                wave, sample_rate = sf.read(enrollment_paths[speaker], dtype="float32")
                if wave.ndim > 1:
                    wave = wave.mean(axis=1)
                features.append(
                    feature_extractor(
                        wave,
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                )
                masks.append(enrollment_stno_mask(wave, sample_rate=int(sample_rate)))
            enrollment_cache[session] = (speakers, features, masks)
        speakers, enrollment_features, enrollment_masks = enrollment_cache[session]

        source = list(dict(cut["recording"])["sources"])[0]
        audio_path = _resolve_path(source["source"], relative_to=args.cutset.resolve().parent)
        mixture, sample_rate = load_cut_audio(audio_path, cut)
        mixture_features = feature_extractor(
            mixture,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        activity_cut = activity_by_id[f"{cut_id}-mask-sortformer"]
        custom = dict(activity_cut.get("custom") or {})
        probability_path = _resolve_path(
            custom.get("sortformer_probabilities_path"),
            relative_to=args.activity_cutset.resolve().parent,
        )
        probabilities = np.load(probability_path, allow_pickle=False)
        oracle_mapping = dict(custom.get("sortformer_slot_to_speaker") or {})
        transcript_spans = _transcript_spans(cut)
        for slot, truth in sorted(oracle_mapping.items()):
            if truth not in speakers:
                continue
            transcript = target_timestamped_text(transcript_spans, truth)
            if not transcript:
                continue
            slot_mask = sortformer_soft_stno_masks(
                probabilities,
                {slot: "target"},
                ["target"],
                require_all_speakers=True,
            )[0]
            labels = tokenizer(
                [transcript] * len(speakers),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=448,
            ).input_ids
            decoder_start = int(model.config.decoder_start_token_id)
            if labels.shape[1] and bool(labels[:, 0].eq(decoder_start).all()):
                labels = labels[:, 1:]
            batch = {
                "input_features": mixture_features["input_features"]
                .repeat(len(speakers), 1, 1)
                .to(args.device, dtype=dtype),
                "attention_mask": mixture_features["attention_mask"]
                .repeat(len(speakers), 1)
                .to(args.device),
                "stno_mask": slot_mask.unsqueeze(0)
                .repeat(len(speakers), 1, 1)
                .to(args.device, dtype=dtype),
                "enrollments": {
                    "input_features": torch.cat(
                        [features["input_features"] for features in enrollment_features]
                    ).to(args.device, dtype=dtype),
                    "attention_mask": torch.cat(
                        [features["attention_mask"] for features in enrollment_features]
                    ).to(args.device),
                    "stno_mask": torch.stack(enrollment_masks).to(args.device, dtype=dtype),
                },
                "labels": labels.to(args.device),
                "upp_labels": labels.to(args.device),
                "use_cache": False,
            }
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=args.device.startswith("cuda"),
                ),
            ):
                logits = model(**batch).logits
            losses = per_sequence_cross_entropy(logits, batch["labels"])
            records.append(
                {
                    "cut_id": cut_id,
                    "session": session,
                    "slot": slot,
                    "truth": truth,
                    "reference_words": len(transcript.split()),
                    **rank_enrollments(
                        losses.detach().cpu().tolist(),
                        speakers,
                        truth=truth,
                    ),
                }
            )
        print(f"processed cut {cut_index + 1}/{len(cuts)}", flush=True)

    correct = sum(record["correct"] for record in records)
    result = {
        "cutset": str(args.cutset),
        "activity_cutset": str(args.activity_cutset),
        "initial_adapter_dirs": [str(path) for path in args.initial_adapter_dir],
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "enrollment_gate_override": args.enrollment_gate_override,
        "enrollment_gate_scale": args.enrollment_gate_scale,
        "enrollment_gate_values": gate_values,
        "slot_count": len(records),
        "top1_correct": correct,
        "top1_accuracy": correct / max(1, len(records)),
        "mean_reciprocal_rank": float(
            np.mean([1.0 / int(record["truth_rank"]) for record in records])
        ),
        "mean_truth_margin": float(np.mean([record["truth_margin"] for record in records])),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
