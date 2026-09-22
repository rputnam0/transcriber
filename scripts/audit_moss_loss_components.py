#!/usr/bin/env python3
"""Measure actual teacher-forced loss by target type on fixed training examples."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
        MossTranscribeDiarizeProcessor,
    )
    from train_moss_diarization import DataCollator, load_samples, DEFAULT_MODEL, DEFAULT_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="0,100,479,480,800,1247")
    args = parser.parse_args()
    torch.set_num_threads(4)
    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
            **({} if Path(args.model).is_dir() else {"revision": DEFAULT_REVISION}),
        )
        .to("mps")
        .eval()
    )
    samples = load_samples(str(args.manifest), forbidden_sessions={"37", "39", "56", "62", "63"})
    collator = DataCollator(processor, 2048, max_activity_speakers=8, text_pad_multiple=128)
    totals = defaultdict(lambda: dict(tokens=0, nll=0.0, weighted_nll=0.0))
    for index in map(int, args.indices.split(",")):
        sample = samples[index]
        batch = collator([sample])
        labels, weights = batch.pop("labels"), batch.pop("loss_weights")
        for key in ["activity_labels", "activity_mask"]:
            batch.pop(key)
        prefix = int(torch.nonzero(labels[0] != -100)[0])
        encoded = processor.tokenizer(
            sample["target"], add_special_tokens=False, return_offsets_mapping=True
        )
        with torch.inference_mode():
            inputs = {k: v.to("mps") for k, v in batch.items()}
            output = model(**inputs, use_cache=False)
            loss = torch.nn.functional.cross_entropy(
                output.logits[0, :-1].float(),
                labels[0, 1:].to("mps"),
                ignore_index=-100,
                reduction="none",
            ).cpu()
        for position, (start, end) in enumerate(encoded["offset_mapping"]):
            hits = [s for s in sample["loss_spans"] if s["start"] < end and s["end"] > start]
            kind = max(hits, key=lambda s: s["weight"])["kind"] if hits else "other"
            value = float(loss[prefix - 1 + position])
            group = totals[kind]
            group["tokens"] += 1
            group["nll"] += value
            group["weighted_nll"] += value * float(weights[0, prefix + position])
        del output, inputs, loss
        torch.mps.empty_cache()
    total = sum(v["weighted_nll"] for v in totals.values())
    for values in totals.values():
        values["mean_nll"] = values["nll"] / max(values["tokens"], 1)
        values["weighted_loss_share"] = values["weighted_nll"] / max(total, 1e-9)
    result = dict(
        model=args.model, indices=args.indices, split="training only", components=dict(totals)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
