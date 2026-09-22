"""Small free-decoding regression check; fixed speaker IDs, no oracle remapping.

Activity references use approximate source-ASR utterance boundaries, not human DER gold.
This smoke check cannot promote a checkpoint by itself.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from score_moss_manifest import parse_target


def activity_counts(reference, predicted, seconds, speaker_count=6):
    times = (np.arange(int(np.ceil(seconds * 50))) + 0.5) / 50

    def matrix(segments):
        out = np.zeros((len(times), speaker_count), bool)
        for segment in segments:
            speaker = segment["speaker"]
            if speaker in [f"S{i+1:02d}" for i in range(speaker_count)]:
                out[:, int(speaker[1:]) - 1] |= (times >= segment["start"]) & (
                    times < segment["end"]
                )
        return out

    truth, pred = matrix(reference), matrix(predicted)
    known = {f"S{i+1:02d}" for i in range(speaker_count)}
    unknown = {s["speaker"] for s in predicted} - known
    unknown_fp = 0
    for speaker in unknown:
        active = np.zeros(len(times), bool)
        for segment in predicted:
            if segment["speaker"] == speaker:
                active |= (times >= segment["start"]) & (times < segment["end"])
        unknown_fp += int(active.sum())
    return dict(
        tp=int((truth & pred).sum()),
        fp=int((~truth & pred).sum()) + unknown_fp,
        fn=int((truth & ~pred).sum()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=24)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_batch_inference import decode_issues, generate_batch
    from train_moss_diarization import DEFAULT_MODEL, DEFAULT_REVISION

    torch.set_num_threads(4)
    device = torch.device("mps")
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(args.model),
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=True
    )
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()][
        : args.limit
    ]
    names = list(model.config.stable_speaker_names)
    for row in rows:
        for name, speaker_id in row["metadata"]["stable_session_speaker_ids"].items():
            if names[int(speaker_id[1:]) - 1] != name:
                raise ValueError("Checkpoint identity order differs from reference")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        model=str(args.model.resolve()),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        limit=args.limit,
    )
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Evaluation cache provenance changed")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    reports = []
    for offset in range(0, len(rows), 4):
        group = rows[offset : offset + 4]
        records = [
            dict(audio=r["conversation"][1]["content"], sha256=r["metadata"]["audio_sha256"])
            for r in group
        ]
        paths = [args.output / (r["metadata"]["cut_id"] + ".json") for r in group]
        if not all(path.exists() for path in paths):
            raw = generate_batch(
                model,
                processor,
                records,
                prompt_kwargs={"prompt": group[0]["conversation"][0]["content"]},
                device=device,
                max_new_tokens=512,
            )
            for row, text, path in zip(group, raw, paths):
                prediction = parse_target(text)
                reference = parse_target(row["conversation"][-1]["content"])
                counts = activity_counts(
                    reference, prediction, sf.info(row["conversation"][1]["content"]).duration
                )
                result = dict(
                    cut_id=row["metadata"]["cut_id"],
                    case=row["metadata"]["case"],
                    counts=counts,
                    raw_text=text,
                    segments=prediction,
                    issues=decode_issues(text),
                )
                path.write_text(json.dumps(result, indent=2) + "\n")
        reports.extend(json.loads(path.read_text()) for path in paths)
        print("DECODED", offset + len(group), flush=True)
    summary = defaultdict(lambda: dict(tp=0, fp=0, fn=0, clips=0, decode_issues=0))
    for result in reports:
        item = summary[result["case"]]
        item["clips"] += 1
        item["decode_issues"] += bool(result["issues"])
        for key, value in result["counts"].items():
            item[key] += value
    for item in summary.values():
        item["named_activity_micro_f1"] = (
            2 * item["tp"] / max(1, 2 * item["tp"] + item["fp"] + item["fn"])
        )
    result = dict(
        cases=dict(summary),
        promoted=False,
        metric="Fixed-identity activity micro F1 at 50 Hz against approximate source-ASR segment boundaries; no speaker permutation. Six clips per case in the default smoke check, not human DER.",
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
