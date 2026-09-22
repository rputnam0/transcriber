"""Run a bounded, non-promoting continuation with held-out loss by acoustic case."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=20)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh run directory; existing checkpoints are preserved")
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_moss_diarization.py")),
        "--train_jsonl",
        str(args.dataset / "train.jsonl"),
        "--eval_jsonl",
        str(args.dataset / "dev.jsonl"),
        "--model_name_or_path",
        str(args.base),
        "--output_dir",
        str(args.output),
        "--forbidden_sessions",
        "1,2,3,15,36,37,39,40,48,50,56,62,63",
        "--use_activity_conditioning",
        "false",
        "--attn_implementation",
        "eager",
        "--max_steps",
        str(args.max_steps),
        "--learning_rate",
        "2e-6",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "2",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_checkpointing",
        "true",
        "--save_strategy",
        "no",
        "--logging_steps",
        "5",
        "--report_to",
        "none",
        "--dataloader_num_workers",
        "0",
        "--dataloader_pin_memory",
        "false",
        "--text_pad_multiple",
        "128",
        "--max_length",
        "4096",
        "--seed",
        "20260921",
        "--data_seed",
        "20260921",
        "--warmup_steps",
        "2",
        "--promote_trainable_parameters_fp32",
        "true",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(
        command,
        check=True,
        env=dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            PYTORCH_ENABLE_MPS_FALLBACK="1",
            OMP_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
        ),
    )
    before = json.loads((args.output / "baseline_dev_results.json").read_text())
    after = json.loads((args.output / "candidate_dev_results.json").read_text())
    report = dict(
        max_steps=args.max_steps,
        promoted=False,
        interpretation="Teacher-forced speaker-tag loss only. Free decoding, word preservation, and human-review regression checks are required before deployment.",
        loss_by_case={
            k: dict(baseline=v, candidate=after[k], delta=after[k] - v)
            for k, v in before.items()
            if k.endswith("_loss")
        },
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
