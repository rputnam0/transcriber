#!/usr/bin/env python3
"""Evaluate immutable checkpoints on the frozen development selector as they become ready."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--steps", default="100,200,300")
    p.add_argument("--moss-python", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    args = p.parse_args()
    environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    for step in map(int, args.steps.split(",")):
        checkpoint = args.checkpoint_root / f"checkpoint-{step}"
        deadline = time.monotonic() + 7200
        while not (checkpoint / "trainer_state.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Checkpoint {step} did not complete")
            time.sleep(15)
        output = args.run_root / f"selector_{args.label}_{step}"
        output.mkdir(exist_ok=True)
        shutil.copy2(args.run_root / "selector_manifest.json", output / "manifest.json")
        commands = [
            [
                str(args.moss_python),
                "scripts/test_moss_mac.py",
                "infer",
                "--output",
                str(output),
                "--split",
                "dev",
                "--model",
                str(checkpoint),
                "--batch-size",
                "4",
            ],
            [
                sys.executable,
                "scripts/test_moss_mac.py",
                "score",
                "--output",
                str(output),
                "--split",
                "dev",
                "--corpus",
                str(args.corpus),
                "--baseline",
                str(args.corpus),
                "--reference-root",
                str(args.references),
                "--identity",
                str(args.identity),
            ],
        ]
        for name, command in zip(["infer", "score"], commands, strict=True):
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True
                )
        scores = json.loads((output / "dev_scores.json").read_text())["metrics"]
        print(
            json.dumps(
                dict(
                    step=step,
                    results={
                        k: {
                            "named_word_error_rate": v["named_word_error_rate"],
                            "macro_f1": v["macro_f1"],
                            "brief_overlap": v["categories"]["brief_overlap"],
                        }
                        for k, v in scores.items()
                    },
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
