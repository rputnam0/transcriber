"""Resolve the pinned, public ASR and alignment weights into a local model manifest."""

import argparse
from pathlib import Path
from run_mac_asr_quality import save

MODELS = {
    "mlx-community/Qwen3-ASR-1.7B-bf16": "e1f6c266914abc5a46e8756e02580f834a6cf8a7",
    "mlx-community/Qwen3-ForcedAligner-0.6B-8bit": "0e1a68e91d815300c7c9754b2a7639378b23db15",
}


def main():
    from huggingface_hub import snapshot_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    models = {
        name: dict(path=snapshot_download(name, revision=revision), revision=revision)
        for name, revision in MODELS.items()
    }
    save(args.output, models)


if __name__ == "__main__":
    main()
