#!/usr/bin/env python3
"""Cache full-file ASR and diarization so identity experiments never rerun recognition."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from transcriber.diarization import _annotation_to_segments, _load_pipeline, load_audio_mono
from transcriber.parakeet_backend import load_model, transcribe_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-speakers", type=int, default=7)
    parser.add_argument("--refresh-asr", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    info = args.audio.stat()
    identity = {
        "path": str(args.audio.resolve()),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "max_speakers": args.max_speakers,
        "model": "pyannote/speaker-diarization-community-1",
        "device": args.device,
    }
    receipt = args.output / "input.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Cache belongs to different inputs/settings; choose a new directory")
    receipt.write_text(json.dumps(identity, indent=2))
    started = time.monotonic()
    asr_path = args.output / "asr.json"
    if args.refresh_asr or not asr_path.exists():
        model = load_model("parakeet", compute_type="float32")
        segments = transcribe_file(str(args.audio), model)
        asr_path.write_text(json.dumps(segments, indent=2))
        print("ASR_READY", len(segments), time.monotonic() - started, flush=True)
    diar_path = args.output / "diarization.json"
    if not diar_path.exists():
        wave = load_audio_mono(str(args.audio), sample_rate=16000)
        pipeline = _load_pipeline(identity["model"], device=args.device, hf_token=get_token())
        if str(pipeline.device) != args.device:
            raise RuntimeError(f"Requested {args.device}, got {pipeline.device}")

        def hook(step_name, artifact, **kwargs):
            if kwargs.get("completed") == kwargs.get("total"):
                print("DIAR_STAGE", step_name, round(time.monotonic() - started, 1), flush=True)

        result = pipeline(
            {"waveform": torch.from_numpy(np.asarray(wave).copy())[None, :], "sample_rate": 16000},
            max_speakers=args.max_speakers,
            hook=hook,
        )
        payload = {
            "segments": [vars(t) for t in _annotation_to_segments(result)],
            "exclusive_segments": [
                vars(t) for t in _annotation_to_segments(result.exclusive_speaker_diarization)
            ],
            "duration": len(wave) / 16000,
            "device": args.device,
            "wave_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
        }
        diar_path.write_text(json.dumps(payload, indent=2))
    print("FULL_FILE_READY", args.output, round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
