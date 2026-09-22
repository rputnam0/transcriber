#!/usr/bin/env python3
"""Measure masked zero-padding damage and verify the app's batch invariance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from transcriber.diarization import (
    _resolve_embedder,
    extract_embeddings_for_segments,
    load_audio_mono,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--remove-gated-silence", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    wave = load_audio_mono(str(args.audio), sample_rate=16000)
    wave = wave[int(args.start * 16000) :]
    if args.remove_gated_silence:
        frames = wave[: len(wave) // 320 * 320].reshape(-1, 320)
        active = np.sqrt(np.mean(frames**2, axis=1)) > 0.005
        active = np.convolve(active.astype(int), np.ones(5), mode="same") > 0
        wave = frames[active].reshape(-1)
    clip = wave[:48000]
    if len(clip) != 48000:
        raise ValueError("Need at least three seconds of audio after trimming.")
    token = get_token()
    embedder = _resolve_embedder(
        model_name="pyannote/speaker-diarization-community-1", hf_token=token, device="cpu"
    )
    tensor = torch.from_numpy(clip.copy())[None, None, :]
    reference = embedder(tensor, masks=torch.ones((1, 48000)))[0]

    def cosine(left, right):
        return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))

    padded_scores, app_scores = {}, {}
    for seconds in [3, 6, 12, 30]:
        padded = torch.nn.functional.pad(tensor, (0, seconds * 16000 - 48000))
        mask = torch.zeros((1, seconds * 16000))
        mask[:, :48000] = 1
        padded_scores[str(seconds)] = cosine(reference, embedder(padded, masks=mask)[0])
        results, _ = extract_embeddings_for_segments(
            "",
            [(0, 3, "probe"), (0, seconds, "neighbor")],
            token,
            force_device="cpu",
            pre_pad=0,
            post_pad=0,
            audio_waveform=np.tile(clip, 10),
            audio_sample_rate=16000,
        )
        app_scores[str(seconds)] = cosine(reference, results[0].embedding)
    output = {
        "model": "pyannote/speaker-diarization-community-1",
        "clip_seconds": 3,
        "legacy_masked_padding_cosine": padded_scores,
        "fixed_app_batch_cosine": app_scores,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
