#!/usr/bin/env python3
"""Prepare session-disjoint mono mixtures and speech labels from isolated source tracks.

Source VAD labels are weak supervision, not human gold. Inference never sees the stems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import binary_closing, binary_dilation

SR = 16000
NAMES = [
    "bfschmity",
    "jessev567890",
    "joeeeenathan",
    "kinglizard7958",
    "traceritops",
    "travisaurus6985",
]
SPLITS = {55: "train", 61: "train", 62: "dev", 63: "test"}


def vad_probabilities(wave, model_path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, 64), np.float32)
    result = np.empty((len(wave) + 511) // 512, np.float32)
    for i in range(len(result)):
        chunk = np.asarray(wave[i * 512 : (i + 1) * 512], np.float32)
        chunk = np.pad(chunk, (0, 512 - len(chunk)))[None, :]
        joined = np.concatenate([context, chunk], axis=1)
        output, state = session.run(
            None, {"input": joined, "state": state, "sr": np.array(SR, np.int64)}
        )
        context = chunk[:, -64:]
        result[i] = output[0, 0]
    return result


def speech_mask(probabilities):
    # Bridge <=96ms VAD gaps and retain 64ms around speech for consonants.
    mask = binary_closing(probabilities >= 0.5, structure=np.ones(3))
    return binary_dilation(mask, structure=np.ones(5))


def decode(source, destination):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            str(destination),
        ],
        check=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vad-model", type=Path, required=True)
    p.add_argument("--early-audio", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sample_rate": SR,
        "speaker_names": NAMES,
        "sessions": {},
        "reference_type": "isolated-speaker Silero VAD, not human gold",
        "split_rule": SPLITS,
        "vad_model_sha256": hashlib.sha256(args.vad_model.read_bytes()).hexdigest(),
    }
    for number, split in SPLITS.items():
        folder = args.output / f"session{number}"
        folder.mkdir(exist_ok=True)
        source_path = args.audio_root / f"session{number}.zip"
        metadata = {"split": split, "tracks": {}, "evaluation_windows": []}
        with zipfile.ZipFile(source_path) as archive, tempfile.TemporaryDirectory() as tmp:
            for member in sorted(archive.namelist()):
                if not member.endswith(".ogg"):
                    continue
                name = Path(member).stem.split("-", 1)[1]
                if name not in NAMES:
                    raise ValueError(f"Unexpected speaker: {name}")
                pcm = folder / f"{name}.f32"
                if not pcm.exists():
                    extracted = Path(tmp) / Path(member).name
                    with archive.open(member) as src, extracted.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    decode(extracted, pcm)
                    extracted.unlink()
                wave = np.memmap(pcm, dtype=np.float32, mode="r")
                vad_path = folder / f"{name}.vad.npy"
                if not vad_path.exists():
                    np.save(vad_path, vad_probabilities(wave, args.vad_model))
                probabilities = np.load(vad_path)
                metadata["tracks"][name] = {
                    "pcm": str(pcm.resolve()),
                    "vad": str(vad_path.resolve()),
                    "duration": len(wave) / SR,
                    "speech_seconds": float(speech_mask(probabilities).sum() * 0.032),
                }
                print(
                    "TRACK_READY",
                    number,
                    name,
                    round(metadata["tracks"][name]["speech_seconds"]),
                    flush=True,
                )
        duration = min(t["duration"] for t in metadata["tracks"].values())
        # Freeze coverage across the session before running any candidate model.
        if split != "train":
            metadata["evaluation_windows"] = [
                {"start": float(s), "duration": 60.0}
                for s in np.linspace(120, max(120, duration - 180), 12).round(2)
            ]
        manifest["sessions"][str(number)] = metadata
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    early = args.output / "early1.f32"
    if not early.exists():
        decode(args.early_audio, early)
    wave = np.memmap(early, dtype=np.float32, mode="r")
    probabilities = vad_probabilities(wave, args.vad_model)
    np.save(args.output / "early1.vad.npy", probabilities)
    quiet = ~binary_dilation(probabilities > 0.15, structure=np.ones(15))
    frames = np.asarray(wave[: len(wave) // 512 * 512]).reshape(-1, 512)
    noise_frames = frames[quiet[: len(frames)]]
    if len(noise_frames) < 10:
        raise RuntimeError("Insufficient nonspeech frames to estimate early-session noise")
    spectrum = np.mean(np.abs(np.fft.rfft(noise_frames * np.hanning(512), axis=1)) ** 2, axis=0)
    np.savez(
        args.output / "early_noise.npz",
        power=spectrum,
        rms=np.sqrt(np.mean(noise_frames**2)),
        speech_fraction=float((probabilities > 0.5).mean()),
    )
    manifest["early_noise"] = {
        "file": "Session 1",
        "nonspeech_seconds": len(noise_frames) * 0.032,
        "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(noise_frames**2)), 1e-10))),
        "use": "unlabelled target-domain noise statistics only; no speaker pseudo-labels",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("CORPUS_READY", args.output, flush=True)


if __name__ == "__main__":
    main()
