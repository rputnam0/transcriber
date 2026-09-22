#!/usr/bin/env python3
"""Fine-tune pretrained segmentation into six named, overlapping activity channels.

Train: Sessions 55/61; development: fixed Session 62 windows. Source VAD is weak
supervision. No test references are accessed by training or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from evaluate_early_named_diarization import activity_metrics
from prepare_early_domain_corpus import NAMES, SR
from train_early_speaker_identity import augment, session_tracks
from transcriber.diarization import DEFAULT_DIARIZATION_MODEL, _load_pipeline


def load_data(manifest, split):
    return {
        s: (m, *session_tracks(m)) for s, m in manifest["sessions"].items() if m["split"] == split
    }


def mix_window(waves, start, duration=10):
    lo, hi = int(start * SR), int((start + duration) * SR)
    return sum(np.array(w[lo:hi]) for w in waves.values())


def targets(masks, times):
    return np.stack(
        [
            (
                masks[n][np.clip((times / 0.032).astype(int), 0, len(masks[n]) - 1)]
                if n in masks
                else np.zeros(len(times), bool)
            )
            for n in NAMES
        ],
        axis=-1,
    )


def predict_window(model, wave, device, rf):
    duration = len(wave) / SR
    times = (np.arange(int(duration * 50)) + 0.5) / 50
    output = np.zeros((len(times), 6), np.float64)
    weights = np.zeros(len(times), np.float64)
    # Five-second stride and triangular blending limit artificial chunk-boundary effects.
    starts = list(np.arange(0, max(duration - 10, 0) + 0.01, 5))
    if not starts or starts[-1] < duration - 10:
        starts.append(max(0, duration - 10))
    model.eval()
    with torch.no_grad():
        for start in starts:
            chunk = wave[int(start * SR) : int((start + 10) * SR)]
            pred = (
                model(torch.from_numpy(chunk.copy())[None, None].to(device))
                .sigmoid()[0]
                .cpu()
                .numpy()
            )
            centers = start + rf.start + rf.duration / 2 + np.arange(len(pred)) * rf.step
            eligible = (times >= start) & (times < start + len(chunk) / SR)
            local = times[eligible] - start
            blend = np.maximum(np.minimum(local, 10 - local), 0.1)
            for i in range(6):
                output[eligible, i] += np.interp(times[eligible], centers, pred[:, i]) * blend
            weights[eligible] += blend
    return (output / np.maximum(weights[:, None], 1e-10)).astype(np.float32)


def evaluate(model, data, device, rf):
    predictions, references = [], []
    for _, (meta, waves, masks) in data.items():
        for window in meta["evaluation_windows"]:
            start, duration = window["start"], window["duration"]
            predictions.append(
                predict_window(model, mix_window(waves, start, duration), device, rf)
            )
            references.append(targets(masks, start + (np.arange(int(duration * 50)) + 0.5) / 50))
    return np.concatenate(predictions), np.concatenate(references)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--init-weights", type=Path, help="Warm-start weights; optimizer and RNG restart"
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260918)
    rng = np.random.default_rng(20260918)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    train, dev = load_data(manifest, "train"), load_data(manifest, "dev")
    noise = np.load(args.corpus / "early_noise.npz")["power"]
    pipeline = _load_pipeline(DEFAULT_DIARIZATION_MODEL, device=args.device, hf_token=get_token())
    model = pipeline._segmentation.model
    rf = model.receptive_field
    model.classifier = torch.nn.Linear(model.classifier.in_features, 6).to(args.device)
    model.activation = torch.nn.Identity()
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location="cpu", weights_only=True))
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "seed": 20260918,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "torch": torch.__version__,
                "device": args.device,
                "manifest_sha256": hashlib.sha256(
                    (args.corpus / "manifest.json").read_bytes()
                ).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "initial_weights_sha256": (
                    hashlib.sha256(args.init_weights.read_bytes()).hexdigest()
                    if args.init_weights
                    else None
                ),
                "base_model": DEFAULT_DIARIZATION_MODEL,
            },
            indent=2,
        )
    )
    # Keep the low-level acoustic filters stable; adapt recurrent context and named head.
    for parameter in model.sincnet.parameters():
        parameter.requires_grad_(False)
    groups = [
        dict(params=list(model.lstm.parameters()) + list(model.linear.parameters()), lr=1e-4),
        dict(params=model.classifier.parameters(), lr=1e-3),
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    eligible = {n: [s for s, (_, _, m) in train.items() if n in m] for n in NAMES}
    active = {
        (s, n): np.flatnonzero(mask) for s, (_, _, m) in train.items() for n, mask in m.items()
    }
    # Balanced sample anchors plus moderately weighted positive frames retain rare speakers.
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.full((6,), 3.0, device=args.device))
    started, best, history = time.monotonic(), -1, []
    for step in range(1, args.steps + 1):
        batch, labels = [], []
        for _ in range(args.batch_size):
            name = rng.choice(NAMES)
            session = rng.choice(eligible[name])
            _, waves, masks = train[session]
            anchor = rng.choice(active[session, name]) * 0.032
            duration = min(len(w) for w in waves.values()) / SR
            start = np.clip(anchor - rng.uniform(1, 9), 0, duration - 10)
            wave = mix_window(waves, start)
            if rng.random() < 0.5:
                wave = augment(wave, noise, rng)
            batch.append(wave)
            labels.append((masks, start))
        model.train()
        x = torch.from_numpy(np.stack(batch))[:, None].to(args.device)
        logits = model(x)
        centers = rf.start + rf.duration / 2 + np.arange(logits.shape[1]) * rf.step
        y = torch.from_numpy(
            np.stack([targets(m, start + centers) for m, start in labels]).astype(np.float32)
        ).to(args.device)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(
                "TRAIN",
                step,
                round(loss.item(), 4),
                round(time.monotonic() - started, 1),
                flush=True,
            )
        if step % 100 == 0 or step == args.steps:
            probability, reference = evaluate(model, dev, args.device, rf)
            alternatives = {
                str(t): activity_metrics(reference, probability >= t)
                for t in [0.3, 0.4, 0.5, 0.6, 0.7]
            }
            threshold = max(alternatives, key=lambda t: alternatives[t]["macro_f1"])
            metrics = alternatives[threshold]
            record = dict(
                step=step,
                loss=float(loss.item()),
                threshold=float(threshold),
                metrics=metrics,
                dev_binary_cross_entropy=float(
                    -np.mean(
                        reference * np.log(np.maximum(probability, 1e-7))
                        + (~reference) * np.log(np.maximum(1 - probability, 1e-7))
                    )
                ),
            )
            history.append(record)
            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                torch.save(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    args.output / "best.pt",
                )
                (args.output / "selection.json").write_text(json.dumps(record, indent=2))
                np.savez_compressed(
                    args.output / "dev_predictions.npz",
                    probability=probability,
                    reference=reference,
                )
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            print("DEV", step, threshold, round(metrics["macro_f1"], 4), flush=True)
    print("TRAINING_COMPLETE", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
