#!/usr/bin/env python3
"""Train and compare speaker identities with session-disjoint, source-VAD supervision.

Only train sessions provide speaker examples. Development chooses the model by balanced
cross entropy; the test session is evaluated once afterwards. Source tracks NEVER enter
the diarizer. These automatically derived references are not human-labelled gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token
from scipy.signal import fftconvolve
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss

from prepare_early_domain_corpus import NAMES, SR, speech_mask
from transcriber.diarization import DEFAULT_DIARIZATION_MODEL, _resolve_embedder


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def embed(waves, embedder):
    """Never pad unequal audio lengths: padding changes WeSpeaker normalization."""
    out = np.empty((len(waves), embedder.dimension), np.float32)
    for length in sorted({len(w) for w in waves}):
        indices = [i for i, w in enumerate(waves) if len(w) == length]
        for first in range(0, len(indices), 24):
            selected = indices[first : first + 24]
            tensor = torch.from_numpy(np.stack([waves[i] for i in selected]))[:, None]
            vectors = embedder(tensor)
            if not np.isfinite(vectors).all():
                raise ValueError("Nonfinite speaker embedding")
            out[selected] = unit(vectors)
    return out


def augment(wave, noise_power, rng):
    """Target-noise spectrum, gentle channel EQ, and short reverberation; no pitch shift."""
    wave = wave.copy()
    n = len(wave)
    # Smooth channel changes, bounded to +/-6 dB.
    response = np.interp(np.linspace(0, 1, n // 2 + 1), np.linspace(0, 1, 7), rng.uniform(-6, 6, 7))
    wave = np.fft.irfft(np.fft.rfft(wave) * 10 ** (response / 20), n=n)
    if rng.random() < 0.6:
        size = int(rng.uniform(0.06, 0.3) * SR)
        impulse = rng.normal(size=size) * np.exp(-np.arange(size) / (size / 6)) * 0.02
        impulse[0] = 1
        wave = fftconvolve(wave, impulse)[:n]
    noise = rng.normal(size=n)
    shape = np.interp(
        np.linspace(0, 1, n // 2 + 1), np.linspace(0, 1, len(noise_power)), np.sqrt(noise_power)
    )
    noise = np.fft.irfft(np.fft.rfft(noise) * shape, n=n)
    rms = np.sqrt(np.mean(wave**2))
    noise *= rms / max(np.sqrt(np.mean(noise**2)), 1e-10) / 10 ** (rng.uniform(8, 25) / 20)
    wave += noise
    return (wave / max(np.max(np.abs(wave)), 1)).astype(np.float32)


def session_tracks(meta):
    waves, masks = {}, {}
    for name, item in meta["tracks"].items():
        waves[name] = np.memmap(item["pcm"], np.float32, mode="r")
        masks[name] = speech_mask(np.load(item["vad"]))
    return waves, masks


def training_features(manifest, output, embedder, noise_power):
    path = output / "train_features.npz"
    if path.exists():
        return dict(np.load(path))
    rng = np.random.default_rng(20260918)
    clean, altered, labels, provenance = [], [], [], []
    for session, meta in manifest["sessions"].items():
        if meta["split"] != "train":
            continue
        waves, masks = session_tracks(meta)
        for name, wave in waves.items():
            mask = masks[name]
            # Non-overlapping source windows, >=75% speech, spread across the whole session.
            candidates = []
            for start in np.arange(10, len(wave) / SR - 3, 3):
                if mask[int(start / 0.032) : int((start + 3) / 0.032)].mean() >= 0.75:
                    candidates.append(start)
            selected = rng.choice(candidates, min(120, len(candidates)), replace=False)
            if len(selected) < 10:
                raise ValueError(f"Too little training speech: {session}/{name}")
            clips = [np.array(wave[int(s * SR) : int(s * SR) + 3 * SR]) for s in selected]
            clean.extend(embed(clips, embedder))
            altered.extend(embed([augment(w, noise_power, rng) for w in clips], embedder))
            labels.extend([NAMES.index(name)] * len(clips))
            provenance.extend([(int(session), float(s)) for s in selected])
            print("TRAIN_FEATURES", session, name, len(clips), flush=True)
    result = dict(
        clean=np.array(clean),
        augmented=np.array(altered),
        labels=np.array(labels),
        provenance=np.array(provenance),
    )
    np.savez(path, **result)
    return result


def evaluation_features(manifest, split, output, embedder):
    """Evaluate raw MONO sums, including naturally co-occurring source noise.

    Strong single-owner examples need >=75% target activity and <=15% competing activity.
    One-second and three-second crops are separately retained. Hard overlap is evaluated
    with diarization activity separately, not converted into misleading one-owner labels.
    """
    path = output / f"{split}_features.npz"
    if path.exists():
        return dict(np.load(path))
    vectors, labels, durations, provenance = [], [], [], []
    for session, meta in manifest["sessions"].items():
        if meta["split"] != split:
            continue
        waves, masks = session_tracks(meta)
        for name in NAMES:
            if name not in masks:
                continue
            for seconds in [1, 3]:
                candidates = []
                # Use fixed evaluation windows only; equally cap each speaker/duration.
                for window in meta["evaluation_windows"]:
                    for start in np.arange(
                        window["start"], window["start"] + 60 - seconds, seconds
                    ):
                        lo, hi = int(start / 0.032), int((start + seconds) / 0.032)
                        target = masks[name][lo:hi].mean()
                        others = np.stack([v[lo:hi] for k, v in masks.items() if k != name])
                        if target >= 0.75 and others.any(axis=0).mean() <= 0.15:
                            candidates.append(start)
                if not candidates:
                    continue
                indices = np.linspace(0, len(candidates) - 1, min(80, len(candidates))).astype(int)
                selected = np.array(candidates)[indices]
                clips = [
                    sum(
                        np.array(w[int(s * SR) : int(s * SR) + seconds * SR])
                        for w in waves.values()
                    )
                    for s in selected
                ]
                vectors.extend(embed(clips, embedder))
                labels.extend([NAMES.index(name)] * len(clips))
                durations.extend([seconds] * len(clips))
                provenance.extend([(int(session), float(s)) for s in selected])
                print("EVAL_FEATURES", split, name, seconds, len(clips), flush=True)
    result = dict(
        vectors=np.array(vectors),
        labels=np.array(labels),
        durations=np.array(durations),
        provenance=np.array(provenance),
    )
    np.savez(path, **result)
    return result


def probabilities(model, x):
    return softmax(x @ model["weights"].T + model["bias"], axis=1)


def scores(model, data):
    p = probabilities(model, data["vectors"])
    y, pred = data["labels"], p.argmax(axis=1)
    per = {}
    for i, name in enumerate(NAMES):
        mask = y == i
        per[name] = dict(
            n=int(mask.sum()),
            accuracy=float((pred[mask] == i).mean()) if mask.any() else None,
            loss=float(-np.log(np.maximum(p[mask, i], 1e-12)).mean()) if mask.any() else None,
        )
    return dict(
        macro_loss=float(np.mean([v["loss"] for v in per.values() if v["n"]])),
        macro_accuracy=float(np.mean([v["accuracy"] for v in per.values() if v["n"]])),
        micro_accuracy=float((pred == y).mean()),
        loss=float(log_loss(y, p, labels=range(6))),
        per_speaker=per,
        confusion=confusion_matrix(y, pred, labels=range(6)).tolist(),
        by_duration={
            str(s): float((pred[data["durations"] == s] == y[data["durations"] == s]).mean())
            for s in np.unique(data["durations"])
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["sessions"]) != {"55", "61", "62", "63"}:
        raise ValueError("Wait for complete corpus preparation")
    # Cache identities include supervision, input paths, augmentation code and source model.
    receipt = dict(
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=DEFAULT_DIARIZATION_MODEL,
        speakers=NAMES,
        seed=20260918,
        torch=torch.__version__,
    )
    receipt_path = args.output / "identity.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Changed experiment inputs; use a new output directory")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    torch.set_num_threads(4)
    embedder = _resolve_embedder(
        model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device=args.device
    )
    noise = np.load(args.corpus / "early_noise.npz")["power"]
    train = training_features(manifest, args.output, embedder, noise)
    dev = evaluation_features(manifest, "dev", args.output, embedder)
    centroids = np.stack(
        [unit(train["clean"][train["labels"] == i].mean(axis=0)) for i in range(6)]
    )
    candidates = {}
    for temperature in [5, 10, 20]:
        candidates[f"centroid_{temperature}"] = dict(
            weights=centroids * temperature, bias=np.zeros(6)
        )
    for kind in ["clean", "augmented"]:
        x = (
            train["clean"]
            if kind == "clean"
            else np.concatenate([train["clean"], train["augmented"]])
        )
        y = train["labels"] if kind == "clean" else np.tile(train["labels"], 2)
        for c in [0.1, 1, 10, 100]:
            head = LogisticRegression(
                C=c, class_weight="balanced", max_iter=2000, random_state=20260918
            )
            head.fit(x, y)
            assert np.array_equal(head.classes_, np.arange(6))
            candidates[f"{kind}_C{c}"] = dict(weights=head.coef_, bias=head.intercept_)
    dev_scores = {name: scores(model, dev) for name, model in candidates.items()}
    selected = min(dev_scores, key=lambda name: dev_scores[name]["macro_loss"])
    report = dict(
        selection_rule="minimum dev macro speaker cross entropy; no test tuning",
        reference_type=manifest["reference_type"],
        speakers=NAMES,
        dev=dev_scores,
        selected=selected,
    )
    (args.output / "selection.json").write_text(json.dumps(report, indent=2))
    model = candidates[selected]
    np.savez(args.output / "speaker_model.npz", **model, centroids=centroids, names=NAMES)
    print("SELECTED", selected, dev_scores[selected], flush=True)
    # Freeze the selection before loading/evaluating test examples.
    test = evaluation_features(manifest, "test", args.output, embedder)
    report["test"] = scores(model, test)
    report["test_baseline_centroid"] = scores(candidates["centroid_10"], test)
    report["train_counts"] = dict(zip(NAMES, np.bincount(train["labels"]).tolist()))
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print("TEST", report["test"], flush=True)


if __name__ == "__main__":
    main()
