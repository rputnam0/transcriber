#!/usr/bin/env python3
"""End-to-end named activity evaluation on fixed, held-out mono mixture windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from attribute_cached_diarization import bind_clusters, cluster_features
from prepare_early_domain_corpus import NAMES, SR
from train_early_speaker_identity import session_tracks
from transcriber.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    _annotation_to_segments,
    _load_pipeline,
)


def activity(turns, bindings, frames, guarded):
    result = np.zeros((frames, len(NAMES)), bool)
    times = (np.arange(frames) + 0.5) / 50
    for turn in turns:
        binding = bindings[turn["speaker"]]
        name = binding.get("speaker" if guarded else "proposed_speaker")
        if name in NAMES:
            result[:, NAMES.index(name)] |= (times >= turn["start"]) & (times < turn["end"])
    return result


def activity_metrics(reference, predicted):
    nref, npred = reference.sum(axis=1), predicted.sum(axis=1)
    correct = (reference & predicted).sum(axis=1)
    misses, false = np.maximum(nref - npred, 0), np.maximum(npred - nref, 0)
    confusion = np.minimum(nref, npred) - correct
    per = {}
    for i, name in enumerate(NAMES):
        tp = (reference[:, i] & predicted[:, i]).sum()
        ref, pred = reference[:, i].sum(), predicted[:, i].sum()
        per[name] = dict(
            reference_seconds=float(ref / 50),
            predicted_seconds=float(pred / 50),
            precision=float(tp / max(pred, 1)),
            recall=float(tp / max(ref, 1)),
            f1=float(2 * tp / max(ref + pred, 1)),
        )
    single, overlap = nref == 1, nref > 1
    return dict(
        reference_speaker_seconds=float(nref.sum() / 50),
        weak_named_activity_error_rate=float(
            (misses + false + confusion).sum() / max(nref.sum(), 1)
        ),
        missed_speaker_seconds=float(misses.sum() / 50),
        false_speaker_seconds=float(false.sum() / 50),
        confused_speaker_seconds=float(confusion.sum() / 50),
        exact_on_single=float((reference[single] == predicted[single]).all(axis=1).mean()),
        exact_on_overlap=float((reference[overlap] == predicted[overlap]).all(axis=1).mean()),
        overlap_seconds=float(overlap.sum() / 50),
        macro_f1=float(np.mean([v["f1"] for v in per.values() if v["reference_seconds"] > 0])),
        per_speaker=per,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    model = dict(np.load(args.model))
    baseline = dict(model, weights=model["centroids"] * 10, bias=np.zeros(6))
    torch.set_num_threads(4)
    pipeline = _load_pipeline(DEFAULT_DIARIZATION_MODEL, device=args.device, hf_token=get_token())
    if str(pipeline.device) != args.device:
        raise RuntimeError("Pipeline device mismatch")
    refs = []
    predictions = {
        k: [] for k in ["trained_forced", "trained_guarded", "centroid_forced", "centroid_guarded"]
    }
    windows = []
    for session, meta in manifest["sessions"].items():
        if meta["split"] != args.split:
            continue
        waves, masks = session_tracks(meta)
        for index, window in enumerate(meta["evaluation_windows"]):
            start, duration = window["start"], window["duration"]
            folder = args.output / f"session{session}_{index:02d}"
            folder.mkdir(exist_ok=True)
            lo, hi = int(start * SR), int((start + duration) * SR)
            wave = sum(np.array(w[lo:hi]) for w in waves.values())
            frames = int(duration * 50)
            times = start + (np.arange(frames) + 0.5) / 50
            reference = np.stack(
                [
                    (
                        masks[name][(times / 0.032).astype(int)]
                        if name in masks
                        else np.zeros(frames, bool)
                    )
                    for name in NAMES
                ],
                axis=1,
            )
            turns_path = folder / "turns.json"
            if turns_path.exists():
                turns = json.loads(turns_path.read_text())
            else:
                result = pipeline(
                    {"waveform": torch.from_numpy(wave)[None, :], "sample_rate": SR}, max_speakers=7
                )
                turns = [vars(t) for t in _annotation_to_segments(result)]
                turns_path.write_text(json.dumps(turns, indent=2))
            features_path = folder / "features.npz"
            if features_path.exists():
                features = dict(np.load(features_path))
            else:
                features = cluster_features(wave, turns, pipeline._embedding)
                np.savez(features_path, **features)
            bindings = {
                "trained": bind_clusters(features, model),
                "centroid": bind_clusters(features, baseline),
            }
            for kind in predictions:
                method, mode = kind.split("_")
                predictions[kind].append(
                    activity(turns, bindings[method], frames, mode == "guarded")
                )
            refs.append(reference)
            windows.append(dict(session=session, index=index, **window, bindings=bindings))
            print("WINDOW_READY", session, index, flush=True)
    reference = np.concatenate(refs)
    report = dict(
        split=args.split,
        speakers=NAMES,
        seconds=len(reference) / 50,
        reference_type="isolated source Silero VAD; weak supervision, not human DER",
        inference_inputs="mono sum only; reference stems withheld from diarization/identity",
        metrics={k: activity_metrics(reference, np.concatenate(v)) for k, v in predictions.items()},
        windows=windows,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        args.output / "frame_predictions.npz",
        reference=reference,
        **{k: np.concatenate(v) for k, v in predictions.items()},
    )
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
