#!/usr/bin/env python3
"""Compare global versus windowed identity on a prespecified continuous ten-minute mix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from attribute_cached_diarization import bind_clusters, cluster_features
from diarize_named_recording import windowed_diarization
from evaluate_early_named_diarization import activity_metrics
from prepare_early_domain_corpus import NAMES, SR
from refine_named_turns import corrected_names, named_activity, turn_features
from train_early_speaker_identity import session_tracks
from transcriber.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    _annotation_to_segments,
    _load_pipeline,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    session, meta = next(
        (s, m) for s, m in manifest["sessions"].items() if m["split"] == args.split
    )
    waves, masks = session_tracks(meta)
    start, duration = 1200, 600
    wave = sum(np.array(w[start * SR : (start + duration) * SR]) for w in waves.values())
    times = start + (np.arange(duration * 50) + 0.5) / 50
    reference = np.stack([masks[n][(times / 0.032).astype(int)] for n in NAMES], axis=1)
    model = dict(np.load(args.identity))
    policy = json.loads(args.policy.read_text())["policy"]
    pipeline = _load_pipeline(DEFAULT_DIARIZATION_MODEL, device="mps", hf_token=get_token())
    path = args.output / "global.json"
    if path.exists():
        global_data = json.loads(path.read_text())
    else:
        result = pipeline(
            {"waveform": torch.from_numpy(wave)[None, :], "sample_rate": SR}, max_speakers=7
        )
        turns = [vars(t) for t in _annotation_to_segments(result)]
        bindings = bind_clusters(cluster_features(wave, turns, pipeline._embedding), model)
        names = corrected_names(
            turns, bindings, turn_features(wave, turns, pipeline._embedding), model, policy
        )
        global_data = dict(turns=turns, names=names, bindings=bindings)
        path.write_text(json.dumps(global_data, indent=2))
    global_pred = named_activity(global_data["turns"], global_data["names"], duration * 50)
    windowed = windowed_diarization(wave, model, policy, args.output / "windows")
    window_pred = named_activity(
        windowed["segments"], [t["speaker"] for t in windowed["segments"]], duration * 50
    )
    metrics = {
        k: activity_metrics(reference, v)
        for k, v in [("global", global_pred), ("windowed", window_pred)]
    }
    report = dict(
        session=session,
        split=args.split,
        start=start,
        duration=duration,
        metrics=metrics,
        reference_type="isolated source VAD, not human gold",
    )
    if args.split == "dev":
        report["selected"] = max(metrics, key=lambda k: metrics[k]["macro_f1"])
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        args.output / "frames.npz",
        reference=reference,
        global_pred=global_pred,
        windowed=window_pred,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
