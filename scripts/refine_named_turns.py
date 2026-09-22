#!/usr/bin/env python3
"""Select conservative local identity corrections on development recordings only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from attribute_cached_diarization import bind_clusters
from evaluate_early_named_diarization import activity_metrics
from prepare_early_domain_corpus import NAMES, SR
from train_early_speaker_identity import embed, probabilities, session_tracks
from transcriber.diarization import DEFAULT_DIARIZATION_MODEL, _resolve_embedder


def turn_features(wave, turns, embedder):
    clips, indices, lengths = [], [], []
    for index, turn in enumerate(turns):
        intervals = [(turn["start"], turn["end"])]
        for other in turns:
            if other["speaker"] == turn["speaker"]:
                continue
            remaining = []
            for start, end in intervals:
                if other["start"] >= end or other["end"] <= start:
                    remaining.append((start, end))
                else:
                    if start < other["start"]:
                        remaining.append((start, other["start"]))
                    if other["end"] < end:
                        remaining.append((other["end"], end))
            intervals = remaining
        if not intervals:
            continue
        start, end = max(intervals, key=lambda pair: pair[1] - pair[0])
        # Equal-length buckets contain real audio only: never zero pad speaker crops.
        length = min(3.0, np.floor((end - start) * 4) / 4)
        if length < 0.5:
            continue
        start += ((end - start) - length) / 2
        clips.append(np.array(wave[int(start * SR) : int(start * SR) + int(length * SR)]))
        indices.append(index)
        lengths.append(length)
    vectors = embed(clips, embedder) if clips else np.empty((0, embedder.dimension))
    return dict(vectors=vectors, indices=np.array(indices), durations=np.array(lengths))


def apply_local_identity(names, features, model, policy, allowed=None):
    """Override individual turns only with the frozen strong-evidence policy.

    Preserve the caller's hybrid fallback and never rewrite a whole decoder cluster.
    Session eligibility restricts candidates without renormalizing confidence.
    """
    names = list(names)
    changes = []
    if not policy.get("enabled") or not len(features["vectors"]):
        return names, changes
    identities = model["names"].tolist()
    p = probabilities(model, features["vectors"])
    cosine = features["vectors"] @ model["centroids"].T
    for row, index in enumerate(features["indices"]):
        winner = int(p[row].argmax())
        order = np.sort(p[row])
        if (
            (allowed is None or identities[winner] in allowed)
            and features["durations"][row] >= policy["minimum_seconds"]
            and p[row, winner] >= policy["posterior"]
            and order[-1] - order[-2] >= 0.5
            and cosine[row, winner] >= 0.4
        ):
            index = int(index)
            candidate = identities[winner]
            if candidate != names[index]:
                changes.append(
                    dict(
                        turn_index=index,
                        before=names[index],
                        after=candidate,
                        seconds=float(features["durations"][row]),
                        posterior=float(p[row, winner]),
                        margin=float(order[-1] - order[-2]),
                        cosine=float(cosine[row, winner]),
                    )
                )
                names[index] = candidate
    return names, changes


def corrected_names(turns, bindings, features, model, policy):
    names = [bindings[t["speaker"]].get("proposed_speaker") for t in turns]
    return apply_local_identity(names, features, model, policy)[0]


def named_activity(turns, names, frames=3000):
    times = (np.arange(frames) + 0.5) / 50
    out = np.zeros((frames, 6), bool)
    for turn, name in zip(turns, names):
        if name in NAMES:
            out[:, NAMES.index(name)] |= (times >= turn["start"]) & (times < turn["end"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--policy", type=Path, help="Frozen dev policy; required when evaluating test")
    args = p.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    report = json.loads((args.evaluation / "results.json").read_text())
    if report["split"] != "dev" and not args.policy:
        raise ValueError("Test needs frozen dev policy")
    model = dict(np.load(args.model))
    torch.set_num_threads(4)
    e = _resolve_embedder(model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device="mps")
    windows = []
    for window in report["windows"]:
        session, index = window["session"], window["index"]
        folder = args.evaluation / f"session{session}_{index:02d}"
        path = folder / "turn_features.npz"
        turns = json.loads((folder / "turns.json").read_text())
        if path.exists():
            features = dict(np.load(path))
        else:
            waves, _ = session_tracks(manifest["sessions"][session])
            start, end = int(window["start"] * SR), int((window["start"] + window["duration"]) * SR)
            wave = sum(np.array(w[start:end]) for w in waves.values())
            features = turn_features(wave, turns, e)
            np.savez(path, **features)
        bindings = bind_clusters(dict(np.load(folder / "features.npz")), model)
        windows.append((turns, features, bindings))
    policies = [dict(enabled=False)]
    if args.policy:
        policies.append(json.loads(args.policy.read_text())["policy"])
    else:
        policies.extend(
            dict(enabled=True, posterior=p, minimum_seconds=d)
            for p in [0.95, 0.9, 0.8]
            for d in [1.0, 0.75, 0.5]
        )
    reference = np.load(args.evaluation / "frame_predictions.npz")["reference"]
    comparisons, arrays = [], []
    for policy in policies:
        prediction = np.concatenate(
            [
                named_activity(turns, corrected_names(turns, b, f, model, policy))
                for turns, f, b in windows
            ]
        )
        metrics = activity_metrics(reference, prediction)
        comparisons.append(dict(policy=policy, metrics=metrics))
        arrays.append(prediction)
    selected = (
        len(policies) - 1
        if args.policy
        else max(range(len(policies)), key=lambda i: comparisons[i]["metrics"]["macro_f1"])
    )
    result = dict(
        split=report["split"],
        selection_rule="dev macro F1; ties prefer unchanged baseline, then stricter corrections",
        policy=policies[selected],
        comparisons=comparisons,
        selected=selected,
    )
    (args.evaluation / "turn_refinement.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.evaluation / "refined_frames.npz", reference=reference, predicted=arrays[selected]
    )
    print(json.dumps(comparisons[selected], indent=2), flush=True)


if __name__ == "__main__":
    main()
