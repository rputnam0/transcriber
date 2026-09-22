#!/usr/bin/env python3
"""Expand the frozen augmented logistic enrollment recipe using training sessions only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token
from sklearn.linear_model import LogisticRegression

from prepare_early_domain_corpus import NAMES
from train_early_speaker_identity import training_features, unit
from transcriber.diarization import DEFAULT_DIARIZATION_MODEL, _resolve_embedder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--previous-features", type=Path, required=True)
    p.add_argument("--noise", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    previous = dict(np.load(args.previous_features))
    previous_sessions = set(previous["provenance"][:, 0].astype(int))
    train_sessions = {int(s) for s, m in manifest["sessions"].items() if m["split"] == "train"}
    if not previous_sessions <= train_sessions or train_sessions != {43, 45, 47, 55, 61}:
        raise ValueError("Enrollment source session leakage")
    extra = {
        s: m
        for s, m in manifest["sessions"].items()
        if int(s) in train_sessions - previous_sessions
    }
    skipped = []
    for session, meta in extra.items():
        for name, track in list(meta["tracks"].items()):
            if track["duration"] < 30:
                skipped.append(dict(session=session, speaker=name, reason="track shorter than 30s"))
                del meta["tracks"][name]
    torch.set_num_threads(4)
    embedder = _resolve_embedder(
        model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device="mps"
    )
    new = training_features(
        dict(sessions=extra), args.output, embedder, np.load(args.noise)["power"]
    )
    train = {k: np.concatenate([previous[k], new[k]]) for k in previous}
    if set(train["provenance"][:, 0].astype(int)) != train_sessions:
        raise ValueError("Incomplete enrollment")
    x = np.concatenate([train["clean"], train["augmented"]])
    y = np.tile(train["labels"], 2)
    head = LogisticRegression(C=100, class_weight="balanced", max_iter=2000, random_state=20260918)
    head.fit(x, y)
    assert np.array_equal(head.classes_, np.arange(6))
    centroids = np.stack(
        [unit(train["clean"][train["labels"] == i].mean(axis=0)) for i in range(6)]
    )
    np.savez(
        args.output / "speaker_model.npz",
        weights=head.coef_,
        bias=head.intercept_,
        centroids=centroids,
        names=NAMES,
    )
    np.savez(args.output / "all_train_features.npz", **train)
    receipt = dict(
        train_sessions=sorted(train_sessions),
        skipped=skipped,
        counts=dict(zip(NAMES, np.bincount(train["labels"]).tolist())),
        recipe="existing augmented_C100; no new hyperparameter search",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        previous_features_sha256=hashlib.sha256(args.previous_features.read_bytes()).hexdigest(),
        new_feature_sessions=sorted(train_sessions - previous_sessions),
        test_used=False,
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
