"""Frozen turn-level policy and cross-session human-reference retrieval on reviewed turns.

This is an offline experiment. It never modifies production predictions or reviews.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import get_token
from refine_named_turns import apply_local_identity, turn_features
from train_early_speaker_identity import probabilities
from transcriber.diarization import DEFAULT_DIARIZATION_MODEL, _resolve_embedder


def score(rows, prediction):
    scored = [r for r in rows if r["truth"] is not None and r["positive_label_eligible"]]
    results = {}
    for session in ["all", *sorted({r["session"] for r in scored})]:
        group = scored if session == "all" else [r for r in scored if r["session"] == session]
        fixed = broken = right = 0
        for r in group:
            value = prediction[(r["session"], r["turn_id"])]
            right += value == r["truth"]
            fixed += r["predicted"] != r["truth"] and value == r["truth"]
            broken += r["predicted"] == r["truth"] and value != r["truth"]
        results[str(session)] = dict(
            correct=right,
            total=len(group),
            accuracy=right / len(group),
            errors_fixed=fixed,
            previously_correct_broken=broken,
        )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument(
        "--deployments",
        type=Path,
        default=Path.home() / ".cache/transcriber/moss-training-20260919/deployment",
    )
    args = p.parse_args()
    rows = [
        r
        for r in json.loads((args.analysis / "turns.json").read_text())
        if not r["legacy"] and r["verdict"] != "ungraded"
    ]
    model = dict(np.load(".outputs/early_domain_20260918/identity/speaker_model.npz"))
    config = json.loads(Path("config/early_session_rosters.json").read_text())
    names = config["display_names"]
    handles = model["names"].tolist()
    policy = json.loads(
        Path(".outputs/early_domain_20260918/diarization_dev/turn_refinement.json").read_text()
    )["policy"]
    torch.set_num_threads(4)
    embedder = _resolve_embedder(
        model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device="mps"
    )
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["session"], r["cut_id"])].append(r)
    observations = []
    for (session, cut), group in sorted(grouped.items()):
        base = args.deployments / f"session{session}_selected_20260919"
        record = json.loads((base / "predictions" / f"{cut}.json").read_text())
        path = args.analysis / "features" / f"session{session}_{cut}.npz"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            features = dict(np.load(path))
        else:
            wave, _ = sf.read(record["audio"], dtype="float32")
            features = turn_features(wave, record["segments"], embedder)
            np.savez(path, **features)
        probs = (
            probabilities(model, features["vectors"])
            if len(features["vectors"])
            else np.empty((0, len(handles)))
        )
        cosines = (
            features["vectors"] @ model["centroids"].T
            if len(features["vectors"])
            else np.empty((0, len(handles)))
        )
        binding = json.loads((base / "attribution" / f"{cut}.json").read_text())["bindings"]
        # Exercise the production decision function; display-name baselines are
        # deliberately preserved when there is no qualifying override.
        baseline_names = [None] * len(record["segments"])
        local_names, _ = apply_local_identity(
            baseline_names, features, model, policy, config["session_rosters"][str(session)]
        )
        for row in group:
            matches = [
                i
                for i, t in enumerate(record["segments"])
                if abs(t["start"] + record["start"] - row["start"]) < 1e-5
                and abs(t["end"] + record["start"] - row["end"]) < 1e-5
                and t["text"] == row["text"]
                and t["speaker"] == row["local_speaker"]
            ]
            if len(matches) != 1:
                raise ValueError("Ambiguous review/prediction turn join")
            index = matches[0]
            fidx = np.flatnonzero(features["indices"] == index)
            obs = dict(row, vector=None, local_prediction=None, local_qualifies=False)
            allowed = config["session_rosters"][str(session)]
            mapping = {f"S{i+1:02}": h for i, h in enumerate(record["identity_names"])}
            direct = mapping.get(row["local_speaker"])
            obs["direct_candidate"] = names[direct] if direct in allowed else row["predicted"]
            b = binding.get(row["local_speaker"], {})
            obs["reliable_cluster_candidate"] = (
                row["predicted"] if b.get("speaker") else obs["direct_candidate"]
            )
            if len(fidx):
                j = int(fidx[0])
                winner = int(probs[j].argmax())
                obs.update(
                    vector=features["vectors"][j].tolist(),
                    local_prediction=names[handles[winner]],
                    local_seconds=float(features["durations"][j]),
                    local_posterior=float(probs[j, winner]),
                    local_cosine=float(cosines[j, winner]),
                    local_qualifies=local_names[index] is not None,
                )
            observations.append(obs)
        print("FEATURES", session, cut, flush=True)
    strategies = {
        name: {}
        for name in [
            "baseline",
            "decoder_when_in_roster",
            "reliable_cluster_else_decoder",
            "frozen_local_policy",
            "cross_session_human_retrieval",
        ]
    }
    changes = []
    for row in observations:
        key = (row["session"], row["turn_id"])
        base = row["predicted"]
        strategies["baseline"][key] = base
        strategies["decoder_when_in_roster"][key] = row["direct_candidate"]
        strategies["reliable_cluster_else_decoder"][key] = row["reliable_cluster_candidate"]
        strategies["frozen_local_policy"][key] = (
            row["local_prediction"] if row["local_qualifies"] else base
        )
        # Predeclared conservative retrieval rule. No fitting or threshold selection
        # uses labels from the held-out recording. All turns in that session stay out.
        refs = [
            r
            for r in observations
            if r["session"] != row["session"]
            and r["positive_label_eligible"]
            and r["vector"] is not None
            and r["local_seconds"] >= 1
            and r["overlap_fraction"] == 0
        ]
        pred = base
        if (
            row["vector"] is not None
            and row["local_seconds"] >= 0.75
            and row["overlap_fraction"] == 0
        ):
            scores = {}
            for name in config["display_names"].values():
                vectors = [r["vector"] for r in refs if r["truth"] == name]
                if len(vectors) >= 3:
                    similarities = np.array(vectors) @ np.array(row["vector"])
                    scores[name] = float(np.sort(similarities)[-3:].mean())
            order = sorted(scores.items(), key=lambda x: -x[1])
            if len(order) >= 2 and order[0][1] >= 0.6 and order[0][1] - order[1][1] >= 0.1:
                pred = order[0][0]
        strategies["cross_session_human_retrieval"][key] = pred
        for name, preds in strategies.items():
            if preds[key] != base:
                changes.append(
                    dict(
                        session=row["session"],
                        turn_id=row["turn_id"],
                        strategy=name,
                        before=base,
                        after=preds[key],
                        truth=row["truth"],
                    )
                )
    report = dict(
        experiments={name: score(rows, preds) for name, preds in strategies.items()},
        coverage=dict(
            reviewed=len(rows),
            has_local_embedding=sum(r["vector"] is not None for r in observations),
            wrong_with_local_embedding=sum(
                r["verdict"] == "wrong" and r["vector"] is not None for r in observations
            ),
        ),
        policy=policy,
        retrieval_rule="Other sessions only; reference clean seconds >=1, no predicted overlap, >=3 reference turns/class; query >=0.75s; top-3 mean cosine >=0.6 and margin >=0.1. Thresholds not tuned on these grades.",
        exclusions="One wrong turn without correction and three unsure turns excluded from identity-accuracy comparison.",
        changes=changes,
        production_changed=False,
    )
    (args.analysis / "identity_experiment.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.analysis / "local_observations.json").write_text(
        json.dumps(observations, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
