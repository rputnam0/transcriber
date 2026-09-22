#!/usr/bin/env python3
"""Apply a trained cross-session identity head to cached full-file diarization and words."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from train_early_speaker_identity import embed, probabilities, unit
from transcriber.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    DiarizationResult,
    DiarizationTurn,
    _resolve_embedder,
    load_audio_mono,
)
from transcriber.srt import _fmt_srt_ts, write_srt
from transcriber.transcript_pipeline import _assign_word_speakers, asr_result_from_segments


def cluster_features(wave, turns, embedder):
    """Collect only uncontested voice, then use fixed three-second speech chunks.

    Concatenation occurs only inside a diarizer cluster; it never uses reference labels.
    Short-only clusters retain their real duration and are never zero padded.
    """
    labels = sorted({turn["speaker"] for turn in turns})
    matrix, owners, seconds = [], [], {}
    for label in labels:
        pieces = []
        for turn in turns:
            if turn["speaker"] != label:
                continue
            intervals = [(turn["start"], turn["end"])]
            for other in turns:
                if other["speaker"] == label:
                    continue
                remaining = []
                for start, end in intervals:
                    if other["end"] <= start or other["start"] >= end:
                        remaining.append((start, end))
                    else:
                        if start < other["start"]:
                            remaining.append((start, other["start"]))
                        if other["end"] < end:
                            remaining.append((other["end"], end))
                intervals = remaining
            pieces.extend(
                wave[int(s * 16000) : int(e * 16000)] for s, e in intervals if e - s >= 0.1
            )
        speech = np.concatenate(pieces) if pieces else np.array([], np.float32)
        seconds[label] = len(speech) / 16000
        clips = [
            speech[i : i + 48000]
            for i in range(0, len(speech), 48000)
            if len(speech[i : i + 48000]) >= 8000
        ]
        if not clips:
            continue
        matrix.extend(embed(clips, embedder))
        owners.extend([label] * len(clips))
    return dict(
        vectors=np.array(matrix),
        owners=np.array(owners),
        labels=np.array(labels),
        clean_seconds=np.array([seconds[k] for k in labels]),
    )


def bind_clusters(features, model):
    result = {}
    names = model["names"].tolist()
    for label, seconds in zip(features["labels"], features["clean_seconds"]):
        vectors = features["vectors"][features["owners"] == label]
        if not len(vectors):
            result[str(label)] = dict(
                speaker=None,
                proposed_speaker=None,
                review_required=True,
                clean_seconds=float(seconds),
                reason="no uncontested speech",
            )
            continue
        # Mean posterior prevents a single aberrant chunk from dominating a long cluster.
        p = probabilities(model, vectors).mean(axis=0)
        order = np.argsort(p)[::-1]
        centroid = unit(vectors.mean(axis=0))
        cosines = model["centroids"] @ centroid
        winner = int(order[0])
        agreement = float((probabilities(model, vectors).argmax(axis=1) == winner).mean())
        reliable = p[winner] >= 0.65 and cosines[winner] >= 0.35 and agreement >= 0.6
        result[str(label)] = dict(
            speaker=names[winner] if reliable else None,
            proposed_speaker=names[winner],
            review_required=bool(not reliable or p[winner] < 0.85 or agreement < 0.85),
            mean_posterior=float(p[winner]),
            margin=float(p[winner] - p[order[1]]),
            cosine=float(cosines[winner]),
            chunk_agreement=agreement,
            clean_seconds=float(seconds),
            probabilities=dict(zip(names, p.tolist())),
            cosines=dict(zip(names, cosines.tolist())),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--names", type=Path, help="Optional JSON handle-to-display-name mapping")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--turn-policy", type=Path, help="Frozen development-selected local correction policy"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = dict(np.load(args.model))
    data = json.loads((args.cache / "diarization.json").read_text())
    asr = json.loads((args.cache / "asr.json").read_text())
    features_path = args.cache / "identity_features.npz"
    if features_path.exists():
        features = dict(np.load(features_path))
    else:
        wave = load_audio_mono(str(args.audio), sample_rate=16000)
        if hashlib.sha256(wave.tobytes()).hexdigest() != data["wave_sha256"]:
            raise ValueError("Audio differs from cached diarization")
        embedder = _resolve_embedder(
            model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device=args.device
        )
        features = cluster_features(wave, data["segments"], embedder)
        np.savez(features_path, **features)
    bindings = bind_clusters(features, model)
    cluster_bindings = dict(bindings)
    corrections = []
    if args.turn_policy:
        from refine_named_turns import corrected_names, turn_features

        policy = json.loads(args.turn_policy.read_text())["policy"]
        local_path = args.cache / "turn_features.npz"
        if local_path.exists():
            local = dict(np.load(local_path))
        else:
            wave = load_audio_mono(str(args.audio), sample_rate=16000)
            embedder = _resolve_embedder(
                model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device=args.device
            )
            local = turn_features(wave, data["segments"], embedder)
            np.savez(local_path, **local)
        names = corrected_names(data["segments"], bindings, local, model, policy)
        original = data["segments"]
        regular = []
        for i, (turn, name) in enumerate(zip(original, names)):
            key = f"{turn['speaker']}@{i}"
            binding = dict(bindings[turn["speaker"]])
            if name != binding.get("proposed_speaker"):
                corrections.append(
                    dict(
                        start=turn["start"],
                        end=turn["end"],
                        before=binding.get("proposed_speaker"),
                        after=name,
                    )
                )
                binding.update(
                    speaker=name,
                    proposed_speaker=name,
                    review_required=True,
                    reason="strong local voice differs from global cluster",
                )
            bindings[key] = binding
            regular.append(dict(turn, speaker=key))
        exclusive = []
        for turn in data["exclusive_segments"]:
            candidates = [
                i
                for i, item in enumerate(original)
                if item["speaker"] == turn["speaker"]
                and min(item["end"], turn["end"]) > max(item["start"], turn["start"])
            ]
            if candidates:
                i = max(
                    candidates,
                    key=lambda i: min(original[i]["end"], turn["end"])
                    - max(original[i]["start"], turn["start"]),
                )
                turn = dict(turn, speaker=regular[i]["speaker"])
            exclusive.append(turn)
        data = dict(data, segments=regular, exclusive_segments=exclusive)
    diar = DiarizationResult(
        segments=[DiarizationTurn(**v) for v in data["segments"]],
        exclusive_segments=[DiarizationTurn(**v) for v in data["exclusive_segments"]],
        metadata={},
    )
    assigned = _assign_word_speakers(asr_result_from_segments(asr), diar)
    display = json.loads(args.names.read_text()) if args.names else {}
    for segment in assigned:
        cluster = segment.get("speaker")
        binding = bindings.get(cluster, {})
        handle = binding.get("speaker")
        proposed = binding.get("proposed_speaker")
        segment.update(
            cluster=cluster,
            speaker_handle=handle,
            proposed_speaker=proposed,
            review_required=binding.get("review_required", True),
        )
        segment["speaker"] = (
            display.get(handle, handle)
            if handle
            else (f"Uncertain: {display.get(proposed,proposed)}" if proposed else "Unknown")
        )
        for word in segment.get("words", []):
            word.update(cluster=word.get("speaker"), speaker=segment["speaker"])
    (args.cache / "named.json").write_text(json.dumps(assigned, indent=2))
    text = "\n".join(
        f"[{_fmt_srt_ts(s['start']).replace(',', '.')}–"
        f"{_fmt_srt_ts(s['end']).replace(',', '.')}] {s['speaker']}: {s['text']}"
        for s in assigned
    )
    (args.cache / "named.txt").write_text(text + "\n")
    write_srt(
        args.cache / "named.srt",
        [
            (i + 1, s["start"], s["end"], f"{s['speaker']}: {s['text']}")
            for i, s in enumerate(assigned)
        ],
    )
    report = dict(
        bindings=cluster_bindings,
        local_corrections=corrections,
        segments=len(assigned),
        named_segments=sum(bool(s["speaker_handle"]) for s in assigned),
        review_segments=sum(s["review_required"] for s in assigned),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        audio_sha256=data["wave_sha256"],
        duration=data["duration"],
        caution="Model posteriors are not calibrated early-domain accuracy; no early human reference.",
    )
    (args.cache / "attribution.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
