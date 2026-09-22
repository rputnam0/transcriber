#!/usr/bin/env python3
"""Score named transcript words against time-matched isolated-track ASR (NOT human gold)."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import get_token

from attribute_cached_diarization import bind_clusters, cluster_features
from prepare_early_domain_corpus import NAMES, SR
from refine_named_turns import corrected_names, turn_features
from train_early_speaker_identity import session_tracks
from transcriber.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    _annotation_to_segments,
    _load_pipeline,
)
from transcriber.parakeet_backend import load_model, transcribe_file
from transcriber.transcript_pipeline import _choose_turn_label
from transcriber.diarization import DiarizationTurn


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def predict_labels(segments, regular, exclusive, smooth):
    output = []
    for segment in segments:
        words = segment.get("words", [])
        labels = [_choose_turn_label(w["start"], w["end"], exclusive, regular) for w in words]
        # Sentence consistency is only an experimental comparator; do not silently ship it.
        if smooth and labels:
            votes = {
                label: sum(
                    min(w["end"] - w["start"], 0.6)
                    for w, assigned in zip(words, labels)
                    if assigned == label
                )
                for label in set(labels)
                if label
            }
            if votes:
                winner = max(votes, key=votes.get)
                if votes[winner] / max(sum(votes.values()), 1e-8) >= 0.8:
                    labels = [winner] * len(labels)
        output.extend((word, label) for word, label in zip(words, labels))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    model = dict(np.load(args.identity))
    policy = json.loads(args.policy.read_text())["policy"]
    torch.set_num_threads(4)
    pipeline = _load_pipeline(DEFAULT_DIARIZATION_MODEL, device="mps", hf_token=get_token())
    asr_model = load_model("parakeet", compute_type="float32")
    results = []
    with tempfile.TemporaryDirectory() as temp:
        audio = Path(temp) / "clip.wav"
        for session, meta in manifest["sessions"].items():
            if meta["split"] != args.split:
                continue
            waves, masks = session_tracks(meta)
            for index, window in enumerate(meta["evaluation_windows"]):
                folder = args.output / f"{session}_{index:02d}"
                folder.mkdir(exist_ok=True)
                start = window["start"]
                lo, hi = int(start * SR), int((start + window["duration"]) * SR)
                clips = {name: np.array(w[lo:hi]) for name, w in waves.items()}
                clips["mix"] = sum(clips.values())
                transcripts = {}
                for name, wave in clips.items():
                    path = folder / f"{name}.json"
                    if path.exists():
                        transcripts[name] = json.loads(path.read_text())
                    else:
                        sf.write(audio, wave, SR, subtype="FLOAT")
                        transcripts[name] = transcribe_file(str(audio), asr_model)
                        path.write_text(json.dumps(transcripts[name]))
                diar_path = folder / "diarization.json"
                if diar_path.exists():
                    data = json.loads(diar_path.read_text())
                else:
                    wave = clips["mix"]
                    result = pipeline(
                        {"waveform": torch.from_numpy(wave)[None, :], "sample_rate": SR},
                        max_speakers=7,
                    )
                    turns = [vars(t) for t in _annotation_to_segments(result)]
                    exclusive = [
                        vars(t)
                        for t in _annotation_to_segments(result.exclusive_speaker_diarization)
                    ]
                    features = cluster_features(wave, turns, pipeline._embedding)
                    bindings = bind_clusters(features, model)
                    local = turn_features(wave, turns, pipeline._embedding)
                    names = corrected_names(turns, bindings, local, model, policy)
                    regular = [dict(t, speaker=n or "Unknown") for t, n in zip(turns, names)]
                    for turn in exclusive:
                        candidates = [
                            i for i, t in enumerate(turns) if t["speaker"] == turn["speaker"]
                        ]
                        i = max(
                            candidates,
                            key=lambda i: max(
                                0,
                                min(turn["end"], turns[i]["end"])
                                - max(turn["start"], turns[i]["start"]),
                            ),
                        )
                        turn["speaker"] = names[i] or "Unknown"
                    data = dict(regular=regular, exclusive=exclusive)
                    diar_path.write_text(json.dumps(data))
                regular = [DiarizationTurn(**t) for t in data["regular"]]
                exclusive = [DiarizationTurn(**t) for t in data["exclusive"]]
                refs = {}
                for name, segments in transcripts.items():
                    if name == "mix":
                        continue
                    for segment in segments:
                        for word in segment.get("words", []):
                            center = (word["start"] + word["end"]) / 2
                            frame = min(int((start + center) / 0.032), len(masks[name]) - 1)
                            key = normalized(word["word"])
                            if key and masks[name][frame]:
                                refs.setdefault(key, []).append((center, name))
                base = predict_labels(transcripts["mix"], regular, exclusive, False)
                smooth = predict_labels(transcripts["mix"], regular, exclusive, True)
                for (word, label), (_, smoothed) in zip(base, smooth):
                    center = (word["start"] + word["end"]) / 2
                    eligible = {
                        name
                        for t, name in refs.get(normalized(word["word"]), [])
                        if abs(center - t) <= 0.5
                    }
                    results.append(
                        dict(
                            session=session,
                            window=index,
                            start=float(start + word["start"]),
                            word=word["word"],
                            predicted=label,
                            sentence_consistent=smoothed,
                            reference=next(iter(eligible)) if len(eligible) == 1 else None,
                        )
                    )
                print("WORD_WINDOW_READY", session, index, flush=True)
    known = [r for r in results if r["reference"]]
    metrics = {}
    for method in ["predicted", "sentence_consistent"]:
        per = {
            name: dict(
                n=sum(r["reference"] == name for r in known),
                correct=sum(r["reference"] == name and r[method] == name for r in known),
            )
            for name in NAMES
        }
        metrics[method] = dict(
            micro_accuracy=sum(r[method] == r["reference"] for r in known) / max(len(known), 1),
            macro_accuracy=float(np.mean([v["correct"] / v["n"] for v in per.values() if v["n"]])),
            per_speaker=per,
        )
    report = dict(
        split=args.split,
        reference_type="isolated-track automatic ASR + source VAD + same-word match within 0.5 s; not human gold",
        recognized_words=len(results),
        uniquely_matched_words=len(known),
        matching_coverage=len(known) / max(len(results), 1),
        metrics=metrics,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    (args.output / "words.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
