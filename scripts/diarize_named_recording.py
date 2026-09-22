#!/usr/bin/env python3
"""Named diarization in overlapping windows, using the evaluated one-minute inference path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from attribute_cached_diarization import bind_clusters, cluster_features
from refine_named_turns import corrected_names, turn_features
from transcriber.diarization import (
    DEFAULT_DIARIZATION_MODEL,
    DiarizationResult,
    DiarizationTurn,
    _annotation_to_segments,
    _load_pipeline,
    load_audio_mono,
)
from transcriber.srt import _fmt_srt_ts, write_srt
from transcriber.transcript_pipeline import _assign_word_speakers, asr_result_from_segments


def windowed_diarization(wave, model, policy, output, device="mps"):
    output.mkdir(parents=True, exist_ok=True)
    pipeline = _load_pipeline(DEFAULT_DIARIZATION_MODEL, device=device, hf_token=get_token())
    duration = len(wave) / 16000
    starts = list(np.arange(0, max(0, duration - 60) + 0.001, 50))
    if starts[-1] < duration - 60:
        starts.append(duration - 60)
    centers = [s + min(60, duration - s) / 2 for s in starts]
    boundaries = [0] + [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])] + [duration]
    regular, exclusive, receipts = [], [], []
    for index, start in enumerate(starts):
        path = output / f"window_{index:04d}.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            clip = wave[int(start * 16000) : int(min(start + 60, duration) * 16000)]
            result = pipeline(
                {"waveform": torch.from_numpy(clip.copy())[None, :], "sample_rate": 16000},
                max_speakers=7,
            )
            turns = [vars(t) for t in _annotation_to_segments(result)]
            exc = [vars(t) for t in _annotation_to_segments(result.exclusive_speaker_diarization)]
            features = cluster_features(clip, turns, pipeline._embedding)
            bindings = bind_clusters(features, model)
            local = turn_features(clip, turns, pipeline._embedding)
            names = corrected_names(turns, bindings, local, model, policy)
            named = []
            for turn, name in zip(turns, names):
                binding = bindings[turn["speaker"]]
                named.append(
                    dict(
                        turn,
                        speaker=name or "Unknown",
                        review_required=binding["review_required"]
                        or name != binding.get("proposed_speaker"),
                        original_cluster=turn["speaker"],
                    )
                )
            for turn in exc:
                candidates = [i for i, t in enumerate(turns) if t["speaker"] == turn["speaker"]]
                best = max(
                    candidates,
                    key=lambda i: max(
                        0, min(turn["end"], turns[i]["end"]) - max(turn["start"], turns[i]["start"])
                    ),
                )
                turn.update(
                    speaker=names[best] or "Unknown", review_required=named[best]["review_required"]
                )
            data = dict(start=float(start), regular=named, exclusive=exc, bindings=bindings)
            path.write_text(json.dumps(data, indent=2))
        lower, upper = boundaries[index : index + 2]
        for key, dest in [("regular", regular), ("exclusive", exclusive)]:
            for t in data[key]:
                a, b = max(lower, start + t["start"]), min(upper, start + t["end"])
                if b > a:
                    dest.append(dict(t, start=float(a), end=float(b), window=index))
        receipts.append(
            dict(start=float(start), core=[float(lower), float(upper)], bindings=data["bindings"])
        )
        print("NAMED_WINDOW", index, round(float(start), 1), flush=True)
    return dict(segments=regular, exclusive_segments=exclusive, duration=duration, windows=receipts)


def export_transcript(asr, data, output):
    regular = [DiarizationTurn(t["start"], t["end"], t["speaker"]) for t in data["segments"]]
    exclusive = [
        DiarizationTurn(t["start"], t["end"], t["speaker"]) for t in data["exclusive_segments"]
    ]
    assigned = _assign_word_speakers(
        asr_result_from_segments(asr), DiarizationResult(regular, exclusive, {})
    )
    compact = []
    for s in assigned:
        s["review_required"] = not s["speaker"] or any(
            t.get("review_required")
            for t in data["segments"]
            if t["speaker"] == s["speaker"]
            and min(t["end"], s["end"]) > max(t["start"], s["start"])
        )
        s["overlapping_speakers"] = sorted(
            {
                t["speaker"]
                for t in data["segments"]
                if t["speaker"] != s["speaker"]
                and min(t["end"], s["end"]) > max(t["start"], s["start"])
            }
        )
        if s["overlapping_speakers"]:
            s["review_required"] = True
        if (
            compact
            and s["speaker"]
            and compact[-1]["speaker"] == s["speaker"]
            and s["start"] - compact[-1]["end"] <= 0.75
        ):
            prev = compact[-1]
            prev["end"] = s["end"]
            prev["text"] += " " + s["text"]
            prev["words"].extend(s.get("words", []))
            prev["review_required"] |= s["review_required"]
            prev["overlapping_speakers"] = sorted(
                set(prev["overlapping_speakers"] + s["overlapping_speakers"])
            )
        else:
            compact.append(s)
    (output / "named.json").write_text(json.dumps(compact, indent=2))
    text = "\n".join(
        f"[{_fmt_srt_ts(s['start'])}–{_fmt_srt_ts(s['end'])}] "
        f"{s['speaker'] or 'Unknown'}{' [review]' if s['review_required'] else ''}: {s['text']}"
        for s in compact
    )
    (output / "named.txt").write_text(text + "\n")
    write_srt(
        output / "named.srt",
        [
            (i + 1, s["start"], s["end"], f"{s['speaker'] or 'Unknown'}: {s['text']}")
            for i, s in enumerate(compact)
        ],
    )
    return dict(
        segments=len(compact),
        named_segments=sum(bool(s["speaker"]) for s in compact),
        review_segments=sum(s["review_required"] for s in compact),
        duration=data["duration"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--asr", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    wave = load_audio_mono(str(args.audio), sample_rate=16000)
    identity = dict(
        wave_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
        model_sha256=hashlib.sha256(args.identity.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        window=60,
        stride=50,
    )
    receipt = args.output / "input.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Input cache mismatch")
    receipt.write_text(json.dumps(identity, indent=2))
    model = dict(np.load(args.identity))
    policy = json.loads(args.policy.read_text())["policy"]
    data = windowed_diarization(wave, model, policy, args.output / "windows")
    (args.output / "diarization.json").write_text(json.dumps(data, indent=2))
    summary = export_transcript(json.loads(args.asr.read_text()), data, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
