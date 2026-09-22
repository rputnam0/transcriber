#!/usr/bin/env python3
"""Re-transcribe isolated sources in speech-bounded crops; preserve long-window references.

Owner labels are known. Lexical content remains automatic supervision, not human gold.
Crop selection uses source VAD and existing source words, never candidate predictions.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from prepare_early_domain_corpus import NAMES, SR
from test_moss_mac import save


def speech_crops(mask, duration, *, merge_gap=0.8, padding=0.35, max_core=14.0):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    runs = []
    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        start, end = lo * 0.032, min(hi * 0.032, duration)
        if end - start < 0.16:
            continue
        if runs and start - runs[-1][1] <= merge_gap:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    crops = []
    for start, end in runs:
        start, end = max(0, start - padding), min(duration, end + padding)
        left = start
        while left < end:
            right = min(left + max_core, end)
            crops.append(
                dict(
                    core_start=left,
                    core_end=right,
                    audio_start=max(start, left - 0.4),
                    audio_end=min(end, right + 0.4),
                )
            )
            left = right
    return crops


def crop_words(segments, crop):
    result = []
    for segment in segments:
        words = []
        for word in segment.get("words", []):
            shifted = dict(
                word,
                start=word["start"] + crop["audio_start"],
                end=word["end"] + crop["audio_start"],
            )
            midpoint = (shifted["start"] + shifted["end"]) / 2
            if crop["core_start"] <= midpoint < crop["core_end"]:
                words.append(shifted)
        if words:
            result.append(
                dict(
                    start=words[0]["start"],
                    end=words[-1]["end"],
                    text=" ".join(w["word"] for w in words),
                    words=words,
                )
            )
    return result


def missing_speech_crops(mask, duration, original):
    """Recover wholly omitted speech regions; preserve existing words and their context."""
    words = [w for s in original for w in s.get("words", [])]
    missing = []
    for crop in speech_crops(mask, duration):
        if any(
            min(w["end"] + 0.1, crop["core_end"]) > max(w["start"] - 0.1, crop["core_start"])
            for w in words
        ):
            continue
        # Tiny isolated crops can destabilize Parakeet. Supply at least five seconds
        # of surrounding source audio; keep words only in the original speech region.
        center = (crop["audio_start"] + crop["audio_end"]) / 2
        crop["audio_start"] = max(0, min(crop["audio_start"], center - 2.5))
        crop["audio_end"] = min(duration, max(crop["audio_end"], crop["audio_start"] + 5))
        crop["audio_start"] = max(0, min(crop["audio_start"], crop["audio_end"] - 5))
        missing.append(crop)
    return missing


def main():
    import soundfile as sf
    import mlx.core as mx
    from train_early_speaker_identity import session_tracks
    from transcriber.parakeet_backend import load_model, transcribe_file

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "dev", "test"], required=True)
    args = p.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    model = load_model("parakeet", compute_type="float32")
    # Bound the MLX cache when processing thousands of distinct crop lengths.
    mx.set_cache_limit(2 * 1024**3)
    totals = dict(split=args.split, source_crops=0, words_before=0, words_after=0, files=0)
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "source.wav"
        for session, meta in sorted(manifest["sessions"].items()):
            if meta["split"] != args.split:
                continue
            waves, masks = session_tracks(meta)
            windows = (
                json.loads((args.corpus / f"windows_{session}.json").read_text())
                if args.split == "train"
                else meta["evaluation_windows"]
            )
            for index, window in enumerate(windows):
                folder = args.output / f"words_{args.split}" / f"{session}_{index:02d}"
                for name in NAMES:
                    target = folder / f"{name}.json"
                    if target.exists():
                        continue
                    original = (
                        args.corpus
                        / f"words_{args.split}"
                        / f"{session}_{index:02d}"
                        / f"{name}.json"
                    )
                    before = json.loads(original.read_text()) if original.exists() else []
                    results, crops = list(before), []
                    if name in waves:
                        start, duration = window["start"], window["duration"]
                        wave = waves[name][round(start * SR) : round((start + duration) * SR)]
                        duration = min(duration, len(wave) / SR)
                        mask = masks[name][int(start / 0.032) : int((start + duration) / 0.032)]
                        crops = missing_speech_crops(mask, duration, before)
                        for crop in crops:
                            clip = np.array(
                                wave[
                                    round(crop["audio_start"] * SR) : round(crop["audio_end"] * SR)
                                ]
                            )
                            if not len(clip):
                                continue
                            sf.write(audio, clip, SR, subtype="FLOAT")
                            results.extend(crop_words(transcribe_file(str(audio), model), crop))
                        mx.clear_cache()
                    results.sort(key=lambda s: s["start"])
                    save(target, results)
                    evidence = dict(
                        crops=crops,
                        words_before=sum(len(s.get("words", [])) for s in before),
                        words_after=sum(len(s.get("words", [])) for s in results),
                    )
                    save(folder / f"{name}.audit.json", evidence)
                    totals["source_crops"] += len(crops)
                    totals["words_before"] += evidence["words_before"]
                    totals["words_after"] += evidence["words_after"]
                    totals["files"] += 1
                print("SOURCE_REFERENCE_READY", args.split, session, index, flush=True)
    save(args.output / f"{args.split}_summary_this_run.json", totals)
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
