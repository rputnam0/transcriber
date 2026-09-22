"""Build identity-only stem mixtures with session-disjoint development controls.

Owners come from labeled stems; source ASR supplies approximate boundaries/context,
not lexical training targets. All human reviews are reserved for evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from build_moss_diarization_dataset import group_words_into_segments
from prepare_early_domain_corpus import NAMES, SR
from prepare_moss_mac_training import TRAIN, DEV, TEST, authored_record, eligible_words
from train_early_speaker_identity import session_tracks

CASES = ["clear", "short_interruption", "rapid_switches", "overlap"]


def identity_only(record):
    """Cover whitespace too, so unspecified tokens cannot acquire default loss."""
    target = record["conversation"][-1]["content"]
    meta = record["metadata"]
    speakers = [{**s, "weight": 1.0} for s in meta["loss_spans"] if s["kind"] == "speaker_tag"]
    meta["loss_spans"] = [
        dict(start=0, end=len(target), kind="context_only", weight=0.0),
        *speakers,
    ]
    meta.update(
        eos_loss_weight=0.0,
        activity_supervised=False,
        supervision_source="known_stem_owner",
        human_annotations_used=False,
    )
    return record


def load_pool(corpus, references, split):
    sessions = TRAIN if split == "train" else DEV
    manifest = json.loads((corpus / "manifest.json").read_text())
    pool, banks = [], {}
    for number in sessions:
        meta = manifest["sessions"][str(number)]
        if meta["split"] != split:
            raise ValueError("Source split mismatch")
        waves, masks = session_tracks(meta)
        banks[number] = waves
        windows = (
            json.loads((corpus / f"windows_{number}.json").read_text())
            if split == "train"
            else meta["evaluation_windows"]
        )
        for index, window in enumerate(windows):
            folder = references / f"words_{split}" / f"{number}_{index:02d}"
            transcripts = {n: json.loads((folder / f"{n}.json").read_text()) for n in NAMES}
            words = eligible_words(transcripts, masks, window["start"])
            segments = group_words_into_segments(
                words, turn_gap_seconds=0.3, maximum_segment_seconds=6
            )
            for segment in segments:
                duration = segment["end"] - segment["start"]
                if 0.2 <= duration <= 6 and len(segment["words"]) / duration <= 8:
                    pool.append(
                        dict(
                            session=number,
                            window_start=window["start"],
                            segment=segment,
                            reference_file=str(folder / f"{segment['speaker']}.json"),
                        )
                    )
    return pool, banks


def duration(item):
    return item["segment"]["end"] - item["segment"]["start"]


def schedule(case, target, context, rapid, rng, overlap_fraction, continuation=None):
    """Onsets refer to complete utterances; never truncate a word to hit a duration bin."""
    if case == "clear":
        return [(context, 0.2, 0.0)]
    if case == "short_interruption":
        # A short interjection between two context turns; A-B-A identity sequence.
        onset = 0.2 + duration(context) + rng.choice([0.04, 0.10, 0.20])
        return [
            (context, 0.2, 0.0),
            (target, onset, 0.0),
            (continuation or context, onset + duration(target) + 0.10, 0.0),
        ]
    if case == "rapid_switches":
        out, onset = [], 0.2
        for item in rapid:
            out.append((item, onset, 0.0))
            onset += duration(item) + rng.choice([0.0, 0.04, 0.10, 0.20])
        return out
    if case == "overlap":
        onset = 0.2 + duration(context) - duration(target) * overlap_fraction
        return [(context, 0.2, 0.0), (target, onset, rng.choice([-6.0, 0.0, 6.0]))]
    raise ValueError(case)


def mix(placements, banks, rms):
    seconds = max(onset + duration(item) for item, onset, _ in placements) + 0.2
    wave = np.zeros(round(seconds * SR), np.float32)
    segments, sources = [], []
    for item, onset, relative_db in placements:
        segment = item["segment"]
        begin = item["window_start"] + segment["start"]
        end = item["window_start"] + segment["end"]
        source = banks[item["session"]][segment["speaker"]]
        padding = 0.08
        lo, hi = round((begin - padding) * SR), round((end + padding) * SR)
        if lo < 0 or hi > len(source):
            raise ValueError("Source crop extends beyond labeled stem")
        clip = np.array(source[lo:hi])
        # Short fades inside real context padding, never across labeled speech.
        fade = min(160, len(clip) // 2)
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
        actual_rms = float(np.sqrt(np.mean(clip**2)))
        if actual_rms < 1e-5:
            raise ValueError("Speech reference points to silent audio")
        clip *= rms * 10 ** (relative_db / 20) / actual_rms
        offset = round((onset - padding) * SR)
        wave[offset : offset + len(clip)] += clip
        shift = onset - segment["start"]
        segments.append(
            {
                **segment,
                "start": onset,
                "end": onset + duration(item),
                "word_spans": [
                    {**w, "start": w["start"] + shift, "end": w["end"] + shift}
                    for w in segment["word_spans"]
                ],
            }
        )
        sources.append(
            dict(
                session=item["session"],
                speaker=segment["speaker"],
                source_start=begin,
                source_end=end,
                onset=onset,
                relative_db=relative_db,
                reference_file=item["reference_file"],
            )
        )
    wave /= max(1.0, float(np.max(np.abs(wave))) / 0.98)
    return wave, segments, sources


def build_split(args, split, count):
    pool, banks = load_pool(args.corpus, args.references, split)
    rng = random.Random(args.seed + (split == "dev"))
    short = {
        n: [p for p in pool if p["segment"]["speaker"] == n and duration(p) <= 1] for n in NAMES
    }
    long = {
        n: [p for p in pool if p["segment"]["speaker"] == n and duration(p) >= 2.5] for n in NAMES
    }
    if any(not short[n] or not long[n] for n in NAMES):
        raise ValueError("Need short and clear reference speech for every identity")
    root = args.output / split
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        case = CASES[i % 4]
        # Balanced rotating identities, plus extra Jesse/Schmitty examples in each case.
        block = i // 4
        name = NAMES[block % 6]
        if split == "train" and block % 4 == 3:
            name = rng.choice(NAMES[:2])
        target_bin = [0.25, 0.5, 0.75, 1.0][(i // 24) % 4]
        nearest = sorted(short[name], key=lambda p: abs(duration(p) - target_bin))[:50]
        target = rng.choice(nearest)
        other = rng.choice([n for n in NAMES if n != name])
        context = rng.choice(long[other] if case != "clear" else long[name])
        rapid = [
            target,
            rng.choice(short[other]),
            rng.choice(short[name]),
            rng.choice(short[other]),
        ]
        fraction = [0.25, 0.5, 0.75, 1.0][(i // 4) % 4]
        placements = schedule(case, target, context, rapid, rng, fraction, rng.choice(long[other]))
        wave, segments, sources = mix(placements, banks, 10 ** (rng.uniform(-26, -18) / 20))
        audio = root / f"{case}_{i:04d}.wav"
        sf.write(audio, wave, SR, subtype="FLOAT")
        record = identity_only(
            authored_record(audio.stem, audio, segments, str(sources[0]["session"]), "identity")
        )
        record["metadata"].update(
            case=case,
            split=split,
            source_sessions=sorted({s["session"] for s in sources}),
            sources=sources,
            audio_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            target_duration=duration(target) if case != "clear" else None,
            requested_duration_bin=target_bin if case != "clear" else None,
            overlap_fraction=fraction if case == "overlap" else 0,
        )
        records.append(record)
    path = args.output / f"{split}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return dict(
        samples=len(records),
        cases=dict(Counter(r["metadata"]["case"] for r in records)),
        sources=sorted({s for r in records for s in r["metadata"]["source_sessions"]}),
        short_pool={n: len(v) for n, v in short.items()},
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=384)
    p.add_argument("--dev-count", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260921)
    args = p.parse_args()
    if min(args.train_count, args.dev_count) < 24:
        p.error("Use at least 24 examples per split for all speakers and cases")
    if (args.output / "summary.json").exists():
        raise ValueError("Experiment already built; choose a fresh output to preserve provenance")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        split: build_split(args, split, count)
        for split, count in [("train", args.train_count), ("dev", args.dev_count)]
    }
    summary.update(
        seed=args.seed,
        final_test_sessions_untouched=TEST,
        human_training_targets=0,
        unsure_training_targets=0,
        loss="speaker tags only; words, timestamps and EOS masked",
        limitations="Source ASR boundaries are automatic. Stem ownership is known; cross-talk and boundary errors can remain. No pitch shifting or assumed microphone/noise simulation.",
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
