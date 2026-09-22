#!/usr/bin/env python3
"""Build session-disjoint real and synthetic MOSS training audio on the Mac.

The track owner is a known label. Word text/timing is automatic Parakeet supervision,
filtered with source VAD, not a human transcript. All model inputs are mono mixtures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

import numpy as np
import soundfile as sf

from prepare_early_domain_corpus import NAMES, SR, decode, speech_mask, vad_probabilities
from test_moss_mac import save

TRAIN = [43, 45, 47, 55, 61]
DEV = [39, 62]
TEST = [37, 56]


def training_duration(meta):
    durations = [t["duration"] for t in meta["tracks"].values()]
    # A participant can leave early. Do not truncate the whole recording to that track.
    return max(durations) if min(durations) < 0.95 * max(durations) else min(durations)


def mono_window(waves, start, duration):
    result = np.zeros(round(duration * SR), np.float32)
    for wave in waves.values():
        clip = np.asarray(wave[round(start * SR) : round((start + duration) * SR)])
        result[: min(len(clip), len(result))] += clip[: len(result)]
    return result


def ingest_session(number, args):
    root = args.output / f"session{number}"
    root.mkdir(parents=True, exist_ok=True)
    source = args.archives / f"session{number}.zip"
    metadata = dict(
        split="train" if number in TRAIN else "dev" if number in DEV else "test",
        tracks={},
        evaluation_windows=[],
    )
    with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory() as tmp:
        for member in sorted(archive.namelist()):
            if Path(member).suffix.lower() not in {".ogg", ".flac", ".wav"}:
                continue
            name = Path(member).stem.split("-", 1)[1]
            if name not in NAMES:
                raise ValueError(f"Unrecognized source owner {name}")
            pcm = root / f"{name}.f32"
            if not pcm.exists():
                extracted = Path(tmp) / Path(member).name
                with archive.open(member) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                decode(extracted, pcm)
                extracted.unlink()
            wave = np.memmap(pcm, np.float32, mode="r")
            vad = root / f"{name}.vad.npy"
            if not vad.exists():
                np.save(vad, vad_probabilities(wave, args.vad_model))
            metadata["tracks"][name] = dict(
                pcm=str(pcm.resolve()),
                vad=str(vad.resolve()),
                duration=len(wave) / SR,
                speech_seconds=float(speech_mask(np.load(vad)).sum() * 0.032),
            )
            print("INGEST", number, name, flush=True)
    if not metadata["tracks"]:
        raise ValueError(f"No labeled audio tracks in {source.name}")
    duration = min(t["duration"] for t in metadata["tracks"].values())
    if number not in TRAIN:
        metadata["evaluation_windows"] = [
            dict(start=float(s), duration=60.0)
            for s in np.linspace(120, duration - 180, 12).round(2)
        ]
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata["archive_sha256"] = digest.hexdigest()
    save(root / "metadata.json", metadata)
    return str(number), metadata


def ingest(args):
    original = json.loads((args.original / "manifest.json").read_text())
    manifest = dict(
        sample_rate=SR,
        speaker_names=NAMES,
        sessions={},
        split_rule=dict(train=TRAIN, dev=DEV, test=TEST),
        reference_type="known track owners; automatic source VAD and ASR",
    )
    for number in [55, 61, 62]:
        manifest["sessions"][str(number)] = original["sessions"][str(number)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = [
            executor.submit(ingest_session, n, args)
            for n in TRAIN + DEV + TEST
            if str(n) not in manifest["sessions"]
        ]
        for future in as_completed(jobs):
            key, metadata = future.result()
            manifest["sessions"][key] = metadata
            save(args.output / "manifest.partial.json", manifest)
    save(args.output / "manifest.json", manifest)


def transcribe(args):
    from train_early_speaker_identity import session_tracks
    from transcriber.parakeet_backend import load_model, transcribe_file

    manifest_path = args.output / "manifest.json"
    if args.allow_partial and not manifest_path.exists():
        manifest_path = args.output / "manifest.partial.json"
    manifest = json.loads(manifest_path.read_text())
    model = load_model("parakeet", compute_type="float32")
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "source.wav"
        for number, meta in sorted(manifest["sessions"].items()):
            if meta["split"] != args.split:
                continue
            waves, masks = session_tracks(meta)
            if args.split == "train":
                duration = training_duration(meta)
                if duration < 240:
                    raise ValueError(f"Training session {number} is unexpectedly short")
                windows = [
                    dict(start=float(s), duration=30.0)
                    for s in np.linspace(90, duration - 120, 96).round(2)
                ]
            else:
                windows = meta["evaluation_windows"]
            save(args.output / f"windows_{number}.json", windows)
            for index, window in enumerate(windows):
                folder = args.output / f"words_{args.split}" / f"{number}_{index:02d}"
                folder.mkdir(parents=True, exist_ok=True)
                start, duration = window["start"], window["duration"]
                lo, hi = int(start * SR), int((start + duration) * SR)
                for name in NAMES:
                    path = folder / f"{name}.json"
                    if path.exists():
                        continue
                    if name not in waves:
                        save(path, [])
                        continue
                    clip = np.array(waves[name][lo:hi])
                    mask = masks[name][int(start / 0.032) : int((start + duration) / 0.032)]
                    if mask.sum() * 0.032 < 0.15 or np.sqrt(np.mean(clip**2)) < 1e-5:
                        save(path, [])
                        continue
                    sf.write(audio, clip, SR, subtype="FLOAT")
                    save(path, transcribe_file(str(audio), model))
                print("TRANSCRIBED", args.split, number, index, flush=True)


def eligible_words(transcripts, masks, start):
    from evaluate_named_words import normalized

    words = []
    for name, segments in transcripts.items():
        for segment in segments:
            for word in segment.get("words", []):
                s, e = float(word["start"]), float(word["end"])
                center = start + (s + e) / 2
                frame = min(int(center / 0.032), len(masks[name]) - 1)
                if 0.02 <= e - s <= 1.8 and masks[name][frame] and normalized(word["word"]):
                    words.append(dict(start=s, end=e, text=word["word"].strip(), speaker=name))
    return words


def authored_record(cut, audio, segments, session, mode="onset"):
    from build_moss_brief_overlap_crops import render_weighted_target
    from build_moss_diarization_dataset import DEFAULT_PROMPT
    from moss_identity_prompt import identity_prompt

    onsets = {}
    for segment in segments:
        name = segment["speaker"]
        onsets[name] = min(onsets.get(name, float("inf")), segment["start"])
    speaker_ids = {n: f"S{i+1:02d}" for i, n in enumerate(sorted(onsets, key=onsets.get))}
    if mode == "identity":
        speaker_ids = {n: f"S{NAMES.index(n)+1:02d}" for n in onsets}
    target, weights, activity = render_weighted_target(segments, speaker_ids)
    return dict(
        conversation=[
            dict(
                role="user",
                message_type="text",
                content=identity_prompt(NAMES) if mode == "identity" else DEFAULT_PROMPT,
            ),
            dict(role="user", message_type="audio", content=str(audio.resolve())),
            dict(role="assistant", message_type="text", content=target),
        ],
        metadata=dict(
            cut_id=cut,
            session=session,
            mono_input_only=True,
            stable_session_speaker_ids=speaker_ids,
            loss_spans=weights,
            activity=activity,
            activity_supervised=True,
            reference_segments=segments,
            naming_mode=mode,
        ),
    )


def build(args):
    from build_moss_diarization_dataset import group_words_into_segments
    from train_early_speaker_identity import session_tracks

    manifest = json.loads((args.output / "manifest.json").read_text())
    tag = f"_{args.dataset_tag}" if args.dataset_tag else ""
    audio_root = args.output / f"training_audio{tag}"
    audio_root.mkdir(exist_ok=True)
    records, pool = [], []
    wave_bank = {}
    for number in TRAIN:
        meta = manifest["sessions"][str(number)]
        assert meta["split"] == "train" and number not in DEV + TEST
        waves, masks = session_tracks(meta)
        wave_bank[number] = waves
        windows = json.loads((args.output / f"windows_{number}.json").read_text())
        for index, window in enumerate(windows):
            folder = (args.reference_root or args.output) / "words_train" / f"{number}_{index:02d}"
            transcripts = {n: json.loads((folder / f"{n}.json").read_text()) for n in NAMES}
            start = window["start"]
            words = eligible_words(transcripts, masks, start)
            segments = group_words_into_segments(
                words, turn_gap_seconds=0.3, maximum_segment_seconds=6
            )
            # Exclude implausibly fast pseudo-labels and empty crops.
            segments = [
                s for s in segments if len(s["words"]) / max(s["end"] - s["start"], 0.1) <= 8
            ]
            if not segments:
                continue
            wave = mono_window(waves, start, 30)
            wave /= max(1.0, float(np.max(np.abs(wave))) / 0.98)
            audio = audio_root / f"real_{number}_{index:02d}.wav"
            sf.write(audio, wave, SR, subtype="FLOAT")
            records.append(authored_record(audio.stem, audio, segments, str(number), args.naming))
            for segment in segments:
                duration = segment["end"] - segment["start"]
                if 0.2 <= duration <= 6:
                    pool.append(dict(segment=segment, session=number, window_start=start))
    rng = random.Random(20260919)
    brief_by_name = {
        n: [
            u
            for u in pool
            if u["segment"]["speaker"] == n and u["segment"]["end"] - u["segment"]["start"] <= 1.8
        ]
        for n in NAMES
    }
    long_pool = [u for u in pool if u["segment"]["end"] - u["segment"]["start"] >= 2.5]
    gains = []
    for index in range(768):
        name = NAMES[index % len(NAMES)]
        short = rng.choice(brief_by_name[name])
        main = rng.choice([u for u in long_pool if u["segment"]["speaker"] != name])
        relative = rng.uniform(-12, 3)
        gain = 10 ** (rng.uniform(-24, -18) / 20)
        main_onset = 0.3
        short_duration = short["segment"]["end"] - short["segment"]["start"]
        main_duration = main["segment"]["end"] - main["segment"]["start"]
        short_onset = rng.uniform(main_onset + 0.15, main_onset + main_duration - 0.1)
        short_onset = min(short_onset, 7.7 - short_duration)
        mixture = np.zeros(8 * SR, np.float32)
        segments = []
        for item, onset, rms in [
            (main, main_onset, gain),
            (short, short_onset, gain * 10 ** (relative / 20)),
        ]:
            segment = item["segment"]
            source = wave_bank[item["session"]][segment["speaker"]]
            begin = item["window_start"] + segment["start"]
            end = item["window_start"] + segment["end"]
            padding = 0.08
            clip = np.array(source[int((begin - padding) * SR) : int((end + padding) * SR)])
            # Fade only the padding to avoid teaching hard splicing clicks.
            fade = min(160, len(clip) // 2)
            clip[:fade] *= np.linspace(0, 1, fade)
            clip[-fade:] *= np.linspace(1, 0, fade)
            clip *= rms / max(float(np.sqrt(np.mean(clip**2))), 1e-5)
            offset = int((onset - padding) * SR)
            mixture[offset : offset + len(clip)] += clip
            shift = onset - segment["start"]
            segments.append(
                dict(
                    segment,
                    start=onset,
                    end=segment["end"] + shift,
                    word_spans=[
                        dict(w, start=w["start"] + shift, end=w["end"] + shift)
                        for w in segment["word_spans"]
                    ],
                )
            )
        mixture /= max(1.0, float(np.max(np.abs(mixture))) / 0.98)
        audio = audio_root / f"synthetic_{index:04d}.wav"
        sf.write(audio, mixture, SR, subtype="FLOAT")
        record = authored_record(audio.stem, audio, segments, str(main["session"]), args.naming)
        record["metadata"].update(
            source_sessions=[main["session"], short["session"]], relative_db=relative
        )
        records.append(record)
        gains.append(relative)
    target = args.output / f"train_{args.naming}{tag}.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = dict(
        samples=len(records),
        real_samples=len(records) - 768,
        synthetic_samples=768,
        utterance_pool=len(pool),
        brief_pool={n: len(v) for n, v in brief_by_name.items()},
        train_sessions=TRAIN,
        dev_sessions=DEV,
        final_test_sessions=TEST,
        relative_interruption_db_range=[min(gains), max(gains)],
        naming=args.naming,
        word_supervision="automatic source-track Parakeet timestamps filtered with VAD",
        input_seconds=sum(sf.info(r["conversation"][1]["content"]).duration for r in records),
    )
    summary["reference_root"] = str(args.reference_root or args.output)
    save(args.output / f"training_summary_{args.naming}{tag}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["ingest", "transcribe", "build"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archives", type=Path)
    p.add_argument("--original", type=Path)
    p.add_argument("--vad-model", type=Path)
    p.add_argument("--split", choices=["train", "dev", "test"], default="train")
    p.add_argument("--naming", choices=["onset", "identity"], default="onset")
    p.add_argument("--reference-root", type=Path)
    p.add_argument("--dataset-tag", default="")
    p.add_argument(
        "--allow-partial", action="store_true", help="Transcribe already-ingested sessions"
    )
    args = p.parse_args()
    {"ingest": ingest, "transcribe": transcribe, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
