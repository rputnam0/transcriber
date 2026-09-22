#!/usr/bin/env python3
"""Test cross-session voice examples prepended to public MOSS audio.

Enrollment comes only from training tracks. Map anonymous IDs by their overlap with
known enrollment intervals, then remove the enrollment audio from target predictions.
No target reference enters preparation, inference, or name binding.
"""
import argparse
import hashlib
import json
from pathlib import Path

from test_moss_mac import save, MODEL


def prepare(args):
    import numpy as np
    import soundfile as sf
    from prepare_early_domain_corpus import NAMES, SR, speech_mask

    corpus = json.loads((args.corpus / "manifest.json").read_text())
    rows = [json.loads(line) for line in args.training_manifest.read_text().splitlines()]
    profiles, pieces, cursor = [], [], 0
    for name in NAMES:
        candidates = []
        for row in rows:
            meta = row["metadata"]
            if not meta["cut_id"].startswith("real_"):
                continue
            session = meta["session"]
            if corpus["sessions"][session]["split"] != "train":
                raise ValueError("Enrollment must come only from training sessions")
            for segment in meta["reference_segments"]:
                duration = segment["end"] - segment["start"]
                if segment["speaker"] == name and 3.0 <= duration <= 3.8:
                    candidates.append((session != "61", abs(duration - 3.5), meta, segment))
        candidates.sort(key=lambda x: x[:2])
        for _, _, meta, segment in candidates:
            session = meta["session"]
            index = int(meta["cut_id"].split("_")[-1])
            start = (
                json.loads((args.corpus / f"windows_{session}.json").read_text())[index]["start"]
                + segment["start"]
            )
            track = corpus["sessions"][session]["tracks"][name]
            duration = segment["end"] - segment["start"]
            mask = speech_mask(np.load(track["vad"]))[
                int(start / 0.032) : int((start + duration) / 0.032)
            ]
            if not len(mask) or mask.mean() < 0.75:
                continue
            source = np.memmap(track["pcm"], np.float32, mode="r")
            clip = np.array(
                source[round((start - 0.12) * SR) : round((start + duration + 0.12) * SR)]
            )
            clip *= 0.08 / max(float(np.sqrt(np.mean(clip**2))), 1e-8)
            clip /= max(1.0, float(np.abs(clip).max()) / 0.98)
            profiles.append(
                dict(
                    speaker=name,
                    source_session=session,
                    source_start=start,
                    start=cursor / SR + 0.12,
                    end=(cursor + len(clip)) / SR - 0.12,
                    sha256=hashlib.sha256(clip.tobytes()).hexdigest(),
                )
            )
            pieces.extend([clip, np.zeros(round(0.35 * SR), np.float32)])
            cursor += len(clip) + round(0.35 * SR)
            break
        else:
            raise ValueError(f"No clean training enrollment for {name}")
    pieces.append(np.zeros(2 * SR, np.float32))
    prefix = np.concatenate(pieces)
    original = json.loads(args.selector.read_text())
    records = []
    for row in original:
        target, rate = sf.read(row["audio"], dtype="float32")
        assert rate == SR and hashlib.sha256(target.tobytes()).hexdigest() == row["sha256"]
        wave = np.r_[prefix, target]
        audio = args.output / "audio" / f"{row['cut_id']}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio, wave, SR, subtype="FLOAT")
        records.append(
            dict(
                row,
                audio=str(audio.resolve()),
                duration=len(wave) / SR,
                sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                target_record=row,
            )
        )
    save(args.output / "manifest.json", records)
    save(
        args.output / "enrollment.json",
        dict(
            profiles=profiles,
            prefix_seconds=len(prefix) / SR,
            prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
        ),
    )
    print(
        f"Prepared {len(records)} target clips with {len(prefix)/SR:.2f}s of training-only enrollment"
    )


def finalize(args):
    enrollment = json.loads((args.raw / "enrollment.json").read_text())
    raw_records = json.loads((args.raw / "manifest.json").read_text())
    if args.limit:
        raw_records = raw_records[: args.limit]
    records = []
    for record in raw_records:
        original = record["target_record"]
        prediction = json.loads((args.raw / "predictions" / f"{record['cut_id']}.json").read_text())
        mapping, evidence = {}, {}
        turns = prediction["segments"]
        for label in {t["speaker"] for t in turns}:
            votes = {}
            for profile in enrollment["profiles"]:
                votes[profile["speaker"]] = sum(
                    max(0, min(t["end"], profile["end"]) - max(t["start"], profile["start"]))
                    for t in turns
                    if t["speaker"] == label
                )
            name = max(votes, key=votes.get)
            share = votes[name] / max(sum(votes.values()), 1e-8)
            if votes[name] >= 1.2 and share >= 0.65:
                mapping[label] = name
            evidence[label] = dict(votes=votes, winning_fraction=share, accepted=label in mapping)
        shift = enrollment["prefix_seconds"]
        kept = [
            dict(
                t, start=max(0, t["start"] - shift), end=min(original["duration"], t["end"] - shift)
            )
            for t in turns
            if t["start"] >= shift - 0.3 and t["end"] > shift
        ]
        result = dict(prediction, **{k: v for k, v in original.items() if k not in prediction})
        result.update(original)
        result.update(
            model=MODEL + "+enrollment-prefix",
            base_model=MODEL,
            segments=kept,
            enrollment_mapping=mapping,
            enrollment_evidence=evidence,
            inference_audio_sha256=prediction["sha256"],
            prefix_seconds=shift,
            crossed_boundary_turns=sum(
                t["start"] < shift - 0.3 and t["end"] > shift for t in turns
            ),
        )
        save(args.output / "predictions" / f"{record['cut_id']}.json", result)
        records.append(original)
    save(args.output / "manifest.json", records)
    save(args.output / "enrollment.json", enrollment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "finalize"])
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    {"prepare": prepare, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
