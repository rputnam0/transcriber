"""Prepare six energy-stratified minute windows per isolated source track.

--tracks is private JSON: [{"audio": "/path/to/stem.ogg", "speaker": "Display Name"}].
Optional legacy TXT parsing/alignment supports the Character Name HH:MM:SS format.
Run in requirements/mac-asr.lock.txt's environment. Outputs contain private audio/text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import soundfile as sf


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(tracks, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    prepared = []
    for index, track in enumerate(tracks, 1):
        pcm = output / f"track{index}.f32"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                track["audio"],
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "f32le",
                str(pcm),
            ],
            check=True,
        )
        wave = np.memmap(pcm, np.float32, mode="r")
        prepared.append(dict(track, pcm=str(pcm.resolve()), duration=len(wave) / 16000))
        candidates = []
        for start in range(0, int(len(wave) / 16000) - 59, 60):
            clip = wave[start * 16000 : (start + 60) * 16000]
            rms = np.sqrt(np.mean(clip.reshape(-1, 1600) ** 2, axis=1))
            db = 20 * np.log10(np.maximum(rms, 1e-10))
            active = db[db > -55]
            candidates.append(
                dict(
                    start=start,
                    peak_db=float(max(db)),
                    digital_silence=not bool(np.any(clip)),
                    speech_db=float(np.median(active)) if len(active) else -200,
                    active_fraction=float(np.mean(db > -55)),
                )
            )
        silence = sorted(candidates, key=lambda c: (c["peak_db"], c["active_fraction"]))[:2]
        speech = [c for c in candidates if c["active_fraction"] >= 0.15 and c not in silence]
        quiet = sorted(speech, key=lambda c: c["speech_db"])[:2]
        loud = sorted(
            [c for c in speech if c not in quiet], key=lambda c: c["speech_db"], reverse=True
        )[:2]
        if len(silence) + len(quiet) + len(loud) != 6:
            raise ValueError("Track has insufficient speech for six stratified windows")
        for stratum, rows in [("silence", silence), ("quiet", quiet), ("loud", loud)]:
            for row in rows:
                start = row["start"]
                clip = np.asarray(wave[start * 16000 : (start + 60) * 16000])
                cut_id = f"track{index}_{start:05d}"
                audio = output / "samples" / f"{cut_id}.wav"
                audio.parent.mkdir(exist_ok=True)
                sf.write(audio, clip, 16000, subtype="FLOAT")
                records.append(
                    dict(
                        row,
                        cut_id=cut_id,
                        audio=str(audio.resolve()),
                        duration=60,
                        sha256=hashlib.sha256(clip.tobytes()).hexdigest(),
                        speaker=track["speaker"],
                        stratum=stratum,
                    )
                )
    save(output / "manifest.json", records)
    save(output / "tracks.json", prepared)


def parse_legacy(path):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"(.*?) (\d\d):(\d\d):(\d\d) (.*)", line)
        if not match:
            raise ValueError("Legacy transcript must use Name HH:MM:SS text")
        speaker, hour, minute, second, text = match.groups()
        result.append(
            dict(
                speaker=speaker.replace("Kaladin Shash", "Kaladen Shash"),
                start=int(hour) * 3600 + int(minute) * 60 + int(second),
                text=text,
            )
        )
    return result


def align_legacy(output, legacy, models):
    from mlx_audio.stt import load

    aligner = load(models["mlx-community/Qwen3-ForcedAligner-0.6B-8bit"]["path"])
    manifest = json.loads((output / "manifest.json").read_text())
    result = []
    for track in json.loads((output / "tracks.json").read_text()):
        wave = np.memmap(track["pcm"], np.float32, mode="r")
        rows = [r for r in legacy if r["speaker"] == track["speaker"]]
        samples = [r for r in manifest if r["speaker"] == track["speaker"]]
        for index, row in enumerate(rows):
            end = min(
                track["duration"],
                row["start"] + 60,
                next(
                    (r["start"] for r in rows[index + 1 :] if r["start"] > row["start"]),
                    track["duration"],
                ),
            )
            if not any(row["start"] < s["start"] + 60 and end > s["start"] for s in samples):
                continue
            if end <= row["start"]:
                end = row["start"] + 1
            clip = np.asarray(wave[round(row["start"] * 16000) : round(end * 16000)])
            aligned = aligner.generate(audio=clip, text=row["text"], language="English").segments
            result.extend(
                dict(
                    speaker=track["speaker"],
                    word=w["text"],
                    start=row["start"] + w["start"],
                    end=row["start"] + w["end"],
                )
                for w in aligned
            )
    save(output / "old_words.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, help="Private track/speaker input JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--models", type=Path, help="Pinned model manifest for legacy alignment")
    args = parser.parse_args()
    if args.tracks:
        prepare(json.loads(args.tracks.read_text()), args.output)
    if args.legacy:
        legacy = parse_legacy(args.legacy)
        save(args.output / "old.json", legacy)
        if args.models:
            align_legacy(args.output, legacy, json.loads(args.models.read_text()))


if __name__ == "__main__":
    main()
