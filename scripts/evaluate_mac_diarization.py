#!/usr/bin/env python3
"""Cross-session speaker enrollment and a mono-only Mac diarization smoke evaluation.

The isolated evaluation tracks are used ONLY for an explicitly labelled energy proxy.
This is not a human-labelled DER/WER benchmark. Audio remains in the output directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from transcriber.diarization import (
    DiarizationResult,
    DiarizationTurn,
    diarize_audio,
    extract_embeddings_for_segments,
)
from transcriber.speaker_bank import SpeakerBank
from transcriber.transcript_pipeline import (
    _aggregate_speaker_embeddings,
    _assign_word_speakers,
    asr_result_from_segments,
)

SR = 16000


def decode(path, start, duration):
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(duration),
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "-",
        ]
    )
    wave = np.frombuffer(raw, np.float32)
    return np.pad(wave, (0, max(0, int(duration * SR) - len(wave))))[: int(duration * SR)]


def tracks(archive, start, duration):
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory() as temp:
        for member in sorted(source.namelist()):
            if not member.lower().endswith(".ogg"):
                continue
            path = Path(temp) / Path(member).name
            with source.open(member) as src, path.open("wb") as dst:
                import shutil

                shutil.copyfileobj(src, dst)
            handle = path.stem.split("-", 1)[1]
            yield handle, decode(path, start, duration)


def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def speech_enrollment(args, selected, token):
    """Remove gated silence from known single-speaker enrollment, never from eval sources."""
    cached = args.output / "enrollment61_speech.npz"
    bank_root = args.output / "speech_speaker_bank"
    if cached.exists() and (bank_root / "session61" / "bank.json").exists():
        data = np.load(cached)
        return data["names"].tolist(), data["vectors"]
    bank = SpeakerBank(bank_root, profile="session61", scoring_whiten=False)
    if not bank.is_empty:
        raise RuntimeError("Incomplete speech enrollment; use a fresh output directory.")
    names, profiles, stats = [], [], {}
    for name, wave in tracks(args.enrollment_zip, 120, 1200):
        clips = np.concatenate(
            [
                wave[int((item["start"] - 120) * SR) : int((item["end"] - 120) * SR)]
                for item in selected[name]
            ]
        )
        frames = clips.reshape(-1, 320)
        active = np.sqrt(np.mean(frames**2, axis=1)) > 0.005
        occupancy = float(active.mean())
        active = np.convolve(active.astype(int), np.ones(5), mode="same") > 0
        speech = frames[active].reshape(-1)
        count = len(speech) // (3 * SR)
        if count < 3:
            raise RuntimeError(f"Insufficient active enrollment speech for {name}")
        payload = [(3 * i, 3 * (i + 1), name) for i in range(count)]
        embeddings, _ = extract_embeddings_for_segments(
            "enrollment",
            payload,
            token,
            force_device="cpu",
            pre_pad=0,
            post_pad=0,
            audio_waveform=speech,
            audio_sample_rate=SR,
        )
        for entry in embeddings:
            bank.add_embedding(
                name,
                entry.embedding,
                source=args.enrollment_zip.name,
                extra={"speech_only": True, "chunk_seconds": 3},
            )
        profiles.append(unit(np.mean([entry.embedding for entry in embeddings], axis=0)))
        names.append(name)
        stats[name] = {
            "raw_seconds": 60,
            "raw_active_fraction": occupancy,
            "embedded_speech_seconds": 3 * count,
        }
    bank.save()
    np.savez(cached, names=names, vectors=profiles)
    (args.output / "speech_enrollment_stats.json").write_text(json.dumps(stats, indent=2))
    return names, np.stack(profiles)


def bind(embeddings, names, profiles):
    result = {}
    for label, vector in embeddings.items():
        scores = profiles @ unit(np.asarray(vector))
        order = np.argsort(scores)[::-1]
        result[label] = {
            "speaker": names[order[0]],
            "cosine": float(scores[order[0]]),
            "margin": float(scores[order[0]] - scores[order[1]]),
            "scores": dict(zip(names, scores.tolist())),
        }
    return result


def activity(turns, labels, duration):
    times = np.arange(int(duration * 50)) / 50 + 0.01
    result = np.zeros((len(labels), len(times)), bool)
    for turn in turns:
        result[labels.index(turn.speaker)] |= (times >= turn.start) & (times < turn.end)
    return result


def energy_proxy(waves, names, diarization, bindings, duration):
    """Threshold sensitivity on dominant, singly active source-track frames."""
    labels = sorted({turn.speaker for turn in diarization.segments})
    predicted = activity(diarization.segments, labels, duration)
    rms = np.sqrt(np.mean(waves.reshape(len(names), -1, 320) ** 2, axis=2))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    output = []
    for threshold in [-50, -45, -40, -35]:
        reference = db > threshold
        single = reference.sum(axis=0) == 1
        # Exclude boundary frames and require a 10 dB dominance margin.
        ordered = np.sort(db, axis=0)
        single &= ordered[-1] - ordered[-2] >= 10
        for shift in [-2, -1, 1, 2]:
            single &= np.roll(reference.argmax(axis=0), shift) == reference.argmax(axis=0)
        single[:2] = single[-2:] = False
        named = np.zeros_like(reference)
        for index, label in enumerate(labels):
            if label in bindings:
                named[names.index(bindings[label]["speaker"])] |= predicted[index]
        correct = (named == reference).all(axis=0)
        covered = single & named.any(axis=0)
        overlap = reference.sum(axis=0) > 1
        output.append(
            {
                "threshold_dbfs": threshold,
                "eligible_single_source_seconds": float(single.sum() / 50),
                "energy_overlap_seconds": float(overlap.sum() / 50),
                "exact_named_activity_on_single_source": float(correct[single].mean()),
                "single_source_coverage": float(covered.sum() / max(single.sum(), 1)),
                "per_speaker": {
                    name: {
                        "seconds": float((single & reference[i]).sum() / 50),
                        "exact_accuracy": (
                            float(correct[single & reference[i]].mean())
                            if (single & reference[i]).any()
                            else None
                        ),
                    }
                    for i, name in enumerate(names)
                },
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrollment-zip", type=Path, required=True)
    parser.add_argument("--evaluation-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=1418.11)
    parser.add_argument("--duration", type=float, default=300)
    args = parser.parse_args()
    if args.enrollment_zip.resolve() == args.evaluation_zip.resolve():
        parser.error("Enrollment and evaluation must use different sessions.")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from huggingface_hub import get_token

    torch.set_num_threads(6)
    token = get_token()
    started = time.monotonic()
    enrollment = args.output / "enrollment61.npz"
    selected = {}
    if enrollment.exists():
        data = np.load(enrollment)
        names = data["names"].tolist()
        vectors = data["vectors"]
        selected = json.loads((args.output / "enrollment_windows.json").read_text())
    else:
        names, vectors = [], []
        bank = SpeakerBank(args.output / "speaker_bank", profile="session61", scoring_whiten=False)
        if not bank.is_empty:
            raise RuntimeError(
                "Incomplete enrollment: use a new output directory to avoid duplicates."
            )
        for name, wave in tracks(args.enrollment_zip, 120, 1200):
            windows = wave.reshape(-1, 5 * SR)
            rms = np.sqrt(np.mean(windows**2, axis=1))
            eligible = np.flatnonzero(rms >= max(0.005, np.percentile(rms, 75)))
            if len(eligible) < 12:
                raise RuntimeError(f"Too little enrollment activity for {name}")
            indices = eligible[np.linspace(0, len(eligible) - 1, 12).astype(int)]
            payload = [(float(i * 5), float(i * 5 + 5), name) for i in indices]
            embeddings, _ = extract_embeddings_for_segments(
                "enrollment",
                payload,
                token,
                force_device="cpu",
                pre_pad=0,
                post_pad=0,
                audio_waveform=wave,
                audio_sample_rate=SR,
            )
            if len(embeddings) != 12:
                raise RuntimeError(f"Incomplete embeddings for {name}")
            for entry in embeddings:
                bank.add_embedding(name, entry.embedding, source=args.enrollment_zip.name)
            names.append(name)
            vectors.append(unit(np.mean([entry.embedding for entry in embeddings], axis=0)))
            selected[name] = [{"start": 120 + a, "end": 120 + b} for a, b, _ in payload]
            print("ENROLLED", name, len(embeddings), flush=True)
        vectors = np.stack(vectors)
        np.savez(enrollment, names=names, vectors=vectors)
        (args.output / "enrollment_windows.json").write_text(json.dumps(selected, indent=2))
        bank.save()
    mixture = args.output / "probe62.wav"
    reference_path = args.output / "probe62_references.npz"
    if mixture.exists() and reference_path.exists():
        waves = np.load(reference_path)["waves"]
        # The existing baseline is sorted by Session62 archive member name.
        with zipfile.ZipFile(args.evaluation_zip) as archive:
            eval_names = [
                Path(p).stem.split("-", 1)[1]
                for p in sorted(archive.namelist())
                if p.endswith(".ogg")
            ]
    else:
        extracted = list(tracks(args.evaluation_zip, args.start, args.duration))
        eval_names = [name for name, _ in extracted]
        waves = np.stack([wave for _, wave in extracted]) * 0.4
        sf.write(mixture, waves.sum(axis=0), SR, subtype="FLOAT")
        np.savez(reference_path, waves=waves)
    baseline_path = args.output / "baseline62.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        diarization = DiarizationResult(
            [DiarizationTurn(**t) for t in baseline["segments"]],
            [DiarizationTurn(**t) for t in baseline["exclusive_segments"]],
            baseline["metadata"],
        )
    else:
        diarization = diarize_audio(str(mixture), hf_token=token, max_speakers=6, device="cpu")
        baseline = {}
    embeddings = _aggregate_speaker_embeddings(
        str(mixture),
        diarization,
        hf_token=token,
        diarization_model_name=None,
        force_device="cpu",
        quiet=True,
    )
    original = bind(baseline.get("baseline_embeddings", {}), names, vectors)
    raw_enrollment_bindings = bind(embeddings, names, vectors)
    names, vectors = speech_enrollment(args, selected, token)
    bindings = bind(embeddings, names, vectors)
    baseline_clean_enrollment = bind(baseline.get("baseline_embeddings", {}), names, vectors)
    asr_path = args.output / "asr62.json"
    if asr_path.exists():
        asr = json.loads(asr_path.read_text())
    else:
        from transcriber.parakeet_backend import load_model, transcribe_file

        asr = transcribe_file(str(mixture), load_model("parakeet", compute_type="float32"))
        asr_path.write_text(json.dumps(asr, indent=2))
    segments = _assign_word_speakers(asr_result_from_segments(asr), diarization)
    for segment in segments:
        raw = segment.get("speaker")
        match = bindings.get(raw)
        segment["speaker_raw"] = raw
        # Prespecified guards, not thresholds fitted to the evaluation references.
        segment["speaker"] = (
            match["speaker"] if match and match["cosine"] >= 0.5 and match["margin"] >= 0.1 else raw
        )
    (args.output / "named62.json").write_text(json.dumps(segments, indent=2))
    (args.output / "named62.txt").write_text(
        "\n".join(f"[{s['start'] + args.start:8.2f}] {s['speaker']}: {s['text']}" for s in segments)
        + "\n"
    )
    results = {
        "enrollment_session": args.enrollment_zip.name,
        "evaluation_session": args.evaluation_zip.name,
        "evaluation_start_seconds": args.start,
        "evaluation_duration_seconds": args.duration,
        "reference_type": "isolated-source energy proxy; not human-labelled speech or DER/WER",
        "raw_enrollment_seconds_per_speaker": 60,
        "new_bindings": bindings,
        "original_bindings": original,
        "new_embeddings_raw_enrollment_bindings": raw_enrollment_bindings,
        "old_embeddings_speech_enrollment_bindings": baseline_clean_enrollment,
        "speech_profile_cosines": (vectors @ vectors.T).tolist(),
        "speech_profile_names": names,
        "new_energy_proxy": energy_proxy(waves, eval_names, diarization, bindings, args.duration),
        "guarded_energy_proxy": energy_proxy(
            waves,
            eval_names,
            diarization,
            {k: v for k, v in bindings.items() if v["cosine"] >= 0.5 and v["margin"] >= 0.1},
            args.duration,
        ),
        "original_energy_proxy": (
            energy_proxy(waves, eval_names, diarization, original, args.duration)
            if original
            else None
        ),
        "wall_seconds_this_run": time.monotonic() - started,
        "asr_words": sum(len(s.get("words", [])) for s in asr),
        "output_segments": len(segments),
    }
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print("EVALUATION_READY", args.output / "evaluation.json", flush=True)


if __name__ == "__main__":
    main()
