"""Fresh, resumable MOSS diarization using the specified trained checkpoint on MLX."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

import soundfile as sf
from mlx_speech_batch import greedy_batch
from moss_batch_inference import decode_issues, repair_speaker_brackets, recover_compact_segments
from moss_identity_prompt import identity_prompt
from run_mac_asr_quality import save


def parse(text, duration):
    pattern = r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\]\s*(.*?)\[(\d+(?:\.\d+)?)\]"
    text = repair_speaker_brackets(text)
    items = [
        SimpleNamespace(start=float(a), end=float(b), speaker=s, text=t.strip())
        for a, s, t, b in re.findall(pattern, text, re.DOTALL)
    ]
    items, recovered = recover_compact_segments(text, items)
    segments = [
        dict(
            start=max(0.0, p.start),
            end=min(duration, p.end),
            speaker=p.speaker,
            text=p.text.strip(),
        )
        for p in items
        if min(duration, p.end) > max(0.0, p.start) and p.text.strip()
    ]
    return sorted(segments, key=lambda x: (x["start"], x["end"], x["speaker"])), recovered


def run(args):
    import mlx.core as mx
    from mlx_audio.stt import load

    model = load(str(args.model))
    config = json.loads((args.model / "config.json").read_text())
    names = config["stable_speaker_names"]
    prompt = identity_prompt(names)
    model_hash = hashlib.sha256()
    for path in sorted(args.model.glob("*.safetensors")):
        with path.open("rb") as stream:
            while data := stream.read(8 * 1024 * 1024):
                model_hash.update(data)
    provenance = dict(
        runtime="mlx-audio",
        model=str(args.model.resolve()),
        checkpoint_sha256=model_hash.hexdigest(),
        decoder="greedy_equal_length_v1",
        max_tokens=2048,
        stable_speaker_names=names,
    )
    plan = json.loads(args.plan.read_text())
    for entry in plan["recordings"]:
        if args.sessions and entry["session"] not in args.sessions:
            continue
        target = args.output / f"session{entry['session']}"
        source = Path(entry["deployment"])
        target.mkdir(parents=True, exist_ok=True)
        for filename in ["manifest.json", "source.json"]:
            dest = target / filename
            if dest.exists() and dest.read_bytes() != (source / filename).read_bytes():
                raise ValueError("Input changed")
            shutil.copy2(source / filename, dest)
        records = json.loads((target / "manifest.json").read_text())
        if args.limit:
            records = records[: args.limit]
        pending = []
        for record in records:
            path = target / "predictions" / f"{record['cut_id']}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if (
                    previous.get("provenance") != provenance
                    or previous["sha256"] != record["sha256"]
                ):
                    raise ValueError("Prediction cache mismatch")
                continue
            pending.append(record)
        while pending:
            batch = [pending.pop(0)]
            while (
                pending
                and len(batch) < args.batch_size
                and pending[0]["duration"] == batch[0]["duration"]
            ):
                batch.append(pending.pop(0))
            waves = []
            for r in batch:
                wave, rate = sf.read(r["audio"], dtype="float32")
                if rate != 16000 or hashlib.sha256(wave.tobytes()).hexdigest() != r["sha256"]:
                    raise ValueError("Audio mismatch")
                waves.append(wave)
            started = time.monotonic()
            texts, complete = greedy_batch(
                model, waves, family="moss", prompt=prompt, max_tokens=2048
            )
            for r, wave, text, done in zip(batch, waves, texts, complete, strict=True):
                issues = decode_issues(text) + ([] if done else ["token_budget"])
                repair = None
                if issues:
                    original = text
                    result = model.generate(
                        wave,
                        prompt=prompt,
                        max_tokens=2048,
                        temperature=0.0,
                        repetition_penalty=1.1,
                        verbose=False,
                    )
                    text = result.text.strip()
                    repair = dict(
                        original_raw_text=original,
                        original_issues=issues,
                        remaining_issues=decode_issues(text),
                        repetition_penalty=1.1,
                    )
                segments, recovered = parse(text, r["duration"])
                result = dict(
                    **r,
                    model=str(args.model.resolve()),
                    provenance=provenance,
                    identity_names=names,
                    raw_text=text,
                    segments=segments,
                    batch_size=len(batch),
                    elapsed_seconds=(time.monotonic() - started) / len(batch),
                )
                if repair:
                    result["decode_repair"] = repair
                if recovered:
                    result["parser_repair"] = recovered
                save(target / "predictions" / f"{r['cut_id']}.json", result)
            mx.clear_cache()
            print(
                f"MOSS session {entry['session']}: {batch[-1]['cut_id']} / {len(records)}; batch={len(batch)} {time.monotonic()-started:.2f}s",
                flush=True,
            )
        entry["baseline_deployment"] = entry["deployment"]
        entry["deployment"] = str(target.resolve())
        save(args.output / "processing_plan.json", plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sessions", type=int, nargs="*")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
