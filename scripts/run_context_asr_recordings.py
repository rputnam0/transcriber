"""Full-recording Qwen ASR with overlapping context and independent word alignment.

A 30-second ownership interval is decoded with three seconds of real context on
both sides. Each aligned word belongs to the interval containing its midpoint.
All raw outputs, including words outside that interval, remain in the audit cache.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from mlx_speech_batch import greedy_batch
from run_mac_asr_quality import save


def prepare(entry, output, context=3.0):
    source = Path(entry["baseline_deployment"])
    old = json.loads((source / "manifest.json").read_text())
    pcm = Path(old[0]["audio"]).parent.parent / "mono.f32"
    wave = np.memmap(pcm, np.float32, mode="r")
    if len(wave) != round((old[-1]["start"] + old[-1]["duration"]) * 16000):
        raise ValueError("PCM duration differs from source manifest")
    records = []
    for row in old:
        core = wave[round(row["start"] * 16000) : round((row["start"] + row["duration"]) * 16000)]
        if hashlib.sha256(core.tobytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source PCM differs from hashed original chunks")
        lo = max(0, round((row["start"] - context) * 16000))
        hi = min(len(wave), round((row["start"] + row["duration"] + context) * 16000))
        audio = output / "audio" / f"{row['cut_id']}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        clip = np.asarray(wave[lo:hi])
        sha = hashlib.sha256(clip.tobytes()).hexdigest()
        if not audio.exists():
            sf.write(audio, clip, 16000, subtype="FLOAT")
        records.append(
            dict(
                cut_id=row["cut_id"],
                audio=str(audio.resolve()),
                sha256=sha,
                start=lo / 16000,
                duration=(hi - lo) / 16000,
                core_start=row["start"],
                core_end=row["start"] + row["duration"],
            )
        )
    target = output / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != records:
        raise ValueError("Changed ASR audio preparation")
    save(target, records)
    return records


def restore_original_words(text, aligned):
    """Restore ASR punctuation instead of publishing the aligner's cleaned text."""
    tokens = []
    prefix = []
    for token in text.split():
        if any(c.isalnum() for c in token):
            tokens.append(" ".join(prefix + [token]))
            prefix = []
        elif tokens:
            tokens[-1] += " " + token
        else:
            prefix.append(token)

    def norm(word):
        return "".join(c.lower() for c in word if c.isalnum())

    if len(tokens) != len(aligned) or any(
        norm(t) != norm(a["text"]) for t, a in zip(tokens, aligned)
    ):
        raise ValueError("Aligner changed lexical tokens; cannot silently discard original words")
    return [{**a, "text": t} for t, a in zip(tokens, aligned, strict=True)]


def alignment_issues(segments, duration):
    issues = []
    previous = -1.0
    for s in segments:
        if not 0 <= s["start"] <= s["end"] <= duration + 0.02:
            issues.append("out_of_bounds")
        if s["start"] < previous:
            issues.append("nonmonotonic")
        if s["start"] == s["end"]:
            issues.append("zero_duration_word")
        previous = s["start"]
    return sorted(set(issues))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sessions", type=int, nargs="*")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    from mlx_audio.stt import load
    import mlx.core as mx

    models = json.loads(args.models.read_text())
    asr = models["mlx-community/Qwen3-ASR-1.7B-bf16"]
    align = models["mlx-community/Qwen3-ForcedAligner-0.6B-8bit"]
    model = load(asr["path"])
    aligner = load(align["path"])
    provenance = dict(
        asr=asr,
        aligner=align,
        temperature=0,
        context_seconds=3.0,
        schema=1,
        decoder="greedy_equal_length_v1",
        max_tokens=1024,
    )
    plan = json.loads(args.plan.read_text())
    for entry in plan["recordings"]:
        if args.sessions and entry["session"] not in args.sessions:
            continue
        target = args.output / f"session{entry['session']}"
        records = prepare(entry, target)
        if args.limit:
            records = records[: args.limit]
        pending = []
        for r in records:
            path = target / "predictions" / f"{r['cut_id']}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["provenance"] != provenance or cached["sha256"] != r["sha256"]:
                    raise ValueError("Changed ASR cache provenance")
            else:
                pending.append(r)
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
                    raise ValueError("ASR input mismatch")
                waves.append(wave)
            started = time.monotonic()
            texts, done = greedy_batch(model, waves, family="qwen", max_tokens=1024)
            for r, wave, text, complete in zip(batch, waves, texts, done, strict=True):
                # Raw output has a language header in Qwen's greedy stream.
                if "<asr_text>" in text:
                    text = text.split("<asr_text>", 1)[1].strip()
                if text.startswith("language English"):
                    text = text[len("language English") :].strip()
                aligned = (
                    aligner.generate(audio=wave, text=text, language="English").segments
                    if text
                    else []
                )
                issues = alignment_issues(aligned, r["duration"])
                aligned = restore_original_words(text, aligned)
                if not complete:
                    issues.append("token_budget")
                words = []
                for s in aligned:
                    start = r["start"] + max(0.0, min(r["duration"], s["start"]))
                    end = r["start"] + max(0.0, min(r["duration"], s["end"]))
                    midpoint = (start + end) / 2
                    if r["core_start"] <= midpoint < r["core_end"]:
                        words.append(dict(text=s["text"], start=start, end=end))
                save(
                    target / "predictions" / f"{r['cut_id']}.json",
                    dict(
                        **r,
                        provenance=provenance,
                        text=text,
                        aligned_segments=aligned,
                        words=words,
                        issues=issues,
                        elapsed_seconds=(time.monotonic() - started) / len(batch),
                    ),
                )
            mx.clear_cache()
            print(
                f"ASR session {entry['session']}: {batch[-1]['cut_id']} / {len(records)}; batch={len(batch)} {time.monotonic()-started:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
