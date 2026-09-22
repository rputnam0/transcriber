"""Re-decode failed full-recording ASR passages, retaining every original output."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import soundfile as sf
from run_context_asr_recordings import alignment_issues, restore_original_words
from run_mac_asr_quality import save


def text_issues(text):
    tokens = re.findall(r"\w+", text.lower())
    for width in range(1, 7):
        needed = max(8, (40 + width - 1) // width)
        for start in range(len(tokens) - width * needed + 1):
            block = tokens[start : start + width]
            if tokens[start : start + width * needed] == block * needed:
                return ["runaway_phrase_repeat"]
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    args = p.parse_args()
    pending = []
    for path in sorted(args.root.glob("session*/predictions/*.json")):
        record = json.loads(path.read_text())
        reasons = text_issues(record["text"])
        if "token_budget" in record["issues"]:
            reasons.append("token_budget")
        if reasons:
            pending.append((path, record, reasons))
    if not pending:
        print("No failed ASR passages", flush=True)
        return
    from mlx_audio.stt import load
    import mlx.core as mx

    models = json.loads(args.models.read_text())
    model = load(models["mlx-community/Qwen3-ASR-1.7B-bf16"]["path"])
    aligner = load(models["mlx-community/Qwen3-ForcedAligner-0.6B-8bit"]["path"])
    for path, record, reasons in pending:
        wave, rate = sf.read(record["audio"], dtype="float32")
        if rate != 16000 or hashlib.sha256(wave.tobytes()).hexdigest() != record["sha256"]:
            raise ValueError("Repair source differs from original inference")
        original = path.parent.parent / "failed_originals" / path.name
        if not original.exists():
            save(original, record)
        result = model.generate(
            wave,
            language="English",
            temperature=0.0,
            max_tokens=1024,
            repetition_penalty=1.1,
            verbose=False,
        )
        text = result.text.strip()
        if text_issues(text) or result.generation_tokens >= 1024:
            raise RuntimeError(f"Repair failed for {path}")
        aligned = (
            aligner.generate(audio=wave, text=text, language="English").segments if text else []
        )
        aligned = restore_original_words(text, aligned)
        words = []
        for word in aligned:
            start = record["start"] + max(0, min(record["duration"], word["start"]))
            end = record["start"] + max(0, min(record["duration"], word["end"]))
            if record["core_start"] <= (start + end) / 2 < record["core_end"]:
                words.append(dict(text=word["text"], start=start, end=end))
        record.update(
            text=text,
            aligned_segments=aligned,
            words=words,
            issues=alignment_issues(aligned, record["duration"]),
            decode_repair=dict(
                original=str(original.resolve()),
                reasons=reasons,
                original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                decoder="mlx_audio_native_single",
                repetition_penalty=1.1,
                generation_tokens=result.generation_tokens,
                remaining_issues=[],
            ),
        )
        save(path, record)
        mx.clear_cache()
        print(
            "REPAIRED", path.parent.parent.name, path.stem, len(text.split()), "words", flush=True
        )


if __name__ == "__main__":
    main()
