"""Resumable, audio-hashed local ASR pass using dedicated Apple Silicon recognizers.

Interfaces: research citations C1-C4 in mac_asr_quality_20260921_sources.md.
Raw recognizer text is retained; this script performs no editorial rewriting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time

import soundfile as sf


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--engine", choices=["mlx", "whisper"], required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hotwords", type=Path)
    args = p.parse_args()
    records = json.loads(args.manifest.read_text())
    if args.limit:
        records = records[: args.limit]
    hotwords = json.loads(args.hotwords.read_text()) if args.hotwords else []
    provenance = dict(
        model=args.model,
        engine=args.engine,
        language="English",
        temperature=0,
        hotwords=hotwords,
        schema=1,
    )
    if args.engine == "mlx":
        from mlx_audio.stt import load

        model = load(args.model)
        if "granite-4.0" in args.model:
            # mlx-audio 0.4.4 double-transposes already converted bf16 pointwise
            # convs. Restore MLX (out, kernel=1, in) only for that observed shape.
            import mlx.core as mx
            from mlx.utils import tree_flatten

            fixes = [
                (k, v.transpose(0, 2, 1))
                for k, v in tree_flatten(model.parameters())
                if any(n in k for n in ("up_conv.weight", "down_conv.weight"))
                and v.ndim == 3
                and v.shape[1] > 1
                and v.shape[2] == 1
            ]
            model.load_weights(fixes, strict=False)
            mx.eval(model.parameters())
            provenance["granite_pointwise_layout_fix"] = [k for k, _ in fixes]
    else:
        import mlx_whisper
    import mlx.core as mx

    for index, record in enumerate(records, 1):
        target = args.output / f"{record['cut_id']}.json"
        current = dict(provenance, audio_sha256=record["sha256"])
        if target.exists():
            if json.loads(target.read_text())["provenance"] != current:
                raise ValueError("ASR cache provenance mismatch; use a new output directory")
            continue
        wave, rate = sf.read(record["audio"], dtype="float32")
        if rate != 16000 or wave.ndim != 1:
            raise ValueError("ASR requires prepared 16 kHz mono")
        if hashlib.sha256(wave.tobytes()).hexdigest() != record["sha256"]:
            raise ValueError("Prepared audio was modified")
        started = time.monotonic()
        if args.engine == "whisper":
            result = mlx_whisper.transcribe(
                wave,
                path_or_hf_repo=args.model,
                language="en",
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
                initial_prompt=", ".join(hotwords) if hotwords else None,
                hallucination_silence_threshold=2.0,
                verbose=None,
            )
            text, segments = result["text"].strip(), result["segments"]
        else:
            result = model.generate(
                wave,
                language="English",
                max_tokens=1024,
                temperature=0.0,
                hotwords=hotwords or None,
                verbose=False,
            )
            text = result.text.strip()
            segments = [asdict(s) if is_dataclass(s) else s for s in (result.segments or [])]
        save(
            target,
            dict(
                provenance=current,
                cut_id=record["cut_id"],
                start=record["start"],
                duration=len(wave) / rate,
                text=text,
                segments=segments,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        mx.clear_cache()
        print(
            f"ASR {index}/{len(records)} {record['cut_id']} {time.monotonic()-started:.2f}s {len(text.split())} words",
            flush=True,
        )


if __name__ == "__main__":
    main()
