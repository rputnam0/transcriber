from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from moss_activity_conditioning import (
    install_activity_conditioning,
    load_activity_weights,
    serialized_activity_summary,
)
from moss_target_speaker_conditioning import (
    install_target_speaker_conditioning,
    load_target_speaker_weights,
    serialized_target_activity_summary,
)


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"


def normalized_segments(segments: Iterable[object], *, duration: float) -> list[dict]:
    output = []
    for segment in segments:
        start = max(0.0, float(segment.start))
        end = min(duration, float(segment.end))
        text = str(segment.text).strip()
        speaker = str(segment.speaker).strip()
        if end <= start or not text or not speaker:
            continue
        output.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run joint MOSS transcription and diarization on one mono recording."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--prompt")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        DEFAULT_PROMPT,
        build_transcription_messages,
        generate_transcription,
    )

    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    duration = float(sf.info(args.audio).duration)
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    started = time.time()
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    model_path = Path(str(args.model))
    if not model_path.is_dir():
        model_kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(str(args.model), **model_kwargs).to(device).eval()
    activity_adaptor = None
    if bool(getattr(model.config, "activity_conditioning", False)):
        checkpoint = model_path
        if not checkpoint.is_dir():
            raise ValueError("Activity-conditioned model must be a local checkpoint")
        activity_adaptor = install_activity_conditioning(
            model,
            max_speakers=int(getattr(model.config, "activity_max_speakers", 8)),
        )
        load_activity_weights(model, checkpoint)
    target_adaptor = None
    if bool(getattr(model.config, "target_speaker_conditioning", False)):
        checkpoint = model_path
        if not checkpoint.is_dir():
            raise ValueError("Target-conditioned model must be a local checkpoint")
        target_adaptor = install_target_speaker_conditioning(
            model,
            profile_seconds=float(getattr(model.config, "target_profile_seconds", 4.0)),
            gap_seconds=float(getattr(model.config, "target_gap_seconds", 0.5)),
            attention_heads=int(getattr(model.config, "target_attention_heads", 8)),
        )
        load_target_speaker_weights(model, checkpoint)
    processor_source = DEFAULT_MODEL if model_path.is_dir() else str(args.model)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        revision=args.revision,
        trust_remote_code=True,
    )
    messages = build_transcription_messages(args.audio, args.prompt or DEFAULT_PROMPT)
    if target_adaptor is not None:
        target_adaptor.clear_target_activity_logits()
    generated = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    text = str(generated["text"])
    segments = normalized_segments(parse_transcript(text), duration=duration)
    payload = {
        "model": args.model,
        "revision": args.revision,
        "audio": str(args.audio.resolve()),
        "duration": duration,
        "joint_transcription_and_diarization": True,
        "uses_reference_activity": False,
        "uses_isolated_audio": False,
        "activity_conditioning": activity_adaptor is not None,
        "target_speaker_conditioning": target_adaptor is not None,
        "prompt": args.prompt or DEFAULT_PROMPT,
        "raw_text": text,
        "speaker_count": len({segment["speaker"] for segment in segments}),
        "segment_count": len(segments),
        "segments": segments,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        ),
    }
    activity_summary = serialized_activity_summary(activity_adaptor)
    if activity_summary:
        payload.update(activity_summary)
    target_summary = serialized_target_activity_summary(target_adaptor)
    if target_summary:
        payload.update(target_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"raw_text", "segments"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
