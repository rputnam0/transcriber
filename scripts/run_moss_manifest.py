from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

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
from run_moss_transcribe_diarize import normalized_segments


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_unique_records(
    rows: Iterable[Mapping[str, object]],
    *,
    sessions: set[str] | None = None,
    max_records: int = 0,
    max_records_per_session: int = 0,
) -> list[dict]:
    selected = []
    seen = set()
    session_counts: dict[str, int] = defaultdict(int)
    for raw_row in rows:
        row = dict(raw_row)
        metadata = dict(row.get("metadata") or {})
        cut_id = str(metadata.get("cut_id") or "")
        session = str(metadata.get("session") or "")
        if not cut_id or cut_id in seen or (sessions and session not in sessions):
            continue
        if max_records_per_session > 0 and session_counts[session] >= max_records_per_session:
            continue
        conversation = list(row.get("conversation") or [])
        if len(conversation) != 3:
            raise ValueError(f"Malformed conversation for {cut_id}")
        selected.append(
            {
                "cut_id": cut_id,
                "session": session,
                "prompt": str(conversation[0].get("content") or ""),
                "audio": str(conversation[1].get("content") or ""),
            }
        )
        seen.add(cut_id)
        session_counts[session] += 1
        if max_records > 0 and len(selected) >= max_records:
            break
    return selected


def resolve_processor_source(model: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return DEFAULT_MODEL if Path(model).is_dir() else model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one loaded MOSS checkpoint over a mono conversation manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--processor")
    parser.add_argument("--prompt", help="Override the manifest prompt for every recording.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sessions")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-records-per-session", type=int, default=0)
    parser.add_argument(
        "--activity-only",
        action="store_true",
        help="Run the mono encoder/activity heads without autoregressive transcript decoding.",
    )
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    sessions = (
        {value.strip() for value in args.sessions.split(",") if value.strip()}
        if args.sessions
        else None
    )
    selected = select_unique_records(
        _read_jsonl(args.manifest),
        sessions=sessions,
        max_records=args.max_records,
        max_records_per_session=args.max_records_per_session,
    )
    if not selected:
        raise ValueError("No manifest records selected")
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if not Path(args.model).is_dir():
        model_kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device).eval()
    activity_adaptor = None
    if bool(getattr(model.config, "activity_conditioning", False)):
        if not Path(args.model).is_dir():
            raise ValueError("Activity-conditioned model must be loaded from a local checkpoint")
        activity_adaptor = install_activity_conditioning(
            model,
            max_speakers=int(getattr(model.config, "activity_max_speakers", 8)),
        )
        load_activity_weights(model, Path(args.model))
    target_adaptor = None
    if bool(getattr(model.config, "target_speaker_conditioning", False)):
        if not Path(args.model).is_dir():
            raise ValueError("Target-conditioned model must be loaded from a local checkpoint")
        target_adaptor = install_target_speaker_conditioning(
            model,
            profile_seconds=float(getattr(model.config, "target_profile_seconds", 4.0)),
            gap_seconds=float(getattr(model.config, "target_gap_seconds", 0.5)),
            attention_heads=int(getattr(model.config, "target_attention_heads", 8)),
        )
        load_target_speaker_weights(model, Path(args.model))
    processor_source = resolve_processor_source(args.model, args.processor)
    processor_kwargs = {"trust_remote_code": True}
    if not Path(processor_source).is_dir():
        processor_kwargs["revision"] = args.revision
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)

    records = []
    started = time.time()
    for index, item in enumerate(selected, 1):
        audio = Path(item["audio"])
        duration = float(sf.info(audio).duration)
        if activity_adaptor is not None:
            activity_adaptor.clear_activity_logits()
        if target_adaptor is not None:
            target_adaptor.clear_target_activity_logits()
        messages = build_transcription_messages(audio, args.prompt or item["prompt"])
        if args.activity_only:
            if activity_adaptor is None:
                raise ValueError("--activity-only requires an activity-conditioned checkpoint")
            import numpy as np
            import soxr

            samples, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
            samples = np.asarray(samples.mean(axis=1), dtype=np.float32)
            expected_rate = int(processor.feature_extractor.sampling_rate)
            if sample_rate != expected_rate:
                samples = soxr.resample(samples, sample_rate, expected_rate).astype(np.float32)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = processor(text=[prompt], audio=[samples], return_tensors="pt")
            feature_inputs = {
                key: batch[key].to(
                    device=device,
                    dtype=dtype if batch[key].is_floating_point() else None,
                )
                for key in (
                    "input_features",
                    "audio_feature_lengths",
                    "audio_chunk_mapping",
                )
            }
            with torch.inference_mode():
                model.model.get_audio_features(**feature_inputs)
            raw_text = ""
            segments = []
        else:
            generated = generate_transcription(
                model,
                processor,
                messages,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                device=device,
                dtype=dtype,
            )
            raw_text = str(generated["text"])
            segments = normalized_segments(parse_transcript(raw_text), duration=duration)
        record = {
            **item,
            "duration": duration,
            "raw_text": raw_text,
            "speaker_count": len({segment["speaker"] for segment in segments}),
            "segment_count": len(segments),
            "segments": segments,
            "uses_reference_activity": False,
            "uses_isolated_audio": False,
        }
        activity_summary = serialized_activity_summary(activity_adaptor)
        if activity_summary:
            record.update(activity_summary)
        target_summary = serialized_target_activity_summary(target_adaptor)
        if target_summary:
            record.update(target_summary)
        records.append(record)
        print(
            f"{index}/{len(selected)} {item['cut_id']}: "
            f"{len(segments)} segments, {records[-1]['speaker_count']} speakers",
            flush=True,
        )
    payload = {
        "manifest": str(args.manifest),
        "model": args.model,
        "revision": args.revision,
        "processor": processor_source,
        "joint_transcription_and_diarization": True,
        "uses_reference_activity": False,
        "uses_isolated_audio": False,
        "activity_conditioning": activity_adaptor is not None,
        "target_speaker_conditioning": target_adaptor is not None,
        "activity_only": args.activity_only,
        "record_count": len(records),
        "elapsed_seconds": time.time() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
