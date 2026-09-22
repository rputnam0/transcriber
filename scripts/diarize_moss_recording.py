#!/usr/bin/env python3
"""Run the evaluated 30-second MOSS recipe on a complete recording, with resumable caches.

Prepare/attribute in the app environment; infer in the separate MOSS environment.
Each chunk's anonymous IDs are independently bound to the enrolled speaker roster.
No isolated source track or reference transcript is read by this deployment runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from test_moss_mac import MODEL, infer, save
from moss_batch_inference import decode_issues


def pending_prediction_indices(records, output, model):
    pending = []
    for index, record in enumerate(records):
        path = output / "predictions" / f"{record['cut_id']}.json"
        if not path.exists():
            pending.append(index)
        else:
            cached = json.loads(path.read_text())
            if cached["sha256"] != record["sha256"] or cached["model"] != model:
                raise ValueError("Stale prediction; use a fresh output directory")
            if decode_issues(cached.get("raw_text", "")):
                pending.append(index)
    return pending


def infer_recording(args):
    """Process every chunk, recycling MPS workers before caches exhaust unified RAM."""
    if args.device != "mps":
        return infer(args)
    records = json.loads((args.output / "manifest.json").read_text())
    environment = dict(os.environ)
    # Defaults permit allocation beyond the recommended working set. Keep this job
    # below it and ask the allocator to collect unused buffers earlier.
    environment.update(
        PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.75", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.6"
    )
    pending = pending_prediction_indices(records, args.output, args.model)
    while pending:
        stop = min(len(records), pending[0] + args.worker_chunks)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "infer-worker",
            "--output",
            str(args.output),
            "--model",
            args.model,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.batch_size),
            "--attention",
            args.attention,
            "--worker-end",
            str(stop),
        ]
        print(f"WORKER: source chunks {pending[0]}..{stop - 1} of {len(records)}", flush=True)
        result = subprocess.run(command, env=environment)
        if result.returncode == 86 and args.batch_size > 1:
            # Retry the unsaved audio with a smaller batch, retaining the same model,
            # precision and token budget. Never skip a memory-intensive passage.
            command[command.index("--batch-size") + 1] = "1"
            print("MPS memory limit reached; retrying pending chunks individually", flush=True)
            subprocess.run(command, env=environment, check=True)
        elif result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        remaining = pending_prediction_indices(records, args.output, args.model)
        if len(remaining) >= len(pending):
            raise RuntimeError("Inference worker made no progress")
        pending = remaining


def infer_worker(args):
    try:
        return infer(args)
    except RuntimeError as error:
        if "MPS" in str(error) and "out of memory" in str(error):
            print(str(error), file=sys.stderr, flush=True)
            raise SystemExit(86) from error
        raise


def prepare(args):
    import numpy as np
    import soundfile as sf

    args.output.mkdir(parents=True, exist_ok=True)
    stat = args.audio.stat()
    source = dict(path=str(args.audio.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    provenance = args.output / "source.json"
    if provenance.exists() and json.loads(provenance.read_text()) != source:
        raise ValueError("Recording changed; use a new output directory")
    pcm = args.output / "mono.f32"
    if not pcm.exists():
        temporary = pcm.with_suffix(".partial")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(args.audio),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(pcm)
        save(provenance, source)
    wave = np.memmap(pcm, np.float32, mode="r")
    records = []
    for index, lo in enumerate(range(0, len(wave), 30 * 16000)):
        clip = np.asarray(wave[lo : lo + 30 * 16000])
        cut_id = f"{index:05d}"
        audio = args.output / "audio" / f"{cut_id}.wav"
        audio.parent.mkdir(exist_ok=True)
        if not audio.exists():
            sf.write(audio, clip, 16000, subtype="FLOAT")
        records.append(
            dict(
                cut_id=cut_id,
                split="deployment",
                start=lo / 16000,
                duration=len(clip) / 16000,
                audio=str(audio.resolve()),
                sha256=hashlib.sha256(clip.tobytes()).hexdigest(),
            )
        )
    save(args.output / "manifest.json", records)
    print(f"Prepared {len(records)} clips / {len(wave) / 16000:.1f} seconds", flush=True)


def timestamped_turns(record, turns, names, bindings):
    """Keep uncertain names and cut boundaries visible for review, including overlapping turns."""
    output = []
    for turn, name in zip(turns, names, strict=True):
        binding = bindings.get(turn["speaker"], {})
        reasons = []
        if name in (None, "Unknown"):
            reasons.append("unknown speaker")
        elif binding.get("review_required", True):
            reasons.append("uncertain enrollment")
        if name and binding.get("proposed_speaker") not in (None, name):
            reasons.append("identity methods disagree")
        if turn["start"] < 0.15 or turn["end"] > record["duration"] - 0.15:
            reasons.append("chunk boundary")
        if any(
            other["speaker"] != turn["speaker"]
            and min(other["end"], turn["end"]) > max(other["start"], turn["start"])
            for other in turns
        ):
            reasons.append("overlapping speech")
        output.append(
            dict(
                start=record["start"] + turn["start"],
                end=record["start"] + turn["end"],
                speaker=name or "Unknown",
                text=turn["text"],
                local_speaker=turn["speaker"],
                cut_id=record["cut_id"],
                review_required=bool(reasons),
                review_reasons=reasons,
            )
        )
    return output


def readable_paragraphs(turns, max_gap=1.2, max_span=45.0):
    """Join adjacent fragments without crossing another voice or changing raw turns."""
    paragraphs = []
    for turn in turns:
        previous = paragraphs[-1] if paragraphs else None
        if (
            previous
            and previous["speaker"] == turn["speaker"]
            and 0 <= turn["start"] - previous["end"] <= max_gap
            and turn["end"] - previous["start"] <= max_span
        ):
            previous["text"] += " " + turn["text"]
            previous["end"] = turn["end"]
            previous["review_required"] |= turn["review_required"]
            previous["source_turns"] += 1
        else:
            paragraphs.append(
                dict(
                    start=turn["start"],
                    end=turn["end"],
                    speaker=turn["speaker"],
                    text=turn["text"],
                    review_required=turn["review_required"],
                    source_turns=1,
                )
            )
    return paragraphs


def write_readable(output, turns):
    from transcriber.srt import _fmt_srt_ts

    paragraphs = readable_paragraphs(turns)
    (output / "readable.txt").write_text(
        "\n\n".join(
            f"[{_fmt_srt_ts(p['start'])}] {p['speaker']}"
            f"{' [review]' if p['review_required'] else ''}: {p['text']}"
            for p in paragraphs
        )
        + "\n"
    )


def attribute(args):
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import get_token

    from attribute_cached_diarization import bind_clusters, cluster_features
    from refine_named_turns import apply_local_identity, corrected_names, turn_features
    from transcriber.diarization import DEFAULT_DIARIZATION_MODEL, _resolve_embedder
    from transcriber.srt import _fmt_srt_ts, write_srt

    torch.set_num_threads(4)
    identity = dict(np.load(args.identity))
    policy = json.loads(args.policy.read_text())["policy"] if args.policy else None
    allowed = None
    if args.method == "moss_hybrid_local_names":
        if policy is None or not getattr(args, "session", None):
            raise ValueError("Hybrid local attribution requires --policy and --session")
        roster_config = json.loads(args.rosters.read_text())
        allowed = roster_config["session_rosters"][str(args.session)]
    embedder = _resolve_embedder(
        model_name=DEFAULT_DIARIZATION_MODEL,
        hf_token=get_token(),
        device=args.device,
    )
    provenance = dict(
        method=args.method,
        identity_sha256=hashlib.sha256(args.identity.read_bytes()).hexdigest(),
        policy=policy,
    )
    if allowed is not None:
        provenance.update(allowed_speakers=allowed, local_identity_version=1)
    records = json.loads((args.output / "manifest.json").read_text())
    all_turns = []
    for record in records:
        prediction_path = args.output / "predictions" / f"{record['cut_id']}.json"
        prediction = json.loads(prediction_path.read_text())
        if prediction["sha256"] != record["sha256"]:
            raise ValueError("Prediction audio differs from manifest")
        current = dict(
            provenance, prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        )
        cached = args.output / "attribution" / f"{record['cut_id']}.json"
        if cached.exists():
            result = json.loads(cached.read_text())
            if result["provenance"] != current:
                raise ValueError("Attribution configuration changed; use a fresh cache")
        else:
            wave, _ = sf.read(record["audio"], dtype="float32")
            if hashlib.sha256(wave.tobytes()).hexdigest() != record["sha256"]:
                raise ValueError("Chunk audio changed")
            turns = prediction["segments"]
            baseline = getattr(args, "baseline", None)
            if baseline:
                previous = json.loads((baseline / "attribution" / cached.name).read_text())
                for key in ["identity_sha256", "prediction_sha256"]:
                    if previous["provenance"][key] != current[key]:
                        raise ValueError(f"Baseline {key} mismatch")
                bindings = previous["bindings"]
            else:
                bindings = bind_clusters(cluster_features(wave, turns, embedder), identity)
            corrections = []
            if args.method in {"moss_direct_names", "moss_hybrid_names", "moss_hybrid_local_names"}:
                roster = prediction.get("identity_names")
                if not roster:
                    raise ValueError("Checkpoint has no stable speaker roster")
                mapping = {f"S{i+1:02d}": name for i, name in enumerate(roster)}
                names = [
                    (
                        bindings.get(t["speaker"], {}).get("proposed_speaker")
                        if args.method in {"moss_hybrid_names", "moss_hybrid_local_names"}
                        else None
                    )
                    or mapping.get(t["speaker"])
                    for t in turns
                ]
                if args.method == "moss_hybrid_local_names":
                    names, corrections = apply_local_identity(
                        names, turn_features(wave, turns, embedder), identity, policy, allowed
                    )
            elif args.method == "moss_local_names":
                if policy is None:
                    raise ValueError("Local corrections require the frozen policy")
                names = corrected_names(
                    turns, bindings, turn_features(wave, turns, embedder), identity, policy
                )
            else:
                names = [bindings.get(t["speaker"], {}).get("proposed_speaker") for t in turns]
            result = dict(
                provenance=current,
                bindings=bindings,
                turns=timestamped_turns(record, turns, names, bindings),
            )
            if args.method == "moss_hybrid_local_names":
                result["local_corrections"] = corrections
                for correction in corrections:
                    result["turns"][correction["turn_index"]][
                        "local_identity_correction"
                    ] = correction
            save(cached, result)
        all_turns.extend(result["turns"])
        print("ATTRIBUTED", record["cut_id"], flush=True)
    all_turns.sort(key=lambda t: (t["start"], t["end"]))
    save(
        args.output / "named.json",
        dict(
            source=json.loads((args.output / "source.json").read_text()),
            provenance=provenance,
            segments=all_turns,
            reference_used=False,
            review_note="Automatic draft. Names use recording handles. Segment timing is coarse; review flags are heuristics, not calibrated probabilities.",
        ),
    )
    write_srt(
        args.output / "named.srt",
        [
            (i, t["start"], t["end"], f"{t['speaker']}: {t['text']}")
            for i, t in enumerate(all_turns, 1)
        ],
    )
    (args.output / "named.txt").write_text(
        "\n\n".join(
            f"[{_fmt_srt_ts(t['start'])}] {t['speaker']}"
            f"{' [review]' if t['review_required'] else ''}: {t['text']}"
            for t in all_turns
        )
        + "\n"
    )
    write_readable(args.output, all_turns)
    print(
        f"Exported {len(all_turns)} turns; {sum(t['review_required'] for t in all_turns)} flagged",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "infer", "infer-worker", "attribute"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--method",
        choices=[
            "moss_named",
            "moss_local_names",
            "moss_direct_names",
            "moss_hybrid_names",
            "moss_hybrid_local_names",
        ],
        default="moss_named",
    )
    parser.add_argument("--session", type=int)
    parser.add_argument(
        "--baseline", type=Path, help="Reuse verified cluster evidence from an earlier deployment"
    )
    parser.add_argument("--rosters", type=Path, default=Path("config/early_session_rosters.json"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--worker-chunks", type=int, default=32)
    parser.add_argument("--worker-end", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_chunks < 1:
        parser.error("--worker-chunks must be positive")
    args.split, args.limit = "deployment", args.worker_end if args.command == "infer-worker" else 0
    args.repair_repetition = args.command == "infer-worker"
    {
        "prepare": prepare,
        "infer": infer_recording,
        "infer-worker": infer_worker,
        "attribute": attribute,
    }[args.command](args)


if __name__ == "__main__":
    main()
