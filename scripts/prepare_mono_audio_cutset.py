from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def padded_duration(duration: float, *, chunk_seconds: float) -> float:
    if duration <= 0.0 or chunk_seconds <= 0.0:
        raise ValueError("Audio and chunk durations must be positive")
    return math.ceil(duration / chunk_seconds) * chunk_seconds


def mono_cut_id(session_number: int, offset_seconds: float) -> str:
    return f"session_{session_number}_w000000_c{int(round(offset_seconds * 1000)):06d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize one mono recording and create exact 30-second inference cuts."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-number", type=int, default=0)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    args = parser.parse_args()

    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording

    source_info = sf.info(args.audio)
    duration = float(source_info.duration)
    output_duration = padded_duration(duration, chunk_seconds=args.chunk_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_audio = args.output_dir / "mono_16k.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.audio),
            "-af",
            f"apad=whole_dur={output_duration:.3f}",
            "-t",
            f"{output_duration:.3f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(normalized_audio),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = Recording.from_file(
        normalized_audio,
        recording_id=f"session_{args.session_number}_mono",
    )
    cuts = []
    offset = 0.0
    while offset < output_duration - 1e-6:
        cut_id = mono_cut_id(args.session_number, offset)
        cuts.append(
            MonoCut(
                id=cut_id,
                start=offset,
                duration=args.chunk_seconds,
                channel=0,
                recording=recording,
                supervisions=[],
                custom={
                    "source_audio": str(args.audio.resolve()),
                    "source_duration": duration,
                    "inference_only": True,
                },
            )
        )
        offset += args.chunk_seconds
    cutset_path = args.output_dir / "mono_cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(cutset_path)
    summary = {
        "source_audio": str(args.audio.resolve()),
        "normalized_audio": str(normalized_audio.resolve()),
        "source_duration": duration,
        "padded_duration": output_duration,
        "chunk_seconds": args.chunk_seconds,
        "cut_count": len(cuts),
        "session": f"Session {args.session_number}",
        "cutset": str(cutset_path),
        "inference_uses_isolated_audio": False,
    }
    (args.output_dir / "mono_cutset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
