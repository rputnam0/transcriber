"""Make compact AAC listening copies; model inference always uses original sources."""

import argparse
import json
from pathlib import Path
import subprocess
from run_mac_asr_quality import save


def prepare(source, target):
    provenance = dict(
        source=str(source.resolve()),
        bytes=source.stat().st_size,
        mtime_ns=source.stat().st_mtime_ns,
        codec="aac",
        bitrate="96k",
        channels=1,
        sample_rate=48000,
    )
    metadata = target.with_suffix(".source.json")
    if target.exists():
        if not metadata.exists() or json.loads(metadata.read_text()) != provenance:
            raise ValueError("Listening copy provenance mismatch")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".partial.m4a")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(target)
    save(metadata, provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for e in json.loads(args.plan.read_text())["recordings"]:
        target = args.output / f"session{e['session']}.m4a"
        prepare(Path(e["audio"]), target)
        print("LISTENING AUDIO", e["session"], target.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
