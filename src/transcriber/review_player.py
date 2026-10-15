"""Build an offline listening/correction workspace from Qwen worker outputs.

python -m transcriber.review_player --session TITLE REQUEST RESPONSE --output DIR
Repeat --session to put multiple sessions into one review player.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def build_review(
    sessions: list[list[str]],
    output: Path,
    mapping: dict | None = None,
    feedback_path: Path | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    mapping = mapping or {}
    data = []
    for number, (title, request_path, response_path) in enumerate(sessions):
        request = json.loads(Path(request_path).read_text())
        response = json.loads(Path(response_path).read_text())
        if len(request["files"]) != len(response["files"]):
            raise ValueError("Request/response track count mismatch")
        for track, (source, segments) in enumerate(zip(request["files"], response["files"])):
            source = Path(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = output / f"audio-{number}-{track}-{digest[:12]}.m4a"
            if not target.exists():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(source),
                        "-ac",
                        "1",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "64k",
                        str(target),
                    ],
                    check=True,
                )
            handle = source.stem.split("-", 1)[-1]
            label = mapping.get(handle, source.stem)
            activity = response.get("activity", [None] * len(response["files"]))[track]
            data.append(
                dict(
                    id=f"{number}-{track}",
                    session=title,
                    speaker=label,
                    audio=target.name,
                    audio_sha256=digest,
                    segments=segments,
                    activity=activity,
                )
            )
    payload = dict(version=1, tracks=data)
    payload["dataset_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if feedback_path:
        prior = json.loads(feedback_path.read_text())
        old_tracks = {track["id"]: track for track in prior["tracks"]}
        seeds = []
        for entry in prior["feedback"]:
            old = old_tracks[entry["track_id"]]
            match = next((t for t in data if t["audio_sha256"] == old["audio_sha256"]), None)
            if match is None:
                raise ValueError("Cannot carry feedback to different audio")
            seeds.append(
                dict(
                    entry,
                    id=f"{match['id']}:prior:{entry['id']}",
                    track_id=match["id"],
                    session=match["session"],
                    speaker=match["speaker"],
                    segment_index=None,
                    source_dataset_id=prior["dataset_id"],
                )
            )
        payload["seed_feedback"] = seeds
    # JSON in a script element must escape '<' even when the content is data.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("review_player.html").read_text()
    target = output / "index.html"
    target.write_text(template.replace("__REVIEW_DATA__", encoded))
    (output / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        nargs=3,
        action="append",
        required=True,
        metavar=("TITLE", "REQUEST", "RESPONSE"),
    )
    parser.add_argument("--mapping", type=Path, help="JSON mapping of track handle to speaker name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feedback", type=Path, help="Carry corrections from identical source audio"
    )
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text()) if args.mapping else None
    print(build_review(args.session, args.output, mapping, args.feedback).resolve())


if __name__ == "__main__":
    main()
