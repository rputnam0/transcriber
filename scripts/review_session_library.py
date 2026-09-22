"""Prepare and serve full-session speaker annotation, separate from inference outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import secrets
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from review_speakers import byte_range

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "tools" / "session_review"


def prepare(plan_path, exports, cache):
    """Snapshot completed exports; prepare seekable audio without touching model caches."""
    entries = []
    for entry in json.loads(plan_path.read_text())["recordings"]:
        n = entry["session"]
        if n == 1:  # The existing reviewer owns the original human annotations.
            continue
        transcript = exports / f"Session {n}.turns.json"
        if not transcript.exists():
            continue
        raw = transcript.read_bytes()
        dataset = json.loads(raw)
        audit = dataset["provenance"]
        named = Path(entry["deployment"]) / "named.json"
        if hashlib.sha256(named.read_bytes()).hexdigest() != audit["source_transcript_sha256"]:
            raise ValueError(f"Session {n}: stale transcript export")
        digest = hashlib.sha256(raw).hexdigest()
        folder = cache / f"session{n}" / digest
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / "transcript.json"
        if not snapshot.exists():
            snapshot.write_bytes(raw)
        source = Path(entry["audio"])
        audio = source
        if source.suffix.lower() not in {".m4a", ".mp3"}:
            audio = cache / f"session{n}" / "audio.m4a"
            if not audio.exists():
                temporary = audio.with_name("audio.preparing.m4a")
                print(f"Preparing browser audio: Session {n}", flush=True)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "24000",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "80k",
                        "-threads",
                        "2",
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ],
                    check=True,
                )
                temporary.replace(audio)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if abs(duration - audit["duration_seconds"]) > 0.3:
            raise ValueError(f"Session {n}: review audio duration differs from transcript")
        entries.append(
            {
                "id": str(n),
                "title": f"Session {n}",
                "audio": str(audio),
                "transcript": str(snapshot),
                "review": str(folder / "review.json"),
                "duration": duration,
                "note": audit.get("source_note", ""),
                "roster": [audit["display_names"][s] for s in audit["allowed_speakers"]],
            }
        )
        print(f"Ready: Session {n}, {len(dataset['segments'])} turns", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "library.json"
    manifest.write_text(json.dumps(entries, indent=2) + "\n")
    return manifest


def make_server(manifest, port=8767):
    entries = json.loads(manifest.read_text())
    sessions = {}
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    for entry in entries:
        raw = Path(entry["transcript"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        dataset = json.loads(raw)
        state = {
            "transcript_sha256": digest,
            "session": entry["id"],
            "revision": 0,
            "roster": entry["roster"],
            "reviews": {},
        }
        review_path = Path(entry["review"])
        if review_path.exists():
            state = json.loads(review_path.read_text())
            if state["transcript_sha256"] != digest or state["session"] != entry["id"]:
                raise ValueError("Review does not match this transcript/session")
        sessions[entry["id"]] = {
            "entry": entry,
            "segments": dataset["segments"],
            "digest": digest,
            "review": state,
        }

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body, mime="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def session(self):
            return sessions.get(parse_qs(urlsplit(self.path).query).get("session", [""])[0])

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/sessions":
                self.send_bytes(
                    json.dumps(
                        [
                            {k: s["entry"][k] for k in ("id", "title", "duration", "note")}
                            for s in sessions.values()
                        ]
                    ).encode()
                )
            elif path in {"/api/data", "/audio"}:
                s = self.session()
                if s is None:
                    self.send_error(404)
                    return
                if path == "/api/data":
                    with lock:
                        self.send_bytes(
                            json.dumps(
                                {
                                    "title": s["entry"]["title"],
                                    "session": s["entry"]["id"],
                                    "roster": s["entry"]["roster"],
                                    "duration": s["entry"]["duration"],
                                    "note": s["entry"]["note"],
                                    "segments": s["segments"],
                                    "review": s["review"],
                                    "token": token,
                                }
                            ).encode()
                        )
                    return
                audio = Path(s["entry"]["audio"])
                size = audio.stat().st_size
                requested = self.headers.get("Range")
                try:
                    start, end = byte_range(requested, size)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206 if requested else 200)
                self.send_header("Content-Type", mimetypes.guess_type(audio.name)[0] or "audio/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if requested:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if self.command == "HEAD":
                    return
                try:
                    with audio.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            block = stream.read(min(65536, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif path in {"/", "/app.js", "/style.css"}:
                name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                mime = {"/": "text/html", "/app.js": "text/javascript", "/style.css": "text/css"}[
                    path
                ]
                self.send_bytes((UI / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            s = self.session()
            if urlsplit(self.path).path != "/api/review" or s is None:
                self.send_error(404)
                return
            if self.headers.get("X-Review-Token") != token:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 8_000_000:
                    raise ValueError("Invalid payload size")
                data = json.loads(self.rfile.read(length))
                if data["transcript_sha256"] != s["digest"] or data["session"] != s["entry"]["id"]:
                    raise ValueError("Transcript/session mismatch")
                roster = data["roster"]
                if (
                    not isinstance(roster, list)
                    or len(roster) != len(s["entry"]["roster"])
                    or any(not isinstance(x, str) or not x.strip() or len(x) > 100 for x in roster)
                    or len(set(roster)) != len(roster)
                ):
                    raise ValueError("Each voice needs a distinct nonempty name")
                if not isinstance(data["reviews"], dict):
                    raise ValueError("Invalid reviews")
                for key, item in data["reviews"].items():
                    if (
                        not key.isdigit()
                        or str(int(key)) != key
                        or not 0 <= int(key) < len(s["segments"])
                    ):
                        raise ValueError("Unknown turn")
                    if item.get("verdict") not in ("correct", "wrong", "unsure", "ungraded"):
                        raise ValueError("Unknown verdict")
                    slot = item.get("speaker_slot")
                    if slot is not None and (type(slot) is not int or not 0 <= slot < len(roster)):
                        raise ValueError("Unknown speaker slot")
                    if (
                        not isinstance(item.get("note", ""), str)
                        or len(item.get("note", "")) > 5000
                    ):
                        raise ValueError("Invalid note")
                with lock:
                    if data.get("revision") != s["review"]["revision"]:
                        self.send_bytes(
                            b'{"error":"Another tab saved changes. Reload before editing."}',
                            status=409,
                        )
                        return
                    data["revision"] += 1
                    review = Path(s["entry"]["review"])
                    review.parent.mkdir(parents=True, exist_ok=True)
                    temporary = review.with_suffix(".tmp")
                    temporary.write_text(json.dumps(data, indent=2) + "\n")
                    temporary.replace(review)
                    s["review"] = data
                    self.send_bytes(
                        json.dumps({"saved": True, "revision": data["revision"]}).encode()
                    )
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self.send_bytes(json.dumps({"error": str(error)}).encode(), status=400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / ".outputs/all_single_file_20260919/processing_plan.json"
    )
    parser.add_argument("--exports", type=Path, default=ROOT / "transcripts/early_sessions")
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache/transcriber/speaker-review/library"
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--serve-only", action="store_true")
    args = parser.parse_args()
    manifest = (
        args.cache / "library.json"
        if args.serve_only
        else prepare(args.plan, args.exports, args.cache)
    )
    if args.prepare_only:
        return
    server = make_server(manifest, args.port)
    print(f"Session reviewer: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
