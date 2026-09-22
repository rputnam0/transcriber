"""Serve a private, local audio/transcript reviewer; no ML dependencies required."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

UI = Path(__file__).resolve().parents[1] / "tools" / "speaker_review"


def byte_range(value: str | None, size: int) -> tuple[int, int]:
    """Resolve one HTTP byte range, rejecting malformed or unsatisfiable ranges."""
    if not value:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("Invalid byte range")
    left, right = match.groups()
    if not left:
        if int(right) == 0:
            raise ValueError("Empty suffix")
        return max(0, size - int(right)), size - 1
    start = int(left)
    end = min(int(right), size - 1) if right else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, end


def make_server(audio: Path, transcript: Path, review: Path, port: int = 8765):
    raw = transcript.read_bytes()
    dataset = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    initial = {
        "transcript_sha256": digest,
        "roster": ["Dungeon master", "Player 1", "Player 2", "Player 3"],
        "reviews": {},
    }
    if review.exists():
        initial = json.loads(review.read_text())
        if initial.get("transcript_sha256") != digest:
            raise ValueError(
                "Review belongs to a different transcript; choose another --review path"
            )
    state = {"value": initial}

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

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/data":
                with lock:
                    payload = {
                        "title": "Session 1 · Speaker review",
                        "segments": dataset["segments"],
                        "review": state["value"],
                        "token": token,
                    }
                    self.send_bytes(json.dumps(payload).encode())
            elif path == "/audio":
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
                self.send_header("Content-Type", "audio/mp4")
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
            elif path in ("/", "/app.js", "/style.css"):
                name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                mime = {"/": "text/html", "/app.js": "text/javascript", "/style.css": "text/css"}[
                    path
                ]
                self.send_bytes((UI / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/review":
                self.send_error(404)
                return
            if self.headers.get("X-Review-Token") != token:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 2_000_000:
                    raise ValueError("Invalid payload size")
                data = json.loads(self.rfile.read(length))
                if data.get("transcript_sha256") != digest:
                    raise ValueError("Transcript mismatch")
                roster = data["roster"]
                if len(roster) != 4 or any(not isinstance(s, str) or not s.strip() for s in roster):
                    raise ValueError("Four nonempty roster names are required")
                if len(set(roster)) != 4:
                    raise ValueError("Roster names must be distinct")
                for key, item in data["reviews"].items():
                    if not key.isdigit() or not 0 <= int(key) < len(dataset["segments"]):
                        raise ValueError("Unknown turn")
                    if item.get("verdict") not in ("correct", "wrong", "unsure", "ungraded"):
                        raise ValueError("Unknown verdict")
                    if item.get("speaker_slot") not in (None, 0, 1, 2, 3):
                        raise ValueError("Unknown speaker slot")
                review.parent.mkdir(parents=True, exist_ok=True)
                with lock:
                    temp = review.with_suffix(".tmp")
                    temp.write_text(json.dumps(data, indent=2) + "\n")
                    temp.replace(review)
                    state["value"] = data
                self.send_bytes(b'{"saved":true}')
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self.send_bytes(json.dumps({"error": str(error)}).encode(), status=400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Browser-ready .m4a audio")
    parser.add_argument("--transcript", type=Path, required=True, help="MOSS named.json")
    parser.add_argument("--review", type=Path, required=True, help="Separate human review JSON")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.audio.stat()
    server = make_server(args.audio, args.transcript, args.review, args.port)
    print(f"Speaker reviewer: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
