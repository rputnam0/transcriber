"""Local transcript reader with byte-range audio seeking; no annotation mutation."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit
from review_speakers import byte_range


def make_server(root, port):
    root = Path(root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            relative = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
            parts = Path(relative).parts
            if ".." in parts or (len(parts) > 1 and not (len(parts) == 2 and parts[0] == "audio")):
                self.send_error(404)
                return
            file = root / relative
            if not file.is_file():
                self.send_error(404)
                return
            size = file.stat().st_size
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
            self.send_header(
                "Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            )
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with file.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        data = stream.read(min(65536, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--port", type=int, default=8768)
    args = p.parse_args()
    server = make_server(args.root, args.port)
    print(f"Transcript reader: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
