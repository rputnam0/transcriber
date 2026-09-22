"""Check real seeking and review persistence without loading speech models."""

import importlib.util
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

SPEC = importlib.util.spec_from_file_location(
    "review_speakers", Path(__file__).parents[1] / "scripts" / "review_speakers.py"
)
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, (0, 99)),
        ("bytes=10-19", (10, 19)),
        ("bytes=90-", (90, 99)),
        ("bytes=-8", (92, 99)),
        ("bytes=90-999", (90, 99)),
    ],
)
def test_range(header, expected):
    assert reviewer.byte_range(header, 100) == expected


@pytest.mark.parametrize(
    "header", ["bytes=100-", "bytes=8-2", "bytes=-0", "bytes=", "bytes=0-1,3-4"]
)
def test_bad_range(header):
    with pytest.raises(ValueError):
        reviewer.byte_range(header, 100)


def test_seek_save_and_reload(tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(bytes(range(100)))
    transcript = tmp_path / "named.json"
    original = {"segments": [{"start": 1, "end": 2, "speaker": "original", "text": "Hi"}]}
    transcript.write_text(json.dumps(original))
    review = tmp_path / "review.json"
    server = reviewer.make_server(audio, transcript, review, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(Request(base + "/audio", headers={"Range": "bytes=40-49"})) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 40-49/100"
            assert response.read() == bytes(range(40, 50))
        with urlopen(base + "/api/data") as response:
            data = json.load(response)
        state = data["review"]
        state["reviews"]["0"] = {"verdict": "wrong", "speaker_slot": 1, "note": "test"}
        body = json.dumps(state).encode()
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(base + "/api/review", data=body))
        assert rejected.value.code == 403
        with urlopen(
            Request(base + "/api/review", data=body, headers={"X-Review-Token": data["token"]})
        ) as response:
            assert json.load(response)["saved"] is True
        assert json.loads(review.read_text())["reviews"]["0"]["speaker_slot"] == 1
        assert json.loads(transcript.read_text()) == original
        with urlopen(base + "/api/data") as response:
            assert json.load(response)["review"] == state
        with pytest.raises(HTTPError) as rejected:
            urlopen(base + "/../named.json")
        assert rejected.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    restarted = reviewer.make_server(audio, transcript, review, port=0)
    restarted.server_close()
    transcript.write_text('{"segments": []}')
    with pytest.raises(ValueError, match="different transcript"):
        reviewer.make_server(audio, transcript, review, port=0)
