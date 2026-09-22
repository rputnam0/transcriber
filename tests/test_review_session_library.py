"""Exercise isolated reviews, six-person rosters, seeking, and stale-tab protection."""

import importlib.util
import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "review_session_library", SCRIPTS / "review_session_library.py"
)
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


def test_multi_session_review_and_restart(tmp_path):
    entries = []
    for n, count in [(2, 4), (36, 6)]:
        audio = tmp_path / f"{n}.m4a"
        audio.write_bytes(bytes(range(100)))
        transcript = tmp_path / f"{n}.json"
        transcript.write_text(
            json.dumps({"segments": [{"start": 1, "end": 2, "speaker": "DM", "text": "Hello"}]})
        )
        entries.append(
            {
                "id": str(n),
                "title": f"Session {n}",
                "audio": str(audio),
                "transcript": str(transcript),
                "review": str(tmp_path / f"review{n}.json"),
                "duration": 2,
                "note": "",
                "roster": [f"Voice {i}" for i in range(count)],
            }
        )
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps(entries))
    original = Path(entries[1]["transcript"]).read_bytes()
    server = reviewer.make_server(manifest, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urlopen(base + path) as response:
            return json.load(response)

    def post(state, token, session="36"):
        with urlopen(
            Request(
                base + f"/api/review?session={session}",
                data=json.dumps(state).encode(),
                headers={"X-Review-Token": token},
            )
        ) as response:
            return json.load(response)

    try:
        assert len(get("/api/sessions")) == 2
        data = get("/api/data?session=36")
        assert len(data["roster"]) == 6
        state = data["review"]
        state["reviews"]["0"] = {"verdict": "wrong", "speaker_slot": 5, "note": "overlap"}
        with pytest.raises(HTTPError) as error:
            post(state, "wrong-token")
        assert error.value.code == 403
        result = post(state, data["token"])
        assert result == {"saved": True, "revision": 1}
        assert get("/api/data?session=2")["review"]["reviews"] == {}
        assert get("/api/data?session=36")["review"]["reviews"]["0"]["speaker_slot"] == 5
        with pytest.raises(HTTPError) as error:
            post(state, data["token"])
        assert error.value.code == 409
        with pytest.raises(HTTPError) as error:
            post(state, data["token"], session="2")
        assert error.value.code == 400
        state["revision"] = 1
        state["reviews"]["0"]["speaker_slot"] = 6
        with pytest.raises(HTTPError) as error:
            post(state, data["token"])
        assert error.value.code == 400
        with urlopen(
            Request(base + "/audio?session=36", headers={"Range": "bytes=80-89"})
        ) as response:
            assert response.status == 206
            assert response.read() == bytes(range(80, 90))
            assert response.headers["Content-Range"] == "bytes 80-89/100"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/audio?session=36", headers={"Range": "bytes=100-"}))
        assert error.value.code == 416
        assert Path(entries[1]["transcript"]).read_bytes() == original
        saved = json.loads(Path(entries[1]["review"]).read_text())
        assert saved["revision"] == 1
        assert saved["reviews"]["0"]["speaker_slot"] == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    restarted = reviewer.make_server(manifest, 0)
    restarted.server_close()
    Path(entries[1]["transcript"]).write_text('{"segments": []}')
    with pytest.raises(ValueError, match="does not match"):
        reviewer.make_server(manifest, 0)
