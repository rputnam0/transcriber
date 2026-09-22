"""Protect absolute timing and uncertainty in full-recording exports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diarize_moss_recording import timestamped_turns  # noqa: E402


def test_overlapping_local_ids_keep_both_names_and_absolute_offsets():
    record = {"start": 60, "duration": 30, "cut_id": "00002"}
    turns = [
        {"start": 2, "end": 5, "speaker": "S01", "text": "long speech"},
        {"start": 3, "end": 3.5, "speaker": "S02", "text": "yes"},
    ]
    bindings = {
        "S01": {"proposed_speaker": "alice", "review_required": False},
        "S02": {"proposed_speaker": "bob", "review_required": False},
    }
    result = timestamped_turns(record, turns, ["alice", "bob"], bindings)
    assert [(t["start"], t["end"], t["speaker"]) for t in result] == [
        (62, 65, "alice"),
        (63, 63.5, "bob"),
    ]
    assert all(t["review_reasons"] == ["overlapping speech"] for t in result)


def test_direct_identity_disagreement_is_visible_without_silently_overriding_it():
    record = {"start": 0, "duration": 30, "cut_id": "00000"}
    turns = [{"start": 0, "end": 1, "speaker": "S02", "text": "hello"}]
    bindings = {"S02": {"proposed_speaker": "alice", "review_required": False}}
    result = timestamped_turns(record, turns, ["bob"], bindings)[0]
    assert result["speaker"] == "bob"
    assert result["review_required"]
    assert result["review_reasons"] == ["identity methods disagree", "chunk boundary"]


def test_readable_paragraphs_keep_interruptions_and_raw_evidence():
    from diarize_moss_recording import readable_paragraphs

    turns = [
        dict(start=0, end=1, speaker="alice", text="one", review_required=False),
        dict(start=1.2, end=2, speaker="alice", text="two", review_required=True),
        dict(start=1.5, end=1.8, speaker="bob", text="yes", review_required=True),
        dict(start=2.2, end=3, speaker="alice", text="three", review_required=False),
    ]
    paragraphs = readable_paragraphs(turns)
    assert [p["text"] for p in paragraphs] == ["one two", "yes", "three"]
    assert paragraphs[0]["review_required"]
    assert paragraphs[1]["speaker"] == "bob"
    assert turns[0]["text"] == "one"
    assert len(turns) == 4


def test_readable_paragraphs_do_not_merge_self_overlap_or_long_silence():
    from diarize_moss_recording import readable_paragraphs

    turns = [
        dict(start=0, end=2, speaker="alice", text="one", review_required=False),
        dict(start=1, end=3, speaker="alice", text="two", review_required=False),
        dict(start=10, end=11, speaker="alice", text="three", review_required=False),
    ]
    assert len(readable_paragraphs(turns)) == 3


def test_worker_restarts_process_all_audio_and_preserve_completed_chunks(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import diarize_moss_recording as runner

    records = [dict(cut_id=str(i), sha256=f"audio-{i}") for i in range(10)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    original = json.dumps({"sha256": "audio-0", "model": "fixed-model", "text": "preserve"})
    (predictions / "0.json").write_text(original)
    commands = []

    def worker(command, **kwargs):
        commands.append(command)
        stop = int(command[command.index("--worker-end") + 1])
        for record in records[:stop]:
            path = predictions / f"{record['cut_id']}.json"
            if not path.exists():
                path.write_text(json.dumps(dict(sha256=record["sha256"], model="fixed-model")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", worker)
    args = SimpleNamespace(
        output=tmp_path,
        model="fixed-model",
        device="mps",
        dtype="float32",
        worker_chunks=3,
        batch_size=4,
        attention="eager",
    )
    runner.infer_recording(args)
    assert len(commands) == 3
    assert len(list(predictions.glob("*.json"))) == 10
    assert json.loads(manifest.read_text()) == records
    assert (predictions / "0.json").read_text() == original


def test_memory_retry_does_not_skip_the_failed_passage(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import diarize_moss_recording as runner

    (tmp_path / "manifest.json").write_text(json.dumps([dict(cut_id="0", sha256="audio")]))
    attempts = []

    def worker(command, **kwargs):
        batch = command[command.index("--batch-size") + 1]
        attempts.append(batch)
        if batch == "4":
            return SimpleNamespace(returncode=86)
        (tmp_path / "predictions").mkdir()
        (tmp_path / "predictions/0.json").write_text(
            json.dumps(dict(sha256="audio", model="fixed"))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", worker)
    args = SimpleNamespace(
        output=tmp_path,
        model="fixed",
        device="mps",
        dtype="float32",
        worker_chunks=32,
        batch_size=4,
        attention="eager",
    )
    runner.infer_recording(args)
    assert attempts == ["4", "1"]


def test_nonprogressing_worker_cannot_report_completion(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    import diarize_moss_recording as runner

    (tmp_path / "manifest.json").write_text(json.dumps([dict(cut_id="0", sha256="audio")]))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    args = SimpleNamespace(
        output=tmp_path,
        model="fixed",
        device="mps",
        dtype="float32",
        worker_chunks=32,
        batch_size=4,
        attention="eager",
    )
    with pytest.raises(RuntimeError, match="no progress"):
        runner.infer_recording(args)
