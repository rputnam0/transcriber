import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from restore_mac_voice_assets import restore  # noqa: E402
from transcribe_mac_recording import build_plan  # noqa: E402


def backup(tmp_path, member="rosters.json"):
    source = tmp_path / "backup"
    source.mkdir()
    payload = b"a tiny stand-in for model weights"
    (source / "part000").write_bytes(payload)
    with zipfile.ZipFile(source / "metadata.zip", "w") as archive:
        archive.writestr(member, "{}")
    manifest = dict(
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        metadata_archive="metadata.zip",
        metadata_sha256=hashlib.sha256((source / "metadata.zip").read_bytes()).hexdigest(),
        parts=[
            dict(name="part000", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        ],
    )
    (source / "backup_manifest.json").write_text(json.dumps(manifest))
    return source, payload


def test_restore_verifies_and_reconstructs_assets(tmp_path):
    source, payload = backup(tmp_path)
    output = tmp_path / "restored"
    restore(source, output)
    assert (output / "moss-checkpoint/model.safetensors").read_bytes() == payload
    assert (output / "rosters.json").exists()
    with pytest.raises(ValueError, match="already has a model"):
        restore(source, output)


def test_restore_rejects_corrupt_part(tmp_path):
    source, payload = backup(tmp_path)
    (source / "part000").write_bytes(b"x" * len(payload))
    output = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore(source, output)
    assert not (output / "moss-checkpoint/model.safetensors").exists()


def test_restore_accepts_checksummed_zip_parts(tmp_path):
    source, payload = backup(tmp_path)
    with zipfile.ZipFile(source / "part000.zip", "w") as archive:
        archive.writestr("part000", payload)
    path = source / "backup_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["parts"][0].update(
        archive_name="part000.zip",
        archive_sha256=hashlib.sha256((source / "part000.zip").read_bytes()).hexdigest(),
    )
    path.write_text(json.dumps(manifest))
    (source / "part000").unlink()
    restore(source, tmp_path / "restored")
    assert (tmp_path / "restored/moss-checkpoint/model.safetensors").read_bytes() == payload


def test_restore_rejects_archive_path_escape(tmp_path):
    source, _ = backup(tmp_path, "../outside.json")
    with pytest.raises(ValueError, match="escapes"):
        restore(source, tmp_path / "restored")
    assert not (tmp_path / "outside.json").exists()


def test_plan_requires_explicit_attendance_and_keeps_whole_recording(tmp_path):
    rosters = tmp_path / "rosters.json"
    rosters.write_text(json.dumps(dict(session_rosters={"70": ["voice_a", "voice_b"]})))
    plan = build_plan(tmp_path / "full.wav", 70, tmp_path / "work", rosters)
    row = plan["recordings"][0]
    assert row["baseline_deployment"] == row["deployment"]
    assert row["audio"].endswith("full.wav")
    with pytest.raises(ValueError, match="roster"):
        build_plan(tmp_path / "full.wav", 71, tmp_path / "work", rosters)
