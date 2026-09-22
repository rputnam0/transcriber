from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_persistent_mono_named_pipeline import (  # noqa: E402
    cut_binding_maps,
    retarget_enrollment_manifest,
    write_empty_cut_result,
)


def test_retarget_enrollment_manifest_deduplicates_speakers_without_eval_audio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    alice = tmp_path / "alice.wav"
    bob = tmp_path / "bob.wav"
    sf.write(alice, np.zeros(80), 80)
    sf.write(bob, np.zeros(80), 80)
    rows = [
        {
            "session": "Session 64",
            "speaker_id": "Alice",
            "materialized": {"positive_enrollment_paths": [str(alice)]},
            "uses_evaluation_session_audio": False,
        },
        {
            "session": "Session 67",
            "speaker_id": "Alice",
            "materialized": {"positive_enrollment_paths": [str(alice)]},
            "uses_evaluation_session_audio": False,
        },
        {
            "session": "Session 64",
            "speaker_id": "Bob",
            "materialized": {"positive_enrollment_paths": [str(bob)]},
            "uses_evaluation_session_audio": False,
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "runtime.jsonl"

    summary = retarget_enrollment_manifest(source, output, session="Session 0")
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["speaker_count"] == 2
    assert {row["session"] for row in written} == {"Session 0"}
    assert all(not row["uses_evaluation_session_audio"] for row in written)
    assert all(
        Path(row["materialized"]["positive_enrollment_paths"][0]).is_absolute() for row in written
    )


def test_retarget_enrollment_manifest_rejects_same_session_profiles(tmp_path: Path) -> None:
    enrollment = tmp_path / "alice.wav"
    sf.write(enrollment, np.zeros(80), 80)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "session": "Session 0",
                "speaker_id": "Alice",
                "materialized": {"positive_enrollment_paths": [str(enrollment)]},
                "uses_evaluation_session_audio": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        retarget_enrollment_manifest(source, tmp_path / "runtime.jsonl", session="Session 0")
    except ValueError as error:
        assert "cross-session" in str(error)
    else:
        raise AssertionError("Expected same-session enrollment to be rejected")


def test_empty_activity_mapping_writes_renderable_silent_result(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "records": [
                    {"cut_id": "cut-silent", "one_to_one_mapping": {}},
                    {
                        "cut_id": "cut-speech",
                        "one_to_one_mapping": {"speaker_0": "Alice"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    mappings = cut_binding_maps(binding, mode="one_to_one")
    output = tmp_path / "silent.json"
    write_empty_cut_result(output, cut_id="cut-silent", session="Session 0")

    assert mappings["cut-silent"] == {}
    assert mappings["cut-speech"] == {"speaker_0": "Alice"}
    assert json.loads(output.read_text(encoding="utf-8"))["records"] == []
