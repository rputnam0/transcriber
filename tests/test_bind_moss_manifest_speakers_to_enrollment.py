from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from bind_moss_manifest_speakers_to_enrollment import validate_mono_output  # noqa: E402


def test_validate_mono_output_rejects_privileged_record_audio() -> None:
    with pytest.raises(ValueError, match="privileged"):
        validate_mono_output(
            {
                "uses_reference_activity": False,
                "uses_isolated_audio": False,
                "records": [{"uses_isolated_audio": True}],
            }
        )


def test_validate_mono_output_accepts_mixed_waveform_records() -> None:
    validate_mono_output(
        {
            "uses_reference_activity": False,
            "uses_isolated_audio": False,
            "records": [{"uses_reference_activity": False, "uses_isolated_audio": False}],
        }
    )
