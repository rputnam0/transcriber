from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from calibrate_speaker_binding_confidence import (  # noqa: E402
    assigned_margin_rows,
    calibrate_threshold,
)


def test_calibration_uses_assigned_margin_and_maximizes_eligible_coverage() -> None:
    payload = {
        "records": [
            {
                "cut_id": "cut",
                "oracle_mapping_diagnostic_only": {
                    "speaker_0": "Alice",
                    "speaker_1": "Bob",
                },
                "one_to_one_mapping": {
                    "speaker_0": "Alice",
                    "speaker_1": "Carol",
                },
                "binding_evidence": {
                    "speaker_0": {"scores": {"Alice": 0.8, "Bob": 0.4}},
                    "speaker_1": {"scores": {"Carol": 0.6, "Bob": 0.55}},
                },
            }
        ]
    }

    rows = assigned_margin_rows([payload], mode="one_to_one")
    calibrated = calibrate_threshold(rows, target_precision=0.95, threshold_step=0.1)

    assert calibrated["selected"]["threshold"] == 0.1
    assert calibrated["selected"]["accepted"] == 1
    assert calibrated["selected"]["precision"] == 1.0
