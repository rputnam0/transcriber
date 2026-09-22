import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "audit_single", Path(__file__).parents[1] / "scripts/audit_single_file_sessions.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_full_recording_including_last_partial_chunk():
    records = [{"start": 0, "duration": 30}, {"start": 30, "duration": 2.75}]
    assert module.check_coverage(records, 32.75) == 32.75


@pytest.mark.parametrize(
    "records,duration",
    [
        ([{"start": 0, "duration": 30}], 3600),
        ([{"start": 30, "duration": 30}], 60),
        ([{"start": 0, "duration": 10}, {"start": 11, "duration": 10}], 21),
        ([{"start": 0, "duration": 10}, {"start": 9, "duration": 10}], 19),
        ([], 0),
    ],
)
def test_reject_truncation_gaps_overlaps_and_empty_input(records, duration):
    with pytest.raises(ValueError):
        module.check_coverage(records, duration)
