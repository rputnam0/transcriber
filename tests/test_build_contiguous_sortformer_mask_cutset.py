from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_contiguous_sortformer_mask_cutset import (  # noqa: E402
    group_contiguous_cuts,
    split_probabilities,
)


def test_group_contiguous_cuts_splits_gaps_and_windows() -> None:
    cuts = [
        SimpleNamespace(id="session_64_w000100_c000000", duration=30.0),
        SimpleNamespace(id="session_64_w000100_c030000", duration=30.0),
        SimpleNamespace(id="session_64_w000100_c090000", duration=30.0),
        SimpleNamespace(id="session_64_w000200_c000000", duration=30.0),
    ]

    runs = group_contiguous_cuts(cuts)

    assert [[cut.id for cut in run] for run in runs] == [
        ["session_64_w000100_c000000", "session_64_w000100_c030000"],
        ["session_64_w000100_c090000"],
        ["session_64_w000200_c000000"],
    ]


def test_split_probabilities_preserves_all_frames_proportionally() -> None:
    probabilities = np.arange(24, dtype=np.float32).reshape(12, 2)

    chunks = split_probabilities(probabilities, [10, 20, 10])

    assert [len(chunk) for chunk in chunks] == [3, 6, 3]
    assert np.array_equal(np.concatenate(chunks), probabilities)
