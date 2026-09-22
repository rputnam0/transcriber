from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from select_balanced_mono_eval_cuts import (  # noqa: E402
    evenly_spaced_indices,
    select_balanced,
    select_contiguous,
)


def test_evenly_spaced_indices_include_both_ends() -> None:
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]


def test_select_balanced_takes_equal_count_per_session() -> None:
    cuts = [
        SimpleNamespace(id=f"session_{session}_w{index:02d}")
        for session in (64, 67)
        for index in range(5)
    ]

    selected = select_balanced(cuts, cuts_per_session=3)

    assert [cut.id for cut in selected] == [
        "session_64_w00",
        "session_64_w02",
        "session_64_w04",
        "session_67_w00",
        "session_67_w02",
        "session_67_w04",
    ]


def test_select_contiguous_chooses_longest_run_per_session() -> None:
    cuts = [
        SimpleNamespace(id="session_64_w000100_c000000"),
        SimpleNamespace(id="session_64_w000100_c030000"),
        SimpleNamespace(id="session_64_w000100_c090000"),
        SimpleNamespace(id="session_64_w000200_c000000"),
        SimpleNamespace(id="session_64_w000200_c030000"),
        SimpleNamespace(id="session_64_w000200_c060000"),
    ]

    selected = select_contiguous(cuts, cuts_per_session=2)

    assert [cut.id for cut in selected] == [
        "session_64_w000200_c000000",
        "session_64_w000200_c030000",
    ]


def test_select_contiguous_excludes_previously_scored_cuts() -> None:
    cuts = [
        SimpleNamespace(id=f"session_64_w{window:06d}_c{offset:06d}")
        for window in (100, 200)
        for offset in (0, 30_000, 60_000)
    ]
    excluded = {cut.id for cut in cuts if "w000100" in cut.id}

    selected = select_contiguous(
        cuts,
        cuts_per_session=2,
        excluded_cut_ids=excluded,
    )

    assert [cut.id for cut in selected] == [
        "session_64_w000200_c000000",
        "session_64_w000200_c030000",
    ]
