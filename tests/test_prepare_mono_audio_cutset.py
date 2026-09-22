from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_mono_audio_cutset import mono_cut_id, padded_duration  # noqa: E402


def test_padded_duration_rounds_up_to_complete_model_chunks() -> None:
    assert padded_duration(61.2, chunk_seconds=30.0) == 90.0
    assert padded_duration(60.0, chunk_seconds=30.0) == 60.0


def test_mono_cut_id_matches_persistent_sortformer_grouping_contract() -> None:
    assert mono_cut_id(0, 60.0) == "session_0_w000000_c060000"
