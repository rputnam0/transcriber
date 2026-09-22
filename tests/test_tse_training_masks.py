from __future__ import annotations

import sys
import random
from pathlib import Path

import torch
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_usef_tse_domain_adapter import (  # noqa: E402
    ForcedReferenceIndex,
    _activity_loss_components,
    _forced_reference_coverage,
    _interferer_only_mask,
    _overlap_mask,
    _overlap_spans,
    _sample_chunk_start,
    _validate_forced_reference_coverage,
)


def test_overlap_sampler_centers_chunk_on_forced_speaker_overlap(tmp_path):
    reference_path = tmp_path / "forced.jsonl"
    reference_path.write_text(
        (
            '{"session":"Session 1","window_start":0.0,"window_end":20.0,'
            '"words":[{"speaker":"Alice","start":10.0,"end":11.0,"text":"hello"},'
            '{"speaker":"Bob","start":10.5,"end":11.5,"text":"there"}]}\n'
        ),
        encoding="utf-8",
    )
    forced = ForcedReferenceIndex(reference_path)
    row = {
        "session": "Session 1",
        "window_start": 0.0,
        "window_end": 20.0,
        "duration": 20.0,
        "speaker_id": "Alice",
    }

    overlaps = _overlap_spans(row, forced_references=forced)
    chunk_start = _sample_chunk_start(
        row,
        rng=random.Random(7),
        chunk_seconds=4.0,
        active_probability=1.0,
        overlap_probability=1.0,
        forced_references=forced,
    )

    assert overlaps == [{"start": 10.5, "end": 11.0}]
    assert chunk_start <= 10.75 <= chunk_start + 4.0


def test_interferer_only_mask_excludes_target_active_overlap():
    target_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    raw_non_owner_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    interferer_only = _interferer_only_mask(target_mask, raw_non_owner_mask)

    assert torch.equal(interferer_only, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    assert torch.all(interferer_only[target_mask.bool()] == 0.0)
    assert torch.equal(
        _overlap_mask(target_mask, raw_non_owner_mask),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )


def test_activity_loss_components_report_overlap_without_suppressing_it():
    estimate = torch.tensor([[0.0, 0.5, 0.5, 0.0]])
    target = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    target_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    raw_non_owner_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    components = _activity_loss_components(
        estimate,
        target,
        target_mask,
        raw_non_owner_mask,
        active_weight=4.0,
        silence_weight=2.0,
        non_owner_weight=3.0,
        overlap_recon_weight=0.5,
    )

    assert components["loss"].item() > 0.0
    assert components["overlap_fraction"].item() == 0.25
    assert components["interferer_only_fraction"].item() == 0.5


def test_forced_reference_coverage_gate_fails_missing_target_rows(tmp_path):
    reference_path = tmp_path / "forced.jsonl"
    reference_path.write_text(
        (
            '{"session":"Session 1","window_start":0.0,"window_end":10.0,'
            '"words":[{"speaker":"Alice","start":0.0,"end":0.3,"text":"hello"}]}\n'
        ),
        encoding="utf-8",
    )
    forced = ForcedReferenceIndex(reference_path)
    rows = [
        {
            "row_id": "has-ref",
            "session": "Session 1",
            "window_start": 0.0,
            "window_end": 10.0,
            "speaker_id": "Alice",
            "target_word_count": 1,
        },
        {
            "row_id": "missing-ref",
            "session": "Session 1",
            "window_start": 0.0,
            "window_end": 10.0,
            "speaker_id": "Bob",
            "target_word_count": 1,
        },
    ]

    coverage = _forced_reference_coverage(rows, forced)

    assert coverage["row_coverage"] == 0.5
    assert coverage["forced_target_word_coverage"] == 0.5
    assert coverage["missing_row_ids"] == ["missing-ref"]
    with pytest.raises(ValueError, match="forced-reference coverage gate failed"):
        _validate_forced_reference_coverage(
            label="train",
            coverage=coverage,
            require_forced_reference=True,
            min_row_coverage=0.98,
            min_word_coverage=0.98,
        )
