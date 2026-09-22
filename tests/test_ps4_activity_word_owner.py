from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_ps4_activity_word_owner import (  # noqa: E402
    EnrollmentSelector,
    LauraActivityHead,
    PS4ActivityHead,
    PS4FeatureExtractor,
    _frame_loss_and_metrics,
    _sample_group_chunk_start,
    _subset_batch_candidates,
    _word_owner_loss,
)


class _StaticStemCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def wav_for_member(self, row, member):
        return self.path


class _DummyPS4(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.separator = torch.nn.Module()
        self.separator.separation = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        )
        for parameter in self.parameters():
            parameter.requires_grad = False


def test_activity_head_preserves_candidate_and_frame_axes():
    head = PS4ActivityHead(channels=16, layers=2, dropout=0.0)

    logits = head(torch.randn(3, 32, 128, 17))

    assert logits.shape == (3, 17)


def test_laura_activity_head_reduces_frequency_without_losing_frames():
    head = LauraActivityHead(channels=16, layers=2, dropout=0.0)

    logits = head(torch.randn(2, 256, 17, 11))

    assert logits.shape == (2, 17)


def test_feature_extractor_unfreezes_only_requested_tail_blocks():
    extractor = PS4FeatureExtractor(_DummyPS4(), trainable_separator_blocks=1)

    trainable = [name for name, value in extractor.model.named_parameters() if value.requires_grad]
    extractor.set_trainable_mode(True)

    assert trainable == ["separator.separation.2.weight", "separator.separation.2.bias"]
    assert sorted(extractor.trainable_state_dict()) == sorted(trainable)
    assert not extractor.model.separator.separation[1].training
    assert extractor.model.separator.separation[2].training


def test_group_sampler_centers_on_cross_speaker_overlap():
    words = [
        {"speaker": "Alice", "start": 10.0, "end": 11.0},
        {"speaker": "Bob", "start": 10.5, "end": 11.5},
    ]

    start = _sample_group_chunk_start(
        words,
        duration=20.0,
        chunk_seconds=4.0,
        overlap_probability=1.0,
        active_probability=1.0,
        rng=random.Random(7),
    )

    assert start <= 10.75 <= start + 4.0


def test_candidate_subset_prefers_speakers_active_in_chunk():
    batch = {
        "mixture": torch.zeros(3, 8),
        "enrollments": torch.zeros(3, 4),
        "target_sources": torch.zeros(3, 8),
        "target_masks": torch.zeros(3, 8),
        "speakers": ["Alice", "Bob", "Carol"],
        "words": [
            {"speaker": "Alice", "start": 1.0, "end": 1.2},
            {"speaker": "Carol", "start": 2.0, "end": 2.2},
        ],
        "chunk_start": 0.0,
        "chunk_seconds": 4.0,
    }

    selected = _subset_batch_candidates(batch, limit=2, rng=random.Random(7))

    assert selected["speakers"] == ["Alice", "Carol"]
    assert selected["mixture"].shape[0] == 2
    assert selected["target_sources"].shape[0] == 2


def test_enrollment_selector_avoids_silent_region_and_caches_choice(tmp_path):
    sample_rate = 8000
    wave = np.zeros(sample_rate * 30, dtype=np.float32)
    wave[sample_rate * 20 :] = 0.2
    path = tmp_path / "target.wav"
    sf.write(path, wave, sample_rate)
    selector = EnrollmentSelector(seconds=10.0, sample_rate=sample_rate, min_rms=0.01)
    row = {
        "row_id": "row-1",
        "target_member": "target.wav",
        "positive_enrollment_spans": [{"start": 0.0, "end": 30.0, "duration": 30.0}],
    }

    selected = selector.load(row, cache=_StaticStemCache(path))
    selected_again = selector.load(row, cache=_StaticStemCache(path))

    assert torch.allclose(selected, selected_again)
    assert torch.sqrt(torch.mean(selected.square())).item() > 0.19


def test_word_owner_loss_compares_all_enrolled_candidates():
    logits = torch.tensor(
        [
            [5.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0],
        ]
    )
    words = [
        {"speaker": "Alice", "start": 0.1, "end": 0.9},
        {"speaker": "Bob", "start": 2.1, "end": 2.9},
    ]

    loss, metrics = _word_owner_loss(
        logits,
        words=words,
        speakers=["Alice", "Bob"],
        chunk_start=0.0,
        chunk_seconds=4.0,
        frame_hop_seconds=1.0,
    )

    assert loss.item() < 0.02
    assert metrics["words"] == 2
    assert metrics["correct"] == 2


def test_frame_activity_loss_keeps_simultaneous_speakers_positive():
    logits = torch.tensor([[5.0, 5.0], [5.0, -5.0]])
    masks = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    loss, metrics = _frame_loss_and_metrics(logits, masks)

    assert loss.item() < 0.02
    assert metrics == {"tp": 3, "fp": 0, "fn": 0}
