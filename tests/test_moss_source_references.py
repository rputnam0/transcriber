"""Keep speech-crop reference recovery aligned and prevent seam duplicates."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repair_moss_source_references import (
    crop_words,
    speech_crops,
    missing_speech_crops,
)  # noqa: E402


def test_short_speech_after_long_silence_gets_its_own_crop():
    mask = np.zeros(1900, bool)
    mask[500:520] = True
    mask[1400:1430] = True
    crops = speech_crops(mask, 60)
    assert len(crops) == 2
    assert crops[0]["audio_start"] < 16 < crops[0]["audio_end"]
    assert crops[1]["audio_start"] < 44.8 < crops[1]["audio_end"]
    assert all(c["audio_end"] - c["audio_start"] < 2 for c in crops)


def test_context_overlap_has_exactly_one_owner_at_core_boundary():
    crops = speech_crops(np.ones(1000, bool), 30)
    seam = crops[0]["core_end"]
    recovered = []
    for crop in crops[:2]:
        local = seam - crop["audio_start"]
        recovered.extend(
            crop_words(
                [{"words": [{"start": local - 0.1, "end": local + 0.1, "word": "hello"}]}], crop
            )
        )
    assert len(recovered) == 1
    assert recovered[0]["words"][0]["start"] == seam - 0.1


def test_recovery_keeps_existing_speech_and_supplies_context_for_missing_turn():
    mask = np.zeros(1900, bool)
    mask[500:520] = True
    mask[1400:1430] = True
    original = [{"words": [{"start": 16.1, "end": 16.5, "word": "known"}]}]
    crops = missing_speech_crops(mask, 60, original)
    assert len(crops) == 1
    assert crops[0]["core_start"] > 40
    assert crops[0]["audio_end"] - crops[0]["audio_start"] >= 5
