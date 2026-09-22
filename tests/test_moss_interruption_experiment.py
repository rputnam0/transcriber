import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_moss_interruption_experiment import identity_only, mix, schedule
from prepare_moss_mac_training import authored_record
from train_moss_diarization import load_samples, loss_weights_from_offsets


def item(name, duration):
    return dict(
        session=43,
        window_start=1.0,
        reference_file="source.json",
        segment=dict(
            start=0.2,
            end=0.2 + duration,
            speaker=name,
            words=["hello"],
            word_spans=[dict(start=0.2, end=0.2 + duration, text="hello", speaker=name)],
        ),
    )


def test_synthetic_overlap_keeps_both_owners_and_shifted_word_times():
    a, b = item("bfschmity", 3), item("jessev567890", 0.5)
    placements = schedule("overlap", b, a, [], random.Random(1), 0.5)
    banks = {
        43: {name: np.full(16000 * 10, 0.1, np.float32) for name in ["bfschmity", "jessev567890"]}
    }
    wave, segments, sources = mix(placements, banks, 0.05)
    assert wave.ndim == 1 and np.isfinite(wave).all()
    assert [s["speaker"] for s in segments] == ["bfschmity", "jessev567890"]
    assert segments[0]["end"] - segments[1]["start"] == pytest.approx(0.25)
    assert all(s["word_spans"][0]["start"] == s["start"] for s in segments)
    assert len(sources) == 2


def test_rapid_switches_preserve_four_alternating_utterances():
    a, b = item("bfschmity", 0.5), item("jessev567890", 0.3)
    placements = schedule("rapid_switches", a, b, [a, b, a, b], random.Random(3), 0)
    assert [p[0]["segment"]["speaker"] for p in placements] == ["bfschmity", "jessev567890"] * 2
    for (previous, onset, _), (_, next_onset, _) in zip(placements, placements[1:]):
        gap = next_onset - onset - (previous["segment"]["end"] - previous["segment"]["start"])
        assert -1e-8 <= gap <= 0.200001


def test_clear_control_is_single_speaker_and_uncut():
    a, b = item("bfschmity", 3), item("jessev567890", 0.5)
    placements = schedule("clear", b, a, [], random.Random(1), 0)
    assert placements == [(a, 0.2, 0.0)]


def test_identity_only_loss_masks_words_timestamps_whitespace_and_eos(tmp_path):
    record = identity_only(
        authored_record(
            "test", tmp_path / "audio.wav", [item("bfschmity", 3)["segment"]], "43", "identity"
        )
    )
    target = record["conversation"][-1]["content"]
    weights = loss_weights_from_offsets(
        [(i, i + 1) for i in range(len(target))], record["metadata"]["loss_spans"]
    )
    assert "".join(c for c, w in zip(target, weights) if w) == "[S01]"
    assert record["metadata"]["eos_loss_weight"] == 0
    assert not record["metadata"]["activity_supervised"]


def test_unsure_label_is_excluded_even_if_identity_is_present(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    good = authored_record("clear", audio, [item("bfschmity", 3)["segment"]], "43", "identity")
    unsure = json.loads(json.dumps(good))
    unsure["metadata"].update(verdict="unsure", corrected_speaker="bfschmity")
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps(unsure) + "\n" + json.dumps(good) + "\n")
    assert len(load_samples(str(manifest), forbidden_sessions=set())) == 1


def test_secondary_stem_cannot_cross_train_dev_boundary(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    row = authored_record("mixture", audio, [item("bfschmity", 3)["segment"]], "43", "identity")
    row["metadata"]["source_sessions"] = [43, 39]
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="forbidden holdout"):
        load_samples(str(manifest), forbidden_sessions={"39"})


def test_decoding_metric_does_not_permute_wrong_identities_or_ignore_unknowns():
    from evaluate_moss_interruption_decode import activity_counts

    reference = [dict(start=0, end=1, speaker="S01")]
    assert activity_counts(reference, reference, 1) == dict(tp=50, fp=0, fn=0)
    for wrong in ["S02", "S99"]:
        assert activity_counts(reference, [dict(start=0, end=1, speaker=wrong)], 1) == dict(
            tp=0, fp=50, fn=50
        )


def test_decoding_metric_keeps_overlapping_reference_speakers():
    from evaluate_moss_interruption_decode import activity_counts

    reference = [dict(start=0, end=1, speaker="S01"), dict(start=0.5, end=1, speaker="S02")]
    assert activity_counts(reference, reference[:1], 1) == dict(tp=50, fp=0, fn=25)
