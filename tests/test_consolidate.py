from __future__ import annotations

from pathlib import Path

import pytest

from transcriber.consolidate import choose_speaker, save_outputs


def test_choose_speaker_ignores_numeric_only_fuzzy_matches():
    mapping = {
        "autbot_80hd_55561_0": "B. Ver",
        "travisaurus6985_0": "Cyrus Schwert",
        "kinglizard7958_0": "Leopold Magnus",
    }

    assert choose_speaker("1.flac", mapping, return_match=True) == ("1", False)
    assert choose_speaker("9.flac", mapping, return_match=True) == ("9", False)


def test_choose_speaker_preserves_alpha_stem_fuzzy_matching():
    mapping = {
        "kinglizard7958_0": "Leopold Magnus",
    }

    assert choose_speaker("kinglizard7958.flac", mapping, return_match=True) == (
        "Leopold Magnus",
        True,
    )
    assert choose_speaker("6-kinglizard7958.ogg", mapping, return_match=True) == (
        "Leopold Magnus",
        True,
    )


def test_save_outputs_raises_when_output_dir_is_not_writable(monkeypatch, tmp_path):
    blocked_dir = tmp_path / "blocked"
    original_mkdir = Path.mkdir

    def fake_mkdir(self: Path, *args, **kwargs):
        if self == blocked_dir:
            raise PermissionError("drive mount unavailable")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    with pytest.raises(PermissionError):
        save_outputs(
            base_stem="Session 77",
            output_dir=str(blocked_dir),
            per_file_segments=[("input.wav", [{"start": 0.0, "end": 1.0, "text": "hello"}])],
            consolidated_pairs=[("00:00:00", "Speaker", "hello")],
            diar_by_file=None,
            exclusive_diar_by_file=None,
            write_srt_file=False,
            write_jsonl_file=False,
        )


def timed_segment(text, start=0.0, speaker="Alice"):
    words = [
        {"word": word, "start": start + i * 0.5, "end": start + i * 0.5 + 0.25}
        for i, word in enumerate(text.split())
    ]
    return {
        "text": text,
        "start": start,
        "end": words[-1]["end"],
        "speaker": speaker,
        "words": words,
    }


def test_sentence_and_pause_splits_preserve_words_and_metadata():
    from transcriber.consolidate import split_segments

    segment = timed_segment('Yeah. Okay. We go now then stop! "Really?"')
    # A long pause inside an unfinished sentence.
    for word in segment["words"][5:]:
        word["start"] += 2
        word["end"] += 2
    segment["speaker_raw"] = "track_1"
    result = split_segments([segment], pause_threshold=1.0)
    assert [s["text"] for s in result] == ["Yeah.", "Okay.", "We go now", "then stop!", '"Really?"']
    assert [w for s in result for w in s["words"]] == segment["words"]
    for s in result:
        assert s["start"] == s["words"][0]["start"]
        assert s["end"] == s["words"][-1]["end"]
        assert s["speaker"] == "Alice"
        assert s["speaker_raw"] == "track_1"
    assert segment["text"] == 'Yeah. Okay. We go now then stop! "Really?"'


def test_multitrack_interleave_and_legacy_mode():
    from transcriber.consolidate import consolidate, split_multitrack_segments

    tracks = [
        ("alice.wav", [timed_segment("Hello. Next sentence.")]),
        ("bob.wav", [timed_segment("Okay. Yes.", start=0.25, speaker="Bob")]),
    ]
    assert [s for _, s, _ in consolidate(tracks)] == ["Alice", "Bob", "Alice", "Bob"]
    assert [t for _, _, t in consolidate(tracks)] == ["Hello.", "Okay.", "Next sentence.", "Yes."]
    assert len(consolidate(tracks, sentence_level=False)) == 2
    assert split_multitrack_segments(tracks[:1]) == tracks[:1]


def test_untimed_or_incomplete_words_keep_text():
    from transcriber.consolidate import split_segments

    incomplete = timed_segment("Hello. More text.")
    incomplete["words"] = incomplete["words"][:1]
    untimed = {"text": "Yeah. Okay.", "start": 1, "end": 3}
    assert split_segments([incomplete, untimed]) == [incomplete, untimed]


def test_punctuation_only_fragments_merge_but_short_replies_survive():
    from transcriber.consolidate import split_segments

    result = split_segments([timed_segment("Yes. ... Okay. Dr. Smith agrees.")])
    assert [s["text"] for s in result] == ["Yes. ...", "Okay.", "Dr. Smith agrees."]


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_invalid_pause_threshold(threshold):
    from transcriber.consolidate import split_segments

    with pytest.raises(ValueError, match="pause_threshold"):
        split_segments([], threshold)


def test_roster_reports_unknown_names_without_changing_output(caplog):
    from transcriber.consolidate import check_speaker_roster

    pairs = [("00:00:00", "Kaladin Shash", "Yeah.")]
    assert check_speaker_roster(pairs, ["Kaladen Shash"]) == ["Kaladin Shash"]
    assert "Kaladin Shash" in caplog.text
    assert pairs == [("00:00:00", "Kaladin Shash", "Yeah.")]
    assert check_speaker_roster(pairs, None) == []
    assert check_speaker_roster(pairs, ["Kaladin Shash"]) == []


def test_all_exports_use_split_records(tmp_path):
    import json
    from transcriber.consolidate import consolidate, split_multitrack_segments

    tracks = [
        ("alice.wav", [timed_segment("Yeah. Okay.")]),
        ("bob.wav", [timed_segment("Hello.", speaker="Bob")]),
    ]
    records = split_multitrack_segments(tracks)
    pairs = consolidate(records, sentence_level=False)
    save_outputs("Session 4", str(tmp_path), records, pairs, None)
    jsonl = [json.loads(line) for line in (tmp_path / "Session 4.jsonl").read_text().splitlines()]
    assert len(jsonl) == len(pairs) == 3
    assert [r["start"] for r in jsonl] == sorted(r["start"] for r in jsonl)
    assert "Alice: Okay." in (tmp_path / "Session 4.srt").read_text()
    assert [w for r in jsonl if r["speaker"] == "Alice" for w in r["words"]] == tracks[0][1][0][
        "words"
    ]
    assert "00:00:00,500 --> 00:00:00,750" in (tmp_path / "Session 4.srt").read_text()


def test_shipped_speaker_mapping_uses_canonical_names():
    import yaml

    root = Path(__file__).resolve().parents[1]
    mapping = yaml.safe_load((root / "config/speaker_mapping.example.yaml").read_text())
    assert choose_speaker("3-bfschmity_0.ogg", mapping) == "Kaladen Shash"
    assert choose_speaker("traceritops_0.ogg", mapping) == "Cletus Cobbington"
