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
