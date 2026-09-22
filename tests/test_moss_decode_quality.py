import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from moss_batch_inference import decode_issues  # noqa: E402


def test_runaway_vocalization_cannot_pass_as_complete_transcription():
    raw = "[0.48][S03] The next turn.[2.40][9.28][S02] A" + "h" * 4000
    assert decode_issues(raw) == ["runaway_character_repeat", "unfinished_turn"]


def test_abrupt_generation_limit_is_detected_even_without_repetition():
    assert decode_issues("[1.2][S01] Here is an unfinished response") == ["unfinished_turn"]


def test_short_vocalization_and_genuine_repeated_words_remain_valid():
    assert decode_issues("[1.2][S01] Ahhhh. No no no no![3.4]") == []
    assert decode_issues("") == []


def test_speaker_bracket_repair_preserves_the_words_and_timestamps():
    from moss_batch_inference import repair_speaker_brackets

    raw = "[15.28][S02} Right behind here.[16.88][17.12][S02] Flanking?[18.48]"
    fixed = repair_speaker_brackets(raw)
    assert fixed == "[15.28][S02] Right behind here.[16.88][17.12][S02] Flanking?[18.48]"
    assert repair_speaker_brackets("Use {S02} in your notes.") == "Use {S02} in your notes."


def test_missing_identity_does_not_drop_later_words_or_inherit_a_voice():
    from types import SimpleNamespace
    from moss_batch_inference import recover_compact_segments

    raw = "[1][S01] Before.[2][3] Missing identity.[4][5][S02] After.[6]"
    before = SimpleNamespace(start=1.0, end=2.0, speaker="S01", text="Before.")
    parsed, recovered = recover_compact_segments(raw, [before])
    assert [p.text for p in parsed] == ["Before.", "Missing identity.", "After."]
    assert recovered[0]["speaker"].startswith("UNTAGGED_")
    assert recovered[1]["speaker"] == "S02"
    assert [(p.start, p.end) for p in parsed] == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert parsed[0] is before


def test_recovery_does_not_split_bracketed_numbers_inside_parsed_speech():
    from types import SimpleNamespace
    from moss_batch_inference import recover_compact_segments

    original = SimpleNamespace(start=1.0, end=20.0, speaker="S01", text="I rolled [12] yesterday.")
    parsed, recovered = recover_compact_segments("[1][S01]I rolled [12] yesterday.[20]", [original])
    assert parsed == [original]
    assert recovered == []


def test_cough_marker_does_not_drop_bounded_speech_without_a_speaker_id():
    from moss_batch_inference import recover_compact_segments

    raw = "[1.20][cough] Oh, good session.[3.68][3.68][S03] Sweet.[4.72]"
    parsed, recovered = recover_compact_segments(raw, [])
    assert [(p.start, p.end, p.text) for p in parsed] == [
        (1.2, 3.68, "[cough] Oh, good session."),
        (3.68, 4.72, "Sweet."),
    ]
    assert recovered[0]["speaker"].startswith("UNTAGGED_")
    assert recovered[1]["speaker"] == "S03"
    assert recover_compact_segments(raw, parsed) == (parsed, [])
