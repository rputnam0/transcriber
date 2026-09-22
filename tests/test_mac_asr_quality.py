from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_context_asr_recordings import alignment_issues, restore_original_words  # noqa: E402
from run_moss_mlx_recordings import parse  # noqa: E402
from fuse_asr_diarization import fuse_words, word_owners  # noqa: E402


def turn(start, end, name, text, ident=0):
    return dict(
        start=start,
        end=end,
        speaker_handle=name,
        text=text,
        turn_id=ident,
        local_speaker="S01",
        cut_id="00000",
        review_reasons=[],
    )


def word(start, end, text):
    return dict(start=start, end=end, text=text, cut_id="00000")


def test_alignment_restores_punctuation_and_contractions():
    aligned = [dict(text="I'm", start=0, end=1), dict(text="warforged", start=1, end=2)]
    assert [w["text"] for w in restore_original_words("I'm war-forged!", aligned)] == [
        "I'm",
        "war-forged!",
    ]
    with pytest.raises(ValueError, match="lexical"):
        restore_original_words("I am warforged!", aligned)


def test_alignment_flags_invalid_and_collapsed_times():
    assert alignment_issues([dict(start=1, end=1), dict(start=-1, end=2)], 1) == [
        "nonmonotonic",
        "out_of_bounds",
        "zero_duration_word",
    ]


def test_compact_parser_keeps_unattributed_cough_and_clamps():
    segments, recovered = parse("[0.2][cough] Good session.[3.0][3.1][S01] Bye.[9.0]", 5)
    assert segments[0]["text"] == "[cough] Good session."
    assert segments[0]["speaker"].startswith("UNTAGGED")
    assert segments[1]["end"] == 5
    assert len(recovered) == 1


def test_same_local_id_can_have_different_human_voice_names():
    turns = [turn(0, 1, "DM", "Good morning", 0), turn(2, 3, "Jesse", "Roll initiative", 1)]
    words = [
        word(0, 0.4, "Good"),
        word(0.5, 0.9, "morning"),
        word(2, 2.4, "Roll"),
        word(2.5, 2.9, "initiative."),
    ]
    result, _ = fuse_words(words, turns)
    assert [r["speaker"] for r in result] == ["DM", "Jesse"]
    assert " ".join(r["text"] for r in result) == "Good morning Roll initiative."


def test_overlap_uses_words_to_distinguish_voices():
    turns = [turn(0, 3, "DM", "You enter the dark cave", 0), turn(1, 2, "Jesse", "Wait stop", 1)]
    words = [
        word(0.1, 0.3, "You"),
        word(0.4, 0.8, "enter"),
        word(1, 1.3, "Wait"),
        word(1.3, 1.7, "stop"),
    ]
    owners, methods = word_owners(words, turns)
    assert owners == [0, 0, 1, 1]
    assert set(methods) == {"lexical_and_time"}


def test_missing_overlap_retained_but_never_duplicate_or_invent_speech():
    turns = [turn(0, 3, "DM", "You enter the dark cave", 0), turn(1, 2, "Jesse", "Wait stop", 1)]
    words = [
        word(0.1, 0.3, "You"),
        word(0.4, 0.8, "enter"),
        word(1, 1.3, "the"),
        word(1.4, 1.8, "dark"),
        word(2, 2.5, "cave"),
    ]
    result, audit = fuse_words(words, turns)
    supplement = [r for r in result if r["speaker"] == "Jesse"]
    assert len(supplement) == 1 and supplement[0]["text"] == "Wait stop"
    assert supplement[0]["review_required"]
    assert audit["qwen_words"] == 5
    result, _ = fuse_words(words + [word(1, 1.1, "Wait"), word(1.2, 1.3, "stop")], turns)
    assert [r["text"] for r in result].count("Wait stop") == 1
    assert sum(len(r.get("words", [])) + len(r.get("alternate_asr_words", [])) for r in result) == 7


def test_no_evidence_stays_unknown():
    result, audit = fuse_words([word(10, 11, "Hello!")], [turn(0, 2, "DM", "Goodbye")])
    assert result[0]["speaker"] == "Unknown"
    assert result[0]["review_required"]
    assert audit["unresolved_words"] == 1


def test_overlap_route_keeps_raw_asr_alternative_without_publishing_twice():
    turns = [turn(0, 4, "DM", "Go into the dark room.", 0), turn(1, 2, "Jesse", "Wait!", 1)]
    recognized = [
        word(0.1, 0.4, "Go"),
        word(0.5, 0.9, "inside"),
        word(1.1, 1.4, "Wait!"),
        word(2.1, 2.4, "the"),
        word(2.5, 3, "room."),
    ]
    result, audit = fuse_words(recognized, turns)
    assert [t["text"] for t in result] == ["Go into the dark room.", "Wait!"]
    assert sum(len(t["alternate_asr_words"]) for t in result) == len(recognized)
    assert audit["overlap_turns_retained"] == 2


def test_foreign_chunk_overlap_is_not_emitted_twice():
    turns = [turn(0, 2, "DM", "Yes.", 0), turn(1, 2, "Jesse", "No!", 1)]
    turns[0]["cut_id"] = turns[1]["cut_id"] = "earlier"
    result, _ = fuse_words([word(1, 1.5, "No!")], turns, core_cut_id="00000")
    assert len(result) == 1 and result[0]["asr_source"] == "qwen3_asr"


def test_small_alignment_gap_requires_one_consistent_voice():
    ending = word(2.5, 2.7, "end.")
    owners, methods = word_owners([ending], [turn(0, 2, "DM", "The end")])
    assert owners == [0] and methods == ["lexical_and_time"]
    owners, methods = word_owners([ending], [turn(0, 2, "DM", "The finish")])
    assert owners == [0] and methods == ["nearby_single_voice"]
    owners, methods = word_owners(
        [ending], [turn(0, 2, "DM", "The finish"), turn(3.4, 4, "Jesse", "Hello")]
    )
    assert owners == [None]


def test_runaway_repetition_is_flagged_without_rejecting_brief_interjections():
    from repair_context_asr import text_issues

    assert text_issues("Yeah. " * 80) == ["runaway_phrase_repeat"]
    assert text_issues("We are going over there. " * 12) == ["runaway_phrase_repeat"]
    assert not text_issues("Yeah, yeah. No, no, wait. I can go there.")


def test_review_audit_does_not_score_identical_simultaneous_phrases(tmp_path):
    import json
    from audit_retranscribed_reviews import audit

    annotations = tmp_path / "reviews.json"
    annotations.write_text(
        json.dumps(
            [
                dict(
                    legacy=False,
                    positive_label_eligible=True,
                    verdict="correct",
                    session=15,
                    start=65.92,
                    end=66.88,
                    text="forty five gold",
                    truth="Leopold Magnus",
                    predicted="Leopold Magnus",
                    turn_id=30,
                )
            ]
        )
    )
    (tmp_path / "Session 15.turns.json").write_text(
        json.dumps(
            dict(
                segments=[
                    dict(start=65.68, end=66.88, text="Forty five gold.", speaker="Kaladen Shash"),
                    dict(start=65.92, end=66.88, text="Forty five gold.", speaker="Leopold Magnus"),
                ]
            )
        )
    )
    result = audit(annotations, tmp_path)
    assert result["comparable"] == 0
    assert result["broken"] == 0
    assert result["records"][0]["competing_speaker_matches"]
