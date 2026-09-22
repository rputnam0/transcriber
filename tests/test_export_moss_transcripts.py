import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "export_moss", Path(__file__).parents[1] / "scripts/export_moss_transcripts.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_valid_name_unchanged_even_when_other_scores_win():
    handle, reason, _ = module.resolve_name(
        {"speaker": "DM"}, {"probabilities": {"absent": 1.0}}, ["DM", "player"]
    )
    assert handle == "DM"
    assert reason == "original valid-roster assignment"


def test_absent_voice_without_evidence_remains_unconfirmed():
    assert module.resolve_name({"speaker": "absent"}, {}, ["DM", "player"])[0] is None


def test_roster_prior_cannot_override_poor_voice_match():
    binding = {
        "probabilities": {"DM": 0.1, "player": 0.01, "absent": 0.89},
        "cosines": {"DM": 0.12},
    }
    assert module.resolve_name({"speaker": "absent"}, binding, ["DM", "player"])[0] is None
    binding["cosines"]["DM"] = 0.7
    name, reason, evidence = module.resolve_name({"speaker": "absent"}, binding, ["DM", "player"])
    assert name == "DM"
    assert "tentative" in reason
    assert evidence["original_posterior"] == 0.1


def test_paragraphs_preserve_interruptions_and_distinct_unknowns():
    def turn(i, handle, start, end, cluster):
        return {
            "turn_id": i,
            "speaker_handle": handle,
            "speaker": handle or "Unconfirmed speaker",
            "start": start,
            "end": end,
            "cut_id": "00000",
            "local_speaker": cluster,
            "text": str(i),
            "roster_review_required": handle is None,
        }

    raw = [
        turn(0, "DM", 0, 2, "S01"),
        turn(1, None, 1, 1.5, "S02"),
        turn(2, None, 1.6, 1.9, "S03"),
        turn(3, "DM", 2, 3, "S01"),
        turn(4, "DM", 3.1, 4, "S01"),
    ]
    rows = module.paragraphs(raw)
    assert [r["text"] for r in rows] == ["0", "1", "2", "3 4"]
    assert [r["text"] for r in raw] == ["0", "1", "2", "3", "4"]


def test_html_escapes_transcript_and_speaker():
    row = {
        "turn_id": 0,
        "speaker": "<evil>",
        "start": 1,
        "text": "<script>bad()</script>",
        "roster_review_required": True,
    }
    output = module.reader("Test", [row], "draft")
    assert "&lt;script&gt;bad()&lt;/script&gt;" in output
    assert "<evil>" not in output
    assert "check speaker" in output
