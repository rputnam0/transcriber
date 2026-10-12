import importlib.util
import json
from pathlib import Path


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load_script("score_multitrack_asr")
prepare = load_script("prepare_multitrack_asr_study")


def test_edits_report_missing_speech_and_empty_reference_insertions():
    assert score.edit_counts(["quiet", "speech"], ["speech"]) == dict(
        deletions=1, insertions=0, substitutions=0
    )
    assert score.edit_counts([], ["thank", "you"]) == dict(
        deletions=0, insertions=2, substitutions=0
    )
    assert score.edit_counts(["one"], ["two"]) == dict(deletions=0, insertions=0, substitutions=1)


def test_silent_audio_overrides_hallucinated_reference(tmp_path):
    (tmp_path / "silence.json").write_text(json.dumps({"text": "Get a good camera."}))
    manifest = [
        dict(
            cut_id="silence",
            speaker="A",
            start=0,
            duration=60,
            digital_silence=True,
            stratum="silence",
        )
    ]
    candidates = {"candidate": [dict(speaker="A", start=0, end=1, word="Thank you.")]}
    result = score.score(manifest, tmp_path, candidates)["totals"]["candidate / ALL"]
    assert result["reference_words"] == 0
    assert result["silent_words"] == 2
    assert result["proxy_wer"] is None


def test_repetition_flag_and_fractional_window_ownership():
    assert score.repeated_phrase("I saw it I saw it".lower().split())
    assert not score.repeated_phrase("I saw it yesterday".lower().split())
    rows = [dict(speaker="A", start=59.9, end=60.3, word="Hello.")]
    assert score.extract(rows, dict(speaker="A", start=0, duration=60)) == ""
    assert score.extract(rows, dict(speaker="A", start=60, duration=60)) == "Hello."


def test_legacy_parsing_preserves_dialogue_and_canonical_name(tmp_path):
    path = tmp_path / "old.txt"
    path.write_text("Kaladin Shash 01:02:03 Very quiet.\n")
    assert prepare.parse_legacy(path) == [
        dict(speaker="Kaladen Shash", start=3723, text="Very quiet.")
    ]
