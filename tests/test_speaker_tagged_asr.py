from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from score_speaker_tagged_asr import _parse_tagged_tokens, _score_clip  # noqa: E402


def test_parse_tagged_tokens_assigns_current_speaker():
    tokens = _parse_tagged_tokens("[S0] Hello there [S1] Dragon! wagon")

    assert tokens == [
        {"token": "hello", "speaker": "S0", "raw": "Hello"},
        {"token": "there", "speaker": "S0", "raw": "there"},
        {"token": "dragon", "speaker": "S1", "raw": "Dragon"},
        {"token": "wagon", "speaker": "S1", "raw": "wagon"},
    ]


def test_score_clip_uses_lexical_sequence_and_oracle_mapping():
    reference = [
        {"speaker": "Alice", "start": 0.0, "end": 0.3, "text": "hello"},
        {"speaker": "Bob", "start": 0.4, "end": 0.7, "text": "dragon"},
        {"speaker": "Alice", "start": 0.8, "end": 1.1, "text": "again"},
    ]
    # "wrong" is ignored by lexical sequence matching; S1 maps to Bob.
    predicted = "[S0] hello wrong [S1] dragon [S0] again"

    score = _score_clip(reference, predicted)

    assert score["reference_words"] == 3
    assert score["predicted_words"] == 4
    assert score["lexical_matched_words"] == 3
    assert score["many_to_one_mapping"] == {"S0": "Alice", "S1": "Bob"}
    assert score["many_to_one_accuracy"] == 1.0
    assert score["one_to_one_accuracy"] == 1.0


def test_score_clip_reports_overlap_slice():
    reference = [
        {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "hello"},
        {"speaker": "Bob", "start": 0.5, "end": 1.2, "text": "there"},
    ]

    score = _score_clip(reference, "[S0] hello [S1] there")

    assert score["overlap_words"] == 2
    assert score["many_to_one_overlap_accuracy"] == 1.0
