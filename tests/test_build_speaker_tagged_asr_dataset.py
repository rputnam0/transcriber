from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_speaker_tagged_asr_dataset import _clip_words, tagged_transcript  # noqa: E402


def test_tagged_transcript_assigns_tags_by_first_appearance():
    words = [
        {"speaker": "Dungeon Master", "text": "hello"},
        {"speaker": "Kaladen Shash", "text": "there"},
        {"speaker": "Dungeon Master", "text": "again"},
    ]

    transcript, mapping = tagged_transcript(words, max_speakers=4)

    assert transcript == "[S0] hello [S1] there [S0] again"
    assert mapping == {"Dungeon Master": "[S0]", "Kaladen Shash": "[S1]"}


def test_tagged_transcript_rejects_more_than_max_speakers():
    words = [
        {"speaker": "A", "text": "one"},
        {"speaker": "B", "text": "two"},
        {"speaker": "C", "text": "three"},
    ]

    transcript, mapping = tagged_transcript(words, max_speakers=2)

    assert transcript == ""
    assert mapping == {"A": "[S0]", "B": "[S1]"}


def test_clip_words_rebases_times():
    words = [
        {"speaker": "A", "start": 9.0, "end": 10.5, "text": "before"},
        {"speaker": "B", "start": 12.0, "end": 13.0, "text": "inside"},
    ]

    clipped = _clip_words(words, start=10.0, end=15.0)

    assert clipped == [
        {"speaker": "A", "start": 0.0, "end": 0.5, "text": "before"},
        {"speaker": "B", "start": 2.0, "end": 3.0, "text": "inside"},
    ]
