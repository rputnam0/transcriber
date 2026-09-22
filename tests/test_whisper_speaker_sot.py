from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = _load("train_whisper_speaker_sot")
RUN = _load("run_whisper_speaker_sot")


def test_serialized_transcript_emits_fixed_tags_at_speaker_changes() -> None:
    words = [
        {"speaker": "Cyrus Schwert", "start": 0.0, "end": 0.2, "token": "hello"},
        {"speaker": "Cyrus Schwert", "start": 0.2, "end": 0.4, "token": "there"},
        {"speaker": "Dungeon Master", "start": 0.3, "end": 0.5, "token": "wrong"},
    ]

    text = TRAIN.build_serialized_transcript(words)

    assert text == "<|0.04|> hello there <|0.08|> wrong"


def test_uniform_reference_preserves_turn_identity_and_text() -> None:
    rows = [
        {
            "session": "Session 1",
            "window_start": 30.0,
            "window_end": 60.0,
            "speaker_id": "Cyrus Schwert",
            "word_spans": [
                {
                    "speaker": "Cyrus Schwert",
                    "start": 2.0,
                    "end": 4.0,
                    "text": "Hello there!",
                }
            ],
        }
    ]

    references = TRAIN.build_uniform_turn_references(iter(rows))
    words = references[("Session 1", 30.0, 60.0)]["words"]

    assert [word["token"] for word in words] == ["hello", "there"]
    assert [word["normalized"] for word in words] == ["hello", "there"]
    assert all(word["speaker"] == "Cyrus Schwert" for word in words)
    assert words[0]["source_span_start"] == words[1]["source_span_start"] == 2.0


def test_serialized_transcript_keeps_overlapped_fifo_turns_contiguous() -> None:
    words = [
        {
            "speaker": "Cyrus Schwert",
            "start": 0.0,
            "end": 0.2,
            "token": "hello",
            "source_span_start": 0.0,
            "source_span_end": 1.0,
        },
        {
            "speaker": "Dungeon Master",
            "start": 0.2,
            "end": 0.4,
            "token": "wrong",
            "source_span_start": 0.2,
            "source_span_end": 0.8,
        },
        {
            "speaker": "Cyrus Schwert",
            "start": 0.4,
            "end": 0.6,
            "token": "there",
            "source_span_start": 0.0,
            "source_span_end": 1.0,
        },
    ]

    text = TRAIN.build_serialized_transcript(words)

    assert text == "<|0.04|> hello there <|0.08|> wrong"


def test_parse_tagged_words_recovers_known_speakers_case_insensitively() -> None:
    words = RUN.parse_tagged_words("<|0.04|> hello <|0.08|> there")

    assert words == [
        {"token": "hello", "speaker": "Cyrus Schwert"},
        {"token": "there", "speaker": "Dungeon Master"},
    ]


def test_generation_limit_reserves_whisper_prefix_tokens() -> None:
    assert RUN.generation_token_limit(448, max_target_positions=448) == 444
    assert RUN.generation_token_limit(448, max_target_positions=448, reserved_tokens=5) == 443
    assert RUN.generation_token_limit(128, max_target_positions=448) == 128


def test_unsuppress_control_tokens_preserves_other_whisper_suppression() -> None:
    assert RUN.unsuppress_control_tokens([0, 58, 60, 100], [58, 60]) == [0, 100]


def test_score_tagged_transcript_counts_lexical_and_speaker_errors() -> None:
    reference = [
        {"speaker": "Cyrus Schwert", "start": 0.0, "end": 0.4, "token": "hello"},
        {"speaker": "Dungeon Master", "start": 0.2, "end": 0.5, "token": "there"},
        {"speaker": "Cyrus Schwert", "start": 0.6, "end": 0.8, "token": "again"},
    ]

    score = RUN.score_tagged_transcript(reference, "<|0.04|> hello there again")

    assert score["lexical_recall"] == 1.0
    assert score["correct_speaker_words"] == 2
    assert score["speaker_attributed_word_accuracy"] == 2 / 3
    assert score["overlap_reference_words"] == 2
    assert score["overlap_correct_speaker_words"] == 1


def test_aggregate_scores_handles_no_lexical_matches() -> None:
    score = RUN.score_tagged_transcript(
        [{"speaker": "Cyrus Schwert", "start": 0.0, "end": 0.2, "token": "hello"}],
        "<|0.08|> goodbye",
    )

    aggregate = RUN.aggregate_scores([score])

    assert aggregate["lexical_recall"] == 0.0
    assert aggregate["matched_word_speaker_accuracy"] == 0.0


def test_unique_chunk_examples_removes_candidate_duplicates() -> None:
    examples = [
        {"key": ("Session 1", 0.0, 30.0), "chunk_start": 0.0, "speaker": speaker}
        for speaker in ("A", "B")
    ]

    unique = TRAIN.unique_chunk_examples(examples)

    assert len(unique) == 1


def test_speaker_class_weights_upweight_rare_speakers() -> None:
    weights = TRAIN.balanced_speaker_weights({"common": 100, "rare": 4})

    assert weights["rare"] > 1.0
    assert weights["common"] < 1.0
    assert weights["rare"] <= 3.0


def test_speaker_token_mask_uses_tag_character_offsets() -> None:
    mask = TRAIN.build_speaker_token_mask([50367, 7751, 50369, 456], [50367, 50369])

    assert mask == [True, False, True, False]
