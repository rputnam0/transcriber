from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_moss_diarization_dataset import (  # noqa: E402
    DEFAULT_PROMPT,
    activity_metadata,
    anonymous_target,
    anonymous_target_with_loss_spans,
    build_dataset,
    group_words_into_segments,
    reference_segment_metadata,
)


def test_default_prompt_requests_official_speaker_timestamp_format() -> None:
    assert "[S01]" in DEFAULT_PROMPT
    assert "时间戳" in DEFAULT_PROMPT


def test_target_keeps_overlapped_speaker_turns_separate() -> None:
    words = [
        {"speaker": "Alice", "start": 0.0, "end": 0.4, "text": "hello"},
        {"speaker": "Bob", "start": 0.2, "end": 0.6, "text": "yes"},
        {"speaker": "Alice", "start": 0.5, "end": 0.8, "text": "there"},
    ]

    segments = group_words_into_segments(
        words,
        turn_gap_seconds=0.75,
        maximum_segment_seconds=15.0,
    )
    target = anonymous_target(segments, duration=1.0)

    assert target == "[0.00][S01] hello there[0.80][0.20][S02] yes[0.60]"
    assert activity_metadata(segments) == [
        {"speaker": "S01", "speaker_index": 0, "start": 0.0, "end": 0.4},
        {"speaker": "S02", "speaker_index": 1, "start": 0.2, "end": 0.6},
        {"speaker": "S01", "speaker_index": 0, "start": 0.5, "end": 0.8},
    ]
    reference_metadata = reference_segment_metadata(segments)
    assert [segment["speaker"] for segment in reference_metadata] == ["S01", "S02"]
    assert [word["overlap"] for word in reference_metadata[0]["word_spans"]] == [True, True]
    assert reference_metadata[1]["word_spans"][0]["overlap"] is True


def test_pause_interruption_is_not_labeled_as_simultaneous_speech() -> None:
    words = [
        {"speaker": "Alice", "start": 0.0, "end": 0.4, "text": "I"},
        {"speaker": "Bob", "start": 0.5, "end": 0.8, "text": "yes"},
        {"speaker": "Alice", "start": 0.9, "end": 1.2, "text": "agree"},
    ]

    segments = group_words_into_segments(
        words,
        turn_gap_seconds=0.75,
        maximum_segment_seconds=15.0,
    )
    target, spans, brief_words = anonymous_target_with_loss_spans(
        segments,
        duration=2.0,
    )

    assert target == "[0.00][S01] I agree[1.20][0.50][S02] yes[0.80]"
    assert brief_words == 0
    assert all(span["kind"] != "brief_overlap_word" for span in spans)
    assert activity_metadata(segments) == [
        {"speaker": "S01", "speaker_index": 0, "start": 0.0, "end": 0.4},
        {"speaker": "S02", "speaker_index": 1, "start": 0.5, "end": 0.8},
        {"speaker": "S01", "speaker_index": 0, "start": 0.9, "end": 1.2},
    ]
    assert all(not segment["overlap"] for segment in reference_segment_metadata(segments))


def test_anonymous_target_weights_brief_overlap_text() -> None:
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 3.0, "words": ["hello", "there"]},
        {"speaker": "Bob", "start": 1.0, "end": 1.5, "words": ["yeah"]},
    ]

    target, spans, brief_words = anonymous_target_with_loss_spans(
        segments,
        duration=6.0,
    )

    assert target == "[0.00][S01] hello there[3.00][1.00][S02] yeah[1.50]"
    assert brief_words == 1
    brief_span = next(span for span in spans if span["kind"] == "brief_overlap_word")
    assert target[brief_span["start"] : brief_span["end"]] == "yeah"
    assert brief_span["weight"] == 4.0


def test_dataset_uses_mono_cut_and_repeats_overlap(tmp_path: Path) -> None:
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"fixture")
    cutset = tmp_path / "cuts.jsonl.gz"
    cut = {
        "id": "session_49_w001800_c000000",
        "duration": 30.0,
        "recording": {"sources": [{"source": str(audio)}]},
    }
    with gzip.open(cutset, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(cut) + "\n")
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        json.dumps(
            {
                "session": "Session 49",
                "window_start": 1800.0,
                "words": [
                    {"speaker": "Alice", "start": 0.0, "end": 0.4, "text": "one"},
                    {"speaker": "Bob", "start": 0.2, "end": 0.6, "text": "two"},
                    {"speaker": "Alice", "start": 0.5, "end": 0.8, "text": "three"},
                    {"speaker": "Bob", "start": 0.7, "end": 1.0, "text": "four"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "moss.jsonl"

    summary = build_dataset(
        cutset=cutset,
        forced_word_reference=reference,
        output_jsonl=output,
        summary_path=tmp_path / "summary.json",
        prompt="Transcribe.",
        excluded_sessions={"Session 34"},
        minimum_words=4,
        turn_gap_seconds=0.75,
        maximum_segment_seconds=15.0,
        overlap_repeat=2,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["source_examples"] == 1
    assert summary["training_records"] == 2
    assert rows[0]["conversation"][1]["content"] == str(audio.resolve())
    assert rows[0]["metadata"]["mono_input_only"] is True
    assert rows[0]["metadata"]["brief_overlap_word_count"] == 4
    assert rows[0]["metadata"]["activity_supervised"] is True
