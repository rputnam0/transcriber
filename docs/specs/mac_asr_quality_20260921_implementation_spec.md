# Implementation specification

## Build Plan
Implemented a resumable candidate ASR runner, a shared clip comparison, and a full recording queue. The released Qwen ASR and MOSS inference both run through MLX in an isolated Python 3.12 environment; enrollment and export use the application environment. See the [installation and run guide](../mac-single-file-transcription.md).

## Data and Interface Mapping
Audio: float32 mono 16 kHz, immutable clip hashes. Times: seconds relative to clip plus explicit recording offset. Words from the selected ASR, word timing from its aligner (C1,C2,C3). MOSS turns supply overlapping voice evidence (C7), with frozen voice enrollment and attendance constraints from the repository.

## Algorithm Procedure
1. Verify sources, prepare balanced clips from actual sessions and separate labeled-stem development audio.
2. Cache model outputs and runtime measurements. Compare lexical errors using approximate source references; inspect examples of disagreement without claiming human WER.
3. Rerun the user-selected MOSS checkpoint for each full source. Run selected ASR across all source intervals and align output words.
4. Recompute speaker attribution using local evidence and the new MOSS turns; retain simultaneous speech evidence and flag conflicts rather than silently duplicating words.
5. Export timestamped named TXT/SRT/HTML/JSON with full coverage and provenance checks, then publish complete sessions.

## Acceptance Gates and Kill Criteria
- Every source sample covered; a 15-minute excerpt cannot stand in for full Session 1.
- No stale cache or changed model accepted under the same provenance.
- Word timestamps bounded and ordered; chunk-edge reconciliation tested.
- Every output word comes from an audio recognizer, not an LLM rewriting dialogue.
- No out-of-roster speaker assignment without an explicit roster update.
- Report disagreements, omitted overlap risks and absence of human lexical ground truth.

## Spec Delta
Existing pipeline used MOSS for both words and voices. This run evaluates dedicated ASR and explicitly uses the new pilot for diarization at user request. Prior release and review snapshots are retained.

## Implemented interfaces

- `transcribe_mac_recording.py`: complete-recording orchestration and resume entry point.
- `resolve_mac_asr_models.py`: pinned public ASR/alignment model resolution.
- `restore_mac_voice_assets.py`: checksum-verified private checkpoint and enrollment restoration from a complete multipart backup.
- `run_mac_asr_quality.py`: pinned-model, SHA-verified paired candidate inference; records the Granite 4 layout repair.
- `mlx_speech_batch.py`: equal-real-duration greedy MLX batching with a per-sample token cap.
- `run_moss_mlx_recordings.py`: fresh trained-checkpoint inference over complete source manifests; separate resumable prediction caches and parser/decode repairs.
- `run_context_asr_recordings.py`: 30-second ownership intervals, ±3 seconds real context, independent forced alignment, punctuation restoration and alignment issue records.
- `attribute_mac_recordings.py`: frozen enrollment and individual-turn correction, followed by allowed-roster export.
- `fuse_asr_diarization.py`: lexical/time attribution and complete-overlap routing, retaining dedicated-ASR alternatives and word accounting.
- `finish_mac_asr_release.py`: full-source coverage, model/hash/roster/timestamp/accounting gates and standard transcript exports.
- `serve_transcript_reader.py`: localhost-only reader with byte-range seeking for multi-hour audio.
- `summarize_mac_asr_benchmark.py`, `score_mac_asr_release.py`, and `audit_retranscribed_reviews.py`: reproducible ASR, integration and frozen-review comparisons.

Late-aligned words may bridge at most 1.25 seconds to nearby turns only when their speaker evidence agrees. These are flagged. Overlap routing uses actual different-speaker intersection of at least 0.15 seconds, preserves complete MOSS turns, and does not duplicate their flattened Qwen alternative in the main transcript. Neither heuristic is a calibrated confidence score.

The reader/export preserves older annotations by using a new release directory. Existing transcript IDs are not reused as if they still identified the same utterance. The review comparison matches eligible words in the original time interval and explicitly reports unmatched cases.
