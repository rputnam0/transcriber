# Session 1 speaker reviewer

This is the historical excerpt reviewer. The complete Session 1 recording has since
been processed with the confirmed four-person roster; see the [current workflow and
reader](../../docs/mac-single-file-transcription.md). Keep this review snapshot intact
so its annotations continue to refer to the original predictions.

A local audio player for grading the original MOSS fine-tune's Session 1 draft.
This snapshot's recording is 922.906 seconds (15 minutes 23 seconds), with 325 timed
segments. It has four people, according to the user. The existing model output
used six candidate identities; the review tool preserves that output rather than
silently inventing the four-person mapping.

## Run on this Mac

From the repository root:

```sh
.venv/bin/python scripts/review_speakers.py \
  --audio "$HOME/.cache/transcriber/speaker-review/session1/audio.m4a" \
  --transcript "$HOME/.cache/transcriber/moss-training-20260919/deployment/session1_selected_20260919/named.json" \
  --review "$HOME/.cache/transcriber/speaker-review/session1/review.json"
```

Open <http://127.0.0.1:8765/>. No model loading or network service is needed.
The server binds only to loopback. Audio and human labels stay outside Git.
Keep this command running while reviewing; restart it to resume later.

## Review

1. Rename the four slots at the top (DM first, then three players). Recording
   handles work best when comparing them with model predictions.
2. Click a timestamp to play with one second of context. Overlapping segments
   remain separate, and all active segments are highlighted.
3. Grade the **original speaker prediction** Correct, Wrong, or Unsure. Choosing a
   correction slot also grades the prediction by exact name match; if using an
   alias, judge the prediction explicitly before recording conclusions.
4. Flag word/timing errors separately, and add notes for missing speech or overlap.
5. Use Next ungraded, search, and filters to move quickly. Disable Follow audio to
   review a particular turn while playback continues.
6. Export review downloads JSON containing every original prediction, your grades,
   and corrected speakers. The server also saves `review.json` atomically after
   every change; browser storage is a recovery backup. Use one reviewing tab at a
   time (multiple tabs use last-save-wins semantics).

Keys outside fields/buttons: Space play/pause, C correct, W wrong, U unsure,
1–4 assign a slot, N next ungraded. Undo grade clears a verdict and speaker
correction while retaining separate word-error flags and notes.

The displayed score is **correct / (correct + wrong) among judged turns**.
Unsure and ungraded turns are excluded. It is not word-level accuracy or an
unbiased whole-session estimate if you only review flagged turns. Missing speech
is not counted in this turn score. To estimate session quality, review the whole
15-minute recording, including pauses and overlaps, or sample across the full
recording without selecting on model confidence.

## Evidence and limitations

This uses the already completed checkpoint-300 fine-tune with hybrid name binding.
On two held-out, later-session mono mixes, named-word error was 27.5%, versus
30.8% for public MOSS. Those references were automatic; this is not a calibrated
probability of correct speaker attribution on the early recording. See
`docs/analysis/moss_finetune_20260919/README.md` for methodology and caveats.

The player does not rerun inference or force six predictions into four names.
The confirmed Session 1 roster and human labels can support a later constrained
attribution pass and a separate evaluation. Preserve the uncorrected predictions
when assessing that improvement.

Validation: 11 Python tests cover byte-range seeking, invalid ranges, saving,
reload state, transcript mismatch protection, and original transcript preservation.
Browser checks covered all 325 turns, audio metadata, a correction surviving a
reload, undo, and the 77-turn flagged filter. Test grades were undone.
