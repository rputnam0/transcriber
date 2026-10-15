# Full Session 47 local transcription trial

The complete 2-hour-36-minute Session 47 was processed on the user's Mac with
Qwen3-ASR-1.7B, the pinned Qwen forced aligner, Silero VAD, and the same 34 canonical
campaign hints used in the listening pilot. No recognition or speech-detection
settings were changed in response to the pilot corrections for this run.

The fresh run completed in **1,747.52 seconds (29 minutes 8 seconds)**, including
model setup, speech detection, recognition, alignment, retries, and cache writes.
This excludes downloading the source ZIP and encoding the browser playback copies.
Five source tracks total 12.63 hours; their detected, padded speech regions total
149.1 minutes. **80.3% of combined track time was excluded from recognition.**
The output contains 23,564 aligned words and three explicitly flagged passages.
This is a runtime and pipeline validation, not a measured improvement in WER.

| Speaker | Source minutes | Retained minutes | Processing minutes | Flagged passages |
| --- | ---: | ---: | ---: | ---: |
| Zariel Torgan | 155.9 | 17.7 | 5.7 | 1 |
| Kaladen Shash | 155.9 | 16.7 | 5.0 | 2 |
| Dungeon Master | 155.9 | 87.8 | 10.5 | 0 |
| Cyrus Schwert | 134.3 | 11.6 | 4.1 | 0 |
| Leopold Magnus | 155.9 | 15.3 | 3.8 | 0 |

The shorter Cyrus recording is the length of its source track. Start coordinates
remain on the original session timeline. None of the historical transcripts were
replaced. The earlier listening workspace and its 19-entry feedback snapshot remain
separate from this full-session review.

## Review and outputs

Private results are in `outputs/qwen-full/session47/`:

- `request.json`, `response.json`, and `cache/` retain settings, raw recognition,
  alignment evidence, retries, and timing.
- `transcript/` contains the new TXT, SRT, and JSONL exports and an unchanged copy of
  the legacy transcript for comparison.
- `review/` contains the complete listening player and AAC playback copies.
- `summary.json` records track-level metrics and exact flagged intervals.
- `source-metadata.json` records the archive hash and source identity.

Temporary ZIP extraction was removed after verification. Complete AAC playback
copies remain in the review folder; originals remain in Drive. Rebuilding from
the original OGG recordings requires downloading them again and updating the
request's source paths. Content-based inference caches are retained.

Serve the review using:

```sh
.venv/bin/python -m transcriber.review_server outputs/qwen-full --port 52545
```

Open `http://127.0.0.1:52545/session47/review/`. The new **Next flagged passage**
button selects flagged intervals across speakers without creating a correction.
Use **Replay selection** to listen. Hour-long timestamps display as hours, minutes,
and seconds. Feedback remains local to this browser origin and dataset until
exported. The review uses its own dataset identity; pilot corrections are not
silently transferred to different source audio hashes.

## Interpretation

Two flagged passages are on Kaladen's track and one is on Zariel's. The failures
include mismatched alignment tokens and invalid timestamps. One very short region
copied a partial name bank and was caught by alignment validation; the existing
whole-bank retry does not cover every partial copy. The raw evidence is preserved,
while the published output shows a review marker instead of invented dialogue.

The Dungeon Master's track completed without flagged failures. This does not mean
it is error-free: ordinary substitutions, short noise interpreted as speech, and
quiet words omitted by detection still need listening review. User feedback from
the pilot supports the approach qualitatively. Archive replacement still requires
human evaluation; errors-only corrections cannot establish an overall WER.

Validation: all exported segment and word timestamps were checked against the
source timeline, and all five browser audio durations matched within 0.1 seconds.
Browser checks verified all three flag selections, seeking and word highlighting
at 1:50:41, hour-formatted timestamps, and zero feedback entries created by replay.
Nine review-builder/server tests passed; `git diff --check` passed. A scan for
dense vocabulary output found no unflagged instance, but that heuristic does not
exclude all partial hint copying or other hallucinations.
