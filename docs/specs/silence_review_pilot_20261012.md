# Speech filtering and listening review

The first ten minutes of Sessions 45 and 47 were retranscribed locally using
Silero speech detection before Qwen recognition and word alignment. Original
recordings and session-time coordinates are preserved. Quiet gaps are omitted
from recognition; the review player can skip them during listening or play the
original timeline for an omissions audit.

| Sample | Original combined tracks | Retained, including padding | Excluded from ASR | Processing time |
| --- | ---: | ---: | ---: | ---: |
| Session 45, six speakers | 3,600 s | 536.408 s | 85.1% | 232.36 s |
| Session 47, five speakers | 3,000 s | 579.722 s | 80.7% | 76.70 s |

These are completed, fresh schema-6 runs including detection, decoding and
alignment. The extra short calls make Session 45 slower than the earlier
fixed-window run. Filtering is not proof of improved WER. Short vocalizations and
noises remain difficult: there were 41 retry-without-hints events in Session 45
and two in Session 47. One Session 45 Zariel-track interval, 180.902–189.466 s,
still exhausted its output budget and is explicitly marked for listening review.
Session 47 has no flagged recognition/alignment failures. Successful output can
still contain errors; absence of a flag is not an accuracy guarantee.

The private listening workspace is `outputs/listening-review/index.html`, served
with `python -m transcriber.review_server outputs/listening-review --port 8769`.
It offers synchronized word highlighting, phrase replay, playback speed, silence
skipping, corrected text and notes, issue categories, explicit missing-speech
intervals, browser-local saving and JSON export/import. Original audio remains
available to verify omitted speech. Source audio hashes and original transcript
text accompany the feedback, so it can support later accuracy evaluation.

The user's first two corrections were preserved when replacing the preliminary
player data, matching by original audio hash and retaining the old timestamps.
They are stored privately in `user-feedback-preserved.json` and seeded into the
new player. Test-only feedback was excluded. Rebuilding does not apply corrections
silently to the model output.

Validation: 66 focused tests passed; lint/format checks passed for changed Python
files. Browser checks verified word highlighting, phrase seeking, persistence
after reload, feedback export, both sessions in the selector and the two preserved
corrections. The local server handles byte-range requests, including suffix/open
ranges and invalid ranges, to support media seeking. The models, detector and
inference remain local; feedback is not uploaded.

Detector settings, setup, commands and remaining limitations are documented in
[the Mac isolated-track guide](../mac-isolated-track-transcription.md).

## Human listening review, 2026-10-12

The user reported reviewing approximately 60–70%, primarily the Dungeon Master,
and spot-checking other speakers. Their assessment was that most transcription
was correct, campaign names were correct, and speaker labels looked appropriate,
with a few garbled words and minor fixes. The precise reviewed intervals were not
logged, so this is qualitative acceptance of the pilot, not a measured WER or a
complete audit of either session.

The feedback export was preserved under the private
`outputs/listening-review/reviews/feedback-2026-10-12T17-31-43.257Z.json`.
It contains 19 entries: two inherited from the earlier model-output version and
17 attached to current segments, all in Session 45. The inherited entries overlap
current corrections and must not be counted as independent errors. One current
entry retains the original text while noting unclear background noise.

Corrections identify several useful follow-up cases: laughter and microphone
noise interpreted as words, casual phrases confused with names or ordinary words,
game terminology such as “Dex save,” “succeed” misheard as “sixty,” and a word
assigned to the neighboring segment. Campaign-name accuracy and speaker-label
quality are supported by the user's listening assessment; the export itself is
an error-focused sample rather than a comprehensive review log.

These corrections are retained as evaluation evidence without replacing raw
model output. The next useful validation is one complete session, with special
attention to short utterances and noise. Testing changes against these cases
requires separate held-out reviewed audio to assess generalization. Neither a
lower WER than the legacy method nor archive-wide quality has yet been measured.
