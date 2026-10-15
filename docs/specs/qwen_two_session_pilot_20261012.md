# Local Qwen pilot: Sessions 45 and 47

This records the 2026-10-12 pilot decision. On 2026-10-15, Qwen became the
automatic Apple Silicon backend for known isolated tracks by user choice. The
accuracy limitations below still apply.

The Mac runtime is practical, but this pilot does **not establish lower WER** or
justify replacing the archive. Qwen produces promising repairs alongside new
errors. Name hints also introduce a quiet-input hallucination that required a
targeted safeguard.

## Paired inputs and method

Compared the original archived transcript with newly generated Qwen output for
the first ten minutes of each session, on every available isolated speaker track:

| Session | Material | Tracks | Track-minutes | Measured processing time |
| --- | --- | ---: | ---: | ---: |
| 45 | Combat, short turns, pauses | 6 | 60 | 98.25 seconds |
| 47 | Recap, narration, conversation | 5 | 50 | 109.37 seconds, including resume |

These are two session **samples**, not two complete re-transcriptions. Session 45
uses the archived audio-transcribe repository's ten-minute golden-example ZIP and
its original Session 45 transcript. Session 47 uses the campaign's Drive audio ZIP
and the D&D summary repository's archived transcript. Original transcript lines
were selected by start time; the final original line can extend past the sample
cutoff. Their word counts must not be used to infer omissions or accuracy.

The archived Colab helper uses WhisperX with Whisper large-v3. Exact package
versions and per-session historical execution settings were not established;
this is an old-output/new-output comparison, not a fresh controlled Whisper run.

Candidate: Qwen3-ASR-1.7B BF16 through MLX, Qwen 0.6B 8-bit forced alignment,
34 canonical campaign hints, deterministic generation, 30-second cores with
three seconds of surrounding audio. Hardware: Apple M5 Pro, 48 GB unified memory.
Both models were already downloaded. All inference was offline on the Mac.
Revision and dependency pins, vocabulary and audio hashes are retained in the
chunk provenance.

## Runtime and capacity

Session 45 is a fresh inference run under cache schema 3, including loading,
audio decoding, recognition, alignment and output. The earlier failed schema-2
attempt is excluded. Session 47's fresh run stopped when the filesystem ran out
of space after 103.93 seconds; after removing this test's temporary downloaded
ZIP, the cached resume completed in 5.44 seconds. The sum is 109.37 seconds of
process time, excluding download, preparation, idle and disk cleanup time. Its
response JSON's approximately five-second timing is **cached resume time**, not
fresh inference throughput.

The process footprint reported by macOS peaked around 8.2 GB decimal. Extrapolating
these samples gives approximately **30–35 minutes for a three-hour session with
five or six similar tracks**. This is a rough estimate: activity, silence, hints,
track count and output length change the workload. Historical Whisper runtime
was not measured, so no speedup over Colab is claimed.

Disk capacity is a real constraint: the run exhausted available storage. Removing
the temporary downloaded archive restored approximately 1.3 GiB at the time of
the check. Free substantially more space before full-session or archive runs;
keep sufficient room for archives, extracted audio, caches and exports.

## Text observations

These are candidate improvements/regressions based on the paired text, **not
listening-verified corrections**:

| Sample | Original | Qwen | Interpretation to verify by listening |
| --- | --- | --- | --- |
| 45, Leopold ~07:24 | “hurdle a stone” | “hurl a stone” | Plausible ordinary-word improvement |
| 45, Leopold ~06:55 | “I have casted it” | “I upcasted it” | Plausible game terminology improvement |
| 45, DM ~01:05 | “simultaneously” | “Some things” | Apparent regression |
| 47, DM ~08:11 | “Elvin farm” | “elven form” | Plausible ordinary-word improvement |
| 47, DM ~08:11 | “furbolgs” | “furballs” | Apparent game terminology regression |

Qwen still renders dice notation inconsistently (“sixty eight” for the old “68”)
and varies unlisted fantasy names. A vocabulary bank helps particular terms; it
does not establish a general transcription-quality improvement.

On quiet input, Qwen copied the complete 34-term hint list. The original timestamp
validation stopped the first Session 45 attempt. The backend now preserves the
echo and retries without hints when the entire output matches a bank of at least
four terms, ignoring punctuation and case. There were **10 such retries in
Session 45 and 17 in Session 47**. The safeguard is intentionally narrow: partial
hint copying and other invented speech are not comprehensively detected.
Zero-duration aligned words (77 and 49 respectively) remain flagged in the audit;
they require review and are not automatically counted as transcription errors.

## Decision and remaining work

Keep Qwen opt-in and retain the original archive. Installation is complete on this
Mac, including offline models, the campaign vocabulary and a ready local config.
Before a wholesale replacement:

1. Listen to and correct a representative, fixed sample across the two sessions,
   including quiet speech, silence, combat rules, narration, numbers and names.
   Transcribe what is audible, marking unintelligible portions instead of guessing.
2. Score both candidates against that independent reference using identical
   normalization. Report overall WER, insertions/deletions/substitutions and a
   separate name/term score. Do not treat the old transcript as ground truth.
3. Evaluate conservative speech detection and a smaller session-specific bank
   against that same reference, checking that quiet words are retained. The exact
   echo retry alone is insufficient evidence for unattended archive conversion.
4. Free disk space and then run a complete session to validate long-recording
   behavior and the runtime estimate before scaling to the archive.

Private, ignored artifacts are under `outputs/qwen-comparison/`: `review.html`
contains playable isolated audio beside both versions; `summary.json` records
timing and audit totals; session folders retain plain text, JSONL, raw decodes,
requests and logs. The downloaded full Session 47 ZIP was removed after preparing
the review clips; originals in Drive and both repositories were not changed.

Validation: both sampled sessions completed, 53 focused Qwen/CLI tests passed,
Ruff and Black passed for the changed backend and tests, and `git diff --check`
passed. This validates execution and the regression fix, not recognition accuracy.
