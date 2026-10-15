# Sentence-level multitrack transcripts

The regular CLI assigns each isolated Craig stem its mapped speaker, splits its timed
ASR words at sentence-final punctuation (`.`, `?`, `!`, `…`, including closing quotes),
and at pauses of at least 0.8 seconds. It then interleaves lines by their unrounded start
times. TXT timestamps retain the standard `Character Name HH:MM:SS text` format;
JSONL and SRT retain fractional-second boundaries and are ordered by start time.
Multitrack SRT cues include the speaker display name. Single-file sessions retain their
existing segmentation.

All three exports use the same split records. Word order, word start/end times, and
speaker metadata are preserved. Short replies such as `Yeah.` and `Okay.` remain
separate; only punctuation-only fragments attach to neighbouring words. Common titles
such as `Dr.` do not trigger punctuation splits. There is no character-length cutoff:
long sentences without pauses remain intact. Segments with missing timing or incomplete
word coverage retain their original text and boundaries to avoid losing dialogue.

Configuration (CLI flags override config):

```yaml
multitrack_sentence_level: true
multitrack_pause_threshold: 0.8
speaker_mapping: config/speaker_mapping.example.yaml
speaker_roster:
  - Dungeon Master
  - Kaladen Shash
  - Zariel Torgan
  - Leopold Magnus
```

Use `multitrack_sentence_level: false` or `--no-multitrack-sentence-level` to keep the
legacy ASR segment lines. Set `--multitrack-pause-threshold 1.2` to require longer pauses.
The threshold must be positive and finite. `speaker_roster` is an optional list of exact
allowed display names, also accepted as quoted arguments to `--speaker-roster`.
Unexpected output names produce a warning and retain their dialogue. Configure the
attendees of the particular session; the shipped mapping includes later participants.

Canonical campaign names are Dungeon Master, Kaladen Shash, Zariel Torgan, Leopold
Magnus, Cyrus Schwert, and Cletus Cobbington. The example also includes B. Ver for other
recordings. Map `bfschmity_0` to Kaladen Shash, correcting the old Kaladin Shash label.
Roster checks identify configuration mistakes; they do not measure acoustic speaker
accuracy or bleed between recording tracks.

For Apple Silicon multitrack ZIPs and mapped directories, `backend: auto` now
selects Qwen. The Whisper path described in this report remains available with
`--backend mlx-whisper`; unmapped directories also use it automatically. Other
platforms use `faster`. `--backend parakeet` explicitly selects an older recognizer.
The following details describe the Whisper path from the Session 4 study. Its default compute
type is float16; MLX also accepts float32. MLX uses the Apple GPU regardless of the
pyannote `--device cpu` setting. Its `batch_size` argument is unused: windows decode
sequentially to bound memory. See the [ASR study](multitrack-asr-study.md) for the
comparison, historical settings, limitations, and throughput.

The local path decodes 60-second cores with three seconds of real context on each
side. Each word belongs to the core containing its midpoint, preventing duplicated
context at boundaries. It skips only cores whose decoded samples are **all exactly
zero**, preserving quiet speech without an amplitude/VAD cutoff. Whisper's own
no-speech test and two-second hallucination-silence heuristic remain enabled.
Temperature is zero, language is English, and previous-text conditioning is disabled.
For sentence-level multitrack exports, timed words are joined across decoder segments
and core windows before applying the configured sentence/pause splitter. Native
decoder boundaries therefore do not force short fragments. Disabling sentence-level
output retains the original recognizer segments. Word timing uses Whisper
cross-attention/DTW; the historical notebook did not align.

Explicitly disable `speaker_bank` and `postprocess` for isolated-stem reruns that
only need local ASR. Copy Drive audio locally before running. Set `hf_cache_root:
.hf_cache` in local config to cache model weights and verified per-window recognition.
Audio hashes, model-file hashes, decoder settings and the recognizer version identify the
cache; altered cache provenance fails rather than silently reusing text. A rerun
reuses completed windows, including when the original extraction path has changed.
The `.transcribe.json` completion marker records the resolved backend/model, compute
type and default model revision. Keep local configuration, audio, caches and
exports outside Git.

```sh
uv run transcribe '.audio/session4/Session 4.zip' \
  --backend mlx-whisper --config config/transcriber.yaml \
  --output-dir transcripts/multitrack_final
```

The CLI creates `transcripts/multitrack_final/Session 4/Session 4.{txt,jsonl,srt}`.
Compare per-speaker concatenated words against the original before replacing any
consumer transcript. Whitespace may change as lines split; ASR differences must be
reported separately. Review statistics should include full-line and dialogue-only
character lengths so speaker/timestamp prefixes do not obscure the comparison.

## Session 4 rerun (2026-10-11)

A local Apple Silicon run used Parakeet TDT 0.6B v3, MLX revision
`ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, `parakeet-mlx` 0.5.1,
MLX 0.31.0, Python 3.12.8, float16, and a 0.8-second pause threshold. The
four-track source was copied from Drive; no source or downstream transcript was
modified, and no paid provider calls were made. Outputs and detailed comparison
remain private under `transcripts/multitrack_resplit/`.

| Measure | Existing transcript | Rerun |
| --- | ---: | ---: |
| Lines | 932 | 2,485 |
| Mean full-line characters | 179.52 | 73.77 |
| Maximum full-line characters | 549 | 445 |
| Mean dialogue characters | 156.02 | 50.19 |
| Maximum dialogue characters | 525 | 421 |
| Dungeon Master lines | 325 | 1,062 |
| Kaladen Shash lines | 233 | 382 |
| Leopold Magnus lines | 143 | 372 |
| Zariel Torgan lines | 231 | 669 |

Full-line lengths exclude the newline and include the name/timestamp prefix. The
old Kaladin Shash label is counted as Kaladen Shash in this comparison. The rerun
falls within the requested 70–110 full-line character range. Long uninterrupted
sentences are retained, so the maximum remains above that range.

No output line combines source tracks, and every track has one canonical output
speaker. Of 2,485 output records, 2,452 have matching word/text inventories and
word-derived boundaries. Another 33 original model segments have inconsistent
sentence text versus timed-word inventories; they were kept intact. For example,
the model can emit a different word order in its timed tokens than in its sentence
text. The fallback avoids rewriting these sentences using those inconsistent tokens.
Synthetic regression tests verify exact preservation of input word records,
timestamps, and order through splitting.

**The rerun is not word-equivalent to the existing transcript.** Concatenated
whitespace-separated token counts (including punctuation) changed as follows:

| Speaker | Existing tokens | Rerun tokens |
| --- | ---: | ---: |
| Dungeon Master | 14,591 | 13,801 |
| Kaladen Shash | 4,146 | 2,560 |
| Leopold Magnus | 3,342 | 2,975 |
| Zariel Torgan | 6,181 | 5,637 |

These include wording, punctuation, fillers, and omission differences from fresh
ASR, rather than a segmentation-only transformation of the existing text. The
configured Apple Silicon multitrack model was used, but the historical transcript
has no saved model/decoder metadata, so its exact runtime/settings cannot be
established from the available config/history. Do not interpret the smaller lines
as proof of unchanged recognition or improved ASR accuracy. Review the private
per-speaker comparison before swapping the downstream transcript. The original
transcript's SHA-256 was verified unchanged.

Verification: `make lint`; focused `uv run pytest -q tests/test_consolidate.py
 tests/test_parakeet_backend.py tests/test_cli.py tests/test_session_reassignment.py`
(57 passed); the complete four-stem CLI run; private export/source/roster and
per-speaker word comparisons. A pre-existing Black formatting failure in
`tests/test_session_reassignment.py` was corrected without changing its behavior.
