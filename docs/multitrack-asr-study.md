# Session 4 multitrack ASR decision

Whisper large-v3 is the local multitrack default. The paired study found substantial
speech deletion in the previous Parakeet rerun, including Kaladen's quiet passages.
The change retains sentence splitting from timed words, canonical track names and
the configured roster check. Audio, detailed predictions and final exports are
private; this report contains aggregate evidence only. No paid provider calls were made.

## What the notebook actually does

The supplied `DM_Helper` JSON notebook calls
`whisperx.load_model("large-v3", "cuda", compute_type="float16")` and
`model.transcribe(audio, batch_size=32)`, loading 16 kHz audio through
`whisperx.load_audio`. It requests CUDA matmul/cuDNN TF32. Its saved output is a
**Session 48** run dated 2025-10-18, not Session 4. Notebook metadata says Colab
T4; saved logs confirm Pyannote VAD, automatic language detection per file,
Torch 2.8.0+cu126, pyannote.audio 3.4.0, Lightning 2.5.5 and Python 3.12 package
paths. The logs also warn that Pyannote disabled TF32, so the request is not proof
that TF32 stayed enabled during recognition.

The installation cells are unpinned for WhisperX and include an old Torch 2.0.0 /
CUDA 11.8 request that does **not** match the saved runtime. Numeric VAD thresholds,
beam size and decoder defaults are inherited rather than explicitly configured;
no saved package/model manifest establishes their exact versions for Session 4.
`Drafts/DM_Helper_old.ipynb` uses large-v3 / float16 / batch 16 without alignment;
`Drafts/WhisperX.ipynb` uses large-v2 / float16 / batch 16 and saves a Session 8
run, also without alignment. The single-file draft does align a Session 15 mix,
which is a different workflow. None identifies a saved Session 4 execution.
These are recoverable notebook settings; the historical Session 4 model and full
decoder configuration remain unproven.

**There is no forced alignment or diarization call.** In particular,
`whisperx.align` and an alignment model are absent. It processes each `.flac`/`.ogg`
stem separately, maps the filename to a speaker, strips the segment text, truncates
start timestamps to whole seconds, sorts all stems' segments by floating-point
start, and writes one TXT line per recognizer segment. The coarse lines come from
that segment handling; alignment is not an explanation for better Colab wording.

The d14ee96 rerun used Parakeet TDT 0.6B v3,
`mlx-community/parakeet-tdt-0.6b-v3` revision
`ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, float16,
parakeet-mlx 0.5.1, MLX 0.31.0, Python 3.12.8. Its 2,485 lines were a fresh
recognition pass, rather than a segmentation-only edit of Colab text.

At study start, d14ee96 was in open PR #5, absent from origin/main. This branch
includes its changes so the new path can be reviewed and run independently.

## Paired reference and scoring

Session 4 lasts 3:21:26; four stems total 13:25:36. Decode sources to mono float32
16 kHz before selection. Select six non-overlapping 60-second windows per speaker:
two lowest-energy windows, two quiet speech windows and two loud speech windows.
Speech selection uses the median of 100 ms frames above -55 dBFS, requiring at
least 15% active frames. This threshold selects study windows; **it is not an
inference filter**. The two lowest-energy windows for each stem are digital zero.
This yields 24 minutes, including six windows from Kaladen, eight quiet-speech
windows, eight loud-speech windows and eight digital-silence windows.

| Speaker | Silence starts (s) | Quiet starts (s) | Loud starts (s) |
| --- | --- | --- | --- |
| Dungeon Master | 960, 5040 | 540, 480 | 10740, 2340 |
| Kaladen Shash | 1560, 1680 | 7080, 5040 | 11160, 7440 |
| Leopold Magnus | 60, 480 | 0, 9840 | 1380, 1680 |
| Zariel Torgan | 600, 2100 | 1860, 6600 | 2520, 60 |

The reference is isolated-track Qwen3-ASR-1.7B bf16, revision
`e1f6c266914abc5a46e8756e02580f834a6cf8a7`, English, temperature zero,
maximum 1,024 tokens, no hotwords, using mlx-audio 0.5.5 / MLX 0.32.2 and the
pinned [Mac ASR environment](../requirements/mac-asr.lock.txt). It is the dedicated
ASR selected in the [single-file decision report](specs/mac_asr_quality_20260921_decision_report.md).
It produced 1,035 normalized speech-reference words. Raw Qwen also hallucinated
an unrelated 15-word camera sentence on **each** digital-zero window (120 words).
Those reference windows are corrected to empty using the source samples, and
those raw predictions are retained privately. Neither Qwen nor word count alone
is trustworthy evidence of speech.

Colab has no word/end timestamps. For the paragraphs intersecting study windows,
Qwen's independent forced aligner places **the original Colab text** within its
start-to-next-start interval, capped at 60 seconds. Alignment revision is
`0e1a68e91d815300c7c9754b2a7639378b23db15`
(`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`). Scoring never selects a whole
paragraph solely by its start timestamp. Parakeet and the final local candidate
use their native word times. Each window selects words by midpoint.

Normalize lowercase ASCII letters/digits and internal apostrophes; ignore
punctuation/case for WER, without expanding numbers or contractions. Count
minimum-edit substitutions, deletions and insertions. Report deletion counts
against the same reference inventory. Silent words are candidate words in the
eight exact-zero windows. Repetition flags identify adjacent repeated phrases
of at least three words; they can include real repetitions and are not automatic
proof of hallucination.

These are **automatic-reference proxy WERs**, not human gold accuracy. Selection
is deterministic and stratified, not random or an unbiased full-session estimate.
Adjacent selected windows and timestamp uncertainty affect boundary errors.
Colab forced alignment adds another source of boundary error. Fantasy names,
fillers, punctuation and reference omissions still require listening/review.

## Results and decision

| Candidate | Speaker | Proxy WER | Deleted / reference words | Silence words | Repeated-phrase windows |
| --- | --- | ---: | ---: | ---: | ---: |
| Colab | Dungeon Master | 6.41% | 11 / 421 | 0 | 0 |
| Colab | Kaladen Shash | 8.95% | 8 / 190 | 0 | 0 |
| Colab | Leopold Magnus | 10.42% | 9 / 192 | 0 | 0 |
| Colab | Zariel Torgan | 21.55% | 24 / 232 | 0 | 1 |
| Colab | ALL | 11.01% | 52 / 1035 | 0 | 1 |
| Parakeet | Dungeon Master | 30.64% | 114 / 421 | 0 | 0 |
| Parakeet | Kaladen Shash | 28.95% | 49 / 190 | 0 | 0 |
| Parakeet | Leopold Magnus | 46.88% | 75 / 192 | 0 | 0 |
| Parakeet | Zariel Torgan | 11.21% | 19 / 232 | 0 | 0 |
| Parakeet | ALL | 28.99% | 257 / 1035 | 0 | 0 |
| Whisper | Dungeon Master | 6.41% | 11 / 421 | 0 | 0 |
| Whisper | Kaladen Shash | 8.42% | 9 / 190 | 0 | 0 |
| Whisper | Leopold Magnus | 10.94% | 7 / 192 | 0 | 0 |
| Whisper | Zariel Torgan | 10.78% | 10 / 232 | 0 | 0 |
| Whisper | ALL | 8.60% | 37 / 1035 | 0 | 0 |

The completed local pipeline scores **8.60% proxy WER**, versus Colab's 11.01%
and Parakeet's 28.99%. It deletes 37/1,035 reference words (3.57%), versus
52/1,035 (5.02%) and 257/1,035 (24.83%). Across quiet windows, proxy WER is 9.36% for Whisper, 10.19% for Colab and
38.67% for Parakeet; deletions are 18, 15 and 166 of 481 reference words.
The final eight silence windows have no
candidate words; the unguarded Whisper sample pass emitted 16 and raw Qwen emitted
120. Colab's one repeated-phrase flag is not established as a hallucination.

Whisper is the closest **tested local candidate** to this reference overall. It
ties Colab for Dungeon Master, slightly trails it for Leopold, and improves the
aggregate through Kaladen and Zariel. The Colab/Whisper differences are modest;
Parakeet's deletion deficit is much larger. In Kaladen's quiet 7,080-second window,
Parakeet deletes 29 of 36 reference words, versus 5 for the final Whisper path and
3 for Colab. The omitted clauses are present in both Qwen and the old transcript,
supporting real speech omissions as a material cause of the smaller Parakeet output.
This does not classify every whole-session word-count difference as deleted speech,
or establish Whisper as better than Qwen against human truth.

Use pinned local Whisper with the silence guard and native word times as the
permanent multitrack default. The notebook supplies no evidence that forced
alignment improved its multitrack wording, so a WhisperX alignment dependency is
unnecessary for this fix. Alignment would improve placement, not recover words
that the recognizer never produced.

The preliminary unguarded MLX Whisper minute-window pass scored 10.14% proxy WER,
32 deletions and 16 words on digital silence. Its decoder is English, float16,
temperature zero, no previous-text conditioning, native word timestamps and a
two-second hallucination-silence heuristic. MLX Whisper 0.4.3 uses pinned
`mlx-community/whisper-large-v3-mlx` revision
`49e6aa286ad60c14352c404340ded53710378a11`. Unlike the notebook it decodes
sequentially, uses native Whisper no-speech handling rather than WhisperX VAD,
and has cross-attention/DTW word timing. These are practical Apple GPU adaptations;
this is not a bit-identical WhisperX/CUDA reproduction.

Punctuation/casing is generally readable in Colab and Parakeet. Colab includes
fillers and long multi-sentence paragraphs; Parakeet reads cleanly but omits
substantial speech. Whisper sometimes emits long lowercased runs without sentence
punctuation, notably in a Kaladen sample. Pause splitting helps, but does not invent
punctuation, rewrite names in dialogue, or enforce a character-length cutoff.
One sampled quiet Kaladen window contains a short final-word echo; the three-word
phrase flag does not count single-word repetitions. Silence guards are not proof
that every non-silent prediction is correct.

## Permanent path and final artifact

`backend: auto` selects MLX Whisper for Apple Silicon multitrack input and
faster-whisper elsewhere. Explicit backend choices, mixed-recording diarization
requests, CUDA selection and single-file behavior remain available.
For the study, speaker-bank matching and generative postprocessing are disabled;
stem filenames supply the four canonical names. No learned voice attribution is
needed for known isolated tracks.

The MLX backend processes 60-second cores with three seconds of real context,
assigns words by midpoint, joins timed words across decoder boundaries for
sentence-level multitrack exports, and skips cores whose decoded samples are all exactly
zero. It never applies a loudness threshold to reject quiet speech. Model files,
source PCM and contextual windows are hashed; decoder settings and recognizer
version identify resumable cache entries. Missing word timing or incomplete
word/text coverage fails instead of silently dropping recognized text.

Final exports contain **3,265 lines and 28,752 timed words**. All three formats
have matching text/names and valid bounds; sentence splitting preserves the complete
owned timed-word inventory. Mean full-line length is **67.55 characters** (dialogue
44.05); the maximum is 1,271 (dialogue 1,247). The long line reflects a long
unpunctuated recognizer run: the splitter preserves speech and does not impose an
arbitrary character cutoff. There are 41 zero-duration native Whisper words,
but no zero-duration export lines or SRT cues. They remain a timing-review limitation.

| Speaker | Colab words | Parakeet words | Final Whisper words | Final lines |
| --- | ---: | ---: | ---: | ---: |
| Dungeon Master | 14,591 | 13,801 | 14,670 | 1,061 |
| Kaladen Shash | 4,146 | 2,560 | 4,283 | 822 |
| Leopold Magnus | 3,342 | 2,975 | 3,431 | 576 |
| Zariel Torgan | 6,181 | 5,637 | 6,368 | 806 |

Word counts here use final whitespace-separated export tokens, matching the prior
comparison. Intermediate recognizer-text counts can differ slightly because joining
timed words normalizes whitespace. The complete final word inventory is 28,752.

Fresh inference with model weights already cached took approximately **30 minutes
57 seconds** for this **3:21:26** four-stem session: about **6.5× session real time**,
or **26× aggregate stem real time**. This covers all four tracks, totaling 13:25:36.
Of 808 one-minute cores, 681 were decoded and 127 were exact digital silence.
Raw recognition yielded 4,611 decoder segments; their word streams are joined before
the existing sentence/pause splitter so decoder boundaries do not force fragments.
The complete offline cache rerun reproduces all three final exports byte-for-byte.


The measured host is an Apple M5 Pro with 48 GB unified memory, Python 3.12.8,
MLX 0.31.0 and mlx-whisper 0.4.3 for the application run. Throughput is measured
on this Mac; it is not a CUDA benchmark or a promise for other hardware.

Private outputs are `transcripts/multitrack_final/Session 4/Session 4.txt`,
`.jsonl`, and `.srt`. The downstream `dnd_summary` transcript is untouched and
must be swapped only after review. Original and local audio/transcript copies
have matching SHA-256 hashes. Raw source tracks extracted for this study are
removed after verification; the private ZIP and prepared evidence remain local.

## Reproduce

Use `uv sync` for the application and an in-worktree inference environment:

```sh
uv venv --python 3.12 .venv-asr
uv pip install --python .venv-asr/bin/python -r requirements/mac-asr.lock.txt
HF_HOME="$PWD/.hf_cache" .venv-asr/bin/python scripts/resolve_mac_asr_models.py \
  --output docs/analysis/multitrack_asr_study/models.json
```

Create private track JSON containing `audio` paths and canonical `speaker` names
in the order Zariel, Dungeon Master, Kaladen, Leopold. Prepare windows, decode the
Qwen reference, and align existing coarse text:

```sh
.venv-asr/bin/python scripts/prepare_multitrack_asr_study.py \
  --tracks /path/to/private-tracks.json --output docs/analysis/multitrack_asr_study
HF_HOME="$PWD/.hf_cache" .venv-asr/bin/python scripts/run_mac_asr_quality.py \
  --manifest docs/analysis/multitrack_asr_study/manifest.json \
  --output docs/analysis/multitrack_asr_study/qwen --engine mlx \
  --model mlx-community/Qwen3-ASR-1.7B-bf16
.venv-asr/bin/python scripts/prepare_multitrack_asr_study.py \
  --output docs/analysis/multitrack_asr_study --legacy /path/to/colab.txt \
  --models docs/analysis/multitrack_asr_study/models.json
```

Run the normal CLI with local `hf_cache_root: .hf_cache`,
`speaker_mapping: config/speaker_mapping.example.yaml`, a four-name `speaker_roster`,
`speaker_bank.enabled: false`, and `postprocess.enabled: false`:

```sh
uv run transcribe '.audio/session4/Session 4.zip' \
  --config config/transcriber.yaml --output-dir transcripts/multitrack_final
uv run python scripts/score_multitrack_asr.py \
  --manifest docs/analysis/multitrack_asr_study/manifest.json \
  --reference docs/analysis/multitrack_asr_study/qwen \
  --candidate colab=docs/analysis/multitrack_asr_study/old_words.json \
  --candidate parakeet=/path/to/prior/Session4.jsonl \
  --candidate 'whisper=transcripts/multitrack_final/Session 4/Session 4.jsonl' \
  --output docs/analysis/multitrack_asr_study/final_scores.json
```

Verification: `make lint`; focused `uv run pytest -q tests/test_mlx_whisper_backend.py
 tests/test_multitrack_asr_study.py tests/test_cli.py tests/test_consolidate.py
 tests/test_parakeet_backend.py tests/test_cli_transcribe.py tests/test_session_reassignment.py`;
complete four-track CLI inference and identical offline cache replay; paired scoring and private export/source/roster audit.
The focused suite passed **87 tests**, and `make lint` passed. The full repository test suite was not run, as requested. `make setup` synced the
environment but its bare `pre-commit install` command was not on PATH; shared Git
hooks were left untouched. The explicit `uv run` lint/test commands work locally.
