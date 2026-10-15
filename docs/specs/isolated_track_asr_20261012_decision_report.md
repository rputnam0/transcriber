# Improving transcription of isolated speaker tracks

Decision date: 2026-10-12. Decision-only research assessment; at the time of this
assessment no new model inference or archive retranscription was performed. Scope: local hardware or user-managed
Colab, English campaign recordings, one known speaker per source track.

## Decision Summary

Test Qwen3-ASR-1.7B with a session-specific vocabulary first, against the current
isolated-track recognizers. Make Parakeet-TDT with NeMo GPU phrase boosting the
second primary candidate. Include Canary-Qwen-2.5B as an independent English
recognizer and Granite Speech 5.0 TurboCTC as a compact comparator. Keep Whisper
large-v3, both unchanged and with vocabulary hints, in the comparison.

The working hypothesis is that modern recognition plus vocabulary supplied during
decoding will improve both ordinary words and fantasy names. This is a testable
recommendation, not a measured improvement on this archive. Reject it if a frozen,
human-transcribed test set shows no useful gain or increased omissions/name
hallucinations. Model benchmarks establish candidates, not the archive's WER.

## Repository findings

- `src/transcriber/asr.py:112` calls faster-whisper with VAD and word timestamps,
  but supplies no language, initial prompt, or hotwords. Faster-whisper is a
  Whisper inference implementation, not a separate newer family of learned models
  (C1, L1).
- The ordinary CLI supports faster-whisper and Parakeet. Its auto selection can
  choose Parakeet for isolated tracks on supported CPU platforms. The Parakeet
  default is TDT 0.6B v3. Therefore record the resolved backend/model, not just
  `--model large-v3`: that alias can resolve to Parakeet (L2).
- The mixed-recording workflow already uses Qwen3-ASR-1.7B. Its batch helper builds
  the Qwen prompt with a null third argument; the production recording runner
  does not pass a vocabulary. Its three seconds of surrounding context means
  AUDIO context, not a campaign glossary (L3).
- The experimental `run_mac_asr_quality.py` already accepts a hotword JSON list;
  this is reusable scaffolding, not an end-to-end vocabulary bank in the ordinary
  CLI (L3).
- The prior development comparison had 1,083 reference words across 16 clips:
  Qwen 238 edits (21.98% proxy WER), Whisper 242 (22.35%), Granite 5 244 (22.53%).
  Qwen's lead over Whisper was four edits. The references were automatic isolated
  stem transcripts used to evaluate mixed speech, not human truth. They cannot
  establish accuracy on the separate tracks or validate their original transcripts
  (L4). Do not reuse them as gold labels for this decision.

The track-to-person mapping supplies attribution; the vocabulary supplies spellings
of names SPOKEN in the dialogue. These are different functions. Preserve concurrent
turns when merging tracks; separate files remove acoustic competition within a
clean track, not simultaneous speech on the shared session timeline.

## Variant Comparison

| Priority | Candidate / mechanism | Fit and expected benefit | Limitation / decision |
| --- | --- | --- | --- |
| 1 | Qwen3-ASR-1.7B with context vocabulary | Existing Mac workflow; official local Python inference accepts context; trained to use contextual information (C2–C4) | First quality hypothesis; verify vocabulary in the actual MLX adapter and compare with official CUDA behavior |
| 2 | Parakeet-TDT-0.6B-v3 with NeMo GPU-PB | Existing family in CLI; decoder phrase boosting targets uncommon names without retraining (C5–C7) | CUDA/Colab experiment; current Mac wrapper does not expose this feature; tune false insertions as well as recall |
| 3 | Canary-Qwen-2.5B | Independent English ASR candidate with strong published results (C8) | Different model from Qwen3-ASR; no assumption that Qwen's context API or generic Canary boosting applies |
| 4 | Granite Speech 5.0 470M TurboCTC | Compact English recognizer with documented Transformers and MLX paths (C9) | Already in prior mixed-audio trial; no verified turnkey name-prompt mechanism in examined card |
| Control | Whisper large-v3 unchanged and with hints | Separates vocabulary benefit from model replacement; low integration cost (C1) | Current wrapper needs to expose supported arguments; hints are not guaranteed spellings |

Four approach families inform this ranking: prompt-conditioned ASR, explicit
decoder phrase boosting, a second acoustic recognizer for disputed spans, and
supervised fine-tuning. Start with the first two. Use the third to identify or
re-decode difficult passages after measuring complementarity. Defer fine-tuning
until corrected examples show a stable residual problem; training on garbled old
transcripts would make the supervision unreliable.

Published evidence for progress: Qwen's own comparison reports 8.45% versus 9.76%
WER for Whisper large-v3 on GigaSpeech and 7.39% versus 9.90% on Common Voice
English (C4, Table 3). These are author-run benchmarks, not predicted improvements
for campaign speech. Canary's card reports 5.63% mean WER on its Open ASR suite
(C8); do not compare that mean directly with Qwen's individual datasets.

Voxtral Realtime has open weights but prioritizes streaming, which this archive
does not need (C10). It is an expansion candidate if the first comparison is
inconclusive. Paid hosted services are excluded by the user's preference.

## Vocabulary bank proposal

Maintain a private, versioned campaign dictionary and a smaller session subset.
Use confirmed notes and human spellings; do not infer canonical names from old
ASR output. Suggested fields (design proposal, not an existing CLI schema):

```yaml
terms:
  - canonical: "Vaelith"
    kind: character
    aliases: ["Vae"]
    pronunciation_note: "VAY-lith"
    active_sessions: [12, 13]
    source: "human-confirmed campaign notes"
```

The names above are illustrative. Add players, characters, NPCs, places, factions,
spells, and unusual equipment. An alias is something people actually say; a common
ASR misspelling belongs in a separate error record. Do not silently expand spoken
nicknames into full names. Pronunciation notes aid review; the inspected interfaces
do not establish a shared phoneme-lexicon API.

Compile canonical terms into each backend's supported form: Qwen context text,
Whisper hotwords/initial prompt, or a NeMo boosting phrase list (C1–C3, C6). Supply
session-relevant names across every player's track, since anyone may mention them.
Test irrelevant terms and near-homophones as negative controls. A glossary can
bias ambiguous audio; it cannot guarantee a name was spoken. Preserve raw output
and provenance for any subsequent spelling edits.

## Quality improvements beyond the checkpoint

These are engineering hypotheses to evaluate independently, not claimed measured
gains:

1. Decode original isolated audio and retain source-time offsets. Validate the
   track roster and synchronization before merging results.
2. Use speech-aware chunks with acoustic padding. Retain quiet and very short
   turns in the test set; aggressive silence removal can conceal recognition
   failures. Compare the unfiltered waveform with the chosen VAD configuration.
3. Audit chunk joins, empty decodes, repeated phrases, generation truncation, and
   speech-like spans with no words. Coverage of audio files alone does not prove
   coverage of speech.
4. Keep denoising optional and compare it against originals; do not assume cleaner
   sounding audio produces lower recognition error.
5. Align selected words to source audio for timestamps. Qwen provides a separate
   forced aligner (C2). Successful alignment is not evidence that the words are
   correct.
6. Use a second ASR system to flag disagreements and re-listen to the AUDIO. Do
   not accept majority vote or a text-only LLM rewrite as truth. A fluent rewrite
   cannot establish missing acoustic evidence.

## Proposed evaluation and archive rollout

Start with 60–90 minutes pooled across speakers and several sessions, including
ordinary speech and deliberately difficult excerpts. This size is a pragmatic
pilot, not a statistical guarantee. Expand when per-speaker or name counts are
too small. Separate a tuning set and a frozen test set by session; balance speakers
across them. Keep targeted difficult clips separate from a representative sample
when reporting aggregate error.

Humans should transcribe while listening, with canonical spellings available and
without using a candidate's output as unquestioned truth. Mark truly unintelligible
spans and report their exclusion. Establish one policy for fillers, contractions,
numbers, and incomplete words. Never normalize away a wrong name.

Run the current baseline, hinted Whisper, Qwen without/with vocabulary, Parakeet
without/with boosting, Canary-Qwen, and Granite on identical source spans. Tune
only on development sessions. Cache source hashes, model revisions, runtime,
precision, prompts, chunking/VAD settings, vocabulary version, and predictions.

Measure:

- Normalized word error rate, with substitutions/deletions/insertions separately.
- Exact canonical-name precision and recall, plus false name insertions on
  negative examples; retain case-sensitive spelling accuracy separately.
- Missed turns and short replies, invented words in silence, and chunk-boundary
  duplication/omission.
- Per-speaker and per-session results, elapsed runtime, peak memory, and human
  correction time. Report confidence intervals by resampling sessions where the
  number of sessions supports it; a small pilot is not conclusive.

Proposed practical promotion target: at least 10% relative WER reduction versus
the actual current isolated-track baseline, at least 25% reduction in incorrect
name occurrences, and no material increase in missed turns or false names. These
are project targets, not literature results. A 10% relative reduction from 20%
WER is 18% WER. Freeze tolerances for the guardrail metrics after the baseline
pilot and before opening the held-out results. If only name quality improves,
consider the vocabulary feature separately from the checkpoint replacement.

After a winner passes, run one complete session to check silence handling, offsets,
merging, and review effort. Then reprocess the archive into a new revision while
preserving originals and human corrections. No full-archive quality claim or
replacement is justified by this research alone.

## Original vs Adaptation Resolution

Official Qwen Transformers/CUDA inference is the API reference. The existing Mac
MLX implementation is an adaptation; test vocabulary forwarding and batch/single
parity before relying on it. NeMo GPU-PB documentation describes decoder support,
not automatic feature parity with parakeet-mlx. Pin an isolated Colab environment:
the repository currently constrains NeMo to major version 2, while some consulted
documentation is nightly. Granite's card specifies Transformers 5.16+ and MLX
Audio 0.5.1+ (C9); do not upgrade the working application environment implicitly.
Colab GPU fit, peak memory, and throughput remain unmeasured.

## No-Go Check Result

| condition | result | evidence / action |
| --- | --- | --- |
| Official support for candidate mechanisms | pass for research shortlist | C1–C9 |
| Runtime and adapter compatibility | unresolved for execution | Isolated environments; smoke/parity tests |
| Human gold accuracy evidence | absent; blocks superiority claims | New isolated-track benchmark required |
| Measurable decision criteria | proposed | WER, names, omissions, false insertions, review effort |
| Applicability to track attribution | pass with source validation | Use file identity and shared timeline |
| Core OCR/transcription uncertainty | pass | Native documentation and source inspected |

## Locked Defaults and Spec Delta

Local/Colab only. No diarization for verified single-person tracks. Preserve
simultaneous turns across files. English decoding where applicable. No automatic
text-only editorial rewrite in scored ASR. Retain original transcripts. Keep
private vocabulary/audio outside tracked research documents.

Compared with the existing mixed-recording work, the new decision target is pure
lexical accuracy on stems, with explicit vocabulary conditioning and independent
human references. It does not replace the Qwen/MOSS mixed-recording architecture.
The research is complete; readiness below applies to constructing the experiment,
not to selecting or deploying a production winner. See the associated
[open items](isolated_track_asr_20261012_open_items.md) and
[evidence table](../sources/isolated_track_asr_20261012_evidence_table.md).

status = ready_for_implementation
