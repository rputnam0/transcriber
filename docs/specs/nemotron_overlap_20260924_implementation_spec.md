# Paired Nemotron overlap experiment

## Build Plan

1. Create an isolated Python environment. Pin model revision and runtime commit;
   run an official short-audio smoke test before adapting repository interfaces.
   Test CPU correctness first if CUDA is unavailable; measure Mac acceleration
   separately. Never upgrade the existing Qwen/MOSS environment in place (E5).
2. Export frame probabilities and intervals from Nemotron with explicit timestamps.
   Reuse `build_contiguous_sortformer_mask_cutset.py` where compatible, preserving
   its distinction between predictions and diagnostic oracle mapping (E4/E7).
3. Add overlap-aware scoring alongside the existing word-owner score (E6).
4. Bind anonymous slots with held-out acoustic enrollment. Reuse the enrollment
   machinery, validating it against the current voice assets; never assume
   arrival-order channel numbers are the trained MOSS identity IDs (E2/E7).
5. Compare unchanged MOSS, unchanged Nemotron, and a conservative detector-assisted
   MOSS candidate on the same recordings. Keep recognizer and enrollment fixed.

This is a proposed build sequence, not an implemented runner.

## Data and Interface Mapping

Input: mono 16 kHz audio, immutable hash, recording offset in seconds.
Output: eight independent activity columns, explicit frame times, overlapping
intervals, model/runtime fingerprints, operating settings and measured runtime.
Default output step is 10 ms (N1/N2). Do not infer the time grid by stretching all
frames to audio duration; account for padding and the final partial frame.

Bind slots from sufficiently clean single-speaker regions using enrollment from
other recordings. Preserve uncertainty when evidence is weak. Reset state at true
recording boundaries and gaps; carry state across contiguous chunks. Validate
late-arriving speakers and re-entry after silence (N3).

## Locked Defaults For Coding

Official NeMo offline configuration (N1): model `nvidia/Nemotron-3-Diarization`,
`spkcache_len=264`, `fifo_len=40`, `chunk_len=340`, `chunk_right_context=40`,
`spkcache_update_period=300`. These settings use 80 ms encoder frames; they are
not output-frame counts. Retain raw probabilities with `include_tensor_outputs=True`.

The old repository runner defaults to cache 188 and update 340; explicitly override
both. Its `_apply_streaming_config` looks for the validation method on the modules
object; the new official example calls it on the model. Verify that interface during
adaptation rather than assuming the old runner validates the new configuration.

## Evaluation Requirements

Proposed initial review budget: 30–60 minutes across at least three recordings,
including natural two-voice overlap, short quiet interruptions, rapid alternation,
clean speech, and chunk boundaries. This is an engineering budget, not a statistical
guarantee. Record speaker activity and both speakers' words by listening; use stems
as supporting evidence when available, checking cross-talk.

Development: existing training/development split policy and synthetic mixtures.
Final evaluation: newly frozen natural passages, disjoint from fitting and threshold
selection. Previously inspected Sessions 37/56 passages are regression data, not
a fresh holdout. Human speaker-only grades cannot supervise words or boundaries.

Report, separately:

- DER with overlap included and zero collar, broken into miss/false alarm/confusion.
- Overlap detection precision/recall/F1 and missed secondary-speaker duration.
- Short interruption recall, clean-speech errors, late-speaker/re-entry errors.
- Named activity accuracy using acoustic enrollment, with unknown rate; reference
  permutation is a separate diagnostic, never a deployable naming result.
- Time-and-speaker-matched word recall, insertions and speaker-attributed WER on
  human transcripts; retain automatic-reference results under a proxy label.
- Runtime and peak memory on the actual host, including ASR retries.

Use paired recording-level uncertainty intervals when enough independent recordings
exist. Report per-recording deltas; a tiny review sample cannot establish significance.

## Acceptance Gates and Kill Criteria

These are proposed engineering gates, not thresholds supplied by NVIDIA.

| gate | criterion | failure action |
| --- | --- | --- |
| Mechanical | Complete source coverage, valid timestamps, finite probabilities, state/tail checks pass | Repair runner before scoring |
| Detector | Better overlap recall at matched development-calibrated precision on frozen evaluation; show uncertainty | Reject detector-assisted routing if unsupported |
| Naming | No newly wrong names on frozen known-correct regression cases; report unknowns | Keep existing attribution |
| Words | Improved secondary-speaker word recall without higher attributed error/insertions on the paired human set | Do not promote transcript changes |
| Clean speech | No new errors on the frozen clean regression set | Narrow routing or reject candidate |
| Deployment | Runtime/memory fits a measured operating budget agreed before full-session rollout | Keep experimental |

## Training and lexical recovery ablations

First compare current decoder with reference activity versus predicted activity as
a diagnostic on labeled clips. If even reference activity fails, prioritize a
conditioned decoder/extractor rather than more diarizer training.

For fine-tuning, use source-owned stems with independently checked boundaries,
session-disjoint splits, varied relative loudness and overlap lengths, and clean
speech replay. Tune decision thresholds before weights. Compare a small adaptation
of Nemotron with the existing MOSS pilot; avoid reusing its speaker-token-only loss
as if that supervised activity or lexical recovery. Architecture-specific optimizer
and loss choices need a separate, revision-verified training smoke test.

For decoder experiments, first add contextual MOSS retries only on detector/MOSS
disagreements. Retain alternates and deduplicate with acoustic/time evidence.
Evaluate existing activity-conditioned Parakeet or target-ASR paths only after
runtime and eight-channel compatibility checks. Their existence is not evidence
of benefit with the new model (L4).

## Spec Delta and Traceability

Relative to the current Mac ASR specification, add an independent activity stream,
multi-label scoring, and explicit disagreement records (E2/E6/E8). No default
recognizer, private checkpoint, roster or published transcript changes are proposed
until the gates pass. Log any changes to these experimental defaults with their
reason and validation result before interpreting comparisons.
