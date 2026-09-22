# Mono Sortformer Target-ASR Results

## Constraint

Isolated Discord tracks are privileged training supervision only. Their near-zero inactive
channels are not representative of a mono recording and must never be used as the production
VAD or as an evaluation input.

Production and held-out evaluation use only the mixed mono waveform. Clean tracks may provide:

- target transcript text and speaker identity during training;
- oracle slot-to-speaker assignment when constructing training examples;
- reference labels for scoring.

## Corpus

- Train sessions: 27, excluding held-out Sessions 63, 64, 66, and 67.
- Train cuts: 1,072 non-overlapping 30-second mono cuts, 8.93 hours.
- Train overlap: 890 cuts contain overlap; 14.18% of labeled speech frames overlap.
- Dev cuts: 90 mono cuts from Sessions 63 and 66, 45 minutes.
- Dev overlap: 76 cuts contain overlap; 15.54% of labeled speech frames overlap.

The clean source activity is retained as a teacher. Training variants include corrupted teacher
masks and masks/probabilities predicted by the adapted Sortformer from the mono mixture.

## Models

Baseline ASR checkpoint:

`outputs/multitalker_parakeet_adapter_kd_500step_source_v2/multitalker_parakeet_domain_adapter_final.nemo`

Selected mono-soft checkpoint:

`outputs/multitalker_parakeet_clean_soft_kernels_1000step_v1/multitalker_parakeet_domain_adapter_final.nemo`

The selected model starts from the baseline, freezes the acoustic encoder and RNNT decoder, and
updates only the 4.2M speaker/background kernel parameters. Its training set is a 50/50 mixture
of clean teacher examples and raw mono-Sortformer probability examples. The probability tensors,
not postprocessed RTTM turns, condition the target-speaker ASR path.

## Held-Out Results

Scores use oracle one-to-one slot-to-name assignment independently in each 30-second cut. They
measure the recognition upper bound and do not measure production global name binding.

| Model | Matched / reference | Recall | Precision |
| --- | ---: | ---: | ---: |
| 500-step baseline | 3,822 / 6,959 | 54.92% | 62.86% |
| Selected mono-soft | 3,861 / 6,959 | 55.48% | 64.40% |
| All-speaker soft continuation | 3,855 / 6,959 | 55.40% | 64.68% |

On the 76 overlap-containing clips, the selected model moves recall from 53.23% to 53.87% and
precision from 62.26% to 64.09%. Non-overlap recall is unchanged at 67.65%.

The selected model gains 39 matched words across dev: 16 cuts improve, 12 regress, and 62 tie.
A paired cut bootstrap gives a recall-delta 95% interval of -0.07 to +1.24 percentage points,
with 95.7% of bootstrap samples positive. Treat the gain as encouraging but not conclusive.

## Rejected Result

Training the RNNT decoder and speaker kernels for 1,500 steps on clean, corrupted, and hard
Sortformer masks was harmful. On the heavy-overlap probe, attributed recall fell from 38.76% to
23.26%. This checkpoint is not a candidate for production.

## Implications

1. Clean-track energy VAD is useful as a teacher but creates a real train/inference mismatch if
   used as the only conditioning source.
2. Raw mono Sortformer probabilities are better aligned with deployment than thresholded turns
   and preserve weak evidence for brief interruptions.
3. Correcting this mismatch yields a small held-out improvement, not the 90%+ breakthrough.
4. The current result still uses oracle per-cut name assignment. Enrollment-based global identity
   binding and calibrated ambiguous-crosstalk review remain required.
5. The modest ceiling supports moving beyond post-hoc slot decoding toward direct enrollment-
   conditioned speaker-attributed ASR rather than further mask threshold or smoothing sweeps.

## Direct Enrollment-Conditioned Target ASR

The first corrected SE-DiCoW adaptation exposed two material data-path bugs:

1. Energy-aligned activity regions carried the complete transcript on the first short VAD island,
   so target-ASR training treated transcript timing and activity timing as the same supervision.
2. `mono-sortformer-soft` cuts stored mono probability tensors, but the trainer ignored those
   tensors and rebuilt STNO masks from clean-track supervisions.

The repaired corpus preserves original transcript spans separately, trains with SE-DiCoW's native
timestamp labels, and constructs continuous STNO probabilities from mono Sortformer output. Clean
energy remains limited to isolated enrollment preparation and privileged teacher-mask variants.

Historical enrollment profiles contain 30 seconds per speaker from four training sessions. No
profile uses Sessions 63, 64, 66, or 67. On the first ten Session 63 cuts, historical-profile
binding names 23/29 streams correctly, compared with 24/29 using same-session enrollment.

An all-linear rank-8 LoRA plus FP32 FDDT/gate adaptation (9.36M trainable parameters) improves the
production-shaped first-ten score from 67.28% recall / 64.04% precision to 70.83% / 73.33% cutwise.
Chronological session-speaker aggregation, which avoids penalizing real words shifted across coarse
30-second reference boundaries, scores the same output at 75.16% / 77.82%.

On a frozen contiguous 18-cut test slice from Sessions 64 and 67, the selected step-400 model moves
strict session-aggregate recall/precision from 67.19% / 67.28% to 69.57% / 75.83%. It uses only:

- one mixed mono waveform per cut;
- mono Sortformer probabilities;
- cross-session training enrollment profiles.

The result is a real held-out gain, but it is below the reliability objective. A conservative
review renderer keeps uncertain segments and marks weak enrollment binding, crosstalk, unmapped
streams, and duplicate overlapping hypotheses. At the dev-calibrated high-confidence binding
threshold, test naming is perfect but covers only 31% of reference words, so the current review
volume is not yet occasional.

Two larger adaptations were rejected:

- Identity-aware wrong-enrollment negatives collapsed to an emit-nothing solution by step 200.
- Full 219.2M-parameter speaker-communication-block adaptation regressed fresh-dev strict recall to
  59.46%, versus 64.62% for the selected LoRA model.

## Continued Target ASR And True Contiguous Mono

A 1,200-step continuation of the corrected LoRA path was selected at step 900. On the frozen
18-cut contiguous dev set it improves recall/precision from 64.62% / 66.59% for the original
step-400 adapter to 69.78% / 75.99%. A paired enrollment-ranking continuation then trains each
positive transcript against the same activity mask with a wrong enrollment. Unlike the rejected
empty-transcript negative, the positive ASR loss remains anchored and the ranking term only asks
the correct enrollment to score the transcript above the wrong enrollment. The selected step-300
checkpoint reaches 70.97% / 76.93% on full dev.

The largest newly discovered production mismatch was independent 30-second Sortformer inference.
The input is a single mono file, but the old evaluation reset the streaming model and its speaker
cache at every ASR cut. Running Sortformer once over each 270-second mono sequence raises its mean
per-cut activity F1 from 0.604 to 0.647. Sortformer slot identities still swap over long runs, so
one global slot-to-name assignment is harmful; enrollment binding remains local to each ASR cut.

With persistent mono activity, local cross-session enrollment binding, and the paired target-ASR
checkpoint, full-dev session-aggregate recall/precision reaches 74.67% / 81.55%. Reference-only
slice diagnostics use clean-track activity solely to label scoring subsets and report:

| Dev recovery slice | Reset diarization | Persistent diarization |
| --- | ---: | ---: |
| Ordinary non-overlap | 69.20% | 74.61% |
| Overlap | 59.05% | 65.02% |
| Brief turns (<=2 s) | 63.91% | 70.20% |
| Brief overlap | 58.63% | 69.78% |

A final 12-cut confirmation set from Sessions 64 and 67 was frozen before paired-model or
persistent-diarization inference and excludes all 32 previously scored test cuts. On its 1,040
reference words, the frozen system scores 64.71% recall / 77.09% precision. Replacing persistent
diarization with 30-second resets, while keeping the same ASR model and binding policy, scores
62.88% / 68.55%. Persistent context therefore generalizes primarily by suppressing false and
wrong-speaker text. Brief-overlap recovery moves from 59.09% to 61.36% on this harder slice.

Oracle slot-to-name assignment on that same untouched slice reaches 65.48% recall / 89.14%
precision. Correct naming removes most of the remaining precision error, while lexical recovery
remains the principal recall ceiling. The dev-calibrated binding margin accepts 13/32 final
reference slots with 100% naming precision; 60/103 rendered transcript segments still require
review. This is meaningful progress, but the occasional-correction objective remains unmet.

Important deployment rule: isolated Discord energy is used only for privileged training teachers,
clean enrollment preparation, and evaluation-slice labels. Persistent diarization, enrollment
binding, target-ASR decoding, and uncertainty rendering consume the mono waveform plus historical
cross-session enrollment only.

## Broad-Mask And Enrollment-Likelihood Audit

The paired model does not encode enough speaker identity in its native transcript likelihood. On
the fixed four-cut dev probe, teacher-forcing each target transcript under every roster enrollment
gives only 25.0% top-1 accuracy. A curriculum that presents a generic all-speech mask on half of
positive and absent-speaker examples does not solve this:

| Checkpoint | Target-mask recall | Target-mask precision | Generic-mask recall | Generic-mask precision |
| --- | ---: | ---: | ---: | ---: |
| Paired champion | 80.27% | 83.84% | 80.80% | 16.38% |
| Broad step 150 | 71.47% | 86.45% | 66.93% | 14.05% |
| Broad step 300 | 41.33% | 86.11% | - | - |
| Broad step 450/600 | 35.73% | 87.01% | - | - |

The generic mask preserves lexical evidence but causes every enrollment pass to emit similar text.
The broad curriculum therefore learns suppression rather than enrollment-conditioned ownership.
This family is rejected; it is not routed into production.

The audit also found that SE-DiCoW multiplies each enrollment cross-attention block by a learned
gate. All eight official and adapted gates remain near zero (roughly 0.0001-0.0029). Replacing the
gate pattern with +0.05 destabilizes ASR. Scaling the original signed pattern by 5x is less severe,
but a ranking-focused continuation reaches only 33.3% held-out enrollment top-1 and 66.93% recall /
50.40% precision on decoded dev. Opening the gate is not a deployable tweak; speaker conditioning
must be trained end to end at substantially greater scale.

## New-Session Holdout

Session 34 was frozen before the broad-mask checkpoint sweep and opened only after the paired
champion remained selected. It contains a contiguous 300-second mono excerpt, ten 30-second cuts,
914 scored words, and five active roster speakers. Its enrollment profiles contain no Session 34
audio and were built only from later training sessions.

| System | Recall | Precision | Sequence recall | Sequence precision |
| --- | ---: | ---: | ---: | ---: |
| Persistent mono + historical enrollment | 79.65% | 67.91% | 75.82% | 64.65% |
| Oracle Sortformer slot names | 69.58% | 66.60% | 66.19% | 63.35% |
| Clean reference activity | 55.91% | 88.72% | 53.94% | 85.59% |

The clean-activity row is a scoring diagnostic, never a permissible inference mode. Slice scoring
uses isolated tracks only to define categories:

| Session 34 recovery slice | Sequence recall |
| --- | ---: |
| Ordinary non-overlap | 80.81% |
| Overlap | 44.88% |
| Brief turns (<=2 s) | 30.89% |
| Brief overlap | 7.94% |

The renderer marks 74/115 segments for review. This is not the occasional-correction objective.
The system is useful as a best-available mono baseline and review-aware workflow, but strong brief
interruption and overlap recovery still require a genuinely speaker-conditioned SA-ASR/TSE model
trained on a larger balanced multitrack corpus.

## Runnable Mono Workflow

`scripts/run_persistent_mono_named_pipeline.py` accepts one arbitrary conversation file plus a
cross-session enrollment manifest. It performs:

1. 16 kHz mono normalization and exact 30-second inference cuts.
2. One persistent Sortformer pass over each contiguous sequence.
3. Per-cut slot binding against historical enrollment profiles.
4. Six enrollment-conditioned target-ASR passes with the selected adapter chain.
5. Chronological rendering with weak-binding, crosstalk, unmapped-stream, and duplicate review
   reasons.

The loader honors cut offsets when multiple cuts share one recording, skips truly silent cuts,
and clips padded-tail timestamps to the source duration. Enrollment manifests marked as using the
evaluation session are rejected. At runtime, no clean stem, clean-track energy threshold, or
Discord inactive-channel silence is read.
