# Speaker Isolation Literature Decision Report

## 1. Decision Summary

- Selected variant: explicitly target-conditioned speaker extraction/enhancement, using closed-set one-hot conditioning as the current live baseline.
- Retired after measurement: activity-only TS-VAD/PVAD-style top-1 word assignment. It remains useful as candidate-set metadata, but not as the main ownership decision.
- Retired after measurement: small from-scratch direct word-source CNNs over mixed word crops.
- Retired after measurement: speaker-independent ConvTasNet fine-tuning, including a pretrained Libri2Mix initialization.
- Retired after measurement: pause-bounded local region aggregation of current mixed word embeddings.
- Recent literature update: next TSE work should use direct/context-rich enrollment conditioning, not only a frozen speaker centroid.
- Rejected variants: more drop-in blind separation, more clean speaker-bank expansion by itself, ExtraTrees over current word embeddings, sequence routers over current candidate evidence, and an immediate speaker-attributed ASR rewrite.
- Decision date: 2026-06-01
- Decision owner: Codex

## 2. Variant Comparison

| variant_id | description | strengths | weaknesses | applicability | decision | citation_ids |
| --- | --- | --- | --- | --- | --- | --- |
| V-001 | TS-VAD/PVAD-style closed-set target-speaker activity model | Directly predicts which known speakers are active at each frame; avoids separated-source permutation | Local oracle source-RMS top-1 is only 38.00% on 300 hard words, so activity does not identify word ownership under overlap | Diagnostic metadata only | retired-as-primary | CIT-004, CIT-005, CIT-013 |
| V-002 | Target-speaker extraction conditioned on enrollment audio | Produces the isolated waveform the user asked about; uses existing labeled speaker audio; literature specifically targets mixtures with known reference speech | More compute and model complexity; waveform quality may still not improve word labels unless trained with downstream objective | Immediate next serious path | selected | CIT-001, CIT-002, CIT-003, CIT-013 |
| V-003 | High-capacity blind separation/fine-tuned separator | Mature model families exist and can improve source quality on benchmarks | Still has source permutation/assignment problem; repo drop-in, small PIT, and pretrained ConvTasNet fine-tune runs did not beat mixed audio | Only if target-conditioned or word-aware | rejected-as-next-step | CIT-006, CIT-007, CIT-008, CIT-009, CIT-013 |
| V-004 | Direct word-source classifier on real flattened word crops | Optimizes the exact word-speaker objective and may preserve information Titanet embeddings discard | Small raw/log-spectrogram CNNs failed badly under leave-group-out evaluation | Retired for from-scratch CNNs; pretrained fine-tuning remains separate | retired-as-primary | CIT-009, CIT-013 |
| V-005 | Full speaker-attributed ASR | Solves words and speakers jointly; uses linguistic context | Large architecture rewrite and likely training-data burden | Long-term fallback | deferred | CIT-011, CIT-012 |
| V-006 | More clean labeled speaker-bank embeddings only | Easy and already available | Local oracle results show clean voice ID is already high; mixed overlap is the bottleneck | Not sufficient | rejected | CIT-013 |
| V-007 | Context-rich or embedding-free TSE conditioning | Newer TSE variants use enrollment/mix cross-attention, multi-level enrollment features, noisy positive/negative enrollments, or flow-matching extraction | More complex than one-hot or centroid conditioning and still needs domain-matched evaluation | Design constraint for the next serious extractor | selected-design-upgrade | CIT-014, CIT-015, CIT-016, CIT-017, CIT-018, CIT-019 |

## 3. Rejection Rationale

Blind separation is no longer the best next experiment. The literature supports strong separation architectures, but it also documents permutation and continuous-evaluation issues; the repo has now measured the practical failure mode repeatedly: separated sources do not automatically become better speaker labels.

Activity-only word assignment has also been measured and retired as a primary path. The true-stem
source-RMS oracle reaches only 38.00% top-1 accuracy on 300 hard overlap rows, which means "who is
active/loudest" is not the same as "who produced this ASR word."

Activity fusion was tested because learned target activity has some complementary oracle coverage.
The 400-step activity model has a 68.33% oracle union with mixed LDA, but confidence, margin,
negative-entropy, and learned routers all stay at or below the mixed baseline; a 1600-step activity
run is no better deployably. Activity remains metadata, not a calibrated ownership signal.

Small direct word-crop classifiers have now been measured too. A raw-waveform CNN reached 12.60%
overall / 15.98% hard accuracy, and a log-spectrogram CNN reached 35.88% overall / 29.65% hard
accuracy, both below the frozen mixed Titanet/LDA baseline.

Pretrained speaker-independent ConvTasNet fine-tuning has now also been measured. The fine-tuned
Libri2Mix ConvTasNet reached only 43.33% oracle separated-source accuracy on 300 hard rows versus
57.67% mixed same-row accuracy. This keeps the target-conditioned requirement in place.

Closed-set one-hot conditioned TasNet is the first measured extractor to beat mixed audio on the
full hard-overlap set, but only modestly: 57.97% versus 56.63%, with a mixed-plus-extracted oracle
union of 71.22%. It is promoted as a baseline for future target-conditioned work, not as a solved
path.

The deployable all-candidate version is also not solved: extracting once per known speaker and
choosing with the best learned selector/router reaches 56.68% on the full hard-overlap set versus
56.63% mixed. However, an oracle over mixed plus any candidate classifier output reaches 78.27%,
so the remaining target-extraction problem is selector/calibration plus extractor quality, not just
"run the extractor for every speaker."

Richer shallow selectors over the same all-candidate cache were measured next. The best bounded
nonlinear selector, ExtraTrees over posterior/cosine scalar features, reaches 57.32% on the full
hard-overlap set. Raw candidate embeddings and row-level logistic stacks do not help. This retires
post-hoc shallow selector tuning for the current extractor as a plausible upper-90 route.

Frozen speaker-embedding backend swaps were measured too. SpeechBrain ECAPA reaches 53.55% on the
hard-overlap set, and Titanet+ECAPA concatenation reaches 55.85% with LDA, both below the existing
Titanet baseline at 56.63%. WavLM x-vectors were worse at 46.92% hard-overlap accuracy. The next
useful work should change source extraction or word attribution, not simply swap speaker encoders.

Lightweight word-aware text fusion was measured with optimistic reference-word text. Text-only
logistic context models remain near chance, and the best confidence router reaches 55.76% on hard
overlap versus 56.63% mixed audio. This retires shallow transcript-context fusion; a future
speaker-attributed ASR path would need a real joint audio/text architecture.

Objective tweaks to the current small one-hot TasNet were measured next. A small-CNN speaker
auxiliary and a frozen-Titanet identity auxiliary both hurt speaker-label accuracy. Adding a log-STFT
reconstruction loss is harmless but only moves full hard-overlap true-target extraction from 57.97%
to 58.06%, while the best deployable STFT candidate selector reaches 57.14%. This is not enough to
change the conclusion: the current small extractor family is plateauing.

A stronger one-hot conditioned STFT U-Net mask extractor was measured after that. With the strict
real-stem cache it reached only 53.33% on the 300-row hard smoke versus 59.67% mixed; with the
larger 660-crop cache and a wider/longer model it tied mixed at 59.67%. This retires generic
mask-reconstruction U-Nets as the next upper-90 path unless the objective changes toward speaker
attribution or the model is replaced with a much stronger pretrained target-speaker extractor.

An off-the-shelf pretrained ECAPA-conditioned TSE Conv-TasNet was measured as that stronger
target-speaker extractor candidate. After fixing the actual ONNX condition input name and testing
raw/peak/no/unit conditioning variants, it reached only 40.00% on a 60-row hard smoke versus
51.67% mixed. This retires the current public model as a drop-in for these Discord mixtures.

Frozen WavLM hidden-state pooling was also measured as a direct SSL word-attribution feature. The
best cheap leave-group classifier reached 28.67% on the 300-row hard smoke versus 59.67% mixed,
so pooled SSL hidden states do not preserve enough target ownership signal without fine-tuning or a
word/extraction objective.

Distribution-matched target-extractor training was measured next. A leaky eval-stem diagnostic
trained on the same selected flattened-window stem crops reached 78.33% on the 300-row hard smoke,
and the non-leaky leave-window version reached 63.33% on the same smoke. On all 2,172 hard-overlap
words, leave-window eval-stem training reached 61.28% versus 56.63% mixed. This is the strongest
learned-extractor evidence so far, but it is still far below the 90.24% oracle-mask ceiling and is
not yet a deployable split. The next serious extractor path should use a larger distribution-matched
domain corpus with strict held-out sessions and a speaker/word-attribution objective.

A larger/deeper version of the same eval-stem TasNet changes the frontier substantially. With
`enc_feats=192`, `bottleneck=192`, `cond_dim=96`, `layers=8`, `stacks=3`, and 1600 training steps,
leave-window true-target accuracy reaches 73.07% on all hard-overlap rows versus 56.63% mixed.
The stricter held-out-session check reaches 64.13% versus 53.64% mixed. This promotes
high-capacity domain target extraction as the live baseline and shows the earlier small TasNet was
under-capacity, not merely architecturally wrong.

That stricter held-out-session check is now measured. With both extractor and speaker-ID scoring
split by session, one-hot eval-stem training reaches 55.62% on all hard-overlap rows versus 53.64%
mixed. Clean-bank centroid conditioning and supplementing session folds with production-stem crops
both underperform the strict 300-row mixed baseline. The direction survives, but only weakly; the
small TasNet plus SI-SNR/L1 objective is not the upper-90 path.

The all-candidate deployability check was also rerun for the stronger leave-window eval-stem
extractor folds. The non-deployable true-condition classifier preserves the 61.28% full hard-set
result, but the best built-in deployable selector/router reaches only 56.86% versus 56.63% mixed.
A richer selector sweep over scalar posterior/cosine features, raw candidate embeddings, trees,
MLPs, and row-level models tops out at 56.95%. The oracle mixed-plus-any-candidate classifier
reaches 77.99%, so the candidate set contains recoverable speaker information, but the current
extractor/selector interface does not expose a reliable "requested speaker is present" score.

The larger/deeper all-candidate check improves the candidate oracle but not deployable selection.
The true-condition classifier reaches 73.02%, and oracle mixed-plus-any-candidate reaches 84.44%,
but the best richer selector/router reaches only 57.64% versus 56.63% mixed. The primary blocker is
now candidate calibration or word-ownership supervision, not true-target extraction capacity alone.

Temporal decoding over those all-candidate scores was measured next. Fixed-penalty Viterbi and
learned-transition HMM smoothing over mixed, candidate-self, selector, and fused emissions topped
out at 57.04% on the full hard-overlap set, below the previous richer per-word selector at 57.64%.
This retires sequence smoothing over current candidate scores as the bridge to the 84.44% oracle
candidate union.

Full-context temporal decoding was then measured to check whether hard-only evaluation was hiding
useful speaker-turn anchors from easy words. Decoding complete reference-word windows and scoring
only the hard rows raises hard accuracy from 56.63% raw to 57.50% with the best global Viterbi
penalty; nested hard-selected Viterbi reaches 57.04%. Turn context is therefore a small polish, not
a path to the requested 90%+ overlap accuracy.

Two candidate-interface follow-ups were measured on the 300-row hard smoke. A word-owner calibrated
TasNet trained with positive target crops and non-owner silence targets reached at most 61.00%
deployable accuracy versus 59.67% mixed; a positive-heavy variant restored true-condition accuracy
to 64.33% but still selected deployably at only 61.00%. Richer selectors over that cache topped out
at 60.00%. Repeating the same positive-heavy silence objective with the larger/deeper TasNet
improved true-condition accuracy to 67.67% and oracle mixed-plus-any-candidate accuracy to 84.33%,
but the best deployable rule reached only 60.67%, and richer selectors topped out at 57.00%. A
joint owner-head follow-up also failed to bridge the gap: pairwise BCE owner training produced
10.33% owner-logit argmax, row-softmax owner training improved that to 36.00% but only tied mixed
deployably, and a stronger owner-loss weight reached only 60.33% deployably while damaging the
true-condition and oracle candidate scores. A matched eval-stem STFT U-Net reached 62.33%
true-target accuracy versus 59.67% mixed, below the matched eval-stem TasNet at 63.33%. A wider
1600-step ratio-mask U-Net follow-up tied mixed at 59.67%, far below the larger/deeper TasNet at
69.00%. These results retire silence-only candidate calibration, simple owner heads on the current
separator hidden state, and small objective/architecture swaps around the current extractor family
as an upper-90 route.

Fixed-channel closed-set separation was also measured to test whether named output streams remove
the selector problem. A six-channel word-owner separator reached 60.67% deployably versus 59.67%
mixed while preserving an 83.33% oracle mixed-plus-candidate score. An all-source fixed-channel
separator initially reached only 59.67% deployably with an 82.67% oracle score. Increasing the
owner loss weight to 5.0 and training the all-source fixed-channel model for 1600 steps improved
true-channel accuracy to 69.33% and oracle mixed-plus-candidate accuracy to 85.00%, but the owner
head still scored only 30.33% by argmax and the best deployable result tied mixed at 59.67%. This
confirms the issue is not just target-conditioned request calibration; the current waveform models
still do not expose a reliable word-owner signal.

An exact-word source-energy oracle was also measured to check whether the earlier activity failure
was caused by using 2.0s crops. Even with true stems, exact word-span RMS reached 58.67%, and
word-span plus 50 ms padding reached 60.33% on the 300-row hard smoke. Word-local activity is
better than 2.0s crop activity but still not a high-accuracy ownership signal.

Lightweight WavLM direct word-owner fine-tuning was measured as a stronger pretrained-audio
alternative to the earlier frozen-feature tests. Head-only training on the 300 hard-row smoke
reached 25.67% versus 57.00% mixed, unfreezing the last two encoder layers reached 12.67%, and
head-only training on all 5,953 rows reached 36.10% overall / 27.67% hard versus 82.93% / 56.63%
mixed. This retires WavLM as a direct word-crop classifier in the current lightweight form.

Pairwise speaker-conditioned attribution over the current mixed Titanet embeddings was measured
as a direct candidate-scoring alternative to multiclass LDA. The best routed pairwise model reaches
54.51% on hard overlap versus 56.63% for mixed LDA, so classifier geometry around collapsed mixed
embeddings is also retired.

Pause-bounded local region aggregation was measured to check whether individual word crops were too
short to identify speakers reliably. The best 300-row smoke reached 61.33% versus 59.67% mixed,
but the same setting tied mixed audio at 56.63% on the full 2,172-row hard-overlap set. This
retires phrase-level aggregation of current mixed Titanet embeddings as a primary fix.

The 2024-2025 literature scan strengthens the target-extraction recommendation but changes the
implementation shape. Contextual TSE, USEF-TSE, multi-level enrollment representations, FlowTSE,
positive/negative noisy enrollment, and TargetVoice all point away from a single static speaker
centroid as the conditioning bottleneck. The next extractor should let the model attend to
enrollment audio or enrollment features directly, and the repo's labeled speaker timelines can
generate both target-present and target-absent enrollment examples.

A first positive/negative enrollment smoke was measured. Mixed positive/negative enrollment
snippets plus one-hot identity reached 64.67% on the 300-row hard smoke versus 59.67% mixed and
63.33% for the prior matched one-hot TasNet, but the full hard-set result was only 58.84% versus
56.63% mixed and below the earlier eval-stem TasNet full result of 61.28%. Removing one-hot identity
dropped to 51.00%. Direct enrollment context remains promising as a design direction, but this
small raw-enrollment encoder is not the upper-90 route.

A shallow spectral-enrollment U-Net was also measured and retired. Feeding the STFT mask U-Net
positive/negative enrollment spectral profiles plus one-hot identity reached only 56.00% on the
300-row hard smoke versus 59.67% mixed.

A direct enrollment/mix attention TasNet was also measured. The small mixed-enrollment attention
variant reached 61.00% on the 300-row hard smoke, clean target-enrollment attention reached 58.67%,
and a larger/deeper mixed-enrollment attention run reached 64.33%, all against the same 59.67%
mixed baseline. This confirms that direct enrollment context can help, but the current tokenized
attention block still trails the simpler positive/negative global-enrollment encoder at 64.67% and
the larger/deeper one-hot extractor at 69.00%.

A lightweight ASR-aware candidate selector was measured as a narrow proxy for speaker-attributed
ASR. Each candidate waveform from the larger/deeper all-candidate extractor was transcribed with
faster-whisper and matched against the reference word plus same-speaker context. `base.en` on
60 rows reached only 53.33% with the best deployable ASR router versus 55.00% mixed, although its
oracle ASR-or-mixed union was 68.33%. `small.en` on 24 rows tied mixed with the learned selector,
while raw ASR argmax was only 29.17%. Candidate text evidence is complementary but not calibrated
enough over the current extracted waveforms.

Waveform-level candidate quality selectors were measured on the larger/deeper 300-row candidate
cache. RMS, peak, residual, mixture-correlation, center/edge energy, zero-crossing, spectral-band,
centroid, bandwidth, flatness, and row-level stacks reached at most 60.00% with waveform+scalar
features, versus 59.67% mixed and 61.00% for the existing learned selector. The oracle
mixed-plus-any-candidate score remains 83.67%, so the problem is not lack of candidate diversity;
the post-hoc waveform diagnostics are not calibrated enough to choose the owner.

Aggregate candidate-class evidence was also measured on the larger/deeper full hard candidate
cache. Summing, noisy-oring, or voting LDA class probabilities across all extracted candidates
reached at most 56.45% after routing versus 56.63% mixed. The missing selector signal is therefore
not a simple per-speaker evidence aggregation bug.

Word-aware sequence modeling over the same candidate cache also fails. Direct BiLSTM tagging over
full reference-word windows reaches only 49.45-51.43% on hard-overlap words, and a narrower action
router that chooses among mixed LDA and candidate-derived class rules reaches only 48.85-51.15%,
below the 56.63% mixed hard baseline. Easy neighboring words and speaker-turn context therefore do
not calibrate the current extracted candidates.

Unbounded target-extractor masks were tested as a small architecture fix. Larger/deeper softplus
and ReLU TasNet masks reached 65.33% and 66.67% true-target accuracy on the 300-row smoke,
respectively, below the larger/deeper sigmoid-mask benchmark at 69.00%. Mask activation alone is
not the path to the oracle-mask ceiling.

Clean speaker-bank growth is also not the primary missing piece. Clean/oracle-isolated voice classification is already near the user's desired regime, while short mixed-overlap crops remain much lower.

Full speaker-attributed ASR is real and relevant, but it would replace too much of the current ASR pipeline before we know whether a smaller known-speaker activity model can close the gap.

## 4. Original vs Adaptation Resolution

| conflict_id | original_variant | adaptation_variant | normative_choice | compatibility_note | citation_ids |
| --- | --- | --- | --- | --- | --- |
| CFG-001 | Blind PIT separation | Enrollment-conditioned target extraction | target extraction for waveform isolation | The repo has named speakers and enrollment audio, so the model should be told which speaker to isolate instead of separating unnamed streams first. | CIT-001, CIT-002, CIT-003, CIT-006, CIT-013 |
| CFG-002 | Target extraction waveform first | Activity, direct from-scratch classifier, or speaker-independent separator first | target-conditioned extraction with selector/word-aware objective | Activity, small direct CNNs, and speaker-independent separator fine-tuning have failed. One-hot target extraction is modestly positive but far below oracle-mask accuracy. | CIT-001, CIT-002, CIT-003, CIT-004, CIT-005, CIT-013 |
| CFG-003 | Post-hoc speaker embeddings only | ASR-integrated speaker attribution | post-hoc plus activity now, SA-ASR later | Speaker-attributed ASR is a credible destination, but the current repo can test target activity without replacing transcription. | CIT-011, CIT-012 |

## 5. No-Go Check Result

| condition | status | evidence |
| --- | --- | --- |
| Missing primary citation for core equation or official citation for core rule | pass | Target extraction, activity detection, separation, CSS, and SA-ASR claims are supported by primary papers. |
| Unresolved unit mismatch | pass | Audio frame durations, probabilities, and word accuracy metrics are implementation choices, not literature-unit conflicts. |
| Unidentifiable parameter set | pass | Frame thresholds, loss weights, and word aggregation thresholds can be calibrated against existing multitrack labels. |
| Applicability regime mismatch | pass | TS-VAD/PVAD and target extraction assume known/enrolled speakers, matching the repo's labeled-speaker archive. |
| Missing measurable acceptance gates | pass | Gates are defined against same-row mixed baselines, hard-overlap slices, and oracle-mask ceiling in the implementation spec. |
| Core transcription-uncertain claim unresolved | pass | Sources used are digital papers or repo documents; no OCR-uncertain core claim remains. |

## 6. Locked Defaults

| default_id | field | locked_value | reason | citation_ids |
| --- | --- | --- | --- | --- |
| L-001 | next_experiment | `one_hot_target_extraction_plus_selector_or_word_aware_objective` | One-hot target extraction is the first positive learned extractor, but its full hard-set ceiling remains too low without better selection or supervision. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-002 | speaker_set | `closed_set_from_speaker_bank` | The known cast/enrollment setup is the repo's advantage and matches TS-VAD/PVAD conditioning. | CIT-004, CIT-005, CIT-013 |
| L-003 | primary_metric | `word_speaker_accuracy_on_flattened_audio` | The user-facing failure is word speaker labeling, not signal SDR. | CIT-009, CIT-013 |
| L-004 | hard_slice | `target_share <= 0.90` | This is where mixed embeddings collapse and oracle isolation shows recoverable signal. | CIT-013 |
| L-005 | integration_gate | `beat_same_row_mixed_baseline_by_10pp_on_hard_overlap_or_reach_half_oracle_gap` | Prevents integrating separator/activity models that look plausible but do not improve the measured bottleneck. | CIT-009, CIT-013 |
| L-006 | activity_use | `candidate_metadata_only` | Activity top-k can describe possible speakers, but not top-1 word ownership. | CIT-004, CIT-005, CIT-013 |
| L-007 | all_candidate_target_extraction | `diagnostic_until_extractor_or_word_objective_improves` | Current all-candidate selector and richer shallow selectors stay near mixed audio, while oracle mixed-plus-any-candidate reaches 78.27%. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-008 | frozen_embedding_backend | `titanet_small_until_extraction_changes` | ECAPA, WavLM, and ECAPA fusion underperformed Titanet on the hard-overlap slice. | CIT-009, CIT-013 |
| L-009 | text_context_fusion | `retired_for_lightweight_models` | Reference-text context/fusion underperformed mixed audio on hard overlap. | CIT-011, CIT-012, CIT-013 |
| L-010 | current_tasnet_auxiliary_losses | `retired_as_standalone_fix` | STFT loss barely improves true-target extraction, and speaker-ID auxiliary losses hurt. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-011 | pairwise_mixed_embedding_attribution | `retired_for_current_embeddings` | Pairwise candidate scoring underperformed multiclass LDA on hard overlap. | CIT-009, CIT-013 |
| L-012 | generic_stft_unet_masking | `retired_until_objective_or_pretraining_changes` | A larger one-hot U-Net mask extractor only tied the mixed 300-row smoke baseline and did not beat the prior one-hot TasNet. | CIT-001, CIT-002, CIT-009, CIT-013 |
| L-013 | public_pretrained_tse_dropin | `retired_for_current_model` | The ECAPA-conditioned public TSE model scored 40.00% versus 51.67% mixed on the interface smoke. | CIT-001, CIT-002, CIT-009, CIT-013 |
| L-014 | frozen_ssl_hidden_pooling | `retired_without_finetuning` | Pooled WavLM hidden states scored at most 28.67% versus 59.67% mixed on the 300-row hard smoke. | CIT-009, CIT-013 |
| L-015 | distribution_matched_extractor_training | `promoted_for_larger_balanced_domain_training` | Eval-stem leave-window true-target training reaches 61.28%, but strict leave-session drops to 55.62% versus 53.64% mixed, and the best eval-stem all-candidate selector reaches only 56.95%. The direction is real but weak and not deployable yet. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-016 | small_tasnet_session_generalization | `retired_as_upper_90_route` | Strict leave-session, centroid conditioning, and production-stem supplementation do not produce a large gain. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-017 | current_all_candidate_interface | `retired_until_word_ownership_objective_changes` | Production-stem and eval-stem all-candidate extraction stay near mixed audio with deployable selectors; the larger/deeper cache has an 84.44% oracle mixed-plus-any-candidate score, but per-word selectors top out at 57.64% and temporal decoding tops out at 57.04%. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-018 | word_owner_silence_objective | `retired_as_current_tasnet_fix` | Positive/non-owner-silence training reaches only 61.00% deployable accuracy for the small model, 60.67% for the larger/deeper model, 60.33% for the owner-head variant, and 60.67% for word-owner fixed speaker channels on the 300-row hard smoke, while richer selectors remain at or below 60.33%. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-019 | eval_stem_stft_unet | `retired_as_current_small_mask_fix` | Matched eval-stem STFT U-Net masking reaches 62.33% on the 300-row smoke, below the matched eval-stem TasNet at 63.33%; a wider 1600-step version ties mixed at 59.67%. | CIT-001, CIT-002, CIT-009, CIT-013 |
| L-020 | word_local_energy_activity | `metadata_only` | Exact true-stem word-span energy reaches only 58.67%, and word plus 50 ms padding reaches 60.33% on the 300-row hard smoke. | CIT-004, CIT-005, CIT-013 |
| L-021 | lightweight_wavlm_direct_word_classifier | `retired_as_current_word_attribution_fix` | WavLM head fine-tuning reaches only 36.10% overall and 27.67% hard accuracy when trained on all rows, far below mixed Titanet. | CIT-009, CIT-011, CIT-012, CIT-013 |
| L-022 | local_region_embedding_aggregation | `retired_as_primary_fix` | Pause-bounded region aggregation reaches 61.33% versus 59.67% mixed on the 300-row smoke, but ties mixed at 56.63% on the full hard-overlap set. | CIT-009, CIT-013 |
| L-023 | next_tse_conditioning | `direct_enrollment_or_positive_negative_context` | Recent TSE papers show gains from enrollment/mix attention, embedding-free conditioning, multi-level enrollment features, and positive/negative noisy enrollment; local centroid conditioning failed. | CIT-014, CIT-015, CIT-016, CIT-018 |
| L-024 | small_posneg_enrollment_encoder | `retired_as_standalone_fix` | Mixed positive/negative enrollment plus one-hot improves the 300-row smoke to 64.67% but reaches only 58.84% on the full hard-overlap set, below the matched eval-stem TasNet full result. | CIT-013, CIT-014, CIT-018 |
| L-025 | high_capacity_eval_stem_tasnet | `promoted_as_true_target_baseline` | Larger/deeper one-hot TasNet reaches 73.07% leave-window and 64.13% strict-session full hard accuracy, a major improvement over the small model. | CIT-001, CIT-002, CIT-003, CIT-013 |
| L-026 | high_capacity_candidate_selector | `blocked_by_calibration_objective` | Larger/deeper all-candidate extraction reaches 84.44% oracle mixed-plus-any-candidate, but richer deployable selectors top out at 57.64%. | CIT-009, CIT-013 |
| L-027 | shallow_spectral_enrollment_unet | `retired_as_current_context_fix` | Spectral-enrollment U-Net scores 56.00% on the 300-row smoke versus 59.67% mixed. | CIT-013, CIT-014, CIT-016 |
| L-028 | tokenized_enrollment_attention_tasnet | `retired_as_current_context_fix` | Larger/deeper enrollment/mix attention reaches 64.33% on the 300-row smoke, below positive/negative global enrollment at 64.67% and larger one-hot extraction at 69.00%. | CIT-013, CIT-014, CIT-016, CIT-018 |
| L-029 | per_candidate_asr_text_matching | `retired_as_current_selector_fix` | `base.en` text matching has a 68.33% oracle ASR-or-mixed union on 60 rows, but the best deployable router reaches only 53.33% versus 55.00% mixed; `small.en` ties mixed on 24 rows. | CIT-009, CIT-011, CIT-012, CIT-013 |
| L-030 | waveform_quality_candidate_selectors | `retired_as_current_selector_fix` | Waveform-level RMS, residual, correlation, and spectral features over high-cap candidate outputs reach at most 60.00% on the 300-row smoke, below the existing learned selector at 61.00%. | CIT-009, CIT-013 |
| L-031 | stronger_fixed_channel_owner_supervision | `retired_as_current_selector_fix` | All-source fixed speaker channels with owner loss weight 5.0 and 1600 steps improve true-channel accuracy to 69.33% and oracle mixed-plus-candidate accuracy to 85.00%, but owner argmax is only 30.33% and the best deployable rule ties mixed at 59.67%. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-032 | activity_confidence_fusion | `retired_as_current_selector_fix` | Synthetic target activity can have a 68.33% oracle union with mixed, but confidence, margin, entropy, and learned routers do not beat mixed on the 300-row smoke. | CIT-004, CIT-005, CIT-009, CIT-013 |
| L-033 | aggregate_candidate_class_evidence | `retired_as_current_selector_fix` | Sum, noisy-or, and vote routers over all larger/deeper candidate class probabilities reach at most 56.45% versus 56.63% mixed on the full hard set. | CIT-009, CIT-013 |
| L-034 | unbounded_tasnet_masks | `retired_as_current_extractor_fix` | Larger/deeper ReLU and softplus masks reach 66.67% and 65.33% true-target accuracy on the 300-row smoke, below the 69.00% sigmoid-mask benchmark. | CIT-001, CIT-002, CIT-003, CIT-013 |
| L-035 | sequence_candidate_routing | `retired_as_current_selector_fix` | Direct full-window BiLSTM tagging reaches 49.45-51.43% hard accuracy, and action routing reaches 48.85-51.15%, below the 56.63% mixed hard baseline. | CIT-009, CIT-013 |

## 7. Spec Change Protocol

Any future implementation that changes the primary metric, split policy, speaker conditioning source, or integration gate must update this decision report and the implementation spec before code is promoted beyond experiment scripts.

## 8. Final Status (Machine Readable)

status = ready_for_implementation
