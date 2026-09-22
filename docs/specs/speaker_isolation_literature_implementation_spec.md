# Speaker Isolation Literature Implementation Spec

## 1. Build Plan

1. Keep `scripts/speaker_id_target_activity_sweep.py` as a diagnostic script rather than changing production transcription first.
2. Reuse the existing multitrack/flattened-window evaluation harness and speaker-bank enrollment assets.
3. Generate frame-level multi-label activity targets from aligned stems for each known speaker.
4. Train a TS-VAD/PVAD-style closed-set model that takes mixed-audio features plus speaker conditioning and predicts per-speaker frame activity probabilities.
5. Aggregate frame probabilities over each ASR/evaluation word span and combine them with the current Titanet/LDA word embedding score only after reporting the activity-only result.
6. Since the activity-only, direct-from-scratch classifier, and speaker-independent separator smokes missed the gate, promote explicitly target-conditioned speaker extraction. Closed-set one-hot conditioning is now the baseline variant to beat.
7. For the next serious extractor, replace static centroid-only conditioning with direct enrollment/context conditioning: enrollment/mix cross-attention, multi-level enrollment features, or positive/negative target-present/target-absent enrollment segments.

Experiment result update: `/tmp/codex_target_activity_s400.json` shows 27.67% learned activity
top-1 accuracy on 300 hard rows, while `/tmp/codex_target_activity_oracle_only.json` shows that
true-stem source-RMS activity reaches only 38.00% top-1. Activity is therefore candidate metadata,
not the next primary architecture.

Activity fusion update: `/tmp/codex_target_activity_fusion_s400_v2.json` and
`/tmp/codex_target_activity_s1600_v2.json` add confidence, margin, negative-entropy, and learned
routers between mixed Titanet/LDA and synthetic target activity. The 400-step activity model has a
68.33% oracle union with mixed, but deployable fusion reaches only 57.33%; the 1600-step model
reaches only 57.67% deployably. Do not spend more effort on standalone activity-confidence routing.

Direct classifier result update: `/tmp/codex_direct_word_source_group_s160.json` and
`/tmp/codex_direct_word_source_logmag_group_s160.json` show that small from-scratch raw/logmag
CNNs underperform the mixed Titanet/LDA baseline. This branch is retired unless replaced by
fine-tuning a substantially stronger pretrained audio/speaker model.

Pretrained separator result update: `/tmp/codex_pretrained_real_stem_s60_lr1e4_s400.json` shows
that fine-tuning an 8 kHz pretrained Libri2Mix ConvTasNet still underperforms the mixed baseline
after separated-source scoring. Future separator work must condition on the enrolled target speaker
or optimize a word-aware downstream objective.

Target-conditioned result update: `/tmp/codex_conditioned_tasnet_onehot_s60_s2400_full.json`
shows that a one-hot conditioned TasNet reaches 57.97% on all 2,172 hard-overlap words versus
56.63% mixed. It is a real but small positive. Its full hard-set mixed-plus-extracted oracle union
is 71.22%, so the next implementation must improve either the extractor itself or the selector
objective before this can approach the requested 90%+ target.

All-candidate selector update: `/tmp/codex_conditioned_tasnet_onehot_candidates_full.json` tests
the deployable form of the target-extraction idea by extracting once per known speaker and choosing
among candidates without the truth label. The best full hard-set selector/router reaches only
56.68% versus 56.63% mixed, while the oracle mixed-plus-any-candidate classifier reaches 78.27%.
This means the current extractor creates useful alternate hypotheses, but the selector/calibration
problem remains unsolved.

Richer selector update: `/tmp/codex_conditioned_tasnet_candidate_selector_features_all_full.json`
tests deployable logistic, tree, MLP, and row-level selectors over LDA posteriors, clean-bank
cosine features, and raw candidate embeddings. The best result is 57.32% on the full hard-overlap
set. Future work should not spend more effort on shallow post-hoc selection for this extractor;
it should change the extractor objective, use a stronger target-speaker extraction architecture, or
train an ASR/word-aware attribution model.

Embedding-backend update: `/tmp/codex_speechbrain_ecapa_embedding_sweep_full.json` and
`/tmp/codex_titanet_ecapa_fusion_sweep.json` show that SpeechBrain ECAPA and Titanet+ECAPA
concatenation do not beat the existing Titanet baseline. The hard-overlap scores are 53.55% for
ECAPA LDA and 55.85% for fused LDA versus 56.63% for the Titanet mixed baseline. Frozen encoder
swaps are therefore retired until a new extraction or word-attribution stage changes the input
distribution.

WavLM update: `/tmp/codex_wavlm_xvector_embedding_sweep_full.json` tests
`microsoft/wavlm-base-plus-sv` x-vector embeddings on the same rows and clean target-stem cache.
The best hard-overlap score is 46.92%, far below Titanet. Keep Titanet as the frozen speaker
embedding backend until extraction changes the input distribution.

Text-context update: `/tmp/codex_text_context_sweep.json` tests an optimistic word-aware feature
using reference-word text. Text-only context models score at most 22.47% on hard overlap, and the
best text-confidence router scores 55.76% versus 56.63% mixed audio. Do not integrate shallow
transcript-context features; a future ASR-aware path should be a true joint attribution model.

Extractor-objective update: `/tmp/codex_conditioned_tasnet_stftaux_s60_s2400_full.json`,
`/tmp/codex_conditioned_tasnet_titanetaux_s300.json`, and
`/tmp/codex_conditioned_tasnet_aux_s800.json` show that simple objective tweaks do not unlock the
current small TasNet. STFT reconstruction loss gives a tiny true-target hard-set gain
(58.06% versus 57.97%), but deployable all-candidate selection still tops out at 57.14%.
Small-CNN and frozen-Titanet speaker-ID auxiliary losses hurt. Future extractor work should change
architecture or training data/objective more substantially rather than tuning these auxiliaries.

STFT U-Net update: `/tmp/codex_unet_mask_s60_s1200.json`,
`/tmp/codex_unet_mask_s120min0_s1200.json`, and
`/tmp/codex_unet_mask_s120min0_s2400_c48.json` show that a stronger one-hot conditioned
time-frequency mask model is not enough by itself. The strict-cache run scored 53.33% versus
59.67% mixed on the 300-row hard smoke; the larger-cache wider run tied mixed at 59.67% but did not
beat the earlier one-hot TasNet smoke at 62.67%. Do not spend more effort on reconstruction-only
U-Net masking without adding a speaker-attribution loss, a better selector objective, or a
pretrained target-speaker extraction backbone.

Pretrained TSE update: `/tmp/codex_pretrained_tse_peak_s60.json`,
`/tmp/codex_pretrained_tse_none_s60.json`, and
`/tmp/codex_pretrained_tse_unit_peak_s60.json` test an off-the-shelf ECAPA-conditioned streaming
TSE Conv-TasNet. The script had to infer the ONNX condition input name (`cond_embedding`) from the
graph. After that fix, raw ECAPA centroids, no amplitude normalization, peak normalization, and
unit-normalized centroids all scored 40.00% versus 51.67% mixed on the same 60 hard rows. Do not
integrate this public model as a drop-in. A pretrained-target-extraction path remains plausible
only if the model is adapted or trained on this domain and measured by word speaker accuracy.

WavLM-hidden update: `/tmp/codex_wavlm_hidden_last4_s300_fast.json` tests pooled last-four-layer
WavLM hidden states as direct word-attribution features. On the 300-row hard smoke, the best cheap
leave-group classifier scores 28.67% versus 59.67% mixed. Do not spend more on frozen SSL pooling;
future SSL work needs fine-tuning, target-speaker extraction supervision, or ASR-integrated word
ownership loss.

Eval-stem target-extractor update: `/tmp/codex_eval_stem_tasnet_s300_s1200.json`,
`/tmp/codex_eval_stem_tasnet_lgo_s300_s800.json`, and
`/tmp/codex_eval_stem_tasnet_lgo_full_s800.json` train the one-hot conditioned TasNet on flattened
eval-window stems instead of synthetic production-stem mixtures. The leaky same-row diagnostic
scores 78.33% on 300 hard rows, proving the small extractor can learn much more when the supervised
mixture distribution matches evaluation. The non-leaky leave-window run scores 63.33% on the same
smoke and 61.28% on all 2,172 hard-overlap rows versus 56.63% mixed. Promote
distribution-matched domain training, but require stricter held-out-session validation and a
speaker/word-attribution objective before production integration.

High-capacity eval-stem update: `/tmp/codex_eval_stem_tasnet_lgo_s300_big_s1600.json`,
`/tmp/codex_eval_stem_tasnet_lgo_full_big_s1600.json`,
`/tmp/codex_eval_stem_tasnet_session_s300_big_s1600.json`, and
`/tmp/codex_eval_stem_tasnet_session_full_big_s1600.json` show that the small TasNet was
under-capacity. The larger/deeper one-hot model reaches 69.00% on the 300-row leave-window smoke,
73.07% on all 2,172 leave-window hard rows, 68.00% on the 300-row strict-session smoke, and
64.13% on all strict-session hard rows. Use this larger TasNet as the current true-target baseline
to beat.

Strict-session eval-stem update: `/tmp/codex_eval_stem_tasnet_session_s300_s800.json`,
`/tmp/codex_eval_stem_tasnet_session_full_s800.json`,
`/tmp/codex_eval_stem_tasnet_session_centroid_s300_s800.json`, and
`/tmp/codex_eval_stem_tasnet_session_supp_s300_s800.json` add held-out-session extractor training
and session-level speaker-ID scoring. One-hot conditioning scores 57.33% versus 54.33% mixed on
the 300-row smoke and 55.62% versus 53.64% mixed on all hard-overlap rows. Clean-bank centroid
conditioning and production-stem supplementation each score only 51.00% on the 300-row strict
session smoke. Do not integrate this small TasNet as a production path. The next extractor
experiment needs a larger balanced domain corpus and a stronger speaker/word-attribution objective.

Eval-stem all-candidate update: `/tmp/codex_eval_stem_candidates_lgo_s300.json` and
`/tmp/codex_eval_stem_candidates_lgo_full.json` run the leave-window eval-stem extractor once per
known speaker and select among candidates without the truth label. On all 2,172 hard-overlap rows,
the non-deployable true-condition classifier reaches 61.28%, but the best deployable
selector/router reaches only 56.86% versus 56.63% mixed. The oracle mixed-plus-any-candidate
classifier reaches 77.99%. Do not integrate all-candidate extraction as a post-hoc selector; the
next implementation must change the extractor training objective or add direct word-ownership
supervision so the correct candidate is identifiable.

Richer eval-stem selector update: `/tmp/codex_eval_stem_candidate_selector_features_all_full.json`
tests the same scalar posterior/cosine features, candidate embeddings, tree ensembles, MLPs, and
row-level selectors used for the production-stem candidate cache. The best full hard-overlap score
is 56.95%, still only +0.32 points over mixed. This confirms selector-only tuning is not the
missing bridge between the 61.28% true-condition result and the 77.99% oracle candidate union.

High-capacity candidate update: `/tmp/codex_eval_stem_candidates_lgo_s300_big_s1600.json`,
`/tmp/codex_eval_stem_candidates_lgo_full_big_s1600.json`,
`/tmp/codex_eval_stem_candidate_selector_features_full_big_s1600_fast.json`, and
`/tmp/codex_eval_stem_candidate_selector_features_full_big_s1600_all.json` rerun all-candidate
selection for the stronger extractor. True-condition full hard accuracy is 73.02%, and oracle
mixed-plus-any-candidate accuracy is 84.44%, but the best deployable selector/router reaches only
57.64% versus 56.63% mixed. The next implementation must train a calibrated candidate/ownership
objective inside the extractor or move to a joint word-attribution model; post-hoc selectors remain
insufficient.

Candidate temporal-decoding update:
`/tmp/codex_eval_stem_candidate_temporal_full_big_s1600.json` applies fixed-penalty Viterbi and
learned-transition HMM decoding over mixed, candidate-self, selector, and fused candidate emissions
from the strongest full hard-overlap all-candidate cache. The best score is 57.04% for a simple
mixed+self average emission, versus 56.63% mixed and 57.64% for the previous richer per-word
selector. Temporal smoothing over the current candidate scores is therefore not the missing bridge
to the 84.44% oracle candidate union.

Full-context temporal hard-slice update:
`/tmp/codex_full_context_temporal_hard_reference_nested.json` decodes complete reference-word
windows with mixed Titanet probabilities, then scores only the hard `target_share <= 0.90` rows.
The best global Viterbi setting reaches 57.50% hard accuracy versus 56.63% mixed raw, while nested
hard-selected Viterbi reaches 57.04%. Easy-word turn anchors help a little, but full-window
temporal context does not solve overlap speaker ownership.

Word-owner objective update: `/tmp/codex_word_owner_tasnet_s300_v1.json`,
`/tmp/codex_word_owner_tasnet_s300_w1.json`,
`/tmp/codex_word_owner_tasnet_s300_pos75.json`, and
`/tmp/codex_word_owner_candidate_selector_pos75_all.json` train a TasNet variant where the truth
candidate outputs the target stem crop and every non-owner candidate outputs silence. The best
deployable 300-row score is 61.00% versus 59.67% mixed. A positive-heavy loss restores
true-condition accuracy to 64.33%, but deployable selection remains at 61.00%, and richer selectors
reach only 60.00%. The larger/deeper repeat
(`/tmp/codex_word_owner_tasnet_s300_pos75_big_s1600.json` and
`/tmp/codex_word_owner_candidate_selector_s300_pos75_big_s1600_fast.json`) improves
true-condition accuracy to 67.67% and oracle mixed-plus-any-candidate accuracy to 84.33%, but the
best deployable rule is still only 60.67%, and richer selectors top out at 57.00%. Do not continue
with silence-only calibration as the main fix; it does not make the correct candidate sufficiently
identifiable.

Owner-head update: `/tmp/codex_word_owner_head_tasnet_s300_balanced_s800.json`,
`/tmp/codex_word_owner_head_tasnet_s300_rowsoft_s800.json`, and
`/tmp/codex_word_owner_head_tasnet_s300_rowsoft_w5_s800.json` add an explicit owner/presence head
to the larger/deeper TasNet. Pairwise BCE owner training fails as a candidate ranker
(10.33% owner-logit argmax). Row-softmax training improves owner-logit argmax to 36.00%, but the
best deployable result still only ties mixed at 59.67%; increasing the owner-head loss weight
reaches 60.33% deployably while hurting true-condition and oracle-candidate accuracy. Simple owner
heads on the current separator hidden state are therefore retired.

Fixed-channel closed-set update: `/tmp/codex_closed_set_word_owner_s300_s800.json`,
`/tmp/codex_closed_set_all_sources_s300_s800.json`,
`/tmp/codex_closed_set_word_owner_selector_s300_s800_fast.json`, and
`/tmp/codex_closed_set_all_sources_selector_s300_s800_fast.json` test a one-pass separator whose
output channels are fixed to the six known speakers. Word-owner channel targets reach 60.67%
deployably versus 59.67% mixed, with an 83.33% oracle mixed-plus-candidate score. All-source
channel targets reach only 59.67% deployably, with an 82.67% oracle score. Fixed output identity
therefore does not solve the candidate-calibration problem for this data/model size.

Stronger fixed-channel owner-supervision update:
`/tmp/codex_closed_set_all_sources_s300_w5_s1600.json` repeats the all-source channel target with
owner loss weight 5.0 for 1600 steps. It improves true-channel accuracy to 69.33% and oracle
mixed-plus-any-candidate accuracy to 85.00%, but owner-logit argmax remains 30.33% and the best
deployable selector/router ties mixed at 59.67%. Do not keep scaling fixed output channels unless
the ownership signal itself is redesigned.

Eval-stem U-Net update: `/tmp/codex_eval_unet_s300_s800.json` trains a one-hot STFT U-Net mask
directly on matched flattened eval-window mixtures and stems. It reaches 62.33% true-target
accuracy on the 300-row hard smoke versus 59.67% mixed, below the matched eval-stem TasNet at
63.33%. Do not replace the current matched TasNet branch with this small U-Net.

High-capacity eval-stem U-Net update: `/tmp/codex_eval_unet_s300_big_s1600_c64.json` widens the
same ratio-mask distillation model to `base_channels=64`, `cond_channels=16` and trains for
1600 steps. It ties mixed at 59.67% on the 300-row smoke, below both the 800-step smaller U-Net
at 62.33% and the larger/deeper TasNet at 69.00%. Do not keep scaling this shallow STFT U-Net
backbone as the route to the oracle-mask ceiling.

TasNet mask-activation update: `/tmp/codex_eval_stem_tasnet_softplus_s300_big_s1200.json` and
`/tmp/codex_eval_stem_tasnet_relu_s300_big_s1200.json` test unbounded softplus/ReLU masks in the
larger/deeper eval-stem TasNet. They reach 65.33% and 66.67% true-target accuracy on the 300-row
hard smoke, below the 69.00% larger/deeper sigmoid-mask benchmark. Keep the bounded sigmoid mask
as the current baseline unless the backbone or objective changes more substantially.

Word-local activity oracle update: an exact-word true-stem source-energy diagnostic reaches 58.67%,
and word-span plus 50 ms padding reaches 60.33% on the 300-row hard smoke. This rules out a simple
"choose the locally loudest true source" target as a hidden upper-90 proxy; activity features remain
metadata unless learned jointly with a word/ASR objective.

WavLM fine-tuning update: `/tmp/codex_wavlm_finetune_hard300_head_lr1e3_s300.json`,
`/tmp/codex_wavlm_finetune_hard300_last2_lr1e4_s160.json`, and
`/tmp/codex_wavlm_finetune_all_head_lr1e3_s300.json` test direct word-owner classification with
`microsoft/wavlm-base-plus-sv`. The head-only hard smoke scores 25.67% versus 57.00% mixed;
unfreezing the last two encoder layers scores 12.67%; training the head on all 5,953 rows scores
36.10% overall and 27.67% on hard overlap versus 82.93% and 56.63% mixed. Do not continue with
lightweight pooled-WavLM word classification. A future ASR-aware path should use an architecture
that models words and speakers jointly, not a post-hoc crop classifier.

Pairwise attribution update: `/tmp/codex_pairwise_titanet_attribution_sweep.json` tests a
speaker-conditioned candidate scorer over mixed Titanet embeddings and clean speaker centroids.
The best routed hard-overlap score is 54.51%, below the 56.63% mixed LDA baseline. Do not replace
the current LDA backend with pairwise scoring over the same collapsed mixed embeddings.

Region aggregation update: `/tmp/codex_region_aggregation_s300.json` and
`/tmp/codex_region_aggregation_full_best.json` test pause-bounded local averaging of existing
Titanet word embeddings. The best smoke setting reaches 61.33% versus 59.67% mixed on 300 hard
rows, but the same setting ties mixed audio at 56.63% on all 2,172 hard-overlap rows. Do not spend
more effort on phrase/window smoothing of current mixed embeddings as the main fix.

Recent TSE literature update: CIT-014 through CIT-019 add 2024-2025 evidence for richer enrollment
conditioning: enrollment/mix T-F attention, embedding-free cross-attention, multi-level enrollment
spectral plus speaker features, flow-matching TSE, noisy positive/negative enrollment comparison,
and compact low-latency TSE. For this repo, the most actionable variant is positive/negative
enrollment plus direct enrollment-feature attention because labeled Discord timelines can produce
target-present and target-absent examples without asking the user for new clean enrollment audio.

Positive/negative enrollment smoke update:
`/tmp/codex_eval_stem_posneg_enroll_mixed_s300.json`,
`/tmp/codex_eval_stem_posneg_enroll_mixed_nohot_s300.json`,
`/tmp/codex_eval_stem_posneg_enroll_target_s300.json`, and
`/tmp/codex_eval_stem_posneg_enroll_mixed_full.json` test a small raw-enrollment encoder attached
to the eval-stem TasNet. Mixed positive/negative snippets plus one-hot identity score 64.67% on
300 hard rows versus 59.67% mixed, but only 58.84% on all 2,172 hard-overlap rows versus 56.63%
mixed. Removing one-hot identity drops to 51.00%. This supports richer enrollment conditioning as a
research direction, but the small raw-enrollment encoder is not strong enough to integrate.

Spectral-enrollment U-Net update: `/tmp/codex_eval_spectral_enroll_unet_mixed_s300.json` tests a
shallow U-Net that receives positive/negative enrollment spectral profiles, their difference, and
one-hot identity. It reaches only 56.00% versus 59.67% mixed on the 300-row hard smoke. Do not
continue with this shallow spectral-context form.

Enrollment/mix attention update:
`/tmp/codex_eval_stem_enroll_attention_mixed_s300_v2.json`,
`/tmp/codex_eval_stem_enroll_attention_target_s300_v2.json`, and
`/tmp/codex_eval_stem_enroll_attention_mixed_s300_big_s1600.json` test a TasNet whose mixture
frames attend over tokenized positive and negative enrollment snippets. The small mixed-enrollment
model reaches 61.00% versus 59.67% mixed, the clean-target enrollment diagnostic reaches 58.67%,
and the larger/deeper mixed-enrollment model reaches 64.33%. This is positive but still below the
64.67% simple positive/negative global-enrollment smoke and the 69.00% larger one-hot true-target
extractor on the same 300-row protocol. Treat the current tokenized attention block as retired
unless the next version changes the backbone, supervision, or word-ownership interface.

ASR-aware candidate attribution update:
`/tmp/codex_candidate_asr_attribution_s12_cpu_context.json`,
`/tmp/codex_candidate_asr_attribution_s12_base_context.json`,
`/tmp/codex_candidate_asr_attribution_s60_base_context_learned.json`, and
`/tmp/codex_candidate_asr_attribution_s24_small_context.json` test a narrow
speaker-attributed-ASR proxy: transcribe each extracted candidate waveform and match the transcript
against the reference word plus same-speaker context. The best 60-row `base.en` deployable router
reaches 53.33% versus 55.00% mixed, despite a 68.33% oracle ASR-or-mixed union. The 24-row
`small.en` run ties mixed at 50.00% with a learned selector while raw ASR argmax is only 29.17%.
Keep full speaker-attributed ASR as a future architecture, but do not continue lightweight
per-candidate text matching over the current extracted waveforms as the selector fix.

Waveform-selector update:
`/tmp/codex_eval_stem_candidate_waveform_selector_s300_big_s1600.json` and
`/tmp/codex_eval_stem_candidate_waveform_selector_s300_big_s1600_row.json` add waveform-level
quality features over the larger/deeper all-candidate extractor outputs: RMS, peak, residual,
mixture correlation, center/edge energy, zero-crossing, spectral-band ratios, centroid, bandwidth,
flatness, and row-level feature stacks. The best waveform+scalar selector/router reaches 60.00%
on the 300-row smoke versus 59.67% mixed and 61.00% for the existing learned selector. Treat
waveform-quality metadata as retired for the current candidate outputs; the extractor must produce
a more calibrated ownership signal rather than relying on post-hoc waveform diagnostics.

Aggregate candidate-class evidence update:
`/tmp/codex_eval_stem_candidates_lgo_full_big_s1600_aggregate.json` sums, noisy-ors, and votes
LDA class probabilities across all candidate outputs from the larger/deeper extractor. The best
router reaches 56.45% on the full hard set versus 56.63% mixed. This rules out a simple
per-speaker evidence aggregation bug in the candidate selector.

Sequence candidate-router update: `/tmp/codex_sequence_candidate_tagger_prob_s240.json`,
`/tmp/codex_sequence_candidate_tagger_prob_s48.json`,
`/tmp/codex_sequence_candidate_action_router_s96.json`, and
`/tmp/codex_sequence_candidate_action_router_s48_hw2.json` test full-word-window BiLSTM models
over mixed probabilities plus candidate evidence. Direct speaker tagging reaches only 49.45-51.43%
hard accuracy, and a narrower action router over mixed/candidate rules reaches 48.85-51.15% hard
accuracy, below the 56.63% mixed baseline. Do not continue sequence routing over the current
candidate features as the selector fix.

## 2. Data and Interface Mapping

| field | meaning | units | source_equation_or_rule | citation_ids |
| --- | --- | --- | --- | --- |
| `mixed_waveform` | Flattened mono audio window to label | samples at repo working sample rate | Target activity/extraction operates on mixed multi-speaker audio | CIT-001, CIT-004, CIT-005 |
| `speaker_profile[s]` | Enrollment representation for known speaker `s` | unitless embedding or reference waveform | TS-VAD uses speaker vectors; PVAD conditions on speaker embedding or verification score; target extraction uses reference speech | CIT-001, CIT-002, CIT-004, CIT-005 |
| `speaker_enrollment_context[s]` | Raw or feature-level enrollment examples for target-present and target-absent segments | waveform or T-F feature tensors | Recent TSE variants use enrollment/mix attention, embedding-free target features, or positive/negative enrollment comparison | CIT-014, CIT-015, CIT-016, CIT-018 |
| `frame_activity[t, s]` | Whether speaker `s` is active in frame `t` | probability or binary label | TS-VAD predicts activity for each speaker at each frame | CIT-004 |
| `target_activity[t]` | One-vs-rest target/non-target/non-speech labels when running a PVAD form | probability vector | Personal VAD outputs non-speech, target speech, and non-target speech probabilities | CIT-005 |
| `target_mask[t, f]` | Optional learned target-speaker mask for waveform enhancement | unitless mask | SpeakerBeam estimates `M = g(|Y|, |A|)` and `S_hat_0 = M * Y` | CIT-002 |
| `word_score[word, s]` | Speaker ownership score for a word span | logit/probability | Aggregate frame activity over word start/end and optionally fuse with embedding backend | CIT-004, CIT-005, CIT-013 |
| `word_speaker_accuracy` | Primary evaluation metric | percent | Downstream metric avoids SDR-only trap for practical separation systems | CIT-009, CIT-013 |

## 3. Algorithm Procedure (Coding Form)

```text
inputs:
  mixed audio windows with word spans and reference speaker labels
  aligned per-speaker stems or clean labeled tracks
  speaker-bank enrollment clips/embeddings for the closed speaker set

prepare_labels:
  resample all stems to the model sample rate
  compute frame RMS or VAD activity per speaker
  mark frame_activity[t, s] = 1 when speaker s exceeds the calibrated activity threshold
  preserve multi-label frames when speakers overlap
  write split metadata that prevents same-window stem crops from leaking into evaluation

train_activity_model:
  extract log-mel or learned waveform features from mixed_waveform
  create speaker_profile[s] from enrollment clips
  for each frame and speaker, predict p(active_s_at_t | mixed_features, speaker_profile[s])
  optimize multi-label BCE or per-speaker target/non-target loss
  balance batches across speaker, target-share bucket, and active-speaker count

score_words:
  for each word span, average or max-pool p(active_s_at_t) over frames inside the span
  optionally blend with current speaker-bank LDA score after reporting activity-only metrics
  assign the speaker with the highest calibrated word_score
  emit uncertainty when top-two scores are too close or all activity scores are below threshold

evaluate:
  compare same rows against mixed embedding baseline
  report full word speaker accuracy, hard-overlap accuracy, target-share buckets, and active-speaker-count buckets
  serialize model settings, thresholds, split IDs, and raw metrics to JSON
```

## 4. Numerical Stability and Fallbacks

| scenario | risk | fallback/disable rule | citation_ids |
| --- | --- | --- | --- |
| Very short word spans | Frame average is unstable | Expand to a fixed context window for features but aggregate only around the word center; report separate short-word bucket. | CIT-004, CIT-013 |
| Multiple speakers active with close probabilities | Speaker assignment can become arbitrary | Emit low-confidence metadata and let temporal decoding smooth only after activity-only metrics are reported. | CIT-004, CIT-005 |
| Activity model beats frame F1 but not word accuracy | Optimizing the wrong metric | Block production integration; retune word aggregation or move to target extraction. | CIT-009, CIT-013 |
| Separated waveform improves SDR but worsens word labels | Signal metric mismatch | Reject the separator unless same-row word speaker accuracy improves. | CIT-009, CIT-013 |
| Enrollment profile corrupted or missing | Target-conditioned model fails silently | Fall back to current speaker-bank LDA scorer and mark target-activity disabled for that speaker. | CIT-004, CIT-005, CIT-013 |

## 5. Acceptance Gates and Kill Criteria

| gate_id | metric | threshold | comparison target | fail_action |
| --- | --- | --- | --- | --- |
| G-001 | hard-overlap word speaker accuracy (`target_share <= 0.90`) | at least +10 percentage points | same-row mixed embedding baseline | keep as experiment only |
| G-002 | hard-overlap oracle-gap closure | at least 50% of gap between mixed baseline and oracle mask | local oracle-mask ceiling | move to target extraction if missed |
| G-003 | full flattened word speaker accuracy | exceed current temporal decode champion | best existing flattened-audio pipeline | do not integrate |
| G-004 | target-share bucket regression | no bucket above 0.75 target share may regress by more than 2 points | same-row mixed baseline | recalibrate thresholds |
| G-005 | split leakage audit | zero leaked evaluation windows/crops | split metadata | invalidate run |
| G-006 | deployable candidate selection | selector/router must recover at least half of its oracle mixed-plus-any-candidate gain | all-candidate oracle for the same extractor | change extractor objective or candidate representation |
| G-007 | word-local activity proxy | must beat same-row mixed by at least 10 points before becoming primary | exact word-span source-energy or learned target activity | keep as metadata only |

## 6. Evaluation Requirements

- Use the same flattened sessions/windows as the current speaker-ID ledger whenever possible.
- Report same-row mixed baseline and activity model metrics in one JSON artifact.
- Include buckets by target energy share, active-speaker count, speaker name, session/window, word duration, and confidence margin.
- Preserve an activity-only score before blending with embedding/LDA scores or temporal decoding.
- Compare against the oracle-mask hard-overlap ceiling so failures are interpretable as model capacity, label quality, or task ambiguity.

## 7. Traceability Map

| implementation_decision | claim_ids | citation_ids | rationale |
| --- | --- | --- | --- |
| Keep target-speaker activity as diagnostic metadata | C-004, C-005, C-013 | CIT-004, CIT-005, CIT-013 | The activity-only MVP failed top-1 word ownership, but top-k activity may still help candidate analysis. |
| Keep target-speaker extraction as stage 2 | C-001, C-002, C-003 | CIT-001, CIT-002, CIT-003 | The user's masking idea is literature-backed, but it should be tried with a stronger target-conditioned model after activity baseline. |
| Do not use blind separators as drop-ins | C-006, C-007, C-008, C-009, C-013 | CIT-006, CIT-007, CIT-008, CIT-009, CIT-013 | Strong separation papers do not remove source identity/permutation or practical downstream metric issues. |
| Evaluate by word speaker accuracy | C-009, C-013 | CIT-009, CIT-013 | LibriCSS warns signal metrics can be weakly tied to ASR; repo objective is speaker-labeled words. |
| Keep speaker conditioning closed-set | C-004, C-005 | CIT-004, CIT-005 | The repo's unusual advantage is abundant known-speaker enrollment audio. |

## 8. Spec Delta (Required)

| delta_id | baseline_spec | change_summary | rationale | citation_ids |
| --- | --- | --- | --- | --- |
| D-001 | `speaker_bank_diarization_tuning` | Add an overlap-specialized known-speaker activity model before speaker-bank embedding assignment. | Existing tuning improves non-overlap/clean cases but does not solve mixed-overlap words. | CIT-004, CIT-005, CIT-013 |
| D-002 | `speaker_id_oracle_mask_sweep` | Treat oracle mask as a ceiling and target extraction as stage 2, not as proof that any generic separator will work. | Literature and local runs show source permutation/domain mismatch must be handled explicitly. | CIT-001, CIT-002, CIT-006, CIT-009, CIT-013 |
| D-003 | current ASR pipeline | Defer full speaker-attributed ASR until modular activity/extraction experiments plateau. | SA-ASR is promising but too invasive for the next narrow experiment. | CIT-011, CIT-012 |

## 9. Locked Defaults For Coding (Required)

| default_id | field | locked_value | reason | citation_ids |
| --- | --- | --- | --- | --- |
| L-001 | `experiment_kind` | `one_hot_target_extraction_plus_word_ownership_objective` | One-hot target extraction is the first positive learned extractor, but deployable post-hoc selectors do not recover its oracle candidate gain. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-002 | `model_output` | `per_speaker_frame_activity_probability` | Supports overlap as multi-label activity instead of forcing one speaker per frame. | CIT-004 |
| L-003 | `speaker_conditioning` | `speaker_bank_enrollment_embedding_or_reference` | Uses known speaker data explicitly. | CIT-001, CIT-004, CIT-005 |
| L-004 | `primary_eval` | `same_row_word_speaker_accuracy` | Prevents SDR-only or source-quality-only success. | CIT-009, CIT-013 |
| L-005 | `hard_eval_slice` | `target_share <= 0.90` | Measures the overlap failure mode that dominates the gap. | CIT-013 |
| L-006 | `promotion_rule` | `pass_acceptance_gates_before_pipeline_integration` | Keeps experimental models from adding complexity without real speaker-label gains. | CIT-009, CIT-013 |
| L-007 | `fixed_channel_owner_supervision` | `retired_as_current_selector_fix` | Stronger all-source fixed speaker channels improve true-channel and oracle scores, but do not produce a deployable owner signal. | CIT-001, CIT-002, CIT-003, CIT-009, CIT-013 |
| L-008 | `mask_activation_swap` | `retired_as_current_extractor_fix` | ReLU/softplus masks do not beat the larger/deeper sigmoid TasNet true-target benchmark. | CIT-001, CIT-002, CIT-003, CIT-013 |
| L-009 | `sequence_candidate_routing` | `retired_as_current_selector_fix` | Full-window BiLSTM tagging and action routing over current candidate features both underperform mixed LDA on hard overlap. | CIT-009, CIT-013 |
