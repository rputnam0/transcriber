# Speaker Breakthrough Big Bets Decision Report

## Decision Summary

- Do not continue shallow selectors, embedding swaps, smoothing, crop-length tweaks, confidence
  routers, or small TasNet/U-Net variants.
- Do not conclude that 90%+ word speaker accuracy is impossible; local oracle-mask and clean-source
  results show the signal exists.
- Do conclude that the current modular pipeline shape is unlikely to reach upper-90 accuracy via
  incremental post-hoc tweaks.
- Selected next work: build the data/license/model feasibility foundation for a bigger bet, starting
  with a serious external TSE feasibility check and a domain-scale corpus inventory.

## Variant Comparison

| variant_id | description | expected upside | risk | decision | evidence |
| --- | --- | --- | --- | --- | --- |
| BB-V-001 | Domain-scale target-speaker extraction with direct enrollment context and word-ownership loss | Best match to oracle-mask evidence and user hypothesis | Requires corpus engineering and real training budget | selected | BB-C-001, BB-C-003, BB-C-004, BB-C-005, BB-C-006 |
| BB-V-002 | Speaker-attributed ASR or joint diarization-ASR | Learns words and speakers jointly rather than assigning speaker labels after Whisper | Larger rewrite and label-prep burden | selected as parallel research track | BB-C-008, BB-C-009 |
| BB-V-003 | Larger, cleaner multitrack train/eval corpus | Reduces tiny-window overfitting and improves held-out validation | Requires data inventory and manifest work | prerequisite | BB-C-001, BB-C-003 |
| BB-V-004 | Serious pretrained modern TSE/meeting model adaptation | Fastest chance of a non-toy breakthrough if weights and license fit | External model mismatch/licensing may block | selected for feasibility spike | BB-C-004, BB-C-005, BB-C-006, BB-C-007 |
| BB-V-005 | More post-hoc selectors or small local variants | Cheap to run | Already repeatedly fails to harvest oracle candidate signal | retired | BB-C-002 |

## Locked Defaults

| default_id | field | locked_value | reason |
| --- | --- | --- | --- |
| BB-L-001 | banned_next_work | `shallow_selector_embedding_swap_smoothing_crop_tweak_confidence_router_small_tasnet_unet` | This branch is mined out in local evidence. |
| BB-L-002 | primary_metric | `word_speaker_accuracy_on_flattened_audio` | Signal-quality metrics alone can be misleading for conversation audio. |
| BB-L-003 | promotion_gate | `held_out_session_hard_overlap_gain_and_oracle_gap_closure` | Bigger models must prove they generalize beyond tiny eval-stem windows. |
| BB-L-004 | TSE_conditioning | `direct_enrollment_context_positive_negative_or_cross_attention` | Static centroids and frozen embeddings were not enough locally. |
| BB-L-005 | corpus_policy | `manifest_before_training` | A serious training run needs balanced sessions, speakers, and overlap examples. |

## Immediate Next Actions

1. Create a WSL corpus inventory report from all labeled multitrack/domain audio.
2. License-check and load-test ClearerVoice, WeSep, and USEF-TSE candidates.
3. Pick one external TSE candidate for a thin adapter smoke: enrollment + mixed crop in, extracted waveform out, repo word-speaker score out.
4. In parallel, design the domain-scale TSE manifest: target/enrollment/negative-enrollment/source waveform triples plus word-owner labels.
5. Treat speaker-attributed ASR as a separate architecture spike, starting from Sortformer/Reverb/JEDIS-style options rather than Whisper post-labeling.

## Drive Holdout Update

- Google Drive `Dnd/audio` discovery and WSL caches now cover Sessions 64-67.
- A scorer bug was fixed: predicted words are consumed once, so matched words can no longer exceed
  predicted words.
- Under the corrected scorer, the old best cached baseline is 62.36% total word-speaker accuracy,
  not 69.43%.
- The same control profile on new six-speaker held-out Drive sessions scores 49.55% on Session 64
  and 52.61% on Session 67.
- The failure pattern is consistent with this report's decision: pyannote/profile post-labeling is
  collapsing identities after flat mixed ASR, while word evidence has already been dropped or merged.

## External TSE Feasibility Update

- USEF-TSE TFGridNet was loaded from the official repo/checkpoint path and run against a held-out
  manifest with 22 materialized test rows from Sessions 64/67.
- License is CC BY-NC 4.0, so this remains research-only unless replaced or relicensed.
- Best checkpoint tested was `chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar`.
- Overall held-out audio metric: mixture SI-SDR -7.814 dB, USEF estimate -7.269 dB, SI-SDRi +0.545
  dB.
- The gain is real but too small and uneven for adoption: Cyrus and David improve, low target-share
  speakers often regress, and this does not yet answer word ownership.
- Updated decision: serious pretrained TSE is worth using as an adaptation seed/reference, but an
  off-the-shelf model does not break through. The next bet must train on the domain manifest with
  direct enrollment and word-owner supervision.

## Domain Adapter Update

- Added `scripts/train_usef_tse_domain_adapter.py` to fine-tune USEF-TFGridNet on manifest chunks
  with direct enrollment, target-source reconstruction, and transcript-derived activity weighting.
- Added `scripts/score_tse_word_ownership.py` to probe whether extracted speaker tracks can assign
  labeled transcript spans by energy.
- Added `scripts/score_tse_speaker_attributed_asr.py` to transcribe extracted/source speaker tracks
  independently and score the resulting speaker-attributed words.
- Added `scripts/build_tse_forced_word_reference.py` to build CTC forced-aligned word references
  from clean materialized speaker stems with torchaudio `MMS_FA`.
- Pilot training on Sessions 51/58 with dev Session 65 improves held-out Session 64/67 SI-SDRi from
  pretrained USEF's +0.545 dB to +1.676 dB.
- The same pilot improves energy-track word ownership from 44.67% to 47.38%, while clean source
  tracks score 70.17% under the same coarse span timing.
- In a one-window Session 64 speaker-attributed ASR probe, clean source tracks scored 45.72% total
  accuracy, pretrained USEF tracks 18.31%, and domain adapter v2 tracks 21.31%. The adapter again
  beats the pretrained extractor, but extracted-track ASR over-generates and the clean-source result
  exposes the need for better word timing before using this as a promotion metric.
- MMS forced alignment on the four materialized held-out Session 64/67 windows aligned 3,586/3,630
  transcript tokens. With those references, clean source tracks have 93.67% same-speaker temporal
  coverage at 0.96x prediction density, showing that the stretch target is plausible when track
  ownership is known. Pretrained USEF has 94.92% same-speaker coverage but 3.74x prediction density;
  domain adapter v2 has 89.79% coverage and 3.05x density. The extracted estimates contain target
  evidence, but they do not suppress non-owner speech.
- Extended forced alignment to the prior train/dev pilot sessions: Sessions 51/58/65 now have
  10,061/10,153 aligned train/dev words. Added forced target masks and non-owner suppression masks
  to `scripts/train_usef_tse_domain_adapter.py`, plus `scripts/score_tse_forced_mask_energy.py`.
- A 20-step forced-mask smoke from the v2 checkpoint improves held-out SI-SDRi from +1.676 dB to
  +1.897 dB and improves target/non-owner waveform energy from 2.31 dB to 2.55 dB. ASR density
  improves only slightly from 3.05x to 2.96x, while same-speaker coverage slips from 89.79% to
  89.21% and greedy ASR accuracy slips from 22.00% to 21.53%.
- Added suppression-aware checkpoint selection to the trainer and fixed two selection problems:
  non-owner suppression no longer penalizes target/non-owner overlap regions, and dev chunks are
  deterministic across checkpoints. A sampled-dev selector produced a misleading 7.00 dB dev score
  but regressed held-out leakage to 2.38 dB. A fixed-dev selector was valid but only reached 2.46 dB
  held-out leakage and +1.843 dB SI-SDRi, still below the earlier forced-mask smoke.
- Decision update: domain adaptation clearly moves the extraction signal, but energy decoding is
  not enough, and extract-then-transcribe ASR is not enough. Forced masks move the right waveform
  leakage metric but are not yet a word-ownership breakthrough. The current USEF adapter objective
  is producing incremental gains, not the desired accuracy jump. The next scaled run needs a larger
  deterministic dev set and a stronger word-owner/suppression head, or a move to
  speaker-attributed ASR/joint diarization-ASR.
- External audit incorporated: the next owner-deciding model should be multi-label target-speaker
  activity plus word ownership, with target-speaker extraction demoted to an auxiliary/enhancement
  branch. Patched the P0 metric/runner issues before using the larger Drive eval: independent
  lexical-aware ASR scoring with `speaker_attributed_lexical_accuracy` as the primary ASR
  attribution metric, ASR cache config hashing, SI-SDR padding, measured self-reference SI-SDR,
  overlap-add inference metadata/resume handling, explicit interferer-only and overlap masks,
  nearest-neighbor mask resampling, and forced-word ownership scoring.
- Built a larger WSL Drive-backed manifest from `/mnt/g/My Drive/DND/Audio`: train Sessions 49-62
  (221 rows, 33,981 target words), dev Sessions 63/66 (33 rows, 5,042 words), and test Sessions
  64/67 (34 rows, 5,397 words). Forced MMS references aligned 10,401/10,470 dev/test tokens
  (99.34%).
- On the 67-row dev/test eval, clean source tracks have 9.42 dB target/non-owner energy and 77.23%
  forced-word owner accuracy. The best current forced-mask extractor improves SI-SDRi to +2.30 dB
  but only reaches 2.60 dB target/non-owner energy and 50.18% forced-word owner accuracy. This is a
  stronger no-go for extract-then-independent-ASR as the owner-deciding mechanism.
- Added a Drive-backed TS-VAD-style frame activity harness,
  `scripts/train_tsvad_word_owner_baseline.py`. Training only on forced dev windows fit the dev split
  to 66.40% but fell to 23.47% on held-out Sessions 64/67. Training on non-materialized Drive train
  Sessions 49-62 plus forced dev improved held-out forced-word ownership to 41.82%, still below the
  old post-hoc diarization line.
- Added a direct enrollment-conditioned word-owner head,
  `scripts/train_direct_word_owner_baseline.py`. With the same train49-62+dev data, it reached
  40.92% held-out forced-word ownership. This rejects hand-built spectral enrollment features and
  small MLP heads as the breakthrough path, even when the objective is word ownership.
- Added `scripts/train_wavlm_word_owner_baseline.py` to test a stronger frozen speech backbone for
  the same word-owner objective. Frozen `microsoft/wavlm-base-plus` features with enrollment
  profiles can fit the forced dev windows to 61.52%, but dev-to-test is 13.77%, and train49-62 plus
  forced dev reaches only 21.89% on held-out Sessions 64/67. This rejects "swap log spectra for
  frozen WavLM features" as the missing architecture step. The pretrained representation must be
  adapted inside a speaker-conditioned activity/owner model, or replaced by a real meeting/SA-ASR
  backbone.
- Added `scripts/train_sequence_tsvad_word_owner_baseline.py` to test a temporal
  enrollment-conditioned TS-VAD model with dilated convolutions over frame sequences. It reaches
  only 43.73% on forced-dev self-eval and 39.00% on held-out Sessions 64/67 with train49-62 plus
  forced dev, below the independent-frame TS-VAD MLP. This rejects local frame-BCE sequence capacity
  as the next breakthrough candidate.
- Added `scripts/train_listwise_word_owner_baseline.py` to test whether the direct owner objective
  was the remaining weak point in the local model family. A masked softmax over all enrolled speaker
  candidates fits forced dev best among local owner models at 69.67%, but held-out transfer remains
  poor: 28.55% from dev-only training and 36.98% from train49-62 plus forced dev. Objective shape
  alone is not enough; the bottleneck is representation/backbone transfer.
- Added forced-word train supervision for the serious Sortformer path. MMS/fallback references now
  cover 33,962/34,275 train tokens (99.09%; 1.85% source-span fallback). Ultra-Sortformer 8spk
  trained on these forced-word RTTMs reaches a new best aggregate Sortformer score at 72.76%
  all-word many-to-one and 86.88% high-confidence non-overlap, but overlap-word ownership remains
  about 46.19%. Cleaner labels help, especially away from overlap, but the anonymous diarization
  pipeline still does not approach the 90%+ target.
- Tested the raw corpus-scale hypothesis by expanding forced-label Sortformer training to Sessions
  43-62 plus 65. The handoff grows to 63 train windows / 18,900 seconds / 630 thirty-second chunks,
  but the added forced references are noisier, with 1,351 fallback words in the new Sessions 43-48
  plus 65 subset. The 630-step Ultra-Sortformer 8spk run reaches 71.77% all-word many-to-one,
  69.73% one-to-one, 84.53% high-confidence non-overlap, and about 47.99% estimated overlap
  accuracy. This slightly improves overlap versus the 420-step forced-label champion but regresses
  aggregate and non-overlap accuracy, so more Drive hours alone are not the breakthrough.
- Added a lightweight speaker-attributed ASR smoke using
  `mrfakename/qwen3-asr-1.7b-ami-diarization-fft-r6-20260422` on four 30-second held-out
  Session 64/67 clips. The model is Apache-2.0 and runnable on the RTX 5080, and it emitted
  `[S0]`-style tags on every clip. Scored with lexical sequence alignment and per-clip oracle
  tag-to-known-speaker mapping, it reached only 52.23% many-to-one accuracy over 381 forced words,
  53.28% lexical coverage, 43.02% overlap accuracy, and 36.65% prediction precision proxy. One
  clip entered a repeated "HE'S THE ONE WHO" loop. This confirms joint ASR/diarization is worth
  keeping as a family, but the runnable lightweight AMI finetune is not competitive without
  domain adaptation and repetition/coverage controls.
- Tested that domain adaptation path directly with Qwen LoRA. A forced-word Qwen-style train set
  now exists at `outputs/speaker_tagged_asr_train_forced_v1_max4_120`: 120 train49-62 clips,
  9,399 forced words, and at most four speakers per 30-second clip. A one-sample supervised loss
  and LoRA step fit on the RTX 5080, using about 1.20M trainable parameters. The first smokes were
  negative: 40 steps at 1e-4 collapsed generation to 30.45% many-to-one and 8.14% overlap accuracy,
  while a gentle 5-step 1e-5 run scored 51.97%, essentially zero-shot and still repetitive. This
  retires naïve tiny LoRA on the AMI finetune, not SA-ASR as an architecture class.
- Checked stronger/current SA-ASR feasibility. TagSpeech-AMI is architecturally closer to the goal
  because it emits timestamped speaker-attributed segments, but the HF snapshot available here does
  not include the `model.py` and `utils/xml_utils.py` files referenced by its model-card quick
  start. Rev Reverb diarization V2 is gated with current HF credentials. Both remain candidates
  only after access/packaging is resolved.
- Ran a bounded sequence TS-VAD/word-owner probe on the same forced train labels. The
  from-scratch enrollment-conditioned log-spectrum TCN reaches only 34.30% held-out word-owner
  accuracy and 27.68% on overlap. This keeps TS-VAD/word ownership as the right architectural
  direction, but rejects the local lightweight implementation as the breakthrough path.
- Added `scripts/score_pyannote_oracle_diarization.py` to measure the production diarizer's ceiling
  with oracle cluster-to-known-speaker mapping. On held-out Sessions 64/67, pyannote Community-1 with
  a free speaker count reaches 66.54% many-to-one oracle accuracy and 63.41% one-to-one oracle
  accuracy. Forcing the candidate speaker count is worse at 63.87% many-to-one / 58.44% one-to-one.
  Non-exclusive turns are also slightly worse at 63.41% many-to-one.
- Added `scripts/score_sortformer_oracle_diarization.py` to test NeMo Sortformer as a serious
  pretrained meeting-diarization baseline. The official runnable checkpoints are four-speaker
  models, which mismatches held-out Drive windows with five or six known speakers. The best
  checkpoint tested, `nvidia/diar_streaming_sortformer_4spk-v2.1`, reaches 64.92% many-to-one
  oracle accuracy overall and 76.22% on high-confidence non-overlap words. Offline 4spk v1 reaches
  64.73% overall / 72.79% high-confidence non-overlap; streaming 4spk v2 reaches 60.68% / 71.09%.
  A widened community Ultra-Sortformer 5spk checkpoint then reached 69.88% overall / 81.17%
  high-confidence non-overlap, and the 8spk checkpoint reached 68.33% / 78.57%. This beats pyannote
  overall and proves the speaker cap matters, but it is still below the clean-source oracle regime.
  Sortformer remains a candidate backbone only if adapted beyond anonymous cluster assignment.
- The clean-source forced-word energy oracle is 75.28% on the held-out test split alone. Because the
  clean tracks have large positive target/non-owner energy margins but still miss roughly a quarter
  of forced words, the eval ceiling is now partly a data/timing/source-mapping audit problem, not
  only a model problem. A 90% promotion gate needs cleaner word ownership references or a metric that
  distinguishes true model confusion from forced-alignment/source-label noise.
- Added `scripts/export_sortformer_training_data.py` to turn the Drive windows into NeMo
  Sortformer manifests and RTTMs. The WSL export contains 42 train windows / 12,600 seconds from
  Drive zips with manifest-span labels, plus 12 dev/test windows / 3,600 seconds with forced-word
  labels. Train has 27 six-speaker and 9 seven-speaker windows, so the first full-domain
  Sortformer adaptation used the 8-speaker head while the stronger 5-speaker checkpoint required a
  max5 filter.
- Added `scripts/chunk_sortformer_manifest.py` and `scripts/train_sortformer_domain_adapter.py` for
  the first real Sortformer domain-adaptation runs. Full 300-second windows are not tractable with
  the 8-speaker ATS/PIL permutation loss, but 30-second chunks train cleanly. The 8-speaker adapter
  improves from 68.33% to 71.71% held-out many-to-one oracle accuracy and from 78.57% to 82.87% on
  high-confidence non-overlap words. The filtered 5-speaker adapter is the current best
  Sortformer-family result at 72.36% overall, 71.90% one-to-one, 84.46% high-confidence non-overlap
  many-to-one, and 85.25% one-to-one. This is the first positive larger-bet result, but it still
  relies on oracle cluster mapping and remains below the clean-source top-two regime.
- Added `scripts/widen_sortformer_checkpoint.py` to test whether the stronger 5-speaker checkpoint
  can be used as the initialization for a full 8-speaker model. The transplant is clean: all
  same-shaped tensors copy, and only the two speaker-output projections are row-widened. The raw
  widened checkpoint scores 69.10% overall / 80.05% high-confidence non-overlap. A 50-step
  full-domain adapter reaches 71.73% overall, 70.20% one-to-one, and 83.09% high-confidence
  non-overlap, improving over the vanilla 8-speaker 50-step adapter. A 420-step run regresses to
  69.30% / 81.71%, so longer training on coarse manifest-span train RTTMs is not the breakthrough.
  This shifts the next Sortformer bet toward forced-word train RTTMs or explicit known-speaker
  word-owner supervision.
- Added `scripts/audit_clean_source_word_oracle.py` to split that clean-source result into
  single-winner ownership versus multi-label speaker activity. On held-out Sessions 64/67, clean
  source top-two accuracy is 91.06%, and oracle group/speaker activity thresholds recall the
  reference speaker on 91.14% of forced words. On high-confidence non-overlap words
  (`score >= 0.05`), clean-source winner accuracy is 91.76% and top-two is 99.42%, while pyannote
  free-count oracle mapping is still only 81.68%. This restores the 90% target as a plausible
  activity/word-owner goal on clean reference slices, but it also shows the existing pyannote-style
  modular pipeline is not close enough.

## Enrollment-Conditioned ASR Update

- Built a leakage-safe, domain-scale handoff from Sessions 43-62 and 65: 210 five-minute groups,
  2,100 unique 30-second mono chunks (17.5 hours), 1,156 candidate-speaker rows, and 157,303 target
  words. Sessions 63/66 are development; Sessions 64/67 stayed sealed.
- A fixed-roster Whisper SOT with atomic speaker tokens and direct owner weighting overfits badly:
  75.77% on a training window versus 14.64% attributed-word accuracy over ten held-out Session 63
  chunks. The 45.62% lexical recall but 32.09% matched-word speaker accuracy exposes speaker-label
  transfer, not ASR alone, as the failure.
- Official SE-DiCoW zero-shot reaches 48.71% target-word recall on the available oracle-activity dev
  set. On the selected hard clip it reaches 33.33% target-blind and 46.08% with oracle activity.
- Found and fixed a full-block training bug: optimizer-owned speaker-conditioning weights were
  bfloat16, making 2e-6 updates mostly unrepresentable. The trainer now promotes all 212.1M
  trainable parameters to FP32 and uses CUDA autocast for the frozen backbone.
- The precision-safe 500-step full-block run moves weights materially but remains 33.33% blind and
  46.08% oracle on the hard clip. A bounded 1e-5 run changes decoding but regresses blind recall to
  32.35%. This rejects under-training and precision as the remaining explanations.
- Decision: retain enrollment-conditioned target ASR as the primary architecture family, but
  retire the current `generic VAD -> SE-DiCoW target transcript` shape. The next credible model must
  jointly learn enrollment-conditioned personal VAD/word ownership and expose calibrated abstention
  for crosstalk. It must beat the 72.76% Sortformer champion on Sessions 63/66 before any sealed
  Session 64/67 run.
- Joint personal-VAD follow-up: a two-output target/non-target head was trained on the
  enrollment-conditioned SE-DiCoW encoder with a joint activity and target-ASR objective. Threshold
  0.30 reaches 47.06% attributed recall and 57.14% overlap recall on the selected hard clip, but
  only 20.09% target precision. Across ten Session 63 chunks it reaches 44.58% attributed recall,
  38.36% overlap recall, and 18.07% precision. Its frame behavior is generic-speech detection
  (97.58% target recall, 13.02% precision), not reliable identity-conditioned activity.
- Updated decision: keep the implemented two-pass mask, four-state overlap representation,
  activity metrics, and ambiguity output. Retire the local linear pVAD head. The next model must
  adapt a serious pretrained personal-VAD/TSE activity backbone, or inject speaker enrollment
  before and throughout the activity network as in USEF-TP, rather than probing final encoder
  states with another compact head.

## No-Go Check Result

| condition | status | evidence |
| --- | --- | --- |
| Missing source for the pivot away from incremental tweaks | pass | Local ledger documents repeated failures and oracle/clean-source recoverability. |
| Missing credible bigger-bet families | pass | External TSE, joint diarization-ASR, and corpus-scale training sources are identified. |
| Licensing unresolved for implementation | caution | External checkpoint licenses are explicitly open items before adoption. |
| Data scale unresolved for training | caution | Corpus inventory is the first required implementation action. |
| Risk of declaring 90% impossible | pass | Decision explicitly preserves upper-90 as a stretch target supported by oracle evidence. |

## Final Status

status = ready_for_implementation
