# Speaker Breakthrough Big Bets Implementation Spec

## Scope

This spec replaces incremental speaker-ID experiments as the recommended path toward upper-90 word
speaker accuracy. The goal is not another selector over current mixed/candidate embeddings; the
goal is to change the modeling problem.

## Workstream A: External TSE Feasibility

Candidate priority:

1. ClearerVoice-Studio audio-only target-speaker extraction.
2. WeSep pretrained or trainable TSE recipes.
3. USEF-TSE if license and checkpoint access permit.
4. Penta ECAPA TSE only as a known retired baseline unless domain adaptation is attempted.

Current feasibility result:

- Runner: `scripts/run_usef_tse_manifest.py`
- Report: `docs/specs/speaker_usef_tse_feasibility.md`
- Candidate loaded: USEF-TSE TFGridNet from `/tmp/USEF-TSE`, checkpoint `ZBang/USEF-TSE`
  `chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar`.
- License: CC BY-NC 4.0, research-only unless relicensed or replaced.
- Held-out all-test manifest: 22 rows from Sessions 64/67.
- Result: raw mixture mean SI-SDR -7.814 dB, USEF estimate -7.269 dB, SI-SDRi +0.545 dB.
- Decision: external USEF-TSE is operational and shows target signal, but the gain is far below the
  threshold for production adoption. Treat it as a domain-adaptation seed/reference, not a solution.
- Domain adapter pilot:
  - Trainer: `scripts/train_usef_tse_domain_adapter.py`
  - Word-owner energy probe: `scripts/score_tse_word_ownership.py`
  - Speaker-attributed ASR probe: `scripts/score_tse_speaker_attributed_asr.py`
  - Forced-reference builder: `scripts/build_tse_forced_word_reference.py`
  - Training subset: Sessions 51/58, dev Session 65, 80 steps, 4 second chunks.
  - Held-out Session 64/67 SI-SDRi: +1.676 dB, improving over pretrained +0.545 dB.
  - Energy-track word ownership: pretrained 44.67%, domain adapter 47.38%, clean source tracks
    70.17% under the same coarse span timing.
  - One-window speaker-attributed ASR: clean source tracks 45.72% total accuracy, pretrained USEF
    tracks 18.31%, domain adapter v2 tracks 21.31%. The extracted tracks over-generate words, and
    the clean-source result shows approximate span timing is not good enough for final scoring.
  - Forced-aligned four-window held-out ASR diagnostic: MMS_FA aligns 3,586/3,630 held-out tokens.
    Clean source tracks show 93.67% same-speaker temporal coverage at 0.96x prediction density.
    Pretrained USEF shows 94.92% same-speaker coverage but 3.74x prediction density. Domain adapter
    v2 shows 89.79% same-speaker coverage and lowers density to 3.05x.
  - Forced-mask training smoke:
    - Train/dev forced references for Sessions 51/58/65: 10,061/10,153 aligned words, 99.09%
      coverage.
    - Trainer now supports `--forced-reference-jsonl` and `--non-owner-weight`; non-owner aligned
      words become explicit suppression masks.
    - Held-out SI-SDRi improves from domain adapter v2's +1.676 dB to +1.897 dB.
    - Target/non-owner waveform energy improves from 2.31 dB to 2.55 dB, versus clean source
      9.08 dB.
    - ASR density moves from 3.05x to 2.96x, but same-speaker coverage slips from 89.79% to 89.21%
      and greedy ASR accuracy slips from 22.00% to 21.53%.
  - Suppression-aware selection update:
    - Non-owner suppression now excludes target-overlap regions.
    - Dev evaluation is deterministic across checkpoints within a run.
    - `--selection-metric dev_suppression_score` was tested. A sampled-dev run overfit sampled
      chunks and regressed held-out leakage to 2.38 dB. A fixed-dev run improved dev suppression
      from 2.87 to 2.93 dB, but held-out leakage reached only 2.46 dB and SI-SDRi +1.843 dB.
    - Decision: current USEF adapter training moves suppression slightly, but this model/objective
      shape has not produced a word-ownership breakthrough.
  - Decision: domain adaptation is promising for extraction quality, but a simple energy-owner
    decoder and extract-then-transcribe ASR remain far below target. Scaling training must be paired
    with forced word-level timing and a direct word-owner/suppression objective selected by a
    suppression-aware dev metric, or a joint diarization-ASR model.

Adapter contract:

```text
inputs:
  mixed waveform crop or full window
  enrollment waveform(s) for target speaker
  optional negative enrollment waveform(s)

outputs:
  extracted target waveform
  model metadata: checkpoint, license, sample rate, enrollment duration, runtime

score:
  embed extracted waveform with existing Titanet scorer
  compare same rows to mixed baseline, clean-source/oracle-mask ceiling, and true-target/source oracle
```

Go/no-go:

- Must load locally on WSL.
- Must accept known-speaker enrollment audio without retraining the repo around it.
- Must have usable code and checkpoint license for the intended use.
- Must beat same-row mixed hard-overlap baseline before adaptation work expands.

## Workstream B: Domain-Scale TSE Training

Training examples should come from a corpus manifest, not tiny hand-selected eval windows.

Initial Drive-backed manifest:

- Builder: `scripts/build_speaker_tse_manifest.py`
- Report: `docs/specs/speaker_tse_manifest_drive_v1.md`
- WSL artifact: `.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v1/speaker_tse_manifest.jsonl`
- Current split: train Sessions 58-60, dev Session 61, test Sessions 64/67.
- Current scope: 63 rows, 10,758 target words, no enrollment/window or held-out-session leakage.
- Full held-out materialization for external TSE scoring:
  `.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/speaker_tse_manifest.jsonl`
  with 22 materialized test rows and 3,624 held-out target words.
- Larger non-materialized inventory for domain adaptation:
  `.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v4_large_inventory/speaker_tse_manifest.jsonl`
  from Sessions 49-61 and 64-67. This has 327 rows total, 215 train rows, 67 dev rows, 45
  test rows, 32,234 train target words, 9,772 dev target words, 7,221 test target words, and a
  passing leakage audit.

Required fields:

| field | meaning |
| --- | --- |
| `mixture_path` | Mixed Discord/call audio or generated mixture aligned to source tracks. |
| `target_source_path` | Target speaker source waveform for reconstruction. |
| `speaker_id` | Closed-set known speaker identity. |
| `positive_enrollment_paths` | Target-present examples, preferably outside the eval window. |
| `negative_enrollment_paths` | Target-silent / interferer-present examples for the same window or session. |
| `word_spans` | Word start/end plus speaker owner for word-level objective. |
| `split_id` | Held-out session/window split identifier. |
| `overlap_bucket` | Target-share and active-speaker bucket for balanced sampling. |

Current forced reference artifact:

`.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/forced_word_reference_all_mms/forced_word_reference_groups.jsonl`

This covers the four materialized held-out windows with 98.79% aligned-token coverage. The next
training manifest should extend the same MMS forced-alignment reference to train/dev windows before
adding a word-owner loss.

Current train/dev forced-reference artifact:

`.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v4_large_inventory/forced_word_reference_s51_s58_s65_mms/forced_word_reference_groups.jsonl`

This covers Sessions 51/58/65 with 99.09% aligned-token coverage and is the first usable supervision
artifact for non-owner suppression training.

Objective shape:

```text
loss =
  source_reconstruction_loss(target waveform)
  + speaker_presence_or_word_owner_loss(center word / word span)
  + interferer_only_suppression_loss(forced aligned non-target words outside target-active overlap)
  + overlap_aware_activity_loss(forced target and non-target words active together)
  + optional ASR/phonetic consistency loss
  + optional negative-enrollment contrastive loss
```

The owner loss must be part of model training, not a post-hoc selector.

## Workstream C: Speaker-Attributed ASR

Treat this as a separate architecture path:

- Do not attach another classifier to Whisper words and call it joint ASR.
- Build manifests that pair audio with words, timings, speaker IDs, and optional speaker profiles.
- Evaluate diarization-ASR models by word speaker accuracy and speaker-attributed transcript error.
- Initial baselines can use Sortformer-style diarization plus ASR, but the breakthrough attempt
  should learn speech content and speaker attribution together.
- Speaker-tagged ASR smoke:
  - Added `scripts/score_speaker_tagged_asr.py` to score anonymous `[S0]`-style ASR outputs
    against forced-reference words with normalized lexical sequence alignment and per-clip oracle
    tag-to-known-speaker mapping.
  - Built a four-clip held-out smoke set at `outputs/sa_asr_smoke_v1/sa_asr_smoke_manifest.jsonl`
    from 30-second Session 64/67 slices. It contains 381 forced words and 86 overlap words.
  - `mrfakename/qwen3-asr-1.7b-ami-diarization-fft-r6-20260422` runs through `qwen-asr` on the
    RTX 5080 and emits tags for all clips. Output:
    `outputs/sa_asr_smoke_v1/qwen3_asr_1p7b_ami_diarization/qwen3_asr_speaker_tagged_raw.jsonl`.
  - Scored output:
    `outputs/sa_asr_smoke_v1/qwen3_asr_1p7b_ami_diarization_score_v1/speaker_tagged_asr_summary.json`.
    Aggregate metrics are 52.23% many-to-one lexical speaker accuracy, 37.27% one-to-one,
    53.28% lexical coverage, 43.02% overlap accuracy, and 36.65% prediction precision proxy.
  - Failure modes: the model often maps matched words to coherent anonymous speakers, but misses too
    many reference words, predicts extra anonymous speakers on two-speaker clips, and entered a
    repetition loop on one three-speaker clip. Do not promote this checkpoint without domain
    fine-tuning or decoding controls.
  - Domain adaptation smoke:
    - Added `scripts/build_speaker_tagged_asr_dataset.py` to export forced-word Qwen-style
      first-appearance `[S0]` training clips.
    - Added `scripts/train_qwen_speaker_tagged_lora.py` to run LoRA adaptation and immediate
      held-out prediction export.
    - Train set: `outputs/speaker_tagged_asr_train_forced_v1_max4_120`, with 120 train49-62 clips,
      9,399 forced words, and at most four speakers per 30-second clip.
    - LoRA feasibility: one-sample supervised loss works, `q_proj,v_proj` LoRA trains 1.20M
      parameters, and peak GPU memory is roughly 5-7 GB on the RTX 5080.
    - Result: `outputs/qwen3_speaker_tagged_lora_domain_smoke_v1` at 40 steps / 1e-4 collapses to
      30.45% many-to-one and 8.14% overlap accuracy. The gentle
      `outputs/qwen3_speaker_tagged_lora_domain_smoke_v2_gentle` run at 5 steps / 1e-5 scores
      51.97% many-to-one and 43.02% overlap, essentially matching zero-shot and not fixing
      repetition.
    - Decision: naïve tiny LoRA is not the breakthrough. A future SA-ASR attempt needs a stronger
      runnable base model, more careful validation/checkpointing/decoding, or a larger
      domain-supervised training recipe.
  - TagSpeech-AMI is a stronger conceptual SA-ASR candidate, but the current HF snapshot lacks the
    quick-start code files named in its model card. Rev Reverb diarization V2 is gated with current
    credentials.
- Current Sortformer probe:
  - Scorer: `scripts/score_sortformer_oracle_diarization.py`
  - Best official checkpoint tested: `nvidia/diar_streaming_sortformer_4spk-v2.1`.
  - Held-out Sessions 64/67: 64.92% many-to-one oracle accuracy overall, 76.22% on
    high-confidence non-overlap words.
  - `nvidia/diar_sortformer_4spk-v1` reaches 64.73% overall / 72.79% high-confidence
    non-overlap; `nvidia/diar_streaming_sortformer_4spk-v2` reaches 60.68% / 71.09%.
  - Widened checkpoint update: `devsy0117/ultra_diar_streaming_sortformer_5spk_v1` reaches 69.88%
    overall / 81.17% high-confidence non-overlap, and
    `mago-ai/ultra_diar_streaming_sortformer_8spk_v1` reaches 68.33% / 78.57%.
  - Domain-adapter update: `scripts/chunk_sortformer_manifest.py` and
    `scripts/train_sortformer_domain_adapter.py` make the NeMo fine-tune path reproducible. A
    300-second one-step 8-speaker smoke ran out of memory in the ATS/PIL permutation loss; 30-second
    chunks train cleanly.
  - 8-speaker adapter result: 50 steps reaches 70.63% overall / 81.28% high-confidence non-overlap;
    420 steps reaches 71.71% overall / 82.87% high-confidence non-overlap.
  - 5-speaker filtered adapter result: training
    `devsy0117/ultra_diar_streaming_sortformer_5spk_v1` on <=5-speaker chunks reaches 72.36%
    overall, 71.90% one-to-one, 84.46% high-confidence non-overlap many-to-one, and 85.25%
    one-to-one. This is the current best Sortformer-family result.
  - 5-speaker-to-8-speaker widening update: `scripts/widen_sortformer_checkpoint.py` creates
    `outputs/sortformer_5spk_to_8spk_widened_v1/sortformer_widened_final.nemo` by loading the
    8-speaker checkpoint as the target, copying all same-shaped tensors from the 5-speaker
    checkpoint, copying the first five rows of the speaker-output projections, and keeping the extra
    three rows from the 8-speaker checkpoint.
  - Widened result: raw widened scoring reaches 69.10% overall / 80.05% high-confidence
    non-overlap. A 50-step full-domain adapter reaches 71.73% overall / 83.09% high-confidence
    non-overlap. A 420-step full pass regresses to 69.30% / 81.71%.
  - Decision: relaxing the four-speaker cap and adapting a real meeting backbone both help, but
    anonymous Sortformer plus oracle cluster mapping still falls short of the clean-source oracle
    regime. The widened initialization is useful but not enough; the next credible path is
    forced-word train references or known-speaker word-owner conditioning, not more local selectors
    or more epochs on coarse train spans.
- NeMo export update:
  - Exporter: `scripts/export_sortformer_training_data.py`.
  - Chunker: `scripts/chunk_sortformer_manifest.py`.
  - Trainer: `scripts/train_sortformer_domain_adapter.py`.
  - Widening utility: `scripts/widen_sortformer_checkpoint.py`.
  - Train handoff: `outputs/sortformer_nemo_export_train_v1/train_manifest.jsonl`, 42 windows,
    12,600 seconds, manifest-span RTTMs, max 7 speakers.
  - Dev/test handoff: `outputs/sortformer_nemo_export_devtest_v1/{dev,test}_manifest.jsonl`, 12
    windows, 3,600 seconds, forced-word RTTMs, max 6 speakers.
  - Chunked handoff: `train_chunks_30s_manifest.jsonl` has 420 rows; the max5 filter has 350 rows.
    `dev_chunks_30s_manifest.jsonl` has 60 rows; the max5 filter has 57 rows.
  - Decision: keep 30-second chunking for tractable Sortformer adaptation. Use the 8-speaker head
    for full-domain coverage, and prefer cleaner train labels or known-speaker ownership over more
    coarse-label training of the widened checkpoint.
- Forced-label Sortformer update:
  - Train forced references: `outputs/speaker_tse_drive_v5_49_67_devtest_materialized/forced_word_reference_mms_train49_62_fallback_v1/forced_word_reference_groups.jsonl`.
    The build covers 33,962/34,275 train tokens (99.09%), with 33,328 forced words and 634
    fallback words (1.85%).
  - Exported handoff: `outputs/sortformer_nemo_export_train_forced_v1/train_manifest.jsonl`, 42
    train windows, 12,600 seconds, all using forced-word references; 30-second chunking yields 420
    rows with active-speaker distribution `{1:5, 2:22, 3:93, 4:129, 5:128, 6:40, 7:3}`.
  - 8spk forced 50-step score:
    `outputs/sortformer_8spk_forcedtrain_domain_adapter_30s_50step_v1_oracle_test` reaches
    72.07% all-word many-to-one, 69.15% one-to-one, 78.82% non-overlap, and 84.24%
    high-confidence non-overlap.
  - 8spk forced 420-step score:
    `outputs/sortformer_8spk_forcedtrain_domain_adapter_30s_420step_v1_oracle_test` reaches
    72.76% all-word many-to-one, 69.12% one-to-one, 80.15% non-overlap, and 86.88%
    high-confidence non-overlap.
  - Expanded forced-label 630-step score:
    `outputs/sortformer_8spk_forcedtrain_v6_43_62_65_domain_adapter_30s_630step_v1_oracle_test`
    trains on Sessions 43-62 plus 65 via 630 thirty-second chunks and reaches 71.77% all-word
    many-to-one, 69.73% one-to-one, 78.39% non-overlap, and 84.53% high-confidence non-overlap.
    Estimated overlap accuracy rises to 47.99%, but the aggregate/high-confidence regression means
    the extra sessions did not beat the cleaner train49-62 forced-label run.
  - Decision: forced-word train labels are better supervision than coarse manifest spans, and the
    cleaner train49-62 run sets the best aggregate/high-confidence Sortformer scores, but
    overlap-word ownership remains about 46.19%. Do not mistake the non-overlap lift for a 90%
    path; the next model must optimize known-speaker word ownership or joint speaker-attributed ASR
    directly.
- Current repo probe: transcribing each extracted/source speaker track independently is not enough.
  On one Session 64 held-out window, domain-adapted USEF improves over pretrained USEF but still
  scores only 21.31% total word-speaker accuracy, and clean source tracks score 45.72% with the
  approximate reference timing. This branch needs forced alignment or a model that learns speaker
  ownership while learning words.
- Forced-reference update: with MMS word timings, the same branch shows the target words are often
  recoverable but not selectable. Across four held-out windows, domain-adapted USEF has 89.79%
  same-speaker temporal coverage but 3.05x prediction density. A viable SA-ASR/TSE hybrid must
  suppress non-target decoded words, not merely expose target speech somewhere in the estimate.
- Frozen-feature update: `scripts/train_wavlm_word_owner_baseline.py` tested WavLM mixture and
  enrollment features with a direct word-owner head. It reached only 21.89% held-out accuracy when
  trained on train49-62 plus forced dev, despite fitting dev to 61.52%. Do not treat frozen WavLM
  pooling as the next architecture; use WavLM only inside an adapted speaker-conditioned
  activity/owner model or joint speaker-attributed ASR path.
- Temporal-frame update: `scripts/train_sequence_tsvad_word_owner_baseline.py` tested a dilated
  TCN over enrollment-conditioned frame sequences. It reached 39.00% held-out accuracy, below the
  independent-frame TS-VAD MLP. Do not spend more effort on local frame-BCE sequence capacity unless
  the objective changes to direct word ownership or the encoder/backbone changes materially.
- Forced-label temporal-frame update: the same sequence TS-VAD family trained on train49-62 forced
  references with 1,200 capped sequences reaches only 34.30% held-out word-owner accuracy and
  27.68% overlap accuracy in
  `outputs/sequence_tsvad_word_owner_forcedtrain_probe_v1`. This rejects the local
  from-scratch log-spectrum TCN as the missing piece even when labels are cleaner.
- Listwise-owner update: `scripts/train_listwise_word_owner_baseline.py` tested a masked softmax
  over all speaker candidates per forced word. It fit forced dev to 69.67%, but reached only 36.98%
  held-out accuracy when trained on train49-62 plus forced dev. Do not spend more effort on local
  candidate-MLP objective variants; the next attempt needs a materially stronger meeting,
  speaker-conditioned, or speaker-attributed ASR backbone.

### Target-Speaker Whisper Update

- `scripts/train_whisper_speaker_sot.py` and `scripts/run_whisper_speaker_sot.py` implement a
  fixed-roster serialized-output Whisper experiment with atomic one-token speaker controls. The
  atomic controls avoid Whisper's built-in punctuation suppression and the serializer preserves
  source FIFO order through overlap.
- Its domain-scale run uses 17.5 hours from Sessions 43-62 and 65. It reaches 75.77% on a training
  window but only 14.64% attributed-word accuracy on ten Session 63 chunks. This path is retired
  because fixed speaker IDs without enrollment overfit session/channel cues.
- `scripts/run_se_dicow_target_asr.py` runs the official revision-pinned `BUT-FIT/SE-DiCoW`
  checkpoint in separate per-enrollment passes and scores target-blind and oracle-activity modes.
- `scripts/train_se_dicow_domain_adapter.py` adapts LoRA plus the full 212.1M-parameter speaker
  communication block. It filters missing positive enrollment examples, supports non-materialized
  multitrack manifests, and keeps all optimizer-owned weights FP32 under CUDA autocast.
- A real precision bug was found: direct bfloat16 optimization at 2e-6 produced nearly inert full
  conditioning checkpoints. After the fix, saved deltas are material, but held-out performance is
  still flat at 33.33% blind / 46.08% oracle for 500 steps at 2e-6. A bounded 200-step 1e-5 run
  regresses blind recall to 32.35%.
- Added that joint path: the trainer now supervises target and non-target frame activity on the
  enrollment-conditioned encoder alongside target-ASR, while the runner performs a second decode
  from the predicted four-state mask and records frame precision/recall plus ambiguity rate.
- Result: threshold 0.30 reaches 47.06% attributed recall / 57.14% overlap recall on the hard clip,
  but across ten held-out chunks it reaches only 44.58% attributed recall, 38.36% overlap recall,
  and 18.07% precision. Target-frame recall is 97.58% with only 13.02% precision, so the linear head
  learned generic speech rather than transferable speaker identity.
- Preserve the two-pass inference and confidence instrumentation, but do not continue the local
  linear pVAD head. Replace it with a pretrained, structurally enrollment-aware personal-VAD/TSE
  network. At inference, low-margin simultaneous candidates must remain ambiguous rather than being
  forced into one name.

## Workstream D: Corpus Inventory

Before another serious model training run, produce:

- Total labeled audio hours by speaker.
- Multitrack hours by session.
- Overlap hours by active-speaker count.
- Word counts by speaker and target-share bucket.
- Proposed train/dev/test split with held-out sessions and balanced overlap.
- Leakage audit: no source crops from a held-out eval window in training.

## Stop Rules

Do not spend further effort on:

- Shallow selectors over current candidate embeddings.
- Frozen speaker embedding swaps.
- Temporal smoothing or phrase aggregation over current embeddings.
- Confidence routers over current candidate scores.
- Crop-length tweaks as a standalone experiment.
- Small TasNet/U-Net architecture variants without domain-scale data and a new objective.

## Acceptance Gates

| gate_id | requirement |
| --- | --- |
| BB-G-001 | Same-row hard-overlap accuracy beats mixed by at least 10 points before production integration. |
| BB-G-002 | Held-out-session validation is reported separately from leave-window validation. |
| BB-G-003 | Model closes at least half the gap between mixed baseline and oracle-mask ceiling before being called a breakthrough. |
| BB-G-004 | License and model-card constraints are documented before any external checkpoint is adopted. |
| BB-G-005 | Corpus manifest and leakage audit exist before a domain-scale training run. |
| BB-G-006 | Ordinary, non-overlapped conversation reaches at least 90% named-speaker word accuracy, with at least 95% precision on words emitted as high confidence. |
| BB-G-007 | Brief interruptions and moderate overlap are reported separately and reach at least 70% named-speaker word recall before the system is described as strong on overlap. |
| BB-G-008 | Ambiguous crosstalk can abstain or emit multiple candidates; confidence must be calibrated on held-out sessions rather than treated as a cosmetic score. |
| BB-G-009 | The review queue contains no more than 10% of session words while capturing at least 80% of residual attribution errors, so correction is occasional rather than session-wide relabeling. |

The operating objective is therefore a selective named-speaker transcript, not forced certainty on
every word. The primary score remains all-word named-speaker accuracy, but production promotion also
requires coverage-risk curves for confidence and a per-session review-burden report.
