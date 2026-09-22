# Speaker USEF-TSE Feasibility

## Purpose

This records the first serious pretrained target-speaker extraction test on the Drive-backed
multitrack corpus. It is not another embedding selector test. The input is a flat mixed 300 second
conversation window plus direct target-speaker enrollment audio; the output is an extracted waveform
for that target speaker, scored against the clean source stem.

## Candidate

- Model family: USEF-TSE, TFGridNet variant.
- Source checkout: `/tmp/USEF-TSE` on WSL, commit `bdaeb36`.
- Checkpoint source: Hugging Face `ZBang/USEF-TSE`.
- Tested checkpoints:
  - `chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar`
  - `chkpt/USEF-TFGridNet/wham!/temp_best.pth.tar`
  - `chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar`
- License: CC BY-NC 4.0, so this is research-only unless relicensed or replaced.
- Runtime: CUDA on WSL, 8 kHz model input, 20 second chunks, estimates resampled to 16 kHz for
  repo scoring.

## Repo Adapter

Added experimental runner:

```bash
uv run python scripts/run_usef_tse_manifest.py \
  --manifest .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/speaker_tse_manifest.jsonl \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/usef_tfgridnet_wsj_estimates_all22 \
  --usef-repo /tmp/USEF-TSE \
  --checkpoint-file "chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar" \
  --chunk-seconds 20 \
  --force
```

The runner intentionally keeps USEF external. It does not vendor the model or add the noncommercial
checkpoint as a project dependency.

## Held-Out Test Corpus

All-test materialized manifest:

`<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/speaker_tse_manifest.jsonl`

Build summary:

| split | rows | target words |
| --- | ---: | ---: |
| train | 31 | 5468 |
| dev | 10 | 1666 |
| test | 22 | 3624 |

Test sessions:

| session | rows |
| --- | ---: |
| Session 64 | 11 |
| Session 67 | 11 |

Leakage audit passed: no enrollment/window overlap and no held-out-session leakage.

## Results

Baseline score on all 22 held-out rows:

```bash
uv run python scripts/score_speaker_tse_manifest.py \
  --manifest .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/speaker_tse_manifest.jsonl \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/baseline_score
```

USEF-WSJ score:

```bash
uv run python scripts/score_speaker_tse_manifest.py \
  --manifest .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/speaker_tse_manifest.jsonl \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/usef_tfgridnet_wsj_score_all22 \
  --estimates-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/usef_tfgridnet_wsj_estimates_all22
```

Overall:

| rows | target words | mixture SI-SDR | estimate SI-SDR | SI-SDRi |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 3624 | -7.814 | -7.269 | +0.545 |

By session:

| session | rows | target words | mixture SI-SDR | estimate SI-SDR | SI-SDRi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Session 64 | 11 | 1912 | -7.640 | -6.801 | +0.839 |
| Session 67 | 11 | 1712 | -7.989 | -7.738 | +0.251 |

By speaker:

| speaker | rows | target words | mixture SI-SDR | estimate SI-SDR | SI-SDRi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cletus Cobbington | 2 | 148 | -11.075 | -11.245 | -0.170 |
| Cyrus Schwert | 4 | 694 | -6.532 | -5.265 | +1.267 |
| David Tanglethorn | 4 | 566 | -12.487 | -10.672 | +1.815 |
| Dungeon Master | 4 | 1104 | -1.946 | -1.890 | +0.056 |
| Kaladen Shash | 4 | 611 | -8.433 | -8.484 | -0.051 |
| Leopold Magnus | 4 | 501 | -8.044 | -8.047 | -0.003 |

By target-share bucket:

| bucket | rows | target words | mixture SI-SDR | estimate SI-SDR | SI-SDRi |
| --- | ---: | ---: | ---: | ---: | ---: |
| lt_010 | 5 | 264 | -11.948 | -12.333 | -0.385 |
| 010_020 | 8 | 1032 | -9.783 | -8.723 | +1.060 |
| 020_035 | 8 | 1984 | -4.141 | -3.482 | +0.659 |
| 035_050 | 1 | 344 | -0.782 | -0.612 | +0.170 |

Checkpoint smoke comparison on the first two Session 64 rows:

| checkpoint | mean SI-SDRi |
| --- | ---: |
| TFGridNet wsj0-2mix | +1.565 |
| TFGridNet wham! | -2.142 |
| TFGridNet whamr! | -9.167 |

## Domain Adapter Pilot

Added experimental fine-tuning script:

`scripts/train_usef_tse_domain_adapter.py`

The script keeps USEF external, caches session stems as 16 kHz mono WAV, samples short chunks from
manifest rows, and trains USEF-TFGridNet with:

- target-source reconstruction,
- direct positive enrollment conditioning,
- transcript-derived target activity weighting,
- silence penalty outside target-owned spans.

Pilot command:

```bash
uv run python scripts/train_usef_tse_domain_adapter.py \
  --manifest .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v4_large_inventory/speaker_tse_manifest.jsonl \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_domain_adapter_v2_s51_s58 \
  --stems-cache-root .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_domain_adapter_smoke/_stems \
  --stems-wav-root .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_domain_adapter_smoke/_stems16 \
  --usef-repo /tmp/USEF-TSE \
  --train-sessions "Session 51,Session 58" \
  --dev-sessions "Session 65" \
  --train-steps 80 \
  --dev-batches 4 \
  --batch-size 1 \
  --chunk-seconds 4 \
  --enrollment-seconds 4 \
  --activity-probability 1.0 \
  --learning-rate 5e-6
```

Best checkpoint:

`<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_domain_adapter_v2_s51_s58/best_domain_adapter.pt`

Held-out Session 64/67 SI-SDR comparison:

| model | estimate SI-SDR | SI-SDRi vs mixture |
| --- | ---: | ---: |
| raw mixture | -7.814 | 0.000 |
| pretrained USEF-TFGridNet wsj0-2mix | -7.269 | +0.545 |
| domain adapter v1 interrupted pilot | -6.461 | +1.354 |
| domain adapter v2 80-step pilot | -6.139 | +1.676 |

V2 by speaker:

| speaker | SI-SDRi |
| --- | ---: |
| Cletus Cobbington | +0.806 |
| Cyrus Schwert | +2.675 |
| David Tanglethorn | +2.833 |
| Dungeon Master | +1.381 |
| Kaladen Shash | +0.906 |
| Leopold Magnus | +1.020 |

This is the first domain-trained checkpoint that beats the pretrained USEF baseline on every held-out
speaker bucket.

## Energy Word-Ownership Probe

Added probe:

`scripts/score_tse_word_ownership.py`

This scores each labeled transcript span by assigning it to the extracted speaker track with the
highest energy over that span. It is a rough proxy because the manifest currently has line/span
timing rather than true word-level forced alignment.

Held-out Session 64/67 word-weighted scored accuracy:

| signal tracks | scored accuracy |
| --- | ---: |
| clean source tracks | 70.17% |
| pretrained USEF-TFGridNet wsj0-2mix | 44.67% |
| domain adapter v2 | 47.38% |

Interpretation: the adapter improves target extraction, but simple energy ownership is not enough.
The clean-source ceiling of only 70% under this probe also shows that the current transcript span
timing is too coarse for this to be the final word-speaker metric.

## Speaker-Attributed ASR Probe

Added probe:

`scripts/score_tse_speaker_attributed_asr.py`

This transcribes each speaker/source track independently with faster-whisper, then scores the
resulting word spans against manifest speaker labels. It is closer to the proposed
speaker-attributed ASR path than energy ownership, but it still uses approximate reference word
timing by uniformly splitting transcript spans into words.

One held-out Session 64 window was tested with faster-whisper `small` on CPU/int8 after the WSL CUDA
cuDNN loader failed. The reference contained 934 approximate words.

| signal tracks | predicted words | matched words | correct words | total accuracy | matched accuracy | precision proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean source tracks | 877 | 670 | 427 | 45.72% | 63.73% | 48.69% |
| pretrained USEF-TFGridNet wsj0-2mix | 3107 | 839 | 171 | 18.31% | 20.38% | 5.50% |
| domain adapter v2 | 2415 | 823 | 199 | 21.31% | 24.18% | 8.24% |

Interpretation: domain adaptation again improves over pretrained USEF, but extracted-track ASR
hallucinates or duplicates too much speech to work as a post-extraction decoder. More importantly,
the clean-source track only reaches 45.72% total accuracy under this approximate timing setup, so
this probe is a debugging tool rather than a trustworthy upper-bound metric. The next meaningful
ASR branch needs forced alignment or a true joint diarization-ASR objective, not another
extract-then-transcribe router.

## Forced-Alignment Reference Update

Added forced-reference builder:

`scripts/build_tse_forced_word_reference.py`

The builder uses torchaudio `MMS_FA` on the clean materialized speaker stems to replace uniform
transcript-span word splitting with CTC forced word timings. On the four materialized held-out
Session 64/67 windows it aligned 3,586 of 3,630 transcript tokens:

| groups | transcript tokens | aligned words | coverage | errors |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 3,630 | 3,586 | 98.79% | 1 |

The ASR scorer now accepts `--reference-jsonl` and reports two different surfaces:

- `timed_speaker_hit_rate`: legacy timing-only speaker hit rate. The older `accuracy` field is kept
  as an explicit legacy alias for this metric and should not be treated as lexical correctness.
- `speaker_attributed_lexical_accuracy`: primary ASR attribution metric for this probe. A word only
  counts if the normalized token text matches and the speaker matches.
- `same_speaker_candidate_coverage`: whether the correct speaker track has any ASR word near the
  forced-aligned reference word. This is not deployable accuracy, but it is a useful recoverability
  diagnostic for a future word-owner/suppression objective.

Held-out Session 64/67, faster-whisper `small` CPU/int8, tolerance 0.75 seconds:

| signal tracks | greedy accuracy | same-speaker coverage | nearest-speaker accuracy | prediction density |
| --- | ---: | ---: | ---: | ---: |
| clean source tracks | 56.00% | 93.67% | 77.11% | 0.96x |
| pretrained USEF-TFGridNet wsj0-2mix | 20.86% | 94.92% | 22.81% | 3.74x |
| domain adapter v2 | 22.00% | 89.79% | 25.15% | 3.05x |

Interpretation: the old clean-source "ceiling" was too pessimistic, but the merged-word score is
still the wrong way to evaluate multi-track ASR candidates. Clean source has near-94% same-speaker
coverage with normal prediction density. Both USEF variants also contain target-speaker evidence
near roughly 90-95% of reference words, but they emit about 3-4x too many decoded words. The domain
adapter reduces overgeneration versus pretrained USEF and improves greedy merged accuracy, but it
also loses some target coverage.

This is the strongest evidence so far that the next model objective should be a word-owner or
suppression objective: keep the target coverage while forcing non-owner/interferer words silent.
More SI-SDR-only adaptation is not enough.

## Forced-Mask Training Smoke

Extended the forced-reference builder and trainer:

- `scripts/build_tse_forced_word_reference.py` now supports `--include-nonmaterialized` and
  `--sessions`, so train/dev manifest windows can be aligned directly from the original session
  zips without pre-materializing every row.
- `scripts/train_usef_tse_domain_adapter.py` now accepts `--forced-reference-jsonl` and
  `--non-owner-weight`. When forced references are available, target masks come from target-owned
  aligned words and suppression masks come from non-owner aligned words.
- `scripts/score_tse_forced_mask_energy.py` scores waveform leakage directly as target-word energy
  versus non-owner-word energy.

Forced references for the prior domain-adapter pilot sessions:

```bash
uv run python scripts/build_tse_forced_word_reference.py \
  --manifest .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v4_large_inventory/speaker_tse_manifest.jsonl \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v4_large_inventory/forced_word_reference_s51_s58_s65_mms \
  --include-nonmaterialized \
  --sessions "Session 51,Session 58,Session 65" \
  --stems-cache-root .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_domain_adapter_smoke/_stems \
  --stems-wav-root .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_domain_adapter_smoke/_stems16
```

Result: 12 groups, 10,153 transcript tokens, 10,061 aligned words, 99.09% coverage.

Short forced-mask smoke:

- Initialization: domain adapter v2 best checkpoint.
- Train sessions: 51/58.
- Dev session: 65.
- Steps: 20, best checkpoint at step 10 by dev loss.
- Objective additions: forced target-word mask, forced non-owner suppression mask, `activity_weight`
  0.35, `non_owner_weight` 4.0.

Held-out Session 64/67 results:

| model | SI-SDRi | greedy ASR accuracy | same-speaker coverage | prediction density | target/non-owner energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| pretrained USEF-TFGridNet wsj0-2mix | +0.545 dB | 20.86% | 94.92% | 3.74x | 1.29 dB |
| domain adapter v2 | +1.676 dB | 22.00% | 89.79% | 3.05x | 2.31 dB |
| forced-mask smoke | +1.897 dB | 21.53% | 89.21% | 2.96x | 2.55 dB |
| clean source tracks | n/a | 56.00% | 93.67% | 0.96x | 9.08 dB |

Interpretation: forced masks improve held-out SI-SDRi and directly reduce non-owner waveform energy,
but the effect is small and does not yet improve ASR word ownership. The ASR density moved in the
right direction from 3.05x to 2.96x, but the model is still far from clean-source suppression.
This supports the forced word-owner/suppression objective as the right training direction, while
rejecting this short smoke as a breakthrough.

Next gate: train with forced references as a first-class objective, monitor target/non-owner energy
on dev, and select checkpoints by a suppression-aware metric instead of SI-SDR/dev loss alone.

## Suppression-Aware Selection Update

Updated `scripts/train_usef_tse_domain_adapter.py` again:

- non-owner suppression now excludes regions where target-owned words overlap non-owner words, so
  overlap does not punish the extractor for keeping the target speaker;
- dev evaluation now uses the same deterministic sampled chunks at every checkpoint in a run, so
  checkpoint comparisons are not driven by changing random dev samples;
- `--selection-metric` can select `best_domain_adapter.pt` by `dev_loss`, `dev_si_snr`,
  `dev_target_to_non_owner_db`, or `dev_suppression_score`.

Two selection-aware smokes were run from the domain adapter v2 checkpoint:

| run | dev selection behavior | held-out SI-SDRi | held-out target/non-owner energy | decision |
| --- | --- | ---: | ---: | --- |
| sampled-dev suppression selection | changing dev chunks picked step 25 with 7.00 dB sampled score | +1.776 dB | 2.38 dB | invalid/noisy selector |
| fixed-dev suppression selection | deterministic dev chunks improved 2.87 -> 2.93 dB by step 20 | +1.843 dB | 2.46 dB | valid but below forced-mask smoke |

The earlier forced-mask smoke remains the best forced-mask result so far: +1.897 dB SI-SDRi and
2.55 dB target/non-owner energy. Selection-aware training made the measurement cleaner but did not
produce a better held-out checkpoint.

Interpretation: the current USEF adapter can be nudged toward suppression, but this objective/model
shape is moving slowly and appears sensitive to sampled chunk selection. A scaled version should not
be promoted without a larger deterministic dev set and a stronger word-owner/suppression head.

## Interpretation

USEF-TFGridNet proves that direct-enrollment target extraction can pick up real target-speaker signal
on this corpus: the clean WSJ checkpoint beats the raw mixture across 22 held-out rows. But the gain
is small and uneven. The model helps Cyrus and David, barely moves Dungeon Master, and regresses
low-share Cletus/Kaladen/Leopold cases.

This supports the big-bet direction while rejecting an off-the-shelf adoption. A pretrained TSE model
is useful as an initialization or adapter reference, but the repo still needs domain-scale training on
Discord/call mixtures with direct word-ownership supervision, forced word timing, or a real
speaker-attributed ASR objective.

## Next Gate

The next useful experiment is not another checkpoint swap. Train or adapt a target-speaker extractor
on the manifest rows and add an explicit word-owner loss, or build a speaker-attributed ASR model
with stable forced-aligned word targets. Promotion should require a large held-out word-speaker gain,
not just a small SI-SDR gain.

The larger available local inventory is:

`<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v4_large_inventory/speaker_tse_manifest.jsonl`

| split | rows | target words |
| --- | ---: | ---: |
| train | 215 | 32234 |
| dev | 67 | 9772 |
| test | 45 | 7221 |

This is the better starting point for domain adaptation. It should be materialized selectively or
fully depending on the training budget, with Sessions 64/67 kept as held-out test.

## Auditor-Driven Drive v5 Update

Incorporated the external audit recommendation to treat extraction as a diagnostic/auxiliary branch
rather than the final owner-deciding mechanism. The next primary architecture should be
multi-label target-speaker activity plus word ownership, with mixture-level ASR assigned by
speaker activity posteriors.

WSL Google Drive mount used for the larger eval:

`/mnt/g/My Drive/DND/Audio`

New Drive-backed manifest:

`outputs/speaker_tse_drive_v5_49_67_devtest_materialized/speaker_tse_manifest.jsonl`

Split summary:

| split | sessions | rows | target words |
| --- | --- | ---: | ---: |
| train | 49-62 | 221 | 33981 |
| dev | 63, 66 | 33 | 5042 |
| test | 64, 67 | 34 | 5397 |

Forced references:

`outputs/speaker_tse_drive_v5_49_67_devtest_materialized/forced_word_reference_mms/forced_word_reference_groups.jsonl`

Result: 12 groups, 10,470 transcript tokens, 10,401 aligned words, 99.34% coverage.

P0/P1 code repairs made before interpreting the expanded run:

- lexical-aware word matching added beside the old timed-speaker hit metric;
- ASR summaries now name `speaker_attributed_lexical_accuracy` as the primary metric and keep
  timing-only `accuracy` as a legacy alias;
- ASR cache keys now include ASR model/config/version identity;
- SI-SDR scoring now pads short estimates instead of truncating references;
- hard-coded `source_oracle_si_sdr` was replaced by a measured self-reference field;
- inference runner now records skipped-output metadata, resamples enrollment with its own sample
  rate, and supports overlap-add chunk stitching;
- forced non-owner suppression is guarded by explicit `interferer_only_mask` and `overlap_mask`
  helpers;
- training masks now use nearest-neighbor resampling instead of audio resampling;
- forced-word ownership scoring now uses MMS word intervals instead of coarse transcript spans.

Forced-label meeting-model update:

- Train Sessions 49-62 now have forced/fallback word references for 33,962/34,275 tokens
  (99.09% total coverage, 97.24% forced-only coverage, 1.85% source-span fallback).
- Exporting those references to NeMo RTTMs produced 420 thirty-second Sortformer train chunks.
- Ultra-Sortformer 8spk trained on those chunks reaches 72.76% all-word many-to-one owner
  accuracy, 69.12% one-to-one, 80.15% non-overlap, and 86.88% high-confidence non-overlap on
  held-out Sessions 64/67.
- The improvement is mostly non-overlap. Estimated overlap-word many-to-one accuracy is still only
  about 46.19%, so cleaner diarization labels do not by themselves create the 90%+ path.
- A bounded sequence TS-VAD/word-owner probe using the same forced labels reaches only 34.30%
  held-out accuracy and 27.68% overlap accuracy. The TS-VAD/word-owner architecture remains the
  right direction, but the local from-scratch log-spectrum TCN is too weak; the next version needs
  a stronger pretrained/adapted meeting backbone or joint speaker-attributed ASR.

Drive v6 larger-train Sortformer update:

- Expanded the forced-label training manifest to Sessions 43-62 plus 65 while keeping dev
  Sessions 63/66 and held-out test Sessions 64/67 fixed:
  `outputs/speaker_tse_drive_v6_43_67_devtest_materialized/speaker_tse_manifest.jsonl`.
- The expanded train split has 324 manifest rows and 51,411 target words. Its NeMo handoff has
  63 train windows, 18,900 seconds, and 630 thirty-second chunks:
  `outputs/sortformer_nemo_export_train_forced_v6_43_62_65_v1/train_chunks_30s_manifest.jsonl`.
- The new Sessions 43-48 plus 65 forced-reference build aligned 17,427/17,663 transcript tokens
  but required 1,351 fallback words, so the added material is less clean than the train49-62 build.
- Ultra-Sortformer 8spk trained for 630 steps on the expanded forced-label chunks reaches 71.77%
  all-word many-to-one owner accuracy, 69.73% one-to-one, 78.39% non-overlap, and 84.53%
  high-confidence non-overlap on the same held-out Sessions 64/67 reference.
- Estimated overlap many-to-one accuracy improves from 46.19% to 47.99%, but this does not offset
  the non-overlap and high-confidence regressions. This rejects "just add more forced-label
  sessions" as a standalone Sortformer breakthrough; data cleanliness and known-speaker owner
  conditioning matter more than raw hours in this branch.

Expanded 67-row results:

| system | SI-SDRi | target/non-owner energy | forced-word owner accuracy |
| --- | ---: | ---: | ---: |
| clean source tracks | n/a | 9.42 dB | 77.23% |
| forced-mask smoke extractor | +2.30 dB | 2.60 dB | 50.18% |

The extractor improves SI-SDR substantially on the broader Drive dev/test set, especially dev
(+2.81 dB) versus held-out test (+1.80 dB), but it collapses word-owner separability relative to
clean speaker stems. This supports the auditor's conclusion: do not keep optimizing
extract-then-independent-ASR as the final diarization shape. Use extraction as an auxiliary feature
for a TS-VAD/Sortformer-style known-speaker activity and word-owner model.

## Known-Speaker Activity And Diarizer Ceiling Update

Three owner-deciding diagnostics were added after the external audit:

- `scripts/train_tsvad_word_owner_baseline.py`: speaker-conditioned frame activity with enrollment
  profiles, preserving overlap as independent binary labels.
- `scripts/train_direct_word_owner_baseline.py`: direct enrollment-conditioned word-owner scoring,
  trained with the loss aligned to "which enrolled speaker owns this word/span?"
- `scripts/score_pyannote_oracle_diarization.py`: pyannote diarization scored against forced words
  after oracle cluster-to-known-speaker mapping.
- `scripts/train_wavlm_word_owner_baseline.py`: frozen WavLM mixture/enrollment features with a
  direct word-owner head, used to test whether the weak local spectral representation was the main
  bottleneck.
- `scripts/train_sequence_tsvad_word_owner_baseline.py`: enrollment-conditioned temporal TCN over
  frame sequences, used to test whether adding short-range sequence context to TS-VAD frame
  activity is enough.
- `scripts/train_listwise_word_owner_baseline.py`: masked softmax word-owner objective over all
  speaker candidates for each word, used to test whether independent BCE candidate scoring was the
  main transfer bottleneck.
- `scripts/export_sortformer_training_data.py`: Drive window to NeMo manifest/RTTM export for
  Sortformer adaptation.
- `scripts/chunk_sortformer_manifest.py`: 300-second Sortformer manifests split into tractable
  30-second training windows while preserving RTTM offsets and active-speaker counts.
- `scripts/train_sortformer_domain_adapter.py`: NeMo Sortformer domain-adaptation runner used for
  the 8-speaker full-domain and 5-speaker filtered experiments.
- `scripts/widen_sortformer_checkpoint.py`: 5-speaker-to-8-speaker Sortformer checkpoint transplant,
  used to test whether the stronger 5-speaker backbone can cover full 6/7-speaker windows.
- `scripts/score_speaker_tagged_asr.py`: speaker-attributed ASR scorer for anonymous `[S0]`-style
  outputs, using lexical sequence alignment and per-clip oracle tag-to-known-speaker mapping.
- `scripts/build_speaker_tagged_asr_dataset.py`: forced-word chunk exporter for Qwen-style
  first-appearance `[S0]` speaker-tagged ASR training targets.
- `scripts/train_qwen_speaker_tagged_lora.py`: LoRA adaptation runner for the runnable Qwen3-ASR
  speaker-tagged checkpoint, including immediate held-out prediction export.

Held-out Sessions 64/67 forced-word results:

| system | train/eval setup | word-owner accuracy |
| --- | --- | ---: |
| clean source tracks | forced-word energy oracle | 75.28% |
| Ultra-Sortformer 5spk + 30s max5 domain adapter | many-to-one oracle cluster map | 72.36% |
| Ultra-Sortformer 5spk-to-8spk widened + 50-step adapter | many-to-one oracle cluster map | 71.73% |
| Ultra-Sortformer 8spk + 30s domain adapter | many-to-one oracle cluster map | 71.71% |
| Ultra-Sortformer streaming 5spk v1 | many-to-one oracle cluster map | 69.88% |
| Ultra-Sortformer streaming 8spk v1 | many-to-one oracle cluster map | 68.33% |
| pyannote Community-1, free speaker count | many-to-one oracle cluster map | 66.54% |
| pyannote Community-1, forced candidate count | many-to-one oracle cluster map | 63.87% |
| Sortformer streaming 4spk v2.1 | many-to-one oracle cluster map | 64.92% |
| Sortformer offline 4spk v1 | many-to-one oracle cluster map | 64.73% |
| Sortformer streaming 4spk v2 | many-to-one oracle cluster map | 60.68% |
| Qwen3-ASR 1.7B speaker-tagged AMI finetune | four 30s clips, lexical sequence alignment | 52.23% |
| Qwen3-ASR speaker-tagged + gentle domain LoRA | four 30s clips, lexical sequence alignment | 51.97% |
| Qwen3-ASR speaker-tagged + 40-step domain LoRA | four 30s clips, lexical sequence alignment | 30.45% |
| forced-mask smoke extractor | energy owner from extracted tracks | 50.18% |
| TS-VAD-style frame activity MLP | train 49-62 manifest spans + forced dev | 41.82% |
| direct word-owner MLP | train 49-62 manifest spans + forced dev | 40.92% |
| temporal sequence TS-VAD TCN | train 49-62 manifest spans + forced dev | 39.00% |
| listwise word-owner MLP | train 49-62 manifest spans + forced dev | 36.98% |
| WavLM frozen-feature word-owner MLP | train 49-62 manifest spans + forced dev | 21.89% |

The TS-VAD-style harness did learn the dev windows: train/dev self-eval reached 66.40% overall and
72.62% non-overlap accuracy. It did not generalize from six forced dev windows to held-out sessions.
Adding the wider Drive train sessions improved held-out accuracy to 41.82%, but the hand-built
spectral enrollment representation remained weaker than pyannote's anonymous diarization.

The direct word-owner head did not break through either. This is an important negative result:
changing the loss to word ownership is not enough if the representation is still a small local
spectral profile model. The next real attempt needs a pretrained meeting/TS-VAD backbone or
speaker-attributed ASR architecture with direct enrollment conditioning, not another local MLP over
log spectra.

The frozen WavLM word-owner run tested whether a stronger generic speech representation fixes that
problem. It did not. Dev self-eval reached 61.52%, but dev-to-test collapsed to 13.77%, and a
train49-62 plus forced-dev run reached only 21.89% held-out accuracy. This is worse than the
hand-built spectral TS-VAD/direct baselines, so the missing piece is not simply swapping in a
pretrained frame representation. A viable WavLM-style path would need end-to-end speaker-conditioned
activity/owner training, not frozen pooled features plus a shallow MLP.

The sequence TS-VAD TCN tested the other obvious missing ingredient in the local baseline: temporal
context. It underfit forced dev at 43.73% and reached 39.00% held-out accuracy, below the simpler
independent-frame TS-VAD MLP. This rejects local frame-BCE sequence capacity as the next step. The
remaining plausible directions are direct word-owner/listwise training over speaker candidates,
speaker-attributed ASR, or adapting a serious pretrained meeting model where diarization structure
is already part of the backbone.

The listwise word-owner run tested that direct-owner hypothesis with a better loss: one masked
softmax over speaker candidates per word instead of independent BCE scores. It fit forced dev best
among the local owner models at 69.67%, but transfer remained poor: 28.55% from dev-only to heldout
and 36.98% from train49-62 plus forced dev to heldout. That means the local spectral/enrollment
feature family is the bottleneck, not just the candidate loss. More local MLP objective variants are
unlikely to reach the clean-source/pyannote gap.

The first runnable speaker-attributed ASR smoke used
`mrfakename/qwen3-asr-1.7b-ami-diarization-fft-r6-20260422`, an Apache-2.0 Qwen3-ASR AMI finetune
that emits inline speaker tags. Four 30-second held-out Session 64/67 clips were scored against MMS
forced words after normalized lexical sequence alignment and per-clip oracle tag mapping. It reached
52.23% many-to-one accuracy over 381 forced words, 37.27% one-to-one, 53.28% lexical coverage,
43.02% overlap accuracy, and 36.65% prediction precision proxy. The matched anonymous tags were
often coherent, but lexical coverage and repetition were not acceptable. TagSpeech-AMI is a stronger
conceptual model, but the current HF snapshot lacks the quick-start code files referenced by its
card; Rev Reverb diarization V2 is gated with the current credentials. This keeps SA-ASR alive only
as a domain-fine-tuned or stronger-model path.

Domain LoRA adaptation is technically feasible. The generated train set
`outputs/speaker_tagged_asr_train_forced_v1_max4_120` contains 120 train49-62 clips and 9,399 forced
words, capped at four local speakers per clip. A one-sample supervised-loss probe succeeded, and the
LoRA runner trains 1.20M parameters with roughly 5-7 GB peak GPU memory. The first actual smokes
were negative: 40 steps at 1e-4 destabilized generation and dropped to 30.45% many-to-one with only
8.14% overlap accuracy, while a gentle 5-step 1e-5 adapter stayed near zero-shot at 51.97% and did
not fix repetition. This rejects naïve tiny LoRA as the missing piece, but leaves a more careful
domain SA-ASR recipe or stronger runnable base model as an open big-bet path.

The pyannote oracle result is the cleanest diagnostic for the existing modular pipeline. With
perfect cluster-to-identity mapping, the repo's current anonymous diarization family is still only
about 66.5% on the forced-word heldout. Therefore a better speaker bank or embedding mapper alone
cannot plausibly reach 90% on this eval; the segmentation/overlap/word-owner layer itself must
change.

Added `scripts/score_sortformer_oracle_diarization.py` to test modern NeMo Sortformer diarization
as a stronger pretrained meeting-style baseline. The official runnable checkpoints are four-speaker
models, while the Drive held-out windows often contain five or six known speakers. That speaker-cap
mismatch shows up directly: the best official Sortformer run,
`nvidia/diar_streaming_sortformer_4spk-v2.1`, reaches 64.92% many-to-one oracle accuracy overall
and 76.22% on high-confidence non-overlap words. A community widened Ultra-Sortformer 5-speaker
checkpoint improves this to 69.88% overall and 81.17% high-confidence non-overlap, while the
8-speaker checkpoint reaches 68.33% and 78.57%. Sortformer is therefore relevant as a backbone or
fine-tuning recipe, but even the widened off-the-shelf checkpoints are not the breakthrough.

Domain adaptation is the first positive result from the bigger-bet path. A 300-second 8-speaker
fine-tune smoke ran out of memory in Sortformer's ATS/PIL permutation loss, so the NeMo manifests
were chunked into 30-second windows. The 8-speaker adapter on all chunks reaches 71.71% overall and
82.87% high-confidence non-overlap after 420 steps. Filtering to <=5 active speakers and adapting
the stronger 5-speaker checkpoint reaches 72.36% overall, 71.90% one-to-one, 84.46%
high-confidence non-overlap many-to-one, and 85.25% one-to-one. This beats all off-the-shelf
meeting baselines tested so far, but still leaves a large gap to clean-source top-two and does not
solve known-speaker identity without oracle cluster mapping.

The clean-source oracle result also needs respect. Actual per-speaker tracks score only 75.28% on
held-out forced words despite high target/non-owner energy margins. Before treating 90% as an
engineering gate, audit the forced word references, source-member mappings, and Discord track bleed
on the clean-source misses. Otherwise model experiments may be punished for reference noise.

## Clean-Source Oracle Audit

Added `scripts/audit_clean_source_word_oracle.py` to separate three things that were previously
blurred together:

- single-winner energy ownership: "which clean source is loudest for this word?"
- reference-source recoverability: "is the true speaker active in the clean source?"
- reference quality bands: overlap, forced-align confidence, and very short word duration.

Held-out Sessions 64/67:

| slice | word share | clean winner | clean top-two | pyannote oracle many-to-one |
| --- | ---: | ---: | ---: | ---: |
| all forced words | 100.00% | 75.28% | 91.06% | 66.54% |
| non-overlap | 78.22% | 82.28% | 93.68% | 73.59% |
| forced score >= 0.05 | 63.67% | 85.84% | 97.92% | 75.32% |
| non-overlap, forced score >= 0.05 | 51.63% | 91.76% | 99.42% | 81.68% |
| non-overlap, forced score >= 0.05, duration >= 80 ms | 41.52% | 92.27% | 99.46% | 81.62% |

Dev+test clean-source audit:

| slice | word share | clean winner | clean top-two |
| --- | ---: | ---: | ---: |
| all forced words | 100.00% | 77.23% | 92.68% |
| non-overlap | 78.26% | 84.79% | 95.01% |
| forced score >= 0.05 | 65.46% | 86.72% | 98.21% |
| non-overlap, forced score >= 0.05 | 52.96% | 92.86% | 99.51% |
| non-overlap, forced score >= 0.05, duration >= 80 ms | 42.54% | 93.13% | 99.50% |

The multi-label activity audit is more encouraging than the single-winner energy number. On
held-out Sessions 64/67, oracle group/speaker clean-source activity thresholds recall the reference
speaker for 91.14% of forced words. Across dev+test, reference-active recall is 91.96%, pair ROC AUC
is 0.926, and non-overlap active-pair recall is 93.16%.

Interpretation: the 75% clean-source "word-owner" number was not a hard source ceiling. It was a
harsh single-winner metric that punishes overlap and low-confidence forced words. The signal needed
for a TS-VAD/speaker-attributed ASR objective is closer to the low-90s on clean tracks and near
perfect on high-confidence non-overlap top-two. However, pyannote oracle mapping remains only
81.68% on the same high-confidence non-overlap slice, so the current anonymous diarization backbone
still leaves a large architecture gap.

## Sortformer Oracle Diarization Check

Added scorer:

`scripts/score_sortformer_oracle_diarization.py`

This feeds each materialized flat mixture to NeMo Sortformer, parses the predicted diarization
segments, and scores forced-reference words after oracle mapping anonymous model speakers to known
speakers. It uses the same held-out Sessions 64/67 reference as the pyannote oracle check.

| model | overall many-to-one | overall one-to-one | high-confidence non-overlap many-to-one |
| --- | ---: | ---: | ---: |
| Ultra-Sortformer 8spk v1 + forced-label 420-step 30s adapter | 72.76% | 69.12% | 86.88% |
| Ultra-Sortformer 5spk v1 + max5 30s domain adapter | 72.36% | 71.90% | 84.46% |
| Ultra-Sortformer 8spk v1 + forced-label 50-step 30s adapter | 72.07% | 69.15% | 84.24% |
| Ultra-Sortformer 8spk v1 + expanded forced-label 630-step 30s adapter | 71.77% | 69.73% | 84.53% |
| Ultra-Sortformer 8spk v1 + 30s domain adapter | 71.71% | 68.17% | 82.87% |
| Ultra-Sortformer 5spk-to-8spk widened + 50-step 30s adapter | 71.73% | 70.20% | 83.09% |
| Ultra-Sortformer 5spk-to-8spk widened + 420-step 30s adapter | 69.30% | 68.11% | 81.71% |
| Ultra-Sortformer 5spk-to-8spk widened raw | 69.10% | 65.96% | 80.05% |
| Ultra-Sortformer streaming 5spk v1 | 69.88% | 69.86% | 81.17% |
| Ultra-Sortformer streaming 8spk v1 | 68.33% | 62.83% | 78.57% |
| pyannote Community-1, free count | 66.54% | 63.41% | 81.68% |
| Sortformer streaming 4spk v2.1 | 64.92% | 64.92% | 76.22% |
| Sortformer offline 4spk v1 | 64.73% | 64.47% | 72.79% |
| Sortformer streaming 4spk v2 | 60.68% | 60.29% | 71.09% |
| clean source single-winner oracle | 75.28% | n/a | 91.76% |
| clean source top-two oracle | 91.06% | n/a | 99.42% |

Interpretation: a modern EEND-style diarizer is worth keeping in the candidate set, and relaxing the
four-speaker cap is a real improvement. Forced-word train labels are also genuinely useful: the
8-speaker train49-62 forced-label run remains the strongest aggregate/high-confidence anonymous
diarizer tested so far. Expanding the forced-label train set to Sessions 43-62 plus 65 did not
improve the result, which suggests added label noise/session mismatch can erase the benefit of
scale. The Sortformer branch still does not close the clean-source gap, still needs oracle cluster
mapping, and loses on overlap-word ownership, so the next meeting-model attempt should add
known-speaker enrollment context or direct word ownership rather than rely on anonymous cluster
assignment alone.

## Sortformer NeMo Export

Added exporter:

`scripts/export_sortformer_training_data.py`

This groups the speaker-TSE manifest by session window, writes one mixed audio file and one RTTM per
window, and emits NeMo-compatible diarization manifests with `audio_filepath`, `duration`,
`num_speakers`, and `rttm_filepath`. It uses forced-word references when available and falls back to
manifest transcript spans for train.

WSL exports:

| export | groups | duration | label source | speaker-count distribution |
| --- | ---: | ---: | --- | --- |
| `outputs/sortformer_nemo_export_train_v1/train_manifest.jsonl` | 42 | 12,600 s | manifest spans | 5spk: 6, 6spk: 27, 7spk: 9 |
| `outputs/sortformer_nemo_export_devtest_v1/dev_manifest.jsonl` | 6 | 1,800 s | forced words | max 6spk |
| `outputs/sortformer_nemo_export_devtest_v1/test_manifest.jsonl` | 6 | 1,800 s | forced words | max 6spk |

All exported manifest audio and RTTM paths validate as present. Because most train windows contain
more than five speakers, full-domain adaptation uses the 8-speaker Sortformer head. The 5-speaker
checkpoint remains stronger per comparable speaker-count slice, but it needs filtering or
model-head surgery before it can cover the full corpus.

Chunked NeMo handoffs:

| manifest | rows | speaker-count distribution |
| --- | ---: | --- |
| `outputs/sortformer_nemo_export_train_v1/train_chunks_30s_manifest.jsonl` | 420 | 1spk: 3, 2spk: 12, 3spk: 79, 4spk: 105, 5spk: 151, 6spk: 63, 7spk: 7 |
| `outputs/sortformer_nemo_export_train_v1/train_chunks_30s_max5_manifest.jsonl` | 350 | filtered to <=5 active speakers |
| `outputs/sortformer_nemo_export_devtest_v1/dev_chunks_30s_manifest.jsonl` | 60 | 2spk: 8, 3spk: 13, 4spk: 26, 5spk: 10, 6spk: 3 |
| `outputs/sortformer_nemo_export_devtest_v1/dev_chunks_30s_max5_manifest.jsonl` | 57 | filtered to <=5 active speakers |

The first domain adapters were trained from those chunked manifests. The 8-speaker run uses
`mago-ai/ultra_diar_streaming_sortformer_8spk_v1` and writes
`outputs/sortformer_8spk_domain_adapter_30s_420step_v1/sortformer_domain_adapter_final.nemo`. The
5-speaker filtered run uses `devsy0117/ultra_diar_streaming_sortformer_5spk_v1` and writes
`outputs/sortformer_5spk_domain_adapter_30s_350step_v1/sortformer_domain_adapter_final.nemo`. The
best held-out oracle result is currently the 5-speaker filtered adapter at 72.36% overall and
84.46% high-confidence non-overlap.

The 5-speaker-to-8-speaker widening probe writes
`outputs/sortformer_5spk_to_8spk_widened_v1/sortformer_widened_final.nemo`. A one-step 30-second
training smoke succeeds, proving the widened checkpoint is shape-compatible with NeMo's 8-speaker
ATS/PIL loss. Direct scoring reaches 69.10% overall / 80.05% high-confidence non-overlap. A 50-step
full-domain adaptation reaches 71.73% overall / 83.09% high-confidence non-overlap, with much better
one-to-one mapping than the vanilla 8-speaker 50-step adapter. A 420-step full pass regresses to
69.30% overall / 81.71% high-confidence non-overlap. This rejects "just train the widened checkpoint
longer on coarse span RTTMs" as the next move and points back to forced-word train labels or
known-speaker owner supervision.
