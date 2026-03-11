# 1. Build Plan

1. Keep word-level evaluation in `src/transcriber/multitrack_eval.py` as the acceptance harness.
2. Calibrate the existing closed-set speaker bank before changing diarization models.
3. Score mixed-audio diarization segments individually, then aggregate by diar label.
4. Support whitening and AS-Norm in `src/transcriber/speaker_bank.py` and thread those options through `src/transcriber/cli.py`.

## 2. Data and Interface Mapping

| field | meaning | units | source_equation_or_rule | citation_ids |
| --- | --- | --- | --- | --- |
| `segment.embedding` | diarization segment embedding | unitless | EQ-001 input | CIT-001 |
| `speaker_bank.threshold` | acceptance floor | unitless | thresholded score after EQ-001 / EQ-002 | CIT-002 |
| `speaker_bank.match_per_segment` | enable segment-first matching | boolean | segment evidence rule | CIT-001 |
| `speaker_bank.scoring.whiten` | apply whitening before scoring | boolean | EQ-003 | CIT-006 |
| `speaker_bank.scoring.as_norm.enabled` | use adaptive symmetric normalization | boolean | EQ-002 | CIT-002 |
| `speaker_bank.scoring.as_norm.cohort_size` | top-K cohort size | count | EQ-002 | CIT-002 |

## 3. Algorithm Procedure (Coding Form)

1. Validate the speaker bank exists and has at least two speakers before enabling AS-Norm.
2. Extract diarization-aligned segment embeddings per raw diar label.
3. For each segment embedding:
   - Normalize the embedding.
   - Optionally whiten it using the bank covariance transform.
   - Score it against each speaker cluster/prototype with cosine similarity.
   - Optionally convert the raw score with AS-Norm using the bank's non-target cohort embeddings.
4. Accept segment-level matches only when `score >= threshold` and `top1 - top2 >= margin`.
5. Aggregate candidates for each diar label using mean or vote over all segment candidates.
6. Apply segment-level matches first; only fall back to label-level matches when segment evidence is missing.
7. Propagate the chosen speaker label down to word-level timestamps so the evaluator can score words directly.

## 4. Numerical Stability and Fallbacks

| scenario | risk | fallback/disable rule | citation_ids |
| --- | --- | --- | --- |
| Whitening eigendecomposition fails | invalid transform | fall back to raw normalized embeddings | CIT-006 |
| Cohort variance is near zero | AS-Norm explosion | clamp `sigma` to `1e-6` minimum | CIT-002 |
| Cohort smaller than requested top-K | poor normalization | shrink `k` to available cohort size; disable if empty | CIT-002 |
| Too few segments for a diar label | unstable aggregation | require `min_segments_per_label >= 2` | CIT-001 |

## 5. Acceptance Gates and Kill Criteria

| gate_id | metric | threshold | comparison target | fail_action |
| --- | --- | --- | --- | --- |
| G-001 | Hardest-window word-level speaker accuracy | `> 0.30` | current WSL baseline `0.1774` | block_merge |
| G-002 | Mean top-3-window speaker accuracy | `>= 2.0x baseline` | current WSL baseline mean | block_merge |
| G-003 | Unit/integration tests | all pass | `pytest -q` | block_merge |

## 6. Evaluation Requirements

- Use the same `Session 22` flattened-multitrack harness as baseline.
- Report both coverage and word-level speaker accuracy.
- Keep reference generation fixed and only vary the mixed-audio speaker-identification path.

## 7. Traceability Map

| implementation_decision | claim_ids | citation_ids | rationale |
| --- | --- | --- | --- |
| Segment-level matching is the default path | C-001 | CIT-001 | x-vector style embeddings are most useful when we keep local segment evidence instead of collapsing to a single label vector. |
| Whitening support remains available in the bank scorer | C-003 | CIT-006 | Normalization is a low-risk way to handle cross-session mismatch. |
| AS-Norm is implemented as an optional scorer | C-002 | CIT-002 | The repo already has a bank/cohort structure that can support adaptive score normalization. |
| Fine-tuning is deferred | C-004, C-005 | CIT-003, CIT-004 | We should only pay the retraining cost after backend calibration stops being the bottleneck. |

## 8. Spec Delta (Required)

| delta_id | baseline_spec | change_summary | rationale | citation_ids |
| --- | --- | --- | --- | --- |
| D-001 | Existing speaker-bank flow | Default to segment-level evidence instead of diar-label-only matching | Biggest measured lift in the current harness comes from segment-level matching. | CIT-001 |
| D-002 | Existing cosine/prototype scorer | Add adaptive S-Norm support to the bank scorer | The repo exposed the config but did not implement the normalization path. | CIT-002 |
| D-003 | Existing conservative defaults | Lower default acceptance threshold to `0.40` and require only `2` segments per diar label | Current defaults were too conservative for mixed-audio assignment. | CIT-002 |

## 9. Locked Defaults For Coding (Required)

| default_id | field | locked_value | reason | citation_ids |
| --- | --- | --- | --- | --- |
| L-001 | `speaker_bank.match_per_segment` | `true` | Best observed lift comes from segment-level evidence. | CIT-001 |
| L-002 | `speaker_bank.threshold` | `0.40` | Better fit to mixed-audio bank scores than the earlier overly conservative setting. | CIT-002 |
| L-003 | `speaker_bank.min_segments_per_label` | `2` | Keeps sparse labels from dominating while avoiding over-pruning. | CIT-001 |
| L-004 | `speaker_bank.scoring.as_norm.enabled` | `false` | Available, but requires post-implementation calibration before making it the default. | CIT-002 |
