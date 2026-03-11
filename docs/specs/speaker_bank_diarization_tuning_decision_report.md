## 1. Decision Summary

- Selected variant: segment-level closed-set speaker-bank matching with lower default thresholds, optional whitening, and optional adaptive S-Norm.
- Rejected variants: PCA-driven matching, immediate end-to-end diarization fine-tuning, and replacing the current bank scorer with an LDA/logistic backend as the first move.
- Decision date: 2026-03-10
- Decision owner: Codex

## 2. Variant Comparison

| variant_id | description | strengths | weaknesses | applicability | decision | citation_ids |
| --- | --- | --- | --- | --- | --- | --- |
| V-001 | Segment-level cosine/prototype matching with label aggregation | Uses existing pipeline, biggest measured gain in current harness, easy to validate | Still depends on diarization quality | Immediate | selected | CIT-001 |
| V-002 | Whitening + adaptive S-Norm | Literature-backed robustness for channel/session mismatch | Requires threshold recalibration | Immediate, optional | selected | CIT-002, CIT-006 |
| V-003 | PCA-based speaker matching | Cheap visualization aid | PCA is not a discriminative scoring backend | Not suitable as primary scorer | rejected | CIT-001 |
| V-004 | Immediate pyannote / speaker-model fine-tuning | Can adapt to in-domain audio | More data/compute-heavy; risk of overfitting noisy labels | Phase 2 only | rejected | CIT-003, CIT-004 |
| V-005 | Replace scorer with LDA or logistic classifier by default | Could help if bank embeddings are already well-separated | Our offline leave-one-source check favored whitening+cosine over LDA/logreg on the current bank | Follow-up only | rejected | CIT-001, CIT-006 |

## 3. Rejection Rationale

- PCA is a visualization tool, not a robust closed-set speaker-identification backend.
- Immediate fine-tuning is unjustified until the simpler scoring/backend issues are exhausted.
- A discriminative classifier backend is not yet the best default because our offline bank-only check favored whitening plus cosine-style scoring over LDA/logistic on cross-source holdout.

## 4. Original vs Adaptation Resolution

| conflict_id | original_variant | adaptation_variant | normative_choice | compatibility_note | citation_ids |
| --- | --- | --- | --- | --- | --- |
| CFG-001 | Raw x-vector backend | Adaptive S-Norm on top of cosine/prototype scoring | adaptation | We are not replacing the embedding extractor, only calibrating its scores to better survive cross-session mismatch. | CIT-001, CIT-002 |
| CFG-002 | Raw normalized embeddings | Whitening before cosine/prototype scoring | adaptation | Whitening is treated as a switchable robustness transform, not a new embedding model. | CIT-006 |

## 5. No-Go Check Result

| condition | status | evidence |
| --- | --- | --- |
| Missing primary citation for core equation | pass | CIT-001 and CIT-002 cover the scoring equations/rules. |
| Unresolved unit mismatch | pass | All quantities are unitless embedding/score values. |
| Unidentifiable parameter set | pass | Threshold, margin, and cohort size are directly measurable in the eval harness. |
| Applicability regime mismatch | pass | Sources cover speaker recognition/diarization under mismatch and labeled adaptation. |
| Missing measurable acceptance gates | pass | Gates are defined in the implementation spec against the Session 22 harness. |
| Core OCR-uncertain claim unresolved | pass | All cited sources are digital, not OCR-derived. |

## 6. Locked Defaults

| default_id | field | locked_value | reason | citation_ids |
| --- | --- | --- | --- | --- |
| L-001 | `speaker_bank.match_per_segment` | `true` | Best immediate lift with minimal architectural change. | CIT-001 |
| L-002 | `speaker_bank.threshold` | `0.40` | Current mixed-audio scores need a less conservative floor. | CIT-002 |
| L-003 | `speaker_bank.min_segments_per_label` | `2` | Keeps evidence thresholds practical on short diar labels. | CIT-001 |
| L-004 | `speaker_bank.scoring.as_norm.enabled` | `false` | Implemented but not defaulted until WSL calibration is rerun. | CIT-002 |

## 7. Spec Change Protocol

Any future change to the scorer, threshold family, or default diarizer must be re-run through the same `multitrack_eval.py` windows and documented as an explicit spec delta.

## 8. Final Status (Machine Readable)

status = ready_for_implementation
