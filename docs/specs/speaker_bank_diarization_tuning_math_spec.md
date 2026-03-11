# 1. Objective and Mathematical Scope

- Hypothesis under test: closed-set speaker assignment on mixed-session audio improves materially when we score segment-level embeddings against the speaker bank, then calibrate cosine similarity with whitening and adaptive symmetric score normalization.
- Target observable(s): word-level speaker accuracy and coverage in the `multitrack_eval.py` harness.
- In-scope equations/rules: cosine similarity, adaptive S-Norm, whitening transform, segment-label aggregation.
- Out-of-scope equations/rules: end-to-end diarization re-training, PLDA parameter estimation, overlap-aware resegmentation.

## 2. Symbol Glossary and Units

| symbol | meaning | units | allowed range | citation_id |
| --- | --- | --- | --- | --- |
| `x` | raw speaker embedding | unitless | finite vector | CIT-001 |
| `q` | query embedding after normalization | unitless | `||q||_2 = 1` | CIT-001 |
| `e` | enrollment / bank embedding after normalization | unitless | `||e||_2 = 1` | CIT-001 |
| `s` | raw cosine similarity score | unitless | `[-1, 1]` | CIT-001 |
| `mu_e`, `sigma_e` | candidate-side cohort mean/std | unitless | `sigma_e > 0` | CIT-002 |
| `mu_t`, `sigma_t` | query-side cohort mean/std | unitless | `sigma_t > 0` | CIT-002 |
| `k` | adaptive cohort size | count | integer `>= 1` | CIT-002 |

## 3. Normalized Governing Equations

- Equation ID: EQ-001
- Published form: cosine-style backend similarity between normalized embeddings.
- Normalized form: `s(q, e) = q^T e`
- Assumptions: embeddings are finite and length-normalized.
- Domain of validity: closed-set speaker comparison on fixed-dimensional embeddings.
- Citation IDs: CIT-001
- Source locator: section:2.2
- Normative status: normative
- OCR quality: not_ocr

- Equation ID: EQ-002
- Published form: adaptive symmetric score normalization.
- Normalized form: `as_norm(s, q, e) = 0.5 * (((s - mu_e) / sigma_e) + ((s - mu_t) / sigma_t))`
- Assumptions: query-side and candidate-side cohort statistics are computed from non-target cohort scores, using the top-`k` cohort items.
- Domain of validity: score calibration under domain/language/channel mismatch.
- Citation IDs: CIT-002
- Source locator: section:3;eq:7
- Normative status: normative
- OCR quality: not_ocr

- Equation ID: EQ-003
- Published form: whitening / normalization transform (implementation adaptation).
- Normalized form: `z = normalize(W * (x - mu))`
- Assumptions: `W` is derived from the enrollment covariance eigendecomposition with numerical floor on small eigenvalues.
- Domain of validity: mismatched enrollment/test conditions where raw cosine scores compress.
- Citation IDs: CIT-006
- Source locator: section:Results
- Normative status: non_normative
- OCR quality: not_ocr

## 4. Parameterization and Bounds

| parameter | physical meaning | units | bound/range | source_basis | citation_id |
| --- | --- | --- | --- | --- | --- |
| `threshold` | minimum accepted speaker-bank score | unitless | tuned on eval harness; default `0.40` in repo | engineering_constraint | CIT-002 |
| `margin` | minimum top1-top2 separation | unitless | `>= 0` | engineering_constraint | CIT-002 |
| `cohort_size` | adaptive cohort top-K | count | `1..N-1`; repo default `50` | primary | CIT-002 |
| `min_segments_per_label` | minimum segment evidence before label aggregation | count | integer `>= 1`; repo default `2` | engineering_constraint | CIT-001 |

## 5. Calibration and Identifiability

- Minimum data requirements: labeled enrollment embeddings for each known speaker; at least one additional speaker per cohort for AS-Norm.
- Identifiability conditions: known cast list is closed and training labels are trustworthy.
- Calibration objective: maximize word-level speaker accuracy on flattened multitrack windows without materially reducing coverage.
- Known non-identifiable combinations: overlapping mixed segments can produce embeddings that are not cleanly attributable to a single speaker regardless of backend.

## 6. Applicability and Failure Modes

| condition | behavior | failure mode | mitigation | citation_id |
| --- | --- | --- | --- | --- |
| Raw diar-label embedding only | Under-uses within-label evidence | Misses correct speaker or collapses to `unknown` / dominant speaker | Use per-segment embeddings and aggregate | CIT-001 |
| Channel / session mismatch | Compresses cosine margins | Thresholds reject otherwise plausible matches | Enable whitening and optionally AS-Norm | CIT-002 |
| Small or poor cohort | Noisy normalization | AS-Norm can over-amplify bad comparisons | Keep cohort size bounded and disable when cohort is too small | CIT-002 |
| Noisy fine-tuning labels | Model drift | In-domain fine-tune regresses generalization | Defer fine-tuning until backend errors are isolated | CIT-004 |

## 7. Math-Level Open Risks

- AS-Norm thresholds live on a different score scale than raw cosine thresholds and need empirical calibration in the eval harness.
- Whitening helps cross-source bank classification in our offline test, but the final gate is the mixed-audio word-level eval on WSL.
