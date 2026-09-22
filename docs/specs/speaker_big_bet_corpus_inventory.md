# Speaker Big-Bet Corpus Inventory

This inventory is a planning artifact for domain-scale target-speaker extraction and joint
speaker-attributed ASR work. It intentionally avoids recommending more shallow selectors or
small architecture tweaks.

## Summary

- Production root: `.outputs/speaker_id_baseline_prod_graph`
- Clean accepted bank: 11.45h
- Mixed candidate pool total: 18.50h
- Mixed candidate pool accepted: 14.76h
- Hard-negative records: 998
- Prepared eval windows: 7
- Prepared eval duration: 0.58h
- Prepared eval words: 5956

## Clean Bank By Speaker

| key | value |
| --- | ---: |
| Cletus Cobbington | 359 / 1.04h |
| Cyrus Schwert | 362 / 1.04h |
| David Tanglethorn | 483 / 1.57h |
| Dungeon Master | 1001 / 3.79h |
| Kaladen Shash | 514 / 1.63h |
| Leopold Magnus | 742 / 2.38h |

## Mixed Candidate Pool By Speaker

| key | value |
| --- | ---: |
| Cletus Cobbington | 1296 / 1.08h |
| Cyrus Schwert | 2530 / 1.55h |
| David Tanglethorn | 2684 / 2.23h |
| Dungeon Master | 8377 / 8.28h |
| Kaladen Shash | 3407 / 2.38h |
| Leopold Magnus | 3698 / 2.98h |

## Mixed Candidate Pool By Active Speakers

| key | value |
| --- | ---: |
| 1 | 12882 / 10.80h |
| 2 | 6594 / 5.34h |
| 3 | 1985 / 1.77h |
| 4 | 450 / 0.49h |
| 5 | 73 / 0.10h |
| 6 | 8 / 0.01h |

## Mixed Candidate Pool By Dominant Share

| key | value |
| --- | ---: |
| 030_045 | 249 / 0.23h |
| 045_060 | 1728 / 1.33h |
| 060_075 | 2127 / 1.56h |
| 075_090 | 2477 / 2.14h |
| 090_100 | 15406 / 13.24h |
| lt_030 | 5 / 0.01h |

## Hard Negatives By Pair

| key | value |
| --- | ---: |
| Cletus Cobbington::Cyrus Schwert | 147 |
| Cletus Cobbington::David Tanglethorn | 87 |
| Cyrus Schwert::Leopold Magnus | 196 |
| David Tanglethorn::Leopold Magnus | 144 |
| Dungeon Master::Kaladen Shash | 250 |
| Kaladen Shash::Leopold Magnus | 174 |

## Prepared Eval Words By Speaker

| key | value |
| --- | ---: |
| Cletus Cobbington | 884 |
| Cyrus Schwert | 333 |
| David Tanglethorn | 948 |
| Dungeon Master | 2085 |
| Kaladen Shash | 465 |
| Leopold Magnus | 1241 |

## Prepared Eval Words By Target Share

| key | value |
| --- | ---: |
| 030_045 | 169 |
| 045_060 | 159 |
| 060_075 | 189 |
| 075_090 | 232 |
| 090_100 | 4498 |
| lt_030 | 709 |

## Split Risks

- Prepared eval has too few independent sessions for upper-90 generalization claims.

## Next Manifest Requirements

- Create train/dev/test splits by held-out session, not by individual word/window.
- Balance hard-overlap examples by speaker and target-share bucket before domain-scale TSE.
- Include positive and negative enrollment sources outside each evaluation window.
- Record license and checkpoint provenance before external TSE model adaptation.
- Current prepared eval is too small for final claims; add more held-out sessions.
- Current eval is dominated by Session22/Session61; add unrelated held-out calls.

## 2026-08 Domain-Scale Update

The first target-speaker / speaker-attributed-ASR training handoff now uses Sessions 43-62 and 65
only. It contains 210 non-overlapping five-minute groups, 2,100 unique 30-second mono chunks
(17.5 hours), 1,156 candidate-speaker rows, and 157,303 labeled target words. Sessions 63 and 66
remain held out for development. Sessions 64 and 67 remain sealed and were not inspected during
the current model-selection cycle.

This resolves the immediate training-volume and session-leakage prerequisites. It does not resolve
the evaluation-width risk: two development sessions are enough for rejection tests, but not for a
production reliability claim or for calibrating low-confidence crosstalk handling.
