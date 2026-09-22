# Speaker Breakthrough Big Bets Open Items

## Blocking For Next Implementation

| item_id | blocker_type | severity | description | unblock_condition |
| --- | --- | --- | --- | --- |
| BB-OI-001 | evaluation_inventory | high | A 17.5-hour, 21-session training handoff and leakage-safe split now exist, but development still contains only Sessions 63/66 and is too narrow for production confidence calibration. | Add independent held-out development calls and publish overlap/interruption balance plus per-speaker coverage before setting abstention thresholds. |
| BB-OI-002 | license_check | high | ClearerVoice, WeSep, USEF-TSE, Sortformer, and Reverb have different model/code licenses and commercial constraints. | Record code license, checkpoint license, weights redistribution rights, and production restrictions for any candidate used. |
| BB-OI-003 | checkpoint_feasibility | high | Not every promising paper has usable pretrained weights or an inference API compatible with repo audio. | Run only a thin feasibility probe first: load model, feed one enrollment/mixture pair, emit waveform, and score a tiny known row. |
| BB-OI-004 | compute_budget | medium | The RTX 5080 can train the 212M-parameter SE-DiCoW speaker-conditioning block in FP32 with a frozen bfloat16 backbone, but architecture iteration remains materially slower than prior heads. | Preserve short hard-dev gates and checkpoint comparisons before any multi-epoch run. |
| BB-OI-005 | target_activity | high | Sortformer can cover the roster and reaches 72.76%, but remains anonymous. The new joint SE-DiCoW pVAD path mostly learns generic speech activity (97.58% recall, 13.02% precision) rather than target identity. | Adapt a serious pretrained, structurally enrollment-aware personal-VAD/TSE backbone jointly with target-word decoding; retain the implemented confidence and abstention interface. |

## Non-Blocking Follow-Ups

| item_id | impact | description | proposed_next_step |
| --- | --- | --- | --- |
| BB-OI-101 | high | The current eval windows are too small and can overfit architecture decisions. | Build a larger held-out-session eval set with balanced overlap and speaker coverage. |
| BB-OI-102 | high | Direct enrollment context is still the best TSE design clue. | Prefer positive/negative enrollment, cross-attention enrollment features, or embedding-free TSE over speaker centroids. |
| BB-OI-103 | medium | Speaker-attributed ASR may need different labels from the existing Whisper-word pipeline. | Generate token/word-level manifests that include speaker IDs, word times, and optional enrollment profiles. |
| BB-OI-104 | medium | External model adaptation may fail if sample rate, latency assumptions, or enrollment format mismatch Discord audio. | Add an adapter layer per model rather than folding model quirks into the core transcription code. |
