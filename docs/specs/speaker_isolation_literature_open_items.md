# Speaker Isolation Literature Open Items

## Blocking Items

| item_id | blocker_type | severity | due_phase | description | missing_evidence_or_decision | owner | unblock_condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | no_go | low | literature | No literature blocker remains for starting a target-speaker activity experiment. | n/a | Codex | n/a |

## Non-Blocking Follow-Ups

| item_id | impact | severity | due_phase | description | proposed_next_step |
| --- | --- | --- | --- | --- | --- |
| OI-101 | high | high | implementation | Exact frame-label recipe must be locked for multitrack stems. | Derive per-speaker 20 ms or 40 ms activity labels from aligned stems using RMS/VAD thresholds; record thresholds in the experiment output JSON. |
| OI-102 | high | high | implementation | Split hygiene is critical because the repo has repeated speakers, sessions, and synthetic flattened windows. | Use leave-session or leave-window splits; never let source crops from the same flattened evaluation window appear in target activity training. |
| OI-103 | medium | medium | implementation | Speaker embeddings used for conditioning may differ from embeddings used by the current LDA scorer. | Compare Titanet, pyannote, and learned enrollment summaries as conditioning vectors while holding the frame model fixed. |
| OI-104 | medium | medium | implementation | Overlap labels can be ambiguous at word boundaries and during short backchannels. | Evaluate both frame F1 and word-speaker accuracy; bucket results by target energy share and active-speaker count. |
| OI-105 | medium | medium | implementation | Heavy target-extraction architectures may require more GPU memory than the quick sweeps. | Stage them behind an MVP gate; only fine-tune SepFormer/TF-GridNet/SpEx+-class models if target activity alone cannot close the gap. |
| OI-106 | medium | low | implementation | External pretrained model licenses and Hugging Face access may affect distributability. | Record model source, license, and cache requirements in any future implementation PR. |
| OI-107 | high | high | implementation | Current all-candidate target extraction produces useful oracle candidates, but post-hoc confidence selectors cannot identify the correct speaker reliably. | Add a direct word-ownership or candidate-calibration objective during extractor training, then require the deployable selector to recover at least half of the oracle mixed-plus-any-candidate gain. |
| OI-108 | high | high | implementation | A larger/deeper TasNet now reaches 73.07% leave-window and 64.13% strict-session true-target hard accuracy, and a stronger fixed-channel all-source run reaches 69.33% true-channel accuracy with an 85.00% oracle union, but deployable selectors still stay near mixed. | Make candidate calibration/word ownership part of the extractor objective, or move to an ASR-integrated word-attribution model rather than another shallow selector, phrase-smoothing pass, sequence candidate router, activity-confidence router, aggregate class-evidence router, fixed-channel owner-weighting pass, mask-activation swap, small mask reconstruction tweak, or pooled crop classifier. |
| OI-109 | high | medium | implementation | Recent TSE literature suggests direct enrollment context; the first small positive/negative enrollment encoder improves the 300-row smoke but reaches only 58.84% on the full hard set. | If continuing this direction, replace the small raw encoder with stronger enrollment/mix attention or a pretrained/adapted TSE backbone, then score by hard-overlap word speaker accuracy. |
