## Blocking Items

| item_id | blocker_type | severity | due_phase | description | missing_evidence_or_decision | owner | unblock_condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OI-001 | data_gap | low | implementation | No active blocking items remain. The tuned three-window WSL evaluation completed on 2026-03-10. | none | codex | n/a |

## Non-Blocking Follow-Ups

| item_id | impact | severity | due_phase | description | proposed_next_step |
| --- | --- | --- | --- | --- | --- |
| OI-101 | high | medium | implementation | Calibrate AS-Norm thresholds on the WSL harness before enabling it by default. | Sweep threshold/margin on the updated scorer; the first hard-window check regressed from `0.4097` to `0.1462`, so the feature should remain opt-in for now. |
| OI-102 | medium | medium | implementation | Evaluate `speaker-diarization-community-1` or equivalent against the same windows to separate diarization gains from bank-scoring gains. | Add a diarizer A/B path in the eval harness. |
| OI-103 | medium | low | implementation | Fine-tuning pyannote/SpeechBrain on labeled audio is still a viable phase-2 path if backend tuning plateaus. | Build a held-out split from labeled stems and compare against the tuned non-fine-tuned baseline first. |
