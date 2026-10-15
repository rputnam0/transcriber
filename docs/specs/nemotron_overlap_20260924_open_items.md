# Open items

## Blocking Items

These block execution or adoption, not the completed research assessment.

| item | due phase | missing evidence | unblock condition |
| --- | --- | --- | --- |
| Runtime | First inference | Installed Mac environment lacks model support; repository NeMo pin is older | Isolated official-runtime load and inference succeed; pin revisions |
| Human overlap/word truth | Quality claim | Existing automatic stems and speaker-only reviews are insufficient | Freeze reviewed natural passages with simultaneous activity and both word streams |
| Local gain | Adoption | No Nemotron inference or paired comparison run | Pass detector, naming, word and clean-speech gates |
| Training recipe | Fine-tuning | Existing adapter predates new runtime/model | Verify loss, data format, parameter updates and checkpoint reload with a small smoke |

## Non-Blocking Follow-Ups

- Measure Mac execution speed and parity before choosing a converted runtime; vendor
  CUDA throughput is not a Mac estimate.
- Check that each evaluated recording has at most eight distinct voices; roster
  labels do not establish the actual audible count.
- Audit overlap labels for stem cross-talk and alignment errors. Do not treat unsure
  human labels as training truth.
- Keep current voice assets and annotation snapshots intact. The pre-existing private
  backup incompleteness is a separate issue, not a reason to replace assets here.

## Work completed

Read official release/API sources, inspected current fusion and historical scoring,
reconciled the old and current MOSS release notes, and probed installed package
support. No weights downloaded, inference run, training started, or production code
changed. Artifact validation is the only check required for this documentation change.
