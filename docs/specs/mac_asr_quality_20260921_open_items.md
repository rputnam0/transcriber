# Open items

## Blocking Items
None for local comparative implementation. A missing full Session 1 source blocks completion of that full session, not processing the five other files.

## Non-Blocking Follow-Ups
- Human word-level ground truth is absent. Existing grades concern speaker identity; automatic source-stem text is only a proxy and favors its generating recognizer.
- Latest paper-only models require verifiable local weights/runtime before adoption.
- User explicitly requests the 20-step pilot for this rerun; it is not independently validated as globally superior.
- Overlap can defeat single-stream ASR. Preserve overlap evidence and flag uncertain attributions.

## Validation findings

- Development candidates are close after consistent English normalization; eight minutes are insufficient for a strong statistical ranking.
- The integration validation influenced overlap routing. It cannot also be claimed as an untouched final evaluation.
- The hybrid preserves speaker-overlap recall but has mixed aggregate proxy word-error evidence. Human correction of representative words is the next reliable acceptance measure.
- Forced alignment sometimes collapses word times during simultaneous speech. These are flagged and the joint model's overlapping exchanges are retained with the alternative ASR wording available.
