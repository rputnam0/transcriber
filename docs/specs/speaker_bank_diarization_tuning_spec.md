# Speaker Bank Diarization Tuning Spec

This canonical note summarizes the current decision:

- The immediate fix is to treat speaker naming as a closed-set segment-level recognition problem layered on top of diarization, not a single label-vector lookup.
- The repo now defaults to segment-level speaker-bank matching and only uses label-level fallback when the aggregate score clears the configured threshold.
- Whitening is implemented as an optional backend transform because the literature supports it, but it remains disabled by default after the first calibration run regressed sharply.
- Fine-tuning is feasible with our labeled stems and timestamps, but the literature and the measured gains both point to backend calibration and diarization adaptation as the correct order of operations.

See the split artifacts for details:

- [Sources Ledger](/Users/rexputnam/Documents/projects/transcriber/docs/sources/speaker_bank_diarization_tuning_sources.md)
- [Evidence Table](/Users/rexputnam/Documents/projects/transcriber/docs/sources/speaker_bank_diarization_tuning_evidence_table.md)
- [Math Spec](/Users/rexputnam/Documents/projects/transcriber/docs/specs/speaker_bank_diarization_tuning_math_spec.md)
- [Implementation Spec](/Users/rexputnam/Documents/projects/transcriber/docs/specs/speaker_bank_diarization_tuning_implementation_spec.md)
- [Open Items](/Users/rexputnam/Documents/projects/transcriber/docs/specs/speaker_bank_diarization_tuning_open_items.md)
- [Decision Report](/Users/rexputnam/Documents/projects/transcriber/docs/specs/speaker_bank_diarization_tuning_decision_report.md)
