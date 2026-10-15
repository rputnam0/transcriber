# Nemotron overlap investigation: sources

Checked 2026-09-24 against repository revision `36c66c1`. No model inference performed.

## Source Inventory

| citation_id | source | classification | confidence_tier | scope |
| --- | --- | --- | --- | --- |
| N1 | [NVIDIA model card](https://huggingface.co/nvidia/Nemotron-3-Diarization) | official | A | Release, inference interfaces, operating settings, vendor benchmarks |
| N2 | [NVIDIA launch article](https://huggingface.co/blog/nvidia/nemotron-diarization) | official | A | Activity versus transcription; anonymous identities; overlap representation |
| N3 | [Transformers model documentation](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/nemotron3_diarization.md) | official | A | Native inference and streaming cache contract |
| L1 | [Current Mac workflow](../mac-single-file-transcription.md) and [ASR decision](../specs/mac_asr_quality_20260921_decision_report.md) | implementation_reference | A | Maintained release, limitations |
| L2 | [Previous Sortformer experiments](../specs/speaker_usef_tse_feasibility.md#sortformer-oracle-diarization-check) | implementation_reference | B | Historical proxy results, not current human accuracy |
| L3 | `scripts/score_sortformer_oracle_diarization.py`, `scripts/score_pyannote_oracle_diarization.py`, `scripts/build_contiguous_sortformer_mask_cutset.py`, `scripts/evaluate_contiguous_sortformer_enrollment_binding.py` | implementation_reference | A | Local scoring and probability/enrollment interfaces |
| L4 | `scripts/fuse_asr_diarization.py`, `scripts/evaluate_moss_overlap_fusion.py`, `scripts/evaluate_multitalker_parakeet_cutset.py` | implementation_reference | A | Fusion and existing experimental ASR hooks |
| L5 | Private `docs/analysis/asr_quality_20260921/README.md` and `docs/analysis/local_identity_and_interruptions_20260921/README.md` | implementation_reference | B | Historical measurements; unavailable in public clone |

## Discovery And Triage Notes

Started from the user-linked [Space](https://huggingface.co/spaces/nvidia/nemotron-diarization).
Searched NVIDIA/Hugging Face for Nemotron diarization, overlap, and Sortformer.
Used vendor and maintainer documentation for model claims. Excluded social posts
and unverified Apple runtime ports from recommendations. Space source-file fetches
failed; the model card explicitly links that demo. Did not upload private audio.

## Coverage Against Scope-Locked Claims

Capabilities and published comparisons: N1/N2. Integration: N1/N3/L3/L4.
Actual benefit on our audio: unsupported until a paired local experiment.

## Original vs Adaptation Conflict Log

Earlier Ultra-Sortformer adapters in L2 are different checkpoints, not measurements
of this release. Prefer the official runtime as the numerical reference; validate
any Mac port against it. L5's older interruption memo predates the ASR release;
L1 and the later ASR memo establish the current pilot-based baseline.
