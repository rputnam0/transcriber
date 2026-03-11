# Source Inventory

| citation_id | source_id | title | year | type | classification | confidence_tier | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIT-001 | snyder-2018-xvectors | X-Vectors: Robust DNN Embeddings for Speaker Recognition | 2018 | paper | primary | A | Primary method source for speaker-embedding extraction and the standard x-vector backend stack (`LDA -> PLDA`). URL: <https://www.danielpovey.com/files/2018_icassp_xvectors.pdf> |
| CIT-002 | matejka-2017-asnorm | Analysis of Score Normalization in Multilingual Speaker Recognition | 2017 | paper | primary | A | Primary source for adaptive symmetric score normalization. URL: <https://www.isca-archive.org/interspeech_2017/matejka17b_interspeech.html> |
| CIT-003 | speechbrain-speaker-cls | SpeechBrain Tutorial: Speech Classification from Scratch | 2025 | docs | secondary | B | Official implementation guidance showing that speaker-ID models can be trained/fine-tuned directly from labeled utterances. URL: <https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-classification-from-scratch.html> |
| CIT-004 | pyannote-audio-readme | pyannote.audio README | 2026 | docs | secondary | B | Official guidance that the speaker-embedding model can be fine-tuned and that transfer learning on roughly 10h of annotated data is practical. URL: <https://github.com/pyannote/pyannote-audio> |
| CIT-005 | pyannote-community-1 | pyannote speaker-diarization-community-1 model card | 2025 | docs | secondary | B | Official model card noting improved speaker counting and assignment for multi-speaker diarization. URL: <https://huggingface.co/pyannote/speaker-diarization-community-1> |
| CIT-006 | perez-2024-normalization | Comparative analysis of speaker recognition pipelines under realistic forensic conditions | 2024 | paper | primary | B | Recent primary study showing that embedding-level or score-level normalization improves speaker recognition robustness under mismatched conditions. URL: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11478696/> |

## Root-Index Triage Notes

| index_file | purpose | used_for_evidence | note |
| --- | --- | --- | --- |
| [docs/PR-robust-speaker-identification.md](/Users/rexputnam/Documents/projects/transcriber/docs/PR-robust-speaker-identification.md) | Existing repo design notes | no | Useful local context only; not normative evidence. |
| pyannote README / model card indexes | Source discovery | no | Used to find official fine-tuning and model-capability statements. |

## Coverage Against Scope-Locked Claims

| claim_id | required_for_go | supporting_citation_ids | status | gap_note |
| --- | --- | --- | --- | --- |
| CLM-001 | yes | CIT-001, CIT-002 | supported | Closed-set speaker ID should use embedding backends stronger than raw PCA visualization. |
| CLM-002 | yes | CIT-002, CIT-006 | supported | Whitening and cohort score normalization are justified for mismatched conditions. |
| CLM-003 | yes | CIT-003, CIT-004 | supported | Fine-tuning on labeled speaker data is feasible, but not required before backend calibration. |
| CLM-004 | yes | CIT-005 | supported | Better diarization models can help, but speaker assignment and calibration remain separate levers. |

## Original vs Adaptation Conflict Log

| conflict_id | original_citation_id | adaptation_citation_id | difference_summary | normative_choice | rationale |
| --- | --- | --- | --- | --- | --- |
| CFG-001 | CIT-001 | CIT-002 | Raw x-vector backends typically use LDA/PLDA; later systems add adaptive score normalization for better cross-domain robustness. | adaptation | The repo already operates as a closed-set speaker bank rather than a full PLDA stack, so adaptive score normalization is the best low-risk adaptation we can apply immediately. |
| CFG-002 | CIT-001 | CIT-006 | The original x-vector paper centers on the classic backend; newer evaluation emphasizes normalization under real-world mismatch. | adaptation | Our failure mode is cross-session/channel mismatch, so keeping cosine-style scoring while adding whitening is better aligned to the application than introducing a fresh discriminative backend first. |
