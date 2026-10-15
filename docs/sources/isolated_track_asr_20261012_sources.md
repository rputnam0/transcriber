# Sources: isolated-track ASR

Accessed 2026-10-12. A = direct evidence for the stated interface or observation;
B = useful evidence with an applicability or adaptation gap. Source confidence
does not establish performance on this archive.

## Source Inventory

| citation_id | source_id | title / location | year | type | classification | confidence_tier | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | faster-whisper | [Transcription implementation](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py) | 2026 accessed | repo | official | A | WhisperModel.transcribe; initial_prompt, hotwords, VAD |
| S2 | qwen-card | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 2026 | official_doc | official | A | Local inference and forced alignment |
| S3 | qwen-api | [Qwen inference implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py) | 2026 | repo | official | A | transcribe context argument and prompt construction |
| S4 | qwen-paper | [Qwen3-ASR technical report](https://arxiv.org/html/2601.21337v2) | 2026 | paper | primary | A | Context training; Table 3 author-run benchmarks |
| S5 | parakeet-card | [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 2025 | official_doc | official | A | Local NeMo ASR, timestamps, published evaluations |
| S6 | nemo-boost | [Word boosting](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html) | 2026 accessed | official_doc | official | A | GPU-PB decoder families and tuning |
| S7 | turbobias | [TurboBias](https://arxiv.org/html/2508.07014v1) | 2025 | paper | primary | A | Phrase boosting mechanism; no archive-specific result |
| S8 | canary-card | [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) | 2025 | official_doc | official | A | ASR and text-only LLM modes differ |
| S9 | granite-card | [Granite Speech 5.0 TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | 2026 | official_doc | official | A | English, compact CTC, current runtime paths |
| S10 | voxtral-card | [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | 2026 | official_doc | official | A | Open streaming candidate, lower priority for archive task |
| S11 | nemo-nightly | [NeMo Speech customization](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_language_modeling_and_customization.html) | 2026 accessed | official_doc | official | A | Current decoder support; pin implementation version |
| L1 | repo-whisper | `src/transcriber/asr.py:112` | 2026 inspected | repo | implementation_reference | A | Actual wrapper omits vocabulary parameters |
| L2 | repo-routing | `src/transcriber/cli.py:258`, `:2053`, `:2190`; `src/transcriber/parakeet_backend.py:12` | 2026 inspected | repo | implementation_reference | A | Backend selection, known stems, model aliases |
| L3 | repo-qwen | `scripts/mlx_speech_batch.py:21`; `scripts/run_context_asr_recordings.py:153`; `scripts/run_mac_asr_quality.py:33` | 2026 inspected | repo | implementation_reference | A | Production lacks vocabulary, experiment accepts hotwords |
| L4 | repo-proxy | `docs/analysis/asr_quality_20260921/normalized_benchmark_results.json`; `docs/specs/mac_asr_quality_20260921_decision_report.md` | 2026 | other | implementation_reference | A | Private underlying scores; explicitly automatic-reference proxy |

## Discovery And Triage Notes

Queries covered Qwen context vocabulary, NVIDIA Parakeet/Canary word boosting,
Granite 5 speech, and local Voxtral releases. Original model cards, papers, and
official source were opened before using claims. Commercial results and generic
LLM releases were excluded. Public benchmark numbers were not pooled across
different suites. Runtime feasibility on this exact Colab/Mac setup remains an
experiment requirement.

## Coverage Against Scope-Locked Claims

| claim_id | required_for_go | supporting_citation_ids | status | gap_note |
| --- | --- | --- | --- | --- |
| Current vocabulary gap | yes | L1–L3, S1 | supported | Implementation inspection only |
| Qwen contextual recognition | yes | S2–S4 | supported | MLX path requires parity check |
| Parakeet phrase boosting | yes | S5–S7, S11 | supported | Exact installed NeMo compatibility untested |
| Alternative checkpoints | yes | S8–S10 | supported | No new inference |
| Better archive WER | no for shortlist; yes for rollout | L4 | unsupported | Need human references and paired evaluation |

## Original vs Adaptation Conflict Log

| conflict_id | original_citation_id | adaptation_citation_id | difference_summary | normative_choice | rationale |
| --- | --- | --- | --- | --- | --- |
| CFG-1 | S3 | L3 | MLX custom batch helper bypasses official context API | S3 for API meaning; parity-test MLX | Do not assume prompts are forwarded |
| CFG-2 | S6, S11 | L2 | NeMo GPU-PB is not exposed by current Mac wrapper | Pinned official NeMo for boosting trial | Runtime support is not interchangeable |
