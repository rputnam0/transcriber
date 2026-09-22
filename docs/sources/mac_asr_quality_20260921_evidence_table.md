# Evidence

| claim_id | citation_id | location | rule | confidence_tier | ocr_quality |
| --- | --- | --- | --- | --- | --- |
| E1 | C1 | url:https://huggingface.co/Qwen/Qwen3-ASR-1.7B | ASR and separate forced-alignment interface; general benchmark claims do not prove D&D performance. | A | not_ocr |
| E2 | C2 | url:https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md | Apple Silicon implementation; ASR and word alignment are separate operations. | A | not_ocr |
| E3 | C3 | url:https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/transcribe.py | Word timestamps and disabling previous-text conditioning are available. | A | not_ocr |
| E4 | C4 | url:https://huggingface.co/ibm-granite/granite-4.0-1b-speech | Compact English ASR with an explicitly documented MLX path. | A | not_ocr |
| E5 | C5 | url:https://huggingface.co/blog/ibm-granite/granite-speech-5-0-470m-turboctc | August 2026 encoder-only alternative; benchmark GPU throughput is not Mac throughput. | A | not_ocr |
| E6 | C6 | url:https://github.com/xzf-thu/Mega-ASR | Noise adaptation can degrade clean speech; author uses a quality router. | A | not_ocr |
| E7 | C7 | url:https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize | Joint text, speaker and timestamp output; retained as diarization/overlap evidence. | A | not_ocr |
| E8 | C8 | url:https://arxiv.org/abs/2609.07549 | September 2026 research candidate; paper alone does not establish downloadable Mac-ready weights. | A | not_ocr |
