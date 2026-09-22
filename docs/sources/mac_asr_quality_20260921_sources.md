# Sources

## Source Inventory

| citation_id | title | classification | confidence_tier | notes |
| --- | --- | --- | --- | --- |
| C1 | [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | official | A | ASR and separate forced-alignment interface; general benchmark claims do not prove D&D performance. |
| C2 | [MLX Qwen ASR and aligner](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md) | official | A | Apple Silicon implementation; ASR and word alignment are separate operations. |
| C3 | [Whisper and Apple MLX implementation](https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/transcribe.py) | official | A | Word timestamps and disabling previous-text conditioning are available. |
| C4 | [Granite 4.0 Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | official | A | Compact English ASR with an explicitly documented MLX path. |
| C5 | [Granite 5.0 Turbo CTC announcement](https://huggingface.co/blog/ibm-granite/granite-speech-5-0-470m-turboctc) | official | A | August 2026 encoder-only alternative; benchmark GPU throughput is not Mac throughput. |
| C6 | [Mega-ASR](https://github.com/xzf-thu/Mega-ASR) | official | A | Noise adaptation can degrade clean speech; author uses a quality router. |
| C7 | [MOSS Transcribe Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | official | A | Joint text, speaker and timestamp output; retained as diarization/overlap evidence. |
| C8 | [Qwen-Audio-3.0-ASR report](https://arxiv.org/abs/2609.07549) | primary | A | September 2026 research candidate; paper alone does not establish downloadable Mac-ready weights. |

## Discovery And Triage Notes
Searched Qwen3 ASR, Apple MLX speech, Granite, Canary, Mega-ASR and overlapping multi-speaker ASR on 2026-09-21. Only author/maintainer sources support implementation. Model leaderboard claims are screening evidence, not accuracy estimates for these sessions. Canary was considered, but its documented NeMo/CUDA path is less direct than the shortlisted MLX implementations.
