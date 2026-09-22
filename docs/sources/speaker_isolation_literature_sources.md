# Speaker Isolation Literature Sources

## Source Inventory

| citation_id | source_id | title | year | type | classification | confidence_tier | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIT-001 | wang-2019-voicefilter | VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking | 2019 | paper | primary | A | Target-speaker separation from a mixed signal using reference speech-derived speaker embeddings and a spectrogram mask. URL: https://arxiv.org/abs/1810.04826 |
| CIT-002 | zmolikova-2019-speakerbeam | SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction in Speech Mixtures | 2019 | paper | primary | A | Target speaker extraction with an adaptation utterance; explicitly discusses avoiding label permutation and speaker-count dependence. URL: https://www.fit.vut.cz/research/group/speech/public/publi/2019/zmolikova_IEEEjournal2019_08736286.pdf |
| CIT-003 | ge-2020-spexplus | SpEx+: A Complete Time Domain Speaker Extraction Network | 2020 | paper | primary | A | Time-domain target-speaker extraction using reference speech and tied encoders. URL: https://arxiv.org/abs/2005.04686 |
| CIT-004 | medennikov-2020-tsvad | Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario | 2020 | paper | primary | A | Known-speaker frame activity prediction from acoustic features plus speaker vectors. URL: https://arxiv.org/abs/2005.07272 |
| CIT-005 | ding-2020-personalvad | Personal VAD: Speaker-Conditioned Voice Activity Detection | 2020 | paper | primary | A | Lightweight target-speaker/non-target/non-speech frame classifier conditioned on speaker embedding or verification score. URL: https://arxiv.org/abs/1908.04284 |
| CIT-006 | luo-2019-convtasnet | Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation | 2019 | paper | primary | A | Strong single-channel, speaker-independent time-domain separation baseline. URL: https://arxiv.org/abs/1809.07454 |
| CIT-007 | subakan-2021-sepformer | Attention is All You Need in Speech Separation | 2021 | paper | primary | A | Transformer-based SepFormer separation architecture with strong WSJ0-2/3mix results. URL: https://arxiv.org/abs/2010.13154 |
| CIT-008 | wang-2023-tfgridnet | TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation | 2023 | paper | primary | A | High-performing time-frequency separation model combining full-band, sub-band, and attention modules. URL: https://arxiv.org/abs/2211.12433 |
| CIT-009 | chen-2020-libricss | Continuous Speech Separation: Dataset and Analysis | 2020 | paper | primary | A | Continuous/meeting-style separation benchmark; warns pre-segmented SDR-style evaluation can mislead practical ASR/diarization systems. URL: https://www.microsoft.com/en-us/research/wp-content/uploads/2020/04/ICASSP2020__Continuous_speech_separation__dataset_and_analysis.pdf |
| CIT-010 | maiti-2022-eendss | EEND-SS: Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers | 2022 | paper | primary | A | Joint diarization, speaker counting, and Conv-TasNet-style separation. URL: https://arxiv.org/abs/2203.17068 |
| CIT-011 | kanda-2021-saasr | End-to-End Speaker-Attributed ASR with Transformer | 2021 | paper | primary | A | Monaural multi-talker ASR that jointly counts speakers, recognizes words, and identifies speakers, with optional target speaker profiles. URL: https://arxiv.org/abs/2104.02128 |
| CIT-012 | kanda-2022-transcribe-to-diarize | Transcribe-to-Diarize: Neural Speaker Diarization for Unlimited Number of Speakers using End-to-End Speaker-Attributed ASR | 2022 | paper | primary | A | Uses speaker-attributed ASR internals for diarization and speaker-attributed transcripts. URL: https://arxiv.org/abs/2110.03151 |
| CIT-013 | transcriber-ledger | Speaker ID Experiment Ledger | 2026 | repo | implementation_reference | A | Local measured evidence for this repo: mixed-mono, oracle mask, generic separator, domain separator, target extractor, fixed-channel, and candidate-selector sweeps. Path: docs/speaker-id-experiment-ledger.md |
| CIT-014 | yang-2024-contextual-tse | Target Speaker Extraction by Directly Exploiting Contextual Information in the Time-Frequency Domain | 2024 | paper | primary | A | Uses attention between enrollment and mixed-signal T-F representations instead of only a static speaker embedding. URL: https://arxiv.org/abs/2402.17146 |
| CIT-015 | zeng-2024-usef-tse | USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction | 2024 | paper | primary | A | Embedding-free target extraction with frame-level target-speaker feature extraction via multi-head cross-attention. URL: https://arxiv.org/abs/2409.02615 |
| CIT-016 | zhang-2024-multilevel-tse | Multi-Level Speaker Representation for Target Speaker Extraction | 2024 | paper | primary | A | Combines enrollment spectral features and contextual speaker embeddings to improve TSE generalization. URL: https://arxiv.org/abs/2410.16059 |
| CIT-017 | navon-2025-flowtse | FlowTSE: Target Speaker Extraction with Flow Matching | 2025 | paper | primary | A | Conditional flow-matching target extraction from enrollment and mixed mel-spectrograms, with an optional STFT-conditioned vocoder. URL: https://arxiv.org/abs/2505.14465 |
| CIT-018 | xu-2025-posneg-enrollment | Target Speaker Extraction through Comparing Noisy Positive and Negative Audio Enrollments | 2025 | paper | primary | A | Uses positive target-speaking and negative target-silent noisy enrollment segments to disambiguate target identity. URL: https://arxiv.org/abs/2502.16611 |
| CIT-019 | pallala-2025-targetvoice | TargetVoice: Single Channel Low-Latency Target Speaker Extraction | 2025 | paper | primary | A | Low-latency enrolled-speaker extraction for real-world calls and meetings with compact speaker encoder and extraction block. URL: https://www.isca-archive.org/interspeech_2025/pallala25_interspeech.html |

## Discovery And Triage Notes

| entry | purpose | used_for_evidence | note |
| --- | --- | --- | --- |
| web search: "VoiceFilter Targeted Voice Separation speaker conditioned spectrogram masking" | source discovery | yes | Found the arXiv paper and Google publication page; arXiv abstract used for primary method claim. |
| web search: "SpeakerBeam target speaker extraction adaptation utterance" | source discovery | yes | Found author-hosted IEEE paper PDF; used for problem definition and mask equations. |
| web search: "SpEx+ complete time domain speaker extraction network" | source discovery | yes | Found arXiv paper; used for time-domain target-speaker extraction variant. |
| web search: "TS-VAD target speaker voice activity detection diarization overlap" | source discovery | yes | Found TS-VAD arXiv/Interspeech paper; used for known-speaker activity recommendation. |
| web search: "Personal VAD speaker-conditioned voice activity detection" | source discovery | yes | Found arXiv and Google author page; used for lightweight conditioned VAD option. |
| web search: "Conv-TasNet SepFormer TF-GridNet speech separation arXiv" | source discovery | yes | Found primary sources for strong blind/separation architectures. |
| web search: "LibriCSS continuous speech separation dataset analysis" | source discovery | yes | Found Microsoft paper defining continuous separation and evaluation pitfalls for meeting-like audio. |
| web search: "speaker attributed ASR LibriCSS target speaker profiles" | source discovery | yes | Found SA-ASR and Transcribe-to-Diarize papers; used as a high-integration future path. |
| web search: "2024 target speaker extraction contextual enrollment cross attention" | source discovery | yes | Found ICASSP/TASLP-era TSE work that exploits enrollment context rather than only static speaker embeddings. |
| web search: "2025 target speaker extraction flow matching positive negative enrollment" | source discovery | yes | Found FlowTSE and noisy positive/negative enrollment TSE; used to update the next-architecture recommendation. |
| web search: "Interspeech 2025 low latency target speaker extraction TargetVoice" | source discovery | yes | Found TargetVoice as evidence that compact real-time TSE is active in call/meeting settings. |

## Coverage Against Scope-Locked Claims

| claim_id | required_for_go | supporting_citation_ids | status | gap_note |
| --- | --- | --- | --- | --- |
| CLM-001 | yes | CIT-001, CIT-002, CIT-003 | supported | Literature supports using enrollment/reference speech to condition a model to extract one target speaker from a mixture. |
| CLM-002 | yes | CIT-004, CIT-005 | supported | Literature supports directly predicting target-speaker activity, which maps more directly to word speaker attribution than waveform separation. |
| CLM-003 | yes | CIT-006, CIT-007, CIT-008, CIT-009 | supported | Literature supports strong separation models, but also documents permutation and practical continuous-evaluation issues. |
| CLM-004 | yes | CIT-009, CIT-011, CIT-012 | supported | Meeting-style overlap can be handled by continuous separation or speaker-attributed ASR, but those are larger system rewrites. |
| CLM-005 | yes | CIT-013 | supported | Repo-local experiments show frozen embedding backends succeed on clean/oracle-isolated speech and fail on short mixed overlap, so more clean speaker-bank data alone is not the bottleneck. |
| CLM-006 | yes | CIT-014, CIT-015, CIT-016, CIT-017, CIT-018, CIT-019 | supported | Recent TSE work suggests the next extractor should use richer enrollment/context conditioning, not just a single frozen speaker embedding or centroid. |

## Original vs Adaptation Conflict Log

| conflict_id | original_citation_id | adaptation_citation_id | difference_summary | normative_choice | rationale |
| --- | --- | --- | --- | --- | --- |
| CFG-001 | CIT-006 | CIT-001, CIT-002, CIT-003 | Blind separation estimates all sources and leaves source assignment/permutation to later logic; target extraction uses enrollment speech to choose the source during separation. | adaptation | The repo has known speakers and abundant enrollment audio, so target-conditioned extraction better matches the task than speaker-independent PIT separation. |
| CFG-002 | CIT-001, CIT-002, CIT-003 | CIT-004, CIT-005 | Target extraction produces a waveform or mask; TS-VAD/PVAD predicts frame-level target activity directly. | adaptation | The immediate product metric is word speaker attribution, not separated waveform quality, so target activity is the smaller and more directly measurable next experiment. |
| CFG-003 | CIT-011, CIT-012 | current ASR-plus-diarization pipeline | Speaker-attributed ASR can jointly emit words and speaker labels; the current repo already has ASR, labels, and a speaker bank. | current pipeline plus targeted activity module | Full SA-ASR is promising but too invasive for the next repo experiment; use it as a later reference if target activity plateaus. |
| CFG-004 | CIT-001, CIT-002, CIT-003 | CIT-014, CIT-015, CIT-016, CIT-018 | Earlier TSE often conditions on a single speaker embedding; newer variants compare richer enrollment/mix context or positive/negative segments. | adaptation | Local centroid conditioning failed, so the next TSE architecture should let the model attend to enrollment audio/features directly. |
