# Evidence table

Citation IDs resolve in [sources](nemotron_overlap_20260924_sources.md).

| claim_id | citation_id | location | claim_type | evidence | implementation_note | confidence_tier | ocr_quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | N1 | section:Description; Release Date; Model Architecture | capability | Released September 23, 2026; 100M parameters; eight speaker channels | New candidate, not the old four-speaker checkpoint | A | not_ocr |
| E2 | N2 | section:From audio to speaker activity; Diarization and Speaker Attributed ASR | interface | Independent activity channels preserve simultaneous voices; output does not supply words or real names | Retain multi-label probabilities and separate enrollment/ASR | A | not_ocr |
| E3 | N1 | section:Performance Evaluation Results; Metrics | validation | Offline DER: DIHARD full 19.09→12.73; AMI SDM 21.42→11.14; NOTSOFAR1 MHM 21.77→6.77; overlap included | Vendor comparison against v2.1, not our MOSS; meeting reference protocols matter | A | not_ocr |
| E4 | N1 | section:Setting up Streaming Configuration; Getting Diarization Results | interface | Tensor output available; offline parameters are specified in the implementation plan | Validate actual time grid and full tail coverage | A | not_ocr |
| E5 | N1,N3 | section:NVIDIA NeMo Speech; Software Integration; Transformers usage | prerequisite | NeMo 3 / Python 3.12+ documented; Transformers source installation offered | Isolate experimental dependencies | A | not_ocr |
| E6 | L3 | file:scripts/score_pyannote_oracle_diarization.py::_choose_turn_label; _score_group | limitation | Chooses one anonymous label per word before reference-derived mapping | Cannot be the sole overlap acceptance metric | A | not_ocr |
| E7 | L3 | file:scripts/build_contiguous_sortformer_mask_cutset.py::main | interface | Saves activity probabilities and diagnostic oracle slot mapping separately | Never use reference mapping in deployable inference | A | not_ocr |
| E8 | L4 | file:scripts/fuse_asr_diarization.py::fuse_words | limitation | Overlap routing requires two already-predicted named MOSS turns | Independent detector may reveal missed secondary turns; hypothesis only | B | not_ocr |
| E9 | L1,L5 | section:The handoff between words and voices matters | validation | Prior overlap word-recall proxy: 64.92% retained versus 54.03% naive fusion | No human WER or fresh holdout claim | B | not_ocr |
| E10 | L2 | section:Drive v6 larger-train Sortformer update | validation | More forced-label training improved overlap proxy while hurting other slices | Require clean-speech and naming regression gates | B | not_ocr |
