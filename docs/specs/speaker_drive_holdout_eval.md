# Speaker Drive Holdout Eval

## Scope

This note records a Drive-backed held-out check for the current modular flat-audio speaker
pipeline. It uses Google Drive `Dnd/audio` metadata for session discovery, WSL-local cached audio
for the large ZIPs, and Drive transcript exports for Sessions 64-67.

## Corpus Inputs

- Drive audio folder: `Dnd/audio`
- WSL audio cache:
  - `<repo>/data/prod/Audio/Session 64.zip`
  - `<repo>/data/prod/Audio/Session 65.zip`
  - `<repo>/data/prod/Audio/Session 66.zip`
  - `<repo>/data/prod/Audio/Session 67.zip`
- WSL transcript cache:
  - `<repo>/data/prod/Transcripts/Session 64/Session 64.txt`
  - `<repo>/data/prod/Transcripts/Session 65/Session 65.txt`
  - `<repo>/data/prod/Transcripts/Session 66/Session 66.txt`
  - `<repo>/data/prod/Transcripts/Session 67/Session 67.txt`

Sessions 65 and 66 include `B. Ver`, but the selected baseline speaker bank contains only the six
established speakers. Sessions 64 and 67 were used first because their top windows are clean
six-speaker closed-set holdouts.

## Evaluation Fix

`score_word_speaker_alignment` previously allowed a single predicted word span to match multiple
reference words. That produced impossible summaries such as more matched words than predicted words.
The scorer now consumes each predicted word at most once and reports `predicted_words` alongside
`reference_words`, `matched_words`, and `correct_words`.

The old best cached baseline, `baseline_profile_2cc05c9692a3`, changes from 69.4% total word-speaker
accuracy under the permissive scorer to 62.4% under the corrected one-to-one scorer:

| eval set | reference | predicted | matched | correct | accuracy | matched accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Session22 | 2560 | 2405 | 2236 | 1791 | 69.96% | 80.10% |
| Session61 | 2467 | 2039 | 1941 | 1375 | 55.74% | 70.84% |
| short segment slice | 929 | 777 | 753 | 548 | 58.99% | 72.78% |
| total | 5956 | 5221 | 4930 | 3714 | 62.36% | 75.33% |

## New Drive Holdouts

Control profile: `baseline_profile_2cc05c9692a3`

| session | window | reference | predicted | matched | correct | accuracy | matched accuracy | labels matched | diarization dominant share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4800-5100s | 896 | 677 | 642 | 444 | 49.55% | 69.16% | 4 / 6 | 0.894 |
| 67 | 2580-2880s | 861 | 742 | 714 | 453 | 52.61% | 63.45% | 3 / 6 | 0.914 |

Artifacts:

- `<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/eval/codex_drive_eval/session64_profile2cc_top1/summary.json`
- `<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/eval/codex_drive_eval/session67_profile2cc_top1/summary.json`

## Diagnosis

- The current evaluation number was inflated by a scorer bug; one-to-one scoring lowers the old
  headline baseline.
- The new six-speaker holdouts land well below the old corrected baseline, despite high temporal
  coverage and high diarization dominant-share diagnostics.
- Identity assignment is brittle: the profile matcher maps only 3-4 of 6 pyannote labels and often
  maps multiple raw labels to the same canonical speaker.
- The graph pass rescues unknown labels but also increases several cross-speaker confusions, so it is
  not a path to 90%+.
- The flat mixed ASR output contains fewer predicted words than the clean-stem reference. Any
  post-hoc speaker labeler is therefore working after word evidence has already been dropped or
  merged.

## Implication

This supports the bigger-bet conclusion: a modular Whisper -> pyannote -> embedding/profile
assignment pipeline is not shaped correctly for upper-90 word ownership on overlapped group audio.
The most promising next implementation path remains target-speaker extraction or speaker-attributed
ASR trained/evaluated on held-out multitrack sessions with direct enrollment context and a word-owner
objective.
