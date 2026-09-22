# Speaker TSE Manifest Drive V1

## Purpose

This is the first concrete data artifact for the bigger-bet target-speaker extraction path. It turns
real multitrack DND/Discord sessions into rows shaped for target-speaker extraction or
speaker-attributed ASR training, instead of post-hoc relabeling Whisper words.

## Build

Command run on WSL from `<repo>`:

```bash
uv run python scripts/build_speaker_tse_manifest.py \
  --sessions 58,59,60,61,64,67 \
  --dev-sessions 61 \
  --test-sessions 64,67 \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v1 \
  --windows-per-session 2 \
  --min-speakers 6 \
  --min-target-words 20 \
  --materialize-limit 6
```

Manifest:

`<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v1/speaker_tse_manifest.jsonl`

## Summary

| split | rows | target words |
| --- | ---: | ---: |
| train | 31 | 5468 |
| dev | 10 | 1666 |
| test | 22 | 3624 |

| speaker | rows | target words |
| --- | ---: | ---: |
| Cletus Cobbington | 7 | 740 |
| Cyrus Schwert | 11 | 1405 |
| David Tanglethorn | 11 | 1629 |
| Dungeon Master | 12 | 3491 |
| Kaladen Shash | 11 | 1623 |
| Leopold Magnus | 11 | 1870 |

| target share bucket | rows |
| --- | ---: |
| lt_010 | 16 |
| 010_020 | 23 |
| 020_035 | 14 |
| 035_050 | 9 |
| 050_075 | 1 |

## Leakage Audit

- Train sessions: 58, 59, 60
- Dev session: 61
- Test sessions: 64, 67
- Missing positive enrollment rows: 0
- Missing negative enrollment rows: 0
- Enrollment/window overlap violations: 0
- Held-out session leakage: 0

## Materialized Sanity Check

The first six rows were materialized as actual wav files:

`<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v1/materialized/train`

They cover one complete six-speaker Session 58 window:

| field | value |
| --- | ---: |
| window | 1260-1560s |
| materialized speakers | 6 |
| window transcript words | 979 |
| sum of target words across source rows | 979 |
| exact source-oracle word coverage | 100% |

Audio validation found no missing mixture, target-source, positive-enrollment, or
negative-enrollment files. Each row has a 300 second `mixture.wav`, a matching `target.wav`, two
positive enrollment clips, and two negative enrollment clips.

## Next Gate

The next architecture test should consume this manifest directly. A model run should report:

- mixed baseline score on the same rows,
- extracted-target score for each target speaker,
- source-oracle ceiling for the materialized rows,
- held-out Session 64/67 results separate from train/dev,
- license and checkpoint provenance for any external TSE model.

## V3 All-Test Materialization

For external TSE scoring, the same manifest recipe was rerun with all held-out test rows
materialized:

```bash
uv run python scripts/build_speaker_tse_manifest.py \
  --sessions 58,59,60,61,64,67 \
  --dev-sessions 61 \
  --test-sessions 64,67 \
  --output-dir .outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized \
  --windows-per-session 2 \
  --min-speakers 6 \
  --min-target-words 20 \
  --materialize-limit 40 \
  --materialize-splits test
```

Artifact:

`<repo>/.outputs/speaker_id_baseline_prod_graph/artifacts/tse_manifest/codex_drive_v3_test_all_materialized/speaker_tse_manifest.jsonl`

Summary:

| field | value |
| --- | ---: |
| total rows | 63 |
| materialized test rows | 22 |
| Session 64 test rows | 11 |
| Session 67 test rows | 11 |
| held-out target words | 3624 |
| leakage audit | pass |

Raw mixture baseline on these 22 rows:

| rows | target words | mean target share | mean mixture SI-SDR |
| ---: | ---: | ---: | ---: |
| 22 | 3624 | 0.181505 | -7.814428 |
