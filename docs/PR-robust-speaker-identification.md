# PR: Robust Speaker Identification via Segment-Level Voting, Prototypes, and Calibrated Scoring

## Summary

We want robust identification of unlabeled group audio using a speaker bank trained on labeled audio. The current bank now holds many embeddings per speaker (chunked training), and DBSCAN yields multiple clusters per person. However, matching remains sparse on cross-domain group audio because label-level embeddings and centroid-only scoring suppress otherwise plausible matches.

This PR proposes three core improvements executed in measured, incremental steps with validation at each stage:

1) Segment-level matching with per-label voting
2) Prototype-based scoring (centroid + exemplars) instead of centroid-only
3) Threshold + margin acceptance, with optional score normalization (AS-Norm) and CORAL whitening

## Context and Baseline

- Bank: 1,399 embeddings across 5 speakers after chunked enrollment of Session 22.
- Group audio (session_15.m4a): diarizer produced 15 labels and 4,711 segments; only 1 label matched and 16 segments labeled by the bank.
- Observed cosine scores: several diar labels in 0.57–0.69 range with very small top1–top2 margins (e.g., 0.660 vs 0.659). This indicates near-ambiguities that our current threshold/radius policy rejects.

Our goal is to preserve precision while increasing recall by leveraging per-segment evidence, richer per-speaker prototypes, and a margin rule.

## Proposed Changes

### 1) Segment-Level Matching + Voting

Problem: We currently match one vector per diar label against the bank. A diar label may span multiple styles/conditions; a single vector under-represents this variability.

Change:
- Extract embeddings for aligned ASR segments and score each segment individually.
- Aggregate scores per diar label to decide the assigned person (e.g., duration-weighted mean top score; or majority vote over segment-level top speakers passing a relaxed threshold).
- Write results to the JSONL (`speaker`, `speaker_match`, and `speaker_match_source = speaker_bank_segment`).

Controls:
- `speaker_bank.match_per_segment: true|false` (default false to keep behavior stable).
- `speaker_bank.match_aggregation: mean|vote` (default `mean`).
- `speaker_bank.min_segments_per_label: 3` (avoid labeling when evidence is too thin).

Implementation:
- Reuse `extract_embeddings_for_segments` (already added) to embed per-segment windows with padding and batching.
- Add a new path in CLI `run_transcribe` after alignment: if enabled, perform per-segment matching, then aggregate to diar labels; else, keep the current label-level matching.

Validation (Step 1):
- Baseline run (current): collect match rate stats.
- Enable `match_per_segment` and re-run session_15:
  - Expect increased matched diar labels and segment coverage.
  - Collect: matched labels, matched segments, average top1 cosine, average top1-top2 margin across labels.

Commands:
```
# baseline (already gathered):
uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO

# segment-level voting on (update config to enable):
uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO
```

### 2) Prototype-Based Scoring (Centroid + Exemplars)

Problem: Clusters can be wide/multi-modal. Centroids alone can sit far from some valid points.

Change:
- For each cluster, store a small set of prototypes in addition to the centroid.
  - Option A: k-means per cluster with K=3–5; prototypes = k-means centers.
  - Option B: select M most central points (lowest distance to centroid) as prototypes.
- Match-time score per speaker = max cosine across all prototypes in their clusters (still honor cluster variance/radius when available).

Controls:
- `speaker_bank.prototypes.enabled: true|false` (default true when prototypes available).
- `speaker_bank.prototypes.per_cluster: 3` (K or M per cluster; default 3).
- `speaker_bank.prototypes.method: kmeans|central` (default `central`).

Implementation:
- Extend `SpeakerBank._build_clusters()` to also compute/store prototypes.
- Extend `SpeakerBank.match()` to score against prototypes when enabled, else fallback to centroid path.
- Persist prototypes metadata in `bank.json`.

Validation (Step 2):
- Re-run session_15 with prototypes on and aggregation on/off to isolate the gain from prototypes.
- Record: matched labels/segments, average top1 cosine, average margin. Expect improved margins and slightly higher matches.

### 3) Scoring Policy: Threshold + Margin, Optional Normalization

Problem: Absolute cosine scores are depressed and near-tied by channel differences and label aggregation; we need calibrated acceptance.

Change:
- Add margin rule: accept a speaker only if `score >= threshold` AND `(top1 - top2) >= margin`.
- Expose threshold/margin in config; recommend defaults `threshold: 0.58` and `margin: 0.05` for cross-domain data.
- Optional: AS-Norm (cohort-based normalization) and CORAL whitening toggle for additional calibration.

Controls:
- `speaker_bank.scoring.threshold: 0.58`
- `speaker_bank.scoring.margin: 0.05`
- `speaker_bank.scoring.as_norm.enabled: false` (default false)
- `speaker_bank.scoring.as_norm.cohort_size: 50`
- `speaker_bank.scoring.whiten: false`

Implementation:
- Implement margin check in `SpeakerBank.match()` after ranking top scores for a label/segment.
- Add an optional AS-Norm path (collect impostor cohort from other-speaker prototypes) and CORAL whitening step for session vectors when enabled.

Validation (Step 3):
- Sweep threshold [0.55, 0.60] and margin [0.03, 0.08] using a small script; report matched labels/segments and estimate a precision proxy (e.g., percentage of matches with high margin).
- Keep the best pair for your dataset.

## Detailed Control Flow

1) Train (already in place):
   - Chunked training per speaker using windowing and padding (finished in previous PR).
   - Outputs: bank with clusters and PCA plot.

2) Segment-level inference:
   - After alignment, construct `(start,end,label)` tuples for all segments.
   - `extract_embeddings_for_segments()` → per-segment embeddings.
   - Score each segment against bank prototypes/centroids.
   - Aggregate to diar labels, apply threshold+margin, set final `speaker` for segments.

3) Prototype scoring path:
   - For each speaker cluster: centroid + prototypes (k-means or central points).
   - Score = max cosine vs prototypes for each speaker; keep best speaker if accepted by threshold+margin and radius.

4) Calibration (optional):
   - AS-Norm: compute cohort mean/std over top impostors; normalize scores.
   - CORAL: whiten session vector with transform learned from bank.

## Configuration Additions (YAML)

Under `speaker_bank`:
```
scoring:
  threshold: 0.58
  margin: 0.05
  as_norm:
    enabled: false
    cohort_size: 50
  whiten: false

prototypes:
  enabled: true
  per_cluster: 3
  method: central  # or kmeans

match_per_segment: true
match_aggregation: mean  # or vote
min_segments_per_label: 3
```

## Incremental Validation Plan (Step-by-Step)

Baseline (done):
- Metrics: matched labels=1/15; matched segments≈16/4711; near-threshold scores with tiny margins.

Step 1 — Segment-level matching:
- Enable `match_per_segment: true` with `threshold=0.58`, `margin=0.05`.
- Run: `uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO`.
- Record: matched labels, matched segments, avg top1, avg margin, and a histogram of margins. Expect a noticeable increase in matched segments and some labels crossing acceptance.

Step 2 — Prototypes:
- Enable prototypes with `per_cluster: 3`.
- Run the same test; track changes in top1 and margin. Expect modest lift and better stability.

Step 3 — Calibrate threshold/margin:
- Grid search small ranges with a quick script (no need to change code, can reprocess in memory). Choose the pair that balances recall vs. false positives.

Optional — AS-Norm, CORAL:
- If still borderline, toggle `as_norm` and/or `whiten` and re-evaluate. Only keep if they move the needle positively.

Acceptance:
- Document before/after metrics in the PR description with a clear table (matched labels, matched segments, margins) and attach updated PCA.

## Test Plan

- Unit tests:
  - Segment-level aggregation: simulate segment embeddings with controlled scores; verify label assignment and aggregation modes.
  - Prototypes: ensure prototypes are persisted and match paths include prototype scoring.
  - Threshold+margin gating: verify acceptance/rejection scenarios.
  - Optional: AS-Norm and whitening toggles are no-ops when disabled; basic shape checks when enabled with synthetic data.

- Integration:
  - Re-run training on a single ZIP (Session 22) with existing config.
  - Run transcription on session_15; assert increases in matched segments and diar labels relative to baseline and that results are deterministic within tolerance.

## Risks & Mitigations

- Runtime overhead: segment-level embedding increases inference time. Mitigate with batching (`embed_batch_size`) and CPU workers (`embed_workers`). Allow per-run toggle via `match_per_segment`.
- False positives: adding a margin rule and retaining a conservative threshold helps maintain precision.
- API variance: we already handle whisperx/pyannote embedder differences; continue to guard with try/except and clear logs.

## Rollout

1. Land segment-level matching (default off) and prototype scoring (default on with small per-cluster K).
2. Validate with session_15 under `threshold=0.58` / `margin=0.05`.
3. Tune and optionally introduce AS-Norm/whitening in a follow-up if needed.

## Commands (Quick Reference)

- Train from segments/stems (chunked):
```
uv run transcribe --config config/training.yaml --speaker-bank-train-only "audio/Session 22.zip" --no-quiet --log-level INFO
```
- Transcribe with segment-level matching and prototypes:
```
uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO
```
- Inspect outputs:
```
.open .outputs/session_15/session_15.speaker_bank.json
.open .hf_cache/speaker_bank/default/default.training_summary.json
.open .hf_cache/speaker_bank/default/pca.png
```

## Current Validation Snapshot

- Training (`uv run transcribe --config config/training.yaml --speaker-bank-train-only "audio/Session 22.zip" --no-quiet --log-level INFO`)
  - Bank entries: 2,793 (after ingesting Session 22 with chunked windows)
  - Multiple clusters per speaker with stored prototypes (e.g., `Leopold Magnus_c0/c1/c2`, `Dungeon Master_c0/c1`), each carrying three prototypes for persona coverage.

- Matching (`uv run transcribe audio/session_15.m4a --config config/transcriber.yaml --no-quiet --log-level INFO` with segment aggregation enabled)
  - Matched diar labels: 6 / 15 (baseline was 1 / 15)
  - Matched segments: 864 / 4,711 (~18.3%) assigned to named speakers instead of `unknown`
  - Aggregation: mean voting with margin=0.05; e.g., `SPEAKER_05 → Leopold Magnus` aggregates 43 matched segments (out of 422) with mean score ≈0.81 and margin ≈0.066.

---

This plan incrementally addresses our observed gaps: it replaces brittle label-level decisions with per-segment evidence, enriches per-speaker representation with prototypes, and adds calibrated acceptance to resolve near-ties. Each step includes measurable checkpoints so we can confirm improvements before progressing.
