# Decision report

## Decision Summary
Compare Qwen3-ASR (working default), Whisper large-v3 and Granite on identical local clips before full transcription. Keep the user-requested 20-step MOSS checkpoint as the fresh diarization/overlap pass, then attach named speakers to the selected ASR output with acoustic and timing evidence. Default is falsified if another candidate gives better retained lexical content and fewer omissions/repetitions on the same proxy/reference and early-session comparisons.

## Variant Comparison
1. Modern ASR plus new MOSS attribution: separates word quality from learned voice identity; alignment/overlap integration needs explicit checks (C1,C2,C7).
2. Whisper large-v3 plus attribution: established timestamp support, but overlap/hallucination risk (C3).
3. Granite plus attribution: compact English candidate with MLX support (C4,C5).
4. Routed Mega-ASR: conditional option if severe noise is a measured failure; avoid assuming all game audio is degraded (C6).

## Original vs Adaptation Resolution
Use maintained MLX adaptations for Apple GPU execution and retain exact model revisions. Do not transfer CUDA benchmark speed to this Mac.

## No-Go Check Result

| condition | status | evidence |
| --- | --- | --- |
| Missing primary or official API evidence | pass | C1–C7 |
| Unit/platform mismatch | pass | Explicit 16 kHz mono seconds and MLX on Apple Silicon |
| Missing measurable acceptance gates | pass | Complete coverage, artifact checks, paired clip comparison, timestamp and roster checks |
| Unsupported gold-accuracy claim | pass | Proxy labels explicitly separated from human truth |
| Transcription-uncertain source claim | pass | Sources are native text |

## Locked Defaults
Fresh output caches; no modification of old annotations; greedy English decoding; no generative editorial rewriting; source SHA and model identity checks. User-specified diarizer is interruption_v1_pilot_20260921. Final pipeline choice will be recorded after the local comparison.


## Local comparison and implementation outcome

All five dedicated candidates were executed locally, including the official Granite Speech 5.0 TurboCTC weights through the installed MLX implementation. The selected dedicated ASR is Qwen3-ASR-1.7B bf16: normalized development proxy WER 21.98%, versus Whisper large-v3 22.35%, Granite 5 22.53%, and the old MOSS baseline 24.65%. The small lead does not establish universal state of the art. Routed Mega-ASR tied Qwen on development and failed a repetition check on early audio. Granite 4 repeated text and required a documented pointwise-convolution layout repair in the adapter.

The production candidate combines Qwen wording on ordinary speech with the explicitly requested 20-step MOSS checkpoint for fresh voice attribution and complete overlapping exchanges. Qwen's alternate wording is retained for those exchanges. A direct, flat ASR-to-speaker handoff failed the overlap validation and was rejected. The resulting validation preserves the pilot's overlap recall, but does not establish lower aggregate end-to-end WER: the conservative hybrid proxy WER was 34.60%, versus 33.21% for the old baseline. Publish the rerun as a reviewable comparative revision and retain the previous version.

Detailed evidence, caveats, exact commands, model/environment provenance, and the release coverage audit are linked from [the local report](../analysis/asr_quality_20260921/README.md). No human “unsure” grade is a training target. The full Session 1 source remains an input dependency.

status = ready_for_implementation
