# Nemotron 3: overlap improvement decision

## Decision Summary

Investigate Nemotron as an independent overlap detector alongside the current
Qwen/MOSS pipeline. Start with a frozen, unmodified model and a paired evaluation;
then test targeted decoding and only afterward fine-tuning. This report establishes
an experiment design, not a measured improvement or production replacement.

The model's published gains justify a fresh trial (E1–E3). Our earlier NVIDIA
experiments do not test this checkpoint. Its extra speaker capacity fits our
multi-person recordings, but better activity estimates alone cannot restore words
that the recognizer omitted (E2). See the [evidence table](../sources/nemotron_overlap_20260924_evidence_table.md).

## What our repository tells us

The maintained mixed-recording workflow is Qwen wording plus the trained MOSS
pilot, enrollment, and conservative local name correction (L1). The ordinary
pyannote CLI and historical Sortformer experiments are separate paths.

The current overlap router activates only when MOSS already predicts overlapping
named turns. If it misses the quieter voice entirely, the router has no second
turn to retain (E8). That is the most useful place to test an independent detector.

The prior validation found a 64.92% overlap word-recall proxy with retained MOSS
exchanges, falling to 54.03% with naive Qwen attribution. Those are proxy results
on a sample already used for design, not an untouched test or human accuracy (E9).
The MOSS training pilot's overlap activity F1 moved only 0.9196→0.9248 on six
synthetic development clips (L5). Its larger interruption gain is a different result.

Our old Sortformer evaluator forces each word to one speaker. Keep that score for
historical comparison, but measure simultaneous activity independently (E6).

## Variant Comparison

These mechanisms and rankings are engineering hypotheses, not vendor claims.

| rank | approach | expected benefit | main risk | decision |
| --- | --- | --- | --- | --- |
| 1 | Frozen Nemotron plus current naming and ASR | Find missed second voices; improve boundaries | False overlaps or unstable name binding | First experiment |
| 2 | Nemotron-triggered contextual MOSS re-decoding | Recover brief omitted exchanges using existing runtime | Same decoder may repeat the same omission | Follow detector comparison |
| 3 | Fine-tune Nemotron or MOSS on corrected overlap | Adapt to our acoustics and interruption patterns | Label noise, overfitting, clean-speech regression | After zero-shot failure analysis |
| 4 | Activity-conditioned multi-talker ASR or target extraction | Recover both lexical streams | More runtime complexity; separation and identity errors | Conditional on lexical bottleneck |

Reject an immediate replacement of MOSS and blind additional training. Prior
Sortformer adaptation improved one overlap proxy while regressing other slices
(E10). Existing Parakeet/SE-DiCoW experiments are starting points to inspect, not
newly proven solutions. Multiplying a waveform by an activity mask leaves both
voices in simultaneous regions; it is not source separation.

## Original vs Adaptation Resolution

Use official inference as the reference. The existing Mac ASR environment has
Python 3.12.8, Torch 2.14.0 and Transformers 5.17.0, but no Nemotron module or
NeMo installation. Repository Linux dependencies pin NeMo below 3. A separate
environment is required for the documented path (E5). Mac acceleration and speed
remain unmeasured. No package changes or model downloads were made here.

## No-Go Check Result

| condition | result | evidence/action |
| --- | --- | --- |
| Official model/interface support | pass for experiment design | E1–E5 |
| Runtime compatibility | unresolved for execution | Isolated installation and smoke test first |
| Measurable gates | pass | Implementation plan defines paired metrics |
| Reference quality | blocks accuracy claims | Fresh human overlap/word labels required |
| Local superiority | untested | No inference in this investigation |
| Core transcription uncertainty | pass | Native source text and inspected code |

## Locked Defaults

Keep the current release as comparator. Fresh caches and explicit model/runtime
revisions; no production fusion changes during the baseline experiment. Separate
anonymous activity, acoustic name binding, and word recovery in reports. Select
thresholds on development data only. Preserve old human annotations.

## Next action and falsification

Implement the [paired experiment](nemotron_overlap_20260924_implementation_spec.md).
Reject the detector hypothesis if it cannot improve overlap recall at matched
precision, or if downstream names/words regress. Stop diarizer fine-tuning if
reference activity still cannot help the decoder recover missing words: that result
would locate the bottleneck in recognition rather than detection.

Readiness below applies only to constructing the experiment. Runtime execution,
training, and release each retain the gates in [open items](nemotron_overlap_20260924_open_items.md).

status = ready_for_implementation
