# Open items: isolated-track ASR

Implementation follow-up, 2026-10-12: the user selected Qwen on this Mac and the
D&D repository's campaign roster/confirmed terms. The ordinary CLI now has an
explicit Qwen backend, an isolated pinned MLX runtime, canonical-vocabulary loading,
and aligned exports. See the [current workflow](../mac-isolated-track-transcription.md).
Local audio smoke tests establish execution, not WER superiority. The human
benchmark and archive rollout remain outstanding.

## Blocking Items

These block execution or a production accuracy claim, not the completed research.

| item_id | blocker_type | severity | due_phase | description | missing_evidence_or_decision | owner | unblock_condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OI-001 | data_gap | blocking | implementation | No independent human stem references inspected | Representative, corrected references | Project | Freeze tuning/test sessions and transcription policy |
| OI-002 | data_gap | high | implementation | Historical exact recognizer/config unknown | Original run manifests or provenance | Implementer | Record historical baseline where available and reproducible current baseline separately |
| OI-003 | regime_mismatch | high | implementation | Qwen sequential MLX inference and vocabulary forwarding now verified; GPU-PB remains untested | Pinned NeMo smoke tests if later selected | Implementer | No custom batch decoder used in Qwen path; assess parity only if batching is introduced |
| OI-004 | data_gap | medium | implementation | Campaign vocabulary resolved; session-specific reduction optional | User selected canonical campaign.json | Project | Reader imports canonical names and exact short forms, excludes untyped misspelling aliases; use a custom bank for narrower scope |

## Non-Blocking Follow-Ups

| item_id | impact | severity | due_phase | description | proposed_next_step |
| --- | --- | --- | --- | --- | --- |
| OI-101 | runtime | medium | implementation | Colab accelerator/memory and archive size unknown | Measure pilot memory/runtime, then estimate archive cost in GPU hours |
| OI-102 | reliability | medium | implementation | Track offsets, silence patterns, bleed, clipping unmeasured | Inspect representative original stems before choosing preprocessing |
| OI-103 | quality | medium | implementation | Boost strengths and negative-name behavior unmeasured | Tune on development; freeze guardrail tolerances before test |
| OI-104 | scope | low | implementation | Local-only preference resolved | User chose local or Colab; exclude paid transcription APIs |

## Discovery strategy and limits

Inspected repository ASR wrappers, CLI routing, recent Qwen/MOSS work, the MLX batch
helper, and prior proxy score output. Searched official Qwen, NVIDIA, IBM, Mistral,
and faster-whisper sources for current checkpoints and vocabulary mechanisms.
Search snippets and community posts served only as discovery leads. Initial cloud
search results were excluded after the user's local/Colab preference. No new model
was installed or executed, no recording was uploaded, and no measured WER gain is
claimed. Scope is a practical shortlist, not an exhaustive leaderboard census.
