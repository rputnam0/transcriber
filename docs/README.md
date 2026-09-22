# Documentation

## Current workflows

- [Repository README](../README.md): regular CLI, isolated speaker audio, multitrack
  inputs, pyannote speaker bank, configuration, and watch mode.
- [Mac single-file transcription](mac-single-file-transcription.md): install,
  restore private voices, run/resume a complete mixed recording, and listen to named
  transcripts. Includes the completed release and quality limitations.
- [ASR decision report](specs/mac_asr_quality_20260921_decision_report.md),
  [implementation](specs/mac_asr_quality_20260921_implementation_spec.md), and
  [remaining work](specs/mac_asr_quality_20260921_open_items.md): evidence behind the
  selected Qwen/MOSS workflow and unresolved evaluation/backup work.

## Historical experiments and annotations

The other documents under `specs/` and `sources/`, along with `PR-*.md`, retain
research decisions and experiments at the time they were written. Proposed or
experimental configurations there are not the current release defaults. The Mac
guide above is the operational entry point for mixed-recording transcription.

The [single-session reviewer](../tools/speaker_review/README.md) and
[session-library reviewer](../tools/session_review/README.md) preserve older
predictions and human annotations. They are separate from the latest text/audio
reader. Transcript hashes bind annotations to their original exports.

References to `docs/analysis/`, private manifests, recordings, and checkpoints are
local evidence, intentionally excluded from the public repository. A clean clone
contains the code and tests but needs separately restored private assets to identify
the enrolled voices. The Drive checkpoint backup is still incomplete; preserve the
verified local copy until a complete independent backup is verified.

## Archived WSL development

The old WSL snapshots are preserved as Git tags rather than active branches:

- `archive/wsl-speaker-bank-matching`: speaker-bank development snapshot.
- `archive/wsl-transcriber-gpu-wip-20260319`: WSL GPU work-in-progress snapshot.

These snapshots contain unique historical work and are not merged release defaults.
Inspect them with `git show TAG` or use a separate worktree for recovery; `main`
contains the maintained workflow.
