# Full-session speaker review

This annotation tool preserves earlier transcript snapshots. For the latest Qwen/MOSS
exports, including complete Session 1 and Session 40, use the [current transcript
reader](../../docs/mac-single-file-transcription.md). Old grades must not be attached
to new turns merely because a session number matches. The ports and session choices
below describe the historical annotation library, not the latest release reader.

Open http://127.0.0.1:8767/ and choose Session 2, 3, 15, or 36. Session 1 links to the original reviewer on port 8765, preserving its existing annotations.

With **Needs review only** enabled, playback is phrase-by-phrase: one second before the flagged turn through 0.25 seconds after it, then pause for grading. Correct, Wrong, Unsure, or a speaker assignment immediately plays the next ungraded flagged turn, skipping intervening audio. Choose the replacement speaker directly when you know it, or use **Previous phrase** to revisit a grade. Overlapping phrases keep separate grading targets even when another voice is audible. Turning the switch off restores continuous playback.

Click a timestamp to play with one second of context. Audio highlights all simultaneous turns. Use the time-jump box for hours/minutes/seconds, search across the entire transcript, or filter ungraded/flagged turns. The **Speaker** selector filters by the current assigned speaker, including your saved corrections. Choosing a speaker enables **Needs review only**; **All speakers** removes the speaker restriction. The selection is remembered per session, and both the phrase queue and Next ungraded respect it. Renaming a roster voice preserves its selection.

The **Needs review only** switch beside the session picker shows model-flagged speaker/transcription turns. Its setting and the grade filter are remembered separately for each session in this browser. Combine it with **Ungraded** to focus on flagged turns that still need a grade. Pages contain 80 turns; audio-follow and next-ungraded move between pages automatically.

Mark the original speaker correct, assign a replacement voice, mark unsure, flag words/timing, or type a note. Speaker assignments and notes save automatically. The footer reports whether the save reached disk. Export review downloads the original timed transcript together with corrections and annotations. Keyboard shortcuts: Space play/pause, C correct, W wrong, U unsure, N next ungraded, and 1–6 assign a roster voice. Keyboard shortcuts are suspended while editing a field.

Session 15 retains its provisional four-person roster and displays the attendance note. Session 36 offers all six voices. Names can be renamed without detaching prior assignments.

## Run

From the repository root:

```sh
.venv/bin/python scripts/review_session_library.py
```

This snapshots available completed exports and prepares browser audio in the external cache. It can include additional completed sessions on the next run. To start immediately with the already prepared library:

```sh
.venv/bin/python scripts/review_session_library.py --serve-only
```

The service binds to localhost only. It does not start inference or alter model outputs. The original Session 1 service remains separate; see `tools/speaker_review/README.md`.

Review files live under `~/.cache/transcriber/speaker-review/library/sessionN/TRANSCRIPT_SHA256/review.json`. Each immutable transcript snapshot has its own annotation file. Re-exporting a different transcript cannot silently attach old grades to different turns. An old running reviewer continues to use its original snapshot. Existing annotations are never automatically migrated to a changed export.

Writes are atomic. Multiple tabs use revision checks to prevent silent overwrites. If a save reports a conflict, export a backup before reloading. Browser storage also keeps pending changes, but the footer's disk-save confirmation is authoritative.

## Validation

```sh
.venv/bin/python -m pytest -q tests/test_review_session_library.py tests/test_review_speakers.py
node --check tools/session_review/app.js
```

12 tests pass, covering audio byte ranges, isolated per-session persistence, six-voice assignments, stale-tab rejection, and unchanged source transcripts. Browser checks verified seeking at 03:50:00 in Session 2, playback and active-speaker highlighting, six-voice controls, and an isolated correction/note surviving reload. Two simultaneous fixture turns highlighted together. The original Session 1 annotation file still matches its pre-existing SHA-256.

Phrase-mode browser verification: a test clip paused at 2.81 seconds after its 2.5-second turn end; grading immediately sought to 44.18 seconds for the next flagged turn at 45 seconds, skipping the unflagged gap. The overlapping next turn retained its own selection, and the last grade stopped playback with no ungraded flagged turns remaining. All grades for this verification were confined to an isolated test fixture.
