# Timeline Test Loop

Use this loop for the new read-only timeline shell.

## Goals
- launch the timeline demo through the new application architecture
- capture repeatable screenshots
- make visual regressions obvious
- avoid relying on manual ad hoc testing every time

## Commands

### Launch interactive demo
Use the repo virtualenv Python:

```powershell
.venv\Scripts\python.exe run_timeline_demo.py
```

### Capture visual snapshots
```powershell
.venv\Scripts\python.exe run_timeline_capture.py
```

Outputs are written to:

```text
artifacts/timeline-demo/
```

> Note: capture outputs are local runtime artifacts and are **not tracked in Git**.

Expected files:
- `timeline_default.png`
- `timeline_stopped.png`
- `timeline_scrolled.png`

## Current Coverage
- read-only timeline shell renders from `TimelinePresentation`
- multiple layer rows
- take summary/caret affordance
- event blocks
- badges
- playhead
- simple viewport variant

## Rules
- Always use the repo virtualenv Python, not the Windows Store alias.
- Prefer screenshot capture before/after UI changes.
- If the shell changes visually, regenerate captures and compare.
- Add new state variants when new presentation behavior becomes important.

## Next Improvements
- add a simple image diff check
- add event selection + mute/solo visual variants
- add an intent-wired smoke test once interaction is connected
