# Playback Smoke Note

Last updated: 2026-04-18

This is the PB-33 manual smoke handoff note for the current playback remediation slice.

## Automated Proof Already Green

- `./.venv/bin/python -m pytest tests/ui/test_runtime_audio.py tests/ui/test_app_shell_runtime_flow.py tests/testing/test_app_flow_harness.py tests/test_audio_engine.py -q`
- `./.venv/bin/python -m echozero.testing.run --lane gui-lane-b`
- `./.venv/bin/python -m echozero.testing.run --lane appflow`

## Manual Smoke Goal

Verify on the real app path:

1. load song
2. play
3. seek
4. stop
5. switch active playback target
6. confirm audible correctness and smooth playhead behavior

## Operator Checklist

- launch via `run_echozero.py`
- create/open a project
- add one real audio file
- press play on the default song layer
- confirm:
  - audible playback is clean
  - no static/noise burst
  - playhead advances smoothly
- seek to a different time
- confirm:
  - playback resumes from the new position
  - playhead does not jump backward after settling
- stop
- confirm:
  - playhead returns to `0`
  - transport stops cleanly
- create a second playable target:
  - either use stems if available
  - or any second playable layer/take
- set `Active` to the second target without changing selection on purpose
- confirm:
  - audio source changes to the active target
  - selection highlight and active state remain distinct
  - switching active target during or between playback does not produce static
- if stems/event slices are available:
  - set `Active` back and forth at least twice
  - confirm no backward playhead snap during churn

## Record Results Here

- machine / output device:
- sample rate observed by OS/device:
- result:
  - pass / fail
- notes:
- static/noise heard:
  - yes / no
- active-target switching correct:
  - yes / no
- playhead smooth enough:
  - yes / no
- follow-up bug if any:
