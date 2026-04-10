# Phase 3 LD-01 Fixture Freeze

_Date: 2026-04-10_

This document freezes the fixture references for `LD-01` so reruns use the same real-data and MA3 replay inputs.

## Canonical fixture references

- Real-data audio fixture used by the current demo entrypoints:
  - `C:\Users\griff\Desktop\Doechii_NissanAltima_117bpm_SPMTE_v02 [chan 1].wav`
  - This is the default `--audio` path in both `run_timeline_real_data_capture.py` and `run_timeline_real_data_playback.py`.
- Canonical MA3 replay fixture for the sync/reconnect proof path:
  - `tests/fixtures/ma3/reconnect_replay_v1.json`
  - This is the payload replayed by `tests/unit/test_ma3_fixture_replay.py`.

## Working-root conventions for reruns

- Evidence packs live outside tracked source under:
  - `C:\Users\griff\.openclaw\workspace\tmp\phase3-daw-proof\<date>-ld-01\`
- The evidence-pack helper uses these sub-paths for repeatable reruns:
  - raw timeline capture output:
    - `C:\Users\griff\.openclaw\workspace\tmp\phase3-daw-proof\<date>-ld-01\raw\timeline-real-data\`
  - transient working root for the real-data import/analyze flow:
    - `C:\Users\griff\.openclaw\workspace\tmp\phase3-daw-proof\<date>-ld-01\work\real-data\`
- Reuse the same `<date>-ld-01` folder for one operator run. Start a new dated folder for a new proof attempt rather than mixing artifacts across runs.
- Do not redirect Phase 3 outputs back into tracked `artifacts/` or other repo paths.

## Current bootstrap evidence source

- The first evidence-pack bootstrap reuses the existing object-info walkthrough source captured outside Git at:
  - `C:\Users\griff\.openclaw\workspace\tmp\object-info-demo`
- This is an operational bootstrap source, not a replacement for the full clean `LD-01` proof run required by Phase 3 exit.
