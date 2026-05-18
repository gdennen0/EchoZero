# MA3 Transport Handoff - 2026-05-18

Status: reference
Last reviewed: 2026-05-18

## What Changed

- Added app-level external transport normalization in `echozero/application/transport/external.py`.
- Added typed external transport command models in `echozero/application/transport/models.py`.
- Added `TimelineApplication.apply_external_transport_update(...)` so MA3/OSC transport controls apply through canonical timeline intents.
- Moved MA3 section jump behavior out of widget-only runtime logic; jumps now resolve from EchoZero section cues at application time.
- Preserved playing state for seek/move/jump commands; only explicit play/pause/stop/toggle commands alter playing state.
- Fixed MA3 OSC bridge transport consumption so edge/button commands are queued FIFO and scrub/status telemetry can still coalesce latest-only.
- Removed unused legacy Lua helpers that scanned MA3 timecode tracks/events for section seek targets.

## Transport Contract

- Supported external command actions: `play`, `pause`, `stop`, `toggle`/`play_pause`, `seek`/`scrubbed`, `move`, `jump_previous_section`, `jump_next_section`.
- EchoZero timeline/transport is the source of truth.
- Section jumps use the active EchoZero timeline section cues, not MA3 selected cue or timecode ordering.
- `pause` is idempotent. Only `toggle`, `play_pause`, or `toggle_play_pause` toggle.
- `stop` ignores stale playhead values in incoming status payloads.
- `seek`, `move`, and section jumps clamp to the active EchoZero timeline range.

## Tests And Harness Proof

- `.venv/bin/python -m pytest tests/application/test_external_transport_application.py tests/testing/test_ma3_osc_bridge.py tests/ui/test_widget_runtime_transport_sync.py tests/ui/runtime_audio_widget_cases.py -q`
  - Result: `97 passed in 30.57s`.
- `.venv/bin/python -m pytest tests/testing/test_ma3_harness_cli.py::test_ma3_harness_cli_receive_capture_records_inbound_transport tests/testing/test_ma3_harness_cli.py::test_ma3_harness_cli_receive_capture_can_trigger_command -q`
  - Result: `2 passed in 2.37s`.
- `.venv/bin/python -m echozero.testing.run --lane appflow-sync`
  - Result: `4 passed, 10 deselected in 20.90s`.
- `.venv/bin/python -m echozero.testing.run --lane appflow-osc`
  - Result: `64 passed, 12 deselected in 38.49s`.
- `.venv/bin/python -m echozero.testing.run --lane appflow-protocol`
  - Result: `128 passed in 43.73s`.
- `.venv/bin/python -m compileall -q ...changed Python paths...`
  - Result: pass.
- `git diff --check`
  - Result: pass.
- `.venv/bin/python -m ruff check ...changed Python paths...`
  - Result: blocked because `ruff` is not installed in this venv.

## Live MA3 Session Proof

- Target from `config/app-settings.json`: `192.168.1.164:8001`, command path `/cmd`.
- `.venv/bin/python MA3/dev/ma3_harness_cli.py --timeout 3 --json health-check`
  - Result: pass.
  - Live plugin reported `ez_build=2026-05-01.transport-send-debug-1`, `ez_version=2.0`, HitMaker loaded.
- `.venv/bin/python MA3/dev/ma3_harness_cli.py --timeout 2 --json receive-capture --duration-seconds 0.5 --trigger-command 'EZ.JumpToNextSection()'`
  - Result: pass.
  - Captured `transport.jump_next_section` with `action=jump_next_section`, `direction=next`, `source=ez_sections`.
- `.venv/bin/python MA3/dev/ma3_harness_cli.py --timeout 5 --json smoke`
  - Result: blocked.
  - Blocker: live MA3 timed out waiting for `current-song sequence range`.

## Real Project / Showtime Status

- No tracked `.ez` real-world project fixture is present in git.
- Local untracked user projects were preserved and not committed:
  - `NKTest2.ez`
  - `Visual_Updates.ez`
- Best existing repo-local real sample found: `artifacts/demo-suite/20260419-091855/canonical_app_lifecycle/real-app-demo.ez`.
  - This is under generated artifacts, not tracked source truth.
- App/sync proof was run through canonical appflow lanes listed above; live MA3 receive proof was run against the configured MA3 target.

## MA3 Lua Plugin Pool Items To Grab

- Grab/update MA3 plugin pool item: `Ez#2`.
- Required payload item for this transport cleanup: `EZ/ez_timecode.lua`.
- Source file changed in this repo: `MA3/plugins/timecode.lua`.
- Copy-ready package was generated at:
  - `artifacts/ma3-harness-transfer/echozero-ma3-harness-transfer.zip`
- From that package, copy into the MA3 plugin root:
  - `grandMA3/datapools/plugins/Ez#2.xml`
  - `grandMA3/datapools/plugins/EZ/`
- No `TC22`, `HitMaker`, or other plugin pool item changes are required for this transport work.
- After copying, reload MA3 plugins with:
  - `RP`

## Residual Risk / Blockers

- Full live MA3 smoke remains blocked by current-song sequence range timeout on the connected show/session.
- No tracked real-world `.ez` project fixture was available; untracked user `.ez` files were intentionally preserved.
- Ruff could not be run because this venv does not have `ruff` installed.
