# Playback Crackle Elimination Task

Status: active
Last reviewed: 2026-04-30


Task ID: `PB-CR-01`  

This task is the focused execution slice for the audible playback crackle/static bug in the real EZ app path.

It is intentionally narrower than the broader playback remediation board:
- eliminate audible crackle/static during normal EZ playback
- recover enough proof lanes to diagnose playback without unrelated red noise
- add the missing runtime diagnostics needed to identify underrun/device-format faults on real hardware

## Goal

Ship one verified playback path in EZ where:
- default song playback is audibly clean
- seek does not produce crackle/static bursts
- switching the active playback target during or between playback does not produce crackle/static
- event-slice playback and continuous audio playback can both run cleanly through the real app path
- playback diagnostics expose enough information to distinguish device-format problems from callback underruns and track-rebuild churn

## Current Diagnosis

As of 2026-04-29, the repo shows a clear split:

- audio-specific deterministic playback tests are green
- playback-focused app-path subsets are green
- broad automation lanes are not currently usable as crackle proof because they are blocked by unrelated regressions
- no current automated lane proves audible cleanliness on a real output device
- the app/runtime surface does not expose callback underrun telemetry strongly enough to diagnose real-device crackle on its own

Practical conclusion:

- the crackle bug is likely living at the real-device boundary or in playback-churn behavior not modeled by the current fake-stream deterministic tests
- the first engineering move should be observability and proof-lane recovery, not blind tuning

## Execution Update — 2026-04-29

### PB-CR-01A progress

- playback-focused proof lanes are now recovered
- `python -m echozero.testing.run --lane appflow-playback` passed
  - `19 passed`
- `python -m echozero.testing.run --lane ui-automation-playback` passed
  - `1 passed`
- `python -m echozero.testing.run --lane humanflow-playback` passed
  - `21 passed`
- the broader canonical `appflow` lane also passed again after launcher-menu expectation drift was updated
  - `23 passed`

### PB-CR-01B progress

- playback diagnostics now surface through `PlaybackState.diagnostics`
- app-visible/runtime-visible fields now include:
  - `glitch_count`
  - `last_audio_status`
  - `output_device`
  - `stream_latency`
  - `stream_blocksize`
  - `prime_output_buffers_using_stream_callback`
  - `last_transition`
  - `last_track_sync_reason`
- the UI automation backend now includes those playback diagnostics in snapshot artifacts

### Real-device smoke collected

Machine/device observed on 2026-04-29:

- default output device: `MacBook Pro Speakers`
- default output format: `48000 Hz`, `2` output channels

Real EZ runtime smoke via `build_app_shell(...)`:

- default runtime config (`stream_latency=high`, auto blocksize)
  - play clean
  - seek clean
  - `glitch_count=0`
  - `last_audio_status=None`
- aggressive runtime config (`stream_latency=low`, `stream_blocksize=256`)
  - play clean
  - seek clean
  - `glitch_count=0`
  - `last_audio_status=None`

Real-device target-switch stress via live `PlaybackController` on the default output device:

- repeated active playback target switches between continuous audio and event-slice playback while the stream stayed live
- `glitch_count=0`
- `last_audio_status=None`

### Remaining gap

- the full canonical `ui-automation` lane still appears operationally unhealthy in this environment
- it progressed through `tests/ui_automation/test_session.py` and the first `test_echozero_backend.py` case, then stopped making forward progress within the diagnostic window
- playback-specific UI automation proof is green, but the full lane still needs separate operational follow-up

## Evidence Collected

### Audio-specific deterministic proof

Command:

```bash
./.venv/bin/python -m pytest tests/test_audio_engine.py tests/ui/test_runtime_audio.py tests/testing/test_playback_capture.py -q
```

Result:

- pass
- `163 passed`

Interpretation:

- core engine, runtime controller, and simulated playback capture are stable under deterministic test conditions
- this does not prove real hardware audio cleanliness

### Playback-focused app-shell proof

Command:

```bash
./.venv/bin/python -m pytest tests/ui/test_app_shell_runtime_flow.py -q -k "audio or playback"
```

Result:

- pass
- `15 passed`

Interpretation:

- the app shell playback contract is green in targeted automated slices

### Playback-focused harness proof

Command:

```bash
./.venv/bin/python -m pytest tests/testing/test_app_flow_harness.py -q -k "playback"
```

Result:

- pass
- `4 passed`

Interpretation:

- the harness-based playback flows are green in the playback-only subset

### Broad appflow interface

Command:

```bash
./.venv/bin/python -m echozero.testing.run --lane appflow
```

Result:

- blocked by unrelated failure
- `tests/testing/test_app_flow_harness.py::test_app_flow_harness_exposes_launcher_menus`
- failure: extra launcher menu item `Create Project S&nare-Only Model`

Interpretation:

- this lane is not currently usable as crackle proof until the unrelated launcher-menu drift is fixed or the playback-specific subset is broken out as its own lane

### UI automation interface

Command:

```bash
./.venv/bin/python -m echozero.testing.run --lane ui-automation
```

Result:

- blocked by unrelated failure
- `tests/ui_automation/test_echozero_backend.py::test_echozero_backend_imports_song_and_exposes_timeline_targets`
- failure: expected label `Automation Song`, got `automation-import`

Interpretation:

- the canonical automation interface is not currently usable as playback-crackle proof until the unrelated import-label drift is fixed or the playback-specific subset is separated

### Humanflow composite interface

Command:

```bash
./.venv/bin/python -m echozero.testing.run --lane humanflow-all
```

Result:

- blocked by unrelated failures inherited from:
  - `test_app_flow_harness_exposes_launcher_menus`
  - `test_echozero_backend_imports_song_and_exposes_timeline_targets`

Interpretation:

- this lane is not currently usable as playback-crackle proof

### Simulated GUI regression interface

Commands:

```bash
./.venv/bin/python -m echozero.testing.run --lane gui-lane-b
./.venv/bin/python -m pytest tests/testing/test_gui_lane_b.py -q
```

Result:

- stalled/hung in this environment beyond a short diagnostic window
- no useful playback verdict captured

Interpretation:

- this interface currently has operational risk and should be treated as unstable until its runtime behavior is understood

### Existing manual smoke expectations

Relevant note:

- [docs/PLAYBACK-SMOKE-NOTE-2026-04-18.md](archive/PLAYBACK-SMOKE-NOTE-2026-04-18.md)

This note already names the operator checks that matter:

- no static/noise burst
- clean playback after seek
- no static when switching active playback target

But it is still a manual handoff note, not an executable proof lane.

## Diagnosis Gaps

### Missing runtime observability

The real app/runtime surface currently exposes:

- playback output sample rate
- playback output channel count

It does not clearly expose:

- `AudioEngine.glitch_count`
- `AudioEngine.last_audio_status`
- per-session callback underrun history
- current stream latency mode / blocksize at the app-facing diagnostic layer
- whether crackle occurred during:
  - initial play
  - seek
  - active-target switch
  - preview playback

Without that telemetry, the real app can audibly fail while the automation layer still says only “playback happened.”

### Missing real-device proof

The automated green paths are fake-stream or simulated-proof paths.

They do not prove:

- hardware output-device negotiation is correct
- `sounddevice` callback timing is clean on the current machine
- real active-target churn is free of static/crackle on actual output hardware

## Likely Fault Domains

Ranked by current evidence:

1. Real-device stream configuration mismatch
   - `sample_rate`
   - channel count
   - `blocksize`
   - `latency`
   - device-default negotiation

2. Callback underrun during playback churn
   - play
   - seek
   - active-target switch
   - event-slice rebuilds

3. Track replacement during active playback causing discontinuity
   - runtime rebuilds may be clean in deterministic tests but still click/crackle on real hardware timing edges

4. Event-slice plus continuous-audio mixed playback producing callback pressure not seen in isolated tests

5. Playback proof-lane contamination
   - unrelated lane failures are currently masking whether broad app automation would catch playback regressions

## Task Steps

### PB-CR-01A: Recover usable playback proof lanes

- Fix or isolate unrelated failures blocking playback diagnosis:
  - launcher menu drift in `tests/testing/test_app_flow_harness.py`
  - import-label drift in `tests/ui_automation/test_echozero_backend.py`
- If those fixes are intentionally out of scope, add dedicated playback-only lanes under `echozero/testing/run.py` so playback proof is not blocked by unrelated UI regressions.

Done when:

- there is one canonical playback-focused `appflow` lane
- there is one canonical playback-focused `ui-automation` lane

### PB-CR-01B: Add app-visible playback diagnostics

- Surface the following from the real runtime audio path into a diagnostic contract:
  - glitch count
  - last callback status
  - resolved output device id/name
  - resolved sample rate
  - resolved output channel count
  - resolved stream latency mode/value
  - resolved stream blocksize
  - last track-rebuild reason
  - whether the last transition was play, seek, target-switch, or preview

Suggested files:

- `echozero/audio/engine.py`
- `echozero/application/playback/runtime.py`
- `echozero/application/playback/models.py`
- `echozero/ui/qt/app_shell_runtime_services.py`
- `echozero/ui/qt/app_shell_runtime_support.py`

Done when:

- the app/runtime layer can report whether a crackle run coincided with an actual callback glitch or only with a semantic playback transition

### PB-CR-01C: Add playback stress probes that target crackle triggers

- Add automated stress tests for:
  - repeated play/stop churn
  - repeated seek churn during playback
  - active playback-target switching during playback
  - event-slice plus continuous-audio co-playback with route changes
  - playback under multiple output config combinations

At minimum, add proof slices around:

- `tests/ui/test_app_shell_runtime_flow.py`
- `tests/testing/test_app_flow_harness.py`
- `tests/ui/test_runtime_audio.py`
- `tests/test_audio_engine.py`

Done when:

- the repo has automated probes covering the state transitions most likely to cause audible crackle

### PB-CR-01D: Add a real-device playback smoke workflow with recorded diagnostics

- Use the real EZ launcher/app path from [docs/PLAYBACK-SMOKE-NOTE-2026-04-18.md](archive/PLAYBACK-SMOKE-NOTE-2026-04-18.md)
- Record:
  - machine
  - output device
  - sample rate
  - output channels
  - glitch count before/after each operator action
  - last callback status if non-null

Operator actions:

- load song
- play
- seek
- stop
- switch active playback target
- repeat target switching while playing if multiple playable targets exist

Done when:

- one real-device run produces a reproducible crackle/no-crackle matrix with telemetry attached

### PB-CR-01E: Fix the root cause

- After PB-CR-01B through PB-CR-01D establish a concrete repro, fix the actual cause instead of guessing.

Candidate fix areas:

- `echozero/audio/sounddevice_backend.py`
- `echozero/audio/engine.py`
- `echozero/application/playback/runtime.py`
- runtime audio rebuild timing in app-shell integration

Done when:

- the same repro path no longer produces audible crackle/static
- diagnostics no longer show unexpected callback glitch growth on the repaired path

### PB-CR-01F: Final signoff

Required proof:

```bash
./.venv/bin/python -m pytest tests/test_audio_engine.py tests/ui/test_runtime_audio.py tests/testing/test_playback_capture.py -q
./.venv/bin/python -m pytest tests/ui/test_app_shell_runtime_flow.py -q -k "audio or playback"
./.venv/bin/python -m pytest tests/testing/test_app_flow_harness.py -q -k "playback"
./.venv/bin/python -m echozero.testing.run --lane appflow
./.venv/bin/python -m echozero.testing.run --lane ui-automation
./.venv/bin/python -m echozero.testing.run --lane humanflow-all
./.venv/bin/python -m echozero.testing.run --lane gui-lane-b
```

And:

- one documented real-device smoke pass through the launcher/app path

Done when:

- all playback-specific automated proof is green
- broad interfaces are either green or explicitly shown irrelevant
- one real-device smoke run confirms no audible crackle/static

## Done Bar

This task is done only when all are true:

- EZ playback is audibly clean on the real app path
- seek is clean
- active-target switching is clean
- callback/stream diagnostics are visible enough to explain failures
- no broad playback proof lane is blocked by unrelated red noise

## Not Done Yet

This task is not done merely because:

- deterministic fake-stream tests pass
- playback starts at all
- the playhead moves
- simulated capture artifacts look correct
- one unrelated lane happens to be green
