# MA3 Hardware Validation — 2026-05-03

Status: reference
Last reviewed: 2026-05-03


## Status And Authority

This is a validation note for the current localhost MA3 harness lane.
It is not a new canonical architecture spec.

If anything here conflicts with canonical repo docs, the canonical docs win:

- `docs/CANONICAL-EXECUTION-PLAN.md`
- `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
- `docs/STATUS.md`
- `docs/TESTING.md`
- `AGENTS.md`

## Purpose

Capture the current real-MA3 evidence for the EchoZero harness against the
local MA3 target at `127.0.0.1:8001`.

This note closes a specific gap in Wave 2:

- command/reply/identity proof should exist as durable artifacts
- browse proof should exist as durable artifacts
- receive-path observation should be explicit, both while idle and during a
  triggered transport send
- canonical app-boundary pull-workspace hydration should be proven against the
  current localhost target

This note does **not** claim that push, pull, or active receive behavior are
fully validated in the current localhost session.

## Live Target

- host: `127.0.0.1`
- port: `8001`
- command path: `/cmd`
- reply path: `/ez/message`
- app settings path:
  `/Users/march/Documents/GitHub/EchoZero/config/app-settings.json`

## Commands Run

### Targeted Harness Proof

- `./.venv/bin/python -m pytest tests/testing/test_ma3_harness_cli.py -q`

Result:

- passed

Coverage:

- unified CLI ping
- smoke bundle emission
- health-check local-vs-live marker comparison logic
- validation-report artifact emission
- receive-capture transcript logic with simulated inbound transport

### Shared Protocol Proof

- `./.venv/bin/python -m echozero.testing.run --lane appflow-protocol`

Result:

- passed

Coverage:

- MA3 bridge command/reply behavior
- receive/protocol handling
- sync adapter guardrail coverage

### Live Localhost Validation Report

- `./.venv/bin/python MA3/dev/ma3_harness_cli.py --json --ma3-host 127.0.0.1 --ma3-port 8001 validation-report --output-dir artifacts/ma3-harness/live-localhost`

Result:

- passed

Artifacts:

- [artifacts/ma3-harness/live-localhost/summary.json](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost/summary.json:1>)
- [artifacts/ma3-harness/live-localhost/summary.md](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost/summary.md:1>)
- [artifacts/ma3-harness/live-localhost/transcript.json](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost/transcript.json:1>)

### Live Localhost Bundled Validation Report With Triggered Receive

- `./.venv/bin/python MA3/dev/ma3_harness_cli.py --json --ma3-host 127.0.0.1 --ma3-port 8001 validation-report --output-dir artifacts/ma3-harness/live-localhost-bundled --receive-duration-seconds 1 --receive-trigger-command "EZ.Play(1)"`

Result:

- passed

Artifacts:

- [artifacts/ma3-harness/live-localhost-bundled/summary.json](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost-bundled/summary.json:1>)
- [artifacts/ma3-harness/live-localhost-bundled/summary.md](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost-bundled/summary.md:1>)
- [artifacts/ma3-harness/live-localhost-bundled/transcript.json](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost-bundled/transcript.json:1>)

### Canonical App-Boundary Pull Workspace Smoke

- `./.venv/bin/python MA3/dev/ma3_app_path_smoke.py --json --ma3-host 127.0.0.1 --ma3-port 8001`

Result:

- passed

### Live Localhost Receive Idle Capture

- `./.venv/bin/python MA3/dev/ma3_harness_cli.py --json --ma3-host 127.0.0.1 --ma3-port 8001 --transcript-out artifacts/ma3-harness/live-localhost/receive-capture-idle-transcript.json receive-capture --duration-seconds 1 --ping-first`

Result:

- passed

Artifact:

- [artifacts/ma3-harness/live-localhost/receive-capture-idle-transcript.json](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost/receive-capture-idle-transcript.json:1>)

### Live Localhost Receive Triggered Capture

- `./.venv/bin/python MA3/dev/ma3_harness_cli.py --json --ma3-host 127.0.0.1 --ma3-port 8001 --transcript-out artifacts/ma3-harness/live-localhost/receive-capture-play-transcript.json receive-capture --duration-seconds 1 --trigger-command "EZ.Play(1)"`

Result:

- passed

Artifact:

- [artifacts/ma3-harness/live-localhost/receive-capture-play-transcript.json](</Users/march/Documents/GitHub/EchoZero/artifacts/ma3-harness/live-localhost/receive-capture-play-transcript.json:1>)

## Observed Live Results

### Command And Reply Plane

Confirmed:

- `EZ.Ping()` returns `connection.ping`
- `EZ.Version()` returns structured `plugin.version`
- `EZ.GetPluginHealth()` returns structured plugin health

Observed live identity:

- `ez_version = 2.0`
- `ez_build = 2026-05-01.transport-send-debug-1`
- `hitmaker_version = 1.1.0`
- `hitmaker_build = 2026-04-30.hitmaker-x-hit-release-5`

### Local-vs-Live Identity Comparison

Confirmed:

- the MA3-reported plugin markers match the current local live plugin files
  under `/Users/march/MALightingTechnology/gma3_library/datapools/plugins`

This closes the earlier ambiguity where MA3 feedback showed a working version
string but the structured OSC `plugin.version` reply path was missing from the
loaded plugin build.

### Browse Proof

The live validation bundle confirms:

- timecodes can be listed
- current-song sequence range can be resolved
- sequences can be listed
- track groups can be listed
- tracks can be listed

Observed summary from the captured report:

- timecodes: `31`
- sequences: `788`
- current-song range present: `yes`
- first timecode number: `1`
- first timecode track-group count: `1`
- first track-group track count: `1`

### Idle Receive Proof

The live idle receive capture confirms:

- the listener can be armed against the real localhost MA3 target
- the command/reply plane can be verified first via `--ping-first`
- no unsolicited inbound transport traffic arrived during a `1.0` second idle
  capture window

Observed idle receive result:

- `message_count = 0`
- `transport_update_count = 0`

This should be treated as a valid observation, not a failure:

- MA3 did not emit transport updates while idle
- active receive validation therefore requires a real state change or explicit
  transport command during the capture window

### Triggered Receive Proof

The live triggered receive capture confirms:

- the harness can arm the real localhost listener
- the harness can fire a real MA3 transport-send function after the listener is
  ready
- a real transport update arrives back on `/ez/message`

Observed triggered receive result:

- `trigger_command = "EZ.Play(1)"`
- `message_count = 1`
- `message_keys = ["transport.play"]`
- `transport_update_count = 1`
- `latest_transport_update.state = "play"`
- `latest_transport_update.is_playing = true`
- `latest_transport_update.tc = 1`

### Canonical App-Boundary Pull Workspace Proof

The live app-path smoke confirms:

- the canonical Qt runtime can enable MA3 sync against the localhost target
- `OpenPullFromMA3Dialog()` hydrates the manual pull workspace through the real
  app boundary
- the workspace resolves real MA3 browse state, not simulator fixtures

Observed result:

- `workspace_active = true`
- `selected_timecode_no = 1`
- `timecode_count = 31`
- `track_count = 1`
- `source_track_count = 1`
- `available_target_count = 3`

## What This Validation Covers

Covered now:

- real localhost transport path
- real command/reply path
- structured version/health replies
- local-vs-live plugin identity match
- browse/read proof on the live target
- explicit idle receive observation
- explicit triggered receive capture via a real MA3 transport-send command
- one bundled validation-report path that captures identity, browse, and
  triggered receive evidence together
- one non-destructive canonical app-boundary pull-workspace smoke against the
  live localhost target

Not covered yet by this localhost slice:

- push behavior from the canonical app path
- manual pull apply/destination behavior from the canonical app path
- one visible operator end-to-end proof run

## Remaining Wave 2 Gaps

The next required evidence is narrower now:

1. Re-run push validation through the canonical app path against the current
   localhost target.
2. Re-run manual pull apply/destination validation through the canonical app
   path.
3. Roll those results into one operator-facing Wave 2 validation checkpoint.

## Repo Changes Supporting This Slice

Files changed to support this validation pass:

- `MA3/dev/ma3_harness_cli.py`
- `MA3/dev/ma3_harness_common.py`
- `MA3/dev/ma3_plugin_health_check.py`
- `MA3/dev/ma3_reload_plugins.py`
- `MA3/README.md`
- `echozero/infrastructure/sync/ma3_osc.py`
- `echozero/testing/ma3/simulator.py`
- `tests/testing/test_ma3_harness_cli.py`
- `tests/testing/test_ma3_osc_bridge.py`

## Residual Risk

- The current localhost proof shows that the harness is real and the loaded MA3
  plugin identity is known.
- It does **not** yet prove the full Wave 2 scope, because push and pull-apply
  still need a fresh real-app-path capture in the current environment.
