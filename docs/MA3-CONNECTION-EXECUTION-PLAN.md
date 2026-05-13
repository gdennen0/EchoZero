<!--
MA3 connection execution plan for EchoZero public-release readiness.
Exists to convert the council audit into ordered implementation and proof work.
Connects UX, EZ app diagnostics, MA3 Lua protocol, and release validation.
-->

# MA3 Connection Execution Plan

Status: active baseline implemented
Last updated: 2026-05-13

## Scope

Active objective: make MA3 <> EchoZero OSC setup, connection checking, failure diagnosis,
and release proof simple enough for public operators.

Lane: planning/review, then EZ app and MA3 harness execution.

Excluded lanes:

- Foundry
- unrelated playback or pipeline work
- automatic plugin file injection as a public release feature

## Council Summary

### Operator UX

The operator-facing experience needs one primary action: `Run Connection Check`.

That action replaces the current split between `Check Status`, `Ping`, and the MA3 HUD `Test`
button. It should report staged results:

- `Receive Listener`
- `Command Send`
- `MA3 Reply`
- `Plugin Version`
- `Plugin Health`

The user-facing states should be:

- `Not Configured`
- `Bind Failed`
- `Send Failed`
- `Reply Failed`
- `Plugin Missing`
- `Plugin Stale`
- `Connected`
- `Connected With Warnings`

### EZ App Diagnostics

The app should add one shared service in `echozero/application/sync/ma3_connection_check.py`.

Proposed core types:

- `MA3OscConnectionCheckRequest`
- `MA3OscConnectionCheckResult`
- `MA3OscEndpointCheck`
- `MA3OscConnectionState`
- `MA3OscFailureCode`
- `MA3OscConnectionCheckService`

The OSC settings panel and MA3 connection HUD should consume the same result model.

When checking saved/live settings, prefer the active `MA3OSCBridge.ping()` so the app does not
create a second temporary listener that conflicts with the real bridge. Use temporary probes only
for unsaved draft values.

### MA3 Lua / Protocol

The Lua side needs richer truth reporting.

Add a versioned diagnostic envelope through `EZ.ConnectionReport(request_id)` or an enriched
connection status reply. The report should include:

- schema version
- plugin version/build
- target IP/port
- socket/module health
- hook count and hook keys
- last send success/failure
- capabilities
- reasons/warnings

The plugin should centrally track send outcomes in `echozero_osc.lua`.

The hook contract must be corrected before calling the system rock solid. Python currently can mark
hooks active optimistically; Lua can reject a hook path. The bridge should wait for
`subtrack.hooked` or `hooks.error` before considering a track hooked, or Lua should restore
compatible defaults where appropriate.

### Verification / Release

Simulator tests are regression support. Public-release confidence requires real app-path proof with
a real MA3 target.

Release cannot be signed off until:

- automated bridge and app-flow lanes pass
- diagnostics transcript contains target, command list, versions, health, and replies
- real MA3 push, pull, receive, and visible operator flow are proven through the canonical app path
- packaged build and packaged smoke pass

## Target User Experience

### First-Time Setup

1. Operator opens `MA3 OSC Connection`.
2. App shows receive endpoint and MA3 command endpoint fields.
3. Operator selects or enters values.
4. Operator clicks `Run Connection Check`.
5. App shows each stage with pass/fail and exact next action.
6. App shows the callback target sent to MA3.
7. Operator saves only after the result is understandable.

### Reconnect / Failed Push

1. Failed MA3 push opens the same connection surface.
2. App does not use a weaker HUD-only test.
3. App names the failed stage.
4. App offers one immediate recovery action, such as `Pick Free Receive Port`, `Reload MA3 Plugins`,
   or `Copy Diagnostic Report`.

### Diagnostic Report

Every connection check should produce a compact report containing:

- configured receive endpoint
- actual bound receive endpoint
- MA3 command endpoint
- callback target sent to MA3
- commands sent
- replies received
- plugin version/build
- plugin health
- final state
- failure code and recommended fix

## Execution Waves

### Wave 0: Guardrails And Baseline

Purpose: protect the dirty repo and establish test targets before implementation.

Tasks:

- Confirm active changes unrelated to this lane remain untouched.
- Keep plan/docs isolated from unrelated modified files.
- Identify current test entry points that can be run quickly.
- Add or update no production behavior yet.

Acceptance:

- `git status --short` reviewed.
- Plan is captured in this file.
- First implementation slice is bounded to connection diagnostics only.

### Wave 1: Shared EZ Connection Check Service

Purpose: unify proof logic before changing UX copy broadly.

Implementation:

- Add `echozero/application/sync/ma3_connection_check.py`.
- Define typed request/result/stage/failure models.
- Move local receive-bind probe into the service.
- Move send-dispatch probe into the service.
- Move temporary round-trip ping into the service.
- Add optional live-bridge path using `MA3OSCBridge.ping()`.
- Preserve existing behavior for draft settings that differ from the live bridge.

UI integration:

- Update `OscSettingsPanel` to call the shared service.
- Update `MA3ConnectionHUD` to call the shared service.
- Rename `Check Status` to `Run Connection Check`.
- Stop reporting send-only checks as connection success.

Tests:

- Add `tests/application/test_ma3_osc_connection_check_service.py`.
- Cover disabled config.
- Cover invalid send/receive config.
- Cover receive bind failure.
- Cover local send dispatch failure.
- Cover temporary ping success/failure.
- Cover live bridge ping success/failure.

Acceptance:

- Both UI surfaces show the same state vocabulary for equivalent results.
- A live bridge check does not self-conflict with the app's existing receive port.
- Send-only success is labeled as local readiness, not connected.

### Wave 2: Operator Result Card And Recovery Actions

Purpose: make diagnosis obvious without asking operators to interpret raw transport details.

Implementation:

- Add staged display rows for `Receive Listener`, `Command Send`, `MA3 Reply`,
  `Plugin Version`, and `Plugin Health`.
- Show configured and actual endpoints.
- Add port conflict wording for address-in-use errors.
- Add `Copy Diagnostic Report`.
- Add `Pick Free Receive Port` if doing so is low-risk in the settings surface.
- Use the same result rendering in the HUD and settings dialog.

Acceptance:

- Every failure names the failed stage.
- Every failure has one recommended next action.
- The report can be pasted into a bug report without additional context.

### Wave 3: MA3 Lua Diagnostic Report

Purpose: let MA3 explain plugin/socket state instead of forcing EZ to infer everything.

Implementation:

- Add `EZ.ConnectionReport(request_id)` or an additive `connection.report` payload.
- Add central send outcome tracking in `MA3/plugins/echozero_osc.lua`.
- Include plugin and HitMaker health fields already exposed by `EZ.GetPluginHealth()`.
- Include target IP/port and last send details.
- Preserve current `EZ.Ping()`, `EZ.Status()`, `EZ.Version()`, and `EZ.GetPluginHealth()`.
- Update simulator support for the new diagnostic payload.

Tests:

- Extend `tests/testing/test_ma3_osc_bridge.py`.
- Add simulator coverage for `connection.report`.
- Add chunking coverage if the report can exceed one payload.

Acceptance:

- New Python tolerates old Lua with no `ConnectionReport`.
- Old Python ignores new report messages safely.
- Report clearly distinguishes UDP send success from round-trip acknowledgement.

### Wave 4: Hook Truth And Reload Safety

Purpose: remove optimistic sync state that can make the connection appear healthy while live sync is
wrong.

Implementation:

- Fix `HookTrack` / `HookCmdSubTrack` compatibility.
- Make Python wait for `subtrack.hooked` or `hooks.error` before marking a hook active.
- On plugin reload, invalidate cached hooks.
- After `RP`, run `SetTarget`, `Ping`, `Version`, `GetPluginHealth`, `ConnectionReport`,
  then rehook intentionally.

Tests:

- Simulate hook success.
- Simulate hook rejection.
- Simulate dropped or delayed hook reply.
- Simulate plugin reload invalidating hook truth.

Acceptance:

- Python never reports a track hooked unless MA3 confirms it.
- Hook errors surface in diagnostics.
- Reload recovery is explicit and observable.

### Wave 5: Network Hints

Purpose: help operators find likely hardware without making discovery the source of truth.

Implementation:

- Show local IPv4 candidates.
- Show likely same-subnet hints.
- Optionally scan a bounded subnet for responsive hosts if the user explicitly asks.
- Keep `Run Connection Check` as the only connection proof.

Acceptance:

- Discovery output is labeled as hints.
- No release gate depends on discovery alone.

### Wave 6: Hardware And Release Proof

Purpose: close public-release confidence.

Automated gates:

```bash
pytest tests/testing/test_ma3_osc_bridge.py tests/testing/test_ma3_harness_cli.py tests/testing/test_ma3_app_path_smoke.py -q
python -m echozero.testing.run --lane appflow-sync
python -m echozero.testing.run --lane appflow-protocol
python -m echozero.testing.run --lane appflow-all
python scripts/check_repo_hygiene.py
```

Live diagnostics gates:

```bash
python MA3/dev/ma3_plugin_health_check.py --ma3-host <host> --ma3-port <port>
python MA3/dev/ma3_harness_cli.py --json validation-report --ma3-host <host> --ma3-port <port> --output-dir artifacts/ma3-harness/<release-id> --receive-duration-seconds 2 --receive-trigger-command "EZ.Play(1)"
python MA3/dev/ma3_harness_cli.py --json --transcript-out artifacts/ma3-harness/<release-id>/receive-capture-play-transcript.json receive-capture --duration-seconds 2 --trigger-command "EZ.Play(1)"
```

Real app-path gates:

```bash
python MA3/dev/ma3_app_path_smoke.py --json --ma3-host <host> --ma3-port <port>
python MA3/dev/ma3_app_path_push_smoke.py --json --ma3-host <host> --ma3-port <port> --target-track-coord <known-sequenced-track> --cue-number <unused-cue>
```

Packaging gates:

```powershell
powershell -File scripts/build-test-release.ps1
powershell -File scripts/smoke-test-release.ps1
```

Acceptance:

- Evidence states what was real versus simulated.
- Real MA3 target proves connection, browse, pull workspace hydration, bounded push write,
  pull apply/destination behavior, and receive updates through the app boundary.
- Packaged build and smoke pass.

## First Implementation Slice

Start with Wave 1 only.

Files to touch:

- `echozero/application/sync/ma3_connection_check.py`
- `echozero/ui/qt/osc_settings_panel.py`
- `echozero/ui/qt/ma3_connection_hud.py`
- `tests/application/test_ma3_osc_connection_check_service.py`

Optional if runtime injection is needed:

- `echozero/ui/qt/osc_settings_dialog.py`
- `echozero/ui/qt/timeline/widget_action_ma3_push_mixin.py`

Do not touch MA3 Lua in the first slice unless required by an exposed failure.

First-slice completion criteria:

- One shared service owns endpoint validation, bind probe, send probe, and round-trip check.
- Both UI surfaces call that service.
- The old HUD raw `EZ.Ping()` behavior is removed.
- The UI label changes from `Check Status` to `Run Connection Check`.
- Tests prove the service state model.

## Risks And Decisions

### Dirty Worktree

The repo has many pre-existing modified and untracked files. Implementation must avoid unrelated
cleanup and keep diffs scoped to the files named in the active wave.

### UDP Semantics

UDP send success is local dispatch only. It must never be presented as MA3 connected.

### Old Lua Compatibility

EZ must tolerate old MA3 plugins that do not yet implement `ConnectionReport`.

### Plugin Injection

Automatic plugin injection through OSC is not a public-release target. The safe path is:

- detect plugin state
- guide install/reload
- support explicit `RP`
- verify with handshake and health report

## Ready-To-Execute Checklist

- [x] Council audit complete.
- [x] UX, app, Lua/protocol, and verification findings synthesized.
- [x] First implementation slice scoped.
- [x] Wave 1 shared connection-check service implemented.
- [x] Wave 2 operator result rendering and diagnostic report implemented.
- [x] Wave 3 MA3 Lua `EZ.ConnectionReport` and send outcome tracking implemented.
- [x] Wave 4 hook truth hardening implemented.
- [x] Targeted service, UI, simulator, and bridge tests passed.
- [ ] Wave 5 network discovery hints not implemented yet; keep discovery advisory only.
- [ ] Wave 6 live MA3 hardware and packaging proof still required for public-release signoff.
