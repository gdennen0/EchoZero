# Playback Remediation Execution Plan

Last updated: 2026-04-18

This document explains how to execute the playback remediation task board efficiently with bounded parallel work.

Primary source:
- [docs/PLAYBACK-REMEDIATION-TASK-BOARD.md](/Users/march/Documents/GitHub/EchoZero/docs/PLAYBACK-REMEDIATION-TASK-BOARD.md:1)

## Strategy

Use contract-first sequencing with bounded parallelism.

Why:
- playback target vs selection is the root contract flaw
- backend evaluation is important, but the UI/app contract must be fixed first or backend work will anchor to the wrong model
- UI work can begin in parallel only after the new contract shape is explicit

## Immediate Implementation Order

Execute the remaining remediation in this order:

1. finish the explicit playback-target migration
2. remove the remaining `monitored_*` compatibility layer
3. quarantine simulated playback capture/demo helpers from human-path claims
4. rerun app-shell, runtime-audio, widget, and harness proof
5. audit real playback on the real app path for static/noise and device-format errors
6. decide whether to stabilize the current backend or place a temporary backend behind a playback service boundary
7. redesign playhead timing around backend-owned timing snapshots
8. rebuild the playback demo and non-deterministic playback tests on human paths only

## Current Execution Slice

The active slice is:

- replace `monitored_layer_id` / `monitored_take_id` reads and writes with `active_playback_layer_id` / `active_playback_take_id`
- keep `RouteAudioToMaster` only as a compatibility intent alias where needed
- remove any remaining selection-driven playback reroute behavior
- update app-shell refresh preservation to preserve explicit playback target, not monitored fallback
- update targeted proof so it encodes the new contract directly

This slice must complete before backend/device debugging because backend work should not depend on an obsolete selection-coupled playback model.

## Waves

### Wave 0: Coordination

Goal:
- freeze invalid demo patterns
- establish execution lanes

Tasks:
- `PB-00`
- `PB-02`
- `PB-03`

Owner:
- `lead-dev` with optional `review` sidecar

Output:
- invalid proof paths identified and quarantined
- agent ownership declared

### Wave 1: Contract Split

Goal:
- make playback target first-class and independent from selection

Tasks:
- `PB-10`
- `PB-11`
- `PB-12`
- `PB-13`
- `PB-14`

Owner split:
- Worker A:
  - domain/application contract
  - models, intents, orchestrator, assembler
- Worker B:
  - app-shell preservation and refresh behavior
- Verify sidecar:
  - targeted proof slices

Critical rule:
- UI work does not finalize until Wave 1 contract lands

### Wave 2: UI Separation

Goal:
- reflect the new contract in header/object-info/widget behavior

Tasks:
- `PB-40`
- `PB-41`
- `PB-42`
- `PB-43`

Owner split:
- Worker C:
  - header and widget interaction
- Worker D:
  - inspector/object-info contract and related tests
- Verify sidecar:
  - widget/appflow proof

Dependency:
- requires Wave 1 state shape to be explicit

### Wave 3: Backend Decision Gate

Goal:
- choose the near-term playback backend direction

Tasks:
- `PB-20`

Owner:
- Research panel
- `lead-dev` decision

Output:
- decision memo
- recommendation to stabilize current engine or temporarily replace it

### Wave 4: Playback Service Boundary

Goal:
- move playback semantics behind a service contract

Tasks:
- `PB-30`
- `PB-31`
- `PB-32`
- `PB-33`
- `PB-34`

Owner split:
- Worker E:
  - playback service contract and integration
- Worker F:
  - backend/device-format audit and fixes
- Verify sidecar:
  - real playback smoke path
- Review sidecar:
  - contract leakage audit

Dependency:
- backend direction chosen in Wave 3

### Wave 5: Clock and Playhead

Goal:
- replace timer-truth behavior with presentation-clock rendering over authoritative transport snapshots

Tasks:
- `PB-50`
- `PB-51`
- `PB-52`
- `PB-53`
- `PB-54`

Owner split:
- Worker G:
  - transport snapshot model
- Worker H:
  - UI interpolation/playhead rendering
- Verify sidecar:
  - edge-case timing + perf guardrail

### Wave 6: Human-Path Demo/Test Rebuild

Goal:
- rebuild proof surfaces using only real human paths

Tasks:
- `PB-60`
- `PB-61`
- `PB-62`
- `PB-63`

Owner split:
- Worker I:
  - demo/tooling rebuild
- Worker J:
  - human-path functional tests
- Verify sidecar:
  - manual operator proof
- Review sidecar:
  - audit against human-path rule

### Wave 7: Final Signoff

Goal:
- accept playback stabilization as the new baseline

Tasks:
- `PB-70`

Owner:
- `lead-dev`

## Agent Dispersion Plan

### First-Wave Maximum Efficiency

Run these in parallel:

1. Worker A
   - owns:
     - `echozero/application/timeline/models.py`
     - `echozero/application/presentation/models.py`
     - `echozero/application/timeline/intents.py`
     - `echozero/application/timeline/orchestrator.py`
     - `echozero/application/timeline/assembler.py`
     - `tests/application/**`
   - tasks:
     - `PB-10`
     - `PB-11`
     - `PB-12`

2. Worker B
   - owns:
     - `echozero/ui/qt/app_shell.py`
     - app-shell-related tests
   - tasks:
     - `PB-13`
   - note:
     - must adapt to Worker A’s state-model change, not invent parallel state

3. Verify Sidecar
   - owns:
     - no writes by default
   - tasks:
     - prepare proof commands and failure signals for `PB-14`

4. Research Sidecar
   - owns:
     - no writes
   - tasks:
     - continue `PB-20` backend decision memo so Wave 3 is not blocked later

### Second-Wave Maximum Efficiency

After Wave 1 contract merges:

1. Worker C
   - owns:
     - `echozero/ui/qt/timeline/blocks/layer_header.py`
     - `echozero/ui/qt/timeline/widget.py`
     - `tests/ui/**`
   - tasks:
     - `PB-40`
     - `PB-42`

2. Worker D
   - owns:
     - `echozero/application/presentation/inspector_contract.py`
     - object-info related UI tests
   - tasks:
     - `PB-41`

3. Verify Sidecar
   - tasks:
     - `PB-43`

## Immediate Start Recommendation

Begin now with:

1. `PB-10`
2. `PB-11`
3. `PB-12`
4. `PB-13`

Do not begin backend replacement before these land.

## Status Format

Use this compact status line during execution:

`active agents / waiting / blocked / open sessions / risk`

Example:

`4 / 1 / 0 / 5 / medium`
