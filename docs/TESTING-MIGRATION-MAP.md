# Testing Migration Map

Status: reference
Last reviewed: 2026-04-30



This document maps the current EchoZero testing and demo surfaces onto the
canonical primitive and executor contract.
It exists so cleanup is explicit and bounded instead of ad hoc.

## Purpose

EchoZero currently has several overlapping proof surfaces:

- live app automation
- in-process automation provider support
- app-flow harness helpers
- simulated GUI Lane B scenarios
- legacy E2E scenario scaffolding
- demo suite scenario code

The goal is not to delete everything at once.
The goal is to make one contract canonical and route the rest toward it.

## Canonical Target

Target stack:

1. primitive catalog:
   [docs/TESTING-PRIMITIVES.md](TESTING-PRIMITIVES.md)
2. executor contract:
   [docs/TESTING-EXECUTORS.md](TESTING-EXECUTORS.md)
3. scenario schema:
   [docs/TESTING-SCENARIO-SCHEMA.md](TESTING-SCENARIO-SCHEMA.md)
4. live app automation bridge and `packages/ui_automation/**` as public app
   control surface

## Surface Classification

### Keep As Canonical

#### Live app automation

Files:

- `packages/ui_automation/**`
- `echozero/ui/qt/automation_bridge.py`
- `tests/ui_automation/**`
- `run_echozero.py`

Action:

- keep
- standardize on canonical primitive ids
- keep old action names only as temporary aliases

### Keep As Internal Support

#### App harness support

Files:

- `echozero/testing/app_flow.py`
- `tests/testing/test_app_flow_harness.py`

Action:

- keep as internal app-boundary setup support
- adapt to canonical primitive ids
- do not treat as the public automation API

### Keep As Simulated Proof

#### GUI Lane B

Files:

- `echozero/testing/gui_dsl.py`
- `echozero/testing/gui_lane_b.py`
- `tests/testing/test_gui_lane_b.py`
- `tests/gui/scenarios/e2e_core.json`

Action:

- keep for deterministic simulated GUI coverage
- relabel scenarios to canonical primitive ids
- stop adding new GUI-specific action names

Constraint:

- continue labeling as simulated proof, not human-path demo proof

### Transitional Only

#### Legacy E2E scaffolding

Files:

- `echozero/testing/e2e/**`

Action:

- freeze except for compatibility or migration work
- rebuild on top of canonical primitives if retained
- do not expand as a parallel automation model

### Demo/Artifact Support

#### Demo suites

Files:

- `echozero/testing/demo_suite.py`
- `echozero/testing/demo_suite_scenarios.py`
- `echozero/ui/qt/timeline/demo_app.py`

Action:

- keep only for demo/fixture generation and explicitly simulated proof
- migrate reusable flows toward scenario primitives where possible
- stop using demo helpers as proof of app acceptance

## Current To Target Mapping

| Current Surface | Current Shape | Target Shape | Action |
| --- | --- | --- | --- |
| `AutomationSessionBackend.invoke(action_id, target_id, params)` | string action id plus tool-local params | canonical request envelope plus alias resolution | evolve |
| `EchoZeroAutomationProvider` actions | mixed canonical and ad hoc ids | canonical primitive ids in `AutomationAction` | evolve |
| `gui_dsl.py` `SUPPORTED_ACTIONS` | custom action list | canonical primitive ids | replace names |
| `gui_lane_b.py` step handlers | ad hoc action branching | canonical primitive dispatcher | refactor |
| `e2e/scenario.py` `act/assert/capture/wait` | generic step model with freeform `action` | canonical `invoke/assert/capture/wait` schema | adapt or retire |
| `demo_suite_scenarios.py` Python helper flows | direct presentation mutation and demo helper logic | capture-only/demo-only scenarios with explicit proof class | constrain |

## Primitive Alias Migration

The immediate alias work should cover:

- `add_song_from_path` -> `song.add`
- `extract_stems` -> `timeline.extract_stems`
- `extract_drum_events` -> `timeline.extract_drum_events`
- `classify_drum_events` -> `timeline.classify_drum_events`
- `extract_classified_drums` -> `timeline.extract_classified_drums`
- `select_first_event` -> `selection.first_event`
- `nudge` and `nudge_selected_events` -> `timeline.nudge_selection`
- `duplicate` and `duplicate_selected_events` -> `timeline.duplicate_selection`
- `open_push_surface` -> `transfer.workspace_open` with `direction=push`
- `open_pull_surface` -> `transfer.workspace_open` with `direction=pull`
- `apply_transfer_plan` -> `transfer.plan_apply`
- `enable_sync` -> `sync.enable`
- `disable_sync` -> `sync.disable`
- `screenshot` -> `capture.screenshot`

## Recommended Work Order

### Phase 1: Normalize The Public Surface

Scope:

- `packages/ui_automation/**`
- EchoZero provider action inventory
- alias resolver helper

Done when:

- public action inventory exposes canonical ids
- old ids still work as aliases

### Phase 2: Normalize Internal Executors

Scope:

- `echozero/testing/app_flow.py`
- `echozero/testing/gui_lane_b.py`
- `echozero/testing/gui_dsl.py`

Done when:

- internal runners consume canonical ids first
- GUI DSL no longer defines a separate action vocabulary

### Phase 3: Normalize Scenario Files

Scope:

- `tests/gui/scenarios/**`
- any retained `echozero/testing/e2e/**` scenario files

Done when:

- scenario files reference canonical primitive ids
- selector shape is stable and semantic

### Phase 4: Constrain Or Retire Parallel Surfaces

Scope:

- `echozero/testing/e2e/**`
- demo helper flows that behave like alternate test frameworks

Done when:

- no new ad hoc action models are being added
- each surviving surface has a clear canonical role

## File Rename Guidance

Not every filename needs to change immediately.
The first priority is semantic convergence, not churn.

That said, future filenames should prefer contract language:

- `test_echozero_backend_actions.py` is better than `test_echozero_backend.py`
  once action coverage becomes the focus
- `drum_classification_flow.json` is better than `e2e_core.json`
- `testing_primitives.py` is better than another `gui_*` action helper if the
  code is executor-neutral

## Acceptance Criteria

The migration is on track when:

- new tests use canonical primitive ids by default
- providers expose canonical ids in snapshots/actions
- internal scenario runners accept the canonical schema
- simulated proof remains clearly labeled
- the repo stops growing new action vocabularies

## Out Of Scope

- immediate deletion of legacy code
- broad file renames without behavior change
- replacing the app automation bridge

The priority is standardization first, then cleanup.
