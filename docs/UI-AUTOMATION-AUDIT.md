# UI Automation Audit

_Updated: 2026-04-17_

This audit turns the current EchoZero automation work into a reusable delivery
plan for app control, human-style input, and future OpenClaw integration.

## Verdict

The current direction is correct, but it needs one architectural tightening:

- keep the live bridge
- keep pointer and keyboard primitives
- stop treating automation as only flat `targets + actions`
- promote object identity and declared capabilities into the snapshot contract

The scalable model for EchoZero is:

1. app-owned semantic objects
2. declared object actions
3. pointer and keyboard execution against those objects
4. OS-level fallback only for system-owned UI

That is the right fit for a Qt app with custom-painted timeline surfaces.

## Why This Is The Best Approach

Pure cursor automation is not enough:

- custom timeline surfaces do not expose enough structure on their own
- screenshot-only control is brittle
- image matching does not scale to fine editing or reliable regression

Pure semantic invocation is also not enough:

- it can drift away from how a human really uses the app
- it does not prove hit targets, focus, hover, drag, or key flows

The right model is hybrid:

- semantic object model for truth
- human-style pointer and key execution for realism
- OS fallback for native dialogs and packaged app smoke

## Current State

Implemented now:

- one canonical app entrypoint: `run_echozero.py`
- shared launcher surface for real app and harness construction
- semantic backend over the real shell surface
- pointer primitives: move, hover, click, double click, drag, scroll
- live localhost bridge for the real app process
- live client for attach and control
- app-path human-flow lane: `humanflow-all`

Implemented in this pass:

- snapshots now expose first-class automation objects, not only flat targets
- objects carry identity, facts, declared actions, and mapped target ids
- EchoZero snapshots derive these objects from the inspector contract

That is the correct base for reusable object-oriented automation.

## Architecture Standard

Every controllable EchoZero surface should answer four questions:

1. What object is this?
2. What facts describe its current state?
3. What actions can it perform right now?
4. Where is its human hit target on screen?

The control contract should therefore be:

- `targets`: visual/hit-testable regions
- `actions`: globally invokable actions
- `objects`: semantic selected/focused objects with facts and capabilities
- `hit_targets`: precise canvas interaction regions
- `artifacts`: screenshots, pointer state, and app metadata

The object contract should be driven by app truth, not guessed outside the app.

## Recommended Object Model

Use the existing inspector/action model as the canonical source:

- `InspectorObjectIdentity`
- `InspectorSection`
- `InspectorFactRow`
- `InspectorAction`
- `InspectorContextSection`

Do not build a second parallel object vocabulary for automation.
Instead:

- keep enriching the inspector contract
- derive automation objects from it
- keep transport, bridge, and agent wrappers thin

This keeps one truth source for:

- inspector UI
- automation
- future agent tooling

## Gaps

Still missing or incomplete:

- stable live proof for `app.new`, `app.save_as`, `app.open`
- live import flow over the bridge
- native file dialog handling
- explicit focus model in snapshots
- object coverage beyond the current selected inspector object
- packaged app smoke automation
- real MA3 hardware lane

## Concrete Implementation Steps

### Phase 1: Contract hardening

Goal:

- make the snapshot contract stable enough to drive both tests and agent tools

Steps:

1. Keep `AutomationObject` in the reusable core package.
2. Add `focused_target_id` and `focused_object_id` to snapshot artifacts or a
   dedicated focus field.
3. Ensure the live bridge and live client preserve object data losslessly.
4. Add contract tests for object serialization and live bridge transport.

Exit proof:

- core/session tests green
- bridge snapshot round-trip includes object identity, facts, and actions

### Phase 2: Object coverage expansion

Goal:

- expose more than the currently selected object

Steps:

1. Export layer objects for all visible layers.
2. Export event objects for selected and hovered events.
3. Export transfer surfaces and transport controls as explicit action-capable
   objects where that improves discoverability.
4. Add stable object-to-target mapping rules.

Exit proof:

- snapshot can describe visible work surface semantically without querying raw
  widget internals from the agent side

### Phase 3: Lifecycle stabilization

Goal:

- make project lifecycle reliable over the live bridge

Steps:

1. Prove `app.new` live.
2. Prove `app.save_as` live against deterministic temp paths.
3. Prove `app.open` live against a known saved project.
4. Only then promote those flows into `humanflow-all`.

Exit proof:

- stable live tests for new/save_as/open
- no flaky lifecycle test in the default lane

### Phase 4: Human task execution

Goal:

- support commands like “create a new project and add song X from Desktop”

Steps:

1. Harden `add_song_from_path` over the bridge.
2. Add path resolution helpers for Desktop and other common locations.
3. Define an agent-facing action translation layer:
   `new_project`, `open_project`, `add_song_from_path`, `save_project_as`.
4. Add specialized tests for each command before composing them into longer
   scenarios.

Exit proof:

- live attach tests for new + import + save/open
- demo generated from the live bridge, not only the internal scenario runner

### Phase 5: OS fallback lane

Goal:

- cover system-owned UI without corrupting the app-owned contract

Steps:

1. Add explicit fallback adapters for native file dialogs and focus switching.
2. Keep fallback use opt-in and narrow.
3. Record when a scenario used semantic control versus OS fallback.

Exit proof:

- native dialog import smoke
- packaged app smoke lane

## Reuse Boundary

Keep the reusable boundary in `packages/ui_automation`.

The split should remain:

- `core/`: semantic data model
- `adapters/qt/`: generic Qt mechanics
- `adapters/echozero/`: app semantics
- `bridge/`: localhost transport
- agent wrapper later, only after the contract settles

Do not extract this to a separate repo yet.
EchoZero is still the proving ground.

## Immediate Next Steps

The next best implementation order is:

1. add live-client/bridge tests for object serialization
2. add focus metadata to snapshots
3. stabilize live `app.new`
4. stabilize live `app.save_as`
5. stabilize live `app.open`
6. prove `add_song_from_path` live

That sequence keeps the contract clean while moving directly toward reusable
human-task automation.
