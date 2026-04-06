# Autonomous E2E Control Plan - 2026-04-06

## Purpose

Define a practical implementation plan for assistant-driven end-to-end testing in EchoZero that:

- proves Stage Zero behavior with video and report artifacts
- reuses the same harness core for Foundry
- stays aligned with current repo architecture, existing capture scripts, and current CI lanes
- phases scope realistically without broad rewrites

This document is a build plan, not a speculative redesign.

## Grounding References

This plan is grounded in:

- `docs/UI-SOURCE-OF-TRUTH-2026-04-05.md`
- `docs/UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05.md`
- `docs/UI-INTERACTION-CLOSURE-PLAN-2026-04-05.md`
- `docs/PERF-GUARDRAILS.md`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `docs/architecture/DECISIONS.md`
- `run_timeline_capture.py`
- `run_timeline_real_data_capture.py`
- `run_timeline_walkthrough.py`
- `echozero/ui/qt/timeline/test_harness.py`
- `tests/foundry/test_foundry_app.py`
- `tests/foundry/test_ui_smoke.py`
- `.github/workflows/test.yml`

## Non-Goals

- no broad rewrite of Stage Zero or Foundry
- no replacement of current pytest, screenshot, or benchmark lanes
- no fantasy agent stack that depends on unstable vision autonomy for basic correctness
- no mandatory golden-image gate in v0

## Planning Constraints

The implementation must preserve the current repo direction:

- main remains truth; takes remain subordinate comparison lanes
- app-intent proof is preferred over fragile desktop-only automation
- deterministic screenshot capture stays the first recording primitive
- walkthrough recording becomes automated incrementally, not all at once
- Foundry compatibility comes from a shared harness core plus app-specific adapters

## Target Outcome

The end state is a reusable autonomous E2E harness with six separable layers:

1. Scenario DSL
2. Driver abstraction
3. App adapter
4. Recorder
5. Verifier
6. Reporter

The same scenario should be able to run:

- at L1 against app intents for deterministic proof
- at L2 against Qt widgets for interaction proof
- at L3 against the desktop as a human-equivalent fallback

Artifacts from every run should be staged in a stable structure that is easy to review locally, archive in CI, and send over Telegram.

## Architecture

### 1. Scenario DSL

The Scenario DSL is the stable contract for assistant-authored E2E plans.

Requirements:

- declarative and serializable
- app-agnostic at the core
- explicit about preconditions, actions, checkpoints, and assertions
- supports choosing allowed control levels per step
- supports artifact hints such as `record_start`, `checkpoint`, and `record_stop`

Minimum shape:

```yaml
schema: ez.e2e.scenario.v1
app: stagezero
name: timeline_take_promotion_smoke
fixture: real_data_timeline
allowed_levels: [L1, L2, L3]
steps:
  - do: launch
  - do: open_fixture
    target: real_data_timeline
  - do: checkpoint
    id: initial_shell
  - do: toggle_take_lane
    layer: drums
  - do: select_take
    layer: drums
    take: take_2
  - assert:
      kind: presentation_field
      path: layers.drums.is_expanded
      equals: true
  - do: checkpoint
    id: take_selected
```

Rules:

- the DSL describes intent first, not raw clicks first
- raw coordinates are forbidden in the scenario source
- each step may declare a required minimum level only when necessary
- app-specific nouns live in adapters, not in the core schema

### 2. Driver Abstraction

The driver layer executes a scenario step through one of three control levels.

Contract:

- `capabilities() -> DriverCapabilities`
- `launch()`
- `reset_fixture()`
- `execute(step)`
- `capture_checkpoint(checkpoint_id)`
- `inspect() -> RuntimeStateSnapshot`
- `shutdown()`

Selection policy:

- prefer L1 when the adapter exposes the intent deterministically
- escalate to L2 only when the proof requires actual widget interaction
- escalate to L3 only when Qt-level control is unavailable or intentionally being validated at desktop level

This keeps most verification deterministic while still allowing real interaction proof where needed.

### 3. App Adapters

The app adapter maps app-specific concepts to the shared harness contract.

Required adapters:

- `StageZeroAdapter`
- `FoundryAdapter`

Responsibilities:

- expose fixtures
- map scenario nouns to app operations
- surface stable inspection state for verification
- describe app-owned windows and capture targets
- publish artifact metadata relevant to reporting

The adapter is where app semantics belong. The harness core should not know timeline-specific or training-run-specific internals.

### 4. Recorder

The recorder layer produces reviewable evidence.

It should support:

- deterministic window-only video capture
- screenshot checkpoints
- optional event log sidecar
- consistent start/stop behavior tied to scenario steps

The recorder is not the verifier. It records what happened; it does not decide pass/fail.

### 5. Verifier

The verifier turns runtime state and artifacts into acceptance signals.

It should support:

- assertions embedded in the scenario
- adapter-provided structural conformance checks
- matrix generation showing which proof lanes passed
- optional golden-record comparison as a separate lane

### 6. Reporter

The reporter emits human-reviewable outputs.

Minimum outputs:

- `summary.json`
- `summary.md`
- `assertions.json`
- `conformance-matrix.json`
- screenshots
- video path or explicit `not-captured`

The reporter also prepares the artifact bundle for CI upload or Telegram send.

## Control Levels

### L1 App-Intent Driver

Purpose:

- deterministic proof of behavior through stable application interfaces

Stage Zero implementation path:

- reuse timeline demo/test harness patterns
- dispatch timeline intents directly where current surfaces already exist
- inspect presentation models rather than pixels when possible

Foundry implementation path:

- drive `FoundryApp` service boundaries and Foundry UI composition root where needed
- inspect run, artifact, and activity state directly

Best for:

- truth-model assertions
- take/main semantics
- state transitions
- repeatable CI
- low-flake artifact generation

Constraint:

- L1 only proves what the app exposes; it does not validate pointer hit testing or native window behavior

### L2 Qt Interaction Driver

Purpose:

- prove widget-level interaction inside the Qt application without involving the full OS

Implementation path:

- use Qt test primitives and stable widget locators
- target widget object names, semantic roles, or adapter-owned selectors
- map clicks, keypresses, drags, and scrolls to real Qt events

Best for:

- transport clicks
- ruler drag
- take-lane expansion
- selection flows
- Foundry window smoke and bounded UI actions

Constraint:

- widget selectors must be intentionally exposed; do not bind the harness to paint coordinates

### L3 OS Desktop Driver

Purpose:

- human-equivalent fallback when L1/L2 cannot prove the path or when end-to-end desktop behavior itself is the subject

Implementation path:

- desktop window targeting by title/process
- bounded mouse/keyboard automation
- screenshot/video capture at the OS window level

Best for:

- launch/install smoke
- native dialogs
- focus issues
- clipboard or OS-window choreography
- proving that an assistant can recover when a widget-level path is unavailable

Constraint:

- L3 is highest flake and should remain fallback, not default

## Driver Escalation Policy

Use a strict escalation rule:

1. Try L1 if the step is expressible as app intent and the verification target is semantic state.
2. Use L2 if the proof requires actual widget interaction, hover, drag, focus, or keyboard routing.
3. Use L3 only if the app cannot expose the path at L1/L2 or if native desktop behavior is itself under test.

The scenario result should record both:

- `requested_level`
- `actual_level`

This matters for triage and for measuring how much of the system remains desktop-fragile.

## Video Pipeline

### Deterministic Window-Only Recording Path

V0 and V1 should record the app window only, not the full desktop.

Reasons:

- smaller files
- less privacy risk
- less visual noise
- more stable replay artifacts
- easier future golden-record comparison

Recording contract:

- adapter provides target window identity
- recorder crops to the window bounds only
- fixed frame rate and codec profile per environment
- run metadata records window size, DPI assumptions, and recording timestamps

Stage Zero should start by wrapping the existing deterministic screenshot path and add video around walkthrough-style flows.

### Screenshot Checkpoints

Every scenario may emit named checkpoints.

Checkpoint rules:

- checkpoints are mandatory at scenario start and end
- checkpoints are mandatory before and after any high-value interaction cluster
- checkpoint names must be semantic, not numeric only
- screenshot capture should be possible independent of video capture

This preserves useful artifacts even when video capture fails.

### Artifact Staging Conventions

Every run should stage artifacts under one root:

```text
artifacts/e2e/<app>/<scenario>/<run_id>/
```

Required structure:

```text
artifacts/e2e/stagezero/timeline_take_selector_smoke/2026-04-06T070000Z/
  summary.json
  summary.md
  assertions.json
  conformance-matrix.json
  telemetry.json
  checkpoints/
    00_start.png
    10_take_lanes_open.png
    20_end.png
  video/
    window.mp4
  logs/
    runner.log
    app.log
  telegram/
    caption.txt
    attachments.json
```

Telegram staging rules:

- one small caption file with pass/fail, scenario, app, commit, and top failures
- one attachment manifest listing the preferred send order
- prefer first: `summary.md`, key checkpoint PNGs, then video, then JSON sidecars

## Verification Model

### Assertions

Assertions should be explicit and typed.

Initial assertion types:

- `presentation_field`
- `service_state`
- `widget_state`
- `artifact_exists`
- `image_exists`
- `video_exists`
- `log_contains`

Rules:

- L1 assertions should prefer app state over pixel inference
- L2 assertions may use widget state plus checkpoint screenshots
- L3 assertions should minimize raw image reasoning and prefer adapter-readable state when available

### Conformance Matrix Generation

Each run should generate a compact conformance matrix that answers:

- which proof lanes ran
- which control levels were used
- which assertions passed
- which artifacts were produced
- whether the run is reusable as acceptance evidence

Minimum matrix dimensions:

- contract/unit
- app-intent e2e
- widget-interaction e2e
- desktop-fallback e2e
- screenshot evidence
- video evidence
- perf lane
- golden lane

This becomes the bridge between the current verification plan and the new autonomous harness.

### Golden-Record Comparator Lane

Golden comparison should be a separate lane, not the default blocker in v0.

Near-term design:

- compare deterministic screenshots only
- allow adapter-defined ignore regions if required later
- emit diff images and a structured result

What becomes golden first:

- Stage Zero deterministic presentation screenshots
- Foundry bounded window smoke screenshots

What should not be golden first:

- free-running desktop video
- unstable native dialog flows

## CI And Manual Cadence

### CI Cadence

Keep CI practical and narrow.

V0:

- preserve current pytest and perf jobs
- add one small L1 Stage Zero scenario job
- upload E2E artifacts the same way perf artifacts are uploaded today

V1:

- add one L2 Stage Zero interaction scenario
- add one Foundry L1 scenario

V1.5:

- add optional nightly golden lane
- add nightly video capture smoke if runner support is stable

V2:

- add bounded L3 smoke on approved runners only

### Manual Cadence

Use manual runs for the flows CI cannot yet carry cleanly:

- reviewer-visible walkthroughs
- OS-native dialog proof
- failure reproduction with full video
- Telegram-ready artifact bundles for remote review

### Failure Triage Flow

Every failed run should classify into one bucket first:

1. scenario defect
2. harness/driver defect
3. adapter defect
4. app regression
5. environment/runner defect

Triage order:

1. inspect `summary.md`
2. inspect `conformance-matrix.json`
3. inspect final checkpoint and first failing checkpoint
4. inspect video if available
5. inspect logs
6. rerun at lower-flake level if possible

Useful rule:

- if L2 fails but equivalent L1 passes, suspect widget wiring, selector drift, or timing
- if L3 fails but L2 passes, suspect desktop environment, focus, DPI, or native shell issues
- if L1 fails, suspect real app logic or adapter mapping first

## Foundry Compatibility

### Shared Harness Core

The shared harness core should own:

- Scenario DSL parsing
- run orchestration
- driver selection and escalation
- artifact directory creation
- recorder orchestration
- assertion execution
- report generation

This core must not import Stage Zero timeline internals directly.

### Foundry Adapter Contract

The Foundry adapter should expose:

- app launch/bootstrap
- fixture setup for dataset/version/run scenarios
- semantic actions such as `create_dataset`, `plan_version`, `start_run`, `validate_artifact`
- readable inspection state for run status, artifacts, and activity feed
- stable window identity for recording when UI is involved

This aligns with the existing `FoundryApp` composition root and current smoke coverage rather than forcing Foundry into Stage Zero assumptions.

### Stage Zero Adapter Contract

The Stage Zero adapter should expose:

- timeline fixture setup
- scenario actions mapped to current timeline intent model where available
- widget selectors for the surfaces called out in the interaction closure plan
- presentation snapshots for deterministic verification
- current screenshot harness integration for deterministic captures

## Proposed Package Shape

Keep this additive and small:

```text
echozero/e2e/
  scenario/
  drivers/
  adapters/
  recording/
  verify/
  reporting/
```

Suggested first modules:

- `echozero/e2e/scenario/schema.py`
- `echozero/e2e/scenario/parser.py`
- `echozero/e2e/drivers/base.py`
- `echozero/e2e/drivers/app_intent.py`
- `echozero/e2e/drivers/qt_driver.py`
- `echozero/e2e/adapters/base.py`
- `echozero/e2e/adapters/stagezero.py`
- `echozero/e2e/adapters/foundry.py`
- `echozero/e2e/recording/artifacts.py`
- `echozero/e2e/verify/assertions.py`
- `echozero/e2e/reporting/summary.py`

## Phased Scope

### v0

Goal:

- prove the harness shape with deterministic artifacts and no desktop dependency

Scope:

- Scenario DSL v1
- shared artifact staging
- L1 driver only
- Stage Zero adapter only
- screenshot checkpoints only
- assertions + conformance matrix
- markdown/json summary reporting
- one or two Stage Zero scenarios based on current timeline demo and real-data capture flows

Exit condition:

- CI can run at least one deterministic Stage Zero E2E scenario and upload artifacts

### v1

Goal:

- add real Qt interaction proof without destabilizing the deterministic lane

Scope:

- L2 Qt driver
- Stage Zero widget selectors for a narrow set of interactions
- automated walkthrough recording start/stop hooks
- Foundry adapter at L1
- one Foundry semantic E2E scenario

Exit condition:

- one Stage Zero scenario can run at L2 with screenshots and video
- one Foundry scenario can run at L1 with report artifacts

### v1.5

Goal:

- add comparison and review discipline without over-blocking merges

Scope:

- golden comparator lane for deterministic screenshots
- nightly golden job
- richer triage summaries
- Telegram packaging manifest and caption generation

Exit condition:

- nightly runs can compare selected screenshots against approved baselines and emit diffs

### v2

Goal:

- add bounded desktop fallback for native-path proof

Scope:

- L3 OS desktop driver
- desktop-window recording
- native dialog and focus recovery smoke
- limited assistant recovery actions with strict bounds

Exit condition:

- at least one bounded L3 smoke passes on approved runner(s) and produces reviewable evidence

## Practical Design Decisions

### Decision 1: L1 is the default proof lane

Reason:

- it matches the repo bias toward deterministic contracts and existing presentation harnesses

### Decision 2: Video is evidence, not the primary oracle

Reason:

- pass/fail should come from assertions and conformance checks, not free-form video interpretation

### Decision 3: Window-only recording first

Reason:

- stable artifacts matter more than cinematic desktop replay

### Decision 4: Golden lane starts from screenshots, not video

Reason:

- screenshot diffs are tractable now; video goldens are expensive and noisy

### Decision 5: Foundry is a first-class adapter, not a later port

Reason:

- this prevents the Stage Zero harness from hardcoding timeline semantics into the shared core

## Risks And Controls

### Risk: harness drifts into desktop-only automation

Control:

- require every scenario to declare why L1/L2 is insufficient before using L3

### Risk: widget selectors become brittle

Control:

- expose stable object names or semantic selectors in adapters, not paint-coordinate guesses

### Risk: artifact sprawl becomes unreviewable

Control:

- standardize the staging tree and generate one concise `summary.md`

### Risk: golden lane becomes a merge blocker too early

Control:

- keep it nightly and informational until deterministic coverage is proven stable

### Risk: Foundry and Stage Zero diverge

Control:

- make the adapter contract explicit before building app-specific convenience helpers

## Wave 1 Backlog

These are the first tickets for the next implementation wave.

1. Create `echozero/e2e/` package skeleton with shared run/artifact primitives.
2. Define Scenario DSL v1 schema and parser with step validation.
3. Implement artifact staging helper and canonical run directory layout.
4. Implement assertion engine for `presentation_field`, `artifact_exists`, and `image_exists`.
5. Implement conformance matrix generator and `summary.md` reporter.
6. Implement `StageZeroAdapter` with deterministic fixture bootstrap using current timeline harness flows.
7. Implement L1 app-intent driver for Stage Zero timeline scenarios.
8. Add first Stage Zero scenario for deterministic timeline fixture capture.
9. Add second Stage Zero scenario for real-data screenshot flow reuse.
10. Add CI job to run the v0 scenario lane and upload artifacts.

## Wave 2 Backlog

These are the first tickets for the following implementation wave.

1. Implement L2 Qt interaction driver with stable selector contract.
2. Add Stage Zero widget selectors for take toggle, transport, and ruler interactions as they become real controls.
3. Implement recorder abstraction with screenshot plus window-video lifecycle hooks.
4. Automate walkthrough capture through the E2E harness rather than standalone launch only.
5. Implement `FoundryAdapter` on top of `FoundryApp` and current UI smoke boundary.
6. Add first Foundry L1 scenario covering dataset ingest, plan, run, artifact validation, and report generation.
7. Add Telegram packaging output: `caption.txt` and `attachments.json`.
8. Add failure classification to `summary.json` and `summary.md`.
9. Add nightly golden screenshot comparison for selected deterministic scenarios.
10. Document runner requirements for future L3 desktop automation without enabling it by default.

## Recommended First Scenarios

Start with narrow scenarios that already map to current repo capabilities:

- Stage Zero demo fixture screenshot sweep
- Stage Zero real-data screenshot sweep
- Stage Zero take-lane expand/select walkthrough at L1 first, then L2
- Foundry run-to-artifact smoke
- Foundry window-open smoke

## Acceptance For This Plan

This plan is successful if implementation follows these rules:

- one shared harness core
- app-specific adapters, not app-specific harness forks
- deterministic evidence first
- video added as structured evidence, not as the only proof
- practical phasing from v0 to v2
- no broad rewrite required to start

## Immediate Recommendation

Build v0 first around the deterministic screenshot and report lane.

That path has the best leverage because it reuses current Stage Zero capture infrastructure, aligns with the existing verification plan, and forces the shared harness/app-adapter boundary before any desktop automation complexity is introduced.
