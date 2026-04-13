# GUI Input Demo PRD

## Problem Statement

EchoZero currently has two competing demo shapes:

- state-driven demos that inject presentation state directly for quick screenshots
- real-input flows that should prove the same product path an operator will use

The problem is that state-driven demos are cheap but can hide missing runtime seams, broken intent wiring, and UI paths that only fail under real event delivery. Lane B exists to close that gap by driving the GUI with actual input while staying inside the same application pipeline.

## First-Principles Architecture

The architecture requirement is a single pipeline with no bypass:

1. scenario data defines operator intent in a small editable DSL
2. Lane B runner loads the scenario and drives real Qt input against the timeline shell
3. widget input emits canonical timeline intents
4. runtime or orchestrator updates application state
5. assembler rebuilds presentation
6. runner captures trace and artifacts from the resulting UI state

No scenario step is allowed to mutate presentation state directly to fake success. The GUI path must stay observable, replayable, and debuggable.

## Lane Model

- Lane A: state-level contract checks and deterministic presentation coverage
- Lane B: real GUI input simulation against the app shell and timeline widget
- Lane C: full environment or system integration with external dependencies enabled
- Lane D: artifact and release-grade demo packaging, publication, and operator review

Lane B is the first lane that proves operator-visible behavior through actual input dispatch.

## DSL Design Principles

- Semantic selectors over pixel coordinates. Scenarios should target concepts like `layer_id`, `event`, `push surface`, or `sync`.
- Deterministic seeds and fixtures. Runs should begin from a known timeline state and reproducible mock sync data.
- Explicit assertions via trace snapshots. Each step should leave enough structured state to verify the flow without scraping text logs.
- Easy editing. JSON is preferred for the starter slice so product and test contributors can patch scenarios without new tooling.
- Strict validation. Unsupported actions and missing required fields fail fast before GUI execution begins.

## Artifact Strategy

- Video: reserved for later rollout once Lane B flow stability is acceptable
- Screenshots: optional per-step capture for fast visual inspection and regression review
- Trace: required JSON output listing each step, status, error payload, and state snapshot

The trace is the source artifact for CI. Screenshots supplement diagnosis. Video is a later packaging concern, not a prerequisite for the starter slice.

## Rollout Phases

### Phase 1: Starter Lane B

- JSON DSL loader and validator
- offscreen runner using real Qt input
- core actions for selection, nudge, duplicate, push, pull, sync, and screenshot
- starter scenario and focused tests

Acceptance gate:

- starter scenario passes in CI and locally
- trace JSON is written
- push, pull, and sync state transitions are observable in tests

### Phase 2: Broader Scenario Library

- multiple user journeys
- more selectors and assertions
- reusable fixtures and artifact folders

Acceptance gate:

- at least one scenario per major transfer surface
- flaky step rate is understood and tracked

### Phase 3: CI Artifact Publishing

- retain traces and screenshots per run
- summarize failures in lane output
- optional video packaging for review builds

Acceptance gate:

- CI publishes actionable artifacts for failures
- a reviewer can diagnose a failed run without rerunning locally

### Phase 4: Release-Grade Demo Flows

- richer operator stories
- external sync targets or higher-fidelity simulators
- documentation and operator handoff

Acceptance gate:

- demo flows are stable enough for repeated review use
- no known bypass remains between scripted input and user-visible behavior
