# UI Engine Redevelopment Plan

Status: reference
Last reviewed: 2026-04-30



This document defines the redevelopment path for the EchoZero UI engine.
It exists to turn the current UI audit into a reusable internal-library plan
instead of continuing with ad hoc widget cleanup.
It connects the current UI shell, cleanup map, first principles, FEEL rules,
and proof lanes into one redevelopment sequence.

## Goal

Redevelop the EchoZero UI engine as a reusable internal library surface while
preserving the canonical EZ2 app path.

The target outcome is:

- one canonical desktop launcher: `run_echozero.py`
- one canonical EchoZero runtime shell
- a reusable UI engine layer for rendering, layout, interaction, and contracts
- dev/demo/test helpers that do not contaminate the runtime path
- app-facing proof that the redeveloped shell still behaves correctly

This is not a rewrite for its own sake.
The purpose is to preserve what is strong, extract what is reusable, and
remove what is only accidental scaffolding.

## Design Constraints

The redevelopment must preserve these repo rules:

- main is truth
- takes are subordinate lanes, not alternate live truth
- MA3 sync is main-only
- FEEL owns tuning constants
- widget-local workflow logic should not bypass application contracts
- app-facing work is not done until proven through the app path

## Current Read

The current UI already contains the beginnings of a reusable internal engine.

Strong existing candidates:

- `echozero/application/presentation/models.py`
- `echozero/application/presentation/inspector_contract.py`
- `echozero/ui/qt/timeline/blocks/**`
- `echozero/ui/FEEL.py`
- `echozero/ui/style/**`

Current bottlenecks:

- `echozero/ui/qt/timeline/widget.py` carries too much workflow/control logic
- `echozero/ui/qt/app_shell.py` mixes composition root and truth/presentation assembly
- demo/dev helpers still sit too close to canonical runtime paths
- some UI behavior still leaks older take-selection and preview-oriented mental models

## Desired Library Boundaries

The reusable UI engine should eventually be composed from these layers.

### 1. Presentation contract layer

Purpose:

- typed UI-facing data structures
- typed inspector/action contracts
- no Qt dependency
- reusable across EchoZero surfaces and future internal tools

Current seed files:

- `echozero/application/presentation/models.py`
- `echozero/application/presentation/inspector_contract.py`

### 2. FEEL and style system

Purpose:

- all tunable interaction/layout constants
- all theme tokens and QSS tokenization
- one home for visual craft and consistency

Current seed files:

- `echozero/ui/FEEL.py`
- `echozero/ui/style/tokens.py`
- `echozero/ui/style/scales.py`
- `echozero/ui/style/qt/qss.py`
- `echozero/ui/qt/timeline/style.py`

Potential reusable destination:

- `echozero/ui/library/theme.py`
- `echozero/ui/library/surface.py`

### 3. Geometry and time-layout layer

Purpose:

- timeline span math
- scroll bounds and follow-scroll math
- row geometry and hit geometry
- future time-grid and snap math

Current seed files:

- `echozero/ui/qt/timeline/blocks/layouts.py`
- geometry helpers currently embedded in `widget.py`

This layer should become more explicit and less widget-bound.

Potential reusable destination:

- `echozero/ui/library/geometry.py`
- `echozero/ui/library/time_axis.py`

### 4. Paint/layout primitives

Purpose:

- stateless blocks that paint or compose UI from explicit presentation inputs
- reusable rendering primitives for timeline-like surfaces

Current seed files:

- `echozero/ui/qt/timeline/blocks/layer_header.py`
- `echozero/ui/qt/timeline/blocks/take_row.py`
- `echozero/ui/qt/timeline/blocks/event_lane.py`
- `echozero/ui/qt/timeline/blocks/waveform_lane.py`
- `echozero/ui/qt/timeline/blocks/ruler.py`
- `echozero/ui/qt/timeline/blocks/transport_bar_block.py`

Potential reusable destination:

- `echozero/ui/library/paint.py`
- `echozero/ui/library/lane.py` only if a second consumer appears

### 5. View/controller seam

Purpose:

- translate UI events into typed intents/actions
- keep widgets thin
- keep workflow logic and action routing out of view code

Current problem area:

- `echozero/ui/qt/timeline/widget.py`

### 6. EchoZero-specific runtime shell

Purpose:

- bind the reusable UI engine to EchoZero storage, sync, playback, and timeline contracts
- remain EchoZero-specific rather than pretending to be generic

Current seed files:

- `run_echozero.py`
- `echozero/ui/qt/app_shell.py`

## What Belongs To The Library

These are good internal-library targets:

- presentation models and inspector contracts
- FEEL constants and style tokens
- geometry/layout helpers
- paint/layout blocks
- generic-ish timeline interaction utilities
- reusable object-info panel primitives

Concrete near-term library candidates:

- shared shell theme/tokens/QSS
- neutral box/row geometry helpers
- time-axis math and ruler/playhead helpers
- generic chip/button paint helpers

## What Must Stay EchoZero-Specific

These should remain application-bound:

- project storage wiring
- sync service composition
- MA3 transfer workflows
- timeline business rules
- take action semantics
- song/layer/provenance interpretation

Concrete app-owned surfaces:

- `echozero/application/presentation/models.py`
- `echozero/application/presentation/inspector_contract.py`
- `echozero/ui/qt/app_shell.py`
- timeline row semantics such as main/take, stale/edited, and MA3 transfer actions

The library should render and route.
EchoZero-specific application code should decide truth.

## What Must Be Removed From The Core

These do not belong in the reusable engine surface:

- demo walkthrough runners
- preview-only tools mixed into runtime paths
- import-time fixture registration side effects
- workflow heuristics inferred from badges/titles/labels
- stringly action routing in widgets

## Phased Redevelopment

### Phase 1: Freeze the reusable seams

Goal:

- stop further drift while preserving current behavior

Tasks:

- treat presentation contracts as the authoritative UI-facing data model
- keep FEEL/style updates centralized
- document the reusable block layer as intentional library surface
- keep `run_echozero.py` as the sole canonical launcher

Exit criteria:

- no new workflow logic lands in widget-local code without explicit justification
- all new UI work names its proof lane up front
- FEEL/follow-scroll/style guardrails remain green during extraction

### Phase 2: Extract geometry and controller seams

Goal:

- reduce `TimelineWidget` to a thinner shell

Tasks:

- extract geometry/scroll/follow helpers from `widget.py`
- extract manual-pull and object-info panel subsurfaces
- reduce string-based contract-action dispatch in the widget

Preferred first extraction order inside this phase:

1. span/scroll/follow math
2. badge and transfer-plan label helpers
3. object-info panel composition

Exit criteria:

- `TimelineWidget` becomes noticeably smaller
- interaction math is testable without the whole widget
- existing FEEL/follow-scroll tests still pass unchanged

### Phase 3: Separate runtime shell from presentation fallback policy

Goal:

- make `AppShellRuntime` a cleaner EchoZero adapter

Tasks:

- remove demo/fixture side effects from import-time composition
- stop inventing semantic truth in the app-shell view path
- move title/badge/source-label inference to typed application state or remove it

Exit criteria:

- app-shell composition is thinner
- runtime truth comes from typed contracts, not view heuristics

### Phase 4: Quarantine demo/dev surfaces

Goal:

- keep reusable library and canonical runtime clean

Tasks:

- split `demo_app.py` into reusable helpers vs dev-only runner code
- quarantine `demo_walkthrough.py`, preview tools, and fixture utilities
- keep only explicit supported hooks in the canonical launcher

Exit criteria:

- runtime package no longer depends on demo-only helpers unless explicitly justified

### Phase 5: Strengthen proof on the canonical app path

Goal:

- prove the redeveloped shell through the real app path

Tasks:

- strengthen launcher smoke on `run_echozero.py`
- expand app-shell runtime flow coverage
- keep timeline shell contract tests strict
- add real-data proof for follow-scroll and expanded take rows

Exit criteria:

- app-path proof is stronger than demo/fixture proof for UI signoff

## First Implementation Slices

These are the safest first slices after planning:

1. extract span/scroll/follow helpers out of `widget.py`
2. extract badge/label helpers out of `widget.py`
3. extract object-info panel construction out of `widget.py`
4. remove import-time fixture side effects from `app_shell.py`
5. split `demo_app.py` into reusable helper vs dev-only runner

These slices improve the architecture without forcing a single large rewrite.

## Proof Lanes

Redevelopment work should prove itself through:

- `tests/ui/test_timeline_shell.py`
- `tests/ui/test_app_shell_runtime_flow.py`
- `tests/ui/test_timeline_feel_contract.py`
- `tests/ui/test_follow_scroll.py`
- `tests/ui/test_timeline_style.py`
- `tests/ui/test_run_echozero_launcher.py`
- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane gui-lane-b` when visible shell behavior changes

Add perf guardrails if a hot timeline path changes.

## Open Decisions

These decisions still need to be made explicitly:

1. Should `--use-demo-fixture` remain as a test hook in `run_echozero.py`, or be removed from the canonical launcher surface?
2. How much of the timeline interaction layer should become a generic internal library versus an EchoZero-specific engine?
3. Which demo helpers are still strategically useful, and which are just historical drag?

## Risk Notes

- Do not move demo/bootstrap concerns out of the launcher path too early or you risk destabilizing the canonical app entry.
- Manual-pull and transfer work is the highest semantic-risk extraction area; preserve action ids, selection rules, and confirm/apply behavior while extracting it.
- Keep extraction one-way and contract-driven or the UI layer will gain duplicate truth sources and circular imports.

## Working Rule

Every UI redevelopment change should answer four questions:

1. Is this reusable enough to become library surface?
2. If not, should it stay EchoZero-specific or be removed?
3. Does it preserve the main/take/sync truth model?
4. What app-path proof confirms it?
