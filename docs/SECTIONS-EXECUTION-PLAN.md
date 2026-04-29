# Sections Execution Plan

_Last updated: 2026-04-26_

## Goal

Build the first canonical `Sections` workflow around the real operator need:

1. Define song-part boundaries by placing named cues.
2. Preserve imported cue identity exactly.
3. Derive owned song regions from cue order in time.
4. Render section ownership clearly across the timeline.
5. Push and pull section cues through the MA transport lane.
6. Keep markers, locators, and sections as distinct concepts.

This milestone is intentionally not "just improve marker layers." It is the
refactor that turns a marker-adjacent experiment into the actual `Sections`
product lane.

## Scope Note

- This is the active execution doc for `Sections`.
- [SECTIONS-FEATURE-SPEC.md](SECTIONS-FEATURE-SPEC.md) is the product contract.
- [architecture/DECISIONS.md](architecture/DECISIONS.md) remains the broader
  architecture record for prior section/annotation/locator decisions.
- Existing marker-layer proof work is useful implementation evidence, but it is
  not the long-term model.

## Locked Constraints

- User-facing name is `Sections`.
- `Sections` are new. Existing markers remain their own concept.
- Section data is modeled as `section cues`, rendered as derived `section
  regions`.
- The gap before the first cue is valid and must remain neutral/unowned.
- Cue refs are preserved exactly as imported or entered.
- Cue refs are not a sequential requirement.
- Time order determines ownership; cue numbering does not.
- Push and pull stay cue-based; region spans are derived locally in EchoZero.
- Generic timeline regions may be reused as implementation substrate, but
  `Sections` must not be explained to the operator as "just regions."
- Shared MA cue metadata should be standardized for both section cues and
  regular events where applicable.

## Operator Contract

The operator-facing contract for v1 `Sections` is:

- I can create or import section cues.
- Each cue has a preserved cue ref and a readable name.
- The timeline shows which cue owns each part of the song.
- The area before the first cue is visually neutral.
- I can move, rename, and delete cues without renumbering them.
- Pull reconstructs section ownership from imported cue starts.
- Push exports cue-start identity without requiring explicit region-end objects.

## Current Baseline

Already present in the repo:

- generic timeline regions with selection, editing, and backdrop rendering
- song-structure decisions in architecture docs
- marker-like annotation concepts
- ruler locators
- MA push/pull and event metadata paths

Useful current code surfaces:

- `echozero/application/presentation/models.py`
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/ui/qt/app_shell_project_timeline.py`
- `echozero/ui/qt/app_shell_storage_sync.py`
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/region_manager.py`
- `echozero/ui/qt/timeline/blocks/ruler.py`
- `echozero/ui/qt/timeline/widget_canvas_paint_mixin.py`

Current risk:

- recent marker-layer proof work improved cue labeling and transfer proof on the
  marker path; that path must not quietly become the long-term `Sections`
  contract.

## Refactor Direction

The refactor should move in this direction:

- retain marker and locator concepts
- introduce explicit section-cue vocabulary and data shapes
- derive section-owned spans from ordered cues
- standardize shared cue metadata across transport/event boundaries
- let section-specific UI sit on top of reusable region/backdrop primitives
- avoid mass migration of existing marker layers unless later explicitly chosen

## Execution Order

### Slice 0 — Lock contract and name the lane

Status: implemented on 2026-04-26

- Add the feature spec.
- Lock the neutral pre-first-cue gap rule.
- Lock cue-ref preservation and non-sequential numbering.
- Lock the split between sections, markers, and locators.

Done when:

- the product contract exists in-repo
- future work no longer depends on chat memory

### Slice 1 — Canonical section-cue model boundary

Goal:

- establish explicit section-cue vocabulary and app-model support

Scope:

- add canonical section-cue data shapes to the application/presentation layer
- add derived section-region projection from ordered cues
- ensure cue refs are represented as preserved strings, not normalized ints
- keep the pre-first-cue gap implicit and neutral
- avoid forcing a UI rewrite in the same slice

Likely files:

- `echozero/application/presentation/models.py`
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/assembler*`
- `echozero/application/timeline/orchestrator.py`
- focused tests under `tests/application/`

Done when:

- the app can represent canonical section cues directly
- derived regions come from cue ordering
- no automatic cue renumbering exists in the sections path

### Slice 2 — Shared cue metadata transport boundary

Goal:

- unify the cue metadata contract shared by section cues and regular events

Scope:

- define one normalized metadata envelope for:
  - `cue_ref`
  - `label` / `name`
  - `color`
  - `notes`
  - `payload_ref`
- thread that envelope through relevant import/export and storage helpers
- keep markers and events able to carry the same metadata without turning all
  events into sections

Likely files:

- `echozero/ui/qt/app_shell_project_timeline_storage.py`
- MA transport/import-export helpers
- focused tests for serialization/import/export normalization

Done when:

- section cues and regular events can preserve the same cue metadata fields
- transport normalization no longer depends on section-only ad hoc metadata

### Slice 3 — Timeline rendering and editing UX

Status:

- Completed on 2026-04-26 for the shell-side bridge + UX pass.
- Landed as a dedicated `LayerKind.SECTION` flow with canonical
  `timeline.section_cues`, a `Sections` manager dialog, section boundary
  labels, and derived ownership overlays in the ruler and timeline body.
- Residual boundary: Slice 4 still owns end-to-end MA push/pull
  canonicalization beyond the local edit/load/render path.

Goal:

- make section ownership obvious and editable in the shell

Scope:

- add or adapt a dedicated `Sections` lane/surface
- render strong left-edge boundaries plus readable cue identity
- render derived owned section backdrops across the timeline
- preserve neutral styling before the first cue
- support add, move, rename, and delete behaviors according to the spec

Likely files:

- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/region_manager.py`
- `echozero/ui/qt/timeline/blocks/ruler.py`
- `echozero/ui/qt/timeline/widget_canvas_paint_mixin.py`
- `echozero/ui/qt/timeline/style.py`
- focused tests under `tests/ui/`

Done when:

- the operator can clearly see which section owns any point in time
- the pre-first-cue gap is visibly neutral
- section labels read as cue-owned song parts, not marker pills

### Slice 4 — Push / pull canonicalization

Status:

- Completed on 2026-04-26 for the current MA transport path.
- Pull now treats `LayerKind.SECTION` as canonical main-take import, refreshes
  `timeline.section_cues`, and falls back to cue-number text when MA does not
  expose a separate cue ref.
- Push now encodes explicit cue refs into the transport label path and pull
  recovers them from either explicit bridge fields or cue-prefixed labels.
- Focused proof is covered by MA adapter, manual pull, MA3 push, simulated
  bridge, MA3 OSC bridge, transfer-plan, and app-flow harness tests.

Goal:

- make import/export use the section-cue contract end to end

Scope:

- pull imports cue starts and reconstructs derived section ownership
- push exports cue-start identity only
- no explicit region-end transport requirement unless a real MA canonical shape
  is introduced later
- preserve cue refs exactly through round-trip

Likely files:

- MA transfer/import-export boundaries
- app-layer transfer orchestration
- focused tests under `tests/application/` and `tests/testing/`

Done when:

- pull produces canonical section cues with derived regions
- push sends cue-start truth only
- round-trip proof preserves cue refs exactly

### Slice 5 — Marker/path cleanup and migration policy

Goal:

- prevent conceptual drift after sections land

Scope:

- review the old marker-layer proof path
- decide what remains marker-only
- add explicit conversion tooling only if needed
- document whether generic regions remain substrate-only or become removable

Done when:

- sections are no longer mentally or structurally dependent on marker layers
- any migration is explicit, not silent

## What Completes The First Integrated Milestone

The first integrated milestone is complete when all of this is true:

- canonical section-cue data exists in the application boundary
- the timeline renders owned song sections from cue starts
- the pre-first-cue gap is neutral
- cue refs round-trip without renumbering
- pull reconstructs section spans from cue order
- push exports cue starts cleanly
- markers and locators remain distinct and usable

## Current First Refactor Slice

Start with Slice 1 and Slice 2 before the big UI rewrite.

Reason:

- the biggest long-term risk is getting trapped in a marker-layer-derived data
  model
- once the model and transport boundary are right, rendering and editing can be
  rebuilt on stable ground

## Not In This Plan's Done Bar

Not required for the first sections milestone:

- deprecating markers immediately
- deleting generic region plumbing immediately
- automatic migration of all existing marker layers
- live MA hardware proof before the canonical model exists
- ML-generated section analysis as the primary path
