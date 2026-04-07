# Unified Source Of Truth Plan - 2026-04-07

## Purpose

This document is the single canonical execution plan for the `panel-object-inspector` worktree.

It replaces the current split across distillation audit, UI source-of-truth notes, inspector milestone notes, interaction closure notes, and verification notes with one ordered chain:

1. distillation intent
2. first principles
3. application/API contract
4. UX and FEEL constraints
5. real-world DAW reference behavior
6. implementation backlog
7. acceptance gates

If a lower section conflicts with a higher section, the higher section wins.

## Canonical Source Order

### 1. Distillation intent

Primary authority:
- external distillation source referenced by repo audits as `memory/echozero-distillation/DISTILLATION.md`

Local rule:
- that file is not present in this worktree, so distillation intent is treated as upstream authority but cannot be used as a local implementation source directly
- local contributors should use [`docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`](./DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md) as the local proxy for what distillation currently requires until the source is vendored into the repo

### 2. First principles

Primary local authority:
- [`docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`](./architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md)

Non-negotiable rules carried into object inspector work:
- main is truth
- takes are subordinate
- no active-take truth model
- provenance, freshness, and manual modification are distinct
- MA3 sync boundary is main-only
- blank-slate song-version policy remains intact

### 3. Application/API contract

Primary code authorities:
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/timeline/assembler.py`
- `echozero/application/timeline/queries.py`
- `echozero/application/timeline/repository.py`

Supporting service authorities where the panel depends on generated-state behavior:
- `echozero/services/orchestrator.py`
- `echozero/services/dependencies.py`
- `echozero/services/take_actions.py`
- `echozero/persistence/session.py`

Contract rule:
- the inspector may only surface state and edits that can be expressed through application-backed models and intents
- widget-local state may help with transient interaction, but must not become semantic truth

### 4. UX contract

Primary local authorities:
- [`docs/OBJECT-INSPECTOR-MILESTONE-PLAN-2026-04-06.md`](./OBJECT-INSPECTOR-MILESTONE-PLAN-2026-04-06.md)
- [`docs/UI-INTERACTION-CLOSURE-PLAN-2026-04-05.md`](./UI-INTERACTION-CLOSURE-PLAN-2026-04-05.md)
- [`docs/UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05.md`](./UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05.md)

Important constraint:
- older docs reference `docs/UX-DESIGN-DECISIONS.md` and `docs/UX-MICRO-TESTS.md`, but those files are not present in this worktree
- therefore they are not allowed to function as local blocking authorities until restored

### 5. FEEL contract

Primary authority:
- `echozero/ui/FEEL.py`

FEEL owns:
- sizes
- spacing
- timing
- hit-area tuning
- visual density defaults

FEEL does not own:
- truth model
- selection semantics
- edit authority
- provenance rules

### 6. Real-world DAW references

Use these as tie-breakers for interaction feel and information architecture only after the higher layers above are satisfied:

- Ableton Live: inspector/detail panel should stay selection-driven, contextual, and low-friction
- Bitwig Studio: inspector should expose layered device/object properties without pretending unsupported editability
- REAPER: batch-safe property editing should be explicit about mixed values and scope
- Logic Pro / Pro Tools: transport, selection, and timeline-adjacent controls should feel direct and unsurprising

Reference rule:
- DAW precedent can shape presentation and workflow clarity
- DAW precedent cannot override EchoZero truth-model rules, application intent boundaries, or FEEL ownership

### 7. Evidence and audits

Use as evidence, not primary authority:
- [`docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`](./DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md)
- [`docs/UI-CONTRACT-AUDIT-2026-04-03.md`](./UI-CONTRACT-AUDIT-2026-04-03.md)
- [`docs/architecture/UI-BLOCK-AUDIT-2026-04-02.md`](./architecture/UI-BLOCK-AUDIT-2026-04-02.md)

## Deprecated Docs And Rules

These documents are no longer canonical planning sources after this file lands:

- [`docs/UI-SOURCE-OF-TRUTH-2026-04-05.md`](./UI-SOURCE-OF-TRUTH-2026-04-05.md)
  - deprecated as the canonical source-order note
  - retained as historical reconciliation context
- [`docs/OBJECT-INSPECTOR-MILESTONE-PLAN-2026-04-06.md`](./OBJECT-INSPECTOR-MILESTONE-PLAN-2026-04-06.md)
  - deprecated as the top-level plan
  - retained as scope input for the inspector backlog
- [`docs/UI-INTERACTION-CLOSURE-PLAN-2026-04-05.md`](./UI-INTERACTION-CLOSURE-PLAN-2026-04-05.md)
  - deprecated as the execution master plan
  - retained as backlog source material
- [`docs/UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05.md`](./UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05.md)
  - deprecated as a standalone gate document
  - retained as the basis of the acceptance section below

Deprecated rules:
- no plan may treat audits as the source of truth instead of evidence
- no plan may promote absent files into active authorities
- no UI plan may assume active-take truth, fake editability, or FEEL drift is acceptable

## Execution Order

### Phase 0. Lock the authority chain

Outcome:
- contributors know exactly which document or code surface wins when sources conflict

Tasks:
- adopt this document as the only canonical execution plan
- route all future object-inspector decisions through the source order above
- stop creating parallel plan docs unless they are explicitly subordinate addenda

### Phase 1. Re-state the object inspector from first principles

Outcome:
- the inspector is defined as a selection-driven view/edit surface that reflects main-truth application state

Tasks:
- preserve the three panel states: none selected, one selected, multiple selected
- prohibit fake editing for fields without backing intents or domain rules
- define multi-select behavior as batch-safe fields only, with mixed-value handling
- ensure empty state and active-layer fallback remain supportive, not misleading

### Phase 2. Lock the API/application contract before UI growth

Outcome:
- the panel binds to deterministic selection, provenance, stale/manual flags, and edit intents

Tasks:
- remove any remaining active-take truth leakage from timeline-visible contract paths
- make event and object selection preserve enough identity to recover layer, take, and object/event
- expose provenance, freshness, and manual-modified state from application data rather than synthetic defaults
- ensure unsupported types and fields are rendered as read-only or absent, never falsely editable

### Phase 3. Translate the contract into inspector UX and FEEL

Outcome:
- the inspector feels like a real DAW-side property surface without violating EchoZero's model

Tasks:
- map inspector layout density, spacing, and timing to `echozero/ui/FEEL.py`
- keep inspector interaction direct, contextual, and selection-driven
- use DAW precedent for grouping, mixed-value presentation, and transport-adjacent clarity
- avoid hidden semantic state in widgets

### Phase 4. Build in bounded slices

Outcome:
- implementation lands in stable increments with visible proof

Backlog order:
1. Panel shell, docking, open/close persistence, and deterministic selection binding
2. Empty state, active-layer fallback, and single-selection read-only object details
3. Application-backed status chips and metadata for provenance, freshness, and manual edits
4. Single-selection bounded editing for fields already backed by intents, validation, and undo-safe dispatch
5. Unsupported-type handling and unavailable-field handling
6. Multi-selection summary with mixed values and batch-safe edits only
7. Follow-on object-specific affordances only where the application layer already supports them

### Phase 5. Prove the real surface

Outcome:
- the panel is accepted on contract conformance, realistic behavior, and reviewer-visible evidence

Tasks:
- run targeted contract/unit proof
- run real-data smoke when visible timeline/object state changes
- produce walkthrough capture when reviewer-visible behavior changes
- run perf guardrails when paint, assembly, density, zoom, or scroll paths change

## Acceptance Gates

### Gate A. Source-order compliance

Pass when:
- the proposed change cites or clearly follows the canonical source order in this document
- no lower-level doc or widget behavior overrides first principles or application truth

Fail when:
- a change relies on audit prose as authority
- a change revives active-take truth or widget-local semantic truth

### Gate B. Contract completeness

Pass when:
- selection is the sole authority for rendered inspector subject
- provenance, freshness, and manual modification are application-backed
- edits dispatch application intents instead of mutating presentation-only truth

Fail when:
- inspector state can drift from application state
- unsupported fields pretend to be editable

### Gate C. UX and FEEL coherence

Pass when:
- spacing, sizing, and timing are FEEL-backed
- inspector grouping and mixed-value behavior remain consistent with DAW-grade expectations
- no interaction creates surprise relative to selection-driven panel behavior

Fail when:
- magic numbers or ad hoc timing values are introduced
- the UX implies semantics the application does not support

### Gate D. Verification sweep

Required proof lanes by default:
- contract/unit

Required when visible behavior changes:
- real-data smoke
- walkthrough capture

Required when hot paths change:
- perf guardrails

Pass when:
- all required lanes for the change are green and artifacts are produced where applicable

Fail when:
- a visible change lands without the appropriate proof lane

## Implementation Backlog

### P0

- adopt this file as the single planning authority and stop using parallel source-of-truth docs
- kill active-take truth leakage in any timeline or inspector-facing path
- expose real provenance, freshness, and manual-modified fields to the inspector contract
- confirm selection identity is sufficient for none/single/multi inspector states
- wire panel shell and persistence with deterministic selection-driven rendering

### P1

- ship read-only single-selection inspector with empty and no-selection states
- add application-backed status chips, metadata, and unsupported-type handling
- add bounded single-selection editing for intent-backed fields only
- add validation, cancel, commit, and selection-change-during-edit tests

### P2

- ship multi-selection summary state
- add mixed-value presentation and batch-safe edits only
- add richer provenance/source inspection without inventing unsupported edit flows

### P3

- evaluate object-specific affordances already supported by the app layer
- prepare future golden-record readiness without making it a blocker now

## Last Few Hours Accounted Status

- [x] Existing source-order guidance reviewed and folded in from `UI-SOURCE-OF-TRUTH-2026-04-05`
- [x] Object inspector scope and phased rollout reviewed and folded in from `OBJECT-INSPECTOR-MILESTONE-PLAN-2026-04-06`
- [x] Verification lanes and acceptance proof reviewed and folded in from `UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05`
- [x] Distillation conformance risks reviewed and carried forward from `DISTILLATION-CONFORMANCE-AUDIT-2026-04-04`
- [x] FEEL authority confirmed against `echozero/ui/FEEL.py`
- [x] Local application/API authority paths confirmed under `echozero/application/timeline/*`
- [x] Missing local upstream references called out explicitly for absent distillation and UX docs
- [ ] Active-take truth leak removed in implementation
- [ ] Provenance/freshness/manual-modified inspector contract implemented end-to-end
- [ ] Real-world DAW-reference behavior examples captured in dedicated repo docs or fixtures
- [ ] Inspector panel shell implemented and verified
- [ ] Single-selection bounded editing implemented and accepted
- [ ] Multi-selection batch-safe workflow implemented and accepted

## Operating Rule

When conflict appears, resolve it in this order:

1. distillation intent
2. first principles
3. application/API contract
4. UX contract
5. FEEL constants
6. DAW precedent
7. audits and historical notes

Do not start from backlog convenience and reason upward. Start from truth, then contract, then feel, then implementation.
