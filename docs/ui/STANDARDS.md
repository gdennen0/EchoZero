# UI Standards

Status: reference
Last reviewed: 2026-04-30



This document is the canonical UI standards layer for EchoZero.
It is optimized for two audiences at once:

- humans making product and architecture decisions
- agents implementing UI work without creating drift

This file holds the product/UI standards.
Inventories and ownership details live in the companion files under `docs/ui/`.

## How To Use This

- Read this file for product and UI principles.
- Read the inventory files for canonical patterns.
- Read `OWNERSHIP-MAP.md` before changing files.
- Use `CHANGE-CHECKLIST.md` for meaningful UI or architecture work.

If implementation changes a canonical UI pattern, update the corresponding
inventory in the same task or PR.

## 1. Product Outcome Standard

The UI is not successful because it is layered cleanly or styled well.
It is successful when an operator can understand the current state quickly,
act with confidence, recover safely, and continue working without friction.

Every major surface must optimize for:

- comprehension
- confidence
- speed
- reversibility
- strong defaults
- predictable behavior
- low cognitive overhead

Clean outcomes should be built into the system, not assembled ad hoc in view
code.

Repeated behavior should be expressed through shared contracts, reusable
primitives, strong defaults, and bounded configuration.

Configuration may tune behavior, presentation, or workflow scope.
It must not create parallel truth models, hidden semantic branches, or
surface-specific exceptions that weaken consistency.

The default experience should already be good.
Configuration should refine the product, not rescue it.

Before adding controls, workflows, or abstractions, define:

- what decision the user is making
- what action the user is taking
- what feedback proves the action worked
- how the user recovers if it was the wrong action
- what default behavior should feel correct without setup
- what part of the behavior is reusable elsewhere in the product

UI work should be judged by operator outcomes, not only by architectural
cleanliness or visual polish.

A surface should reduce effort, not merely redistribute complexity.

## 2. UI State Standards

Every meaningful surface must explicitly define and render its states.
No screen should improvise state behavior ad hoc.

State logic must remain simple.
State ownership must remain explicit.
Application and UI state must preserve clean boundaries from engine state.

Required state categories:

- empty
- loading
- ready
- busy
- success
- partial
- stale
- disabled
- warning
- error
- syncing
- live
- conflict

Core rules:

- Each state must have a visible presentation and a behavioral contract.
- A user should be able to tell what state the surface is in without inference.
- State should be owned once at the correct layer and observed everywhere else.
- The UI must observe state, not recreate or reinterpret it locally.
- Dependent information should update automatically from shared state or derived
  presentation models.
- Repeated manual update wiring across widgets is a design failure and should be
  removed.
- Derived state is preferred over duplicated state.
- If a property changes in one object, every dependent surface should update
  from the same canonical source or from typed derived state built from that
  source.
- Important state must be explicit and typed, not inferred from labels, titles,
  badges, or visual heuristics.

Behavioral rules:

- Busy states must indicate what is working, what remains interactive, and
  whether the action can be cancelled.
- Loading states must preserve orientation where possible and avoid unnecessary
  visual instability.
- Success states should confirm completion without becoming noisy.
- Partial states must show what succeeded, what failed, and what still requires
  attention.
- Stale states must indicate what source changed and what consequence that has.
- Disabled states must communicate why the control is unavailable.
- Warning states must explain risk before action is taken.
- Error states must explain cause when known and next step when possible.
- Conflict states must never be silent.
- Syncing and live states must be unmistakable when they affect authority,
  timing, or safety.

Architectural rules:

- Engine state, application state, session state, and presentation state must
  have clear ownership and explicit translation boundaries.
- Store minimal canonical state and derive dependent presentation from it.
- Do not keep multiple mutable copies of the same semantic state across UI
  surfaces.
- State transitions for major workflows should be simple enough to enumerate,
  reason about, and test.
- When a surface depends on another object's state, it should subscribe to the
  shared application or presentation contract, not rely on widget-to-widget
  coordination.

## 3. Interaction Standards

Interaction rules must be standardized across the product.
Users should not have to relearn selection, editing, confirmation, or
navigation behavior per surface.

Interaction design must remain predictable, efficient, and reversible.
Complexity should come from the work itself, not from inconsistent controls.

Core principles:

- similar interactions must behave the same across the product
- high-frequency actions must feel fast and low-friction
- direct manipulation should feel clear and trustworthy
- the system should continuously communicate consequence during interaction
- reversible actions are preferred over defensive friction
- keyboard and pointer paths should both be intentional

Canonical interaction domains must be defined for:

- single selection
- additive selection
- range selection
- focus
- active target
- playback target
- armed target
- live target
- drag
- resize
- reorder
- duplicate
- nudge
- confirm
- cancel
- undo
- redo
- hover
- context action
- focus restore

Rules:

- The same selection gesture should produce the same semantic result across
  comparable surfaces.
- Selection, focus, and active-target state must never be ambiguous.
- A user should always know what object they are acting on, what object is
  merely selected, and what object is currently active.
- Drag interactions must communicate target, validity, and consequence
  continuously while dragging.
- Resize and timing interactions must provide precise feedback during
  manipulation.
- High-frequency editing operations should avoid unnecessary dialogs.
- Prefer undoable action flows over confirmation dialogs when technically
  feasible.
- If confirmation is required, the scope and consequence must be explicit
  before execution.
- Cancel behavior must be predictable and should return the user to a stable
  prior state.
- After an action completes, focus should remain predictable and support
  continued work.
- Keyboard accelerators should support expert workflows without creating hidden
  essential behavior.
- Pointer interactions should remain discoverable and not depend on hover-only
  knowledge.
- Context menus should expose secondary actions, not hide the primary workflow.

Governance rules:

- The product must maintain a canonical interaction inventory.
- Reuse an existing interaction pattern before creating a new one.
- If a new interaction is required, define its trigger, target, feedback, state
  impact, cancellation behavior, keyboard behavior, and recovery path.
- New interaction types must be added to the canonical inventory when they are
  introduced, not later.
- The reusable primitive library should keep growing as patterns stabilize.

## 4. Surface Responsibility Standards

Each surface type must have a narrow, documented responsibility.
This keeps the product understandable, prevents shell bloat, and stops
workflow logic from spreading into whichever view was easiest to patch.

A surface should have one primary job.
If it is doing multiple jobs, the boundary is probably wrong.

Modern surface model:

- navigation tells the user where they are
- workspace is where primary work happens
- inspector explains and edits the current selection
- toolbar and transport expose high-frequency global actions and modes
- status and monitoring show system condition, progress, sync, and warnings
- dialogs handle bounded decisions and temporary flows
- inline action areas handle local object actions near the object they affect

Rules:

- If a control changes the selected object directly, it likely belongs in the
  workspace or inspector.
- If a control changes global mode, playback, sync, or session-wide behavior,
  it likely belongs in a toolbar, transport, or status region.
- Primary action should live with the object being acted on.
- Secondary detail should move to the inspector, not crowd the workspace.
- Global system state should remain visible without hijacking the workspace.
- Dialogs must remain bounded.
- Inspectors must not become catch-all workflow engines.
- Status regions must report system condition, not replace primary editing
  flows.
- Navigation should expose structure and location, not duplicate editing
  controls.

Boundary rules:

- A surface may present information from other layers, but it must not absorb
  their responsibility.
- EchoZero-specific truth and workflow semantics must remain in application
  contracts and orchestration, not in the surface definition itself.
- If two surfaces can both affect the same object, one must be primary and the
  other supportive.
- Repeated surface patterns should be promoted into reusable primitives once
  they stabilize.
- If a new surface type or recurring sub-surface pattern is introduced, it must
  be added to the canonical surface inventory and primitive library.

## 5. Progressive Disclosure Standard

Complexity must appear intentionally and in layers, not all at once.
The product should reveal what is necessary for the current task while
preserving access to deeper capability when it becomes relevant.

Progressive disclosure is not about hiding power.
It is about keeping the interface understandable, reducing noise, and exposing
complexity at the right moment.

Rules:

- Default surfaces should prioritize the most common and highest-value actions
  first.
- Essential state, authority, warnings, and consequences must never be hidden
  behind disclosure.
- Most additional capability should be one layer away.
- A second layer of depth is acceptable for advanced or application-wide
  configuration when the structure remains clear.
- Right-click and secondary menus are valid for contextual power, but must not
  hide essential workflow.
- Actions should be self-describing through naming, grouping, and consequence.
- Pipeline and configuration-heavy surfaces should expose strong defaults first,
  then progressively reveal more detail in structured layers.

Depth model:

- primary layer: essential state, common actions, high-confidence controls
- secondary layer: contextual configuration and object-level advanced actions
- tertiary layer: infrequent, advanced, or system-wide configuration

## 6. Product Language Standard

Terminology must be stable across architecture, UI, docs, tests, and
automation.
The product should use one clear language for its objects, actions, states, and
workflows.

Core principles:

- one core concept should have one canonical name
- one action should use one canonical verb set
- labels should describe system reality, not internal implementation
- language should reduce ambiguity, not add personality at the expense of
  clarity
- the same concept should be named the same way everywhere it appears

Rules:

- Labels, badges, inspector fields, menus, dialogs, tooltips, docs, tests, and
  automation references must use the same canonical terms.
- Operational language is preferred over marketing language.
- Warning copy must explain risk or consequence.
- Error copy must explain cause when known and next step when possible.
- Status copy must describe what is true right now.
- Empty-state copy must explain what this area is for and what the user can do
  next.

Canonical terms belong in `GLOSSARY.md` first.
UI-specific usage rules belong here.

## 7. Feedback and Recovery Standard

Every meaningful action must produce clear feedback.
Every risky action must have a recovery path.

The product should never leave the user guessing whether something happened,
what changed, what failed, or what to do next.

Core principles:

- actions must acknowledge user intent clearly
- important outcomes must be visible
- background work must remain understandable while it is happening
- failure must be explainable
- recovery must be designed up front, not patched in later
- reversible systems are preferable to defensive friction

Rules:

- The UI must show what changed, whether it succeeded, and whether more work is
  required.
- Long-running work must expose owner, progress, phase when known, and final
  result.
- Feedback should appear close to the affected object when possible.
- Global system feedback should appear in a stable shell location when the
  impact is broader than one object.
- Prefer undo over confirmation dialogs when technically feasible.
- If undo is not feasible, consequence and scope must be explicit before
  execution.
- Retry should exist when failure is recoverable.
- Cancel should return the user to a stable and understandable state.

## 8. Consistency Standard

The same concept should look, behave, and be named the same way across the
product.
Consistency is operational predictability, not aesthetic repetition.

Rules:

- Identical concepts must keep consistent naming, placement, iconography, color
  meaning, keyboard strategy, interaction pattern, and feedback pattern.
- If a surface intentionally breaks consistency, the reason must be explicit
  and documented.
- Reuse existing patterns before inventing new ones.
- A second implementation of the same concept requires justification.

## 9. Performance as UX Standard

Performance is a product behavior standard, not just an engineering metric.
The UI must feel responsive, trustworthy, and controlled.

Rules:

- Continuous interactions should feel immediate.
- User intent should be acknowledged immediately, even if work continues in the
  background.
- Expensive rendering and recomputation must stay out of interaction hot paths.
- The product should stay usable during background work whenever possible.
- Perf-sensitive surfaces require explicit guardrails and profiling discipline.

## 10. Error Prevention Standard

The best error is the one the user never gets the chance to make accidentally.
The product should prevent invalid actions early, explain risk clearly, and
recover safely when prevention fails.

Rules:

- Invalid actions should be blocked as early as possible at the application
  contract layer and reflected clearly in the UI.
- Dangerous actions must show target, scope, and consequence before commit.
- Sync-affecting, destructive, or authority-changing actions require stronger
  guardrails than routine edits.
- The UI should make valid actions obvious and invalid actions understandable.

## 11. Learnability and Discoverability Standard

Power-user software must still be learnable.
A user should be able to understand what a surface is for, what is selected,
and what the next step is without relying on tribal knowledge.

Rules:

- First-run and empty states should explain purpose and next step.
- Important actions should be discoverable without requiring prior instruction.
- Tooltips, labels, helper text, and grouping should reduce onboarding cost.
- Hidden power features are acceptable.
- Hidden essential workflow is not.

## 12. Reusable UI Library Standard

Reusable building blocks must be treated as first-class product architecture.
The library should grow as real patterns stabilize.

Library-appropriate:

- FEEL constants
- tokens
- QSS/style builders
- geometry helpers
- time-axis math
- paint/layout primitives
- generic interaction helpers
- neutral inspector primitives
- canonical labels and summaries

Not library-appropriate:

- EchoZero truth semantics
- MA3 authority rules
- project storage wiring
- take/main business policy
- app-specific transfer policy
- application-owned orchestration decisions

Rules:

- The reusable layer may render and route.
- The application layer decides truth and consequences.
- Existing reusable seams should be promoted to canonical status before
  inventing new ones.
- New stabilized patterns must be added to the primitive library in the same
  task or PR that introduces them.

## 13. Architectural Responsibility Matrix

Responsibilities must be explicit.
No layer should absorb another layer's job because it was convenient.

Domain:

- invariants
- core entities
- semantic correctness

Engine/core:

- execution mechanics
- pipeline processing
- staleness mechanics
- app-agnostic processing behavior

Application:

- intents
- orchestration
- side effects
- policy
- truth mapping
- workflow consequences

Presentation contract:

- typed UI-facing models
- action contracts
- selection and state shaping
- derived display-ready structures

UI:

- rendering
- event capture
- focus and input behavior
- local visual feedback

Runtime shell:

- composition root
- service wiring
- bootstrapping
- app-specific runtime assembly

## 14. PR and Audit Enforcement Standard

The standards only matter if they are enforced at the moment the product
changes.
Enforcement must be lightweight enough to use every time and strong enough to
stop drift.

For meaningful UI or architecture work, require:

- named proof lane
- ownership statement
- state impact summary
- interaction impact summary
- consistency check
- risk summary
- evidence
- inventory update when canonical patterns changed

Rules:

- New product patterns must be captured immediately in the canonical inventory
  for their domain.
- If implementation changes the product language, interaction model, surface
  model, or primitive library, the standard must change in the same task or PR.
- Review should reject work that adds local heuristics where shared contracts or
  primitives should exist.
- Standards drift is a real regression.
