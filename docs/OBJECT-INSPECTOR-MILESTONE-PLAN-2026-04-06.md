# Object Inspector Milestone Plan - 2026-04-06

## Scope

This document defines the delivery scaffold for the object inspector panel in the `panel-object-inspector` worktree.

The milestone covers:

- a dockable inspector surface tied to current UI layout persistence rules
- selection-driven detail rendering for timeline objects
- a stable read/write contract between selection state, inspector view state, and edit intents
- phased rollout from read-only inspection to bounded editing and multi-select workflows

This milestone does not assume a broad timeline rewrite. It should compose with the current timeline shell, selection model, and panel layout behavior already documented in architecture decisions.

## UI Contract Summary

The object inspector should behave as a deterministic side panel with three primary states:

- no object selected: show active layer summary and neutral empty-state guidance
- one object selected: show object-specific fields, status, provenance, and bounded edit controls
- multiple objects selected: show batch-safe fields, mixed-value handling, and explicit bulk actions only where semantics are clear

The initial contract should preserve these rules:

- selection is the only authority for which object the inspector renders
- inspector fields read from application truth, not widget-local caches
- edits dispatch application intents rather than mutating presentation state directly
- unsupported fields remain visible as read-only or hidden; they do not imply fake editability
- panel open/close and layout persistence follow the existing per-user window layout rules

## Phased Rollout v0-v2

### v0

Goal:
- establish the panel shell and read-only rendering contract

Scope:
- dockable object inspector panel scaffold
- empty state, active-layer fallback, and single-selection read-only detail view
- stable selection-to-inspector binding
- coverage for panel visibility, state transitions, and layout persistence wiring

Exit condition:
- selecting and deselecting supported objects updates the inspector deterministically without write actions

### v1

Goal:
- enable bounded single-object editing on fields already backed by application intents

Scope:
- single-selection editable fields with validation and undo-safe intent dispatch
- read-only display for provenance, status, and metadata payloads
- explicit handling for unsupported object types and unavailable fields
- tests covering edit commit, cancel, validation failure, and selection changes during editing

Exit condition:
- one selected object can be inspected and edited through the panel using application-backed intents only

### v2

Goal:
- extend the inspector to batch and advanced review workflows without breaking the single-object path

Scope:
- multi-selection summary state with mixed-value handling
- batch-safe actions for shared fields only
- richer metadata and provenance presentation
- follow-on support for object-specific affordances such as clip preview or bulk offsets where already supported by the application layer

Exit condition:
- the inspector supports no-selection, single-selection, and multi-selection states with clear behavior and test coverage
