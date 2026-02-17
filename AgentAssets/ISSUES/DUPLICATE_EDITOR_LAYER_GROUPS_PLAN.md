# Duplicate Editor Layer Groups Plan

## Purpose
Maintain a single, living plan that captures the complete restoration strategy for duplicate editor layer groups, along with future learnings and decisions.

## Context
The "Add Editor Layer" dialog can show duplicate groups/layers due to stale `editor_layers` UI state entries that no longer map to current `EventDataItem` IDs.

## Goals
- Eliminate duplicate groups/layers in the dialog.
- Use a single source of truth for layer grouping (`EventDataItem` + `EventLayer`).
- Retain UI presentation overrides without allowing stale groups to persist.

## Non-Goals
- Redesign the Editor timeline UI.
- Change MA3 sync semantics beyond preserving `tc_` groups.
- Alter event data structures beyond what is required for grouping correctness.

## Current Understanding
- `editor_layers` UI state can contain stale `group_id` values.
- Dialog list is built from UI state, not from current `EventDataItem` IDs.
- Duplicate groups are often the same `group_name` with different `group_id`s.

## Best Part Is No Part Strategy
Remove `editor_layers` as a source of truth for dialog content. Build the dialog list from current `EventDataItem` data and use `editor_layers` only for presentation overrides keyed by `(EventDataItem.id, layer_name)`.

## Proposed Architecture (Target State)
1. **Source of truth**: `EventDataItem.id` and `EventLayer.name`.
2. **Presentation overrides**: `editor_layers` stores height, color, visibility, locked state per `(group_id, layer_name)`.
3. **Dialog list**: derived solely from `EventDataItem` instances and their layers.
4. **Stale cleanup**: any UI state entries whose `group_id` does not match a current `EventDataItem.id` are dropped.
5. **Sync groups**: preserve `tc_` and MA3 sync entries.

## Implementation Outline
### Phase 1: Data Source Consolidation
- Adjust dialog list construction to read from `EventDataItem` and `EventLayer`.
- Ensure `group_id` equals `EventDataItem.id` and is stable.
- Remove reliance on `editor_layers` for listing.

### Phase 2: UI Overrides
- Retain `editor_layers` only for display properties.
- Apply overrides when a matching `(group_id, layer_name)` exists.
- Do not create new groups solely from UI state.

### Phase 3: Cleanup + Guardrails
- Drop stale `editor_layers` entries on load.
- Skip saving `editor_layers` when no EventDataItems are loaded.
- Add validations: UI state entries must reference existing EventDataItem IDs.

## Verification
- Dialog list matches current EventDataItems and has no duplicate groups.
- Reopen dialog multiple times with no changes and verify list stability.
- Pull data, re-open dialog, confirm no reappearance of stale groups.

## Open Questions
- Should legacy `group_name`-only layers be auto-migrated to `EventDataItem.id`?
- Are there any non-EventDataItem groups (besides `tc_`) that need preservation?

## Notes Log (Append Only)
Add new findings and decisions below, newest first.

- 2026-01-26: Created plan document. Initial target is removing `editor_layers` as a list source.
