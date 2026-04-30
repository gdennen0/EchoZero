# MA3 Batch Transfer UX Spec (Socratic + First Principles)

Status: reference
Last reviewed: 2026-04-30



## 1) Why this exists

Manual push/pull works for single operations. The next UX jump is handling **multiple layers/tracks in one deliberate transfer session** without turning the workflow into form-entry fatigue.

This spec defines a clean, operator-first batch flow:
- **Push** stays in the editor as a mode.
- **Pull** uses a dedicated timeline-style workspace/modal.
- Both feed a shared **Transfer Plan** with preview + explicit apply.

---

## 2) Product constraints (locked)

1. Manual transfer remains primary.
2. Live sync remains experimental/off by default.
3. Pull always requires explicit EZ destination layer (no implicit target).
4. Diff confirmation is required before destructive apply.
5. No silent destructive clear behavior.
6. MA3 write path remains main-only and metadata-safe.

---

## 3) Socratic seminar (decision pass)

### Q1. What if push and pull both use popups?
**Answer:** This creates unnecessary context switching and weak direct manipulation.

**Decision:**
- Push = in-editor mode.
- Pull = modal workspace (external source browsing requires it).

### Q2. What if we allow batch apply without preflight?
**Answer:** Fast but unsafe; partial failures become unpredictable.

**Decision:**
- Always run plan-wide preflight before apply.
- Show row-level failures and block destructive apply until fixed.

### Q3. What if every row asks for confirmation?
**Answer:** Too noisy, destroys flow.

**Decision:**
- One confirmation per batch apply (after diff preview).
- Row-level warnings in preview table.

### Q4. What if we auto-guess pull destinations?
**Answer:** Feels magical until wrong, then catastrophic.

**Decision:**
- Suggestion allowed, but explicit selection required per row.

### Q5. What if we keep selection interactions different in push vs pull?
**Answer:** Cognitive friction.

**Decision:**
- Unified timeline grammar: click, shift-click, cmd/ctrl-click, drag-lasso.

### Q6. What if we optimize for power users only?
**Answer:** Expert speed rises, approachability collapses.

**Decision:**
- Progressive disclosure:
  - simple defaults for first pass
  - advanced controls hidden behind expanders

---

## 4) First-principles UX goals

1. **Stay in context** (push in editor).
2. **Direct manipulation over forms** (timeline selection, not checkboxes).
3. **One obvious next action** per stage.
4. **Safe by default, fast when intentional**.
5. **Scale from one transfer to many** without a separate mental model.

---

## 5) Experience architecture

## A) Object Info: Sync & Transfer card (per selected layer)

Add a dedicated card:
- Status chip: Off / Observe / Paused / Armed Write
- Mapping summary (if known)
- Actions:
  - Push to MA3…
  - Pull from MA3…
  - Batch Transfer… (opens Transfer Plan workspace)

This is the canonical launch point for sync/transfer actions.

## B) Push Mode (editor-native)

Entering Push Mode does not open a heavy dialog.

UI:
- subtle mode banner: “Push to MA3”
- target-mapping strip (selected layers -> MA3 tracks)
- bottom action bar:
  - Cancel
  - Preview Diff
  - Send N Transfers

Selection uses existing timeline behavior.

## C) Pull Workspace (timeline modal)

Because source is MA3, pull uses a dedicated workspace/modal:
- Left: MA3 track browser (multi-select)
- Center: MA3 event timeline lanes (timeline-style selection)
- Right: EZ destination mapping per selected source track
- Footer:
  - Cancel
  - Preview Import Diff
  - Import N Transfers

Default import mode: **new take**.

---

## 6) Transfer Plan model (shared)

A Transfer Plan is a list of rows.

```text
TransferPlan
- plan_id
- operation_type: push | pull | mixed
- rows: TransferRow[]
- preflight_summary
- diff_summary
```

```text
TransferRow
- row_id
- direction: push | pull
- source_layer_id? (push)
- source_ma3_coord? (pull)
- selected_event_ids[]
- target_ma3_coord? (push)
- target_layer_id? (pull)
- import_mode (pull, default=new_take)
- status: draft | ready | blocked | applied | failed
- issues[]
- preview_diff
```

---

## 7) Interaction model

## Push (multi-layer)
1. User multi-selects layers/events in editor.
2. Enters Push Mode.
3. Assigns MA3 target per source layer (auto-suggest + editable).
4. Can refine event selection with timeline gestures.
5. Preview All.
6. Apply All.

## Pull (multi-track)
1. User opens Pull Workspace.
2. Multi-selects source MA3 tracks.
3. Selects events in source timeline lanes.
4. Sets explicit EZ destination per source track.
5. Preview All.
6. Import All.

---

## 8) Safety + execution semantics

1. **Preflight-first**:
   - validate each row: source selection exists, destination set, mapping valid, metadata constraints.
2. **Diff gate**:
   - plan-wide summary + per-row details.
3. **Apply semantics**:
   - row-by-row execution with progress.
   - no implicit destructive clear behavior.
4. **Failure handling**:
   - failed rows remain actionable with reason.
   - successful rows remain committed.
5. **Auditability**:
   - transfer_id per row + provenance stamps.

---

## 9) UI copy and states (baseline)

- Banner: “Push Mode — Select events and assign MA3 targets”
- Pull title: “Import from MA3”
- Primary actions:
  - Preview Diff
  - Send N Transfers / Import N Transfers

States:
- Draft
- Ready
- Blocked (reason shown inline)
- Applied
- Failed (recoverable)

---

## 10) Commands/intents (proposed)

- `EnterPushMode(layer_ids)`
- `ExitPushMode()`
- `SetPushTargetForLayer(layer_id, target_ma3_coord)`
- `OpenPullWorkspace()`
- `SelectPullSourceTracks(coords)`
- `SetPullTargetForTrack(source_coord, target_layer_id)`
- `BuildTransferPlanFromSelection(...)`
- `PreviewTransferPlan(plan_id)`
- `ApplyTransferPlan(plan_id)`
- `CancelTransferPlan(plan_id)`

---

## 11) Acceptance criteria

1. User can push multiple layer selections in one plan/apply.
2. User can pull multiple MA3 track selections in one plan/apply.
3. Pull rows require explicit EZ target layer before becoming ready.
4. Unified selection gestures work the same in push mode and pull workspace.
5. Plan-wide preflight + diff gate runs before apply.
6. Apply execution reports row-level success/failure deterministically.
7. Existing single push/pull and sync receive lanes remain green.

---

## 12) Implementation slices (recommended)

### Slice A — Information architecture
- Add Sync & Transfer card in Object Info.
- Add Transfer Plan domain/presentation models.

### Slice B — Push Mode
- Editor-native push mode + mapping strip.
- Build plan rows from timeline selection.

### Slice C — Pull Workspace
- MA3 timeline selection modal.
- Explicit per-row target mapping.

### Slice D — Batch preflight/diff/apply
- Plan-level diff + row table.
- Execution progress + deterministic failure handling.

### Slice E — polish
- presets/saved mappings
- keyboard polish
- micro-interactions and copy pass

---

## 13) Out of scope

- Full live bidirectional sync as default workflow.
- Automatic pull destination without explicit confirmation.
- Silent auto-merge behavior.
- Hidden destructive operations.
