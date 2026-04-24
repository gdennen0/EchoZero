# MA3 Manual Push/Pull — Structural Plan & Implementation Guide

Status: historical superseded plan
Last reviewed: 2026-04-23

This doc is kept as structural background only.
If any workflow detail below disagrees with `docs/STATUS.md` or `MA3/README.md`,
those current-truth docs win. That includes the current pull import rule:
new-layer targets import to `main`, existing-layer targets import as a new take.

Current repo truth now lives in:

- `docs/STATUS.md`
- `MA3/README.md`

Active push-lane planning now lives in:

- `docs/MA3-PUSH-V1-EXECUTION-PLAN.md`

Sequence-prep design reference now lives in:

- `docs/architecture/MA3-SEQUENCE-MANAGEMENT-DESIGN-2026-04-22.md`

## 1) Product Direction (Locked)

1. **Live Sync remains experimental** (off by default).
2. **Primary sync UX is manual transfer**:
   - Push EZ → MA3
   - Pull MA3 → EZ
3. **Diff confirmation is required** before destructive apply.
4. **Pull requires explicit EZ layer choice** every time (no implicit target).

---

## 2) Core User Flows

## A) Push selection to MA3
1. User selects EZ timeline events.
2. User clicks **Push to MA3**.
3. App opens **MA3 Track Picker** (track list + search/filter).
4. User picks target MA3 track.
5. App computes diff (EZ selection vs MA3 target state).
6. Diff modal opens with summary and details.
7. User confirms **Apply Push** (or cancels).
8. App sends events and records transfer metadata.

## B) Pull selection from MA3
1. User clicks **Pull from MA3**.
2. App opens **MA3 Browser** (tracks + events).
3. User selects MA3 events to import.
4. App opens **EZ Layer Picker** (mandatory).
5. User selects destination EZ layer.
6. App computes diff (incoming MA3 selection vs chosen EZ layer target state).
7. Diff modal opens with summary and details.
8. User confirms **Apply Pull** (or cancels).
9. App imports to chosen layer (default: new take in selected layer).

---

## 3) Structural Architecture

## 3.1 Application intents/commands

Add intent layer commands (timeline app boundary):
- `OpenPushToMA3Dialog(selection)`
- `ConfirmPushToMA3(targetTrackCoord, selectedEventIds)`
- `OpenPullFromMA3Dialog()`
- `ConfirmPullFromMA3(sourceTrackCoord, selectedMa3EventIds, targetLayerId, importMode)`

Add transfer commands:
- `PushEventsToMA3TrackCommand`
- `PullEventsFromMA3TrackCommand`
- `ApplyPulledEventsToLayerCommand`

## 3.2 Services

1. **MA3CatalogService**
   - list track groups
   - list tracks
   - list events per track

2. **MA3TransferService**
   - push selected EZ events to MA3 track
   - pull selected MA3 events

3. **SyncDiffService**
   - compute added/removed/changed counts
   - produce row-level diff detail model for modal

4. **LayerTargetResolverService**
   - enforces explicit EZ layer target for pull
   - rejects pull if no layer selected

## 3.3 Data contracts

### Pull import target contract
- Required: `target_layer_id`
- Optional: `import_mode`
  - default: `new_take`
  - future: `overwrite_main`, `merge_main`

### Provenance metadata on imported events
- `source = "ma3_manual_pull"`
- `ma3_coord = tc{n}_tg{n}_tr{n}`
- `ma3_event_index`
- `transfer_id`
- `imported_at`

### Provenance metadata on pushed events
- `source = "ez_manual_push"`
- `target_ma3_coord`
- `transfer_id`
- `pushed_at`

---

## 4) UI Components

1. **MA3TrackPickerDialog** (for push)
2. **MA3EventSelectionDialog** (for pull source)
3. **EZLayerPickerDialog** (mandatory for pull destination)
4. **SyncDiffDialog** (shared for push/pull confirmation)

### Diff dialog minimum content
- header: operation type + source/target
- counts: added / removed / modified / unchanged
- table: event time, label/name, action, before/after
- actions:
  - Cancel
  - Apply
  - Copy summary (optional)

---

## 5) Test Strategy (Simple + Deterministic)

## 5.1 Unit tests
- diff computation correctness (stable deterministic output)
- payload mapping EZ↔MA3 event schema
- pull requires `target_layer_id` (hard fail when absent)

## 5.2 Integration tests
- push flow intent → track picker result → diff → apply dispatch
- pull flow intent → MA3 select → layer picker required → diff → apply dispatch
- verify chosen layer receives imported events (default new take)

## 5.3 Existing sync receive lane (keep)
- `test_ma3_communication_service_protocol.py`
- `test_ma3_event_handler.py`
- `test_ma3_event_contract.py`
- `test_ma3_fixture_replay.py`

## 5.4 UI behavior tests
- pull action blocked until destination layer selected
- no silent auto-targeting for pull
- diff modal opens before destructive apply

---

## 6) Implementation Sequence

### Step 1 — Contracts first
- Add intents + transfer command contracts.
- Add pull target requirement (`target_layer_id`) and tests.

### Step 2 — Push UX
- Add Track Picker dialog and push plumbing.
- Add push diff modal gate.

### Step 3 — Pull UX
- Add MA3 event browser + **mandatory EZ layer picker**.
- Add pull diff modal gate.
- Apply pull into selected layer as default new take.

### Step 4 — Shared diff service
- centralize diff generation for both push/pull.
- ensure deterministic tests.

### Step 5 — Live Sync guardrail
- Live sync remains **experimental** and **off by default**.
- The primary sync CTA path remains **manual transfer only**:
  - Push EZ → MA3
  - Pull MA3 → EZ
- Any live-sync state other than `off` requires the experimental flag to be enabled first.
- Reconnect handling must downgrade `armed_write` to `paused`; the user must explicitly re-arm write mode after reconnect.
- Disabling the experimental flag must reset per-layer live-sync guardrail state back to the safe baseline.
- Entering `armed_write` requires explicit user confirmation in the UI before the state is applied.

### Next evolution — Batch transfer UX
- Multi-layer/multi-track transfer UX is specified in:
  - `docs/MA3-BATCH-TRANSFER-UX-SPEC.md`

---

## 7) Acceptance Criteria

Phase accepted when all are true:
1. User can push selected EZ events to chosen MA3 track.
2. User can pull selected MA3 events only after choosing destination EZ layer.
3. Both push and pull show diff modal before apply.
4. No destructive apply without explicit confirmation.
5. Test suites pass:
   - new manual push/pull unit + integration tests
   - existing sync receive lane

---

## 8) Out of Scope (for now)

- Full live bidirectional sync as primary workflow.
- Automatic layer targeting on pull.
- Advanced merge policies as default behavior.

Live sync remains experimental until manual transfer UX is stable and trusted.
