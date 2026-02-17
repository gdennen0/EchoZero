# REFACTOR PROPOSAL: Progress Tracking System

**Command:** `@refactor` progress tracking system

**Date:** 2026-02-09

---

## QUESTIONS (Must Answer)

### 1. What concrete problem does this solve?

Three disconnected progress systems (ProgressTracker, ProgressEventStore, LoadingProgressTracker) with:
- A name collision (`ProgressContext` defined in two files with different meanings)
- A thread safety crash (Qt widgets updated from EventBus thread in SetlistProcessingDialog)
- Dual fallback paths in setlist processing (legacy callbacks running alongside new ProgressContext)
- Dead code (`OperationProgress` event, `BatchProgress` class duplicating `track_progress`)
- INFO-level logging in hot paths flooding logs (3 log lines per progress tick)

### 2. Is there a simpler fix?

No single fix covers it. The root cause is that the system grew organically: ProgressTracker was built for block-level progress, ProgressEventStore was added later for setlist-level progress, and LoadingProgressTracker was built separately for startup. They were never unified. The fallback pattern in SetlistService is a direct consequence of this.

### 3. Can we DELETE instead of reorganize?

Yes. Significant deletion opportunity:
- **DELETE** `BatchProgress` class (~65 lines) -- identical to `track_progress` function
- **DELETE** `OperationProgress` event (~20 lines) -- defined but never published or consumed
- **DELETE** `ProgressBar` backwards-compat alias (~3 lines) -- nothing uses it
- **DELETE** `report_progress_if_available` function (~15 lines) -- never called in codebase
- **DELETE** legacy `progress_callback` parameter chain through facade/service/thread (~50 lines of callback threading)
- **DELETE** fallback path in `SetlistProcessingDialog._update_verbose_info` -- define one path, not two
- **RENAME** `ProgressContext` dataclass in progress_tracker.py to `ProgressTrackerContext` to eliminate name collision

Net: ~170+ lines deleted, zero new functionality added.

### 4. Net complexity change?

**Reduction.** Three systems consolidated to two (ProgressTracker for block-level, ProgressEventStore for operation-level). Loading progress stays separate since it runs before the event system exists, but that is the correct boundary.

### 5. Real problem or imagined?

Real. The thread safety violation in SetlistProcessingDialog is a runtime crash. The name collision creates import confusion. The dual callback/context-manager path means every future change must update both paths or risk silent regression.

---

## RED FLAGS (Reject) -- None Apply

- [x] NOT "more flexible" -- we are removing flexibility (two paths become one)
- [x] NOT "cleaner" -- we are fixing a crash and deleting dead code
- [x] NOT "best practices" -- we are following the project's own stated rule against fallbacks
- [x] Concrete problems identified (crash, name collision, dead code, log flooding)
- [x] NOT "might need later" -- we are removing things, not adding speculative features

---

## GREEN FLAGS (May Proceed)

- [x] Pattern emerged 3+ times (three separate progress systems)
- [x] Clear bugs from current structure (thread safety crash, name collision)
- [x] Fewer lines after (~170+ lines removed)
- [x] Deletion opportunity (BatchProgress, OperationProgress, report_progress_if_available, ProgressBar alias)

---

## REFACTOR PLAN

### Phase 1: Fix the Crash (Critical -- Do First)

**REFACTOR: SetlistProcessingDialog thread safety**

```
Problem: _update_block_progress() modifies Qt widgets from EventBus thread

Change: Use QMetaObject.invokeMethod or a pyqtSignal to marshal the update to the main thread

Before: Direct widget update from event handler (crashes intermittently)
After: Signal-based update, same pattern as MainWindow._on_subprocess_progress

Net: +5 / -2 lines (add signal + slot, remove direct call)
Risk: Low -- follows established MainWindow pattern exactly
```

Files:
- `ui/qt_gui/dialogs/setlist_processing_dialog.py`
  - Add `_block_progress_signal = pyqtSignal(str, int, str)` to class
  - Connect signal to `_update_block_progress` in `__init__`
  - In `on_subprocess_progress` handler, emit signal instead of calling method directly

### Phase 2: Delete Dead Code

**REFACTOR: Remove duplicates and dead code**

```
Problem: BatchProgress duplicates track_progress; OperationProgress event is unused; 
         report_progress_if_available is uncalled; ProgressBar alias is unnecessary

Change: Delete them

Before: 8 progress files with redundant code
After: 8 files, ~170 fewer lines, no unused exports

Net: +0 / -170 lines
Risk: Low -- verify with grep that nothing imports the deleted symbols
```

Files:
- `src/features/execution/application/progress_helpers.py`
  - Delete `BatchProgress` class (lines 227-290)
  - Delete `report_progress_if_available` function (lines 310-329)
- `src/application/events/events.py`
  - Delete `OperationProgress` class (lines 201-221)
- `ui/qt_gui/core/progress_bar.py`
  - Delete `ProgressBar` alias class (lines 209-211)
- Update any `__init__.py` or `__all__` that exports these symbols

### Phase 3: Resolve Name Collision

**REFACTOR: Rename ProgressContext dataclass in progress_tracker.py**

```
Problem: Two classes named ProgressContext with different meanings

Change: Rename the dataclass in progress_tracker.py to ProgressTrackerContext

Before: Import confusion -- which ProgressContext?
After: Unambiguous names

Net: +0 / -0 lines (rename only)
Risk: Low -- search all imports of the dataclass version and update
```

Files:
- `src/features/execution/application/progress_tracker.py`
  - Rename `ProgressContext` to `ProgressTrackerContext`
  - Update all references within the file
- Verify no external imports of this dataclass (it is only used internally by `create_progress_tracker`)

### Phase 4: Eliminate Dual Callback Path in Setlist Processing

**REFACTOR: Remove legacy progress_callback from setlist chain**

```
Problem: SetlistService.process_setlist() runs both legacy callbacks AND ProgressContext,
         violating the "no fallbacks" rule

Change: Remove progress_callback parameter; ProgressContext is the single path.
        SetlistProcessingThread reads from ProgressEventStore or subscribes to events.

Before: Two parallel progress reporting mechanisms
After: One mechanism (ProgressEventStore via ProgressContext)

Net: +10 / -60 lines
Risk: Medium -- requires updating SetlistProcessingThread and SetlistProcessingDialog
       to consume ProgressEventStore instead of legacy callbacks for overall progress.
       Song-level and action-level callbacks can remain (they serve a different purpose:
       updating specific tree items, not overall progress).
```

Files:
- `src/features/setlists/application/setlist_service.py`
  - Remove `progress_callback` parameter from `process_setlist` and `process_song`
  - Remove all `if progress_callback: progress_callback(...)` blocks
- `src/application/api/application_facade.py`
  - Remove `progress_callback` parameter from `process_setlist` and `process_song`
- `ui/qt_gui/core/setlist_processing_thread.py`
  - Remove `progress_callback` from `self.facade.process_setlist()` call
  - Use ProgressEventStore callbacks or polling for overall progress instead
- `ui/qt_gui/dialogs/setlist_processing_dialog.py`
  - Remove fallback path in `_update_verbose_info` -- use store only
  - If store has no state, show "Waiting..." not a parallel tracking system

### Phase 5: Reduce Log Noise

**REFACTOR: Downgrade progress logging from INFO to DEBUG**

```
Problem: 3 INFO log lines per progress tick floods logs

Change: Change progress update logging to DEBUG level

Before: 3000+ INFO lines for a 1000-item batch
After: Same lines at DEBUG, visible only when debugging

Net: +0 / -0 lines (level change only)
Risk: None
```

Files:
- `src/features/execution/application/progress_tracker.py` line 172: `Log.info` -> `Log.debug`
- `ui/qt_gui/main_window.py` line 1288: `Log.info` -> `Log.debug`
- `ui/qt_gui/main_window.py` line 1324: `Log.info` -> `Log.debug`

### Phase 6: Fix Stale Documentation

**REFACTOR: Correct file paths in PROGRESS_TRACKING_SYSTEM.md**

```
Problem: Doc references src/application/services/ but files are in src/shared/application/services/

Change: Update paths to match reality

Before: Wrong paths mislead agents
After: Correct paths

Net: +0 / -0 lines
Risk: None
```

Files:
- `AgentAssets/modules/commands/feature/PROGRESS_TRACKING_SYSTEM.md` lines 10-13

---

## EXECUTION ORDER

| Phase | Priority | Risk | Effort | Dependencies |
|-------|----------|------|--------|--------------|
| 1. Fix crash | Critical | Low | 30 min | None |
| 2. Delete dead code | High | Low | 30 min | None |
| 3. Rename collision | High | Low | 15 min | None |
| 4. Eliminate dual path | Medium | Medium | 2-3 hr | Phases 1-3 |
| 5. Reduce log noise | Low | None | 5 min | None |
| 6. Fix docs | Low | None | 5 min | None |

Phases 1, 2, 3, 5, 6 are independent and can be done in any order or in parallel.
Phase 4 should be done last as it is the largest change and benefits from the cleanup in phases 1-3.

---

## RULES COMPLIANCE

- [x] Evidence of actual pain (crash, name collision, dead code, log flooding)
- [x] Prefer deletion (170+ lines removed)
- [x] Small incremental changes (6 independent phases)
- [x] Not refactoring while fixing bugs (crash fix is Phase 1, separated from refactoring)
- [x] Core value: "best part is no part" -- removing BatchProgress, OperationProgress, fallback paths
- [x] Core value: "simplicity and refinement" -- one progress path instead of two
- [x] No fallbacks -- defining exact workflow per AgentAssets README

---

## WHAT WE ARE NOT DOING

- NOT unifying ProgressTracker and ProgressEventStore into one system. They serve different scopes (block-level vs. operation-level) and that boundary is correct.
- NOT touching LoadingProgressTracker. It runs before the event system exists. Separate is correct.
- NOT adding new features (time estimation, metrics, export). Reduce first.
- NOT changing the block processor progress API. The helpers (progress_scope, track_progress, IncrementalProgress) are good and used correctly by 6+ block processors.

---

*Follows: AgentAssets/modules/commands/refactor/PRESET.md*
