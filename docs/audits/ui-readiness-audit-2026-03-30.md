# UI Integration Readiness Audit
**Date:** 2026-03-30  
**Auditor:** Subagent (code-audit)  
**Scope:** All 74 Python files in `echozero/`  
**Purpose:** Assess readiness for building the PyQt6 Stage Zero Editor  

---

## Summary

| Category | Status | Notes |
|---|---|---|
| Import health | 🟢 GREEN | `from echozero.project import Project` → OK. No circular imports detected. |
| Project class API | 🟢 GREEN | Clean, complete, well-documented. Single entry point for all UI interactions. |
| Domain events | 🟢 GREEN | 8 typed events, all docstrung, all exported from `echozero.domain`. |
| Progress/runtime | 🟢 GREEN | `RuntimeBus` + `ProgressReport` are clean, typed, subscriber-safe. |
| Type hints | 🟡 YELLOW | `Project.import_song` uses `**kwargs` — signature opaque to UI. `on_progress: Any` vs typed `Callable`. |
| Documentation | 🟢 GREEN | Every public class and most methods have docstrings. Minor gaps noted below. |
| Thread safety | 🟡 YELLOW | `run()` is synchronous (blocks Qt main thread). `run_async()` exists but returns handle not a Qt-friendly signal. `analyze()`/`execute_config()` are blocking — must use `run_in_executor` or QThread. |
| Editor package | 🟢 GREEN | `echozero.editor` is purpose-built for the Stage Zero Editor. Coordinator + Pipeline + StaleTracker are all ready. |
| Audio engine | 🟢 GREEN | Lock-free, zero-alloc audio callback. `AudioEngine` is process-agnostic — ideal for UI process. |
| FEEL.py | 🟢 GREEN | All UI constants pre-defined. Nothing hardcoded elsewhere. |
| Knob metadata | 🟢 GREEN | `KnobWidget` enum + `Knob` dataclass give the UI everything needed to auto-generate inspector widgets. |
| Persistence | 🟢 GREEN | Full repository layer, typed entities, crash recovery, autosave. |
| Missing `__all__` exports | 🟡 YELLOW | `pipelines/__init__.py` has no `__all__`. `editor/__init__.py` is well-documented but doesn't export classes. |

---

## UI Contract: Project Class

**Factory methods** (call from main thread — I/O bound, consider `run_in_executor`):

| Method | Signature | Purpose | Thread Safety |
|---|---|---|---|
| `Project.create` | `(name: str, settings?, executors?, registry?, working_dir_root?) → Project` | Create new project | ⚠️ Creates DB + dirs on disk. Use QThread or executor. |
| `Project.open` | `(ez_path: Path, executors?, registry?, working_dir_root?) → Project` | Open .ez archive | ⚠️ Unpacks archive + opens DB. Use QThread or executor. |
| `Project.open_db` | `(working_dir: Path, executors?, registry?) → Project` | Open from working dir (recovery) | ⚠️ Opens DB. Use executor. |

**Properties** (safe from Qt main thread — no I/O):

| Property | Type | Purpose |
|---|---|---|
| `project.name` | `str` | Project name for title bar |
| `project.graph` | `Graph` | DAG of blocks and connections (read-only via `blocks: MappingProxyType`) |
| `project.event_bus` | `EventBus` | Subscribe to domain events for reactive UI |
| `project.storage` | `ProjectStorage` | Direct access to repos if needed |
| `project.is_executing` | `bool` | Whether a pipeline run is in progress |
| `project.is_dirty` | `bool` | Whether there are unsaved changes |
| `project.stale_tracker` | `StaleTracker` | Get stale reasons per block for UI tooltips |
| `project.songs` | `SongRepository` | CRUD for songs (the setlist) |
| `project.song_versions` | `SongVersionRepository` | CRUD for song versions |
| `project.layers` | `LayerRepository` | CRUD for timeline layers |
| `project.takes` | `TakeRepository` | CRUD for takes (pipeline result snapshots) |
| `project.pipeline_configs` | `PipelineConfigRepository` | CRUD for per-song pipeline configs |

**Mutation methods** (safe from main thread — in-memory only, fast):

| Method | Signature | Purpose | Thread Safety |
|---|---|---|---|
| `dispatch` | `(command: Command) → Result[Any]` | Mutate the graph (add/remove/connect blocks, change settings) | ✅ Main thread safe. Synchronous, in-memory. Also saves graph to DB. |

**Execution methods** (⚠️ must NOT block Qt main thread):

| Method | Signature | Purpose | Thread Safety |
|---|---|---|---|
| `run` | `(target?: str) → Result[str]` | Run pipeline synchronously. Returns execution_id. | ❌ **BLOCKS** — use `run_async()` or `run_in_executor` |
| `run_async` | `(target?: str) → Result[ExecutionHandle]` | Run in background thread. Returns handle immediately. | ✅ Non-blocking. Poll `handle.done` or `handle.result()` in worker. |
| `cancel` | `() → None` | Cancel in-flight execution | ✅ Main thread safe (sets threading.Event) |

**Analysis methods** (⚠️ must NOT block Qt main thread — CPU/disk intensive):

| Method | Signature | Purpose | Thread Safety |
|---|---|---|---|
| `analyze` | `(song_version_id: str, template_id: str, knob_overrides?, on_progress?) → Result[AnalysisResult]` | Run pipeline from template + persist results | ❌ **BLOCKS** — always wrap in QThread or executor |
| `execute_config` | `(config_id: str, on_progress?) → Result[AnalysisResult]` | Run pipeline from persisted config + persist results | ❌ **BLOCKS** — always wrap in QThread or executor |

**Song management** (⚠️ I/O — use executor):

| Method | Signature | Purpose | Thread Safety |
|---|---|---|---|
| `import_song` | `(**kwargs) → tuple[SongRecord, SongVersionRecord]` | Import audio as new song | ⚠️ I/O — use executor. **Signature uses `**kwargs` — see issues.** |
| `add_song_version` | `(**kwargs) → SongVersionRecord` | Add audio version to existing song | ⚠️ I/O — use executor. Same `**kwargs` issue. |

**Lifecycle** (use executor for open/close):

| Method | Signature | Purpose | Thread Safety |
|---|---|---|---|
| `save` | `() → None` | Flush and commit graph + all pending changes | ⚠️ DB write — prefer non-blocking via autosave |
| `save_as` | `(ez_path: Path) → None` | Save to .ez archive | ⚠️ I/O — use executor |
| `close` | `() → None` | Flush, stop autosave, release DB | ⚠️ Use executor on app exit |
| `__enter__` / `__exit__` | context manager | Use in `with` blocks | ✅ Convenience wrapper for close() |

---

## Domain Events (UI Should Subscribe)

Subscribe via `project.event_bus.subscribe(EventType, handler)`.

**All events are published on the thread that called `dispatch()`** — typically main thread for graph mutations. Be aware: if you dispatch from a worker thread (e.g., a background loader), events will fire on that thread. In Qt, use a signal/slot bridge.

| Event | Fields | When Fired | UI Action |
|---|---|---|---|
| `BlockAddedEvent` | `block_id, block_type` | After `AddBlockCommand` succeeds | Add block node to canvas |
| `BlockRemovedEvent` | `block_id` | After `RemoveBlockCommand` succeeds | Remove block node from canvas |
| `ConnectionAddedEvent` | `source_block_id, target_block_id` | After `AddConnectionCommand` succeeds | Draw connection wire |
| `ConnectionRemovedEvent` | `source_block_id, target_block_id` | After `RemoveConnectionCommand` succeeds | Remove connection wire |
| `BlockStateChangedEvent` | `block_id, old_state, new_state` | When block execution state changes (FRESH/STALE/ERROR) | Update block status badge |
| `SettingsChangedEvent` | `block_id, setting_key, old_value, new_value` | After `ChangeBlockSettingsCommand` succeeds | Update inspector panel |
| `ProjectLoadedEvent` | `project_id` | After project is loaded from disk | (Optional) refresh entire UI |
| `ProjectSavedEvent` | `project_id` | After project is saved | Clear dirty indicator |

**BlockState enum values** (for status badges):
- `BlockState.FRESH` — outputs are current
- `BlockState.STALE` — needs re-run (upstream changed)
- `BlockState.UPSTREAM_ERROR` — upstream block failed
- `BlockState.ERROR` — this block failed

---

## Progress / Callback Patterns

### RuntimeBus (execution progress — per block, sub-pipeline granularity)

Subscribe via `project._runtime_bus.subscribe(callback)` (or expose via `project.runtime_bus` — see recommendations). The RuntimeBus is separate from the EventBus.

**Report types:**

```python
from echozero.progress import ProgressReport, ExecutionStartedReport, ExecutionCompletedReport

def on_report(report):
    if isinstance(report, ExecutionStartedReport):
        # report.block_id, report.execution_id
        # Show "running" state on block
        pass
    elif isinstance(report, ProgressReport):
        # report.block_id, report.phase, report.percent (0.0–1.0), report.message
        # Update progress bar for block
        pass
    elif isinstance(report, ExecutionCompletedReport):
        # report.block_id, report.execution_id, report.success, report.error
        # Show success/error state on block
        pass

project._runtime_bus.subscribe(on_report)
```

⚠️ **RuntimeBus callbacks fire on the executor thread** (background thread). In Qt, **do not update widgets directly** — emit a Qt signal from the callback and let the signal handler update the UI on the main thread.

### on_progress (analysis progress — per-phase, higher level)

The `analyze()` and `execute_config()` methods accept an `on_progress` callback:

```python
def on_progress(message: str, fraction: float) -> None:
    # message: human-readable phase description
    # fraction: 0.0–1.0
    pass

project.analyze(song_version_id, template_id, on_progress=on_progress)
```

This callback fires on the same thread that called `analyze()`. If you wrap `analyze()` in a `QThread`, the callback fires on that thread — bridge to Qt signals as with RuntimeBus.

### Knob metadata (for auto-generating inspector widgets)

Templates expose fully typed `Knob` definitions. The UI can introspect these without needing to hard-code any UI:

```python
from echozero.pipelines.registry import get_registry
registry = get_registry()
template = registry.get("full_analysis")

for key, knob in template.knobs.items():
    # knob.widget: KnobWidget (SLIDER, DROPDOWN, TOGGLE, FILE_PICKER, etc.)
    # knob.label, knob.description, knob.min_value, knob.max_value
    # knob.options (for DROPDOWN), knob.units, knob.log_scale
    # knob.advanced (hide in basic mode), knob.hidden, knob.group
    pass
```

### StaleTracker (tooltip / stale reason UI)

```python
tracker = project.stale_tracker
# For a specific block:
reasons = tracker.get_reasons(block_id)  # tuple[StaleReason, ...]
tooltip = tracker.summary(block_id)      # "3 changes: ..." or None

# For global stale badge:
count = tracker.stale_count()            # int
all_stale = tracker.get_all_stale()     # dict[block_id, tuple[StaleReason, ...]]
```

### ExecutionHandle (async execution)

```python
result = project.run_async()
if isinstance(result, Ok):
    handle = result.value
    # handle.done: bool
    # handle.execution_id: str
    # handle.cancel(): signal cancellation
    # handle.result(timeout?): Result[str] — block until done or timeout
```

---

## Issues Found

### Critical (blocks UI work)

**C1: `Project.import_song` and `Project.add_song_version` use `**kwargs` — opaque to UI**

```python
# Current (in project.py):
def import_song(self, **kwargs) -> tuple[SongRecord, SongVersionRecord]:
def add_song_version(self, **kwargs) -> SongVersionRecord:
```

The UI developer cannot know the parameter names without digging into `ProjectStorage`. Both methods should have explicit, typed signatures matching `ProjectStorage.import_song`/`add_song_version`.

**C2: `project._runtime_bus` is private — no public accessor**

The UI needs to subscribe to `RuntimeBus` for per-block execution progress. Currently `_runtime_bus` is a private attribute. There is no public `runtime_bus` property on `Project`. UI developers will either access `project._runtime_bus` (fragile) or miss progress entirely.

**Fix:** Add `@property def runtime_bus(self) -> RuntimeBus` to `Project`.

---

### Moderate (should fix soon)

**M1: `on_progress` typed as `Any` instead of `Callable[[str, float], None]`**

In `Project.analyze()` and `Project.execute_config()`, `on_progress: Any = None` should be:
```python
on_progress: Callable[[str, float], None] | None = None
```
This matters for IDE autocomplete when wiring up a progress signal.

**M2: `Project.run()` (synchronous) is the first method listed — discoverability risk**

New UI developers will naturally call `project.run()` first. It blocks the main thread and will freeze the UI. The docstring should warn explicitly. Consider renaming to `run_sync` and making `run_async` the default `run`.

**M3: `executor` thread pool has `max_workers=1` — only one async execution at a time**

`Coordinator` uses `ThreadPoolExecutor(max_workers=1)`. Calling `run_async()` while one is running returns `Err(ExecutionError("Execution already in progress"))`. The UI must handle this case gracefully (disable Run button when `project.is_executing` is True).

**M4: EventBus handlers fire synchronously — Qt widget updates from non-main thread**

If the UI subscribes event handlers that update widgets, and dispatch is called from a non-main thread (e.g., during `import_song` in a worker), the handler fires on the worker thread. This will cause Qt thread violations. The UI needs a Qt signal/slot bridge for all EventBus handlers.

**M5: `PipelineConfigRecord.to_pipeline()` and `with_knob_value()` do JSON round-trips — not cached**

Every call to `with_knob_value()` / `with_block_setting()` does `json.loads + deserialize_graph + serialize_graph + json.dumps`. For a UI with a live "knob scrubber" (drag to update value), this will create lag. Consider debouncing or batching knob updates via `with_knob_values(updates={...})`.

**M6: No `AnalysisResult` progress signal for UI**

`execute_config()` returns `Result[AnalysisResult]` after completion. There is no async variant that yields intermediate results to the UI (layer IDs as they're created, for example). The UI must wait for full completion or use the `on_progress` string callback — no structured intermediate results.

---

### Minor (nice to have)

**m1: `pipelines/__init__.py` has no `__all__` and no public exports**

The `echozero/pipelines/__init__.py` file has only a blank docstring. The UI will need `Pipeline`, `PipelineTemplate`, `PipelineRegistry`, `Knob`, `KnobWidget` — none are importable from `echozero.pipelines` directly.

**m2: `editor/__init__.py` documents classes but doesn't export them**

`from echozero.editor import Coordinator` fails. Developers must `from echozero.editor.coordinator import Coordinator`. Not blocking, but inconsistent.

**m3: `PyTorchAudioClassifyProcessor` uses a dummy classifier (`_predict_event_class` returns hardcoded "kick"/"snare"/"hihat")**

The classification logic is a placeholder. Not a UI issue per se, but the UI should not show confidence scores as real data until this is replaced.

**m4: `ExportAudioDatasetProcessor._default_export_fn` raises `NotImplementedError`**

The default export implementation raises immediately. The UI must always provide a custom `export_fn` or catch `ExecutionError`. Consider adding a soundfile-based default.

**m5: `TranscribeNotesProcessor._default_transcribe` raises `NotImplementedError`**

Same as m4 — requires `basic-pitch` which is not bundled. UI should show a "not available" state for this block type.

**m6: `Graph.blocks` returns `MappingProxyType` but `Graph.connections` returns a plain `list[Connection]` (copy)**

Inconsistent — blocks is a proxy (reflects live state), connections is a snapshot (stale). UI code iterating connections right after a mutation may see the old state if it cached the list. Document this clearly.

**m7: `BlockStateChangedEvent` is intentionally excluded from `DirtyTracker`**

This is correct behavior, but the comment is only in `dirty.py`. UI devs implementing their own dirty tracking might miss this.

---

## Missing Documentation

Files/classes with missing or thin docstrings:

| File | Missing |
|---|---|
| `echozero/pipelines/__init__.py` | No module docstring, no exports |
| `echozero/pipelines/templates/onset_detection.py` | Module docstring present, but `build_onset_detection` function docstring is thin (1 line) |
| `echozero/models/provider.py` | `LocalFileSource.check_available` method docstring missing |
| `echozero/models/provider.py` | `import_os_sep` module-level helper has no docstring |
| `echozero/processors/pytorch_audio_classify.py` | `_create_model_from_config`, `_predict_event_class` are private but shape the public API surface — should document expected model format |
| `echozero/services/waveform.py` | Module-level function `generate_waveform_for_version` has a docstring but no return type annotation for the `None` case |

**All public classes are well-documented** — no class-level docstring gaps found.

---

## Recommendations (Prioritized)

### P0 — Do Before Writing Any UI Code

1. **Add `runtime_bus` property to `Project`** (fix C2)  
   ```python
   @property
   def runtime_bus(self) -> RuntimeBus:
       """Access the runtime bus for per-block execution progress."""
       return self._runtime_bus
   ```

2. **Expand `Project.import_song` and `Project.add_song_version` signatures** (fix C1)  
   Replace `**kwargs` with explicit typed parameters matching `ProjectStorage`.

### P1 — Before First UI Milestone

3. **Fix `on_progress: Any` → `Callable[[str, float], None] | None`** (fix M1)

4. **Add Qt signal bridge pattern to architecture notes**  
   Document the pattern for connecting `EventBus` handlers and `RuntimeBus` subscribers to Qt signals without causing cross-thread widget updates. Suggest a `QObject` bridge class approach.

5. **Add exports to `echozero/pipelines/__init__.py`** (fix m1)  
   At minimum: `Pipeline`, `PipelineTemplate`, `PipelineRegistry`, `Knob`, `KnobWidget`, `knob`, `get_registry`.

6. **Rename `run()` → `run_sync()` and make `run_async` the default `run`** (fix M2)  
   Or add a prominent `# BLOCKING` comment in the docstring: "This blocks the calling thread. Use `run_async()` from Qt."

### P2 — Before Beta

7. **Implement `soundfile`-based default for `ExportAudioDatasetProcessor`** (fix m4)

8. **Add a `Project.analyze_async()` variant** that returns a handle (like `run_async`) for cancellable analysis with progress.

9. **Document `Graph.connections` snapshot behavior** (fix m6) — add a note that `.connections` returns a copy, not a live view.

10. **Add `auto_evaluate` setter exposure on `Project`**  
    `Coordinator.auto_evaluate` is already implemented but not exposed on `Project`. The Stage Zero Editor may want to toggle this.

### P3 — Polish

11. **Real PyTorch classification implementation** (fix m3)  
    The dummy `_predict_event_class` should be replaced before any demo.

12. **Add `__all__` to `echozero/editor/__init__.py`** (fix m2)

---

## Architecture Notes for UI Developer

### The Golden Rule: Project is the UI's Only Door

```
UI
 └─ Project
     ├─ project.dispatch(command)           # mutate graph
     ├─ project.run_async()                 # execute pipeline
     ├─ project.analyze() / execute_config() # analyze song
     ├─ project.event_bus.subscribe(...)    # react to mutations
     ├─ project.runtime_bus.subscribe(...)  # react to execution progress
     ├─ project.songs / layers / takes      # data access
     └─ project.stale_tracker              # why is this block stale?
```

Never import `Graph`, `Pipeline`, `Coordinator`, or `Orchestrator` directly in UI code. Use `Project`.

### Event Flow

```
User Action → dispatch(command) → Pipeline handler → Graph mutation
    → EventBus.publish(event)
        → DirtyTracker marks dirty (autosave ticks)
        → Coordinator.propagate_stale (marks downstream STALE)
        → UI handlers (update canvas)
```

### Thread Model for Qt

```
Qt Main Thread:
  - All widget updates
  - EventBus subscriptions (if dispatched from main thread)
  - project.dispatch(), project.cancel()
  - Read project.is_executing, project.is_dirty

QThread / run_in_executor:
  - project.run()         → use run_async() instead
  - project.analyze()
  - project.execute_config()
  - project.import_song()
  - project.save_as()
  - Project.open() / Project.create()

RuntimeBus callbacks:
  - Fire on executor thread
  - Bridge to Qt: emit a signal from callback, connect to slot on main thread

EventBus callbacks (if dispatch happens on worker thread):
  - Fire on that worker thread
  - Same bridge pattern needed
```

### Knob → Widget Mapping

```python
KnobWidget.SLIDER      → QSlider (min, max, step, units)
KnobWidget.DROPDOWN    → QComboBox (options)
KnobWidget.TOGGLE      → QCheckBox
KnobWidget.TEXT        → QLineEdit
KnobWidget.FILE_PICKER → QLineEdit + QPushButton (browse)
KnobWidget.NUMBER      → QSpinBox / QDoubleSpinBox
KnobWidget.FREQUENCY   → Log-scale QSlider (20–20000 Hz)
KnobWidget.GAIN        → QSlider (-48 to +48 dB)
KnobWidget.MODEL_PICKER → Custom model browser widget
```

### Updating Pipeline Config from UI

```python
# Knob-level update (global — affects all blocks with this setting key)
config = config.with_knob_value("threshold", 0.4, knob_metadata=template.knobs)

# Block-level override (independent of knob — won't be overwritten by knob changes)
config = config.with_block_setting("detect_onsets_1", "threshold", 0.6)

# Clear per-block override → re-link to global knob
config = config.clear_block_override("detect_onsets_1", "threshold")

# Save updated config back to DB
project.pipeline_configs.update(config)
project.storage.commit()
```

### FEEL.py — Where All UI Constants Live

All pixel sizes, colors, thresholds, animation timings, and port/category colors are in `echozero/ui/FEEL.py`. Do not use magic numbers in UI code. Import from FEEL:

```python
from echozero.ui.FEEL import (
    BLOCK_WIDTH_PX, BLOCK_HEIGHT_PX,
    PORT_COLOR_AUDIO, PORT_COLOR_EVENT,
    BLOCK_COLOR_PROCESSOR,
    CLASSIFICATION_COLORS,
    FEEL_SNAP_MAGNETISM_RADIUS_PX,
    ...
)
```

---

*Import test result: `from echozero.project import Project` → **Import OK** (no circular imports, all dependencies resolve)*
