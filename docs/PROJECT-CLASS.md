# EchoZero 2 — Project Class API Reference

`echozero.project.Project`

The `Project` class is the single entry point for all application operations.
The UI talks to `Project`. `Project` talks to everything else.

---

## Overview

```
                    ┌─────────────────┐
                    │     Project     │  ← UI talks here
                    └────────┬────────┘
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
      ┌──────────┐   ┌──────────────┐   ┌──────────────┐
      │  Graph   │   │ProjectStorage│   │ Orchestrator │
      └──────────┘   └──────────────┘   └──────────────┘
            │                │
     ┌──────┴──────┐   ┌─────┴──────┐
     │  Pipeline   │   │   Repos    │
     │ Coordinator │   │(song/take/…│
     └─────────────┘   └────────────┘
```

---

## Factory Methods

These are the **only** ways to create a `Project`. Never call `__init__` directly.

---

### `Project.create(name, settings=None, executors=None, registry=None, working_dir_root=None)`

Create a brand new project.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Project name |
| `settings` | `ProjectSettingsRecord \| None` | Global settings (sample rate, BPM, etc.) |
| `executors` | `dict[str, BlockExecutor] \| None` | Block type → executor map. Use `None` in tests with mocks. |
| `registry` | `PipelineRegistry \| None` | Template registry. Defaults to global singleton. |
| `working_dir_root` | `Path \| None` | Override for `~/.echozero/working/` |

**Returns:** `Project`

**Example:**
```python
from echozero.main import create_project  # with real processors
project = create_project(name="Summer Tour 2026")

# Or bare (for tests):
from echozero.project import Project
project = Project.create(name="Test Project")
```

---

### `Project.open(ez_path, executors=None, registry=None, working_dir_root=None)`

Open an existing project from an `.ez` archive.

If a stale working directory exists (crash recovery scenario), opens it directly without unpacking.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ez_path` | `Path` | Path to the `.ez` file |
| `executors` | `dict[str, BlockExecutor] \| None` | See `create()` |
| `registry` | `PipelineRegistry \| None` | See `create()` |
| `working_dir_root` | `Path \| None` | Override working dir root |

**Returns:** `Project`

**Raises:** `FileNotFoundError` if `.ez` file not found.

---

### `Project.open_db(working_dir, executors=None, registry=None)`

Open directly from a working directory (recovery or dev use).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `working_dir` | `Path` | Path to directory containing `project.db` |
| `executors` | `dict[str, BlockExecutor] \| None` | See `create()` |
| `registry` | `PipelineRegistry \| None` | See `create()` |

**Returns:** `Project`

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Project name (from storage) |
| `graph` | `Graph` | The live block graph (read-only reference) |
| `event_bus` | `EventBus` | Pub/sub bus for domain events |
| `storage` | `ProjectStorage` | Direct access to persistence layer |
| `is_executing` | `bool` | Whether a pipeline run is in progress |
| `is_dirty` | `bool` | Whether there are unsaved changes |
| `stale_tracker` | `StaleTracker` | Query WHY blocks are stale (for UI staleness display) |
| `songs` | `SongRepository` | Shortcut to `storage.songs` |
| `song_versions` | `SongVersionRepository` | Shortcut to `storage.song_versions` |
| `layers` | `LayerRepository` | Shortcut to `storage.layers` |
| `takes` | `TakeRepository` | Shortcut to `storage.takes` |
| `pipeline_configs` | `PipelineConfigRepository` | Shortcut to `storage.pipeline_configs` |

---

## Graph Mutations

### `dispatch(command) → Result[Any]`

Dispatch a command to mutate the graph. This is **the only way** to mutate the graph.

On success:
- Applies the command via the Pipeline handler
- Persists the updated graph to SQLite
- Publishes domain events (via EventBus)
- The Coordinator reacts to events and propagates staleness

On failure:
- Restores the graph to pre-command state
- No events published
- Returns `Err` with the exception

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `Command` | Any registered command (see below) |

**Returns:** `Result[Any]` — `Ok(value)` on success, `Err(exception)` on failure

**Available commands** (from `echozero.editor.commands`):

| Command | Description |
|---------|-------------|
| `AddBlockCommand` | Add a block to the graph |
| `RemoveBlockCommand` | Remove a block and all its connections |
| `AddConnectionCommand` | Wire two block ports together |
| `RemoveConnectionCommand` | Remove a connection |
| `ChangeBlockSettingsCommand` | Update a single setting on a block |

**Example:**
```python
from echozero.editor.commands import ChangeBlockSettingsCommand

result = project.dispatch(ChangeBlockSettingsCommand(
    block_id="DetectOnsets_1",
    setting_key="threshold",
    new_value=0.5,
))

if result.is_ok():
    print("Setting updated")
else:
    print(f"Error: {result.error}")
```

---

## Execution

### `run(target=None) → Result[str]`

Execute the pipeline synchronously. Blocks until complete.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `target` | `str \| None` | Block ID to run up to (inclusive). None = run all. |

**Returns:** `Result[str]` — `Ok(execution_id)` on success, `Err` on failure.

**Thread safety:** Call from the main thread only. Use `run_async()` from the UI.

---

### `run_async(target=None) → Result[ExecutionHandle]`

Execute the pipeline in a background thread. Returns immediately.

**Parameters:** Same as `run()`.

**Returns:** `Result[ExecutionHandle]`

`ExecutionHandle` has:
- `.done` — `bool`, whether execution has finished
- `.cancel()` — signal cancellation
- `.result(timeout=None)` — block until done, returns `Result[str]`
- `.execution_id` — the execution ID (available immediately)

**Example:**
```python
handle_result = project.run_async()
if handle_result.is_ok():
    handle = handle_result.unwrap()
    # Poll from Qt timer or await in async context
    if handle.done:
        result = handle.result()
```

---

### `cancel() → None`

Signal cancellation to any in-flight execution. Non-blocking.
The running execution will stop at the next cancellation checkpoint.

---

## Analysis

### `analyze(song_version_id, template_id, knob_overrides=None, on_progress=None) → Result[AnalysisResult]`

Run a full analysis pipeline against a song version and persist results.

Creates a `PipelineConfigRecord` from the template, then executes it.
Resulting events are persisted as `LayerRecord` + `Take` entries.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `song_version_id` | `str` | ID of the `SongVersionRecord` to analyze |
| `template_id` | `str` | Registered pipeline template ID |
| `knob_overrides` | `dict[str, Any] \| None` | Override default knob values |
| `on_progress` | `Callable[[str, float], None] \| None` | Progress callback: `(message, 0.0–1.0)` |

**Returns:** `Result[AnalysisResult]`

`AnalysisResult` fields:
- `song_version_id: str`
- `pipeline_id: str`
- `layer_ids: list[str]` — created/updated layer IDs
- `take_ids: list[str]` — created take IDs
- `duration_ms: float`

**Example:**
```python
def on_progress(msg, pct):
    print(f"[{pct:.0%}] {msg}")

result = project.analyze(
    song_version_id=version.id,
    template_id="onset_detection",
    knob_overrides={"threshold": 0.3},
    on_progress=on_progress,
)
```

---

### `execute_config(config_id, on_progress=None) → Result[AnalysisResult]`

Execute an existing `PipelineConfigRecord` directly.

Useful for re-running after the user has tweaked knobs in the UI.
The config already contains the full graph — no template rebuild needed.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_id` | `str` | ID of the `PipelineConfigRecord` |
| `on_progress` | `Callable \| None` | Progress callback |

**Returns:** `Result[AnalysisResult]`

---

## Song Management

### `import_song(title, audio_source, artist='', label='Original', default_templates=None, scan_fn=None) → tuple[SongRecord, SongVersionRecord]`

Import an audio file as a new song.

Creates:
- `SongRecord`
- `SongVersionRecord` (copies audio, scans metadata, stores hash)
- Default `PipelineConfigRecord`s from registered templates

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Song title |
| `audio_source` | `Path` | Path to audio file (wav, mp3, flac, etc.) |
| `artist` | `str` | Artist name (optional) |
| `label` | `str` | Version label (default "Original") |
| `default_templates` | `list[str] \| None` | Template IDs for auto-config creation. `None` = all registered. `[]` = none. |
| `scan_fn` | `callable \| None` | Injectable for testing (skips real audio scan) |

**Returns:** `(SongRecord, SongVersionRecord)`

**Raises:** `ValidationError` if audio file is invalid or unreadable.

---

### `add_song_version(song_id, audio_source, label=None, activate=True, scan_fn=None) → SongVersionRecord`

Add a new version of an existing song (e.g., festival edit, radio edit).

Copies all `PipelineConfigRecord`s from the current active version → new version
(same graph, same knobs, new IDs). Optionally sets new version as active.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `song_id` | `str` | Existing `SongRecord` ID |
| `audio_source` | `Path` | Path to new audio file |
| `label` | `str \| None` | Version label. Auto-generated (`"v2"`, `"v3"`, …) if `None`. |
| `activate` | `bool` | Set as active version (default `True`) |

**Returns:** `SongVersionRecord`

**Raises:** `ValueError` if song not found or has no active version.

---

## Lifecycle

### `save() → None`

Flush current graph and all pending DB changes. Clears the dirty flag.
Autosave already commits every 30 seconds — this is for explicit user saves.

---

### `save_as(ez_path) → None`

Save the project as a `.ez` archive. Performs a WAL checkpoint before packing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ez_path` | `Path` | Destination `.ez` file path |

---

### `close() → None`

Close the project gracefully:
- Unsubscribes Coordinator from the EventBus
- Commits any uncommitted changes
- Stops autosave timer
- Releases lockfile
- Closes SQLite connection

**Does NOT delete the working directory** — crash recovery relies on it persisting.

---

### Context manager

`Project` implements `__enter__`/`__exit__`, calling `close()` on exit:

```python
with Project.open(Path("show.ez")) as project:
    # ... work ...
# automatically closed here
```

---

## Thread Safety

| Method | Thread safety |
|--------|--------------|
| `dispatch()` | **Main thread only** — graph mutations are not thread-safe |
| `run()` | **Main thread only** — blocks until complete |
| `run_async()` | Safe to call from main thread; execution runs in background thread |
| `cancel()` | Thread-safe — sets a threading.Event |
| `save()` / `close()` | Main thread only |
| Repository accessors (`songs`, `takes`, …) | Thread-safe (SQLite with RLock) |
| `is_dirty` / `is_executing` | Thread-safe reads |

For PyQt6 integration, call `run_async()` and poll `handle.done` via a `QTimer`.
Never call `run()` from the Qt main thread — it will block the UI.

---

## Event Subscriptions (for UI)

The `project.event_bus` publishes domain events the UI should wire to:

| Event | When | Fields |
|-------|------|--------|
| `BlockAddedEvent` | After `AddBlockCommand` succeeds | `block_id`, `block_type` |
| `BlockRemovedEvent` | After `RemoveBlockCommand` succeeds | `block_id` |
| `ConnectionAddedEvent` | After `AddConnectionCommand` succeeds | `source_block_id`, `target_block_id` |
| `ConnectionRemovedEvent` | After `RemoveConnectionCommand` succeeds | `source_block_id`, `target_block_id` |
| `SettingsChangedEvent` | After `ChangeBlockSettingsCommand` succeeds | `block_id`, `setting_key`, `old_value`, `new_value` |

**Subscribe:**
```python
from echozero.domain.events import BlockAddedEvent

project.event_bus.subscribe(BlockAddedEvent, lambda evt: print(f"Block added: {evt.block_id}"))
```

**Unsubscribe:**
```python
project.event_bus.unsubscribe(BlockAddedEvent, my_handler)
```

All subscriptions on `project.event_bus` are automatically cleaned up when `project.close()` is called.

---

## Result Type

All mutation and execution methods return `Result[T]`:

```python
from echozero.result import is_ok, unwrap

result = project.run()
if is_ok(result):
    execution_id = unwrap(result)
else:
    error = result.error  # the exception
```

Or using the fluent API:
```python
result = project.run()
result.is_ok()          # bool
result.is_err()         # bool
result.unwrap()         # value or raises
result.unwrap_or(None)  # value or default
```
