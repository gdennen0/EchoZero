# EchoZero Full Codebase — Ship-Readiness Audit

**Date:** 2026-03-30
**Auditor:** Chonch
**Scope:** Everything EXCEPT `echozero/audio/` and `echozero/models/` (already audited and fixed)
**Baseline:** 1,405 tests passing, 8.44s
**Method:** File-by-file read of all 50+ source files → correctness analysis → what-if stress testing

---

## Rating Key

- 🔴 **Must fix before ship** — correctness bug, data loss, crash, or security risk
- 🟡 **Should fix** — edge case bug, reliability gap, or likely production footgun
- 🟢 **Nice to have** — polish, performance, future-proofing

---

## 1. Persistence Layer

### 🔴 P1: `unpack_ez()` is vulnerable to zip-slip (path traversal)

**File:** `persistence/archive.py:72`

```python
zf.extractall(working_dir)
```

`extractall()` with no member filtering trusts all filenames in the ZIP. A crafted `.ez` file can contain entries like `../../.bashrc` or `../../../etc/cron.d/evil`. Python 3.12+ warns about this, but doesn't block it by default. Since `.ez` files could come from shared projects or the internet, this is an actual attack surface.

**Fix:** Validate each member's path before extraction:
```python
for member in zf.namelist():
    target = (working_dir / member).resolve()
    if not target.is_relative_to(working_dir.resolve()):
        raise ValueError(f"Zip-slip detected: {member}")
zf.extractall(working_dir)
```

---

### 🔴 P2: `ProjectSession.open()` unpack + open is not atomic — partial unpack leaves broken state

**File:** `persistence/session.py:110`

If `unpack_ez()` succeeds partially (e.g., extracts `manifest.json` but crashes writing `project.db` due to disk full), the working directory exists with a partial state. On next `open()`, the code sees `(working_dir / "project.db").exists()` — if the DB was half-written, `sqlite3.connect` may succeed but `ProjectRepository.list()` crashes with `sqlite3.DatabaseError`. There is no cleanup or retry path.

**Fix:** Unpack to a temp directory first, then atomically rename:
```python
import tempfile
tmp_dir = Path(tempfile.mkdtemp(dir=root))
unpack_ez(ez_path, tmp_dir)
tmp_dir.rename(working_dir)
```
Or: catch exceptions in `open_db()` and offer recovery/delete options.

---

### 🔴 P3: `ProjectSession` DB connection is `check_same_thread=False` without external locking

**File:** `persistence/session.py:65, 97`

```python
conn = sqlite3.connect(str(db_path), check_same_thread=False)
```

The session has `self._lock = threading.RLock()` but only `transaction()`, `commit()`, `save()`, `save_as()`, `save_graph()`, `load_graph()`, `import_song()`, `add_song_version()` acquire it. The repository property accessors (`self.songs`, `self.layers`, etc.) return raw `Repository` objects that share the same connection, and **callers can use those repos from any thread without the lock**.

Example: Thread A calls `session.layers.list_by_version(v_id)` while Thread B calls `session.import_song(...)`. Both execute SQL on the same connection concurrently. SQLite in WAL mode allows concurrent readers, but concurrent writes crash with `sqlite3.OperationalError: database is locked`.

**Fix:** Either:
1. Wrap all repository operations in the session's lock (expensive), or
2. Use a connection-per-thread pool, or
3. Document clearly that repository methods must be called from the session's owning thread, and add `check_same_thread=True` to enforce it (then explicitly exempt the autosave timer)

---

### 🟡 P4: `_autosave_tick()` commits from a Timer thread — SQLite threading violation

**File:** `persistence/session.py:320-328`

```python
def _autosave_tick(self) -> None:
    try:
        if not self._closed and self.dirty_tracker.is_dirty():
            with self._lock:
                self.db.commit()
```

The autosave timer runs on a `threading.Timer` thread. `self.db.commit()` is called from this thread while the connection was opened with `check_same_thread=False`. This works in practice (SQLite WAL + the RLock serializes access), but if the main thread is mid-transaction (e.g., inside `import_song()`'s `with self._lock:` block), the Timer thread will **block on the lock** until the main thread finishes. This is correct but means autosave latency is unbounded — it waits however long the main thread takes.

More critically: if `_autosave_tick` executes between `session.db.execute(...)` and `session.db.commit()` on the main thread (i.e., the main thread does NOT hold the lock), autosave will commit partial work that was meant to be rolled back on error.

**Fix:** Autosave should only commit when the lock is free AND no transaction is in progress. Add an `_in_transaction` flag set by `transaction()` context manager.

---

### 🟡 P5: `save_as()` WAL checkpoint may fail silently

**File:** `persistence/session.py:166`

```python
self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```

If the WAL checkpoint fails (e.g., another connection has a read lock), the PRAGMA returns a row with status info but doesn't raise. The packed `.ez` file will contain a `project.db` that might not include the latest WAL contents. The data isn't lost (it's in the WAL file still), but the archive is incomplete.

**Fix:** Check the checkpoint result:
```python
row = self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
if row[0] != 0:  # 0 = success
    logger.warning("WAL checkpoint returned status %d — archive may be incomplete", row[0])
```

---

### 🟡 P6: Schema migration V1→V2 references `song_pipeline_configs` table that may not exist

**File:** `persistence/schema.py:111`

```python
rows = conn.execute(
    "SELECT id, song_version_id, pipeline_id, bindings, created_at "
    "FROM song_pipeline_configs"
).fetchall()
```

If a V1 database never had any `song_pipeline_configs` entries, this still works (returns empty list). But if a fresh database is created at V2 (no V1 table ever existed), and something triggers `_migrate_v1_to_v2()`, the `SELECT FROM song_pipeline_configs` will raise `sqlite3.OperationalError: no such table`.

This shouldn't happen in normal flow (fresh DBs get V2 directly), but could happen if schema version metadata gets corrupted (set to 1 manually).

**Fix:** Wrap in `try/except` or check `sqlite_master` for table existence.

---

### 🟡 P7: `import_audio()` uses first 16 chars of SHA-256 — collision risk

**File:** `persistence/audio.py:31`

```python
dest_name = f"{audio_hash[:16]}{source_path.suffix}"
```

16 hex chars = 64 bits of entropy. Birthday paradox collision probability hits 50% at ~2³² (4 billion) files. For a desktop app with hundreds of songs, the practical risk is negligible — but two different files with the same 16-char prefix would silently deduplicate to the wrong audio.

**Fix:** Use 32 chars (128 bits) or the full hash. The filename cost is minimal.

---

### 🟡 P8: `TakeRepository._from_row()` crashes on `data_json = NULL` — but schema allows it

**File:** `persistence/repositories/take.py:21`

```python
if row['data_json'] is None:
    raise PersistenceError("Take has no data")
```

The schema declares `data_json TEXT` with no `NOT NULL` constraint. A bug or manual edit could leave `data_json` as NULL. The current code raises `PersistenceError`, which is fine as a guard — but every call to `list_by_layer()` or `get_main()` would crash if any single take in the layer has NULL data. One bad row poisons the entire layer listing.

**Fix:** Skip bad rows with a warning instead of raising, or add `NOT NULL` to schema + migration.

---

### 🟢 P9: `pack_ez()` doesn't include WAL/SHM files

The `.ez` archive only packs `project.db`, not `project.db-wal` or `project.db-shm`. The `PRAGMA wal_checkpoint(TRUNCATE)` before packing is supposed to flush the WAL into the main DB file. If the checkpoint fails (P5), the WAL data is lost from the archive. This is a consequence of P5 — fix P5 and this is fine.

---

## 2. Execution Engine

### 🔴 E1: `ExecutionEngine.run()` catches raw `Exception` and wraps in `RuntimeError` — loses traceback

**File:** `execution.py:155`

```python
except Exception as exc:
    ...
    return err(RuntimeError(f"Executor raised: {exc}"))
```

The original exception's type and traceback are lost. A `FileNotFoundError` from a processor becomes a generic `RuntimeError("Executor raised: [Errno 2] No such file...")`. Callers can't programmatically distinguish error types, and the traceback is gone for debugging.

**Fix:** Chain the exception properly:
```python
return err(ExecutionError(f"Executor for '{block.block_type}' raised: {exc}") from exc)
```
Or better: store the original exception directly in Err.

---

### 🟡 E2: Multi-port detection heuristic can misfire

**File:** `execution.py:173-179`

```python
is_multi_port = (
    isinstance(result_value, dict)
    and len(output_port_names) > 1
    and result_value.keys() <= output_port_names
)
```

If a processor has 2 output ports named `"events_out"` and `"stats"`, and returns `{"events_out": data}` (only one key), `result_value.keys() <= output_port_names` is True. This is treated as multi-port, so `context.set_output(block_id, "events_out", data)` — correct. But `"stats"` has no output set, so downstream blocks connected to `"stats"` get `None` from `context.get_input()`. Silent data loss.

**Fix:** Log a warning for declared ports with no output data. Or require multi-port executors to return ALL declared ports (even if `None`).

---

### 🟡 E3: `ExecutionContext.get_input()` only finds the FIRST matching connection

**File:** `execution.py:51-58`

```python
for conn in self.graph.connections:
    if conn.target_block_id == block_id and conn.target_input_name == input_port_name:
        value = self._outputs.get(...)
        ...
        return value
return None
```

The loop returns on the first matching connection. If an EVENT-type input port has multiple connections (fan-in is allowed for EVENT ports — only AUDIO is restricted), only the first connection's data is returned. This is a silent data loss bug for fan-in scenarios.

**Fix:** For non-AUDIO ports, collect all connected values and return a list (or document that fan-in returns first-connected only).

---

### 🟡 E4: Cancel check only happens between blocks, not during block execution

**File:** `execution.py:141`

```python
if _cancel.is_set():
    return err(OperationCancelledError("Execution cancelled"))
```

If a processor takes 30 seconds (e.g., Demucs separation), cancellation only takes effect after that block finishes. The `cancel_event` is passed in `ExecutionContext` and individual processors CAN check it, but nothing enforces this. Separation could run for minutes with no cancellation possible.

**Fix:** Document that long-running processors MUST check `context.cancel_event.is_set()` periodically. Consider a wrapper/timeout.

---

## 3. Orchestrator / Services

### 🔴 O1: `Orchestrator.execute()` modifies persisted pipeline graph in-place

**File:** `services/orchestrator.py:142-149`

```python
for block_id, block in pipeline.graph.blocks.items():
    if block.block_type == "LoadAudio":
        new_settings = {**dict(block.settings), "file_path": song_version.audio_file}
        updated = _replace(block, settings=BlockSettings(new_settings))
        pipeline.graph.replace_block(updated)
```

The pipeline is deserialized from `config.to_pipeline()`, then its graph is mutated directly. If `execute()` is called twice on the same config, the second deserialization starts clean — so this is safe. But if anyone holds a reference to the deserialized pipeline object, they'd see the mutation. More importantly: `pipeline.graph.blocks.items()` iterates a `MappingProxyType` view over a dict that's being mutated via `replace_block()` inside the loop. In CPython this is technically safe (dict mutation during iteration over a proxy raises `RuntimeError` only if keys are added/removed, and `replace_block` only replaces values), but it's fragile.

**Fix:** Collect block IDs first, then iterate:
```python
load_audio_ids = [bid for bid, b in pipeline.graph.blocks.items() if b.block_type == "LoadAudio"]
for block_id in load_audio_ids:
    block = pipeline.graph.blocks[block_id]
    ...
```

---

### 🟡 O2: `Orchestrator._handle_persist_as_layer_take()` creates a take per domain Layer — but most pipelines return single-layer EventData

**File:** `services/orchestrator.py:250`

```python
for domain_layer in event_data.layers:
```

If a processor returns an `EventData` with 3 layers (e.g., drum onsets split by sub-type), this creates 3 separate `LayerRecord` entries + 3 Takes. This is correct but non-obvious behavior. The mapping from pipeline output → persistence is 1:N based on the `EventData.layers` content, not the pipeline output declaration. This could surprise template authors who declare one output but get multiple layers persisted.

**Fix:** Document this behavior clearly. Consider whether the output should map to one layer (with sub-layers) or multiple top-level layers.

---

### 🟡 O3: `analyze()` (legacy) duplicates 90% of `execute()` — maintenance risk

**File:** `services/orchestrator.py:176-240`

The legacy `analyze()` method duplicates the entire flow of `execute()` but starts from template + bindings instead of a persisted config. Any bug fix to `execute()` must be manually replicated in `analyze()`.

**Fix:** Have `analyze()` create a temporary PipelineConfig and delegate to `execute()`, or extract the shared logic into a private method.

---

## 4. Domain & Serialization

### 🔴 S1: `deserialize_graph()` loads all blocks as `STALE` — fresh project loads trigger full re-run

**File:** `serialization.py:98`

```python
state=BlockState.STALE,
```

Every time a graph is deserialized (project open, config load), all blocks are marked STALE. This means opening a project where all results are cached still shows everything as "stale" in the UI, and auto-evaluate would trigger a full re-run. The STALE state should be determined by whether cached results exist, not by deserialization.

**Fix:** Deserialize as `BlockState.FRESH` (or the serialized state), let the Coordinator/cache determine staleness.

---

### 🔴 S2: `deserialize_pipeline()` reaches into private `_graph` and `_outputs`

**File:** `serialization.py:40-47`

```python
pipeline._graph = graph
...
pipeline._outputs.append(PipelineOutput(out_data["name"], port_ref))
```

This bypasses the Pipeline's `add()` and `output()` methods, which means: no connection validation, no duplicate output name check, no block counter update. A malformed serialized pipeline would silently load invalid state.

**Fix:** Either use the public API (add blocks via `add()`, outputs via `output()`), or validate the deserialized state explicitly.

---

### 🟡 S3: `_handle_change_settings()` mutates `context.graph.blocks[...]` directly

**File:** `editor/pipeline.py:167`

```python
context.graph.blocks[command.block_id] = replace(
    block, settings=BlockSettings(new_entries)
)
```

This writes directly into the Graph's `_blocks` dict via the `MappingProxyType` — wait, no. `context.graph.blocks` returns a `MappingProxyType`, which is read-only. So `context.graph.blocks[command.block_id] = ...` would raise `TypeError`.

Actually looking closer: `context.graph` IS the raw Graph object (not the proxy). So `context.graph.blocks[...]` goes through the property which returns `MappingProxyType`. This means this line would RAISE at runtime.

**Wait — unless the handler is using `context.graph._blocks[...]` or `context.graph` has a different code path.** Let me re-read...

The `CommandContext.graph` is a `Graph` object. `Graph.blocks` returns `MappingProxyType`. Setting `graph.blocks[key] = value` WOULD raise `TypeError: 'mappingproxy' object does not support item assignment`.

This is a **bug that would crash** on any settings change. BUT — the test suite has 1,405 passing tests, including tests that change settings. So either: (a) this code path isn't tested, or (b) something else is going on.

Let me check: the `Coordinator._on_settings_changed` calls `propagate_stale()`, not `_handle_change_settings`. The `Pipeline.dispatch(ChangeBlockSettingsCommand)` calls `_handle_change_settings`. If no test dispatches a `ChangeBlockSettingsCommand` through the editor Pipeline, this bug exists but is latent.

**Fix:** Use `context.graph.replace_block(updated_block)` instead of direct dict assignment.

---

### 🟡 S4: `BlockSettings.__hash__` crashes on non-hashable values

**File:** `domain/types.py:128`

```python
def __hash__(self) -> int:
    return hash(tuple(sorted(self._data.items())))
```

If any setting value is a list, dict, or numpy array, `sorted()` raises `TypeError` (can't compare mixed types) or `hash(tuple(...))` raises `TypeError` (unhashable type in tuple). Since settings commonly contain lists (e.g., frequency bands) or dicts (e.g., classification config), this will crash whenever BlockSettings is used in a set or as a dict key.

**Fix:** Don't implement `__hash__`, or use a JSON-serialized string as the hash input.

---

### 🟡 S5: `PipelineConfig.with_knob_value()` deserializes + re-serializes graph on every knob change

**File:** `persistence/entities.py:150-185`

Each call to `with_knob_value()` does:
1. `json.loads(self.graph_json)` — parse full graph
2. `deserialize_graph(...)` — reconstruct all Block/Port objects
3. Iterate all blocks, update matching settings
4. `serialize_graph(graph)` — re-serialize everything
5. `json.dumps(...)` — back to JSON string

For a graph with 10 blocks and a settings panel with 5 knobs, changing all 5 does this cycle 5 times. `with_knob_values()` exists for batch updates but still does the full cycle once. The performance is ~0.09ms per call (measured in audit WI-D), so this is fine for V1.

**Not a bug, just worth noting** — the batch method exists and should be preferred.

---

### 🟡 S6: `save_project()` / `load_project()` exist alongside `ProjectSession` — two save paths

**File:** `serialization.py:276-296`

`save_project()` writes Graph + TakeLayers to a flat JSON file. `ProjectSession.save_as()` writes to a `.ez` ZIP archive with SQLite. Two completely different persistence formats for the same data. If someone calls `save_project()` instead of `session.save_as()`, they get an incompatible file.

**Fix:** Deprecate `save_project()`/`load_project()` or document they're for testing/debugging only.

---

## 5. Take System

### 🔴 T1: `TakeLayer.promote_to_main()` doesn't validate take is not archived

**File:** `takes.py:103`

A user could promote an archived take to main. An archived take is supposed to be "cold storage" — invisible in the timeline. Promoting it to main would make it the active take while still marked `is_archived=True`, violating the semantic meaning of both flags.

**Fix:** Add: `if take.is_archived: raise TakeLayerError("Cannot promote an archived take")`

---

### 🟡 T2: `merge_events()` additive strategy doesn't deduplicate

**File:** `takes.py:195`

```python
if strategy == "additive":
    return target_events + source_events
```

If the user merges the same take twice, they get duplicate events. An onset at t=1.5 would appear twice. For the "additive" strategy this is technically correct (keep_both), but practically confusing.

**Fix:** Document that additive is truly keep_both with no dedup. Consider a `deduplicate` option.

---

### 🟡 T3: `merge_take_into()` only merges the first layer

**File:** `takes.py:253`

```python
# For now: merge first layer of each. Multi-layer merge TBD.
source_events = source.data.layers[0].events
target_events = target.data.layers[0].events
```

If a take has multiple layers (e.g., separate kick/snare/hihat layers), only the first gets merged. The rest are silently ignored.

**Fix:** Either merge all layers or raise an error for multi-layer merges until the feature is built.

---

### 🟡 T4: `_has_time_match()` is O(n²) for subtract/intersect merges

**File:** `takes.py:224`

```python
def _has_time_match(time, candidates, epsilon):
    return any(abs(time - c) <= epsilon for c in candidates)
```

Called for every target event against all source events. For 10,000 events × 10,000 candidates = 100M comparisons. Onset detection can easily produce thousands of events.

**Fix:** Sort candidates, use `bisect` for O(n log n) instead of O(n²).

---

## 6. Editor Layer

### 🟡 ED1: `_handle_change_settings()` bypasses Graph's `replace_block()` API

**File:** `editor/pipeline.py:167` (same as S3)

```python
context.graph.blocks[command.block_id] = replace(...)
```

As discussed in S3, this writes through the MappingProxyType, which should raise `TypeError`. This is either dead code or a latent crash bug.

---

### 🟡 ED2: `Coordinator.request_run()` is synchronous — blocks the calling thread

**File:** `editor/coordinator.py:84`

```python
result = self._engine.run(plan, cancel_event=self._cancel_event)
```

If called from the UI thread (which it would be via `_on_settings_changed` → `auto_evaluate` → `request_run()`), the entire UI freezes during pipeline execution. Demucs separation takes 30+ seconds.

**Fix:** `request_run()` should dispatch to a worker thread. The auto_evaluate path especially must be async.

---

### 🟡 ED3: `ready_nodes()` recomputes adjacency from scratch every call

**File:** `editor/coordinator.py:28-45`

Every call to `ready_nodes()` rebuilds the dependency map by iterating all connections. For incremental evaluation (auto_evaluate mode), this is called repeatedly. With large graphs (20+ blocks), the overhead is negligible but wasteful.

**Fix:** Cache the adjacency map in the Coordinator and invalidate on connection changes.

---

### 🟡 ED4: `Coordinator.cancel()` sets the event but doesn't wait for execution to stop

**File:** `editor/coordinator.py:96`

```python
def cancel(self) -> None:
    self._cancel_event.set()
```

After `cancel()` returns, `self._executing` may still be True. If the caller immediately calls `request_run()`, both the old (cancelling) and new runs could be in progress simultaneously. The old run's results would be cached and then immediately overwritten by the new run, but there's a window where stale data from the cancelled run gets committed.

**Fix:** `cancel()` should wait until `self._executing` is False, or `request_run()` should wait for any in-progress run to finish.

---

## 7. Archive / ZIP Format

### 🔴 A1: Zip-slip vulnerability in `unpack_ez()`

Same as P1. Repeated here because it's the #1 security issue in the codebase.

---

### 🟡 A2: `pack_ez()` only packs files in `audio/` — ignores subdirectories

**File:** `persistence/archive.py:41-47`

```python
for audio_file in sorted(audio_dir.iterdir()):
    if audio_file.is_file():
```

Only direct children of `audio/` are packed. If future features create subdirectories (e.g., `audio/stems/`), those files are silently excluded from the archive.

**Fix:** Use `rglob('*')` or `walk()` to recursively include all audio files.

---

### 🟡 A3: `is_valid_ez()` only checks for `manifest.json` — doesn't validate `project.db` presence

A `.ez` file with `manifest.json` but no `project.db` passes `is_valid_ez()` but crashes on `open()`.

**Fix:** Also check for `project.db` in the ZIP.

---

## 8. Cross-Cutting Concerns

### 🔴 X1: No input validation on `Event.time` or `Event.duration` — negative values accepted

**File:** `domain/types.py:38-44`

```python
@dataclass(frozen=True)
class Event:
    id: str
    time: float
    duration: float
    ...
```

Events with `time=-1.0` or `duration=-5.0` are valid. A negative-time event would sort before all other events, confuse timeline rendering, and potentially cause buffer underruns in audio playback (seeking to negative position).

**Fix:** Add `__post_init__` validation:
```python
def __post_init__(self):
    if self.time < 0:
        raise ValueError(f"Event time must be >= 0, got {self.time}")
    if self.duration < 0:
        raise ValueError(f"Event duration must be >= 0, got {self.duration}")
```

---

### 🟡 X2: `EventBus` handlers swallow exceptions silently

**File:** `event_bus.py:54`

```python
except Exception as exc:
    logger.warning("EventBus: handler %r raised %r for %s", ...)
```

A broken handler logs a warning and continues. This is correct for resilience but means: if a DirtyTracker handler crashes, the project is never marked dirty, autosave never triggers, and the user loses work without any visible indication.

**Fix:** Add a callback for critical handler failures that the UI can subscribe to (e.g., show a "save manually" warning).

---

### 🟡 X3: `DirtyTracker._unsubscribe()` uses bare `except ValueError: pass`

**File:** `persistence/dirty.py:40`

```python
try:
    self._event_bus.unsubscribe(event_type, self._on_mutation)
except ValueError:
    pass
```

If `EventBus.unsubscribe()` raises something other than `ValueError` (e.g., `KeyError` if the event type was never registered), it would propagate up through `session.close()`, potentially preventing proper cleanup.

**Fix:** Catch `Exception` or at least `(ValueError, KeyError)`.

---

### 🟡 X4: `PipelineConfig.from_pipeline()` and `to_pipeline()` import inside methods

**File:** `persistence/entities.py` — 8 inline imports

```python
def with_knob_value(self, ...):
    import json
    from echozero.serialization import deserialize_graph, serialize_graph
    from echozero.domain.types import BlockSettings
    from dataclasses import replace as _replace
```

Every call to `with_knob_value()`, `with_block_setting()`, etc. does 4 inline imports. Python caches imports after the first call, but it's still ~4 dict lookups per call. More importantly, it's a code smell — the entity depends on serialization, which is a layer violation.

**Fix:** Move graph-manipulation methods to a service or helper class that can import normally. Keep `PipelineConfig` as a pure data entity.

---

## What-If Scenarios

### 🔴 WI-1: What if the user opens the same .ez file twice simultaneously?

`_working_dir_for_path()` returns the same directory for the same `.ez` path. Two `ProjectSession` instances would share the same `project.db` file. SQLite in WAL mode handles concurrent readers, but two writers will deadlock or raise `OperationalError: database is locked`. There's no file lock or PID check to prevent this.

**Fix:** Place a lockfile (e.g., `project.lock`) in the working directory on open. Check on open, clean up on close.

---

### 🔴 WI-2: What if the app crashes during `import_song()`?

`import_song()` does:
1. Create Song row → committed
2. Import audio → file copied
3. Create SongVersion → committed
4. Update Song.active_version_id → committed
5. Create PipelineConfigs → committed
6. `db.commit()` → final commit

Wait — actually, steps 1-5 all run under `with self._lock:` and the final `self.db.commit()` is at step 6. The intermediate operations (`self.songs.create()`, `self.song_versions.create()`, etc.) execute SQL but don't commit individually (repos never commit — Unit of Work pattern).

So if the app crashes between step 1 and step 6, the uncommitted SQL is rolled back by SQLite. The audio file IS already copied to disk (step 2), so you get an orphaned audio file but no orphaned DB rows. **This is correct behavior** — the orphan is harmless and could be cleaned up by a maintenance sweep.

**Verdict: Safe.** The Unit of Work pattern works here.

---

### 🟡 WI-3: What if a pipeline template is updated between config creation and execution?

`PipelineConfig.from_pipeline()` snapshots the full graph at creation time. When `execute()` runs, it deserializes from the stored `graph_json`, not the current template. So template updates don't affect existing configs — by design. But: there's no `template_version` field to track WHICH version of the template created this config. If a template changes to add a new port or remove a block type, old configs become incompatible but there's no way to detect this.

**Fix:** Add `template_version` to PipelineConfig for future migration support.

---

### 🟡 WI-4: What if a processor returns the wrong type for its declared port?

If `DetectOnsets` is declared to output `EVENT` type but the processor returns an `AudioData`, the engine stores it happily. The Orchestrator's `_resolve_target()` checks the actual runtime type, not the declared type. So an `AudioData` on an event port would be routed to `song_version` persistence instead of `layer_take`. This is a silent type mismatch.

**Fix:** Add a type check in `ExecutionEngine.run()` that validates output types against declared port types.

---

### 🟡 WI-5: What if the working directory is on a network drive or synced folder (Dropbox/OneDrive)?

SQLite + WAL mode on network filesystems is **officially unsupported** and can cause silent corruption. The default `WORKING_DIR_ROOT = Path.home() / ".echozero" / "working"` is typically on a local drive, but if `$HOME` is redirected to a network path (common in enterprise environments), every project operation risks data loss.

**Fix:** Warn if `WORKING_DIR_ROOT` is on a network or synced filesystem. Consider `PRAGMA journal_mode=DELETE` fallback for non-local paths.

---

### 🟡 WI-6: What if `save_as()` target path is the same as the source `.ez` file being read from?

If a user opens `project.ez` and then `save_as("project.ez")`:
1. `pack_ez()` writes to `project.ez.tmp`
2. `project.ez.tmp.replace(project.ez)` atomically replaces the original

This is safe because `pack_ez()` reads from the working directory (not the `.ez` file), and the atomic replace ensures no partial write. **Safe.**

---

---

## Summary Table

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| P1 | 🔴 | Archive | Zip-slip path traversal in `unpack_ez()` |
| P2 | 🔴 | Session | Partial unpack leaves broken working directory |
| P3 | 🔴 | Session | DB connection shared across threads without proper locking |
| P4 | 🟡 | Session | Autosave can commit partial transactions |
| P5 | 🟡 | Session | WAL checkpoint failure silently produces incomplete archive |
| P6 | 🟡 | Schema | V1→V2 migration crashes if V1 table never existed |
| P7 | 🟡 | Audio | 16-char hash prefix — theoretical collision risk |
| P8 | 🟡 | Take repo | NULL `data_json` poisons entire layer listing |
| E1 | 🔴 | Execution | Exception wrapping loses type and traceback |
| E2 | 🟡 | Execution | Multi-port detection misses partial returns |
| E3 | 🟡 | Execution | Fan-in `get_input()` returns only first connection |
| E4 | 🟡 | Execution | Cancel only checked between blocks, not during |
| O1 | 🔴 | Orchestrator | Graph mutation during iteration (fragile) |
| O2 | 🟡 | Orchestrator | 1:N output-to-layer mapping non-obvious |
| O3 | 🟡 | Orchestrator | `analyze()` duplicates `execute()` — maintenance risk |
| S1 | 🔴 | Serialization | All deserialized blocks marked STALE — triggers unnecessary re-runs |
| S2 | 🔴 | Serialization | `deserialize_pipeline()` bypasses validation via private access |
| S3 | 🟡 | Editor | `_handle_change_settings()` writes through MappingProxyType — latent crash |
| S4 | 🟡 | Domain | `BlockSettings.__hash__` crashes on list/dict values |
| S5 | 🟡 | Entities | `with_knob_value()` full serialize/deserialize cycle per call |
| S6 | 🟡 | Serialization | Two incompatible save formats coexist |
| T1 | 🔴 | Takes | Can promote archived take to main |
| T2 | 🟡 | Takes | Additive merge doesn't deduplicate |
| T3 | 🟡 | Takes | Multi-layer merge only handles first layer |
| T4 | 🟡 | Takes | O(n²) event matching in subtract/intersect |
| ED1 | 🟡 | Editor | Settings change handler crashes on MappingProxyType write |
| ED2 | 🟡 | Editor | `request_run()` blocks calling thread (UI freeze) |
| ED3 | 🟡 | Editor | Adjacency recomputed from scratch every call |
| ED4 | 🟡 | Editor | `cancel()` doesn't wait for execution to stop |
| A1 | 🔴 | Archive | Zip-slip (same as P1) |
| A2 | 🟡 | Archive | Only packs direct children of audio/ |
| A3 | 🟡 | Archive | `is_valid_ez()` doesn't check for project.db |
| X1 | 🔴 | Domain | Negative Event.time and Event.duration accepted |
| X2 | 🟡 | EventBus | Critical handler failures swallowed silently |
| X3 | 🟡 | Dirty | `_unsubscribe()` catches only ValueError |
| X4 | 🟡 | Entities | Inline imports in hot methods — layer violation |
| WI-1 | 🔴 | Session | Same .ez opened twice → SQLite corruption |
| WI-3 | 🟡 | Config | No template_version tracking |
| WI-4 | 🟡 | Execution | No runtime type check on processor outputs |
| WI-5 | 🟡 | Session | Network filesystem → SQLite corruption risk |

**🔴 Must-fix count: 10** (deduplicated: P1=A1)
**🟡 Should-fix count: 25**
**🟢 Nice-to-have count: 1**

---

## Recommended Fix Priority (4 batches)

### Batch 1 — Security + Data Loss (~45 min)
P1/A1 (zip-slip), WI-1 (double-open lockfile), P2 (partial unpack), X1 (negative events)

### Batch 2 — Correctness (~45 min)
E1 (exception chaining), S1 (STALE on deserialize), S3/ED1 (MappingProxyType crash), T1 (archived promote), O1 (iteration mutation), S2 (private access bypass)

### Batch 3 — Robustness (~30 min)
P3/P4 (threading), P5 (WAL checkpoint), P6 (migration guard), P8 (NULL data_json), S4 (hash crash), ED4 (cancel wait), A2+A3 (archive completeness)

### Batch 4 — Quality (~30 min)
O3 (dedup analyze/execute), E2+E3 (multi-port + fan-in), T3+T4 (merge improvements), ED2 (async run), X2+X3 (error handling), remaining yellows

Total: ~2.5 hours of work, 4 commits.

---

*End of audit. Generated 2026-03-30.*
