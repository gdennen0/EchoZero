# EchoZero Architecture Audit — Structural Alignment

**Date:** 2026-03-30
**Auditor:** Chonch
**Method:** First-principles comparison — Distillation (243 decisions) vs actual codebase
**Scope:** Does the code fit the architecture? Where do the seams not line up?

---

## Executive Summary

The engine layer (Graph, Pipeline builder, ExecutionEngine, Processors) is solid. Clean types, good separation, well-tested. The persistence layer (SQLite, repos, session, archive) is also solid.

**The problem is the middle.** The application layer that's supposed to bridge engine and persistence has structural gaps. Two disconnected worlds — an in-memory engine graph and a SQLite persistence store — with no consistent mechanism to keep them in sync. The command dispatch pattern exists but isn't wired to persistence. The UoW pattern from the distillation is absent. And several architectural concepts from the decisions haven't been expressed in code yet.

This audit identifies **7 structural misalignments** (seams that don't line up) and **5 missing abstractions** (things the distillation specifies that don't exist yet).

---

## S1: Two Disconnected Worlds — Engine Graph vs SQLite

**Severity: STRUCTURAL — must resolve before UI work**

The system has two truth stores that don't talk to each other:

1. **In-memory Graph** — owned by `editor.pipeline.Pipeline`, mutated by commands (AddBlock, ChangeSettings, etc.), cached in `ExecutionCache`, serialized to JSON
2. **SQLite database** — owned by `ProjectSession`, holds Projects, Songs, SongVersions, Layers, Takes, PipelineConfigs

**What connects them:** Only the `Orchestrator`. It builds an engine Pipeline from a PipelineConfig, executes it, then persists outputs as Layers+Takes in SQLite. That's it.

**What's missing:**
- When a user adds a block via `editor.Pipeline.dispatch(AddBlockCommand)`, it modifies the in-memory Graph. **Nothing writes it to SQLite.** The `projects` table has a `graph_json` column, but no code path syncs the editor's live graph to that column.
- When a project loads from SQLite, `graph_json` gets deserialized into a Graph. But there's no code path that hydrates the editor Pipeline's internal graph from the loaded data.
- The DirtyTracker listens to EventBus events (graph mutations) for autosave hints. But autosave calls `session.db.commit()` — which commits **SQLite rows**, not the in-memory graph. If the graph changed but nobody serialized it to `graph_json`, autosave commits nothing useful.

**The distillation says:**
> Save = close connection, zip database + audio into .ez
> Load = extract .ez, open SQLite database

This implies the Graph lives IN the SQLite database (as `graph_json`). The code has the column. But the write path doesn't exist.

**Fix direction:** The application layer needs a `ProjectCoordinator` (or similar) that:
1. On load: deserializes `graph_json` → hydrates editor Pipeline's graph
2. On graph mutation (via editor Pipeline events): serializes graph → writes `graph_json`
3. On save: ensures graph_json is current, then WAL checkpoint + pack_ez

This is the central wiring problem. Everything else downstream (autosave, dirty tracking, undo persistence) depends on this being solved.

---

## S2: No Unit of Work — Commands Bypass Persistence

**Severity: STRUCTURAL — architectural debt**

**The distillation says:**
> Command path: Correlation ID → Validate → UoW (begin → handler → commit → flush events) → Result

> Handlers receive a UoW context (DB connection + event collector). Never call commit/rollback themselves.

**What the code does:** `editor.Pipeline.dispatch()` creates a `CommandContext` with just the Graph and an event list. Handlers mutate the graph directly. On success, events flush to EventBus immediately. **No UoW, no transaction, no DB involvement.**

This is fine for an in-memory-only engine. But the distillation describes a UoW because commands are supposed to be transactional against the database. The editor's command dispatch is completely disconnected from persistence.

**Consequence:** There's no way to atomically:
1. Mutate the graph (in-memory)
2. Persist the mutation (SQLite)
3. Flush events (to UI)
...as one atomic operation. If step 2 fails, step 1 already happened and step 3 already fired. The distillation's "events discarded on rollback" guarantee doesn't hold.

**Fix direction:** Either:
- (A) Wrap `Pipeline.dispatch()` in a UoW that also serializes the graph to SQLite on success. Events flush after commit. Rollback reverts graph + discards events.
- (B) Keep the editor purely in-memory and treat SQLite sync as a separate concern (periodic flush). Simpler but loses atomicity guarantee.

Option (B) is probably the pragmatic V1 choice — the editor operates on the in-memory graph, and persistence is a "save" operation, not per-command. But this should be an explicit decision, not an accident.

---

## S3: Coordinator and Editor Pipeline Own Separate Graphs

**Severity: HIGH — latent wiring bug**

```python
# editor/pipeline.py
class Pipeline:
    def __init__(self, event_bus):
        self._graph = Graph()  # ← creates its own Graph

# editor/coordinator.py
class Coordinator:
    def __init__(self, graph, pipeline, engine, cache, ...):
        self._graph = graph  # ← receives a Graph
```

If the Coordinator is constructed with `Coordinator(graph=some_graph, pipeline=Pipeline(bus))`, they're operating on **different Graph instances**. Commands dispatched through the Pipeline mutate `pipeline._graph`. The Coordinator reads `self._graph`.

The tests work because they carefully wire the same graph instance. But there's no structural enforcement — it's a constructor wiring convention, not a contract.

**The distillation says:**
> One `Pipeline` routes commands/queries to handlers through a middleware chain. Every external interface is a thin adapter.

One Pipeline, one Graph. Not two objects that happen to share a reference.

**Fix direction:** Either:
- (A) Coordinator takes a Pipeline reference and reads `pipeline.graph` (single source of truth)
- (B) Pipeline takes a Graph in its constructor (instead of creating its own), enforcing that both use the same instance
- (C) Pipeline exposes a `graph` property that the Coordinator uses exclusively

Option (B) is cleanest — Pipeline becomes a stateless command router over an externally-owned Graph.

---

## S4: Block Categories Not Enforced at Execution

**Severity: MEDIUM — architectural gap**

**The distillation says three block categories with different execution semantics:**
1. **Processor** — pure transforms, executed by engine
2. **Workspace** — manage own data, manual pull via Take System, engine reads but doesn't execute
3. **Playback** — ephemeral, no persistence, no execution

**The code:** `BlockCategory` enum has `PROCESSOR`, `WORKSPACE`, `ANALYSIS`, `GENERATOR`, `EXPORTER`. The ExecutionEngine doesn't check category — it tries to execute every block in the plan. A Workspace block (like the Editor or ShowManager) would be executed like a Processor if it appeared in the plan.

Also: `ANALYSIS`, `GENERATOR`, `EXPORTER` aren't in the distillation. They're EZ1 holdovers.

**Fix direction:**
1. Align `BlockCategory` enum to distillation: `PROCESSOR`, `WORKSPACE`, `PLAYBACK`
2. `GraphPlanner.plan()` should skip Workspace and Playback blocks (they're terminal/ephemeral)
3. The Coordinator's `ready_nodes()` should also skip non-Processor blocks

---

## S5: No Copy-on-Pull — Mutable Upstream References

**Severity: MEDIUM — violates FP2**

**The distillation says:**
> Copy-on-pull: each block receives a COPY of upstream data, not a reference.
> Deleting or re-running an upstream block does NOT affect downstream blocks' existing data.
> Each block is an island — fully self-contained, portable unit of data.

**The code:** `ExecutionContext.get_input()` returns the raw cached value. No copy. If a downstream processor mutates the `EventData` object it receives (e.g., appending to a list, modifying event metadata), it modifies the upstream block's cached output.

This hasn't caused bugs yet because:
- `EventData` and `AudioData` are frozen dataclasses
- Event tuples are immutable
- Processors return new objects rather than mutating inputs

But `Event.metadata` is `dict[str, Any]` — mutable. And `Event.classifications` is `dict[str, Any]` — mutable. A processor that does `event.metadata["foo"] = "bar"` would mutate the upstream cache.

**Fix direction:** `ExecutionContext.get_input()` should `copy.deepcopy()` the value before returning it. Or: make metadata/classifications use `MappingProxyType` (read-only dict views).

---

## S6: Missing Two-Connection Pattern

**Severity: LOW — optimization, not correctness**

**The distillation says:**
> Two connections from day one: Write connection (protected by Lock, owned by UoW middleware), Read connection (used by queries, no lock needed)

**The code:** `ProjectSession` has one `self.db` connection. All reads and writes go through the same connection, protected by `self._lock`.

This works for V1. The two-connection pattern is an optimization for concurrent read/write access (UI queries shouldn't block behind write transactions). Can defer until the UI layer exists.

---

## S7: Repositories Are Stateful, Not Stateless

**Severity: LOW — style mismatch, not a bug**

**The distillation says:**
> Repositories: Stateless data access helpers. `save(conn, entity)`, `get(conn, id)`. Receive connection from UoW.

**The code:** `BaseRepository.__init__(self, conn)` stores `self._conn`. Repositories are bound to a connection for their lifetime. They're lightweight (no caching, no state beyond the connection ref), but they're not stateless in the distillation's sense.

This is fine for V1. The stateless pattern matters more when you have the two-connection pattern (S6) and need to pass different connections for reads vs writes.

---

## M1: Missing — Bootstrap / Wiring Module

**The distillation says:**
> Module-level wiring. Single `bootstrap.py` constructs entire object graph with explicit constructor calls. Two plain dictionaries: handlers and processors. No container, no locator, no decorators.

**The code:** No `bootstrap.py`. Each test and the Orchestrator wire things manually. The editor's Pipeline auto-registers default handlers in `_register_default_handlers()`.

**Why it matters:** Without a canonical wiring point, there's no single place to see how the system fits together. New code has to reverse-engineer the wiring from tests or the Orchestrator.

**Fix:** Create `echozero/bootstrap.py` that constructs the full object graph. Tests can use it or wire manually. The future UI's `main()` calls bootstrap.

---

## M2: Missing — Application Service Layer

**The distillation describes:**
- Orchestrator (runs pipelines, persists results) ✅ exists
- SetlistProcessor (batch processing) ✅ exists
- But also: project lifecycle (create, open, save, close) and the bridge between editor commands and persistence

**What's missing:** A service that coordinates:
1. Opening a project → loading graph from DB → hydrating editor
2. User edits (graph mutations) → persisting to DB
3. Saving → serializing graph → WAL checkpoint → pack_ez
4. The "load audio → set up pipeline → execute → persist results → update UI" flow as a coherent orchestration

The `ProjectSession` handles DB lifecycle but doesn't know about the editor. The Orchestrator handles execution→persistence but doesn't know about the editor. Nobody owns the full loop.

**Fix:** This is the same problem as S1. The solution is a `ProjectCoordinator` or `AppSession` that owns: the ProjectSession, the editor Pipeline, the Coordinator, and the Orchestrator — and wires them together.

---

## M3: Missing — Undo Stack

**The distillation says:**
> Global undo stack per project. Only editable commands pushed. Pure Python.

> Commands declare whether they're undoable and a reverse factory.

**The code:** Commands have `is_undoable` property. But there's no undo stack. No reverse factory on any command. `Pipeline.dispatch()` doesn't record commands for undo.

This is expected for V1 engine work — undo is an editor concern. But the command infrastructure should be designed to support it. Currently, commands don't carry enough information to reverse themselves (e.g., `RemoveBlockCommand` only has `block_id` — to undo it, you'd need the full block definition that was removed).

**Fix direction:** Add `_snapshot` support to commands: before executing a destructive command, the handler snapshots the affected state. The undo stack stores (command, snapshot) pairs. Undo = restore from snapshot. This is simpler than reverse factories and more reliable.

---

## M4: Missing — Execution Threading

**The distillation says:**
> Background execution uses stdlib ThreadPoolExecutor with bounded workers.
> Per-block threading.Lock keyed by block ID for execution locks.

**The code:** `ExecutionEngine.run()` is synchronous. It runs all blocks sequentially on the calling thread. `Coordinator.request_run()` is also synchronous — it blocks until execution completes.

The audit (Part 1, ED2) flagged this as "request_run blocks calling thread (UI freeze)." The fix is straightforward (ThreadPoolExecutor + Future), but it's not just a bug fix — it's a missing architectural piece.

**Fix direction:** 
1. `ExecutionEngine.run()` stays synchronous (it's the inner loop)
2. `Coordinator.request_run()` submits to a `ThreadPoolExecutor` and returns a Future/callback
3. Completion fires a domain event that the UI can observe

---

## M5: Missing — Per-Song Graph Isolation

**The distillation says:**
> For each song: get active version audio → set audio_path on LoadAudio → execute actions → store results tagged with song_id + version_id

But the Project entity has ONE `graph_json`. The Orchestrator's `execute()` mutates the LoadAudio block's `file_path` in the pipeline's graph to point to the song's audio. This means:
- The persisted graph has one audio path (the last song processed)
- Processing a setlist of 20 songs overwrites the graph's LoadAudio path 20 times

**The distillation's intent:** The graph is a template. Each song execution creates a runtime copy with the song's audio injected. The template graph doesn't change.

**The code almost does this** — the Orchestrator deserializes a fresh Pipeline from PipelineConfig, injects the audio path, executes, and persists results. The PipelineConfig per song version stores the template. But the `graph_json` on the Project entity is ambiguous — is it the template? The last-used state?

**Fix direction:** Clarify that `Project.graph_json` is the **developer's graph template** (the Pipeline Workbench state). PipelineConfigs are per-song-version copies derived from templates. Execution always uses PipelineConfig, never Project.graph_json directly.

---

## Summary Table

| # | Type | Severity | Issue |
|---|------|----------|-------|
| S1 | Structural | **CRITICAL** | Engine graph and SQLite DB are disconnected — no sync mechanism |
| S2 | Structural | HIGH | No UoW — commands don't participate in transactions |
| S3 | Structural | HIGH | Coordinator and Pipeline own separate Graph instances |
| S4 | Structural | MEDIUM | Block categories not enforced at execution |
| S5 | Structural | MEDIUM | No copy-on-pull — upstream references are mutable |
| S6 | Structural | LOW | Single DB connection instead of read/write pair |
| S7 | Structural | LOW | Repositories are stateful (hold conn ref) |
| M1 | Missing | HIGH | No bootstrap.py — no canonical wiring point |
| M2 | Missing | **CRITICAL** | No application service layer bridging editor ↔ persistence |
| M3 | Missing | MEDIUM | No undo stack (commands support it, infrastructure doesn't) |
| M4 | Missing | MEDIUM | No threaded execution (everything synchronous) |
| M5 | Missing | MEDIUM | Per-song graph isolation unclear (template vs instance) |

---

## Recommended Resolution Order

### Phase 1 — Structural Foundation (before any UI work)
1. **S3** — Pipeline takes Graph externally, doesn't create its own
2. **S1 + S2 + M2** — Build the `ProjectCoordinator` (or `AppSession`) that:
   - Owns ProjectSession + editor Pipeline + Coordinator + Orchestrator
   - On graph mutation: serializes to `graph_json` (can be debounced)
   - On project save: ensures graph_json is fresh, WAL checkpoint, pack_ez
   - On project load: deserializes graph_json → populates editor Pipeline
3. **M1** — Create `bootstrap.py` that wires the full object graph

### Phase 2 — Execution Correctness
4. **S4** — Align BlockCategory enum, skip non-Processor blocks in planner
5. **S5** — Copy-on-pull in ExecutionContext.get_input()
6. **M4** — ThreadPoolExecutor in Coordinator.request_run()

### Phase 3 — Editor Infrastructure
7. **M3** — Undo stack with snapshot-based reversal
8. **M5** — Clarify Project.graph_json as template, PipelineConfig as per-song instance

### Phase 4 — Optimization (defer until UI exists)
9. **S6** — Two-connection pattern (read/write split)
10. **S7** — Stateless repositories (receives conn per call)

---

## What's Working Well

To be clear — a lot is right:

- **Domain types are clean.** Frozen dataclasses, typed ports, clear invariants. FP2 is expressed well at the type level.
- **Graph aggregate is solid.** Validation, topological sort, connection rules — all correct.
- **Engine Pipeline builder** (`pipelines/pipeline.py`) is elegant. The `add()` / `output()` API is exactly what the distillation describes.
- **Processor DI pattern** works. Injectable functions, testable without real ML models.
- **Result type** is used consistently at boundaries.
- **Persistence layer** is well-structured. Repos, schema, migrations, archive — all clean.
- **Take system** is a faithful implementation of the distillation's design.
- **Staleness tracking** with human-readable reasons (D280) is exactly right.
- **Event system** (EventBus + RuntimeBus separation) matches the distillation's domain-events-vs-progress-side-channel design.

The engine works. The persistence works. The gap is the application layer that binds them.

---

*Generated 2026-03-30. This audit should be revisited after the Phase 1 structural fixes land.*
