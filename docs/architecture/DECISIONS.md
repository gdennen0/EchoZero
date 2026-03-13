# EchoZero — Architecture Decisions Log

**Purpose:** Running record of architectural decisions made during rewrite planning.  
**Started:** 2026-03-01

---

## Decision Log

### D1: Replace `metadata: Dict[str, Any]` with typed settings
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Block settings stored as individual keyed rows in DB. Block types define hardcoded default keys. Some block types allow users to add rows at runtime (e.g., action items in setlists). Same storage interface for both.  
**Rationale:** Six of seven entities carried an untyped escape hatch. The most important data (block config) lived in an opaque blob with no schema, validation, or migration path.

### D2: Eliminate denormalized block names
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** All relationships use IDs only. Names resolved at the display/query layer.  
**Rationale:** `Connection` and `ActionItem` stored block name copies with no synchronization on rename. Silent data corruption risk. Unanimous panel consensus.

### D3: MANIPULATOR extracted as own abstraction
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Bidirectional communication (ShowManager ↔ Editor) modeled as its own concept, separate from the port/connection directed graph.  
**Rationale:** A "bidirectional port" breaks the directed graph contract. It's a channel, not a port.

### D4: Two block types — Processor and Workspace
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
- **Processor Block:** Pure transform. Pull inputs (copy) → execute in background → store outputs. Isolated, idempotent.  
- **Workspace Block:** Manages own data. No execution pipeline. Manual pull with user-controlled merge options. Editor is the primary Workspace block.  
**Rationale:** Fundamental behavioral split between blocks that derive data (most blocks) and blocks that curate/manage data (Editor, potentially others).

### D5: Non-blocking execution with UI isolation
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
- Processors run on background worker thread, never touch Qt or event bus  
- Progress/completion via Qt signals marshaled to main thread  
- UI stays responsive during execution  
- Execution lock prevents conflicting runs on same block  
- No need for true parallelism or subprocess isolation initially — just non-blocking  
**Rationale:** Griff's requirement: "I still want to be able to execute other blocks, change settings and use the rest of the application."

### D6: Copy-on-pull — each block is an island
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** When a block pulls upstream data, it receives a COPY, not a reference. Each block's stored data is fully self-contained. Deleting or re-running an upstream block does not affect downstream blocks' existing data.  
**Rationale:** Eliminates hidden dependencies. Makes every block an isolated, portable unit of data.

### D7: Events are pure time data — audio bundled as snapshot
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
- EventDataItems contain time/duration/classification data only  
- No `audio_id` cross-references to other blocks' data  
- When events are produced, the source audio is copied and bundled with them through the connection  
- Downstream blocks receive events + audio snapshot through the same wire  
- Audio snapshot is a copy — upstream reload doesn't affect downstream  
**Rationale:** Eliminates the hidden `_lookup_audio_from_events()` pattern. Blocks receive all needed data through explicit connections.

### D8: Waveforms are a display-layer concern
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Waveforms not stored in DataItems. Computed on-demand by the display layer from audio data the block has access to through its connections.  
**Rationale:** Clean separation between processing data and display artifacts.

### D9: One setlist per project (confirmed intentional)
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** One setlist per project is a hard constraint. However, individual song VERSIONING is required — songs need multiple versions (different arrangements), with version selection and history preservation.  
**Rationale:** Rex challenged this as arbitrary. Griff confirmed it's intentional. The versioning need is new and wasn't in the original spec.

### D10: Persistence — SQLite as runtime engine, .ez file is a zip archive
**Date:** 2026-03-01 (revised 2026-03-01)  
**Status:** Decided (revised)  
**Decision:** SQLite is the runtime source of truth with proper schema, foreign keys, and referential integrity. The `.ez` file is a **zip archive** containing the SQLite database file plus audio assets. **Save** = close connection, zip database + audio into `.ez`. **Load** = extract `.ez`, open the SQLite database. SQLite does NOT persist between sessions as a standalone file — it exists only while a project is open. The `.ez` file is what users save, share, and back up. Migrations (versioned SQL scripts) run on the extracted database when opening a project saved with an older app version.  
**Rationale:** Original D10 made SQLite persist as a live database, which conflicts with desktop app UX where projects are files. The zip approach gives SQLite's runtime power (queries, foreign keys, transactions) while keeping projects portable and user-understandable. No JSON serialization layer — the SQLite file IS the data, just bundled.

### D11: Typed port declarations with connection-time validation
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
- Ports declared as typed class attributes on block processor  
- Engine validates type compatibility at connection creation time, not execution time  
- `GENERIC` eliminated or replaced with explicit `ANY` opt-in  
- Cardinality (SINGLE/MULTI) enforced at connection time  
- No runtime filter machinery — graph is type-valid by construction  
**Rationale:** Filtering is a port responsibility. Validate early (connection time), not late (execution time).

### D12: `item_key` / `semantic_key` for stable DataItem identity
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** DataItems carry a stable semantic key that survives re-execution. Engine upserts by `(block_id, item_key)` rather than creating new UUIDs each run.  
**Rationale:** Eliminates orphaned DataItems, enables selective operations, provides stable identity for downstream references.

### D13: Workspace block pull options
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
- Pull is ALWAYS manual, never automatic  
- Two pull actions: **Pull (replace)** — nuke upstream-derived layers, pull fresh, user layers untouched. **Pull (keep existing)** — fetch new data alongside old, no deletions.  
- **Dismiss** — clear "new data available" indicator  
- **Merge** — desired full functionality, deferred until stable event identity is designed (see D19)  
- User edits are sacred: layer-level `source_type` ("upstream" | "user") drives protection. Upstream-derived layers that are manually edited become user-owned.  
**Rationale:** Workspace blocks manage their own data. User edits must be preserved. Merge deferred due to event identity complexity.

### D14: Event classification visual display
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Per-event `classification_state` enum (UNCLASSIFIED, TRUE, FALSE, USER_TRUE, USER_FALSE) with optional `confidence: float`. Single layer contains all classification states. UI renders opacity from state (TRUE/USER_TRUE = opaque, FALSE/USER_FALSE = semi-transparent). USER_ variants protect manual decisions from being overwritten by re-running classifiers. `pull_id` on events enables batch tracking for Keep Both.  
**Rationale:** Classification is data, not display. One layer per classification bucket, visual differentiation via opacity.

### D15: Consolidated into D17

### D16: Execution flow
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
```
1. TRIGGER — User clicks Execute, engine acquires lock
2. PULL — Follow connections → read upstream output → COPY data → bundle into inputs dict
3. EXECUTE — processor.process(inputs, settings) in background worker. Pure function.
4. STORE — Output DataItems stored in block's local data. Manifest updated. Downstream marked STALE.
5. NOTIFY — Completion signal to main thread via Qt signal. Lock released.
```
**Rationale:** Confirmed with Griff. Copy-on-pull, isolated execution, atomic commit, each block is an island.

---

### D17: Group as first-class entity
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** LayerGroup is a real entity with id, name, collapsed state, and ordered list of layers. Layer properties (group_id, group_name, group_index) removed from Layer. SyncState extracted into its own entity. LayerOrder table eliminated — order is structural.

### D18: Consolidated into D14

### D19: Merge pull option — deferred until event identity is designed
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** Merge requires stable event identity across re-executions, which doesn't exist yet. Desired functionality: smart merge with diff/conflict UI, selective accept/reject, preservation of manual edits. Requires stable event identity model (content-hash or deterministic IDs). Full feature — implementation phasing TBD later.

### D20: FULL strategy stops at Workspace boundaries
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** Workspace blocks are terminal inputs in execution graph. Engine reads current output without calling process(). Formal `BlockCategory` enum (PROCESSOR, WORKSPACE) checked in dispatcher.

### D21: Copy semantics — true full copy
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** "Copy" means full copy — metadata AND audio files. Each block writes copies to its own output directory. Block B's data is completely independent of Block A. No shared file references, no reference counting. True isolation. Disk space tradeoff accepted in favor of reliability and predictability.

### D22: Execution lock scope — per-block
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Per-block execution lock prevents re-running same block. NOT a global project lock. UI reads never blocked. STORE phase is a single DB transaction.  
**Rationale:** Griff's core requirement is non-blocking UI — user must be able to execute other blocks, change settings, and use the app while a block runs. A global lock would freeze the app during long ML operations. Per-block lock allows concurrent independent block execution while preventing conflicting runs on the same block. Supersedes Dr. Voss's earlier recommendation for a global lock.

### D23: Cancellation model
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** CancellationToken checked at processor breakpoints. Cancel before STORE = safe rollback. STORE is atomic (single DB transaction), not interruptible.

### D24: System invariants
**Date:** 2026-03-01  
**Status:** Decided (panel consensus)  
**Decision:**  
- Cross-project connections explicitly forbidden (enforced, not implied)  
- `load_project` is transactional (rollback on failure)  
- Block rename propagates atomically or fails  
- No processor touches Qt objects  
- No DataItem lookup bypasses the connection graph  
- `semantic_key` required on every DataItem at creation  
- Re-execution is idempotent (for Processor blocks)  
- Graph validation at connection time (type, cardinality)

### D25: ActionSet — single canonical representation
**Date:** 2026-03-01  
**Status:** Decided (panel recommendation)  
**Decision:** ActionSets must have one canonical form — either normalized DB rows OR embedded JSON, not both. Eliminate the dual representation.

### D26: BlockDataManifest as single source of truth
**Date:** 2026-03-01  
**Status:** Decided (panel recommendation)  
**Decision:** Collapse `block_local_state` and data manifest into a single entity (BlockDataManifest). One place to answer "what data does this block have, where did it come from."

### D27: Partial failure in FULL execution
**Date:** 2026-03-01  
**Status:** Decided (panel consensus)  
**Decision:** Fail fast — stop at first failure, don't commit the failed block. Already-committed ancestors keep their outputs. Downstream marked STALE. Atomic commit per block: stage outputs → verify all succeed → atomic swap to canonical location. Never partial-write.

### D28: DataItem garbage collection
**Date:** 2026-03-01  
**Status:** Decided (panel recommendation)  
**Decision:** Track `execution_id` on DataItems. When a block's outputs are replaced, old DataItems and their files are marked for cleanup. Periodic background pruning of orphaned items.

### D29: Dynamic DataItem rules
**Date:** 2026-03-01  
**Status:** Decided (panel consensus)  
**Decision:**  
- Dynamic cardinality on known port types = fully supported  
- Block-internal state items tagged `internal: true`, cleaned up on block delete  
- Cross-block metadata discovery via ad-hoc repo lookups = PROHIBITED. If Block B needs Block A's data, draw a connection.

---

## Pending / Open Questions

### ~~Command Bus / Event Bus / Facade Architecture~~ → RESOLVED (D30–D34)
Pipeline + adapters architecture decided. CommandSequencer and AudioPlayer conflicts with background worker remain open (see Block-Specific Designs). Undo for Workspace blocks scoped (global stack, editable commands only) — detailed Editor undo interactions still need design.

### ShowManager Sync Layers in Editor
How synced layers (MA3 ↔ Editor) work in the new Workspace block model. SyncBinding is now a separate entity — need to design how live MA3 sync interacts with the pull model, layer identity, and bidirectional communication channel. Griff flagged for dedicated discussion.

### Stale Propagation Mechanism
How downstream blocks learn they're stale after upstream execution. Needs design.

### Block-Specific Designs
- **Editor:** Merge behavior, layer/group structure (panel reviewing)
- **CommandSequencer:** Can't run in background worker (dispatches commands)
- **AudioPlayer:** Needs Qt multimedia (background worker conflict)
- **ShowManager:** Network listener lifecycle doesn't fit command pattern

### MANIPULATOR Runtime Behavior
D3 extracts MANIPULATOR as its own abstraction (channel, not port). The runtime communication protocol — actual bidirectional data flow semantics — still needs design. Ties into ShowManager sync layers discussion.

### UPSTREAM_ERROR Block Status
Should STALE be split into STALE (parent hasn't run yet / parent output changed) and UPSTREAM_ERROR (parent execution failed)? Affects how users understand the state of their pipeline after partial failures.

### Layer Dirty Tracking Granularity
D13 establishes `source_type` for user edit protection. The specific dirty-tracking mechanism needs design: layer-level flag? Which actions trigger it? Confirmation flow when pulling over a modified upstream-derived layer?

### Execution Engine Component Decomposition
Whether to decompose the monolithic execution engine into single-responsibility components (planner, dispatcher, input gatherer, output persister, staleness tracker, event emitter). Panel recommended this but no decision made.

### Subprocess Isolation for Heavy ML Blocks
Rex recommended subprocess isolation for ML inference blocks (Demucs, etc.) rather than redesigning the threading model. Worth exploring when execution architecture is implemented.

### Full Metadata Typing Coverage
D1 replaces `metadata: Dict` with typed settings. Open question: does ANY block type need an extensible/untyped escape hatch, or can every block's config be fully typed?

### Universal Observability
Designed (see OBSERVABILITY.md) but implementation details depend on command bus/event bus decisions.

### Undo for Workspace Blocks
Workspace blocks (Editor) need real undo for user edits. Undo stack scope confirmed global (D33), but Editor-specific undo interactions (layer edits, event classification changes, group operations) need detailed design.

---

## Command Bus / Facade Architecture (2026-03-01)

### D30: Single pipeline, multiple adapters — no facade
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Replace the monolithic `ApplicationFacade` (104 methods, 5,307 lines) with a single `Pipeline` that routes commands/queries to handlers through a middleware chain. Each external interface (Qt GUI, MCP server, CLI) is a thin adapter that translates its native format (button clicks, tool calls, text input) into commands and dispatches them to the same pipeline. Adding a new interface = writing one adapter file, zero changes to pipeline or handlers.  
**Rationale:** The facade pattern breaks down at scale — it became a god object. The pipeline pattern gives one standardized path for all operations, enforces consistent validation/undo/event-emission, and makes new interfaces trivial to add. This is Griff's "one pipeline, multiple interfaces" principle.

### D31: Commands are pure data with self-described validation and undo
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Commands are frozen dataclasses (pure data, no framework deps). Each command declares:  
1. Its fields (the data it needs)  
2. A `validate(ctx)` method (constraints on that data)  
3. Whether it's `undoable` (bool)  
4. A `reverse()` factory (returns the command that undoes this one)  

Handlers are separate classes with injected service dependencies — they do the actual work. Commands never reference the facade or UI. Pipeline middleware orchestrates validation → undo check → handler dispatch → event emission.  
**Rationale:** Eliminates circular dependency (old commands took `facade` as first arg). Commands are testable in isolation. Handlers are testable with mocked services. Self-describing commands mean the pipeline doesn't need to know about specific operations.

### D32: Two command categories — Editable (undoable) and Operational (not undoable)
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
- **Editable commands** (`undoable = True`): Structural edits — add/delete/move blocks, connect/disconnect, rename, edit settings, classify events, reorder layers. These go on the undo stack.  
- **Operational commands** (`undoable = False`): Execute block, pull upstream, load/save project, MA3 sync. These do NOT go on the undo stack. Adapters must warn the user before dispatching ("This action cannot be undone").  
**Rationale:** Industry standard (DAW pattern). Execution involves heavy computation, file I/O, and external side effects that can't be meaningfully reversed. Rather than fake undo, be honest about it. Future optimization: store one generation of previous block output for "revert to previous" (cheaper than full undo, more predictable).

### D33: Global undo stack, editable commands only
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** One undo stack per project. Only editable commands (D32) are pushed. The undo stack is a pipeline middleware — pure Python, not coupled to `QUndoStack`. A Qt adapter can bridge to `QUndoStack` for menu integration if needed, but the core undo system has no framework dependency.  
**Rationale:** Global undo is what every DAW does (Ableton, Logic, Reaper). Per-block undo would confuse users. Decoupling from Qt means the undo system works in CLI/MCP/test contexts.

### D34: CQRS-lite — Commands and Queries through same pipeline, different middleware
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Reads (queries) and writes (commands) both go through the `Pipeline`, but queries skip undo and validation middleware. Queries are also pure data with separate handlers. This gives a single dispatch mechanism while allowing different processing paths.  
```
Command path: Validate → Undo check → Handler → Emit events → Result
Query path:   Handler → Result
```
**Rationale:** One pipeline, one dispatch mechanism, one place to add cross-cutting concerns (logging, auth, rate limiting). But reads and writes have fundamentally different needs — reads don't mutate state, don't need undo, don't emit domain events.

---

## Event Bus + Persistence Architecture (2026-03-01)

### D35: Typed event classes with minimal DomainEvent base
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** Frozen dataclasses. Base carries `timestamp: float`, `correlation_id: str`. Each domain event (`BlockAdded`, `ConnectionCreated`, etc.) inherits from this. No universal ObservableEvent blob. No `source` field — event type provides provenance. Observability metadata (layer, component) derived from event type, not stored.  
**Rationale:** Type safety for handlers, universal metadata for observability/MCP. All 5 panelists agreed.

### D36: Collect-then-publish — events staged during handler, flushed after commit
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** The UoW collects events via `uow.collect(event)`. After successful commit, events are published to the bus. On rollback, events are discarded. No event escapes the transaction boundary.  
**Rationale:** Strongest consensus across the panel. Pre-commit events cause phantom UI updates for rolled-back operations — "events that lie."

### D37: Re-entrant publish uses breadth-first queue
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** Events published by handlers during dispatch are queued and processed after the current batch completes. Never recursive. Predictable ordering.  
**Rationale:** Prevents stack overflow, infinite loops, and makes event ordering deterministic and debuggable.

### D38: Qt adapter is one file in the UI layer — core never imports Qt
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** A subscriber (~30 lines) that marshals events to the Qt main thread via `QCoreApplication.postEvent`. The event bus contract is `Callable[[DomainEvent], None]`. No Qt types in the core. If running CLI/MCP, the adapter simply isn't loaded.  
**Rationale:** Pure Python core enables CLI, MCP, and testing without Qt. Legacy bus required a running Qt app — this was a known pain point.

### D39: Unit of Work middleware wraps every command
**Date:** 2026-03-01  
**Status:** Decided (4-1, Rex dissented)  
**Decision:** ~40 lines of middleware. Handlers receive a UoW context (connection + event collector). They never call commit/rollback. The middleware begins the transaction, calls the handler, commits on success (flushing events), rolls back on failure (discarding events).  
**Rationale:** Enforces the post-commit event rule (D36) architecturally rather than by developer discipline. Rex argued handler-managed transactions are simpler for a single-user app, but the UoW is equally simple and eliminates an entire class of consistency bugs by construction.

### D40: Repositories are stateless — receive connection from UoW
**Date:** 2026-03-01  
**Status:** Decided (4-1)  
**Decision:** Repos are pure data access helpers. `save(conn, entity)`, `get(conn, id)`. No Database reference. No internal locks. Concurrency is the pipeline's job.  
**Rationale:** Direct consequence of D39. If the middleware owns the transaction, repos can't own connections or commit independently.

### D41: STORE phase dispatches through the pipeline
**Date:** 2026-03-01  
**Status:** Decided (4-1, Kai dissented)  
**Decision:** Background execution completes computation, then dispatches a `StoreExecutionResult` command through the pipeline. Same transaction model, same event emission, same middleware. No shadow write path.  
**Rationale:** Two write paths = two sets of bugs. Kai raised deadlock concerns, but the per-block lock (D22) is already held during execution, so there's no contention.

### D42: Separate read and write connections from day one
**Date:** 2026-03-01 (revised 2026-03-01)  
**Status:** Decided (revised per Griff)  
**Decision:** Two SQLite connections: one write connection (owned by UoW middleware, protected by `threading.Lock`) and one read connection (used by query handlers, no lock needed). WAL mode enables concurrent reads during writes natively. Queries never wait on the write lock. Both connections created at bootstrap.  
**Rationale:** Background STORE operations lock the write connection. UI reads shouldn't stutter waiting for that lock. It's ~3 lines of extra setup and prevents a whole category of UI lag. Original "wait and see" approach was premature optimization in reverse — avoiding trivial work that has clear benefit.

### D43: Schema migrations via versioned SQL scripts (on project load)
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** `migrations/` directory with numbered `.sql` files. `schema_version` table inside the SQLite database tracks current version. Migrations run **when opening a project** saved with an older app version — extract the `.ez`, check schema version, run unapplied scripts, then open. Each migration runs in its own transaction. No ORM. During active development, simply delete and recreate.  
**Rationale:** SQLite is the runtime engine bundled inside `.ez` files (D10 revised). Migrations only matter between app versions when a user opens an older project.

### D44: Middleware stack ordering
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**  
```
Command → Correlation (assign ID) → Validation → UoW (begin → handler → commit → flush events) → Result
Query → Handler (read connection) → Result
```
Queries skip validation, UoW, and undo middleware. Direct to handler with a read connection. Correlation ID generated at pipeline entry, stamped on command context, copied to all events created during handling.  
**Rationale:** Clear, predictable, same every time. One sentence: validate, then do the work in a transaction, then announce what happened.

---

## DI / Threading / Error Model (2026-03-01)

### D45: Module-level wiring — no DI framework
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** A single `bootstrap.py` constructs the entire object graph with explicit constructor calls. Two plain dictionaries: `handlers: dict[type[Command], Handler]` and `processors: dict[str, type[Processor]]`. No container, no locator, no decorators, no auto-discovery. Tests call constructors directly with mocks.  
**Rationale:** 17 block types and ~30 handlers is a dictionary, not a framework. Every dependency visible in one file.

### D46: Pipeline is a synchronous function call, thread-agnostic
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** `pipeline.dispatch(command) -> Result[T]` runs on the caller's thread. Any thread can call it. Not an actor, event loop, or thread-bound object.  
**Rationale:** No marshaling infrastructure needed. Background threads call dispatch() directly. Only thread boundary needing explicit marshaling is EventBus → Qt UI (handled by D38 adapter).

### D47: ThreadPoolExecutor for background execution
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** stdlib `concurrent.futures.ThreadPoolExecutor` with bounded workers. Managed by the execution handler, not the pipeline. Per-block `threading.Lock` for execution locks (D22), keyed by block ID.  
**Rationale:** No custom thread infrastructure. stdlib handles thread lifecycle, exception propagation, and future-based completion.

### D48: Single shared write connection with lock
**Date:** 2026-03-01  
**Status:** Decided (4-1, Voss dissented)  
**Decision:** UoW middleware uses a single SQLite write connection created at bootstrap, protected by `threading.Lock`. All write commands (including `StoreExecutionResult` from background threads) serialize through this lock.  
**Rationale:** For a desktop app with a few concurrent block executions, lock contention is negligible. Provides same serialization guarantee as a dedicated writer thread with zero additional infrastructure.

### D49: StoreExecutionResult dispatches through pipeline via direct call
**Date:** 2026-03-01  
**Status:** Decided (4-1, Sage dissented)  
**Decision:** Background worker calls `pipeline.dispatch(StoreExecutionResult(...))` from the worker thread. Goes through full middleware stack. UoW middleware acquires the shared write connection lock (D48), commits, flushes events.  
**Rationale:** One write path, one middleware stack, no special cases.

### D50: Result[T] return type for handlers, exceptions for infrastructure only
**Date:** 2026-03-01  
**Status:** Decided (3-1-1)  
**Decision:** Handlers return `Result[T]` — a frozen dataclass with `value` or `error`. UoW middleware commits on `result.ok`, rolls back on failure. Unexpected exceptions caught by outer try/except, wrapped into `Result.fail(InfrastructureError(...))`. Three error categories: `ValidationError`, `DomainError` (includes cancellation), `InfrastructureError`.  
**Rationale:** Result makes commit/rollback decisions explicit. Exceptions reserved for genuinely unexpected failures.

### D51: Qt-dependent components are UI adapters, not pipeline handlers
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** CommandSequencer, AudioPlayer, and other Qt-dependent components live in the UI layer. They call the pipeline — the pipeline does not call them. Not registered as command handlers.  
**Rationale:** Maintains D38 (core never imports Qt). These are UI concerns that trigger domain operations.

### D52: Standardized progress events for execution
**Date:** 2026-03-01  
**Status:** Decided → Resolved by D53-D56  
**Decision:** Griff requirement: "extremely solid progress events standardized through the application." Resolved in dedicated panel session — see D53-D56 below.

---

## Progress Events, Stale Propagation, Execution Decomposition (2026-03-01)

### D53: Progress is a side-channel, not a domain event
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** Progress uses a dedicated `ProgressChannel` — thread-safe, in-memory pub/sub completely separate from the domain event bus (D35-D38). The domain event bus receives only `ExecutionStarted` and `ExecutionCompleted`/`ExecutionFailed`. All mid-execution progress goes through `ProgressChannel`. Not persisted. Not serialized.  
**Rationale:** Progress fires during execution (before commit). Domain events fire after commit (D36). These are irreconcilable in one system.

### D54: ProgressReport schema
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**
```python
@dataclass(frozen=True)
class ProgressReport:
    execution_id: str
    block_id: str
    phase: str              # "pull", "execute", "store"
    fraction: float         # 0.0–1.0
    stage: str | None       # "Separating vocals", "Loading model", etc.
    items_completed: int | None
    items_total: int | None
    timestamp: float        # time.monotonic()
```
No ETA field. UI can derive ETA from fraction rate if desired.  
**Rationale:** Minimal, sufficient, no promises the system can't keep.

### D55: Processors receive a ProgressReporter callback
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Processors receive a `ProgressReporter` in their `process()` method:
```python
class ProgressReporter(Protocol):
    def report(self, fraction: float, stage: str | None = None,
               items_completed: int | None = None, items_total: int | None = None) -> None: ...
    @property
    def cancellation_token(self) -> CancellationToken: ...
```
Combines progress reporting and cancellation into one object. Reporting is optional — processors that don't call it show indeterminate in UI. Throttle at the consumer (UI at 10-30Hz), not the producer.  
**Rationale:** Single dependency for the processor. Clean protocol. No coupling to event bus or Qt.

### D56: Composite progress via weighted aggregation
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** For FULL execution, the ExecutionCoordinator holds an aggregator that weights each block's fraction by estimated duration (historical average, default to equal weight). Per-block progress maps into the block's slice of overall progress.  
**Rationale:** Users running FULL execution need one coherent progress view, not per-block fragments.

### D57: Four-state DataState enum
**Date:** 2026-03-01  
**Status:** Decided (5-1, Rex dissented)  
**Decision:**
```python
class DataState(Enum):
    FRESH = "fresh"
    STALE = "stale"
    UPSTREAM_ERROR = "upstream_error"
    ERROR = "error"
```
Stored as a column on the blocks table. STALE = upstream output changed. UPSTREAM_ERROR = upstream execution failed. ERROR = this block's own execution failed. FRESH = output is current.  
**Rationale:** Different states require different user actions. STALE → re-run. UPSTREAM_ERROR → fix ancestor first. ERROR → check this block's logs.

### D58: Immediate full-depth stale cascade
**Date:** 2026-03-01  
**Status:** Decided (5-1, Rex dissented)  
**Decision:** Staleness propagates immediately to all descendants when triggered. Triggers: execution completion, settings change, connection change. Cascade stops at Workspace boundaries (D20).  
**Rationale:** False "fresh" is dangerous (user exports stale audio). False "stale" is merely annoying. Full cascade is the safe default.

### D59: Staleness stored as column, not computed
**Date:** 2026-03-01  
**Status:** Decided (5-1, Sage dissented)  
**Decision:** `data_state` column on the `blocks` table. Updated transactionally when propagation triggers fire. StalenessService is a pipeline handler triggered by `ExecutionCompleted`/`ExecutionFailed`/`SettingsChanged`/`ConnectionChanged` domain events.  
**Rationale:** Fast reads (O(1) per block), transactional updates via UoW, explicit state.

### D60: Execution engine decomposed into four components
**Date:** 2026-03-01  
**Status:** Decided (4-1-1)  
**Decision:** Four components:
- **GraphPlanner** — pure function, topological sort, execution order, DAG validation
- **ExecutionCoordinator** — FULL orchestration, sequencing blocks, aggregate progress, fail-fast
- **BlockExecutor** — single block PULL→EXECUTE→STORE, per-block lock, progress reporting
- **StalenessService** — propagation logic, pipeline handler triggered by events after STORE commits

PULL and STORE are phases within BlockExecutor, not separate components. Components interact via direct calls internally; StalenessService triggered via pipeline events.  
**Rationale:** Clean boundaries without over-decomposition. Each component has single responsibility.

### D61: Threads, not subprocesses for ML blocks
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** All block execution uses ThreadPoolExecutor (D47). No subprocess isolation. If a specific processor needs it later, it's an internal detail of that processor.  
**Rationale:** Subprocess adds IPC complexity for progress, error handling, and cancellation. Premature for single-user desktop app.

---

## Pending / Open Questions (Updated)

### Progress during PULL phase
Should BlockExecutor report progress during PULL (copying large audio files)? PULL can take seconds for large projects. The ProgressReport `phase` field supports it.

### Weighted composite progress — historical data
Where does execution time history come from for weighting? Need `execution_history` table or just use equal weights?

### Remaining Entity Design
- Port & Connection (typed ports decided D11, full entity design not done)
- Project (lifecycle, .ez save/load mechanics)
- Setlist (one per project D9, song versioning, action sets D25)
- Full metadata typing

### Feature-Specific Design
- ShowManager sync layers (MA3↔Editor bidirectional sync)
- MANIPULATOR runtime behavior (D3 says channel, not port)
- Editor undo (layer edits, classification changes, group ops)
- Editor merge pull (deferred D19, needs stable event identity)
- Layer dirty tracking
- Block-specific designs (CommandSequencer, AudioPlayer, ShowManager)

### Universal Observability (Finalization)
OBSERVABILITY.md designed. Now that event bus (D35-D38) and progress channel (D53) are decided, finalize how observability layers onto both.

### Q5: Disconnection Behavior
How downstream blocks behave when connections are removed. Stale cascade? Data retention? User notification?

---

## Port, Connection & ShowManager Decisions (2026-03-02)

### D62: Ports are Value Objects on Block
**Date:** 2026-03-02  
**Status:** Decided (6-0)  
**Decision:** Ports have no IDs, no repository. They are frozen value objects stored as part of Block's state. A port is a slot, not an entity.

### D63: Direction is structural, not a field on Port
**Date:** 2026-03-02  
**Status:** Decided (3-3, Sage tiebreaker for B)  
**Decision:** Block holds `input_ports` and `output_ports` collections. Port has no direction field. BIDIRECTIONAL retired as a direction value.

### D64: Control channels are a separate edge type
**Date:** 2026-03-02  
**Status:** Decided (5-1)  
**Decision:** Manipulator reframed as a symmetric control channel. Block gains `control_ports` collection. Connection logic for control channels is a separate edge type. Port direction model is now binary: INPUT / OUTPUT only.

### D65: PortType is a proper Enum with compatible_with()
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** PortType enum: Audio, Event, OSC, Control. Includes `compatible_with()` method with default exact-match, overridable per type. New Port value object = `name` + `port_type` only. Metadata dict eliminated.

### D66: Connection is a Value Object — no ID
**Date:** 2026-03-02  
**Status:** Decided (6-0)  
**Decision:** Connection is a frozen value object. Identity defined by endpoint tuple: (source_block_id, source_output_name, target_block_id, target_input_name). No UUID.

### D67: Fan-out is universal
**Date:** 2026-03-02  
**Status:** Decided (6-0)  
**Decision:** Any output port can connect to N input ports, subject to type compatibility. No per-type restriction.

### D68: Fan-in is declared per PortType
**Date:** 2026-03-02  
**Status:** Decided (5-1)  
**Decision:** PortType has `allows_fan_in` property. Audio=False, OSC=False, Event=True. Attempting fan-in on a 1:1 port raises a domain error — no silent replacement. UI layer may issue delete+connect sequence if user confirms.

### D69: Control connections reuse Connection, stored separately
**Date:** 2026-03-02  
**Status:** Decided (5-1)  
**Decision:** Control channel connections use same Connection value object, stored in `control_connections` collection on Pipeline. Fan-out (one controller → many params) permitted. Fan-in (many controllers → one param) rejected.

### D70: Connection validation lives on the graph aggregate
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** The graph aggregate owns the connection set and validates invariants: type compatibility, fan-in rules, format compatibility, cycle detection. Connection is a dumb value object. Validation no longer lives in ConnectionService.

### D71: PortType compatibility is exact-type-only
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** `compatible_with()` checks only that two ports share the same PortType enum value. No subtypes in PortType.

### D72: Cross-type connections always invalid
**Date:** 2026-03-02  
**Status:** Decided (unanimous)  
**Decision:** Event→Audio, OSC→Event, and all cross-type connections are unconditionally rejected. Coercion needs are served by explicit coercion blocks in the graph.

### D73: PortFormat as optional value object
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** Format metadata (channel count, sample rate class) lives in a separate `PortFormat` value object, not on PortType. Ports may declare a PortFormat. When both endpoints declare formats, graph validates format compatibility as a second gate. No format = connection allowed.

### D74: compatible_with() returns CompatibilityResult
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** `compatible_with()` returns `CompatibilityResult(allowed: bool, reason: str | None)` for rich UI error reporting.

### D75: Compatibility is symmetric
**Date:** 2026-03-02  
**Status:** Decided (unanimous)  
**Decision:** If A is compatible with B, B must be compatible with A. Type system must not introduce directional asymmetry.

### D76: Single Resolution Policy replaces all sync/conflict strategies
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** Collapse SyncStrategy + ConflictStrategy into one concept: Resolution Policy with 4 options: MA3 Master, EZ Master, Union (additive merge), Ask User.

### D77: Sync pipeline integration — 5 commands, 2 queries
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** Sync operations: PushLayerToMA3, PullLayerFromMA3, MapSyncLayer, UnmapSyncLayer, SetSyncResolutionPolicy. Queries: GetSyncStatus, GetDivergence. MA3 responses = domain events. 6 service classes → 2 (sync service + MA3 network adapter).

### D78: Sync layers are configuration, not graph wiring
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** ShowManager's relationship to Editor layers is data references (layer IDs), not port connections. Configured through ShowManager's UI, not by drawing wires. Control ports (D64) exist for block-to-block control, but sync layers are configuration.

### D79: Bidirectional sync stays, simplified
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** Both sides can edit. Resolution policy per layer determines divergence handling. No complex merge algorithms — either one side wins, union, or ask user.

### D80: Sync status reduced to 4 states
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** Unmapped, Synced, Diverged, Error. Pending → transitional (not persisted). Disconnected → Error with reason. AwaitingConnection → Unmapped.

### D81: Fingerprint-based event matching retained
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** Time + duration at millisecond precision (3 decimal places). 50ms tolerance for MA3 frame-level quantization drift. No shared ID space between systems makes temporal fingerprinting the correct approach.

### D82: OSC Server is a project-level service, not a Block
**Date:** 2026-03-02  
**Status:** Decided (unanimous)  
**Decision:** OSC Server is a first-class project entity owned by the Project aggregate. No ports, no graph participation. Infrastructure, not a processing node. MIDI interface analogy.

### D83: Singleton per project, enforced by Project aggregate
**Date:** 2026-03-02  
**Status:** Decided (unanimous)  
**Decision:** One OscGateway instance per loaded project. Singleton enforced by Project aggregate, not Python module-level state. Properly injected via DI.

### D84: IOscGateway protocol interface
**Date:** 2026-03-02  
**Status:** Decided (5-0-1)  
**Decision:** Blocks interact via `IOscGateway` protocol. Send = direct method call. Receive = domain events via event bus. No bidirectional port abstraction — sending and receiving are architecturally different (UDP is fire-and-forget).

### D85: Visual status node in graph UI (view-only)
**Date:** 2026-03-02  
**Status:** Decided (4-2 with conditions)  
**Decision:** OSC Gateway rendered in graph canvas as a non-connectable infrastructure node. Shows status, allows click-to-configure. Not a real Block — no ports, no connection validation participation. Purely a view concern. Must not subclass Block.

### D86: One gateway, multiple endpoint configs
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** No multiple gateway objects. Different network targets (MA3 machines, lighting bridges) are endpoint configurations on one gateway.

### D87: ShowManager becomes pure sync logic
**Date:** 2026-03-02  
**Status:** Decided  
**Decision:** ShowManager holds sync layer records and resolution policies. Zero network knowledge. MA3-specific OSC dialect lives in an MA3 protocol adapter between ShowManager and the generic OSC gateway. Three layers: sync → protocol → wire.

---

## Disconnection Behavior (2026-03-09)

### D88: Disconnection marks downstream STALE, data retained
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** When a connection is removed:  
1. **Stale cascade:** Downstream blocks are marked STALE via the existing D58 cascade mechanism. "Connection change" (already a listed trigger in D58) includes disconnection.  
2. **Data retained:** Downstream blocks keep all stored data (inputs and outputs). Consistent with D6 — each block is an island, its data is its own copy.  
3. **No new state:** No DISCONNECTED enum value. STALE is sufficient — it tells the user "your output may not reflect current inputs" which covers both "upstream changed" and "upstream removed."  
4. **No data loss:** Accidental disconnection is recoverable. Reconnect + re-execute to get fresh data, or keep working with existing output.  
**Rationale:** Follows directly from D6 (copy-on-pull, blocks are islands) and D58 (stale cascade on connection change). Adding a DISCONNECTED state would complicate the DataState enum without providing meaningfully different user action — in both cases the user either re-runs or accepts the existing output.

### D89: Project entity design
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
Project is the top-level aggregate root. Owns:  
- `id: str` (UUID), `name: str`, `version: int` (schema version), `created_at: float`, `modified_at: float`  
- Graph (blocks + connections + control_connections)  
- Setlist (exactly one, per D9)  
- OscGateway config (exactly one, per D83)  
- Global settings (tempo/BPM reference, default timebase, FPS)  

**Lifecycle:** NEW → OPEN → MODIFIED → SAVED (cycle on edits). CLOSED from any state.  
- `CreateProject` — initialize empty SQLite, set schema version  
- `LoadProject` — extract .ez, run migrations (D43), hydrate entities  
- `SaveProject` — close DB connections, copy SQLite, zip with audio, atomic rename to .ez  
- `CloseProject` — cleanup, close connections  

**Dirty tracking:** Boolean flipped on any successful editable command, reset on save. Prompt "unsaved changes" on close/quit.  

**Audio file management:** Audio files stored in project-local directory alongside SQLite while open. LoadAudio copies source into project dir. All blocks use project-relative paths. On save, audio zipped into .ez archive.  
**Rationale:** Project aggregate enforces all invariants (one setlist, one gateway, schema version). Atomic save prevents corruption. Project-relative audio paths keep .ez files portable.

### D90: Setlist entity design
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Setlist:** `id`, `name`, `songs: List[Song]` (ordered), `default_action_set_id`.  
**Song:** `id`, `title`, `artist: str | None`, `order: int`, `active_version_id`, `versions: List[SongVersion]`, `action_set_override_id: str | None`.  
**SongVersion:** `id`, `song_id`, `label: str` (required — e.g., "Album Cut", "Festival Edit"), `audio_file_path: str` (project-relative), `created_at: float`, `notes: str | None`. No version number — versions are parallel arrangements, not sequential revisions. Display order = creation order.  

**Batch processing:** For each song in order: get active version audio → set as SetlistAudioInput source → get ActionSet (override or default) → execute actions → store results tagged with song_id + version_id → report per-song progress.  
**Batch failure: continue on failure.** Log error, mark song as failed, proceed to next. Songs are independent — failure in song N has zero impact on song N+1. User reviews all failures after batch completes.  

**Command categories:** Reorder/add/remove songs, switch active version = editable (undoable). Batch execution = operational (not undoable, per D32).  
**Rationale:** Parallel arrangements (not linear revisions) eliminates version numbering complexity. Continue-on-failure respects song independence unlike block execution (D27) where downstream depends on upstream.

### D91: Block-specific designs — CommandSequencer deleted, AudioPlayer, ShowManager
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**CommandSequencer: DELETED.** Removed from the block type inventory entirely. If sequenced actions are needed in the future, handle via scripting or macros outside the block graph.  

**AudioPlayer → Playback Block.** Third block category:  
- **Playback** — Receives data via connections, provides real-time output (audio). No EXECUTE phase, no STORE, no staleness, no persistence. Ephemeral state only (playing/paused/stopped/position).  
- Lives in UI layer (D51). Playback controls are UI operations, not pipeline commands.  
- Playback position emitted as UI-level event for playhead sync across panels.  
- Appears as a connectable block in the node graph like any other block.  

**ShowManager: Workspace block** with network listener lifecycle.  
- Sync operations (push/pull) are pipeline commands (D77).  
- Incoming MA3 messages: OscGateway receives → domain events → ShowManager handler processes through pipeline.  
- No background "execution" in the Processor sense.  
- Listener start/stop tied to project open/close (OscGateway, D82).  
- Persists sync layer registry.  

**Updated block categories:** Processor (pure transform, background), Workspace (user-managed data, manual pull), Playback (real-time output, ephemeral).  
**Rationale:** CommandSequencer added complexity for marginal value — deleted per "best part is no part." Playback is a genuinely different concern from processing or workspace management — deserves its own lightweight category rather than being forced into either.

### D92: MANIPULATOR removed — pipeline commands replace bidirectional channel
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** MANIPULATOR concept (D3) is removed from the architecture. No bidirectional channel abstraction exists.  

ShowManager ↔ Editor communication uses pipeline commands and domain events:  
- **Inbound (MA3 → Editor):** OscGateway receives → domain event → ShowManager handler → pipeline command writes to Editor layers.  
- **Outbound (Editor → MA3):** Editor publishes "layer changed" domain events (unaware of ShowManager). ShowManager subscribes, decides whether to sync, dispatches push command through pipeline → OscGateway → MA3.  

**Editor remains dumb.** Editor has zero knowledge of ShowManager, MA3, OSC, or sync. It's a pure Workspace block that manages layers/events and publishes change events. ShowManager is the only entity aware of both sides.  

Control ports (D64) retained in spec for potential future use but have zero current implementors. D3 superseded — the "channel" is the pipeline itself.  
**Rationale:** No actual bidirectional data flow exists. All communication is pipeline commands reading from one block and writing through another. MANIPULATOR added a concept with zero concrete implementation. Removing it simplifies the architecture while keeping Editor fully decoupled.

### D93: Take System replaces merge pull — layer dirty tracking via take edits
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** Merge pull (D19) and layer dirty tracking are replaced by the **Take System**. No event matching needed.  

**Model:** Each Editor layer has a **Main layer** (user-curated, syncs to MA3) and zero or more **Takes** (editable snapshots from upstream pulls). Layers are **paired** to an upstream block output — pairing tracks which block + output maps to which Editor layer.  

**Take properties:** `id`, `label` (default: "Take N — [timestamp]"), `source_block_id`, `source_settings_snapshot` (frozen copy of upstream settings at pull time), `created_at`, `events` (editable), `visible`, `order`.  

**Pull behavior:** Pull ALWAYS creates a new take. Replace and Keep Existing (D13) are retired. Pull = new take, that's it. User decides what to promote.  

**Takes are editable.** Users can move, resize, delete, and classify events within a take before promoting. This enables in-take curation (mark false positives, refine positions) before adding to Main.  

**Take → Main actions:**  
- **Add Selection to Main** — Copy selected events to Main (additive). No classification filtering — selection is literal.  
- **Add All to Main** — Copy all events to Main (additive).  
- **Overwrite Main** — Replace Main layer contents with this take's events.  
- **Mark as False/True** — Set classification state on events in-take (D14).  
- Standard event editing (move, resize, delete) within takes.  

**Scope:** Per layer, not per group or per connection. Each paired layer has its own independent take stack.  
**Take limit:** Unlimited. No auto-deletion.  
**All take/promote operations are editable commands (undoable).**  

**Layer pairing:** Tracks `(block_id, output_name)` → Editor layer. Pairing enables "new data available" indicators on the correct layer when upstream re-executes. Unpairing keeps existing takes but stops future pull indicators.  

**Rationale:** Event matching across re-executions is fundamentally unreliable — different settings produce genuinely different events, not "updated" versions. The Take System (inspired by DAW take lanes / comping) eliminates matching entirely. Users visually compare results and cherry-pick, which is how they actually work. D19 (merge deferred) is now superseded.

### D94: No untyped metadata escape hatch
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** Every block type defines fully typed settings. No `Dict[str, Any]` escape hatch. If a new block needs new settings, define them explicitly. If a genuine need arises post-MVP, add it then.  
**Rationale:** D1 killed untyped metadata for good reason. An escape hatch re-introduces the problem.

### D95: Progress reporting during PULL phase — yes
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** BlockExecutor reports progress during PULL phase using `phase: "pull"` (already supported by D54 ProgressReport schema). For small files it's instant and irrelevant. For large files (multi-track stems), the user gets feedback. Throttle at consumer per D55.  
**Rationale:** Basically free — the schema already supports it. Prevents frozen progress bars during large audio copies.

### D97: False event display — visible, visually distinct, toggleable
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
False events (FALSE/USER_FALSE) are displayed in both Main layers and Take layers with distinct visual treatment:  

**Visual hierarchy (strongest to weakest):**  
1. USER_TRUE — full opacity, full color (user explicitly confirmed)  
2. TRUE — full opacity, full color (algorithmically classified)  
3. UNCLASSIFIED — full opacity, neutral/gray color  
4. FALSE — semi-transparent, desaturated  
5. USER_FALSE — semi-transparent, desaturated, subtle X or strikethrough (strongest "no" signal)  

**Toggle:** Per-layer "Show False Events" toggle, persisted. Default ON in Takes (reviewing), OFF in Main (working). Keyboard shortcut for quick toggle.  

**Interaction:** False events are non-interactive by default (can't accidentally select while editing). When visible and explicitly clicked, they can be selected and reclassified (Mark as True). "Add Selection to Main" is literal — no classification filtering, false events included if selected.  

**Rationale:** False events are data, not garbage. Users need to see what was rejected, verify rejections, and reverse mistakes. Visual weight (bright = good, dim = rejected, gray = unreviewed) provides instant at-a-glance understanding.

### D96: Audio file path remapping and auto-save deferred to post-MVP
**Date:** 2026-03-09  
**Status:** Decided (deferred)  
**Decision:** Both features deferred. Neither has architectural impact.  
- **Path remapping:** "File not found" dialog with search/remap. Standard DAW pattern.  
- **Auto-save:** Timer-based periodic save to recovery location. Calls SaveProject to temp path.  
**Rationale:** No architectural decisions needed. Implementation details for post-MVP polish.

### D98: SetlistAudioInput block DELETED
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** SetlistAudioInput removed from block inventory. Batch processing sets `audio_path` directly on the existing LoadAudio block via operational command (not undoable, no stale cascade during batch). 11 block types remain.  
**Rationale:** LoadAudio already loads audio files. A separate block for setlist input is redundant.

### D99: No EZ1 project migration
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** EZ2 does not load EZ1 `.ez` projects. No block type alias map needed. Clean break — fundamentally different data model (Take System, new schemas, deleted blocks).  
**Rationale:** EZ1 was a prototype. Migration cost exceeds value.

### D100: FPS/timebase is per-project
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** FPS stored in Project global settings (D89). All Editor blocks and ShowManager inherit from project. One project = one frame rate. Default: 30fps.  
**Rationale:** A project = one show = one timecode format. Different frame rates within a project would break timecode sync.

### D101: Block rename has no cascade
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** D2 already ensures all references use IDs. Rename only updates block display name. No downstream references use names. Exported files not retroactively renamed.  
**Rationale:** Already solved by D2.

### D102: Batch executor uses block actions directly, never quick actions
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** Quick actions are UI-only (interactive operations). Setlist ActionSets reference block execution commands directly. Separate systems — no filtering or headless mode needed.  
**Rationale:** Clean separation between interactive UI operations and programmable batch operations.

### D103: Configuration validated continuously and at execution
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** `validate_configuration()` runs on every settings change (updating block status in real time) AND as a pre-execution gate in BlockExecutor before PULL phase. Invalid config = immediate ValidationError, no wasted computation. Double-check at execution is cheap insurance.  
**Rationale:** Users should see config errors instantly, not only when they try to execute.

### D104: ShowManager syncs Main layers only, creates layers via pipeline commands
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
ShowManager syncs Main layers only. Takes are working space, never synced.  

**Sync layer creation:** When user maps a sync (MA3 TC → Editor), ShowManager creates a layer in the Editor via pipeline command. From Editor's perspective, it's just a normal layer — Editor has zero knowledge of sync.  

**Editor stays dumb.** All sync intelligence lives in ShowManager:  
- ShowManager maintains sync registry: `{sync_mapping_id: (editor_layer_id, ma3_track_coord, resolution_policy, sync_status)}`  
- Editor publishes "layer changed" domain events as always (unaware of ShowManager)  
- ShowManager subscribes, detects changes in mapped layers, syncs to MA3  
- ShowManager receives MA3 updates, applies resolution policy, pushes events into Editor via pipeline commands  

**Constraints:**  
1. Both user and ShowManager can edit synced layers (normal Editor behavior)  
2. Conflict resolution is ShowManager's job entirely (D76 resolution policies)  
3. User CAN delete a synced layer — ShowManager detects deletion event, updates registry (status → Error/Unmapped)  
4. User CAN create events in synced layers — normal editing, ShowManager picks up changes  
5. ShowManager names layer at creation (e.g., "TC 1"), tracks by layer_id not name  
6. Synced layers do NOT get Takes — data comes from MA3, not upstream analysis  
7. **Sync status icon on layer** — UI adapter queries ShowManager for sync status per layer_id and renders icon (Synced/Diverged/Error). Purely a UI concern — Editor domain has zero awareness.  

**Rationale:** Keeps Editor as a pure Workspace block. All sync complexity is encapsulated in ShowManager. Editor just manages layers and events, same as always.

### D105: Editor data model — full schema
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
Four tables for Editor data:

**layers:** `id`, `editor_block_id` (FK), `name` (unique per editor), `layer_type` (MAIN|TAKE), `group_id` (FK, nullable), `order`, `height` (px, default 60), `color` (hex), `visible`, `locked`, `show_false_events` (default true for TAKE, false for MAIN), `source_type` (UPSTREAM|USER), `paired_block_id` (nullable), `paired_output_name` (nullable), `created_at`.

**takes:** `id`, `layer_id` (FK → MAIN layer this take belongs to), `label`, `source_block_id`, `source_settings_snapshot` (JSON), `order`, `visible`, `created_at`.

**events:** `id`, `layer_id` (FK, nullable), `take_id` (FK, nullable), `time` (seconds), `duration` (seconds, 0=marker), `classification_state` (UNCLASSIFIED|TRUE|FALSE|USER_TRUE|USER_FALSE), `confidence` (nullable float), `label` (nullable), `metadata` (JSON, nullable — analysis-specific payload like pitch, velocity, BPM), `created_at`. Constraint: event has EITHER layer_id OR take_id, never both, never neither.

**layer_groups:** `id`, `editor_block_id` (FK), `name`, `collapsed`, `order`. (Per D17.)

Event `metadata` JSON is analysis output data (varies by producing block type), NOT a settings escape hatch. Schema defined by block type, not user-configurable.  
**Rationale:** Clean relational model with clear ownership. Events belong to exactly one container (Main layer or Take). Takes belong to exactly one Main layer. Layer groups organize layers visually.

### D106: Undo granularity for Editor
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Multi-select operations are atomic.** Any operation on a selection of N events produces ONE undo entry that reverses the entire batch. Move 20 events = one undo. Delete 50 events = one undo that restores all 50. Classify 100 events = one undo.

**Single undo entry per operation:**  
- Create/delete/move/resize event(s) — batch if multi-selected  
- Create/delete/rename layer or take  
- Change event classification (single or batch)  
- Layer reorder, property changes (color, height, visibility, locked)  
- Promote events (Add Selection/All to Main, Overwrite Main)  
- Overwrite Main = atomic macro (clear + add = one undo step)  

**Drag operations commit on mouse-up.** No undo entries during drag. One entry for final delta on release. Prevents 60 undo entries per second of dragging.  

**NOT on undo stack:** Block execution, batch processing (operational commands per D32).  

**Global stack (D33).** Editor operations share stack with node graph operations. Standard DAW behavior.  
**Rationale:** Users expect multi-select operations to undo as one action, not N separate undos. Mouse-up commit is industry standard for drag operations.

### D107: Block deletion cascade — full specification
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Processor block deletion cascade:**  
- Block row → delete  
- Settings rows (keyed by block_id) → cascade delete  
- Connections (block is source or target) → cascade delete  
- Control connections → cascade delete  
- DataItems (block's outputs) → cascade delete + cleanup files  
- Downstream blocks → mark STALE (D88)  
- ActionItems referencing this block → cascade delete (not orphan)  

**Editor block deletion (additional):**  
- All layers, layer groups, takes, events owned by this Editor → cascade delete  
- ShowManager sync mappings referencing these layers → update to Unmapped/Error  

**ShowManager block deletion:**  
- Sync registry entries → cascade delete  
- Layers ShowManager created in Editor → REMAIN. Editor owns those layers. Events are local copies in Editor. Layers become normal unsynced layers — UI adapter finds no ShowManager to query, no sync icon renders. User can continue editing or delete layers manually.  

**All deletions are editable (undoable).** Delete command captures full snapshot before executing. Undo restores everything.  
**Rationale:** Editor always holds its own copy of event data (blocks are islands). ShowManager deletion = sync stops, data stays. No data loss from infrastructure changes.

### D108: Re-execution uses upsert semantics on DataItems
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:** When a Processor block re-executes:  
1. Each output DataItem has a `semantic_key` (D12)  
2. Engine looks up existing DataItems by `(block_id, semantic_key)`  
3. **Found → update content in place, keep same UUID.** Downstream references remain valid.  
4. **Not found → create new DataItem.**  
5. **Leftover old items not in new output → mark for GC (D28).**  

Multi-output blocks (e.g., Separator → drums, bass, vocals, other): all outputs upserted atomically in one STORE transaction. If execution fails mid-output → rollback entire STORE, old DataItems preserved (D27).  
**Rationale:** Stable UUIDs across re-executions mean downstream cached references don't break. Take System source references remain valid. D12 designed semantic_key for exactly this purpose.

### D109: Editor UI — complete interaction catalog
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  

#### Event Interactions (on timeline canvas)

**Creation:**  
- Double-click on empty space in a layer → create marker event (duration=0)  
- Double-click + drag → create clip event with duration  
- Target layer = whatever layer row the cursor is in  

**Selection:**  
- Click event → select (deselect others)  
- Shift+click → add/remove from selection  
- Ctrl/Cmd+click → toggle in selection  
- Rubber band drag on empty space → select all events in rectangle  
- Ctrl/Cmd+A → select all in active layer  
- Ctrl/Cmd+Shift+A → select all in all visible layers  
- Escape → deselect all  

**Moving:**  
- Drag selected event(s) → move horizontally (time) and/or vertically (across layers)  
- **Vertical drag snaps to layers** — events magnetically lock to the nearest layer row. Moving to a different layer row changes the event's layer assignment (data-level change, not just visual).  
- Drag threshold: 3px before move initiates (prevents accidental moves on click)  
- Escape during drag → cancel, snap back to original position  
- All moves on multi-selected events are atomic (D106)  

**Duplicating:**  
- **Option/Alt+drag** → duplicate event(s) and drag the copy. Original stays in place.  
- Ctrl/Cmd+D → duplicate selected events (offset slightly in time)  

**Copy/Paste:**  
- Ctrl/Cmd+C → copy selected events to clipboard  
- Ctrl/Cmd+V → paste at playhead position, into active layer  
- **Copy/paste works across layers** — copy from one layer, select a different layer, paste there  
- Ctrl/Cmd+X → cut (copy + delete)  
- Pasted events maintain relative time offsets if multiple selected  

**Resizing:**  
- Drag left/right edge of block event → resize (change duration)  
- Minimum duration constraint (e.g., 1 frame) to prevent zero-duration block events  

**Deleting:**  
- Delete/Backspace → delete selected events  

**Classification:**  
- Right-click → classify as TRUE/FALSE/USER_TRUE/USER_FALSE  
- Keyboard shortcut: T = mark true, F = mark false  
- Batch: select multiple → classify all (one undo)  

**Snapping (robust, single source of truth):**  
- **TimeGrid service** — single consolidated source of truth for all time calculations  
- Grid modes: auto, frame-based (1f, 2f, 5f, 10f per project FPS D100), second-based (0.1s, 0.5s, 1s)  
- Events snap to grid lines when dragging (magnetic behavior within threshold)  
- **Snap to other events** — edge-to-edge magnetic snapping (event start → event end of adjacent)  
- Hold Alt/Option → temporarily disable all snapping (free positioning)  
- Grid rendering derived from same TimeGrid service (visual grid always matches snap behavior)  
- Grid auto-adjusts density based on zoom level (don't render 1000 grid lines at max zoom-out)  

#### Layer Panel Interactions (left sidebar)

**Layer management:**  
- Click layer header → set as active layer (new events go here)  
- Double-click layer name → rename inline  
- Drag layer header → reorder within group  
- **Drag layers between groups** → move layer to different group  
- Eye icon → toggle visibility  
- Lock icon → toggle locked  
- Color swatch → open color picker  

**Layer groups:**  
- Click group header → collapse/expand  
- Double-click group name → rename  
- **Drag group header → reorder groups** (groups are movable, not just layers within groups)  
- Drag layer into/out of group  
- Right-click group → context menu  

**Take lanes:**  
- Expand arrow on Main layer → show/hide take stack  
- Click take header → select take (for browsing/editing within take)  
- Take lanes render below Main layer, slightly indented, visually distinct (lighter background)  
- "New data available" badge/indicator on Main layer when upstream re-executes and new data is waiting  
- Take lanes show event count and settings snapshot info in header  

#### Event Type Settings & Visibility

**Per-layer event display settings:**  
- Show/hide marker events (duration=0)  
- Show/hide clip events (duration>0)  
- Show false events toggle (D97, default ON in Takes, OFF in Main)  
- Marker shape selector per layer (diamond, circle, star, triangle, line)  
- Event label display: off, on hover, always  
- Waveform in events: off, on (per layer)  

**Per-event-type visual config:**  
- Colors can be assigned by classification state (beyond the opacity hierarchy in D97)  
- Event height within layer can be uniform or scaled by confidence value  

#### Transport & Playback

- Play/Pause (spacebar toggle)  
- Stop (return to 0 or loop start)  
- Rewind / Fast Forward  
- Click on ruler → set playhead position  
- Loop toggle with draggable loop region markers on ruler  
- Current time display (timecode HH:MM:SS:FF per project FPS)  
- Follow modes: off, page (jump), smooth (gradual), center (playhead locked to center)  
- Playhead renders at 60 FPS during playback  

**⚠️ NOTE — OPEN DISCUSSION NEEDED:** Timing and sync across clock, playhead, and audio playback. How does the playhead clock stay in sync with audio output? What's the master clock? Does audio playback drive the playhead, or does a shared clock drive both? Latency compensation? This needs its own dedicated design session.

#### Navigation

- Pan: scroll wheel horizontal, middle-mouse drag, spacebar+drag  
- Zoom: scroll wheel vertical (cursor-centered, 10-1000 px/sec range)  
- Zoom to fit: keyboard shortcut  
- Zoom to selection: keyboard shortcut  
- Home → go to time 0  
- End → go to last event  

#### Standard Keyframe Editor Patterns (reference: Qt Design Studio Timeline, After Effects, Premiere, Logic Pro)
- All time operations internally use a high-precision float (seconds) as canonical format  
- Display format adapts to project FPS (frames) or seconds based on user preference  
- Interpolation between keyframes is NOT applicable (EchoZero events are discrete, not animated properties) — but the selection, move, copy/paste, snap, and multi-select patterns match standard keyframe editor UX  
- Cursor changes on hover: move cursor over event body, resize cursor over edges, crosshair on empty space  

#### Backend Pattern for All UI Operations

Every UI interaction maps to a pipeline command:  
- Event creation → `CreateEvent` command  
- Event move → `MoveEvents` command (batch, contains all selected event deltas)  
- Event resize → `ResizeEvent` command  
- Event delete → `DeleteEvents` command (batch)  
- Event classify → `ClassifyEvents` command (batch)  
- Event duplicate → `DuplicateEvents` command (batch, creates new events)  
- Copy/paste → `PasteEvents` command (creates new events at target layer + time)  
- Layer operations → `CreateLayer`, `DeleteLayer`, `RenameLayer`, `ReorderLayer`, `UpdateLayerProperties`  
- Group operations → `CreateGroup`, `DeleteGroup`, `RenameGroup`, `ReorderGroup`, `MoveLayerToGroup`  
- Take operations → `CreateTake` (via pull), `DeleteTake`, `RenameTake`, `ReorderTake`, `PromoteEvents`, `OverwriteMain`  

All commands are frozen dataclasses (D31), editable/undoable (D32), dispatched through the same pipeline (D30). UI adapter translates mouse/keyboard events into commands. **One pattern, one path, every operation.**

**Rationale:** Comprehensive interaction catalog ensures nothing is missed during implementation. Single pipeline pattern for all operations ensures consistency, undo support, and testability.

### D110: Audio layers in Editor — read-only playback views from graph
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
Editor layers can now contain **events OR audio**. Two layer categories:  
- **Event layers** — editable events/cues (existing behavior). Support Take System, classification, all editing operations.  
- **Audio layers** — read-only waveform visualization of audio flowing through the graph via audio input ports. NOT editable (no trim, split, move, crossfade, time-stretch).  

**Audio layer properties:** `source_block_id`, `source_output_name`, `muted` (default false), `volume` (0.0-1.0, default 1.0), `visible`, `color`, `order`, `name` (default: source block + output, e.g., "Separator: vocals").  

**Playback behavior:**  
- Per-layer mute/unmute and volume control  
- Solo = derived from mute (solo = mute everything except this)  
- All unmuted audio layers sum to stereo master output  
- No buses, effects, or routing at MVP. Architecture allows adding later.  
- Audio layers do NOT get Takes — they reflect current graph audio.  

**Rationale:** Users need to hear individual stems while editing cues. Separator → drums/bass/vocals/other should be visible and independently mutable. Audio stays read-only because EchoZero is a cue editor, not a DAW arranger.

### D111: Audio engine — PortAudio via sounddevice, audio callback as master clock
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Audio engine:** `sounddevice` (Python wrapper for PortAudio). Cross-platform, talks to ASIO (Windows), CoreAudio (macOS), ALSA/PulseAudio (Linux). Battle-tested (Audacity uses PortAudio).  

**Master clock:** Audio callback drives everything (Approach 1 — same as Reaper, Ableton, every serious DAW).  
- System audio driver calls callback every N samples (e.g., 512 @ 44.1kHz ≈ 11.6ms)  
- Callback reads sample position, sums unmuted audio layers with volume, writes to output buffer, advances position atomically  
- Sample counter is THE single source of truth for playback position  
- UI thread reads position via atomic/lock-free variable at 60Hz, renders playhead  
- UI interpolates between reads for smooth visual movement  

**Playback block (D91) becomes the audio engine host.** It manages the `sounddevice` stream, owns the sample counter, and coordinates with the Editor's audio layers.  

**When audio is NOT playing:** System timer drives playhead position (scrubbing, manual positioning). Audio engine position is updated on play to match.  

```python
def audio_callback(outdata, frames, time_info, status):
    pos = engine.position
    for layer in unmuted_audio_layers:
        chunk = layer.read_samples(pos, frames)
        outdata += chunk * layer.volume
    engine.position += frames
```

**Rationale:** Multi-track summing with mute/solo requires a real audio engine — QMediaPlayer can't mix N streams. PortAudio is the industry standard for cross-platform audio I/O. Audio callback as master clock guarantees sample-accurate sync between audio output and playhead. EchoZero's use case (summing a few stems with gain) is trivially within PortAudio's performance envelope.

### D112: Clip events as playable audio clips with waveforms
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Terminology:** Events with duration > 0 are **clip events**. Events with duration = 0 are **marker events**.  

Clip events can function as audio clips:  
- **Waveform display:** Clip events render the source audio waveform inside the event rectangle. Audio slice = `source_audio[event.time : event.time + event.duration]`. Provides visual verification of analysis results (see the transient shape inside a kick event).  
- **Audio playback:** During timeline playback, the audio engine triggers clip event audio at each event's time position. Event layers participate in the same mute/solo/volume system as audio layers.  
- **Opt-in per layer:** "Enable Audio Playback" toggle on event layers. Default OFF. User enables for analysis result layers where hearing the clips adds value.  

**Marker events (duration=0):** No audio clip, no waveform. Time markers only.  

**Audio source:** D7 bundles source audio as a snapshot copy with events through the connection. The audio is already available — clip is just a slice at the event's time/duration.  

**Audio engine integration (D111):** Callback sums both continuous audio layers AND triggered clip event audio from event layers, applying per-layer volume and mute/solo.  

**Rationale:** Seeing AND hearing analysis results makes review dramatically faster. A user can instantly verify onset detection by seeing waveform transients inside events and hearing them during playback. Opt-in prevents unexpected audio from non-analysis layers.

### D113: Snap system — TimeGrid service as single source of truth
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**TimeGrid service** — one object owns all time quantization logic. Grid rendering, snap behavior, and timecode display ALL derive from this single service. No separate grid logic.

**Snap modes (user-selectable):**  
- **Off** — no snapping, free positioning  
- **Auto** — grid density auto-adjusts based on zoom level (fewer lines when zoomed out, more when zoomed in)  
- **Frame-based** — 1f, 2f, 5f, 10f (based on project FPS, D100)  
- **Time-based** — 0.1s, 0.25s, 0.5s, 1s  
- **Beat-based** — 1/4, 1/8, 1/16 (requires BPM, optional, derived from analysis)  

**Snap behavior:**  
- Magnetic threshold: event snaps to nearest grid line when within N pixels (configurable, default 8px)  
- **Snap to grid** — primary snap target  
- **Snap to events** — secondary, edge-to-edge magnetic snapping (event start → nearby event end, and vice versa). Can be toggled independently.  
- **Snap to playhead** — tertiary, events snap to playhead position  
- Hold Alt/Option → temporarily disable ALL snapping  
- Snap applies to: move, resize, create, paste, duplicate  

**Grid rendering:**  
- Derived from same TimeGrid service (visual grid ALWAYS matches snap behavior)  
- Major lines: thicker, labeled with timecode  
- Minor lines: thinner, no label  
- Auto-density: grid lines fade in/out based on zoom level. Never render >1 line per 4 pixels.  
- Beat grid overlay: if BPM is set, show beat markers as a distinct color  

**Time representation:**  
- Internal: `float` (seconds, high precision)  
- Display: adapts per user preference — timecode (HH:MM:SS:FF at project FPS), seconds (SS.mmm), or bars:beats (if BPM set)  
- All conversions go through TimeGrid service  

**Rationale:** One service, one truth. Grid rendering and snap behavior can never disagree. Magnetic snapping with modifier override is industry standard (After Effects, Premiere, Logic, Reaper).

### D114: Event Inspector panel
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
Dockable panel that shows details of the selected event(s).

**Single event selected:**  
- Time (editable — type exact timecode to reposition)  
- Duration (editable — type exact duration)  
- Layer (display, with dropdown to move to different layer)  
- Classification state (dropdown: UNCLASSIFIED/TRUE/FALSE/USER_TRUE/USER_FALSE)  
- Confidence (display, from analysis)  
- Label (editable text field)  
- Metadata (read-only display of analysis payload — pitch, velocity, BPM, etc.)  
- **Audio clip preview:** Mini waveform + play button for clip events. Plays the audio slice for this event. Independent of timeline playback.  
- Source info: which upstream block produced this event, which take it came from  

**Multiple events selected:**  
- Count display ("N events selected")  
- Shared fields show common value or "mixed"  
- Classification dropdown applies to all selected (batch, one undo per D106)  
- Bulk time offset: shift all selected by ±N seconds/frames  
- Bulk duration scale: scale all durations by percentage  

**No events selected:**  
- Panel shows layer info for active layer (name, event count, audio playback status)  

**Rationale:** Precise numerical editing complements mouse-based editing. Audio preview in inspector enables rapid event-by-event review without playing the full timeline.

### D115: Waveform rendering in events and audio layers
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Waveform computation:**  
- Background worker computes waveform peaks from audio data  
- Stored in in-memory LRU cache keyed by `(audio_source_id, sample_rate)`  
- Multiple LOD (level of detail) levels pre-computed: 1 peak per 64/256/1024/4096 samples  
- Cache invalidated when DataItem UUID changes (re-execution with new output)  
- Cache size bounded (configurable, default 256MB). LRU eviction.  
- Computation is async — UI shows placeholder (flat line or loading indicator) until ready  

**Rendering:**  
- LOD selected based on current zoom level (pixels per second)  
- Zoomed out → coarse LOD (fast render, less detail)  
- Zoomed in → fine LOD (more detail, see individual transients)  
- Waveform drawn as filled polygon (min/max peak pairs per pixel column)  
- Color: layer color with slight transparency  
- RMS overlay: optional secondary waveform showing RMS energy (thicker, more opaque)  

**Audio layers:** Full-width waveform spanning entire audio duration.  
**Clip events:** Waveform inside event rectangle, clipped to event boundaries.  

**Thread safety:** Waveform cache is read by UI thread, written by background worker. Thread-safe via read-write lock or concurrent dict.  

**Rationale:** LOD prevents rendering millions of samples at full zoom-out. LRU cache prevents unbounded memory growth. Async computation keeps UI responsive during first load.

### D116: Keyboard shortcuts — default mapping
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
All shortcuts are user-configurable via Settings panel. Defaults:

**Transport:**  
- Space — Play/Pause toggle  
- Home — Go to start  
- End — Go to last event  
- L — Toggle loop  

**Selection:**  
- Ctrl/Cmd+A — Select all in active layer  
- Ctrl/Cmd+Shift+A — Select all in all visible layers  
- Escape — Deselect all  

**Editing:**  
- Delete/Backspace — Delete selected  
- Ctrl/Cmd+D — Duplicate selected  
- Ctrl/Cmd+C — Copy  
- Ctrl/Cmd+V — Paste  
- Ctrl/Cmd+X — Cut  
- Ctrl/Cmd+Z — Undo  
- Ctrl/Cmd+Shift+Z — Redo  

**Classification:**  
- T — Mark selected TRUE  
- F — Mark selected FALSE  
- U — Mark selected UNCLASSIFIED  

**View:**  
- Ctrl/Cmd+0 — Zoom to fit  
- Ctrl/Cmd+Shift+0 — Zoom to selection  
- +/= — Zoom in  
- - — Zoom out  
- G — Toggle grid visibility  
- S — Toggle snap on/off  

**Layers:**  
- Ctrl/Cmd+Shift+N — New layer  

**Snapping modifier:**  
- Alt/Option (held) — Temporarily disable snap  

**Customization model:** JSON config file mapping action names → key combos. Settable via UI preferences panel. Conflicts detected and warned.  

**Rationale:** Industry-standard defaults (matching DAW/NLE conventions). Full customization for power users.

### D117: Context menus — full action catalog
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  

**Right-click on event(s):**  
- Cut / Copy / Paste / Duplicate  
- Delete  
- Classify → TRUE / FALSE / USER_TRUE / USER_FALSE / UNCLASSIFIED  
- Move to Layer → [submenu of available layers]  
- Add to Main (if in a take)  
- Select All in Layer  
- Properties (open Event Inspector)  

**Right-click on empty timeline space:**  
- Create Marker Event  
- Create Clip Event  
- Paste  
- Select All in Layer  
- Zoom to Fit  

**Right-click on layer header:**  
- Rename  
- Duplicate Layer  
- Delete Layer  
- Change Color  
- Toggle Visibility  
- Toggle Lock  
- Toggle Audio Playback (event layers only)  
- Toggle Show False Events  
- Move to Group → [submenu]  
- Remove from Group  

**Right-click on take header:**  
- Add Selection to Main  
- Add All to Main  
- Overwrite Main  
- Rename Take  
- Delete Take  
- Toggle Visibility  

**Right-click on layer group header:**  
- Rename Group  
- Delete Group (layers become ungrouped)  
- Collapse/Expand  
- Add New Layer to Group  

**Right-click on audio layer header:**  
- Mute/Unmute  
- Solo  
- Adjust Volume (slider in context menu or opens inspector)  
- Toggle Visibility (hide waveform)  
- Change Color  

**Rationale:** Every operation accessible via both keyboard shortcut AND context menu. Context menus are discoverable; shortcuts are fast.

### D118: Multi-select behavior
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
**Selection methods:**  
- Click — select single, deselect others  
- Shift+click — add/remove from selection  
- Ctrl/Cmd+click — toggle in selection  
- Rubber band — drag on empty space creates selection rectangle, selects all events within  
- Rubber band + Shift — add to existing selection  
- Ctrl/Cmd+A — select all in active layer  
- Ctrl/Cmd+Shift+A — select all visible layers  

**Multi-select operations (all atomic, one undo per D106):**  
- Move — all selected events move by same delta (time and/or layer)  
- Resize — NOT supported on multi-select (ambiguous which edge). Resize is single-event only.  
- Delete — delete all selected  
- Duplicate — duplicate all selected, maintaining relative positions  
- Copy/Paste — copies all selected with relative time offsets preserved  
- Classify — apply classification to all selected  
- Move to Layer — move all selected to target layer  

**Cross-layer selection:** Selection can span multiple layers. Visual indicator (highlight) on each layer that has selected events.  

**Selection persistence:** Selection survives zoom, scroll, and layer reorder. Selection clears on: click empty space, Escape, or undo/redo that affects selected events.  

**Rationale:** Cross-layer multi-select is essential for batch operations. Resize excluded from multi-select to avoid UX confusion (which edge? proportional?). All batch operations are one undo step.

### D119: ShowManager UI — sync configuration panel
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
ShowManager panel (opened from node graph) has three sections:

**1. Connection Status:** Connected/disconnected indicator, MA3 IP + port, Connect/Disconnect buttons, gear icon for settings (IP, port, auto-connect on project open, auto-reconnect interval, default resolution policy, sync debounce interval).

**2. Sync Layers List:** Each sync layer displays: MA3 track coordinate (TC N), mapped Editor layer name (or "unmapped"), sync status icon (Synced/Diverged/Error/Unmapped), resolution policy dropdown (MA3 Master, EZ Master, Union, Ask User per D76), manual Push/Pull buttons, event count. "Map Layer" opens dropdown of available Editor layers. "Add Sync Layer" creates new mapping. Delete via right-click or X.

**3. Activity Log:** Scrollable timestamped log of sync activity (pushes, pulls, conflicts, errors, connections). Filterable by type. Session-only, not persisted.

**Rationale:** All sync configuration in one panel. Activity log provides transparency into what ShowManager is doing behind the scenes.

### D120: ShowManager ↔ Editor event flow — speed-optimized
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  
Priority is speed. Minimal hops between MA3 and Editor.

**Wire protocol:** OSC over UDP (mandatory — MA3 only speaks OSC). Use minimal OSC parser (`python-osc` or custom ~100 lines). No bloated OSC libraries.

**ShowManager owns the socket directly.** D82 (OSC Gateway as separate service) is superseded. ShowManager holds the UDP socket, runs recv loop, parses OSC, applies sync. No intermediate gateway service, no unnecessary domain event hops for sync data.

**Superseded decisions:** D82 (gateway not a block), D84 (IOscGateway protocol), D85 (visual status node — ShowManager block IS the status node), D86 (one gateway multiple endpoints — ShowManager handles its own config). D83 (singleton) survives — one ShowManager per project.

**Inbound fast path (MA3 → EZ):**  
1. UDP socket receives OSC packet on network thread  
2. Minimal OSC parse → internal event format  
3. ShowManager applies resolution policy  
4. **Direct write to Editor layer event store** (bypasses pipeline command/validation/undo — these aren't user edits)  
5. Batch write wrapped in transaction (data integrity)  
6. ONE domain event emitted after batch: `SyncLayerUpdated(layer_id)` — UI repaints  
7. No undo stack entry (incoming sync is not undoable)  
Target: <50ms from UDP receive to visual update.

**Outbound fast path (EZ → MA3):**  
1. Editor domain event `LayerEventsChanged` fires normally on user edit  
2. ShowManager catches event, starts debounce timer (100ms)  
3. More edits within window → reset timer  
4. Timer fires → batch ALL pending changes → serialize to OSC → UDP send  
5. During drag operations: no sync. Sync on mouse-up (command commit).  
Target: <200ms from edit to MA3 receiving.

**User-initiated sync (Push/Pull buttons in panel):**  
Full pipeline path. Dispatches PushLayerToMA3 / PullLayerFromMA3 commands. Goes through middleware, logged in activity log. These are intentional user actions, not real-time auto-sync.

**Surviving decisions:** D76 (resolution policies), D77 (sync commands for user actions), D78 (config not wiring), D79 (bidirectional), D80 (4 statuses), D81 (fingerprint matching), D83 (singleton), D87 (sync logic — now with direct network access).

**Rationale:** OSC Gateway as a separate service added a hop with zero benefit (only one consumer). ShowManager with direct socket access cuts the inbound path to 2 hops. Hybrid fast path avoids pipeline overhead for high-frequency sync updates while maintaining transaction safety.

### D121: ShowManager ↔ Editor sync flow + conflict resolution via Takes
**Date:** 2026-03-09  
**Status:** Decided  
**Decision:**  

**Sync flows:**  
- **Outbound (EZ → MA3):** Editor publishes LayerEventsChanged → ShowManager catches, debounces (100ms), batches changes → serializes to OSC → UDP send. No sync during drags (sync on mouse-up commit).  
- **Inbound (MA3 → EZ):** UDP receive → OSC parse → ShowManager applies resolution policy → writes to Editor layer.  
- **User-initiated (Push/Pull buttons):** Full pipeline path with logging.  

**Per-policy behavior for inbound:**  
- MA3 Master → direct overwrite of Main layer (fast path)  
- EZ Master → ignore incoming, push local on change  
- Union → additive merge to Main layer  
- Ask User → **create a sync Take** on the mapped layer containing incoming MA3 events. User resolves via standard Take System (Overwrite Main = Take Theirs, delete take = Keep Mine, Add Selection = cherry-pick). No modal dialog needed.  

**Connection loss:** All sync layers → Error status. Auto-reconnect if enabled. On reconnect, full diff on all sync layers, handle per policy.  

**Event identity (MA3 limitation):** MA3 has no stable per-event UID. Events are identified by positional coordinates `(timecode_no, track_group, track, event_layer, event_index)` which change on reorder. **OPEN ITEM:** Investigate MA3 Lua API for workarounds (embed UUID in event name, plugin-side lookup table, custom properties). If no workaround found, fall back to full-state-replacement with best-effort move detection via `(cmd + name)` fingerprint matching.  

**Rationale:** Take-based conflict resolution reuses existing UI with zero new components. Full-state replacement is simple and reliable given MA3's identity limitations. Move detection heuristic covers 90% of cases.

---

## Sprint 5: ML Classification & Model Management (2026-03-12)

### D122: Classify block — core design
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  
New Processor block type: `Classify`. Model-agnostic — the block is a generic "run this PyTorch model against events + audio" machine. The model defines what it classifies.

**Ports:** Input: `events` (Event) + `audio` (Audio). Output: `events` (Event — same events enriched with classification + confidence).  
**Settings:** `model_category: str` (e.g., `"drums.full-mix"`), `confidence_threshold: float` (default 0.5), `batch_size: int` (default 64), `split_layers: bool` (default true), `inference_mode: "auto" | "local" | "cloud"` (default auto).  

**Processing logic:**  
1. Pull events + audio  
2. For each event, extract audio window around event time (window size defined by model metadata — 100ms for drums, 500ms for tonal, 2-4s for energy/structure)  
3. Batch windows into tensors  
4. Run inference via InferenceBackend protocol (D124)  
5. Apply confidence threshold — below threshold → UNCLASSIFIED  
6. Write classification label + confidence to event metadata  
7. If `split_layers=true`, auto-split output into per-label layers (one for kicks, one for snares, etc.)  
8. Output enriched events  

**Input flexibility (Option C):** Works on both full-mix audio and separated stems. Accuracy improves with separated stems. Recommended workflow: Separator → drums stem → DetectOnsets → Classify. Block handles either case — model quality determines accuracy.  

**What the model defines:** Input window size, sample rate requirement, output label taxonomy, architecture.  
**What the block defines:** Window extraction, batching, threshold application, progress reporting, event metadata schema, layer splitting.  
**Adding a new model type = train + package + publish. Zero block code changes.**  
**Rationale:** Model-agnostic design means the block never changes. All intelligence lives in the models. Category-based selection (D123) means users think in terms of "what do I want classified" not "which model version."

---

### D123: Model registry — category lanes + auto-update on Cloudflare
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Three-tier model storage:**  
- **Tier 1: Bundled** — 2-3 core models ship with installer. Work offline, day one.  
- **Tier 2: Cloud registry** — JSON manifest on Cloudflare R2 + CDN at `models.echozero.app/registry.json`. Models download to `~/.echozero/models/`.  
- **Tier 3: Custom** — User drops `.ezmodel` files into `~/.echozero/models/custom/`. Shows in model selector alongside registry models.  

**Category lanes:** Users pick categories, not model versions. Each category is a stable identifier mapping to "best current model for this job":  
- `drums.full-mix`, `drums.separated`, `tonal.full-mix`, `energy.structure`, etc.  
- Category dropdown in block settings shows: name, description, current version, last updated.  
- Optional "Advanced" toggle to pin a specific version (for reproducibility).  

**Auto-update behavior:**  
- Default: auto-update ON per category. New model downloads in background. Next block execution uses new model.  
- Staleness indicator after update: "Model updated since last run. Re-run for latest results."  
- Pin option: user pins category to specific version. Auto-update skips pinned categories. For "I'm mid-tour, don't change anything."  
- Check timing: on first use of Classify block, not on app startup. Periodic background check configurable.  

**Registry manifest schema:**  
```json
{
  "schema_version": 1,
  "updated_at": "2026-03-12T00:00:00Z",
  "categories": [
    {"id": "drums.full-mix", "name": "Drum Classifier (Full Mix)", "description": "...", "latest": "3.1.0", "min_app_version": "2.0.0"}
  ],
  "models": [
    {"id": "drums-full-mix-v3.1.0", "category": "drums.full-mix", "version": "3.1.0", "taxonomy": [...], "window_ms": 100, "sample_rate": 22050, "size_mb": 12, "sha256": "...", "url": "https://models.echozero.app/...", "released_at": "...", "changelog": "..."}
  ]
}
```

**Hosting:** Cloudflare R2 (storage, S3-compatible, free egress) + Cloudflare CDN (delivery).  
**Update flow:** Train model → package as `.ezmodel` → upload to R2 → update registry.json → all instances pick up on next check.  
**Rationale:** Category lanes abstract version complexity from users. Auto-update keeps accuracy improving without user action. Pin option protects live production environments.

---

### D124: Inference abstraction — local vs cloud backend
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**`InferenceBackend` protocol:**  
```python
class InferenceBackend(Protocol):
    def classify(self, windows: list[AudioWindow], model_id: str, category: str) -> list[ClassificationResult]: ...
    def is_available(self) -> bool: ...
    def estimated_latency(self, batch_size: int) -> float: ...
```

**Two implementations:**  
- `LocalInferenceBackend` — loads PyTorch model from cache, runs on CPU/CUDA/MPS. Ships at launch.  
- `CloudInferenceBackend` (future) — sends audio windows to `api.echozero.app/v1/classify`. Requires auth. Handles large models, GPU-accelerated inference, experimental models.  

**Backend selection:** Injected at bootstrap time. Classify block doesn't know which backend it's using. Selection based on: user preference (`inference_mode` setting), model metadata (`"inference": "cloud"` for cloud-only models), availability (no internet → local).  

**Cloud API contract (for future, but boundary matters now):**  
POST `/v1/classify` with category, optional model version, audio windows (base64). Returns label + confidence + alternatives per event.  

**Graceful fallback:** Cloud unavailable → fall back to local if model exists locally. Local model too large → suggest cloud. Both unavailable → clear error.  
**Rationale:** Protocol boundary means swapping local↔cloud is config change, not rewrite. Cloud becomes natural upgrade path for models too large for user hardware.

---

### D125: Model validation, cache management, graceful degradation
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Validation on load:**  
- SHA256 checksum against manifest (registry models)  
- Metadata schema validation  
- `torch.load` with `weights_only=True` (prevents arbitrary code execution)  
- Custom models (Tier 3): same validation minus checksum. One-time warning on first load.  

**Cache structure:**  
```
~/.echozero/models/
  registry/
    drums.full-mix/
      3.1.0.ezmodel
      3.0.0.ezmodel    ← kept for rollback
    tonal.full-mix/
      1.0.0.ezmodel
  custom/
    my-experimental.ezmodel
```

**Retention:** Current + one previous version per category. Older auto-cleaned. Pinned versions never cleaned. `max_cache_mb` setting (default 2GB) with warning before exceeding.  

**Graceful degradation:**  
- Corrupted model file → auto re-download, log incident  
- NaN/Inf outputs → catch, classify affected events as UNCLASSIFIED, warn user, don't crash pipeline  
- App version too old for model → skip with clear error  
- No internet + model not cached → error status: "Model not available offline"  
**Rationale:** Defense in depth. Never crash the pipeline due to model issues. Always recoverable.

---

### D126: GPU detection, performance, cancellation
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Compute device detection:** On first launch + on demand. Detect CPU, CUDA (NVIDIA), MPS (Apple Silicon). Stored in app preferences. Setting: `compute_device: "auto" | "cpu" | "cuda" | "mps"` (default auto). Auto = best available.  

**Memory guard:** Before GPU model load, check available VRAM. Model too large → fall back to CPU with notification. Never OOM crash.  

**Batching:** Default 64, tunable per model via metadata (`recommended_batch_size`). Progress reported per-batch.  

**Cancellation:** Check token between batches. Already-classified events keep labels, remaining stay UNCLASSIFIED.  

**Warm-up:** Single dummy inference on model load to warm PyTorch JIT. Loaded models cached in memory for session duration. Evict on block deletion or category change.  
**Rationale:** GPU is 10-50x faster than CPU for inference. Memory guards prevent the #1 cause of ML app crashes.

---

### D127: Classification result schema + per-label layer split
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Per-event metadata after classification:**  
```python
{"classification": {"label": "kick", "confidence": 0.94, "model_category": "drums.full-mix", "model_version": "3.1.0", "alternatives": [{"label": "tom", "confidence": 0.04}, {"label": "snare", "confidence": 0.02}]}}
```

**Model provenance on every event** (`model_category` + `model_version`) enables: reproducibility, identifying events from old model versions, future "re-classify stale events only" optimization.  

**Integration with D22 classification states:** Classify block sets TRUE + label. Below threshold → UNCLASSIFIED. User override → USER_TRUE/USER_FALSE (protected from re-classification per D22). Re-running Classify skips USER_ events.  

**Layer organization: Option B — per-label layer split (default ON).**  
Output auto-splits into one layer per classification label. "kicks" layer, "snares" layer, etc. Each mapped separately to MA3. Setting `split_layers: bool` (default true) with opt-out to single mixed layer.  
**Rationale:** Per-label layers match the LD's workflow — different lighting responses for different instruments. Mixed layer requires manual sorting. Per-label is what users want 95% of the time.

---

### D128: Telemetry basics
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Opt-in anonymous telemetry:** When user overrides a classification, that's a signal the model was wrong. With opt-in: send `{category, version, predicted_label, corrected_label}` — no audio, no project data.  

**Local quality metrics (always on, no network):**  
- Per-model accuracy tracker based on user corrections  
- Shown in model settings: "drums.full-mix v3.1.0 — 94% accuracy (847 reviewed events)"  

**Setting:** `telemetry: "off" | "anonymous" | "full"` (default off, prompted on first correction).  
**Rationale:** Even just correction pair data (no audio) enables targeted training improvements. Local metrics build user trust in models.

---

### D129: User-powered model improvement flywheel
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Zero-friction data capture:** Users naturally correct misclassifications during editing (drag between layers, right-click reclassify, delete false positives). All stored as USER_TRUE/USER_FALSE per D22. No extra work.  

**Three contribution tiers:**  
- **Tier 1 (correction stats):** Label pairs only `{predicted, corrected, category, version}`. No audio. Lightweight.  
- **Tier 2 (correction + audio):** Correction pair + the misclassified audio window (~4KB each). Separate explicit opt-in. Actual training data.  
- **Tier 3 (full export):** "Export → Training Data" packages all classified events + audio windows + corrections as `.eztraining` file. Power user / community contribution.  

**Privacy design:**  
- Default OFF. All tiers.  
- First correction triggers non-modal banner: "Help improve models? [Yes / Not now / Never]"  
- Tier 2 (audio) requires separate explicit opt-in  
- All data stripped of: filename, project name, artist name, timestamps, user identity  
- Clear data policy page  

**Incentives:** Early access to new model versions (1 week), Discord leaderboard, in-app "Model Contributor" badge. Real incentive: their models get better.  

**Data quality:** Outlier detection on contributions — if one user's corrections consistently disagree with all others, flag for review. Contribution quality score (internal).  

**Flywheel:** Users classify → correct → corrections collected → training pipeline ingests → better model → published → auto-pulled → fewer corrections needed → cycle continues.  
**Rationale:** Every correction is free labeled training data. The more users, the better the models. Data flywheel is the competitive moat.

---

### D130: Stats & visualization
**Date:** 2026-03-12  
**Status:** Decided  
**Decision:**  

**Per-block stats (Classify block panel, after every run):**  
- Total events classified  
- Per-label distribution bar chart  
- Confidence distribution histogram  
- Below-threshold count  
- Processing time + device used  

**After user corrections (accumulates over project):**  
- Accuracy by label: kick 97%, snare 91%, hihat 88%, tom 72%  
- Confusion matrix heatmap (what gets confused with what)  
- Correction rate over time (proves model improvement)  

**Editor timeline visual treatments:**  
- Confidence-based opacity: high confidence = opaque, low = semi-transparent  
- Color by classification: each label gets design system palette color, consistent across projects  
- Uncertainty markers: "?" badge on events between threshold and review threshold (0.5-0.7)  
- Confidence heatmap overlay on waveform (optional): red = uncertain, green = confident  

**Project-level stats ("Model Performance" panel):**  
- Models used across all blocks  
- Overall accuracy per model  
- Recommended actions: "v3.2.0 improves hihat by 15%. [Update]"  
- Export stats as CSV  

**Historical stats (user preferences, cross-project):**  
- Lifetime accuracy per model category  
- Total events classified  
- Contribution stats (if opted in)  
- Model version comparison: "v3.0.0: 89% → v3.1.0: 94% on your data"  

**Model recommendation (future):** If >50% events below threshold, suggest alternative category. Analyze audio features → recommend best model.  
**Ensemble mode (future):** Run N models, majority vote. Higher accuracy than single model.  
**Rationale:** Visualization builds trust. Users need to see the ML is working, where it struggles, and that it's improving. Black box AI = no adoption. Transparent AI = power tool.

---

## Reference Documents
- `SPEC.md` — Behavioral specification (from legacy codebase)
- `docs/architecture/ARCHITECTURE.md` — Vertical feature module architecture
- `docs/architecture/OBSERVABILITY.md` — Universal observability design
- `sass/seminars/` — S.A.S.S. panel analyses and syntheses
