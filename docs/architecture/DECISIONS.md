# EchoZero - Architecture Decisions Log

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
**Status:** Superseded by D92
**Decision:** Bidirectional communication (ShowManager ↔ Editor) modeled as its own concept, separate from the port/connection directed graph.
**Rationale:** A "bidirectional port" breaks the directed graph contract. It's a channel, not a port.

### D4: Two block types - Processor and Workspace
**Date:** 2026-03-01
**Status:** Decided
**Decision:**
- **Processor Block:** Pure transform. Pull inputs (copy) → execute in background → store outputs. Isolated, idempotent.
- **Workspace Block:** Manages own data. No execution pipeline. Manual pull with user-controlled merge options. Editor is the primary Workspace block.
**Note:** D91 adds a third category (Playback). See D91 for full three-category model.
**Rationale:** Fundamental behavioral split between blocks that derive data (most blocks) and blocks that curate/manage data (Editor, potentially others).

### D5: Non-blocking execution with UI isolation
**Date:** 2026-03-01
**Status:** Decided
**Decision:**
- Processors run on background worker thread, never touch Qt or event bus
- Progress/completion via Qt signals marshaled to main thread
- UI stays responsive during execution
- Execution lock prevents conflicting runs on same block
- No need for true parallelism or subprocess isolation initially - just non-blocking
**Rationale:** Griff's requirement: "I still want to be able to execute other blocks, change settings and use the rest of the application."

### D6: Copy-on-pull - each block is an island
**Date:** 2026-03-01
**Status:** Decided
**Decision:** When a block pulls upstream data, it receives a COPY, not a reference. Each block's stored data is fully self-contained. Deleting or re-running an upstream block does not affect downstream blocks' existing data.
**Rationale:** Eliminates hidden dependencies. Makes every block an isolated, portable unit of data.

### D7: Events are pure time data - audio bundled as snapshot
**Date:** 2026-03-01
**Status:** Decided
**Decision:**
- EventDataItems contain time/duration/classification data only
- No `audio_id` cross-references to other blocks' data
- When events are produced, the source audio is copied and bundled with them through the connection
- Downstream blocks receive events + audio snapshot through the same wire
- Audio snapshot is a copy - upstream reload doesn't affect downstream
**Rationale:** Eliminates the hidden `_lookup_audio_from_events()` pattern. Blocks receive all needed data through explicit connections.

### D8: Waveforms are a display-layer concern
**Date:** 2026-03-01
**Status:** Decided
**Decision:** Waveforms not stored in DataItems. Computed on-demand by the display layer from audio data the block has access to through its connections.
**Rationale:** Clean separation between processing data and display artifacts.

### D9: One setlist per project (confirmed intentional)
**Date:** 2026-03-01
**Status:** Decided
**Decision:** One setlist per project is a hard constraint. However, individual song VERSIONING is required - songs need multiple versions (different arrangements), with version selection and history preservation.
**Rationale:** Rex challenged this as arbitrary. Griff confirmed it's intentional. The versioning need is new and wasn't in the original spec.

### D10: Persistence - SQLite as runtime engine, .ez file is a zip archive
**Date:** 2026-03-01 (revised 2026-03-01)
**Status:** Decided (revised)
**Decision:** SQLite is the runtime source of truth with proper schema, foreign keys, and referential integrity. The `.ez` file is a **zip archive** containing the SQLite database file plus audio assets. **Save** = close connection, zip database + audio into `.ez`. **Load** = extract `.ez`, open the SQLite database. SQLite does NOT persist between sessions as a standalone file - it exists only while a project is open. The `.ez` file is what users save, share, and back up. Migrations (versioned SQL scripts) run on the extracted database when opening a project saved with an older app version.
**Rationale:** Original D10 made SQLite persist as a live database, which conflicts with desktop app UX where projects are files. The zip approach gives SQLite's runtime power (queries, foreign keys, transactions) while keeping projects portable and user-understandable. No JSON serialization layer - the SQLite file IS the data, just bundled.

### D11: Typed port declarations with connection-time validation
**Date:** 2026-03-01
**Status:** Decided
**Decision:**
- Ports declared as typed class attributes on block processor
- Engine validates type compatibility at connection creation time, not execution time
- `GENERIC` eliminated or replaced with explicit `ANY` opt-in
- Cardinality (SINGLE/MULTI) enforced at connection time
- No runtime filter machinery - graph is type-valid by construction
**Rationale:** Filtering is a port responsibility. Validate early (connection time), not late (execution time).

### D12: `item_key` / `semantic_key` for stable DataItem identity
**Date:** 2026-03-01
**Status:** Decided
**Decision:** DataItems carry a stable semantic key that survives re-execution. Engine upserts by `(block_id, item_key)` rather than creating new UUIDs each run.
**Rationale:** Eliminates orphaned DataItems, enables selective operations, provides stable identity for downstream references.

### D13: Workspace block pull options
**Date:** 2026-03-01
**Status:** Superseded by D93
**Decision:**
- Pull is ALWAYS manual, never automatic
- Two pull actions: **Pull (replace)** - nuke upstream-derived layers, pull fresh, user layers untouched. **Pull (keep existing)** - fetch new data alongside old, no deletions.
- **Dismiss** - clear "new data available" indicator
- **Merge** - desired full functionality, deferred until stable event identity is designed (see D19)
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
1. TRIGGER - User clicks Execute, engine acquires lock
2. PULL - Follow connections → read upstream output → COPY data → bundle into inputs dict
3. EXECUTE - processor.process(inputs, settings) in background worker. Pure function.
4. STORE - Output DataItems stored in block's local data. Manifest updated. Downstream marked STALE.
5. NOTIFY - Completion signal to main thread via Qt signal. Lock released.
```
**Rationale:** Confirmed with Griff. Copy-on-pull, isolated execution, atomic commit, each block is an island.

---

### D17: Group as first-class entity
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** LayerGroup is a real entity with id, name, collapsed state, and ordered list of layers. Layer properties (group_id, group_name, group_index) removed from Layer. SyncState extracted into its own entity. LayerOrder table eliminated - order is structural.

### D18: Consolidated into D14

### D19: Merge pull option - deferred until event identity is designed
**Date:** 2026-03-01
**Status:** Superseded by D93 (Take System replaces merge concept)
**Decision:** Merge requires stable event identity across re-executions, which doesn't exist yet. Desired functionality: smart merge with diff/conflict UI, selective accept/reject, preservation of manual edits. Requires stable event identity model (content-hash or deterministic IDs). Full feature - implementation phasing TBD later.

### D20: FULL strategy stops at Workspace boundaries
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** Workspace blocks are terminal inputs in execution graph. Engine reads current output without calling process(). Formal `BlockCategory` enum (PROCESSOR, WORKSPACE) checked in dispatcher.

### D21: Copy semantics - true full copy
**Date:** 2026-03-01
**Status:** Decided
**Decision:** "Copy" means full copy - metadata AND audio files. Each block writes copies to its own output directory. Block B's data is completely independent of Block A. No shared file references, no reference counting. True isolation. Disk space tradeoff accepted in favor of reliability and predictability.

### D22: Execution lock scope - per-block
**Date:** 2026-03-01
**Status:** Decided
**Decision:** Per-block execution lock prevents re-running same block. NOT a global project lock. UI reads never blocked. STORE phase is a single DB transaction.
**Rationale:** Griff's core requirement is non-blocking UI - user must be able to execute other blocks, change settings, and use the app while a block runs. A global lock would freeze the app during long ML operations. Per-block lock allows concurrent independent block execution while preventing conflicting runs on the same block. Supersedes Dr. Voss's earlier recommendation for a global lock.

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

### D25: ActionSet - single canonical representation
**Date:** 2026-03-01
**Status:** Decided (panel recommendation)
**Decision:** ActionSets must have one canonical form - either normalized DB rows OR embedded JSON, not both. Eliminate the dual representation.

### D26: BlockDataManifest as single source of truth
**Date:** 2026-03-01
**Status:** Decided (panel recommendation)
**Decision:** Collapse `block_local_state` and data manifest into a single entity (BlockDataManifest). One place to answer "what data does this block have, where did it come from."

### D27: Partial failure in FULL execution
**Date:** 2026-03-01
**Status:** Decided (panel consensus)
**Decision:** Fail fast - stop at first failure, don't commit the failed block. Already-committed ancestors keep their outputs. Downstream marked STALE. Atomic commit per block: stage outputs → verify all succeed → atomic swap to canonical location. Never partial-write.

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

### ~~Command Bus / Event Bus / Facade Architecture~~ → RESOLVED (D30-D34)
Pipeline + adapters architecture decided. CommandSequencer and AudioPlayer conflicts with background worker remain open (see Block-Specific Designs). Undo for Workspace blocks scoped (global stack, editable commands only) - detailed Editor undo interactions still need design.

### ShowManager Sync Layers in Editor
How synced layers (MA3 ↔ Editor) work in the new Workspace block model. SyncBinding is now a separate entity - need to design how live MA3 sync interacts with the pull model, layer identity, and bidirectional communication channel. Griff flagged for dedicated discussion.

### Stale Propagation Mechanism
How downstream blocks learn they're stale after upstream execution. Needs design.

### Block-Specific Designs
- **Editor:** Merge behavior, layer/group structure (panel reviewing)
- **CommandSequencer:** Can't run in background worker (dispatches commands)
- **AudioPlayer:** Needs Qt multimedia (background worker conflict)
- **ShowManager:** Network listener lifecycle doesn't fit command pattern

### MANIPULATOR Runtime Behavior
D3 extracts MANIPULATOR as its own abstraction (channel, not port). The runtime communication protocol - actual bidirectional data flow semantics - still needs design. Ties into ShowManager sync layers discussion.

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
A legacy observability proposal existed before the EZ2 cleanup, but the dedicated doc was removed with the EZ1 historical-doc purge. Any future observability work should derive from the current EZ2 architecture and implementation plans instead.

### Undo for Workspace Blocks
Workspace blocks (Editor) need real undo for user edits. Undo stack scope confirmed global (D33), but Editor-specific undo interactions (layer edits, event classification changes, group operations) need detailed design.

---

## Command Bus / Facade Architecture (2026-03-01)

### D30: Single pipeline, multiple adapters - no facade
**Date:** 2026-03-01
**Status:** Decided
**Decision:** Replace the monolithic `ApplicationFacade` (104 methods, 5,307 lines) with a single `Pipeline` that routes commands/queries to handlers through a middleware chain. Each external interface (Qt GUI, MCP server, CLI) is a thin adapter that translates its native format (button clicks, tool calls, text input) into commands and dispatches them to the same pipeline. Adding a new interface = writing one adapter file, zero changes to pipeline or handlers.
**Rationale:** The facade pattern breaks down at scale - it became a god object. The pipeline pattern gives one standardized path for all operations, enforces consistent validation/undo/event-emission, and makes new interfaces trivial to add. This is Griff's "one pipeline, multiple interfaces" principle.

### D31: Commands are pure data with self-described validation and undo
**Date:** 2026-03-01
**Status:** Decided
**Decision:** Commands are frozen dataclasses (pure data, no framework deps). Each command declares:
1. Its fields (the data it needs)
2. A `validate(ctx)` method (constraints on that data)
3. Whether it's `undoable` (bool)
4. A `reverse()` factory (returns the command that undoes this one)

Handlers are separate classes with injected service dependencies - they do the actual work. Commands never reference the facade or UI. Pipeline middleware orchestrates validation → undo check → handler dispatch → event emission.
**Rationale:** Eliminates circular dependency (old commands took `facade` as first arg). Commands are testable in isolation. Handlers are testable with mocked services. Self-describing commands mean the pipeline doesn't need to know about specific operations.

### D32: Two command categories - Editable (undoable) and Operational (not undoable)
**Date:** 2026-03-01
**Status:** Decided
**Decision:**
- **Editable commands** (`undoable = True`): Structural edits - add/delete/move blocks, connect/disconnect, rename, edit settings, classify events, reorder layers. These go on the undo stack.
- **Operational commands** (`undoable = False`): Execute block, pull upstream, load/save project, MA3 sync. These do NOT go on the undo stack. Adapters must warn the user before dispatching ("This action cannot be undone").
**Rationale:** Industry standard (DAW pattern). Execution involves heavy computation, file I/O, and external side effects that can't be meaningfully reversed. Rather than fake undo, be honest about it. Future optimization: store one generation of previous block output for "revert to previous" (cheaper than full undo, more predictable).

### D33: Global undo stack, editable commands only
**Date:** 2026-03-01
**Status:** Decided
**Decision:** One undo stack per project. Only editable commands (D32) are pushed. The undo stack is a pipeline middleware - pure Python, not coupled to `QUndoStack`. A Qt adapter can bridge to `QUndoStack` for menu integration if needed, but the core undo system has no framework dependency.
**Rationale:** Global undo is what every DAW does (Ableton, Logic, Reaper). Per-block undo would confuse users. Decoupling from Qt means the undo system works in CLI/MCP/test contexts.

### D34: CQRS-lite - Commands and Queries through same pipeline, different middleware
**Date:** 2026-03-01
**Status:** Decided
**Decision:** Reads (queries) and writes (commands) both go through the `Pipeline`, but queries skip undo and validation middleware. Queries are also pure data with separate handlers. This gives a single dispatch mechanism while allowing different processing paths.
```
Command path: Validate → Undo check → Handler → Emit events → Result
Query path:   Handler → Result
```
**Rationale:** One pipeline, one dispatch mechanism, one place to add cross-cutting concerns (logging, auth, rate limiting). But reads and writes have fundamentally different needs - reads don't mutate state, don't need undo, don't emit domain events.

---

## Event Bus + Persistence Architecture (2026-03-01)

### D35: Typed event classes with minimal DomainEvent base
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** Frozen dataclasses. Base carries `timestamp: float`, `correlation_id: str`. Each domain event (`BlockAdded`, `ConnectionCreated`, etc.) inherits from this. No universal ObservableEvent blob. No `source` field - event type provides provenance. Observability metadata (layer, component) derived from event type, not stored.
**Rationale:** Type safety for handlers, universal metadata for observability/MCP. All 5 panelists agreed.

### D36: Collect-then-publish - events staged during handler, flushed after commit
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** The UoW collects events via `uow.collect(event)`. After successful commit, events are published to the bus. On rollback, events are discarded. No event escapes the transaction boundary.
**Rationale:** Strongest consensus across the panel. Pre-commit events cause phantom UI updates for rolled-back operations - "events that lie."

### D37: Re-entrant publish uses breadth-first queue
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** Events published by handlers during dispatch are queued and processed after the current batch completes. Never recursive. Predictable ordering.
**Rationale:** Prevents stack overflow, infinite loops, and makes event ordering deterministic and debuggable.

### D38: Qt adapter is one file in the UI layer - core never imports Qt
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** A subscriber (~30 lines) that marshals events to the Qt main thread via `QCoreApplication.postEvent`. The event bus contract is `Callable[[DomainEvent], None]`. No Qt types in the core. If running CLI/MCP, the adapter simply isn't loaded.
**Rationale:** Pure Python core enables CLI, MCP, and testing without Qt. Legacy bus required a running Qt app - this was a known pain point.

### D39: Unit of Work middleware wraps every command
**Date:** 2026-03-01
**Status:** Decided (4-1, Rex dissented)
**Decision:** ~40 lines of middleware. Handlers receive a UoW context (connection + event collector). They never call commit/rollback. The middleware begins the transaction, calls the handler, commits on success (flushing events), rolls back on failure (discarding events).
**Rationale:** Enforces the post-commit event rule (D36) architecturally rather than by developer discipline. Rex argued handler-managed transactions are simpler for a single-user app, but the UoW is equally simple and eliminates an entire class of consistency bugs by construction.

### D40: Repositories are stateless - receive connection from UoW
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
**Rationale:** Background STORE operations lock the write connection. UI reads shouldn't stutter waiting for that lock. It's ~3 lines of extra setup and prevents a whole category of UI lag. Original "wait and see" approach was premature optimization in reverse - avoiding trivial work that has clear benefit.

### D43: Schema migrations via versioned SQL scripts (on project load)
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** `migrations/` directory with numbered `.sql` files. `schema_version` table inside the SQLite database tracks current version. Migrations run **when opening a project** saved with an older app version - extract the `.ez`, check schema version, run unapplied scripts, then open. Each migration runs in its own transaction. No ORM. During active development, simply delete and recreate.
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

### D45: Module-level wiring - no DI framework
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
**Decision:** Handlers return `Result[T]` - a frozen dataclass with `value` or `error`. UoW middleware commits on `result.ok`, rolls back on failure. Unexpected exceptions caught by outer try/except, wrapped into `Result.fail(InfrastructureError(...))`. Three error categories: `ValidationError`, `DomainError` (includes cancellation), `InfrastructureError`.
**Rationale:** Result makes commit/rollback decisions explicit. Exceptions reserved for genuinely unexpected failures.

### D51: Qt-dependent components are UI adapters, not pipeline handlers
**Date:** 2026-03-01
**Status:** Decided
**Decision:** CommandSequencer, AudioPlayer, and other Qt-dependent components live in the UI layer. They call the pipeline - the pipeline does not call them. Not registered as command handlers.
**Rationale:** Maintains D38 (core never imports Qt). These are UI concerns that trigger domain operations.

### D52: Standardized progress events for execution
**Date:** 2026-03-01
**Status:** Decided → Resolved by D53-D56
**Decision:** Griff requirement: "extremely solid progress events standardized through the application." Resolved in dedicated panel session - see D53-D56 below.

---

## Progress Events, Stale Propagation, Execution Decomposition (2026-03-01)

### D53: Progress is a side-channel, not a domain event
**Date:** 2026-03-01
**Status:** Decided (panel unanimous)
**Decision:** Progress uses a dedicated `ProgressChannel` - thread-safe, in-memory pub/sub completely separate from the domain event bus (D35-D38). The domain event bus receives only `ExecutionStarted` and `ExecutionCompleted`/`ExecutionFailed`. All mid-execution progress goes through `ProgressChannel`. Not persisted. Not serialized.
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
    fraction: float         # 0.0-1.0
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
Combines progress reporting and cancellation into one object. Reporting is optional - processors that don't call it show indeterminate in UI. Throttle at the consumer (UI at 10-30Hz), not the producer.
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
- **GraphPlanner** - pure function, topological sort, execution order, DAG validation
- **ExecutionCoordinator** - FULL orchestration, sequencing blocks, aggregate progress, fail-fast
- **BlockExecutor** - single block PULL→EXECUTE→STORE, per-block lock, progress reporting
- **StalenessService** - propagation logic, pipeline handler triggered by events after STORE commits

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

### Weighted composite progress - historical data
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
A legacy observability proposal existed before the EZ2 cleanup. Now that event bus (D35-D38) and progress channel (D53) are decided, finalize observability directly against the surviving EZ2 architecture docs.

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

### D66: Connection is a Value Object - no ID
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
**Decision:** PortType has `allows_fan_in` property. Audio=False, OSC=False, Event=True. Attempting fan-in on a 1:1 port raises a domain error - no silent replacement. UI layer may issue delete+connect sequence if user confirms.

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

### D77: Sync pipeline integration - 5 commands, 2 queries
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
**Decision:** Both sides can edit. Resolution policy per layer determines divergence handling. No complex merge algorithms - either one side wins, union, or ask user.

### D80: Sync status reduced to 4 states
**Date:** 2026-03-02
**Status:** Decided
**Decision:** Unmapped, Synced, Diverged, Error. Pending → transitional (not persisted). Disconnected → Error with reason. AwaitingConnection → Unmapped.

### D81: Fingerprint-based event matching retained
**Date:** 2026-03-02
**Status:** Decided
**Decision:** Time + duration at millisecond precision (3 decimal places). 50ms tolerance for MA3 frame-level quantization drift. No shared ID space between systems makes temporal fingerprinting the correct approach.

### D82: OSC Gateway is a project-level service, not a Block
**Date:** 2026-03-02
**Status:** Decided (restored 2026-03-16 — D120 reverted)
**Decision:** OSC Gateway is a first-class project entity owned by the Project aggregate. No ports, no graph participation. Infrastructure, not a processing node. MIDI interface analogy.

### D83: Singleton per project, enforced by Project aggregate
**Date:** 2026-03-02
**Status:** Decided (unanimous)
**Decision:** One OscGateway instance per loaded project. Singleton enforced by Project aggregate, not Python module-level state. Properly injected via DI.

### D84: IOscGateway protocol interface
**Date:** 2026-03-02
**Status:** Decided (restored 2026-03-16 — D120 reverted)
**Decision:** Blocks interact via `IOscGateway` protocol. Send = direct method call. Receive = domain events via event bus. No bidirectional port abstraction - sending and receiving are architecturally different (UDP is fire-and-forget).

### D85: Visual status node in graph UI (view-only)
**Date:** 2026-03-02
**Status:** Decided (restored 2026-03-16 — D120 reverted)
**Decision:** OSC Gateway rendered in graph canvas as a non-connectable infrastructure node. Shows status, allows click-to-configure. Not a real Block - no ports, no connection validation participation. Purely a view concern. Must not subclass Block.

### D86: One gateway, multiple endpoint configs
**Date:** 2026-03-02
**Status:** Decided (restored 2026-03-16 — D120 reverted)
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
2. **Data retained:** Downstream blocks keep all stored data (inputs and outputs). Consistent with D6 - each block is an island, its data is its own copy.
3. **No new state:** No DISCONNECTED enum value. STALE is sufficient - it tells the user "your output may not reflect current inputs" which covers both "upstream changed" and "upstream removed."
4. **No data loss:** Accidental disconnection is recoverable. Reconnect + re-execute to get fresh data, or keep working with existing output.
**Rationale:** Follows directly from D6 (copy-on-pull, blocks are islands) and D58 (stale cascade on connection change). Adding a DISCONNECTED state would complicate the DataState enum without providing meaningfully different user action - in both cases the user either re-runs or accepts the existing output.

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
- `CreateProject` - initialize empty SQLite, set schema version
- `LoadProject` - extract .ez, run migrations (D43), hydrate entities
- `SaveProject` - close DB connections, copy SQLite, zip with audio, atomic rename to .ez
- `CloseProject` - cleanup, close connections

**Dirty tracking:** Boolean flipped on any successful editable command, reset on save. Prompt "unsaved changes" on close/quit.

**Audio file management:** Audio files stored in project-local directory alongside SQLite while open. LoadAudio copies source into project dir. All blocks use project-relative paths. On save, audio zipped into .ez archive.
**Rationale:** Project aggregate enforces all invariants (one setlist, one gateway, schema version). Atomic save prevents corruption. Project-relative audio paths keep .ez files portable.

### D90: Setlist entity design
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
**Setlist:** `id`, `name`, `songs: List[Song]` (ordered), `default_action_set_id`.
**Song:** `id`, `title`, `artist: str | None`, `order: int`, `active_version_id`, `versions: List[SongVersion]`, `action_set_override_id: str | None`.
**SongVersion:** `id`, `song_id`, `label: str` (required - e.g., "Album Cut", "Festival Edit"), `audio_file_path: str` (project-relative), `created_at: float`, `notes: str | None`. No version number - versions are parallel arrangements, not sequential revisions. Display order = creation order.

**Batch processing:** For each song in order: get active version audio → set as SetlistAudioInput source → get ActionSet (override or default) → execute actions → store results tagged with song_id + version_id → report per-song progress.
**Batch failure: continue on failure.** Log error, mark song as failed, proceed to next. Songs are independent - failure in song N has zero impact on song N+1. User reviews all failures after batch completes.

**Command categories:** Reorder/add/remove songs, switch active version = editable (undoable). Batch execution = operational (not undoable, per D32).
**Rationale:** Parallel arrangements (not linear revisions) eliminates version numbering complexity. Continue-on-failure respects song independence unlike block execution (D27) where downstream depends on upstream.

### D91: Block-specific designs - CommandSequencer deleted, AudioPlayer, ShowManager
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
**CommandSequencer: DELETED.** Removed from the block type inventory entirely. If sequenced actions are needed in the future, handle via scripting or macros outside the block graph.

**AudioPlayer → Playback Block.** Third block category:
- **Playback** - Receives data via connections, provides real-time output (audio). No EXECUTE phase, no STORE, no staleness, no persistence. Ephemeral state only (playing/paused/stopped/position).
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
**Rationale:** CommandSequencer added complexity for marginal value - deleted per "best part is no part." Playback is a genuinely different concern from processing or workspace management - deserves its own lightweight category rather than being forced into either.

### D92: MANIPULATOR removed - pipeline commands replace bidirectional channel
**Date:** 2026-03-09
**Status:** Decided
**Decision:** MANIPULATOR concept (D3) is removed from the architecture. No bidirectional channel abstraction exists.

ShowManager ↔ Editor communication uses pipeline commands and domain events:
- **Inbound (MA3 → Editor):** OscGateway receives → domain event → ShowManager handler → pipeline command writes to Editor layers.
- **Outbound (Editor → MA3):** Editor publishes "layer changed" domain events (unaware of ShowManager). ShowManager subscribes, decides whether to sync, dispatches push command through pipeline → OscGateway → MA3.

**Editor remains dumb.** Editor has zero knowledge of ShowManager, MA3, OSC, or sync. It's a pure Workspace block that manages layers/events and publishes change events. ShowManager is the only entity aware of both sides.

Control ports (D64) retained in spec for potential future use but have zero current implementors. D3 superseded - the "channel" is the pipeline itself.
**Rationale:** No actual bidirectional data flow exists. All communication is pipeline commands reading from one block and writing through another. MANIPULATOR added a concept with zero concrete implementation. Removing it simplifies the architecture while keeping Editor fully decoupled.

### D93: Take System replaces merge pull - layer dirty tracking via take edits
**Date:** 2026-03-09
**Status:** Decided
**Decision:** Merge pull (D19) and layer dirty tracking are replaced by the **Take System**. No event matching needed.

**Model:** Each Editor layer has a **Main layer** (user-curated, syncs to MA3) and zero or more **Takes** (editable snapshots from upstream pulls). Layers are **paired** to an upstream block output - pairing tracks which block + output maps to which Editor layer.

**Take properties:** `id`, `label` (default: "Take N - [timestamp]"), `source_block_id`, `source_settings_snapshot` (frozen copy of upstream settings at pull time), `created_at`, `events` (editable), `visible`, `order`.

**Pull behavior:** Pull ALWAYS creates a new take. Replace and Keep Existing (D13) are retired. Pull = new take, that's it. User decides what to promote.

**Takes are editable.** Users can move, resize, delete, and classify events within a take before promoting. This enables in-take curation (mark false positives, refine positions) before adding to Main.

**Take → Main actions:**
- **Add Selection to Main** - Copy selected events to Main (additive). No classification filtering - selection is literal.
- **Add All to Main** - Copy all events to Main (additive).
- **Overwrite Main** - Replace Main layer contents with this take's events.
- **Mark as False/True** - Set classification state on events in-take (D14).
- Standard event editing (move, resize, delete) within takes.

**Scope:** Per layer, not per group or per connection. Each paired layer has its own independent take stack.
**Take limit:** Unlimited. No auto-deletion.
**All take/promote operations are editable commands (undoable).**

**Layer pairing:** Tracks `(block_id, output_name)` → Editor layer. Pairing enables "new data available" indicators on the correct layer when upstream re-executes. Unpairing keeps existing takes but stops future pull indicators.

**Rationale:** Event matching across re-executions is fundamentally unreliable - different settings produce genuinely different events, not "updated" versions. The Take System (inspired by DAW take lanes / comping) eliminates matching entirely. Users visually compare results and cherry-pick, which is how they actually work. D19 (merge deferred) is now superseded.

### D94: No untyped metadata escape hatch
**Date:** 2026-03-09
**Status:** Decided
**Decision:** Every block type defines fully typed settings. No `Dict[str, Any]` escape hatch. If a new block needs new settings, define them explicitly. If a genuine need arises post-MVP, add it then.
**Rationale:** D1 killed untyped metadata for good reason. An escape hatch re-introduces the problem.

### D95: Progress reporting during PULL phase - yes
**Date:** 2026-03-09
**Status:** Decided
**Decision:** BlockExecutor reports progress during PULL phase using `phase: "pull"` (already supported by D54 ProgressReport schema). For small files it's instant and irrelevant. For large files (multi-track stems), the user gets feedback. Throttle at consumer per D55.
**Rationale:** Basically free - the schema already supports it. Prevents frozen progress bars during large audio copies.

### D97: False event display - visible, visually distinct, toggleable
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
False events (FALSE/USER_FALSE) are displayed in both Main layers and Take layers with distinct visual treatment:

**Visual hierarchy (strongest to weakest):**
1. USER_TRUE - full opacity, full color (user explicitly confirmed)
2. TRUE - full opacity, full color (algorithmically classified)
3. UNCLASSIFIED - full opacity, neutral/gray color
4. FALSE - semi-transparent, desaturated
5. USER_FALSE - semi-transparent, desaturated, subtle X or strikethrough (strongest "no" signal)

**Toggle:** Per-layer "Show False Events" toggle, persisted. Default ON in Takes (reviewing), OFF in Main (working). Keyboard shortcut for quick toggle.

**Interaction:** False events are non-interactive by default (can't accidentally select while editing). When visible and explicitly clicked, they can be selected and reclassified (Mark as True). "Add Selection to Main" is literal - no classification filtering, false events included if selected.

**Rationale:** False events are data, not garbage. Users need to see what was rejected, verify rejections, and reverse mistakes. Visual weight (bright = good, dim = rejected, gray = unreviewed) provides instant at-a-glance understanding.

### D96: Audio file path remapping and auto-save deferred to post-MVP
**Date:** 2026-03-09
**Status:** Partially superseded — Auto-save superseded by D157 (full design decided). Path remapping superseded by D162 (full design decided).
**Decision:** Both features deferred. Neither has architectural impact.
- **Path remapping:** "File not found" dialog with search/remap. Standard DAW pattern. → Fully designed in D162.
- **Auto-save:** Timer-based periodic save to recovery location. Calls SaveProject to temp path. → Fully designed in D157.
**Rationale:** No architectural decisions needed. Implementation details for post-MVP polish.

### D98: SetlistAudioInput block DELETED
**Date:** 2026-03-09
**Status:** Decided
**Decision:** SetlistAudioInput removed from block inventory. Batch processing sets `audio_path` directly on the existing LoadAudio block via operational command (not undoable, no stale cascade during batch). 11 block types remain.
**Rationale:** LoadAudio already loads audio files. A separate block for setlist input is redundant.

### D99: No EZ1 project migration
**Date:** 2026-03-09
**Status:** Decided
**Decision:** EZ2 does not load EZ1 `.ez` projects. No block type alias map needed. Clean break - fundamentally different data model (Take System, new schemas, deleted blocks).
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
**Decision:** Quick actions are UI-only (interactive operations). Setlist ActionSets reference block execution commands directly. Separate systems - no filtering or headless mode needed.
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

**Sync layer creation:** When user maps a sync (MA3 TC → Editor), ShowManager creates a layer in the Editor via pipeline command. From Editor's perspective, it's just a normal layer - Editor has zero knowledge of sync.

**Editor stays dumb.** All sync intelligence lives in ShowManager:
- ShowManager maintains sync registry: `{sync_mapping_id: (editor_layer_id, ma3_track_coord, resolution_policy, sync_status)}`
- Editor publishes "layer changed" domain events as always (unaware of ShowManager)
- ShowManager subscribes, detects changes in mapped layers, syncs to MA3
- ShowManager receives MA3 updates, applies resolution policy, pushes events into Editor via pipeline commands

**Constraints:**
1. Both user and ShowManager can edit synced layers (normal Editor behavior)
2. Conflict resolution is ShowManager's job entirely (D76 resolution policies)
3. User CAN delete a synced layer - ShowManager detects deletion event, updates registry (status → Error/Unmapped)
4. User CAN create events in synced layers - normal editing, ShowManager picks up changes
5. ShowManager names layer at creation (e.g., "TC 1"), tracks by layer_id not name
6. Synced layers do NOT get **analysis** Takes — data comes from MA3, not upstream pipeline analysis. Note: D121's "sync Take" is a distinct conflict-resolution mechanism (Ask User policy), not a pipeline pull Take.
7. **Sync status icon on layer** - UI adapter queries ShowManager for sync status per layer_id and renders icon (Synced/Diverged/Error). Purely a UI concern - Editor domain has zero awareness.

**Rationale:** Keeps Editor as a pure Workspace block. All sync complexity is encapsulated in ShowManager. Editor just manages layers and events, same as always.

### D105: Editor data model - full schema
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
Four tables for Editor data:

**layers:** `id`, `editor_block_id` (FK), `name` (unique per editor), `layer_type` (MAIN|TAKE), `group_id` (FK, nullable), `order`, `height` (px, default 60), `color` (hex), `visible`, `locked`, `show_false_events` (default true for TAKE, false for MAIN), `source_type` (UPSTREAM|USER), `paired_block_id` (nullable), `paired_output_name` (nullable), `created_at`.

**takes:** `id`, `layer_id` (FK → MAIN layer this take belongs to), `label`, `source_block_id`, `source_settings_snapshot` (JSON), `order`, `visible`, `created_at`.

**events:** `id`, `layer_id` (FK, nullable), `take_id` (FK, nullable), `time` (seconds), `duration` (seconds, 0=marker), `classification_state` (UNCLASSIFIED|TRUE|FALSE|USER_TRUE|USER_FALSE), `confidence` (nullable float), `label` (nullable), `metadata` (JSON, nullable - analysis-specific payload like pitch, velocity, BPM), `created_at`. Constraint: event has EITHER layer_id OR take_id, never both, never neither.

**layer_groups:** `id`, `editor_block_id` (FK), `name`, `collapsed`, `order`. (Per D17.)

Event `metadata` JSON is analysis output data (varies by producing block type), NOT a settings escape hatch. Schema defined by block type, not user-configurable.
**Rationale:** Clean relational model with clear ownership. Events belong to exactly one container (Main layer or Take). Takes belong to exactly one Main layer. Layer groups organize layers visually.

### D106: Undo granularity for Editor
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
**Multi-select operations are atomic.** Any operation on a selection of N events produces ONE undo entry that reverses the entire batch. Move 20 events = one undo. Delete 50 events = one undo that restores all 50. Classify 100 events = one undo.

**Single undo entry per operation:**
- Create/delete/move/resize event(s) - batch if multi-selected
- Create/delete/rename layer or take
- Change event classification (single or batch)
- Layer reorder, property changes (color, height, visibility, locked)
- Promote events (Add Selection/All to Main, Overwrite Main)
- Overwrite Main = atomic macro (clear + add = one undo step)

**Drag operations commit on mouse-up.** No undo entries during drag. One entry for final delta on release. Prevents 60 undo entries per second of dragging.

**NOT on undo stack:** Block execution, batch processing (operational commands per D32).

**Global stack (D33).** Editor operations share stack with node graph operations. Standard DAW behavior.
**Rationale:** Users expect multi-select operations to undo as one action, not N separate undos. Mouse-up commit is industry standard for drag operations.

### D107: Block deletion cascade - full specification
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
- Layers ShowManager created in Editor → REMAIN. Editor owns those layers. Events are local copies in Editor. Layers become normal unsynced layers - UI adapter finds no ShowManager to query, no sync icon renders. User can continue editing or delete layers manually.

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

### D109: Editor UI - complete interaction catalog
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
- **Vertical drag snaps to layers** - events magnetically lock to the nearest layer row. Moving to a different layer row changes the event's layer assignment (data-level change, not just visual).
- Drag threshold: 3px before move initiates (prevents accidental moves on click) — **superseded by D179** (platform standard ~4px on Windows, system default on Mac)
- Escape during drag → cancel, snap back to original position
- All moves on multi-selected events are atomic (D106)

**Duplicating:**
- **Option/Alt+drag** → duplicate event(s) and drag the copy. Original stays in place.
- Ctrl/Cmd+D → duplicate selected events (offset slightly in time)

**Copy/Paste:**
- Ctrl/Cmd+C → copy selected events to clipboard
- Ctrl/Cmd+V → paste at playhead position, into active layer
- **Copy/paste works across layers** - copy from one layer, select a different layer, paste there
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
- **TimeGrid service** - single consolidated source of truth for all time calculations
- Grid modes: auto, frame-based (1f, 2f, 5f, 10f per project FPS D100), second-based (0.1s, 0.5s, 1s)
- Events snap to grid lines when dragging (magnetic behavior within threshold)
- **Snap to other events** - edge-to-edge magnetic snapping (event start → event end of adjacent)
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

**⚠️ NOTE - OPEN DISCUSSION NEEDED:** Timing and sync across clock, playhead, and audio playback. How does the playhead clock stay in sync with audio output? What's the master clock? Does audio playback drive the playhead, or does a shared clock drive both? Latency compensation? This needs its own dedicated design session.

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
- Interpolation between keyframes is NOT applicable (EchoZero events are discrete, not animated properties) - but the selection, move, copy/paste, snap, and multi-select patterns match standard keyframe editor UX
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

### D110: Audio layers in Editor - read-only playback views from graph
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
Editor layers can now contain **events OR audio**. Two layer categories:
- **Event layers** - editable events/cues (existing behavior). Support Take System, classification, all editing operations.
- **Audio layers** - read-only waveform visualization of audio flowing through the graph via audio input ports. NOT editable (no trim, split, move, crossfade, time-stretch).

**Audio layer properties:** `source_block_id`, `source_output_name`, `muted` (default false), `volume` (0.0-1.0, default 1.0), `visible`, `color`, `order`, `name` (default: source block + output, e.g., "Separator: vocals").

**Playback behavior:**
- Per-layer mute/unmute and volume control
- Solo = derived from mute (solo = mute everything except this)
- All unmuted audio layers sum to stereo master output
- No buses, effects, or routing at MVP. Architecture allows adding later.
- Audio layers do NOT get Takes - they reflect current graph audio.

**Rationale:** Users need to hear individual stems while editing cues. Separator → drums/bass/vocals/other should be visible and independently mutable. Audio stays read-only because EchoZero is a cue editor, not a DAW arranger.

### D111: Audio engine - PortAudio via sounddevice, audio callback as master clock
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
**Audio engine:** `sounddevice` (Python wrapper for PortAudio). Cross-platform, talks to ASIO (Windows), CoreAudio (macOS), ALSA/PulseAudio (Linux). Battle-tested (Audacity uses PortAudio).

**Master clock:** Audio callback drives everything (Approach 1 - same as Reaper, Ableton, every serious DAW).
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

**Rationale:** Multi-track summing with mute/solo requires a real audio engine - QMediaPlayer can't mix N streams. PortAudio is the industry standard for cross-platform audio I/O. Audio callback as master clock guarantees sample-accurate sync between audio output and playhead. EchoZero's use case (summing a few stems with gain) is trivially within PortAudio's performance envelope.

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

**Audio source:** D7 bundles source audio as a snapshot copy with events through the connection. The audio is already available - clip is just a slice at the event's time/duration.

**Audio engine integration (D111):** Callback sums both continuous audio layers AND triggered clip event audio from event layers, applying per-layer volume and mute/solo.

**Rationale:** Seeing AND hearing analysis results makes review dramatically faster. A user can instantly verify onset detection by seeing waveform transients inside events and hearing them during playback. Opt-in prevents unexpected audio from non-analysis layers.

### D113: Snap system - TimeGrid service as single source of truth
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
**TimeGrid service** - one object owns all time quantization logic. Grid rendering, snap behavior, and timecode display ALL derive from this single service. No separate grid logic.

**Snap modes (user-selectable):**
- **Off** - no snapping, free positioning
- **Auto** - grid density auto-adjusts based on zoom level (fewer lines when zoomed out, more when zoomed in)
- **Frame-based** - 1f, 2f, 5f, 10f (based on project FPS, D100)
- **Time-based** - 0.1s, 0.25s, 0.5s, 1s
- **Beat-based** - 1/4, 1/8, 1/16 (requires BPM, optional, derived from analysis)

**Snap behavior:**
- Magnetic threshold: event snaps to nearest grid line when within N pixels (configurable, default 8px)
- **Snap to grid** - primary snap target
- **Snap to events** - secondary, edge-to-edge magnetic snapping (event start → nearby event end, and vice versa). Can be toggled independently.
- **Snap to playhead** - tertiary, events snap to playhead position
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
- Display: adapts per user preference - timecode (HH:MM:SS:FF at project FPS), seconds (SS.mmm), or bars:beats (if BPM set)
- All conversions go through TimeGrid service

**Rationale:** One service, one truth. Grid rendering and snap behavior can never disagree. Magnetic snapping with modifier override is industry standard (After Effects, Premiere, Logic, Reaper).

### D114: Event Inspector panel
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
Dockable panel that shows details of the selected event(s).

**Single event selected:**
- Time (editable - type exact timecode to reposition)
- Duration (editable - type exact duration)
- Layer (display, with dropdown to move to different layer)
- Classification state (dropdown: UNCLASSIFIED/TRUE/FALSE/USER_TRUE/USER_FALSE)
- Confidence (display, from analysis)
- Label (editable text field)
- Metadata (read-only display of analysis payload - pitch, velocity, BPM, etc.)
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
- Computation is async - UI shows placeholder (flat line or loading indicator) until ready

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

### D116: Keyboard shortcuts - default mapping
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
All shortcuts are user-configurable via Settings panel. Defaults:

**Transport:**
- Space - Play/Pause toggle
- Home - Go to start
- End - Go to last event
- L - Toggle loop

**Selection:**
- Ctrl/Cmd+A - Select all in active layer
- Ctrl/Cmd+Shift+A - Select all in all visible layers
- Escape - Deselect all

**Editing:**
- Delete/Backspace - Delete selected
- Ctrl/Cmd+D - Duplicate selected
- Ctrl/Cmd+C - Copy
- Ctrl/Cmd+V - Paste
- Ctrl/Cmd+X - Cut
- Ctrl/Cmd+Z - Undo
- Ctrl/Cmd+Shift+Z - Redo

**Classification:**
- T - Mark selected TRUE
- F - Mark selected FALSE
- U - Mark selected UNCLASSIFIED

**View:**
- Ctrl/Cmd+0 - Zoom to fit
- Ctrl/Cmd+Shift+0 - Zoom to selection
- +/= - Zoom in
- - - Zoom out
- G - Toggle grid visibility
- S - Toggle snap on/off

**Layers:**
- Ctrl/Cmd+Shift+N - New layer

**Snapping modifier:**
- Alt/Option (held) - Temporarily disable snap

**Customization model:** JSON config file mapping action names → key combos. Settable via UI preferences panel. Conflicts detected and warned.

**Rationale:** Industry-standard defaults (matching DAW/NLE conventions). Full customization for power users.

### D117: Context menus - full action catalog
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
- Click - select single, deselect others
- Shift+click - add/remove from selection
- Ctrl/Cmd+click - toggle in selection
- Rubber band - drag on empty space creates selection rectangle, selects all events within
- Rubber band + Shift - add to existing selection
- Ctrl/Cmd+A - select all in active layer
- Ctrl/Cmd+Shift+A - select all visible layers

**Multi-select operations (all atomic, one undo per D106):**
- Move - all selected events move by same delta (time and/or layer)
- Resize - NOT supported on multi-select (ambiguous which edge). Resize is single-event only.
- Delete - delete all selected
- Duplicate - duplicate all selected, maintaining relative positions
- Copy/Paste - copies all selected with relative time offsets preserved
- Classify - apply classification to all selected
- Move to Layer - move all selected to target layer

**Cross-layer selection:** Selection can span multiple layers. Visual indicator (highlight) on each layer that has selected events.

**Selection persistence:** Selection survives zoom, scroll, and layer reorder. Selection clears on: click empty space, Escape, or undo/redo that affects selected events.

**Rationale:** Cross-layer multi-select is essential for batch operations. Resize excluded from multi-select to avoid UX confusion (which edge? proportional?). All batch operations are one undo step.

### D119: ShowManager UI - sync configuration panel
**Date:** 2026-03-09
**Status:** Decided
**Decision:**
ShowManager panel (opened from node graph) has three sections:

**1. Connection Status:** Connected/disconnected indicator, MA3 IP + port, Connect/Disconnect buttons, gear icon for settings (IP, port, auto-connect on project open, auto-reconnect interval, default resolution policy, sync debounce interval).

**2. Sync Layers List:** Each sync layer displays: MA3 track coordinate (TC N), mapped Editor layer name (or "unmapped"), sync status icon (Synced/Diverged/Error/Unmapped), resolution policy dropdown (MA3 Master, EZ Master, Union, Ask User per D76), manual Push/Pull buttons, event count. "Map Layer" opens dropdown of available Editor layers. "Add Sync Layer" creates new mapping. Delete via right-click or X.

**3. Activity Log:** Scrollable timestamped log of sync activity (pushes, pulls, conflicts, errors, connections). Filterable by type. Session-only, not persisted.

**Rationale:** All sync configuration in one panel. Activity log provides transparency into what ShowManager is doing behind the scenes.

### D120: ShowManager ↔ Editor event flow - speed-optimized
**Date:** 2026-03-09
**Status:** Superseded (reverted 2026-03-16). ShowManager does not own the socket. D82-D86 restored.
**Decision:**
Priority is speed. Minimal hops between MA3 and Editor.

**Wire protocol:** OSC over UDP (mandatory - MA3 only speaks OSC). Use minimal OSC parser (`python-osc` or custom ~100 lines). No bloated OSC libraries.

**ShowManager owns the socket directly.** D82 (OSC Gateway as separate service) is superseded. ShowManager holds the UDP socket, runs recv loop, parses OSC, applies sync. No intermediate gateway service, no unnecessary domain event hops for sync data.

**Superseded decisions:** D82 (gateway not a block), D84 (IOscGateway protocol), D85 (visual status node - ShowManager block IS the status node), D86 (one gateway multiple endpoints - ShowManager handles its own config). D83 (singleton) survives - one ShowManager per project.

**Inbound fast path (MA3 → EZ):**
1. UDP socket receives OSC packet on network thread
2. Minimal OSC parse → internal event format
3. ShowManager applies resolution policy
4. **Direct write to Editor layer event store** (bypasses pipeline command/validation/undo - these aren't user edits)
5. Batch write wrapped in transaction (data integrity)
6. ONE domain event emitted after batch: `SyncLayerUpdated(layer_id)` - UI repaints
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

**Surviving decisions:** D76 (resolution policies), D77 (sync commands for user actions), D78 (config not wiring), D79 (bidirectional), D80 (4 statuses), D81 (fingerprint matching), D83 (singleton), D87 (sync logic - now with direct network access).

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

### D122: Classify block - core design
**Date:** 2026-03-12
**Status:** Decided
**Decision:**
New Processor block type: `Classify`. Model-agnostic - the block is a generic "run this PyTorch model against events + audio" machine. The model defines what it classifies.

**Ports:** Input: `events` (Event) + `audio` (Audio). Output: `events` (Event - same events enriched with classification + confidence).
**Settings:** `model_category: str` (e.g., `"drums.full-mix"`), `confidence_threshold: float` (default 0.5), `batch_size: int` (default 64), `split_layers: bool` (default true), `inference_mode: "auto" | "local" | "cloud"` (default auto).

**Processing logic:**
1. Pull events + audio
2. For each event, extract audio window around event time (window size defined by model metadata - 100ms for drums, 500ms for tonal, 2-4s for energy/structure)
3. Batch windows into tensors
4. Run inference via InferenceBackend protocol (D124)
5. Apply confidence threshold - below threshold → UNCLASSIFIED
6. Write classification label + confidence to event metadata
7. If `split_layers=true`, auto-split output into per-label layers (one for kicks, one for snares, etc.)
8. Output enriched events

**Input flexibility (Option C):** Works on both full-mix audio and separated stems. Accuracy improves with separated stems. Recommended workflow: Separator → drums stem → DetectOnsets → Classify. Block handles either case - model quality determines accuracy.

**What the model defines:** Input window size, sample rate requirement, output label taxonomy, architecture.
**What the block defines:** Window extraction, batching, threshold application, progress reporting, event metadata schema, layer splitting.
**Adding a new model type = train + package + publish. Zero block code changes.**
**Rationale:** Model-agnostic design means the block never changes. All intelligence lives in the models. Category-based selection (D123) means users think in terms of "what do I want classified" not "which model version."

---

### D123: Model registry - category lanes + auto-update on Cloudflare
**Date:** 2026-03-12
**Status:** Decided
**Decision:**

**Three-tier model storage:**
- **Tier 1: Bundled** - 2-3 core models ship with installer. Work offline, day one.
- **Tier 2: Cloud registry** - JSON manifest on Cloudflare R2 + CDN at `models.echozero.app/registry.json`. Models download to `~/.echozero/models/`.
- **Tier 3: Custom** - User drops `.ezmodel` files into `~/.echozero/models/custom/`. Shows in model selector alongside registry models.

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

### D124: Inference abstraction - local vs cloud backend
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
- `LocalInferenceBackend` - loads PyTorch model from cache, runs on CPU/CUDA/MPS. Ships at launch.
- `CloudInferenceBackend` (future) - sends audio windows to `api.echozero.app/v1/classify`. Requires auth. Handles large models, GPU-accelerated inference, experimental models.

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

**Layer organization: Option B - per-label layer split (default ON).**
Output auto-splits into one layer per classification label. "kicks" layer, "snares" layer, etc. Each mapped separately to MA3. Setting `split_layers: bool` (default true) with opt-out to single mixed layer.
**Rationale:** Per-label layers match the LD's workflow - different lighting responses for different instruments. Mixed layer requires manual sorting. Per-label is what users want 95% of the time.

---

### D128: Telemetry basics
**Date:** 2026-03-12
**Status:** Decided
**Decision:**

**Opt-in anonymous telemetry:** When user overrides a classification, that's a signal the model was wrong. With opt-in: send `{category, version, predicted_label, corrected_label}` - no audio, no project data.

**Local quality metrics (always on, no network):**
- Per-model accuracy tracker based on user corrections
- Shown in model settings: "drums.full-mix v3.1.0 - 94% accuracy (847 reviewed events)"

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

**Data quality:** Outlier detection on contributions - if one user's corrections consistently disagree with all others, flag for review. Contribution quality score (internal).

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

### D131: No exposed node graph at launch - plugin system is the unlock
**Date:** 2026-03-13
**Status:** Decided
**Decision:** The node graph is NOT exposed to users at any tier at launch. No "read-only graph" in a middle tier. The graph only becomes user-facing if/when a plugin SDK exists that lets users write custom blocks. Until then, the graph is a developer-only internal tool.
**Rationale:** Pre-determined pipelines don't benefit from graph visualization - it's showing homework nobody asked to see. Graph exposure is only valuable when users can modify it. Griff: "The only way I see this working is a plugin system where people can write their own blocks."

### D132: Two-interface architecture - User Timeline + Developer Workbench
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- **User interface:** Clean timeline. Drag audio, right-click, analyze, see results as sub-tracks. No graph, no pipeline details.
- **Developer interface (Pipeline Workbench):** Node graph for wiring algorithms, swapping detection strategies, A/B testing pipelines, benchmarking. Griff's laboratory.
- These are separate concerns with separate UIs. The user timeline consumes pipeline *results*. The workbench configures pipeline *construction*.
**Rationale:** One interface can't serve both audiences. Users want clean results. Developers want experimental flexibility. Forcing both into one view compromises both.

### D133: Workflow / Pipeline / Contract architecture
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- **Workflow:** User-facing action ("Analyze Drums"). Defines expected input type and output schema (the contract).
- **Pipeline:** The specific chain of algorithms that fulfills a workflow. Swappable, versionable, A/B testable.
- **Contract:** The glue. A workflow's output schema (e.g., `DrumEvent[]`) that any attached pipeline must produce.
- Pipelines are assigned to workflows. Multiple pipelines can fulfill the same workflow. User never sees which pipeline ran.
- Pipeline definition is data (JSON/config), not code. Swap an algorithm = change a config entry.
- UI rendering is keyed to **output types**, not pipelines. One renderer for `DrumEvent[]` regardless of source pipeline.
**Rationale:** Decouples what users see (workflows) from how it works (pipelines). Griff can experiment freely on the pipeline side without touching the user-facing layer. Adding a new analysis type = new workflow + output schema + pipeline. Improving existing = swap pipeline nodes, workflow unchanged.

### D134: Incremental progressive processing - no loading screens
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
1. Audio dropped → waveform renders instantly
2. Onset detection fires first (~1s, lightweight, no ML) → markers appear on timeline
3. Classification model runs on onsets in background → markers upgrade in-place with labels
4. Stem separation (if enabled) → classification refines further
- User sees results "building up" progressively. Each stage upgrades the previous results in-place.
- No "Quick Scan" intermediate - go straight from onsets to full classification.
**Rationale:** Quick Scan was deemed not viable. Progressive enrichment gives immediate feedback while heavy processing continues. Users see the system working, not waiting.

### D135: Event model - clip-based with one-shot/clip display modes
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- Each detected event is a **clip** with start time (onset) and end time (tail end). Not just a point marker.
- The system attempts to isolate each drum hit from onset through the full tail decay.
- **Two display modes** (user-toggleable per sub-track):
  - **One-shot mode:** Shows only the onset marker. Clean, minimal. For trigger-based workflows (lighting cues).
  - **Clip mode:** Shows full rectangle from onset to tail end. For editing, trimming, sample extraction.
- Events rendered like MIDI notes on sub-track lanes. Position = time, width = duration (in clip mode), color = classification type.
- No height dimension needed.
**Rationale:** Griff: "I want to isolate each drum clip with the onset through the end of the tail." This dual-mode approach serves both trigger-mapping (LD workflow) and sample extraction use cases from the same data.

### D136: Editable event boundaries
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- In clip mode, users can grab-and-drag the left edge (onset) or right edge (tail end) to adjust boundaries.
- Snap-to-zero-crossing to prevent audio clicks.
- Waveform preview visible inside clip rectangle when zoomed in, so users can see decay visually while editing.
- Audio preview on hover/scrub for auditory confirmation.
**Rationale:** ML tail detection is imperfect. Users need fast, intuitive manual correction. The affordance is already built into the clip rectangle - just make the edges draggable.

### D137: Tail detection - spectral decay + per-class priors + stem upgrade
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- **Primary method:** Spectral flux decay - track when the spectral content of the hit stops changing, not just amplitude.
- **Per-class decay priors:** Model knows expected decay curves per drum type (kicks ring longer than hats). Used as Bayesian priors to improve estimates.
- **Stem separation upgrade:** When stems are available, tail detection re-runs on the isolated drum channel for dramatically improved accuracy.
- **Pipeline slot:** Tail detection is a swappable node in the pipeline (per D133). Algorithm can be changed without affecting anything else.
- Next-onset cutoff explicitly rejected as a tail detection strategy.
**Rationale:** Griff: "Clip end detection has proven difficult. Need accurate solution. Next onset cutoff is not the solution." Spectral analysis captures the actual character of the decay rather than just volume. Per-class priors prevent obviously wrong tail lengths. Stem separation removes the biggest source of error (other instruments holding up the energy envelope).

### D138: Sub-track layout - collapsed by default, expandable lanes
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- When analysis completes, sub-tracks appear below parent track in **collapsed mode** by default.
- Collapsed = all event classes on one lane, color-coded by type.
- Expandable = click to expand into individual lanes per class (kick lane, snare lane, hat lane, etc.).
- Section analysis (intro/verse/chorus/drop) renders as colored backdrop regions behind all other content.
**Rationale:** Collapsed keeps it clean and scannable. Expanded gives detail when you need it. Same progressive disclosure philosophy as the rest of the UX.

### D139: Section analysis is first-class
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Song structure detection (intro, verse, chorus, bridge, drop, outro) is a first-class workflow, not an afterthought. Sections render as colored regions on the timeline behind event data. Key differentiator - LDs program by section, not by individual hit.
**Rationale:** Griff: "If we can figure out section analysis, that would be so fire." This is how lighting designers actually think - "verse is moody blue, chorus is full blast." Section-level awareness maps directly to the LD workflow.

### D140: A/B comparison mode
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Users can run two configurations (different pipelines, models, or settings) on the same audio and compare results side by side. "Scientist mode" - experiment, compare, choose. Implementation details TBD but architecturally supported by the Workflow/Pipeline split (D133).
**Rationale:** Griff: "Any form of testing, like a scientist, is encouraged." Pipeline architecture makes this natural - run two pipeline definitions against same input, diff the output.

### D141: Conflicting analyses use Take System
**Date:** 2026-03-13
**Status:** Decided
**Note:** Terminology standardized to **Take** for user-facing analysis results. The internal Take System (D93) retains its name for Editor's internal data versioning.
**Decision:** When multiple analyses of the same type run on the same track (e.g., drum analysis with two different pipelines), results are stored as **takes**. No auto-replacement. User can compare takes and choose. This is the A/B mechanism in practice.
**Rationale:** Take model is designed for this exact scenario. Reuse it.

### D142: Time-range analysis supported
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Users can select a time range on a track and run analysis on just that region. Partial analysis results live on the same sub-layer. Details TBD on stitching partial analyses together.
**Rationale:** Not every use case needs full-track analysis. Especially for long sets or when you only care about a specific section.

### D143: Concurrent multi-track processing pipeline
**Date:** 2026-03-13
**Status:** Decided
**Decision:** The processing pipeline must handle multiple tracks and multiple analyses running concurrently as a first-principles design constraint. Not bolted on - built in from the start. Queue management, resource allocation, progress tracking per job.
**Rationale:** Real projects will have multiple tracks each with multiple analysis types. Sequential processing would be unusably slow.

### D144: Overlapping events get stacked indicator
**Date:** 2026-03-13
**Status:** Decided
**Decision:** When events overlap in time (especially in clip mode), a visual indicator/symbol designates stacked events. Prevents hidden events. Specific visual treatment TBD.
**Rationale:** Hidden events are lost data. Users need to know when clips overlap.

### D145: Manual event creation - reference CuePoints app
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Manual event building and addition is a core feature. Draw/pencil tool for creating events by hand. Reference the CuePoints app for interaction patterns. Users should be able to create, place, and edit events as easily as ML-detected ones.
**Rationale:** ML detection is a starting point, not the final word. Manual cue placement is a fundamental LD workflow.

### D146: Positive/negative classification feedback integrated
**Date:** 2026-03-13
**Status:** Decided
**Decision:** The timeline must integrate the classification feedback loop (per D129 user-powered model improvement). Users can mark events as correct (positive) or incorrect (negative), reclassify them, and this feedback flows into the model improvement pipeline. Must be frictionless - one or two clicks max.
**Rationale:** Classification accuracy improves only if correction is effortless. Baked into the timeline, not a separate tool.

### D147: No video support at launch
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Timeline handles audio only. No video/filmstrip support at launch.
**Rationale:** Scope control. Audio analysis is the core product.

### D148: No cross-track comparison at launch
**Date:** 2026-03-13
**Status:** Decided
**Decision:** No side-by-side comparison of different songs (structural alignment, BPM matching between tracks). Out of scope for v1.
**Rationale:** Scope control. Interesting future feature, not core.

### D149: Event verification workflow - review gate for low-confidence events
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- Events below a confidence threshold require explicit user review before they can be exported or used in show programming.
- Known problem: quiet events frequently misclassified as positives by current models.
- **Review flow:** Low-confidence events are visually distinct (opacity, "?" badge per D130). User must confirm or reject before export to ShowManager.
- Positive/negative feedback on reviewed events feeds into model improvement (D146).
- This is a **gate**, not a suggestion - unreviewed low-confidence events cannot flow to show output.
**Rationale:** Wrong classification → wrong lights → bad show. The cost of a missed review is higher than the friction of requiring one. Griff: "My prelim models misclassify quiet events as positives often."

### D150: Track duration limit - no long-form DJ sets
**Date:** 2026-03-13
**Status:** Superseded by D178 (10 min cap decided)
**Decision:** Enforce a maximum track duration. Specific limit TBD, but long DJ sets (1hr+) are explicitly out of scope. Users with long recordings should split into individual tracks before importing. Clear error/guidance if limit exceeded.
**Rationale:** Processing pipeline, memory, and timeline rendering all degrade at extreme durations. Better to set a clear boundary than silently choke. EchoZero analyzes songs, not marathon sets.

### D151: Re-analysis creates a new take - manual edits never lost
**Date:** 2026-03-13
**Status:** Decided
**Note:** Terminology standardized to **Take** for user-facing analysis results.
**Decision:**
- Re-running analysis on a track that already has results creates a **new take**, never overwrites the existing one.
- All takes are preserved and browsable.
- Users can **cherry-pick events across takes** - select events from take 1 and take 2 to compose the best result.
- Manual edits (boundary adjustments, reclassifications, manually created events) are part of the take they were made on and are never destroyed by new analysis runs.
**Rationale:** User work is sacred. An ML re-run should never destroy manual corrections. Takes solve this cleanly - each run is its own snapshot, user composes the final result.

### D152: Block actions during active analysis
**Date:** 2026-03-13
**Status:** Decided
**Decision:** While analysis is actively running on a track, certain actions are blocked on that track:
- No export to ShowManager (partial data)
- No re-running analysis (queue conflict)
- Editing events on the in-progress sub-layer blocked until analysis completes
- Other tracks remain fully interactive - only the processing track is gated.
- Clear visual indicator that the track is in a processing state.
**Rationale:** Partial exports and edit conflicts with in-flight analysis create unpredictable state. Better to block and show progress than allow corrupted output.

### D153: General-purpose annotation layer - built-in
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- Users can create **annotation layers** on any track - not tied to ML analysis.
- Annotations are manually placed markers with user-defined labels, colors, and optional notes.
- Use cases: bookmarks, comments, custom cue points, rehearsal notes, "lighting idea here" markers.
- Annotation layers are first-class sub-layers - same drag/reorder behavior as analysis layers.
- Distinct visual treatment from analysis events so users don't confuse manual annotations with ML output.
**Rationale:** Not every mark on a timeline comes from ML. LDs think in annotations, notes, and markers as much as classified events. This is table stakes for a professional tool.

### D154: Bidirectional ShowManager ↔ Timeline divergence checks
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- ShowManager and Timeline maintain awareness of each other's state.
- If a timeline event that has been exported to ShowManager is deleted, modified, or reclassified → ShowManager is notified and flags the affected cue(s) as divergent.
- If a ShowManager cue is modified or deleted → Timeline can surface that the linked event's show mapping has changed.
- **Two-way sync status**, not two-way auto-sync. Changes are flagged, not auto-propagated. User resolves divergence manually.
- Builds on existing ShowManager sync design (D76-D81) - extends it to individual event granularity.
**Rationale:** One-way export creates orphaned cues and silent breakage. The cost of a wrong cue firing live is too high. Both sides need to know when they've drifted.

### D155: Single-user projects - no collaboration
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Projects are single-user, single-machine. No multi-user collaboration, no conflict resolution, no shared editing. This is a permanent architectural constraint, not a "v1 limitation."
**Rationale:** Collaboration adds massive complexity (CRDTs, conflict resolution, permissions, real-time sync) for a use case that doesn't exist in the LD workflow. One person programs one show.

### D156: Undo limits - standard and reasonable
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Finite undo stack with a reasonable limit (exact number TBD - likely 50-100 steps). Undo does not cross take boundaries - each take's edits are within that take's context. Standard undo/redo behavior otherwise.
**Rationale:** Infinite undo = unbounded memory. Cross-take undo creates paradoxes (undoing into a different analysis state). Keep it simple and predictable.

### D157: Autosave with checkpoints
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- Project autosaves on a timed interval (e.g., every 60s) and on significant actions (analysis complete, export, take creation).
- Checkpoints are recoverable - if the app crashes mid-analysis, user returns to last checkpoint with analysis results up to that point preserved.
- Manual save also available (Ctrl+S).
- Autosave does not create user-visible "versions" - it's crash protection, not version history.
**Rationale:** Losing work to a crash is unacceptable. Analysis runs can be long - losing results because the app closed is a dealbreaker.

### D158: Configurable layer/sub-layer limits
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- Maximum number of sub-layers per track is configurable (default TBD - likely 8-12).
- Maximum number of main tracks is also configurable.
- When limit is reached, user is informed and must remove a layer before adding another.
- Limits exist in a settings/preferences panel - power users can raise them if their hardware supports it.
- Default limits tuned for performance and UI readability on typical hardware.
**Rationale:** Uncapped layers degrade both rendering performance and usability. Configurable limits respect different hardware while preventing the UI from collapsing under its own weight.

### D159: Empty analysis → empty sub-layer created
**Date:** 2026-03-13
**Status:** Decided
**Decision:** If analysis produces zero events, an empty sub-layer is still created. Serves as proof the analysis ran. No ambiguity between "no results" and "forgot to run."
**Rationale:** Empty states should be explicit, not invisible. User needs to know analysis completed even if nothing was found.

### D160: Non-audio file rejection
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Only audio files can be dragged onto the timeline. Non-audio files are rejected with a clear error toast. No future use case planned for non-audio media on the timeline.
**Rationale:** Scope control. Timeline is for audio analysis, not general-purpose media.

### D161: Bulk delete confirmation
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Destructive actions above a threshold (TBD - likely 10-20+ events) trigger a confirmation dialog. Single event deletes proceed without prompt. Undo is always available as a safety net regardless.
**Rationale:** Fat-finger protection. Deleting 200 manually-corrected events by accident and relying on undo is a worse UX than a one-click confirmation.

### D162: Missing audio file → relink flow
**Date:** 2026-03-13
**Status:** Decided
**Decision:** If a project references an audio file that has been moved or renamed externally:
- Project still opens. Analysis results, events, and sub-layers are preserved.
- Waveform display is unavailable for the missing file.
- User is prompted to relocate/relink the audio file via a file browser dialog.
- Once relinked, waveform renders and full functionality resumes.
- Project stores the path, not a copy of the audio file.
**Rationale:** Users reorganize files. Projects shouldn't break permanently because of a folder rename. Relink flow is standard in every DAW and NLE.

### D163: Background processing continues when app loses focus
**Date:** 2026-03-13
**Status:** Decided
**Decision:** Analysis and processing continue at full speed when the app loses focus or is minimized. No pausing, no throttling. Background processing is expected and normal behavior.
**Rationale:** Users will switch to other apps (Ableton, MA3, etc.) while analysis runs. Pausing or throttling would make long analyses unnecessarily slower.

### D164: Draggable layer reordering
**Date:** 2026-03-13
**Status:** Decided
**Decision:**
- **Main tracks** can be drag-reordered among other main tracks.
- **Sub-layers** can be drag-reordered within their parent track only.
- Cross-container moves (moving a sub-layer to a different parent, or promoting a sub-layer to a main track) are NOT exposed in the UI at launch.
- **However**, the underlying data model must support reparenting - layers are not structurally locked to their parent. This is a UI constraint, not an architecture constraint.
- Future unlock: drag sub-layer out of parent → promotes to independent track, or drag into another parent.
**Rationale:** Users expect drag-to-reorder from every timeline editor. Keeping sub-layers within their parent avoids complexity while the architecture stays flexible for future cross-container moves.

### D165: Uniform layer data model
**Date:** 2026-03-14
**Status:** Decided
**Decision:**
- Layers and sub-layers are the **same data structure**. No distinct "sub-layer" type.
- A layer is `{ id, name, color, style, events[], children[] }`.
- Hierarchy is positional (list of lists), not typed. A "top-level" layer is one whose parent is the root. A "sub-layer" is one whose parent is another layer.
- Users can nest/unnest freely (architecture supports it; UI may constrain at launch per D164).
- Events carry an `origin` field (`"ml"` | `"user"`) for provenance - ML classification and manual events coexist on the same layer. "Events" here means all events on the layer (both clip events with duration > 0 and marker events with duration = 0, per D112 terminology).
- Users can manually add events to any layer and create new sub-layers.
**Note on D165 vs D105:** This is the **conceptual/UX data model**. D105 implements layer hierarchy via `group_id` FK in the DB (not a `children[]` field). The conceptual `children[]` list maps to DB rows with the same `group_id`. D165's `events[]` field maps to the D105 `events` table. These are the same data, different representations.
**Rationale:** Eliminates special-casing between layers and sub-layers. A single recursive structure means drag-to-reparent (D164 future unlock) is trivial - no type conversion needed. The list-of-lists hierarchy is natural and extensible.

### D166: Export is ShowManager-driven, not timeline-driven
**Date:** 2026-03-15
**Status:** Decided
**Decision:** MA3 sync flows through ShowManager directly, not through a manual export action from the timeline. Timeline surfaces sync status (divergent/synced indicators per event). Right-click → "Export to..." submenu reserved for future non-MA3 targets (MIDI, CSV, OSC). No export button in the toolbar at launch.
**Rationale:** ShowManager is the bridge to MA3. The timeline's job is to show sync state, not manage export.

### D167: Git-style take model for analysis layers
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Each track has a **main take** (always visible, canonical) and zero or more **takes**.
- Running analysis creates a new take.
- Each take maintains its own independent state (events, classifications, manual edits).
- **Merge** = pull take results into main (additive).
- **Swap** = take becomes main, old main becomes a take (nothing lost).
- Takes accessible via dropdown in track header or right-click → Switch Take.
- Dropdown shows visual preview of each take's events.
- Different tracks can be on different takes independently.
- Delete take with confirmation if it has manual edits.
- Terminology: "take".
**Rationale:** Git mental model is universally understood. Takes give fearless experimentation - run different models/configs without risking main.

### D168: Analysis triggering
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- **Per-track:** Right-click → Analyze (or button in track header) → creates a take on that track.
- **Batch:** Toolbar action → "Analyze All" → runs selected profile across all tracks, each gets its own new take.
- Analysis Profile = preset config (model, sensitivity, categories).
- Running analysis always creates a new take, never overwrites main.
- Profile selection persists as a project-level default but can be overridden per-track.
**Rationale:** Both granular and batch workflows are needed. Always creating a new take protects main from accidental overwrites.

### D169: Take merge conflict resolution
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Merge triggers a **conflict review panel** showing only conflicts (clean merges auto-apply).
- Conflict types: same position different classification, same position different boundaries, one exists / one deleted.
- Per-conflict actions: Keep Main / Keep Take / Keep Both.
- Batch actions: "Keep All Main" / "Keep All Take" for speed.
- Non-conflicting events merge automatically.
- Merge completes only after all conflicts are resolved. Can cancel at any point.
- Conflict markers highlighted on timeline during review with step-through navigation.
**Rationale:** If you're going git, go git. Half-assing merge defeats the purpose.

### D170: Timeline search and filtering
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- **Search (Ctrl+F):** search bar, query by classification, label, annotation content, time range. Results highlighted, ← → to jump between matches.
- **Filter (persistent):** per-track, filter by classification type, origin (ML vs user), confidence threshold, time range. Hidden events not deleted. Filters stack. Clear all in one click.
- Both search and filter should be robust - full regex/wildcard on text, compound filters with AND/OR, confidence range sliders, arbitrary filter chains.
**Rationale:** Search is "find me this thing." Filter is "only show me these things while I work." Different workflows, both need to be powerful.

### D171: Timeline ruler and time display
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Display modes: Time (MM:SS.ms, default), Bars/Beats (requires BPM), Samples, SMPTE (HH:MM:SS:FF — see D175 for timecode details).
- Click ruler to set playhead, drag to create time selection.
- Right-click ruler → "Go to time..." for precise navigation.
- Section regions (D139) display as colored backdrops in the ruler area.
**Rationale:** Standard DAW ruler behavior. Multiple display modes serve different workflows (editing vs live show programming).

### D172: BPM handling
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Auto-detected from analysis with confidence score.
- If confidence is low, flag visually ("BPM may be inaccurate") but still show grid.
- User can override with manual entry or tap tempo.
- BPM is per-song (each song in the setlist has its own BPM, grid, and ruler context).
- Grid nudge/shift to align with audio (drag grid origin over waveform).
**Rationale:** Auto-detect first, manual override always available. Confidence flag prevents false trust in bad detection.

### D173: One song = one audio file
**Date:** 2026-03-15 (clarified 2026-03-16)  
**Status:** Decided  
**Decision:** Each song in the setlist is one audio file, deeply analyzed. Project contains a setlist of multiple songs (D90 still applies). Per-song data: events, stems, classifications, takes, annotations. Not a multi-track DAW — one audio source per song. BPM, grid, ruler are per-song. Combined with D178 (10 min cap), this prevents users from loading hour-long files as a single "song."  
**Rationale:** Scope what a "song" is without killing the setlist concept. One file per song keeps analysis clean. Length cap prevents abuse.

### D174: Downbeat detection
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Accurately detect downbeats throughout the song.
- Beat 1 alignment is secondary - if all downbeats are mapped, grid is correct.
- User can adjust/correct downbeat positions.
- Grid alignment tool: drag grid to snap to a known downbeat, remaining grid auto-aligns.
**Rationale:** Downbeats are the structural backbone. Get those right and bars/beats mode works.

### D175: Timecode support
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- SMPTE display mode in ruler (HH:MM:SS:FF).
- Configurable frame rates: 24, 25, 29.97, 30.
- Generate MTC (MIDI Timecode) and LTC (Linear Timecode) output.
- Listen to incoming SMPTE/MTC for sync (chase mode).
**Rationale:** Critical for MA3 integration - shows run on timecode. LDs need to see, generate, and chase timecode natively.

### D176: Section boundaries are soft
**Date:** 2026-03-15
**Status:** Decided
**Decision:** Sections (verse, chorus, drop, etc.) can have fuzzy/overlapping transitions. No requirement for hard boundaries - gaps and overlaps are valid. "Transition" as a section type is supported. Visual treatment: gradient or fade between adjacent sections.
**Rationale:** Music doesn't have hard cuts between sections. Forcing hard boundaries misrepresents the audio.

### D177: Ruler markers (locators)
**Date:** 2026-03-15
**Status:** Decided
**Decision:** Users can place markers directly on the ruler - separate from layer events/annotations. Navigation bookmarks: "Drop," "Build," "Breakdown," etc. Click to jump playhead. Song-level, not layer-level. Editable labels, configurable colors.
**Rationale:** Standard DAW locators. Quick navigation is essential for long songs and live show programming.

### D178: Song length cap
**Date:** 2026-03-15
**Status:** Decided
**Decision:** Max song length: 10 minutes at launch. Keeps zoom, rendering, and analysis scope manageable. Revisit for DJ sets / live recordings later.
**Rationale:** Scope control. Most songs are 3-5 minutes. 10 minutes covers extended mixes.

### D179: Drag threshold for ruler interaction
**Date:** 2026-03-15
**Status:** Decided
**Decision:** Platform-standard drag threshold (typically 4px on Windows, system default on Mac). Below threshold = click (place playhead). Above = drag (time selection).
**Rationale:** UX standard practice. Prevents accidental selections.

### D180: Hybrid zoom with adaptive grid
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Continuous zoom via scroll wheel, trackpad pinch, or keyboard +/-.
- Grid density auto-scales with zoom level (bars → beats → subdivisions → ms).
- Zoom presets: save and recall named zoom states.
- Quick zoom actions: Zoom to Fit, Zoom to Selection, Zoom to Region.
- Min zoom: full song visible. Max zoom: sample-level.
**Rationale:** Hybrid approach (Reaper/FL style) gives fluidity of continuous zoom with precision of presets. Best of both worlds.

### D181: Reference audio tracks
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- Users can drop additional audio files onto the timeline for playback only (not analysis).
- Use cases: timecode audio (LTC hour), click tracks, backing tracks.
- Reference tracks play in sync with main song but don't get analyzed.
- Visually distinct from main analysis track. Can be muted/soloed independently.
- No event layers on reference tracks - audio playback only.
**Rationale:** LDs need timecode audio and reference tracks alongside the analyzed song. Playback-only keeps analysis scope clean.

### D182: Solo/Mute - audio and event playback
**Date:** 2026-03-15
**Status:** Decided
**Decision:**
- **Audio tracks** (main song, stems, LTC, reference): standard M/S buttons. Mute = silence, Solo = hear only soloed tracks.
- **Event layers**: M/S controls real-time event playback (trigger sounds, send cues, fire OSC/MIDI). Mute = events don't fire. Solo = only this layer fires.
- Events remain visible regardless of mute state - M/S controls playback, not visibility. Visibility is handled by filters.
**Rationale:** M/S is always about what you hear or what fires. Visibility is a separate concern.

### D183: Project file integrity — atomic save
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Use industry-standard atomic save: write to temp file, then rename/swap on success. Crash during save cannot corrupt the project.  
- If core app files are moved/corrupted: app is broken, reinstall required. No self-healing for application-level files.  
- Project-specific files (audio, project folder): handle relocation gracefully via D162's relink flow. If project folder moves, prompt user to locate missing files.  
**Rationale:** Atomic save is the industry standard (SQLite WAL, DAWs, etc.). Separate app integrity from project integrity — app files are the installer's problem, project files are the app's problem.

### D184: Project version compatibility — forward-incompatible
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Projects saved with a newer version of EchoZero cannot be loaded in an older version.  
- On load attempt: clear error message with current version + required version. Subtle nudge to upgrade.  
- Backward compatibility: newer EchoZero CAN load older project formats (migration on open).  
- Project file includes a `min_app_version` field.  
**Rationale:** Forward compatibility is expensive to maintain and limits what new versions can do. Nudge-to-upgrade is the standard approach (Ableton, FL Studio, etc.).

### D185: OSC Gateway auto-reconnect and divergence handling
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- MA3 disconnect/reconnect is expected behavior — handle gracefully.  
- OSC Gateway auto-reconnects on connection loss (configurable interval per D119).  
- On reconnect: automatic full diff of all sync layers.  
- If no divergence: silent reconnect, status returns to "Connected/Synced."  
- If divergence detected: show divergence indicator on affected sync layers. Ask user how to proceed per layer (MA3 Master / EZ Master / Union / manual review via take).  
- Never auto-overwrite on reconnect divergence — always surface it.  
**Rationale:** Live production environments have flaky connections. Auto-reconnect is mandatory. But divergence after disconnect could mean someone edited on the console while disconnected — user must decide.

### D186: Multi-user MA3 sync — console as source of truth
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Multiple EchoZero instances syncing to the same MA3 console is architecturally supported.  
- MA3 is the source of truth — all instances should converge to MA3's state.  
- Each instance manages its own sync layers independently.  
- No direct instance-to-instance communication — all coordination happens through MA3.  
- Conflict risk: two instances pushing different events to same MA3 layer simultaneously. Mitigated by: last-write-wins at MA3 level (MA3's native behavior).  
- **Not a priority for v1.** Document as supported-but-untested. Revisit if demand appears.  
**Rationale:** MA3 already handles concurrent connections from multiple consoles/stations. EchoZero instances are just another data source. No need to build consensus — MA3 IS the consensus layer.

### D187: Variable BPM / tempo changes
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- v1: single BPM per song. Known limitation. Documented in UI ("assumes constant tempo").  
- Future feature: tempo map — series of BPM changes at specific time positions.  
- Architecture note: BPM is already per-song metadata. Extending to a tempo map (list of `{time, bpm}` pairs) is additive, not a redesign.  
- Grid rendering already handles zoom levels — tempo map just changes the grid spacing calculation.  
**Rationale:** Most pop/electronic/EDM (the primary LD market) is constant tempo. Variable BPM matters for live recordings, jazz, prog — future feature, not launch blocker.

### D188: Bundled models — ship with installer
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Installer bundles 2-3 small core models (drums.full-mix at minimum).  
- App works fully offline on day one with bundled models.  
- Cloud registry (D123) provides additional/updated models when online.  
- If cloud registry is down: bundled models still work. Staleness indicator if newer versions exist but can't be downloaded.  
- Bundle size budget: ~50MB total for bundled models (small enough for reasonable installer size).  
**Rationale:** First-run experience must not depend on internet. Bundle the essentials, download the rest.

### D189: Classification confidence — dual-label tolerance
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- When top two classification labels are within a configurable tolerance (default: 15% confidence gap), mark event as ambiguous/dual-label.  
- Visual treatment: event shows both labels (e.g., "hihat/ride") with split color or secondary indicator.  
- Ambiguous events are flagged for review (D149 verification workflow).  
- User resolution: click to confirm one label, which feeds back into correction data (D130 flywheel).  
- If gap > tolerance: winner takes all, single label displayed.  
- Standard practice in ML classification — don't hide uncertainty from the user.  
**Rationale:** Hiding close calls creates false confidence. Surfacing ambiguity feeds the correction flywheel and teaches users to trust the system's honesty.

### D190: Layer management — configurable limits and organization
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Configurable layer limit per D158.  
- Layer panel UX for high layer counts: collapsible groups (D138 sub-track layout), search/filter (D170), drag reorder (D164).  
- "Focus mode": solo-like behavior for the layer panel — collapse all except focused layer group.  
- Layer colors auto-assigned from palette, user-overridable.  
- Exact limits and UX details refined through user testing. Architecture supports arbitrary layer counts.  
**Rationale:** Architecture shouldn't limit layers. UX should manage complexity through organization, not caps. Testing will reveal the practical sweet spot.

### D191: Take switch is undoable
**Date:** 2026-03-16  
**Status:** Decided  
**Supersedes:** Clarifies D156  
**Decision:**  
- Switching takes IS an undoable action.  
- Ctrl+Z after a take switch reverts to the previous take.  
- Then the next Ctrl+Z undoes whatever was done before the switch (in that take's undo stack).  
- Each take maintains its own undo history. Switching takes pushes a "switch" entry onto a global undo stack.  
- Global undo stack: tracks take switches and cross-take operations. Per-take undo stack: tracks edits within that take.  
- D156's "undo does not cross take boundaries" means edits in Take A can't undo edits in Take B. But the ACT of switching is itself undoable.  
**Rationale:** Take switch is a user action — it should be undoable like any other action. The two-stack model (global + per-take) keeps it clean.

### D192: Progressive waveform rendering
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Waveform rendering is progressive: low-resolution overview renders first (instant), high-resolution detail renders in background.  
- Similar to Reaper's "render in place" — background peak calculation.  
- Zoom in before high-res is ready: show what's available at current resolution, refine as data arrives.  
- Cache waveform peaks at multiple zoom levels (overview, mid, detail).  
- Background rendering continues when app loses focus (per D163).  
- 10-min song at 44.1kHz: ~26M samples. Overview peaks (1 peak per 1024 samples) = ~25K points, renders in <100ms. Full detail is the background job.  
**Rationale:** No blank tracks, ever. Progressive rendering gives instant feedback while detail builds. Standard approach in modern DAWs.

### D193: Classification models as competitive moat — flywheel strategy
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Primary competitive moat is classification model accuracy, driven by the user correction flywheel (D130).  
- More users → more corrections → better models → more users. This is the defensible loop.  
- Model-agnostic architecture (D124) means EchoZero can ALSO use external AI APIs if they emerge, but own models are the differentiator.  
- Secondary moats: MA3 integration depth, domain expertise, offline capability, community (Show Packs).  
- Strategy: invest in model training infrastructure early. Every correction captured is a compounding asset.  
**Rationale:** If a competitor ships "audio → events," they won't have the LD-specific training data. The flywheel makes the gap grow over time, not shrink.

### D194: MA2 support — future adapter, not now
**Date:** 2026-03-16  
**Status:** Deferred  
**Decision:**  
- MA2 support is architecturally possible via a separate protocol adapter (same pattern as MA3 Protocol Adapter).  
- Not in v1 scope. MA3 is the beachhead.  
- Architecture note: OSC Gateway (D82) + adapter pattern means adding MA2 = new adapter + MA2-specific OSC dialect mapping. No core changes.  
- Revisit based on user demand post-launch.  
**Rationale:** MA2 is still widely used but is being phased out. MA3 first, MA2 if market demands it.

### D195: Generic event export — XML/CSV/MIDI
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Add basic XML export as a non-ShowManager export path.  
- Supported formats at launch: XML (generic structured), CSV (simple tabular), MIDI (standard music interchange).  
- Export scope: selected layers or all layers. Time format follows ruler mode (time/bars-beats/SMPTE).  
- Export triggered via File menu or right-click layer → "Export Events..."  
- D166 said export is ShowManager-driven. This EXTENDS D166: ShowManager handles MA3 sync, but File → Export handles generic formats. These are separate paths.  
- No custom export plugins at launch. XML/CSV/MIDI cover the basics.  
**Rationale:** Not everyone uses MA3. Audio analysis → event export to generic formats unlocks film scoring, sound design, VJing, academic use. Low effort, high reach. Extends D166 without contradicting it — different export path for different purpose.
**Supersedes:** Extends D166 (adds non-ShowManager export path).

### D196: OSC inbound admission control — bounded ring buffer
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- OSC Gateway uses a bounded ring buffer (128 entries) for inbound messages.  
- When full, drop-oldest. MA3 messages are state messages, not transactions — newer state replaces old.  
- Drop-oldest is semantically correct: a dropped "cue position at T=1.0" is irrelevant if "cue position at T=1.1" arrived after it.  
- Buffer lives in OSC Gateway (not ShowManager — preserving D87 zero-network-knowledge).  
**Rationale:** Prevents OOM from unbounded buffering during high-throughput show playback. Launch-critical.

### D197: Decoupled recv/drain async loops
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Two independent async tasks in OSC Gateway: recv loop (reads UDP into ring buffer) and drain loop (processes from ring buffer → domain events).  
- Recv never blocks on processing. Processing never starves the socket.  
- ~50 lines of code. Eliminates OOM crash category.  
**Rationale:** Standard producer-consumer pattern. Decoupling prevents cascading slowdowns.

### D198: OSC drop visibility in Activity Log
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- When messages are dropped from the ring buffer, OSC Gateway publishes a drop event.  
- ShowManager surfaces it in the Activity Log: "X messages dropped due to high throughput."  
- No modal, no panic — log entry only. Professionals need diagnostic signal.  
**Rationale:** Silent drops are debugging nightmares. Log it, don't interrupt.

### D199: Audio playback — testing-informed channel limit
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- No hardcoded channel mix limit at architecture level.  
- Set a practical limit based on performance testing (main + stems + reference tracks).  
- If mix exceeds tested capacity: log warning, continue playback with best effort (skip lowest-priority reference tracks).  
- Priority order: main audio > stems > reference tracks (user-reorderable).  
**Rationale:** Hardware varies wildly. Test, find the limit, set it conservatively. Don't guess.

### D200: Timeline-audio sync — zero drift, sample-accurate
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Accurate timing is the single most important thing. Zero drift tolerance.  
- Playback block and timeline ruler are sample-locked — both reference the same audio clock.  
- Timeline position derived FROM audio playback position, not independently calculated.  
- Audio callback drives the clock. Timeline follows. Never the reverse.  
- If audio glitches (buffer underrun), timeline pauses with audio — they never desync.  
- All event triggering (OSC sends, MIDI out, visual feedback) is timed from the audio clock.  
**Rationale:** Events at specific times IS the product. If timing drifts, the tool is useless. Audio clock is the single source of truth.

### D201: Audio sample rate handling — resample on load
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Internal processing runs at a standard sample rate (44.1kHz recommended, configurable in settings).  
- On audio load: if source sample rate differs, resample to internal rate.  
- Resampled copy stored in project directory (original untouched on disk).  
- High-quality resampling algorithm (e.g., libsamplerate/SoX quality).  
- Display original sample rate in file info for reference.  
**Rationale:** Consistent internal rate simplifies ML model input, waveform rendering, and buffer management. Resample once on import, not on every read.

### D202: Take accumulation — soft limit with cleanup tools
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Takes accumulate — no auto-deletion.  
- Soft limit warning at a high threshold (e.g., 50 takes per track). Warning only, not enforced.  
- Take manager UI: list all takes, sort by date/name, bulk delete, archive.  
- Exact threshold adjustable after user testing.  
**Rationale:** Takes are cheap (event metadata, not audio copies). Let them accumulate, provide cleanup tools, optimize later.

### D203: Mass conflict resolution — batch tools
**Date:** 2026-03-16  
**Status:** Decided  
**Supersedes:** Extends D169  
**Decision:**  
- D169's side-by-side per-conflict resolution works for small conflict sets.  
- For mass conflicts (50+): batch resolution options:  
  - "Accept all from Take A" / "Accept all from Take B"  
  - "Accept all from Take A where confidence > X%"  
  - Filter conflicts by type/layer, resolve filtered set in bulk  
  - "Review remaining" opens per-conflict view for what's left  
- Quick, clean, minimal clicks for the common case (accept all from one side). Detailed review available for edge cases.  
**Rationale:** 500 conflicts one-by-one is unusable. Batch-first with escape hatch to per-conflict review.

### D204: Redo stack behavior — standard (destroyed on new edit)
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Standard undo/redo behavior: undo creates redo stack, new edit destroys redo stack.  
- No redo branching or undo trees — standard DAW behavior.  
- Per D191: take switches are on the global undo stack, edits within a take are on the per-take stack.  
**Rationale:** Standard behavior. Users expect it. Undo trees are confusing and no DAW does them.

### D205: Non-musical content handling — silence/noise detection
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Pre-classification pass: detect silence and noise-only sections (crowd noise, stage banter, count-ins, dead air).  
- Simple energy/spectral analysis — not ML. Threshold-based:  
  - Below silence threshold → mark as "silence" region, skip classification  
  - Above silence but no transients / low spectral flux → mark as "noise/ambient" region  
- Detected regions shown as subtle background shading on timeline ("silence" / "noise").  
- Events inside noise regions: still detected by onset detection but flagged as low-confidence and filtered from default view.  
- User can toggle "show noise region events" to see them.  
- Click track detection: if periodic, consistent-amplitude transients detected in noise region, flag as "possible click track" — don't classify as musical events.  
- Clean approach: one pre-pass, two thresholds (silence + transient density), no ML overhead.  
**Rationale:** False positives from noise/click bleed are a real workflow problem. Simple energy gating catches 90% of cases without adding complexity. Flag, don't delete — user decides.

### D206: Single-song analysis export/import between projects
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Export a single song's analysis data as a portable package (.ezsong or similar).  
- Package contains: event layers, takes, classifications, annotations, section data. Does NOT contain audio (licensing issues, file size).  
- Import into another project: user provides matching audio file, analysis data maps by timestamp.  
- Use case: LD shares their song analysis with another LD on the same tour.  
- If imported audio duration doesn't match: warning, import anyway, flag events beyond audio end.  
**Rationale:** Collaboration between LDs on the same show is a real workflow. Export analysis, not audio. Simple, avoids copyright issues.

### D207: Disk space exhaustion — graceful failure
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Before starting analysis (especially stem separation): check available disk space against estimated output size.  
- If insufficient: warn user with estimate ("Stem separation needs ~500MB, you have 200MB available"). Don't start.  
- If disk fills mid-operation: catch IO error, abort current operation cleanly, preserve existing project state.  
- Temp files from aborted operations cleaned up on next launch.  
- No corruption risk: atomic writes (D183) + SQLite WAL mode ensure partial writes don't damage the project.  
**Rationale:** Disk full during stem separation is a real scenario. Check first, fail gracefully, never corrupt.

### D208: Cross-platform project compatibility
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- .ez project archives are cross-platform (Windows ↔ Mac).  
- All paths stored as project-relative (no absolute paths in project data).  
- SQLite is inherently cross-platform (same file format on all OS).  
- Audio files stored inside project archive — no external path dependencies.  
- Line endings, path separators normalized on load.  
**Rationale:** LDs work on Mac at home and Windows at the venue (or vice versa). Projects must transfer seamlessly.

### D209: No-beat songs — manual grid required
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- If BPM auto-detection finds no beats (ambient, drones, sound design): no grid displayed.  
- Timeline still works in time mode (MM:SS.ms) and SMPTE mode. Bars/beats mode unavailable without BPM.  
- User can manually set BPM and place a grid if desired.  
- Onset detection and classification still run — they don't require a grid.  
- Section analysis (D139) still works — sections are time-based, not beat-based.  
**Rationale:** Grid is a convenience, not a requirement. Timeline and analysis work without it. Manual override available.

### D210: Deleted layer recovery — undo only
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Deleting a layer is undoable (standard undo).  
- No separate "trash" or "recently deleted" for layers.  
- Bulk delete confirmation per D161 prevents accidents.  
- If undo history is exhausted: layer is gone. Takes serve as the safety net — analysis results exist in other takes.  
**Rationale:** Undo is the standard recovery mechanism. Takes provide a second safety net. A separate layer trash adds complexity with minimal benefit.

### D211: Window state persistence — standard platform behavior
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Window position, size, and layout saved on close, restored on open.  
- If saved display is unavailable (monitor unplugged): fall back to primary display, centered, default size.  
- Panel layout (layer panel width, inspector panel, etc.) saved per-user, not per-project.  
- Standard platform behavior — use OS-provided window management APIs.  
**Rationale:** Standard desktop app behavior. Every DAW does this. Don't overthink it.

### D212: MIDI real-time output — future feature
**Date:** 2026-03-16  
**Status:** Deferred  
**Decision:**  
- Real-time MIDI output during playback (MIDI notes, CC, clock) is architecturally feasible.  
- Playback block already has sample-accurate timing (D200). Adding MIDI output = new output adapter.  
- Not in v1 scope. D195 covers MIDI file export. Live MIDI output is the next step.  
- Architecture note: output adapters pattern (OSC for MA3, future MIDI, future Resolume) means this is additive — no core changes.  
**Rationale:** Live MIDI output unlocks VJ integration, Ableton sync, hardware trigger — huge feature. But export-first, live-second.

### D213: Namespaced classification storage — multi-model safe
**Date:** 2026-03-16  
**Status:** Decided  
**Supersedes:** Replaces flat `label` field on events  
**Decision:**  
- Replace flat `label: str` with `classifications: dict[model_category, ClassificationResult]`.  
- Each Classify block writes to its own namespace slot (e.g., `drums.full-mix → {label: "kick", confidence: 0.92}`).  
- Multiple Classify blocks enriching the same event is expected — each writes its own slot, no overwrites.  
- ClassificationResult: `{label, confidence, model_id, model_version}`.  
**Rationale:** Flat label field would cause silent overwrites when multiple classifiers run. Namespaced dict makes multi-model a feature, not a bug.

### D214: Conflict detection rules — namespace-based
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Different namespace prefixes (drums.* vs tonal.*) = co-occurrence. Both valid, no flag.  
- Same namespace prefix (drums.modelA vs drums.modelB) = potential conflict. Apply D189 dual-label tolerance.  
- Same namespace, same label = corroboration. Show both sources, no flag.  
- Conflict detection only applies within the same semantic space.  
**Rationale:** kick + bass_note at the same time is legitimate (layered sound). hihat vs shaker at the same time from the same model type is a real conflict.

### D215: Per-model layer generation
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Each Classify block generates its own layer set when split_layers=true (D122).  
- An event can appear on both a `kicks` layer (from drums classifier) AND a `bass_notes` layer (from tonal classifier).  
- Layers are prefixed by model category for clarity: "Drums: Kick", "Tonal: Bass Note".  
**Rationale:** Multi-model classification means multi-layer presence. This is the feature — seeing the same moment through different analytical lenses.

### D216: Multi-model event inspector
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Click an event → inspector shows ALL classification annotations from all models.  
- Per annotation: model category, label, confidence, model version.  
- Inline override: user can confirm/reject any individual classification.  
- Conflict indicator when same-namespace models disagree.  
**Rationale:** Users need to see and manage what multiple models said about the same moment.

### D217: Classification chain execution — sequential for v1
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- v1: Classify blocks execute in sequence (chain). Block A enriches events, Block B enriches the same events further.  
- Each block reads existing classifications and adds its own namespace — no interference.  
- Future (v2): parallel fan-out with Merge block for advanced pipelines.  
**Rationale:** Sequential is simple, correct, and sufficient for v1. Parallel execution adds complexity without clear v1 benefit.

### D218: OSC event triggering — change-driven, not continuous
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- OSC messages fire only when events change (create, edit, delete, reclassify) — NOT during playback.  
- Playback is for the user to preview audio + visual feedback locally. OSC sync is for data changes.  
- Multiple layers changing simultaneously: batch into single OSC send cycle per D197 drain loop.  
- No event prioritization needed — change events are relatively infrequent compared to playback-rate updates.  
**Rationale:** This isn't a lighting control protocol during live playback — it's a data sync protocol. Events fire on mutation, not on every playhead tick.

### D219: Background analysis + active editing — take isolation
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Background analysis always writes to a NEW take (per D168). User edits happen on the current take.  
- No read/write contention: analysis writes to take X, user edits take Y. Different data targets.  
- When analysis completes: notification "Analysis complete on take [name]." User can switch/merge at will.  
- If user is editing the SAME take that analysis is targeting (shouldn't happen — but defensive): analysis aborts with message "Take modified during analysis. Re-run required."  
- Standard approach in version-controlled systems — isolation via branching.  
**Rationale:** Takes solve the contention problem architecturally. No locks, no conflicts, no surprises.

### D220: Audio callback deadline miss — standard handling
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Audio callback runs on a high-priority real-time thread (standard PortAudio pattern).  
- If callback misses deadline: audio glitch (underrun). Events tied to that time window fire late (not skipped).  
- Event firing is driven by audio clock position (D200) — if audio stalls, events stall with it. They stay synced.  
- Mitigation: configurable buffer size (larger = more latency but fewer underruns). Default: 256 samples (~6ms at 44.1kHz).  
- Standard DAW approach — every DAW has this tradeoff.  
**Rationale:** Audio clock is truth (D200). If audio hiccups, events follow. Skipping events would be worse than firing them late.

### D221: Close-together events — per-class binary detection
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Events closer than a configurable minimum gap (default: 10ms) from the SAME classifier are merged into one event.  
- Events from DIFFERENT classifiers at the same timestamp are distinct events (D215 — multi-model is a feature).  
- Each classification model type is a binary detector for its class. Kick detector and hihat detector run independently.  
- Flams, ghost notes, tight doubles: detected as separate events if they exceed the minimum gap threshold for their detector.  
- Minimum gap threshold is per-model-category (drums might be 10ms, tonal might be 50ms).  
- Configurable in advanced settings.  
**Rationale:** Binary per-class detection means a kick+hat on the same beat = two events from two detectors. A flam = two events from one detector if gap > threshold. Clean and predictable.

### D222: Sync policy — per-layer configurable
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Already decided in D119: each sync layer has its own resolution policy dropdown (MA3 Master, EZ Master, Union, Ask User).  
- Confirming: this is per-layer granular. Different layers can have different policies simultaneously.  
- Example: position data layer = MA3 Master, classification label layer = EZ Master. Both valid.  
**Rationale:** D119 already covers this. Confirming the per-layer granularity is intentional and correct.

### D223: Multiple output adapters — parallel, low-level
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Architecture supports multiple output adapters running simultaneously against the same event data.  
- MA3 (OSC) + Resolume (OSC on different port/address) + future MIDI (D212) — all can run in parallel.  
- Each adapter is its own lightweight service following the OSC Gateway pattern (D82).  
- Adapters are independent: no coordination between them. Each subscribes to the same domain events.  
- Keep it simple: adapter = socket + protocol translation + send. No shared state between adapters.  
- v1: MA3 adapter only. Architecture supports adding more without core changes.  
**Rationale:** Output adapters are the mechanism for serving multiple markets (D193 strategy). Low-level, independent, parallel. The platform play.

### D224: Analysis exclusion zones — use section markers
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Users can mark sections as "exclude from analysis" using the existing section system (D139/D176).  
- Section type: "Excluded" — analysis pipeline skips these time ranges.  
- Use case: spoken intros, false starts, count-ins, noise sections.  
- Works with D205 (noise detection): auto-detected noise regions could auto-suggest exclusion zones.  
- Existing system, no new UI concepts. Just a new section type.  
**Rationale:** Reuse section markers (D139) instead of inventing a new exclusion mechanism. Clean, discoverable, already designed.

### D225: Manual vs ML comparison — take diff workflow
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- "What I programmed vs what the AI found" is a take comparison workflow.  
- Import manual cues from MA3 (via ShowManager pull) into one take.  
- Run analysis → results go into a different take.  
- Use A/B comparison mode (D140) or take merge conflict view (D169/D203) to diff them.  
- No special "compare mode" needed — existing take + A/B tools cover this naturally.  
**Rationale:** Takes + A/B comparison already solve this. Don't build a separate diff feature when the existing workflow applies.

### D226: Model verification — signed models + weights_only loading
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- All models from the official registry: SHA256 hash verified on download (D125).  
- PyTorch `weights_only=True` loading (D125) — prevents arbitrary code execution from model files.  
- Custom models (.ezmodel dropped into user directory): warning banner "Unverified model — load at your own risk."  
- Future: model signing with EchoZero's key pair. Only signed models load without warning.  
- Community-shared models: must go through registry (SHA256 + review) to be "verified."  
**Rationale:** Model files are a code execution vector. Defense in depth: hash verification + weights_only + user warnings for custom models.

### D227: Mismatched .ezsong import — user's responsibility
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- When importing a .ezsong, user provides the matching audio file.  
- If audio duration doesn't match: warning, import anyway, flag events beyond audio end (per D206).  
- If timestamps don't align because it's a different version/master: that's on the user.  
- .ezsong metadata includes original audio fingerprint (hash) — if it doesn't match, show "Audio file differs from original analysis source" warning.  
- No automatic remapping between song versions — that's D10 territory (song versions, handled separately).  
**Rationale:** We can warn, we can't fix. Timestamp remapping between different audio files is a complex feature (see song versions). For basic export/import, trust the user.

### D228: Minimum system requirements — defined, tested
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Define and publish minimum system requirements:  
  - OS: Windows 10+ / macOS 12+  
  - RAM: 8GB minimum (4GB will struggle with stem separation)  
  - CPU: 4-core minimum  
  - GPU: none required (local inference runs on CPU). GPU recommended for faster classification.  
  - Disk: SSD strongly recommended. HDD will work but analysis is slower.  
  - Disk space: 2GB for app + models, plus project storage  
- Graceful degradation on low-spec hardware:  
  - No GPU: CPU inference, slower but functional  
  - Low RAM: skip stem separation, use full-mix classifiers only  
  - HDD: longer analysis times, larger audio buffer to prevent underruns  
- System check on first launch: warn if below minimum spec.  
**Rationale:** LDs use everything from touring MacBook Pros to decade-old venue PCs. Define the floor, degrade gracefully, warn early.

### D229: Cycle detection in block graph
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Block graph is a DAG (Directed Acyclic Graph) — cycles are structurally invalid.  
- Cycle detection runs on every connection attempt. If adding a connection would create a cycle: reject with error message "Cannot connect — would create a circular dependency."  
- Standard topological sort validation.  
- Most users won't hit this — it's a developer workbench (D132) concern, not timeline UI.  
**Rationale:** Standard graph validation. Prevent the problem at creation time, not at execution time.

### D230: Non-music audio analysis — architecturally supported
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Architecture is accidentally general enough for non-music analysis (podcasts, sound design, field recordings).  
- Onset detection, spectral analysis, energy curves all work on any audio.  
- Classification models are music-specific, but the model-agnostic design (D124) means speech/sound-design models could be added.  
- Not a v1 focus — music/LD market is the beachhead. But don't artificially restrict.  
- If non-music demand appears: add model categories (speech.transcription, sfx.classification), no core changes.  
**Rationale:** The architecture doesn't know it's analyzing music. It analyzes audio. Let the models define the domain, not the platform.

### D231: Multi-sensitivity onset detection — over-detect, then filter
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- DetectOnsets runs at 3 sensitivity levels (low/medium/high), takes the union, deduplicates within 10ms windows.  
- Strategy: over-detect, let the classifier filter false positives.  
- Manual events created by users (events the detector missed) become "missed onset" training signals.  
- Second compounding flywheel: classification corrections improve classifiers, missed-onset corrections improve the detector.  
**Rationale:** Missing an event is worse than detecting a false one (false positives can be filtered/deleted, missed events are invisible). Over-detection + classification filtering is the proven approach.

### D232: ValidateOnsets block — artifact cross-validation
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- New lightweight Processor block: ValidateOnsets.  
- Inputs: full-mix onset detections + stem onset detections.  
- Cross-references within 20ms window. Events detected in stem but NOT in full mix → flagged `artifact_candidate: true`.  
- Artifact candidates: confidence halved, visual treatment (dashed border + ⚠️ tooltip).  
- Still flow to classifier — good models can discriminate real vs artifact.  
- User FALSE corrections on artifact candidates → dedicated artifact training data.  
**Rationale:** Stem separation always produces artifacts. Full-mix cross-validation catches most of them without discarding legitimate quiet events. Artifacts become training data through the flywheel.

### D233: Adaptive onset detection threshold
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- DetectOnsets gets `threshold_mode: "fixed" | "adaptive" (default) | "section-aware"`.  
- Adaptive: sliding window RMS normalization (alpha=0.5 default). Quiet sections → higher sensitivity, loud sections → lower sensitivity.  
- Section-aware: optional `sections` input port. When connected to D139 section analysis, calibrates threshold per-section type.  
- Stacks with D205 noise detection: noise regions suppress onset detection regardless of threshold.  
**Rationale:** One fixed threshold fails on songs with large dynamic range. Adaptive handles this automatically. Section-aware is the premium path for even better accuracy.

### D234: (Reserved)

### D235: (Reserved)

### D236: SyncSnapshot — explicit state tracking for MA3 sync
**Date:** 2026-03-16  
**Status:** Decided  
**Resolves:** D121 open item (MA3 event identity)  
**Decision:**  
- ShowManager maintains a `SyncSnapshot` per sync layer: what EZ last pushed + what MA3 last returned.  
- No stable UIDs from MA3 — accepted as permanent constraint.  
- Snapshot enables clean diff semantics: compare current EZ state vs last-pushed state vs last-pulled MA3 state.  
- On reconnect (D185): pull MA3 state, diff against last snapshot, surface deltas.  
**Rationale:** Snapshot-based sync is actually the correct model for the LD workflow — push before show, pull back occasionally. Not a compromise, it's the right architecture given MA3's constraints.

### D237: MA3 reconciliation algorithm — fingerprint + positional fallback
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Event matching uses `(time, cmd, name)` fingerprint as primary identifier.  
- 5 reconciliation categories: unchanged, moved, modified, deleted-on-MA3, new-on-MA3.  
- Fingerprint collision fallback: positional heuristics (closest match by timestamp).  
- No MA3 Lua sidecar plugin needed — pure OSC-based reconciliation.  
- Algorithm runs on every pull and on reconnect divergence detection.  
**Rationale:** Three-field fingerprint covers ~95% of real-world cases. Positional fallback handles the rest. Clean, no external dependencies.

### D238: MA3 conflict UI — deltas only
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Conflict resolution UI shows only changed events (deltas), not full MA3 state.  
- LD who changed 3 cues reviews 3 cues, not 500.  
- Full-failure fallback: "Import MA3 full state as sync take" — user can then use standard take tools to merge.  
- Integrates with D203 batch conflict resolution for large delta sets.  
**Rationale:** Showing full state is overwhelming and useless. Deltas respect the user's time and attention.

### D239: VersionAlignmentService — DTW-based audio matching
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- `VersionAlignmentService` uses Dynamic Time Warping (librosa DTW) to align two versions of the same song.  
- Produces a `WarpMap`: mapping of timestamps from version A → version B.  
- Per-anchor confidence scores — sections with good alignment vs sections where the versions diverge.  
- WarpMap cached on Song entity for the version pair.  
**Rationale:** DTW is the standard algorithm for audio alignment. librosa provides a solid implementation. Cached map avoids recomputation.

### D240: TransferVersionAnalysis — automated event remapping
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- `TransferVersionAnalysis` command: maps events from version A to version B using WarpMap.  
- Operational command (not undoable) — creates a transfer take on the target version.  
- Events with low WarpMap confidence at their timestamp → flagged UNCERTAIN.  
- ShowManager sync state cleared for the target version — must re-push after review.  
- Transfers: events, section boundaries, annotations. All land in the transfer take, never main.  
**Rationale:** Take-based transfer is fearless — nothing on main is touched. Uncertain flags prevent blind trust in the alignment.

### D241: Version transfer UI — three-step review workflow
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- Step 1: Alignment preview — visual overlay of version A and B waveforms with warp map visualization.  
- Step 2: Config — choose what to transfer (events, sections, annotations, all).  
- Step 3: Inline review — transferred events on timeline. UNCERTAIN events have orange hatching + step-through navigation (next/prev uncertain).  
- Provenance badge "T" on transferred events until user confirms/edits them (takes ownership).  
**Rationale:** Three steps: see the alignment, choose what transfers, review the results. Progressive disclosure of complexity.

### D242: ShowManager safety on version switch
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- When switching to a new song version: ShowManager retains previous version's sync state (for reference/rollback).  
- New version requires explicit push — no automatic sync of unreviewed transferred data to live console.  
- UI: "Version changed. Review transferred events before pushing to MA3."  
**Rationale:** Never automatically push unreviewed data to a live console. One bad push during a show is catastrophic.

### D243: BPM mismatch between song versions — info, not error
**Date:** 2026-03-16  
**Status:** Decided  
**Decision:**  
- If versions have different BPMs: surface as informational during alignment preview ("Album cut: 128 BPM → Festival edit: 130 BPM").  
- DTW handles tempo differences structurally — the warp map accounts for it.  
- No double-correction needed — user doesn't manually adjust BPM AND remap events. DTW does both.  
**Rationale:** BPM difference between versions is expected (festival edits are faster, radio edits are shorter). DTW is built for this.

### D244: IPC Transport — JSON-RPC over HTTP (keep-alive) + WebSocket
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-ipc-transport
**Decision:**
- Core ↔ UI communication uses two channels:
  - **HTTP (keep-alive):** Request/response for commands (add block, execute, save, load). JSON-RPC protocol. Localhost only.
  - **WebSocket:** Real-time push from Core → UI (progress, playback position, sync events, state changes). Stays open for session lifetime.
- Lightweight HTTP server (not a web framework). No Flask/Django.
- JSON everywhere — human-readable, debuggable with curl/browser tools.
- Cross-platform: HTTP/WS work identically on Windows and macOS.
- Future-proof: same pattern scales to cloud (HTTP → HTTPS, WS → WSS) without transport changes.
**Rejected alternatives:** Unix sockets (Windows pain, local-only), gRPC (overkill), ZeroMQ (premature optimization), custom TCP (reinventing HTTP), shared memory (too complex for one dev), stdin/stdout JSON-RPC (no clean bidirectional push), MessagePack-RPC (loses debuggability).
**Rationale:** HTTP keep-alive on localhost is <1ms round-trip. Meets <5ms playback requirement. One dev can implement, debug, and maintain. Lowest complexity budget with maximum debuggability.

### D245: UI Technology — PySide6 with QPainter on Custom QWidget
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-timeline-feasibility
**Decision:**
- UI framework: PySide6 (Qt6 Python bindings).
- Timeline rendering: QPainter on custom QWidget (NOT QGraphicsView).
- Node editor: QPainter on custom QWidget.
- All rendering uses batch passes: backgrounds → waveforms → foreground → labels → playhead → selection.
- Spatial culling via interval tree — only draw visible events.
- Upgrade path if needed: (1) optimize Python layout with Cython/numpy, (2) cached QImage for waveforms, (3) Skia for waveform layer, (4) QOpenGLWidget last resort.
**Rejected alternatives:** QGraphicsView (scene graph overhead, v1's pain point), OpenGL/Vulkan (premature — bottleneck is Python iteration not rasterization), Skia (reasonable future upgrade, not starting point), Dear ImGui (unstable Python bindings, awkward for persistent state UI).
**Rationale:** QPainter on raw QWidget eliminates the QGraphicsView overhead that caused v1's performance issues. 5K events with culling fits in 16ms budget. Griff knows Qt — zero learning curve.

### D246: FEEL.py — Craft Constants Library
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-timeline-feasibility
**Decision:**
- Create a `FEEL.py` file as a first-class component containing ALL tunable UI parameters.
- Constants include: snap magnetism radius, snap easing, scroll inertia decay, playhead width/color, waveform LOD thresholds, resize handle width, hover highlight alpha, ruler tick spacing, etc.
- Agents build machinery that reads FEEL.py. Griff edits the constants.
- No magic numbers in rendering code — every feel-related value imported from FEEL.py.
- FEEL.py must exist BEFORE agents write any UI code.
**Rationale:** The craft/scaffold inversion. LLMs build excellent scaffolding but can't tune "feel." Isolating all feel parameters in one file lets Griff tune the UI in focused sessions without drowning in code. The architecture makes craft tangible and isolated.

### D247: Three-Track Build Strategy
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- **Track A:** Core Engine Foundation (Weeks 1-4) — Domain, Pipeline, Execution, Persistence
- **Track B:** Critical Processor Blocks (Weeks 2-5) — LoadAudio, DetectOnsets (parallel with A)
- **Track C:** Early UI Skeleton (Weeks 3-6) — Node editor, Timeline (parallel with A+B)
- Synchronization point at Week 5-6: first full integration test (LoadAudio → DetectOnsets → Timeline visualization)
- Each track produces versioned, tested artifacts (modules)
- No temporary hacks — if code isn't ready, parallelize further
**Rationale:** Gets Griff seeing results within 2-3 weeks while maintaining architectural integrity. Avoids the waterfall trap of building everything invisible for 8 weeks.

### D248: Build Order Detail
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- Week 1: Domain Layer + Persistence Skeleton
- Week 2: Pipeline Architecture + Unit of Work + LoadAudio begins
- Week 3: Execution Engine + DetectOnsets + Node Editor begins
- Week 4: Persistence completion + Timeline begins
- Week 5-6: Integration sync point (M1: CLI execution, M2: Visual integration)
- Week 6-12: All remaining block processors
- Week 13-20: Full UI shell
- Week 21-32: MA3 integration + polish
- Week 33-44: Beta → Launch
**Rationale:** Dependency-driven ordering with maximum parallelism at each stage.

### D249: Testing as First-Class Architectural Concern
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- Four test levels: Unit (every commit), Processor (every commit), Integration (every PR), System (nightly)
- Golden-file snapshot testing for integration tests
- Mock processors for UI testing (don't wait for real ML models)
- Unit test coverage target: >85% of domain + pipeline code
- Performance assertions in tests (fail if execution > N seconds)
- CI pipeline: pytest + black + isort + mypy --strict
**Rationale:** 40% time investment in testing saves 200% in debugging. Enables safe refactoring and sub-agent code review.

### D250: Test Infrastructure & Mocking
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- Mock processors (MockDetectOnsets, MockClassify) return deterministic fixed data for UI testing
- Test fixture directory: `tests/fixtures/` with audio samples, golden files, project templates
- Reusable conftest.py with standard fixtures
- In-memory SQLite for all persistence tests
**Rationale:** Decouple UI development from ML model availability. Enable parallel development without blocking.

### D251: Minimum Test Infrastructure Before Feature Code
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- Gate: no feature code until test infrastructure exists
- Requirements: pytest working, conftest.py with fixtures, domain entity unit tests (100% coverage), pipeline testable with mock handlers, CI pipeline green, golden file test harness working, mock processor base class, pytest-cov reporting, performance assertions
- Time investment: ~1 week, ~400 lines
**Rationale:** Catch 80% of bugs before they hit Griff's UI. Enable safe refactoring.

### D252: Sub-Agent Parallelization Strategy
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- 6 independent work units for sub-agents:
  1. Event Bus & Progress (~2 days)
  2. Persistence & Repositories (~3 days)
  3. LoadAudio Processor (~2 days)
  4. DetectOnsets Processor (~3 days)
  5. Node Editor UI (~4 days)
  6. Timeline Editor UI (~4 days)
- Maximum parallelism: 2-3 concurrent agents
- Work unit sweet spot: 2-4 days each
- Each unit is complete, tested, and mergeable independently
**Rationale:** Small enough to avoid agent context bloat, large enough to minimize coordination overhead.

### D253: Merge Conflict Prevention
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- **Vertical slicing by module:** Each agent owns a directory. No file edited by two agents simultaneously.
- **Interface-first contracts:** Domain entities are READ-ONLY for sub-agents (Griff writes, others read). Each unit has a public interface defined by Griff.
- **Git discipline:** Feature branches, max 2-3 days before merge, rebase strategy (no merge commits).
- **Code review before merge:** All agents submit PRs. Griff reviews. Automated tests must pass.
**Rationale:** Eliminate merge conflicts by preventing them architecturally, not resolving them reactively.

### D254: Ideal Work Unit Size
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- Too small (<1 day): overhead exceeds value
- Too large (>5 days): context bloat, hard to review, delayed integration
- Sweet spot: 2-4 days per work unit
- Each unit must be complete, tested, and mergeable — no "Part A waiting for Part B"
**Rationale:** Optimizes for review cadence and integration testing frequency.

### D255: Risk Reduction — Spike First
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- **Audio timing (D200):** Spike Week 1 (~2h). Test PortAudio + PySide6 drift. Gate: <5ms over 5 minutes.
- **ML classification:** Spike Week 4 (~1-2 days). Train simple CNN on drum onsets. Gate: >75% accuracy.
- **Execution performance:** Built into Week 3 design. Gate: <10s for 1-minute song.
- **MA3 OSC:** Defer to Week 8. Mock for testing until then.
- Every spike becomes a permanent integration test.
**Rationale:** Address highest-risk components before committing to full implementation.

### D256: Risk-First Integration Testing
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- Every spike produces an integration test that runs permanently
- Audio timing test: verify <5ms drift over 5-minute playback
- ML classification test: verify >75% accuracy on holdout set
- Execution performance test: verify <10s for 1-minute song processing
**Rationale:** Spikes are not throwaway — they become regression guards.

### D257: Role Allocation
**Date:** 2026-03-17
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-strategy
**Decision:**
- **Griff:** Architect + integrator. Define domain + interfaces, review all PRs, run integration tests, spike risks, orchestrate merges. ~40 hrs/week for orchestration (aspirational — actual TBD).
- **Claude Code:** Domain layer, pipeline, execution engine, LoadAudio, DetectOnsets, node editor, timeline, ML spike, classifier, later full UI shell.
- **Codex:** Event bus, persistence, TranscribeNote, TranscribeLib, Separator, AudioFilter, export processors, performance optimization.
- Sub-agents handle ~80% of code volume. Griff handles 100% of architecture + review.
**Rationale:** Leverage sub-agent parallelization while maintaining architectural coherence through single-point review.

### D258: Style Bible (STYLE.md)
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Create `STYLE.md` in repo root as mandatory context for every agent session.
- Covers: naming conventions, module/function size limits, documentation requirements, FP compliance checks, banned anti-patterns, consistency rules.
- Every public class/function gets a docstring. Every file gets a 3-line header (what/why/how).
- No file exceeds 300 lines (500 hard limit). No function exceeds 30 lines (50 hard limit).
- Agent exit checklist included — agents self-verify before submitting.
**Rationale:** Makes slop structurally impossible by making the rules explicit. Makes review fast — Griff can spot violations in 5 seconds, not 5 minutes.

### D259: Three-Tier Review System
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- **Tier 1 (Auto-merge):** Tests, infrastructure, formatting, dependency updates. Chonch merges if CI passes.
- **Tier 2 (Architecture check):** New modules, patterns, schema changes. Chonch verifies against Style Bible + FP1-7, merges if clean.
- **Tier 3 (Griff decision):** Domain model changes, user-facing behavior, anything uncertain. Chonch sends a yes/no question with context. 30 seconds per review.
**Rationale:** ~80% of merges never reach Griff. What does reach him is pre-digested into 30-second decisions.

### D260: Coherence Guardian Role
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Chonch maintains cross-module consistency as "coherence guardian."
- Before spawning agents: writes context preamble (what/why/principles/patterns/anti-patterns).
- After receiving output: checks naming, headers, FP compliance, duplicate patterns.
- Maintains COHERENCE.md: tracks established patterns, module registry, decision queue, technical debt.
**Rationale:** The gap between "each module works" and "the codebase feels cohesive" requires active maintenance.

### D261: Work Unit Lifecycle
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Six-step lifecycle: DEFINE → CONTEXT → EXECUTE → VERIFY → MERGE → RECORD.
- Work unit spec includes: intent (3 lines), deliverables, interface contract, constraints, patterns to follow, anti-patterns to avoid.
- Context package curated per unit (not full distillation dump): STYLE.md + relevant domain + existing patterns.
- Agent exit checklist enforced before submission.
**Rationale:** A well-defined work unit produces good code regardless of which agent runs it. A vague one produces slop regardless of agent quality.

### D262: Three Workstreams
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- EchoZero is three parallel workstreams:
  1. Core Engine (Python) — audio analysis, pipeline, ML, persistence.
  2. Desktop UI (PySide6 → Swift) — user-facing application.
  3. Web Presence (Website + Licensing) — marketing, purchase, activation.
- Independent workstreams with defined interfaces between them.
**Rationale:** Clear separation enables parallel development without coupling.

### D263: Website Is Not Blocking
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Website does NOT block engine or UI work. Separate track.
- Phase 1: Landing page (Webflow, can start after Week 8).
- Phase 2: Purchase + license flow (Stripe, needed by Week 33 for beta).
- Phase 3: User portal (post-launch, only if volume justifies).
**Rationale:** A working product with no website beats a beautiful website with no product.

### D264: Swift Is v2 UI
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- PySide6 is v1 UI. Swift is v2 UI. Sequential, not simultaneous.
- PySide6 first because: same language as Core, faster iteration, cross-platform Day 1, proves product first.
- API Contract enables the swap — Swift UI is just another API client.
- Protect for Swift now: no Qt types in API contract, no Qt in domain logic, FEEL.py transport-agnostic, document behaviors not implementations.
**Rationale:** Prove the product before investing in a native rewrite. The multi-process architecture (D244) already enables the swap.

### D265: Cohesion Problem — Terminology, Feature, Design Drift
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Acknowledge three drift risks: terminology drift (same thing, different names), feature drift (promises vs. reality), design language drift (visual inconsistency).
- Address via GLOSSARY.md, cross-workstream sync points, and design lock.
**Rationale:** Drift is the inevitable result of parallel development by multiple agents. Must be actively managed.

### D266: The Glossary (GLOSSARY.md)
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Maintain `GLOSSARY.md` in repo root. Every concept has ONE canonical name.
- All agents, documentation, UI text, website copy use glossary terms.
- When two documents disagree on terminology, GLOSSARY.md wins.
**Rationale:** One vocabulary across the entire product eliminates terminology drift.

### D267: Single Source of Truth Map
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Maintain a map in COHERENCE.md linking every concept to its authoritative definition location.
- When two documents disagree, the authoritative source wins. Fix the other document.
**Rationale:** Prevents conflicting definitions from accumulating across the growing document set.

### D268: Cross-Workstream Sync Points
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Three mandatory sync points:
  1. **Feature Lock (Week 20):** What the product does is decided. No new features after this.
  2. **Copy Lock (Week 30):** All user-facing text finalized. Glossary terms everywhere.
  3. **Design Lock (Week 32):** Visual design finalized. Consistent across website, UI, installer.
**Rationale:** Between sync points, workstreams evolve independently. Sync points force alignment.

### D269: Griff's Actual Workflow
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Morning check (5-10 min): scan overnight output, answer Tier 3 questions, read daily digest.
- Evening session (30-60 min, when available): deeper review, test app, tune FEEL.py, domain decisions.
- Weekend session (2-4 hours, occasionally): bigger architectural work, domain specs, design decisions.
- Workflow must be productive in 5-minute windows.
**Rationale:** Griff is a touring LD with intermittent availability. The workflow adapts to him, not the other way around.

### D270: Daily Digest
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Chonch produces a daily digest via Telegram. Max 15 lines.
- Contents: what merged (with line counts), what needs input (yes/no questions), progress stats, what's next.
- Tier 3 questions formatted as yes/no with context and defaults.
**Rationale:** Pre-digested information optimized for phone reading in 2 minutes.

### D271: Decision Queue
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Max 5 Tier 3 items in queue at any time. Tracked in COHERENCE.md.
- Each item has a "default if no response in 48h" — what Chonch would do if Griff doesn't respond.
- Items older than 48h: Chonch takes the default action and notes it. Griff can override later.
**Rationale:** Work never blocks on Griff's availability. Defaults prevent queue backup.

### D272: Quality Gate Checklist
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- **Automated (CI):** All tests pass, mypy --strict, black, isort, no file >500 lines, coverage ≥85%.
- **Manual (Chonch):** Docstrings, file headers, naming, FP compliance, no duplicate patterns, no glossary violations, no Qt in domain.
- **Domain (Griff — Tier 3 only):** Domain model correctness, user-facing behavior, FEEL.py changes.
**Rationale:** Layered quality gates catch different types of issues at different costs.

### D273: Zero Tolerance for Broken Tests
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- A merge that breaks any existing test is automatically rejected. Fix first, then merge.
- No "fix it in the next PR." No "it's just a flaky test."
**Rationale:** With multiple agents writing concurrently, one broken test cascades into conflicting fixes. Mainline must always be green.

### D274: Document Maintenance Schedule
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- DECISIONS.md: per decision. DISTILLATION.md: weekly. STYLE.md: when new patterns emerge.
- GLOSSARY.md: when new concepts appear. COHERENCE.md: per merge. FEEL.py: during tuning sessions.
- API-CONTRACT.md: when endpoints change. Daily digest: daily. Memory files: per session.
**Rationale:** Each document has a clear owner and cadence. No orphaned documents.

### D275: Document Staleness Detection
**Date:** 2026-03-20
**Status:** Decided
**Source:** S.A.S.S. Panel — echozero-dev-workflow
**Decision:**
- Every document has a `Last verified: YYYY-MM-DD` line.
- If not verified in 2 weeks, Chonch flags for review.
- "Verified" means someone read it and confirmed it still reflects reality.
**Rationale:** Stale docs create zombie patterns — agents read outdated guidance and implement the wrong thing.

---

## Ingest Elimination, Schema Migration, Song Versions, Ad-Hoc Pipeline UX (2026-03-29)

### D276: Kill Ingest — waveform generation is a block
**Date:** 2026-03-29
**Status:** Decided
**Supersedes:** Eliminates the IngestPipeline concept entirely
**Decision:**
- The "ingest" concept is eliminated. There is no pre-engine shadow pipeline.
- "Add Song" becomes two app-layer operations: **store file** (content-addressed copy) + **create default PipelineConfig** (template factory → serialized graph in DB). Near-instant.
- **Waveform generation becomes a `GenerateWaveform` processor block** in the engine pipeline. It outputs a downsampled peak array for UI rendering.
- All computation on audio is a block. No exceptions, no shadow pipelines. This is FP1 (Pipeline-as-Data) applied consistently.
- The first analysis run includes waveform generation. Cache (content-addressed) ensures it only runs once.
- Metadata scanning (duration, sample rate, channels) moves to LoadAudio — it already derives this. No redundancy.
- **Consequence:** Users see a single progress flow (analysis) instead of two phases (ingest → analysis). "Add Song" is instant; "Analyze" is the first heavy operation.
**Rationale:** Ingest duplicated engine-like work outside the engine. Waveform gen, metadata scanning — all computation that belongs in blocks. Eliminating ingest removes redundant code, simplifies the user mental model, and enforces FP1 consistency. If it's computation on audio, it's a block.

### D277: Schema migration for PipelineConfig — versioned JSON + migration chain
**Date:** 2026-03-29
**Status:** Decided
**Decision:**
- `graph_json` in PipelineConfig includes a `"schema_version": 1` field at the top level.
- On deserialize (`config.to_pipeline()`):
  1. Read `schema_version`
  2. Walk registered migrations: `v1→v2`, `v2→v3`, etc.
  3. Each migration is a **pure function**: JSON dict in → JSON dict out.
  4. New params → default values. Removed params → dropped silently. Renamed params → mapped.
  5. Unknown block types → block marked as `DEGRADED` (skipped on run, warning shown in UI).
  6. Migrated JSON written back to DB on first load (lazy migration).
- **Fail-safe:** If migration throws, config is marked `CORRUPT`. User can still see it, cannot run it. Option: "Reset to Template" rebuilds from the current template version.
- One migration file per version bump. Pure functions, easily testable. No ORM magic.
- Migrations directory: `echozero/migrations/pipeline/`
**Rationale:** Simple, fail-safe, forward-only. Defaults for new params and silent drops for removed params handle 95% of cases. DEGRADED blocks and CORRUPT configs handle the rest without data loss.

### D278: Song update = new SongVersion, configs copied
**Date:** 2026-03-29
**Status:** Decided
**Extends:** D90 (SongVersion entity)
**Decision:**
- When a user updates a track ("Update Track" on existing Song):
  1. App stores new audio file (content-addressed, no collision with old file).
  2. App creates new `SongVersion` linked to the same Song.
  3. App **copies all PipelineConfigs** from the previous active version → new version (same graph, same knob values).
  4. All copied configs start as `PENDING` (never been run on this audio).
  5. Old version's Layers and Takes are **untouched** (historical record).
  6. User hits Analyze → engine runs on new audio with same settings.
- **Version switcher UI:** Song card shows current version. Small caret/dropdown in corner: `v1 ▾`. Click to see all versions with dates. Switch is instant (loads different SongVersion's layers/takes). Active version is bold. Old versions are read-only unless user explicitly "activates" them.
- **No version comparison UI for V1.** Just switch and look. Diff view is a natural V2 feature.
**Rationale:** Song updates are a huge workflow for touring LDs — same song, new arrangement or new master. Copying configs means the user's pipeline settings carry forward automatically. Old versions are preserved for reference/rollback. The version switcher is discoverable but non-intrusive.

### D279: No results pool — layers + takes + ad-hoc pipeline dialog
**Date:** 2026-03-29
**Status:** Decided
**Decision:**
- There is no "results pool" or "staging area" for pipeline outputs. Pipeline runs produce layers with takes. Always.
- **Default pipelines ("Analyze All"):** Auto-create layers, no confirmation. Current behavior.
- **Ad-hoc pipelines (right-click layer → "Run Pipeline..."):** Confirmation dialog with preview stats and three paths:

```
┌─────────────────────────────────────────────┐
│  Note Transcription — 312 notes detected     │
│                                               │
│  ○ Create new layer: "Vocal → Notes"         │
│  ○ Add as take to: [Existing Layer ▾]        │
│                                               │
│  ☐ Continue pipeline → [Select next step ▾]  │
│                                               │
│              [Cancel]  [Apply]                │
└─────────────────────────────────────────────┘
```

  - **Create new layer:** Fresh layer, results as main take. Default for new output types.
  - **Add as take:** Dropdown shows compatible layers (same data type). Results become a non-main take. User can compare, promote later.
  - **Continue pipeline:** Feed results forward as input to another pipeline run. Chain steps on the fly. Only persist at the final step.

- **Custom pipelines (Stage Zero Editor):** Same dialog as ad-hoc.
- **Compatibility filtering:** "Add as take" only shows layers whose data type matches (EventData → event layers, AudioData → audio layers). "Continue pipeline" only shows processors whose input port type matches the current output.
- **Continue pipeline flow:** Each step is another engine run where input = previous output. App holds intermediate results in memory. No new persistence model. User commits when ready.
**Rationale:** The Take System already handles variation within a layer. Layers handle "do I want this analysis at all." No new persistence concepts needed. The timeline IS the workspace — adding a separate staging area is a second place to look. "Continue pipeline" gives exploratory power without orphan management.

### D280: Stale cascade with human-readable reasons
**Date:** 2026-03-29
**Status:** Decided
**Extends:** D57-D59 (DataState, staleness)
**Decision:**
- When a user changes a knob on Block B:
  1. App traces all blocks downstream of B in the graph.
  2. All downstream PipelineConfigs marked `STALE`.
  3. `stale_reason` stored per config: human-readable, specific. Not just "settings changed" but **what** changed.
  4. Example: `"Block 'Separator' setting 'model' changed from 'htdemucs' to 'htdemucs_ft'"`
  5. Multiple changes before re-run compound: user sees all reasons.
- **Timeline/Layer visualization:**
  - Stale layers get a subtle **amber tint** on their header bar.
  - Hover shows the reason string.
  - Small **⟳ icon** appears — click to re-run just that layer.
- **Node graph visualization:**
  - Downstream connections from the changed block go **amber/dashed**.
  - Changed block gets a small dot indicator.
  - Instant visual read on blast radius without being noisy.
- **Global indicator:** If anything is stale, a subtle banner or badge: *"3 layers stale — Re-run All"*. One click to clear the backlog.
- User decides when to re-run. Stale is informational, not blocking (except for export — stale layers cannot be pushed to MA3 without re-run or explicit override).
**Rationale:** Stale cascade is already decided (D58). This adds the UX layer: specific reasons, visual indicators at both timeline and graph level, and one-click re-run. The "why" is the key addition — users need to understand blast radius before committing to a re-run.

### D281: No race condition between add-song and analyze
**Date:** 2026-03-29
**Status:** Decided (resolved by D276)
**Decision:**
- With ingest eliminated (D276), "Add Song" is near-instant (file copy + DB rows + template config creation, milliseconds).
- Analysis is the first heavy operation and is always user-initiated.
- No race condition is possible — there is no async ingest phase that could still be running when the user hits Analyze.
- If the user triggers Analyze, the PipelineConfig exists (created during Add Song) and the audio file is stored. All preconditions are met by construction.
**Rationale:** Killing ingest (D276) eliminates the race condition entirely. No guard needed because the precondition (config + audio stored) is guaranteed before analysis can be triggered.

### D282: Adaptive review loop is EZ-owned and lane-separated
**Date:** 2026-04-26
**Status:** Decided
**Decision:**
- The adaptive review loop is one EZ-owned operator workflow, not a loose handoff between unrelated tools.
- The loop is split into five lanes with explicit ownership:
  - review signal lane
  - Project review-dataset lane
  - shared core-dataset promotion lane
  - model build lane
  - runtime bundle adoption lane
- EZ owns operator actions, launch surfaces, and pending-work adoption policy.
- Foundry owns persisted review signals, dataset versions, train runs, artifact validation, and runtime bundle installation.
- Only explicit review commits create reusable training signal. Silence, untouched Events, and generic edits do not.
- Project-specialized model creation must resolve from a persisted Project review-dataset version, not directly from raw review rows.
- Promoting or adopting a Project-specialized model updates pending work only. It does not silently rewrite already reviewed or already processed main truth.
- Runtime adoption remains policy-based. Saved custom manifest paths stay pinned unless the operator explicitly changes them.
**Rationale:** The review flywheel only becomes trustworthy if review truth, dataset state, training state, and runtime adoption stay separate. This decision keeps the operator flow simple in EZ while preventing hidden truth mutation, accidental data bleed, and fragile path-based coupling between training outputs and future analysis work.

---

## Reference Documents
- `docs/UNIFIED-IMPLEMENTATION-PLAN.md` - Canonical EZ2 implementation authority
- `docs/ARCHITECTURE.md` - Current EZ2 system architecture
- `docs/APP-DELIVERY-PLAN.md` - Canonical app workflow and release contract
