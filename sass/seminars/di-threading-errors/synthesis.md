# S.A.S.S. Synthesis — DI/Bootstrap, Threading Model, Error Model

**Panel:** Maya 🏗️, Kai ⚡, Dr. Voss 🔬, Rex 🔥, Sage 🧮  
**Date:** 2026-03-01

---

## Consensus (All 5 Agree)

1. **Module-level wiring. No DI framework. No container.** A single `bootstrap.py` that constructs everything with explicit constructor calls. Every dependency is visible, greppable, debuggable. Tests call the same constructors with mocks. Strongest consensus of any topic — every panelist said the exact same thing independently.

2. **Processor registry is a plain dict.** `{block_type_string: ProcessorClass}` — 17 entries, built explicitly in bootstrap. No auto-discovery, no decorators, no scanning.

3. **Handler registry is a plain dict.** `{CommandType: handler_instance}` — built explicitly in bootstrap. Pipeline looks up handler by command type.

4. **Pipeline is thread-agnostic.** It's a function call: `pipeline.dispatch(command) -> Result`. Runs on whatever thread calls it. Not an actor, not an event loop, not bound to any thread. Any thread (UI, background, MCP) can submit commands.

5. **`ThreadPoolExecutor` for background work.** stdlib, bounded (e.g., 4 workers for CPU-bound ML inference). No custom thread management framework.

6. **`CancellationToken` is a `threading.Event` wrapper.** Created per-execution, passed to the processor, checked at breakpoints (D23). Stored by block ID so cancel commands can look it up.

---

## Tensions

### T1: How Does StoreExecutionResult Get Back From Background Threads?

This is the single hardest problem in the design. D41 says STORE dispatches through the pipeline. But background threads calling the pipeline means multiple threads hitting the UoW/SQLite write path.

**Option A: Direct pipeline call, single shared write connection with lock** (Maya, Kai, Rex)
Background thread calls `pipeline.dispatch(StoreExecutionResult(...))` directly. The UoW middleware uses a single shared write connection protected by a `threading.Lock`. Writes serialize naturally. Simple. No extra infrastructure.

**Option B: Dedicated write-serializer thread** (Dr. Voss)
All write commands queue into a dedicated thread via `queue.Queue`. The write thread owns the SQLite connection and processes commands sequentially. Callers get a `Future[Result]` back. Eliminates any possibility of write contention.

**Option C: UoW handle continuation** (Sage)  
The original `ExecuteBlock` command's UoW stays open. Background thread receives a `UowHandle` and calls `commit()` directly when done — no re-entry into the pipeline. One logical transaction spanning two threads.

**Resolution: Option A wins.**
- Option B adds a dedicated thread + queue + future machinery for a problem that a `threading.Lock` on the write connection already solves. SQLite in WAL mode with a locked single connection serializes writes without contention errors. Dr. Voss's concern is valid IF the UoW creates new connections per command — but with a single shared connection (D42), the lock serializes access.
- Option C is clever but breaks D41 (everything through the pipeline) and creates a special case where execution UoWs have different lifecycles than normal commands. More complexity, not less.
- Option A is the simplest correct answer: background thread calls `pipeline.dispatch()`, UoW middleware acquires the connection lock, writes, commits, releases. If another thread is writing, it waits. For a desktop app with at most a handful of concurrent block executions, this contention is negligible.

**Important nuance:** The UoW middleware must use a **single shared write connection** (not create new connections). The connection is created at bootstrap, protected by a `threading.Lock`, and shared across all UoW instances. This is what makes Option A safe.

### T2: Error Model — Result[T] vs Exceptions

**Result[T, E] for everything** (Maya, Voss, Sage): Handlers return `Result`. No exceptions cross handler boundaries. Middleware checks result type for rollback decisions. Explicit, type-checkable, no try/except gambling.

**Exceptions for everything** (Rex): Python has exceptions, use them. Handler raises, middleware catches, rollback. Zero ceremony. Futures re-raise across threads via `future.result()`.

**Hybrid** (Kai): Result for domain outcomes, exceptions for infrastructure failures. Both caught by middleware.

**Resolution: Result[T] for handler returns, exceptions caught as infrastructure failures.**

Rex's argument that Result types aren't idiomatic Python is fair, but the benefit is concrete: the UoW middleware doesn't need try/except to decide commit vs rollback. It checks `result.is_ok`. Handler authors can't accidentally raise an unhandled exception that skips rollback logic — if they do, the outer try/except catches it as an infrastructure error and still rolls back.

Keep the Result type simple — not a monadic library, just a frozen dataclass:

```python
@dataclass(frozen=True)
class Result(Generic[T]):
    value: T | None = None
    error: PipelineError | None = None
    
    @property
    def ok(self) -> bool:
        return self.error is None
```

Three error categories, not four (merging Sage's and Maya's taxonomies):
- **ValidationError** — bad input, caught before handler runs
- **DomainError** — business rule violation, handler returns it
- **InfrastructureError** — unexpected failure, caught by middleware from exceptions

CancellationError is just a DomainError subtype — "user cancelled" is a domain concept, not a separate category.

### T3: CommandSequencer and AudioPlayer (Qt-Dependent Components)

Rex raised the clearest point: these are **not pipeline handlers**. They're Qt objects that live in the UI layer and *call* the pipeline — the pipeline doesn't call them. CommandSequencer dispatches commands (it's a UI automation tool). AudioPlayer plays audio (it's a Qt multimedia widget). Neither belongs in the command handler registry.

**Resolution:** Qt-dependent components are UI-layer objects that submit commands to the pipeline like any other adapter. They're not registered as handlers. This aligns with D38 (core never imports Qt).

---

## Decisions

### D45: Module-level wiring — no DI framework
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** A single `bootstrap.py` constructs the entire object graph with explicit constructor calls. Two plain dictionaries: `handlers: dict[type[Command], Handler]` and `processors: dict[str, type[Processor]]`. No container, no locator, no decorators, no auto-discovery. Tests call constructors directly with mocks. CLI/Qt/MCP call the same bootstrap with different configs.  
**Rationale:** 17 block types and ~30 handlers is a dictionary, not a framework. Every dependency visible in one file. The legacy 30-param ServiceContainer is what happens when you centralize with a container — the fix is no container, not a better one.

### D46: Pipeline is a synchronous function call, thread-agnostic
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** `pipeline.dispatch(command) -> Result[T]` runs on the caller's thread. Any thread can call it. The pipeline is not an actor, event loop, or thread-bound object. It's a function call through a middleware chain.  
**Rationale:** Thread-agnostic pipeline means no marshaling infrastructure. Background threads call `dispatch()` directly. The only thread boundary that needs explicit marshaling is EventBus → Qt UI, which is handled by the Qt adapter (D38).

### D47: ThreadPoolExecutor for background execution
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** stdlib `concurrent.futures.ThreadPoolExecutor` with bounded workers (e.g., 4 for CPU-bound ML inference). Managed by the execution handler, not the pipeline. Per-block `threading.Lock` for execution locks (D22), keyed by block ID.  
**Rationale:** No custom thread infrastructure. stdlib handles thread lifecycle, exception propagation, and future-based completion.

### D48: Single shared write connection with lock
**Date:** 2026-03-01  
**Status:** Decided (4-1, Voss dissented)  
**Decision:** The UoW middleware uses a single SQLite write connection created at bootstrap, protected by a `threading.Lock`. All write commands (including `StoreExecutionResult` from background threads) serialize through this lock. No dedicated write-serializer thread.  
**Rationale:** For a desktop app with at most a few concurrent block executions, lock contention is negligible. The lock provides the same serialization guarantee as a dedicated writer thread with zero additional infrastructure. If contention becomes measurable under real workloads, a write-serializer can be added without changing the handler or pipeline API.

### D49: StoreExecutionResult dispatches through pipeline via direct call
**Date:** 2026-03-01  
**Status:** Decided (4-1, Sage dissented)  
**Decision:** Background worker completes computation, then calls `pipeline.dispatch(StoreExecutionResult(...))` from the worker thread. Goes through the full middleware stack (correlation, validation, UoW). The UoW middleware acquires the shared write connection lock (D48), commits, flushes events.  
**Rationale:** One write path, one middleware stack, no special cases. Sage's UoW-handle continuation is clever but creates a separate transaction lifecycle that doesn't go through the pipeline, violating D41.

### D50: Result[T] return type for handlers, exceptions for infrastructure only
**Date:** 2026-03-01  
**Status:** Decided (3-1-1; Maya/Voss/Sage for Result, Kai hybrid, Rex exceptions)  
**Decision:** Handlers return `Result[T]` — a frozen dataclass with `value` or `error`. UoW middleware commits on `result.ok`, rolls back on failure. Unexpected exceptions (infrastructure failures) are caught by an outer try/except in the middleware, wrapped into `Result.fail(InfrastructureError(...))`, and rolled back. Three error categories: `ValidationError`, `DomainError` (includes cancellation), `InfrastructureError`.  
**Rationale:** Result makes commit/rollback decisions explicit without try/except. Exceptions reserved for genuinely unexpected failures. Simple enough that handler authors can't screw it up — return `Result.ok(data)` or `Result.fail(DomainError(...))`.

### D51: Qt-dependent components are UI adapters, not pipeline handlers
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** CommandSequencer, AudioPlayer, and any other Qt-dependent components live in the UI layer. They call the pipeline — the pipeline does not call them. They are not registered as command handlers.  
**Rationale:** Maintains D38 (core never imports Qt). These components are UI concerns that happen to trigger domain operations, not domain operations that happen to need Qt.

---

## Open Questions

1. **Connection pooling for reads:** D42 says single connection to start with read connection added later. With the shared write connection locked during writes, should queries use a separate unlocked read connection from day one to avoid blocking UI reads during STORE? Or wait until profiling shows it matters?

2. **Execution handler returning before completion:** When `ExecuteBlock` is dispatched, the handler submits to the thread pool and returns `Result.ok(Accepted(cancel_token_id))` immediately. The UI needs to know execution finished later — this happens via the `ExecutionCompleted` event (fired when `StoreExecutionResult` commits). Is this flow clear enough, or does it need more design?
