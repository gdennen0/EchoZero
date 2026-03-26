# Sage 🧮 — Data Modeler Response

## Position

The central data modeling question hiding in all three topics is: **who owns the connection, and who owns the lifecycle of state that flows through the pipeline?** Right now the brief has D39 (UoW middleware wraps every command) and D40 (repositories are stateless, receive connection from UoW). Good. But the threading model threatens to shatter this clean ownership. The moment you have a background thread dispatching `StoreExecutionResult` back through the pipeline, you have *two* UoW lifecycles for what is logically *one* operation. That's a data integrity smell. The execution's PULL → EXECUTE → STORE is a single logical transaction that spans threads. The UoW middleware can't pretend each command is independent when the background thread's `StoreExecutionResult` is causally dependent on the original `ExecuteBlock` command.

For DI, the question is really about **where the wiring graph lives as data**. A registry is a data structure — a mapping from command types to handler factories. I want that mapping to be explicit, inspectable, and immutable after bootstrap. No runtime service location. The 17+ processor types are just rows in a registry table: `{block_type: str → processor_factory: Callable}`. Same pattern for handlers: `{command_type: Type → handler_factory: Callable}`. Two flat maps. That's your entire DI "framework." Module-level wiring builds these maps; the pipeline reads them. The maps are the source of truth for "what handles what."

For errors, the key data modeling insight is that an error is *not* an exception — it's a **domain value**. A `Result[T, E]` is data that flows through the pipeline like any other data. Exceptions are for infrastructure failures (disk died, SQLite corrupt). Domain errors (validation failed, block has no input, cancellation requested) are values you return, not throw. This distinction maps perfectly onto the UoW: exceptions trigger rollback via `__exit__`; error Results trigger rollback via explicit check before commit. Both paths are data-driven, both are testable.

## Key Insight

**The `StoreExecutionResult` dispatch (D41) must NOT go through the full middleware stack.** Everyone will assume "it's a command, it goes through the pipeline." But if it does, it gets its own UoW, its own correlation ID, its own middleware chain — severing the causal link to the original execution. Instead, `StoreExecutionResult` should be a **continuation** of the original command's UoW. The background thread should receive a `UowHandle` — a thread-safe reference to a pending UoW that defers commit until the background work completes. The STORE phase calls `uow_handle.commit()` directly. One transaction, two threads. This is the only way to maintain the invariant that PULL → EXECUTE → STORE is atomic.

```python
# The UoW handle bridges threads
class UowHandle:
    def __init__(self, connection: sqlite3.Connection, events: EventCollector):
        self._conn = connection
        self._events = events
        self._lock = threading.Lock()

    def commit(self, result: ExecutionResult) -> None:
        with self._lock:
            repo = ExecutionResultRepo(self._conn)
            repo.store(result)
            self._conn.commit()
            self._events.flush()  # dispatches domain events

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()
            self._events.clear()
```

## Risk

If you ignore the UoW continuity problem, you'll end up with `StoreExecutionResult` as a fully independent command that can fail *separately* from the execution. Imagine: ML inference runs for 5 minutes, produces output, then `StoreExecutionResult` fails validation or hits a concurrent write conflict. Now you've lost 5 minutes of work and the user's execution is in limbo — not stored, not rolled back, just gone. Worse, the original `ExecuteBlock` command already "succeeded" (it returned to the UI), so the UI shows success while the data is inconsistent. You'll chase this bug for weeks.

## Verdict

**Q1: Module-level wiring with two explicit registries.**

No framework. No container. Two dictionaries built in `bootstrap.py`:

```python
# bootstrap.py
def create_app(config: AppConfig) -> Application:
    # Handler registry: command type → factory
    handlers: dict[type, Callable[..., Handler]] = {
        RenameBlock: lambda uow: RenameBlockHandler(uow, BlockRepo),
        ExecuteBlock: lambda uow: ExecuteBlockHandler(uow, BlockRepo, processor_registry, thread_pool),
        # ... all 30+ commands
    }

    # Processor registry: block type → processor class
    processors: dict[str, type[Processor]] = {
        "whisper": WhisperProcessor,
        "rvc": RvcProcessor,
        # ... all 17+ types
    }

    pipeline = Pipeline(
        middleware=[CorrelationMiddleware(), ValidationMiddleware(), UowMiddleware(db_path)],
        handlers=handlers,
    )
    return Application(pipeline=pipeline)
```

Explicit. Greppable. No magic. Tests build their own registries with fakes. CLI/MCP call `create_app()` with different configs. Processors are registered by block type string key — same key stored in SQLite's block table. **The registry key must match the persisted `block_type` column exactly.** That's the join.

**Q2: Thread pool with UoW handle continuation.**

- Pipeline is thread-agnostic. It's a function call. Whoever calls `pipeline.send(cmd)` owns the thread.
- For instant commands (rename): runs synchronously on caller's thread. UoW opens, handler runs, commit, done.
- For execution commands: handler submits work to a `ThreadPoolExecutor` (bounded, e.g., 4 workers). Handler returns *immediately* with `Result[ExecutionStarted]`. The background thread receives a `UowHandle`. On completion, it calls `uow_handle.commit()` — no re-entry into the pipeline.
- Per-block lock (D22): each block has an `asyncio.Lock` (or `threading.Lock`) keyed by block ID. Acquired before PULL, released after STORE. Stored in a `dict[BlockId, Lock]` owned by the execution coordinator.
- `CancellationToken`: created per-execution, stored in `dict[BlockId, CancellationToken]` owned by execution coordinator. Cancel command looks up token by block ID, calls `.cancel()`. Token is passed to processor, checked at breakpoints.
- Qt marshaling for UI notifications: `uow_handle.flush()` dispatches events. Event bus has a Qt adapter that uses `QMetaObject.invokeMethod` or signals to marshal to main thread. The event bus adapter is injected at bootstrap — CLI gets a no-op adapter, Qt gets the marshaling one.

**Q3: `Result[T]` for domain, exceptions for infrastructure.**

```python
@dataclass(frozen=True)
class Success(Generic[T]):
    value: T

@dataclass(frozen=True)
class Failure:
    error: DomainError  # validation, cancellation, business rule
    
Result = Success[T] | Failure
```

- Handlers return `Result[T]`. Always.
- UoW middleware checks: if `Failure`, rollback. If `Success`, commit.
- Infrastructure exceptions (`sqlite3.OperationalError`, `OSError`) propagate as exceptions. UoW middleware catches them in `__exit__`, rolls back, wraps in `Failure(InfrastructureError(...))`.
- Error taxonomy: `ValidationError`, `DomainError`, `CancellationError` (all Result values), `InfrastructureError` (caught exception, wrapped).
- Background thread errors: caught by the thread pool worker, call `uow_handle.rollback()`, dispatch `ExecutionFailed` event through the event bus. UI adapter marshals to main thread and shows notification.
- Handler authors can't screw it up because the return type is `Result[T]` — the type checker enforces it. No try/except in handlers. If they raise, middleware catches it. If they return Failure, middleware handles it. Both paths converge.
