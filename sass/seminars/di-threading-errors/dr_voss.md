# Dr. Voss 🔬 — DI/Bootstrap, Threading Model, Error Model

## Position

The three questions in this brief are not independent — they form a dependency chain, and getting the order wrong will produce a system that looks clean in diagrams but deadlocks under load. Threading dictates the error model (you cannot use exceptions across thread boundaries), and the error model dictates how handlers are authored, which constrains DI. I will address them in reverse causal order: threading first, then errors, then DI.

The threading model is where this design will live or die. D22 says per-block locks, D5 says background threads never touch Qt, D41 says STORE dispatches back through the pipeline. That last one is the trap. If a background worker dispatches `StoreExecutionResult` through the pipeline, and the pipeline wraps commands in UoW middleware that acquires a SQLite write connection — you now have background threads competing for the single SQLite writer. This is not a theoretical concern; it is the *default failure mode* of this architecture. The pipeline must be thread-agnostic in *acceptance* (any thread can submit), but the UoW/write path must be serialized onto a single dedicated writer thread. Anything else is a race condition waiting for a demo day.

The error model must be a `Result[T, E]` type — not exceptions. Exceptions do not cross thread boundaries cleanly, they cannot be pattern-matched by middleware, and they make "did the handler fail or did the infrastructure fail?" ambiguous. A discriminated result type forces handler authors to be explicit, lets middleware make rollback decisions without try/except gambling, and — crucially — can be serialized back from a background thread via a simple queue. The DI story then falls out naturally: module-level wiring with explicit constructor calls, no framework, no magic. Seventeen processor types is not enough to justify a container. A `register_processors()` function with a dictionary is all you need.

## Key Insight

**`StoreExecutionResult` dispatched from a background thread (D41) will deadlock or corrupt if it enters the same UoW middleware path as UI-originated commands.** Everyone will focus on "how do we wire handlers" and "what error type do we use." Nobody will notice that the background→pipeline→UoW path creates a second writer competing for SQLite. You need a dedicated write-serializer — either a single-threaded executor or an asyncio-style event loop for the write path — that *all* threads submit to. The pipeline isn't just "thread-agnostic"; the write side must be thread-*serialized*.

```python
# The subtle bug everyone will miss:
# Background thread A finishes execution, dispatches StoreExecutionResult
# Background thread B finishes execution, dispatches StoreExecutionResult  
# Both enter UoW middleware → both try to acquire write connection
# SQLite: "database is locked" or silent corruption with WAL mode

# Fix: ALL write-commands funnel through a serialized writer
class WriteSerializer:
    def __init__(self):
        self._queue: queue.Queue[tuple[Command, Future]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
    
    def submit(self, command: Command) -> Future[Result]:
        future: Future[Result] = Future()
        self._queue.put((command, future))
        return future
    
    def _run(self):
        while True:
            cmd, future = self._queue.get()
            result = self._pipeline.handle(cmd)  # UoW lives here, single thread
            future.set_result(result)
```

## Risk

If my perspective is ignored, you will ship a system that works perfectly in single-threaded tests, passes every unit test, looks beautiful in code review — and then deadlocks the first time two blocks finish executing within the same SQLite busy timeout window. The error will be intermittent, non-reproducible in development (because your test machine is fast enough to serialize accidentally), and will manifest as "the app hangs for 5 seconds then works" in production. You will blame Qt. It is not Qt. It is your write path.

## Verdict

**Q1 — DI/Bootstrap:** Module-level wiring. No container, no framework. A `bootstrap.py` that constructs everything with explicit `__init__` calls. Processors registered via a plain `dict[BlockType, type[Processor]]` populated by a `register_processors()` function. The pipeline is constructed once, injected everywhere. If you cannot trace every dependency by reading one file top-to-bottom, you have failed. For tests, call the same constructors with test doubles. Seventeen types is a dictionary, not a framework.

**Q2 — Threading:** Three execution contexts, no more. (1) **Caller thread** (UI, CLI, MCP) — submits commands to the pipeline. (2) **Write-serializer thread** — owns the SQLite write connection, runs UoW middleware, processes all mutating commands sequentially. (3) **Worker thread pool** — runs processor execution (the EXECUTE phase from D16). Caller submits command → write-serializer handles TRIGGER/PULL/STORE. Write-serializer dispatches EXECUTE to thread pool. Worker completes → submits `StoreExecutionResult` back to write-serializer (not directly to pipeline). `CancellationToken` is a shared `threading.Event`, passed into the processor at dispatch time, checked at breakpoints per D23. CommandSequencer and AudioPlayer that need Qt's main thread get their results posted via `QMetaObject.invokeMethod` or a Qt-free callback interface that adapters implement.

**Q3 — Error Model:** `Result[T, E]` as a frozen dataclass. No exceptions for domain/validation errors. Exceptions are reserved for infrastructure failures (disk full, SQLite corruption) and are caught exactly once — in the outermost pipeline middleware — wrapped into `Result.failure(InfrastructureError(...))`, logged, and returned. Handler signature: `def handle(self, command: C, uow: UnitOfWork) -> Result[T, DomainError]`. Middleware checks `result.is_failure` for rollback decisions — no try/except in UoW middleware. Background errors return via the write-serializer's `Future[Result]`, which the caller can await or attach a callback to. Four error categories, no more: `ValidationError` (caught before handler), `DomainError` (handler returns it), `CancellationError` (token was set), `InfrastructureError` (caught at boundary).

```python
@dataclass(frozen=True)
class Result(Generic[T, E]):
    value: T | None = None
    error: E | None = None
    
    @property
    def is_ok(self) -> bool: return self.error is None
    
    @staticmethod
    def ok(value: T) -> Result[T, E]: return Result(value=value)
    
    @staticmethod  
    def fail(error: E) -> Result[T, E]: return Result(error=error)

# Handler — impossible to forget error handling
class ExecuteBlockHandler:
    def handle(self, cmd: ExecuteBlock, uow: UnitOfWork) -> Result[ExecutionId, DomainError]:
        block = uow.blocks.get(cmd.block_id)
        if block is None:
            return Result.fail(DomainError("Block not found"))
        # ... proceed
        return Result.ok(execution_id)
```
