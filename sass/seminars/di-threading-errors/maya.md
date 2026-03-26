# Maya 🏗️ — The Architect

## Position

The three questions are actually one question wearing three hats: **what is the application's compositional seam, and how does work flow across it?** DI determines how components find each other. Threading determines where work executes. The error model determines how failure propagates back. If you design these independently you'll get three systems that fight each other at every boundary. The pipeline is the spine — DI wires it, threading schedules it, errors flow through it. Design from the pipeline outward.

For DI, I'm firmly in the **module-level wiring with explicit constructor calls** camp — but structured as a layered `CompositionRoot`, not a single bootstrap blob. No framework, no container, no locator. Python's dynamic nature makes DI containers pointless ceremony; you get zero compile-time safety anyway, so the "convenience" of a registry just adds indirection without payoff. A `CompositionRoot` that explicitly constructs every handler with its deps is greppable, debuggable, and makes illegal states unrepresentable. For the 17+ processors, a simple `dict[type[Command], Handler]` built explicitly in the composition root. Yes, it's verbose. That's the point — every dependency is visible.

For threading, the pipeline must be **thread-agnostic but dispatch-aware**. Any thread can submit a command. The pipeline itself is just a function call on the caller's thread. But certain commands (execution) need to schedule background work and return immediately. This means the pipeline returns a `Result[T]` synchronously for every command — for background work, that result is just "accepted, here's a tracking token." The background thread, upon completion, submits `StoreExecutionResult` back through the pipeline from its own thread. The pipeline doesn't care — it's just a function call. The UoW middleware serializes writes via SQLite's single-writer constraint naturally (connection-level locking). For the error model: `Result[T, E]` everywhere, no exceptions crossing handler boundaries, middleware catches unexpected exceptions and wraps them.

## Key Insight

**The pipeline must be a pure function call, not an actor.** Everyone's instinct will be to make the pipeline "run on" a thread, or give it a queue, or make it an event loop. Wrong. The pipeline is `def dispatch(cmd: Command) -> Result[T, PipelineError]`. Period. It executes on *whatever thread calls it*. This means background threads call `dispatch(StoreExecutionResult(...))` directly — no marshaling, no posting, no callback hell. The UoW middleware handles write serialization (it acquires the write connection, which is inherently serialized). The *only* thread boundary that needs explicit marshaling is pipeline → Qt UI notification, and that's a single adapter at the edge (`signal.emit` in the notification middleware, which is the outermost layer and the *only* place that knows Qt exists).

## Risk

If you make the pipeline thread-bound (e.g., "all commands run on the main thread" or "pipeline has its own worker thread"), you'll reintroduce the exact blocking and marshaling complexity you're trying to escape. Every background completion will need to post back to the pipeline thread, creating a serialization bottleneck. With per-block locks (D22) and a thread-agnostic pipeline, you get natural concurrency — two independent blocks can execute and store concurrently (SQLite WAL mode handles concurrent reads, and write serialization is at the connection level, which is fine for your throughput). A thread-bound pipeline kills this.

If you skip `Result[T, E]` and use exceptions, handler authors *will* forget to catch them, middleware ordering will matter in subtle ways, and you'll end up with the same swallowed-error mess you have today. Exceptions are for *programmer errors* (bugs). Domain failures are values.

## Verdict

### Q1: DI / Bootstrap

**Explicit composition root. No container. No framework.**

```python
# composition_root.py
class CompositionRoot:
    """Single place where the entire object graph is wired."""
    
    def __init__(self, config: AppConfig, db_path: Path):
        # Infrastructure layer
        self._connection_pool = ConnectionPool(db_path)
        self._uow_factory = UnitOfWorkFactory(self._connection_pool)
        
        # Repositories (stateless, receive connection from UoW per D40)
        self._block_repo = BlockRepository()
        self._graph_repo = GraphRepository()
        
        # Processors (registered explicitly)
        self._processor_registry: dict[str, Processor] = {
            "eq": EQProcessor(),
            "compressor": CompressorProcessor(),
            "ml_denoise": MLDenoiseProcessor(config.ml_model_path),
            # ... all 17+, right here, visible, greppable
        }
        
        # Handlers
        self._handlers: dict[type, Handler] = {
            ExecuteBlock: ExecuteBlockHandler(self._processor_registry, self._block_repo),
            RenameBlock: RenameBlockHandler(self._block_repo),
            StoreExecutionResult: StoreExecutionResultHandler(self._block_repo),
            # ...
        }
        
        # Pipeline (middleware stack per D44)
        self._pipeline = Pipeline(
            middleware=[
                CorrelationMiddleware(),
                ValidationMiddleware(),
                UnitOfWorkMiddleware(self._uow_factory),
            ],
            handlers=self._handlers,
        )
    
    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline
```

```python
# main_qt.py — Qt entry point
root = CompositionRoot(config, db_path)
app = QtApp(root.pipeline)  # Qt layer wraps pipeline, adds signal-based notifications

# main_cli.py — CLI entry point  
root = CompositionRoot(config, db_path)
cli = CliApp(root.pipeline)  # No Qt dependency anywhere in the graph
```

No magic. No decorators. No scanning. Every dependency is a constructor argument. Tests construct handlers directly with mocks — they don't need the composition root at all.

### Q2: Threading

**Pipeline is a synchronous function call, thread-agnostic. Background work uses a `ThreadPoolExecutor`. The only thread marshaling is at the Qt notification edge.**

```python
class Pipeline:
    def dispatch(self, command: Command) -> Result[Any, PipelineError]:
        """Runs on caller's thread. Always."""
        ctx = PipelineContext(command)
        return self._run_middleware(ctx)

class ExecuteBlockHandler:
    def handle(self, cmd: ExecuteBlock, uow: UnitOfWork) -> Result[Accepted, PipelineError]:
        token = CancellationToken()
        # Submit to thread pool — returns immediately
        self._executor.submit(self._run_execution, cmd.block_id, token)
        return Ok(Accepted(cancel_token=token))
    
    def _run_execution(self, block_id: BlockId, token: CancellationToken):
        """Runs on background thread."""
        try:
            result = self._processor.process(block_id, token)
            # Dispatch BACK through pipeline from background thread — this is fine,
            # pipeline is just a function call, UoW middleware gives us a fresh connection
            self._pipeline.dispatch(StoreExecutionResult(block_id, result))
        except CancelledError:
            pass  # D23: cancel before STORE = safe rollback
```

- `ThreadPoolExecutor` with `max_workers` sized to CPU cores (ML inference is CPU-bound)
- Per-block lock (D22) acquired inside the handler, not the pipeline
- `CancellationToken` is just an `threading.Event` wrapper — checked at processor breakpoints (D23)
- Qt notification: outermost middleware emits a Qt signal *only if Qt is present* (adapter pattern)

### Q3: Error Model

**`Result[T, E]` for all handler returns. Exceptions are bugs, not flow control. Four error categories.**

```python
@dataclass(frozen=True)
class PipelineError:
    code: str
    message: str
    correlation_id: str

class ValidationError(PipelineError): ...    # Bad input (missing name, invalid connection)
class DomainError(PipelineError): ...        # Business rule violation (cycle in graph, incompatible format)
class InfrastructureError(PipelineError): ... # DB failure, file I/O, OOM
class CancellationError(PipelineError): ...  # User cancelled execution

# Handler signature — no exceptions cross this boundary
class Handler(Protocol[TCmd, TResult]):
    def handle(self, cmd: TCmd, uow: UnitOfWork) -> Result[TResult, PipelineError]: ...
```

```python
# UoW middleware — the ONLY try/except in the system
class UnitOfWorkMiddleware:
    def __call__(self, ctx: PipelineContext, next: NextMiddleware) -> Result[Any, PipelineError]:
        uow = self._uow_factory.create()
        try:
            result = next(ctx, uow)
            if isinstance(result, Ok):
                uow.commit()
                uow.flush_events()  # D44
            else:
                uow.rollback()  # Error result → rollback
            return result
        except Exception as e:
            uow.rollback()
            return Err(InfrastructureError(
                code="UNEXPECTED",
                message=str(e),
                correlation_id=ctx.correlation_id,
            ))
```

Rollback decision is trivial: `Ok` → commit, `Err` → rollback, exception → rollback + wrap. No ambiguity. Handler authors return `Ok(value)` or `Err(SomePipelineError(...))`. They literally cannot get this wrong.

Background errors reach the UI through the same pipeline path: background thread dispatches `StoreExecutionResult`, if that fails the result is an `Err`, the notification middleware (outermost) picks it up and routes it to whatever UI adapter is registered (Qt signal, CLI print, MCP response).
