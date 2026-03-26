# Kai ⚡ — The Pragmatist

## Position

Stop overthinking DI. This is a single-user desktop app with ~20 handlers, not a microservices platform. You don't need a container framework, you don't need a service locator, and you definitely don't need auto-wiring. Write a `bootstrap.py` that constructs everything with explicit constructor calls. It'll be 80-120 lines. You can read it top to bottom and know exactly what depends on what. When a test needs a handler, you call `make_rename_handler(repo=fake_repo)` — done. The "it won't scale" objection is nonsense; you have 17 block types, not 170. If you hit 50 handlers someday and it hurts, *then* add a registry. Not before. The processor registration is the same story — a dictionary literal mapping block type enums to processor classes, built in bootstrap. Fifteen lines.

Threading: the pipeline is thread-agnostic. It's just a function call chain. The caller's thread runs it. UI thread calls `pipeline.send(RenameBlock(...))` → runs synchronously → done. For execution commands, the handler itself spawns the work onto a `ThreadPoolExecutor` (stdlib, no Qt dependency). The background task does its ML work, then calls `pipeline.send(StoreExecutionResult(...))` from the worker thread. The UoW middleware serializes writes through SQLite's single connection — that's your write lock, you already have it. Don't build a custom thread manager. `ThreadPoolExecutor(max_workers=4)` plus per-block `asyncio.Lock` (or `threading.Lock`) is the entire threading model. `CancellationToken` is passed as a field on the command — the handler passes it to the processor, processor checks it at breakpoints. That's it.

Errors: use exceptions for infrastructure failures (DB down, file missing) and a `Result[T]` return for domain outcomes (validation failed, block not found, execution cancelled). Middleware catches unhandled exceptions and wraps them into `Result.error(...)`. Handlers return `Result` explicitly for expected failures. This means UoW middleware does `try: result = next(pipeline) / except: rollback` — simple. For background execution errors, the worker catches exceptions and dispatches `StoreExecutionResult` with an error payload, which flows through the pipeline normally and emits an event that the UI layer can subscribe to.

## Key Insight

Everyone's going to debate container patterns for hours when the real risk is the threading boundary. The *only* hard problem here is: "background thread finishes ML work and needs to write results through the pipeline." If you get that handoff right — worker thread calls `pipeline.send(StoreExecutionResult(...))`, UoW serializes the write, event fires, UI picks it up via signal — everything else is trivial wiring. Spend 80% of your design time on that one interaction, not on DI abstractions.

## Risk

If you ignore the pragmatist take and build an elaborate DI container + thread orchestrator + Result monad hierarchy, you'll spend 3 weeks on infrastructure before a single block processes audio. Then you'll discover the threading model doesn't work under real load and have to rewrite it anyway — except now it's entangled with your container framework. Ship the simple version, stress-test it with actual ML inference, *then* refine.

## Verdict

**Q1 — DI/Bootstrap:** Module-level wiring. One `bootstrap.py` with explicit constructor calls. A plain `dict[type[Command], Callable]` for handler lookup. A plain `dict[BlockType, type[Processor]]` for processor registration. No framework, no decorators, no magic.

```python
# bootstrap.py
def create_app(db_path: str, qt: bool = True) -> App:
    connection = sqlite3.connect(db_path)
    uow_factory = lambda: UnitOfWork(connection)
    block_repo = BlockRepository()
    
    processors = {
        BlockType.DENOISE: DenoiseProcessor,
        BlockType.REVERB: ReverbProcessor,
        # ... 15 more, it's fine
    }
    
    handlers = {
        ExecuteBlock: ExecuteBlockHandler(block_repo, processors, executor),
        RenameBlock: RenameBlockHandler(block_repo),
        # ...
    }
    
    pipeline = Pipeline(middleware=[
        CorrelationMiddleware(),
        ValidationMiddleware(),
        UnitOfWorkMiddleware(uow_factory),
    ], handlers=handlers)
    
    return App(pipeline=pipeline)
```

**Q2 — Threading:** Pipeline is thread-agnostic — caller's thread runs it. Background work via `concurrent.futures.ThreadPoolExecutor(max_workers=4)`. Per-block `threading.Lock` for execution locks (D22). Background worker calls back through pipeline on worker thread. SQLite UoW serializes writes. `CancellationToken` is a command field, plumbed to processor. No custom thread infrastructure.

```python
class ExecuteBlockHandler:
    def handle(self, cmd: ExecuteBlock, uow: UnitOfWork) -> Result:
        lock = self._block_locks[cmd.block_id]  # per-block lock
        if not lock.acquire(blocking=False):
            return Result.error("Block already executing")
        
        def run():
            try:
                processor = self._processors[cmd.block_type]()
                output = processor.process(cmd.input_data, cmd.cancel_token)
                self._pipeline.send(StoreExecutionResult(
                    block_id=cmd.block_id, output=output
                ))
            except CancelledError:
                pass  # D23: cancel before STORE = safe rollback
            except Exception as e:
                self._pipeline.send(StoreExecutionResult(
                    block_id=cmd.block_id, error=str(e)
                ))
            finally:
                lock.release()
        
        self._executor.submit(run)
        return Result.ok("Execution started")
```

**Q3 — Errors:** Two-lane model. `Result[T]` for expected domain outcomes (returned by handlers). Exceptions for unexpected infrastructure failures (caught by middleware). UoW middleware wraps both: exception → rollback + `Result.error()`. Background errors flow through `StoreExecutionResult` with error payload → event emitted → UI notified.

```python
@dataclass(frozen=True)
class Result(Generic[T]):
    value: T | None = None
    error: str | None = None
    
    @property
    def ok(self) -> bool: return self.error is None
    
    @staticmethod
    def success(val: T) -> Result[T]: return Result(value=val)
    
    @staticmethod  
    def fail(msg: str) -> Result: return Result(error=msg)
```

No error taxonomy beyond that. No `ValidationError` vs `DomainError` vs `InfrastructureError` hierarchy. A string message and a boolean. Add categories later if you actually need to branch on error types — you probably won't.
