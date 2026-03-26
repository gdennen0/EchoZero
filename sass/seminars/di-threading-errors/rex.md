# Rex 🔥 — Devil's Advocate

## Position

Everyone's going to overcook this. I can already smell the architecture astronautics. You have a **single-user desktop app** with 17 block types. Not a distributed system. Not a microservice. A desktop app that processes audio. The brief lists 15+ "already decided" architectural decisions and we haven't even written the interesting code yet. The danger here isn't picking the wrong DI pattern — it's spending so long on plumbing that the app never ships.

On DI: you don't need a container. You don't need a registry. You need a `bootstrap.py` that calls constructors. "But what about testability?" Mock the constructor args. "But what about 17 processors?" That's a dict literal — `{BlockType.REVERB: ReverbProcessor, ...}`. Seventeen lines. The current codebase has a `ServiceContainer` with 30+ params and everyone agrees it's bad, so the instinct will be to build a *better* container. Wrong instinct. The problem wasn't that the container was bad — the problem was that a container existed at all for an app this size. On threading: the pipeline should be thread-agnostic. It's just function calls. The *caller* decides which thread to run on. Background execution? Submit to a `ThreadPoolExecutor`. That's it. Stop making threading a pipeline concern. The `StoreExecutionResult` dispatch from a background thread is the only genuinely tricky part, and the answer is: the background task calls `pipeline.send(StoreExecutionResult(...))` directly — the pipeline doesn't care what thread it's on, SQLite writes are serialized by the UoW's connection lock, done.

On errors: `Result[T]` monads in Python are a trap. They look elegant in blog posts and become a tax on every handler author. Python has exceptions. Use them. Define a small hierarchy — `ValidationError`, `DomainError`, `CancellationError` — and let middleware catch them. The UoW middleware does `try: handler(); commit()` / `except: rollback(); raise`. Errors cross thread boundaries via `Future.result()` which re-raises. This is *solved infrastructure*. The only thing you need to build is the UI notification bridge: background thread → `Future` → result callback on main thread → show error. That's a `QMetaObject.invokeMethod` or a signal. Four lines.

## Key Insight

The brief asks three separate questions but they have **one answer**: keep the pipeline dumb. The pipeline is a for-loop over middleware. It doesn't know about threads. It doesn't know about DI. It doesn't know about errors. Middleware knows about errors (catch + rollback). The caller knows about threads (submit to executor or call directly). Bootstrap knows about DI (explicit wiring). The moment you make the pipeline smart about any of these things, you've coupled all three concerns together and created a monster. The legacy `ServiceContainer` with 30 params *is* what happens when you centralize everything.

## Risk

If you ignore this and build a sophisticated DI container + thread-aware pipeline + Result monad, you'll end up with:
1. A bootstrap sequence nobody can debug because resolution happens at runtime
2. Threading bugs because the pipeline "helps" by dispatching to threads, and now you have two things managing threads
3. Handler authors wrapping every return in `Ok()` / `Err()` and silently swallowing errors when they forget to unwrap
4. 2000 lines of infrastructure before the first audio sample gets processed
5. Another rewrite in 18 months because the architecture is too rigid

You've already decided on CQRS-lite, a middleware pipeline, UoW, repositories, commands as frozen dataclasses, per-block locks, cancellation tokens... that's a LOT of pattern for a desktop app. Every additional abstraction layer needs to justify its existence against "just call the function."

## Verdict

**Q1 (DI/Bootstrap):** Module-level wiring. A single `bootstrap.py` that constructs everything with explicit `__init__` calls. Processor registry is a plain dict. No container, no framework, no `@inject` decorators. If you can't read `bootstrap.py` top-to-bottom and understand every dependency, you've failed. Test by passing mocks to the same constructors.

```python
# bootstrap.py — the whole thing
def create_app(db_path: Path, qt: bool = True) -> App:
    conn = sqlite3.connect(db_path)
    uow = UnitOfWork(conn)
    block_repo = BlockRepository()
    processors = {
        BlockType.REVERB: ReverbProcessor(),
        BlockType.ML_DENOISE: MLDenoiseProcessor(model_path=MODELS / "denoise"),
        # ... 15 more, each one line
    }
    executor = ThreadPoolExecutor(max_workers=4)
    pipeline = Pipeline([
        CorrelationMiddleware(),
        ValidationMiddleware(),
        UoWMiddleware(uow),
    ])
    handlers = {
        ExecuteBlock: ExecuteBlockHandler(block_repo, processors, executor),
        RenameBlock: RenameBlockHandler(block_repo),
        # ...
    }
    pipeline.set_handlers(handlers)
    return App(pipeline=pipeline, executor=executor)
```

**Q2 (Threading):** Pipeline is thread-agnostic. `ThreadPoolExecutor` for background work, managed by the handler that needs it (not the pipeline). `StoreExecutionResult` dispatched by calling `pipeline.send()` from the background thread. SQLite writes serialize naturally through UoW connection lock. Cancel via `CancellationToken` passed into the processor — it's just a `threading.Event`. For Qt-dependent things (CommandSequencer, AudioPlayer): those live in the UI layer and are *not* pipeline handlers. They're Qt objects that *call* the pipeline, not the other way around.

```python
class ExecuteBlockHandler:
    def handle(self, cmd: ExecuteBlock, uow: UnitOfWork) -> None:
        block = self.block_repo.get(uow.conn, cmd.block_id)
        token = CancellationToken()
        self.active_tokens[cmd.block_id] = token
        future = self.executor.submit(self._run, block, token, self.pipeline)
        # Future completion notifies UI via callback

    def _run(self, block, token, pipeline):
        processor = self.processors[block.type]
        result = processor.execute(block, token)  # long-running, background thread
        pipeline.send(StoreExecutionResult(block.id, result))  # back through pipeline
```

**Q3 (Error Model):** Exceptions, not Result types. Three categories: `ValidationError` (bad input, caught by validation middleware before handler), `DomainError` (handler logic failures), `CancellationError` (token was set). Infrastructure errors (SQLite, file I/O) propagate as-is. UoW middleware catches all exceptions → rollback. Background errors surface via `Future.result()` re-raising. UI layer wraps `Future` callbacks to show errors. Handler authors just `raise DomainError("block has no input connections")` — zero ceremony.

```python
# UoW middleware — the entire error strategy
class UoWMiddleware:
    def __call__(self, command, next_handler):
        uow = self.uow_factory()
        try:
            result = next_handler(command, uow)
            uow.commit()
            uow.flush_events(self.pipeline)
            return result
        except Exception:
            uow.rollback()
            raise
```

Stop building frameworks. Wire constructors. Raise exceptions. Submit to a thread pool. Ship the app.
