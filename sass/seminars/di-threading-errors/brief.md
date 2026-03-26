# S.A.S.S. Brief — DI/Bootstrap, Threading Model, Error Model

## Context

EchoZero is a desktop audio processing app (Python + Qt) being rewritten. Node-based architecture where blocks process audio and connect via a directed graph. Single user, single machine.

### Already Decided (relevant subset)
- **D5:** Non-blocking execution. Processors run on background thread, never touch Qt. UI stays responsive.
- **D10 (revised):** SQLite is runtime engine. `.ez` file is zip archive (SQLite DB + audio). Save = zip, Load = extract. SQLite doesn't persist between sessions.
- **D16:** Execution flow: TRIGGER → PULL (copy) → EXECUTE (background) → STORE (atomic) → NOTIFY
- **D22:** Per-block execution lock (not global). UI never blocked. Per-block lock allows concurrent independent block execution.
- **D23:** CancellationToken checked at processor breakpoints. Cancel before STORE = safe rollback.
- **D30:** Single pipeline, multiple adapters. One entry point for all operations.
- **D31:** Commands are frozen dataclasses. Handlers are separate classes with injected dependencies.
- **D34:** CQRS-lite — commands and queries through same pipeline, different middleware.
- **D39:** Unit of Work middleware wraps every command. Handlers receive UoW (connection + event collector).
- **D40:** Repositories are stateless — receive connection from UoW.
- **D41:** STORE phase dispatches `StoreExecutionResult` through pipeline.
- **D44:** Middleware stack: Correlation → Validation → UoW → Handler → commit → flush events.

### Legacy State
- `ServiceContainer` takes 30+ constructor params, wired in `bootstrap.py`
- `ApplicationFacade.__init__` does 30+ `getattr(services, ...)` calls
- Threading: background `QThread` workers for execution, `QTimer` for polling, `QCoreApplication.postEvent` for thread marshaling
- Error handling: try/except scattered everywhere, `Log.error()` calls, no consistent pattern. Some methods return `CommandResult` with `ResultStatus`, most just return booleans or None.

## Questions for the Panel

### Q1: Dependency Injection / App Bootstrap
How do handlers get their service dependencies? Options:
- **Constructor injection via registry:** Register handlers with their deps at startup. Pipeline looks up handler by command type.
- **Container/locator:** A service container that handlers query at runtime. (Current pattern, but cleaner.)
- **Factory functions:** Each handler is a function, not a class. A factory wires deps and returns the function.
- **Module-level wiring:** Just wire everything in a bootstrap module with explicit constructor calls. No framework.

How is the pipeline itself wired? What does app startup look like?

Constraints:
- Must work without Qt (CLI, MCP, tests)
- Must be explicit enough to understand without magic
- Must support both command handlers and query handlers
- Current codebase has 17+ block types, each needing a processor. How do processors get registered?

### Q2: Threading Model
The pipeline runs commands. Some commands are instant (rename block). Some trigger long background work (execute block → minutes of ML inference). How does threading work?

Key questions:
- Does the pipeline itself run on a specific thread, or is it thread-agnostic?
- When a command triggers background work (execution), who manages the thread?
- How does the background worker dispatch `StoreExecutionResult` back through the pipeline (D41)?
- Thread pool, single worker thread, or per-block threads?
- How does `CancellationToken` (D23) get plumbed?
- What about CommandSequencer and AudioPlayer, which need the main thread (Qt)?

Constraints:
- UI must never block (D5)
- Per-block execution lock, not global (D22)
- Background workers never touch Qt (D5)
- Pipeline must handle commands from any thread (UI thread, background thread, MCP thread)
- SQLite single-writer limitation (one write connection via UoW)

### Q3: Error Model
How do errors flow through the system? Currently it's a mess of try/except, Log.error(), boolean returns, and CommandResult objects.

Key questions:
- Error taxonomy: what categories of errors exist? (validation, domain, infrastructure, cancellation?)
- What does a handler return? `Result[T]` monad? Exception? Both?
- How does the pipeline middleware handle errors? (catch and wrap? let propagate? return Result?)
- How do errors in background execution get reported to the user?
- How does the UoW middleware know to rollback? (exception vs error result?)

Constraints:
- Errors must reach the user (not swallowed silently)
- Must be testable without Qt
- Must work across threads (background execution errors → UI notification)
- Must be simple enough that handler authors don't screw it up

## Constraints (repeated for emphasis)
- Python desktop app, single user, single machine
- Must work without Qt (CLI, MCP, testing)
- 17+ block processor types to register
- Background execution threads must not block UI
- SQLite single-writer limitation
- This is a rewrite — no backwards compatibility required
