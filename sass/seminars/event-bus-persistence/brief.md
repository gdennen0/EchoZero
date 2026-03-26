# S.A.S.S. Brief — Event Bus + Persistence Architecture

## Context

EchoZero is a desktop audio processing app (Python + Qt) being rewritten from scratch. It uses a node-based architecture where blocks process audio data and connect via a directed graph.

### Already Decided
- **D10:** SQLite is the real database (not a cache). `.ez` file is an export format.
- **D16:** Execution flow: TRIGGER → PULL (copy) → EXECUTE (background) → STORE (atomic) → NOTIFY
- **D22:** Per-block execution lock (not global). UI never blocked.
- **D24:** System invariants including transactional project load, atomic rename propagation
- **D30:** Single pipeline, multiple adapters (no facade). One entry point for all operations.
- **D31:** Commands are pure data (frozen dataclasses). Self-describe validation + undo. Handlers separate.
- **D32:** Two categories: Editable (undoable) and Operational (not undoable, with warning)
- **D33:** Global undo stack, editable commands only, pure Python (not Qt-coupled)
- **D34:** CQRS-lite — commands and queries through same pipeline, different middleware paths

### Legacy State
- EventBus: Thread-safe pub/sub, ~30 typed DomainEvent subclasses, coupled to Qt (QEvent/QCoreApplication.postEvent for thread marshaling). Works but can't exist without Qt.
- Persistence: SQLite as session cache (cleared on load), JSON .ez files are truth. BaseRepository pattern with generic CRUD. Database uses threading.RLock, WAL mode.
- CommandBus: Coupled to QUndoStack/QUndoCommand. Commands take facade as first arg.

### Observability Design (already proposed, not yet decided)
- Universal event schema: layer, component, event_type, payload, timestamp, correlation_id
- Observable mixin for any emitting component
- MCP server as thin event bus projection

## Questions for the Panel

### Q1: Event Bus Architecture
- Event typing: typed event classes per domain event, unified ObservableEvent, or both (typed classes with observable base)?
- Dispatch model: synchronous on publisher's thread? Async? 
- Re-entrancy: if a handler publishes another event during dispatch, what happens? Queue? Immediate? Error?
- How does the Qt UI subscribe without coupling the core? What does the adapter contract look like?

### Q2: Persistence + Transaction Boundaries
- Should command handlers manage their own transactions, or should a Unit of Work middleware wrap every command automatically?
- How do events relate to transactions? (Events fire before commit? After? Collected and flushed?)
- Repository pattern: keep current BaseRepository (repos own a Database reference + lock) or make repos stateless (receive connection from outside)?
- How does the execution STORE phase (D16) fit? It runs on a background thread after heavy computation. Does it dispatch a command through the pipeline, or manage its own transaction?

### Q3: Schema & Migration
- SQLite is now the real DB (D10). What's the migration strategy? Versioned SQL scripts? ORM migrations? 
- Read vs write connections — should queries use separate connections for concurrency during long writes?

### Q4: Integration
- How do the event bus and persistence interact in the pipeline middleware chain? What's the ordering?
- How does correlation_id flow through a command → handler → events → subscribers chain?
- What does the full middleware stack look like end-to-end?

## Constraints
- Python desktop app, single user, single machine
- Must work without Qt (for CLI, MCP, testing)
- SQLite single-writer limitation
- Background execution threads must not block UI
- Near-zero overhead when no subscribers exist
- This is a rewrite — no backwards compatibility required
