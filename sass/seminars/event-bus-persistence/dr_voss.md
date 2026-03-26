# Dr. Voss 🔬 — Response to Event Bus + Persistence Brief

## Position

The central flaw I see in the current trajectory is a failure to treat the event bus and persistence layer as a single transactional unit. They are being discussed as separate systems that "interact" — this is the wrong framing. Events *are* side effects of state transitions. If your transaction rolls back but events already fired, you have an inconsistent system. If events fire before commit, subscribers act on data that may never exist. This is not a theoretical concern — it is the single most common source of subtle, unreproducible bugs in event-driven architectures, and it will bite you hardest in the STORE phase where background threads commit results asynchronously.

The correct architecture is: commands execute within a Unit of Work that *collects* events. Events dispatch **only after successful commit**. No exceptions. No "fire during handler" convenience. This means every handler is pure with respect to external observation — it mutates state and declares intent (events), but the outside world sees nothing until the transaction succeeds. This is the Outbox Pattern adapted for an in-process system, and it is the only pattern that gives you deterministic, testable behavior. The re-entrancy question (Q1) dissolves entirely under this model: handlers cannot trigger cascading events during dispatch because dispatch hasn't happened yet. Events collected during commit-flush are queued and dispatched iteratively, never recursively.

Regarding the Qt decoupling: the event bus must be pure Python with a subscription contract that is a plain callable. The Qt adapter registers a subscriber that posts to Qt's event loop via `QMetaObject.invokeMethod` or a `QEvent`. This is a one-way boundary. The core never imports Qt. The adapter lives in the UI layer. This is non-negotiable if you want CLI and MCP to work, and it is trivially testable — your test subscriber is just a list that appends.

## Key Insight

**Everyone will focus on the dispatch model (sync vs async, threading) and miss the transaction-event boundary problem.** If you get dispatch wrong, you get performance issues. If you get the transaction-event boundary wrong, you get *data corruption*. Specifically: the STORE phase (D16) runs on a background thread after heavy computation. If STORE fires events before its transaction commits, and a subscriber (say, the UI) reads the database, it may see stale data (the write hasn't committed) or trigger a dependent command that races with the uncommitted write. The fix is simple but must be architectural law:

```python
class UnitOfWork:
    def __init__(self, connection):
        self._conn = connection
        self._pending_events: list[DomainEvent] = []

    def collect(self, event: DomainEvent):
        self._pending_events.append(event)

    def commit(self) -> list[DomainEvent]:
        self._conn.commit()
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events  # Caller dispatches AFTER this returns
```

No event leaves the unit of work until `commit()` succeeds. The pipeline middleware calls `event_bus.publish_all(uow.commit())`. This is the only safe ordering.

## Risk

If this perspective is ignored and events fire eagerly (during handler execution or before commit), you will encounter:

1. **Ghost events**: UI updates for operations that rolled back. User sees a block rename, but the rename failed — now the UI label and the database disagree. This is silent corruption.
2. **Re-entrant deadlocks**: Handler A fires event X, subscriber B handles X and dispatches command C, which tries to acquire the same per-block lock (D22) that handler A already holds. Deadlock. On a background thread. No stack trace in the UI. The app just freezes.
3. **Untestable handlers**: If handlers have side effects (event dispatch) interleaved with mutations, you cannot test the mutation in isolation. Every handler test becomes an integration test. Your test suite will be slow, brittle, and eventually abandoned.

These are not hypothetical. I have seen each one in production systems. The re-entrant deadlock in particular is nearly impossible to reproduce — it depends on subscription ordering and thread scheduling.

## Verdict

**Q1 — Event Bus Architecture:**
Typed event classes with a common `DomainEvent` base (frozen dataclass with `timestamp`, `correlation_id`, `source_component`). No unified "ObservableEvent" blob — you lose type safety and subscriber filtering becomes stringly-typed. Dispatch is synchronous on the *caller's* thread, but dispatch only happens post-commit (see above), so the caller is the pipeline middleware, not the handler. Re-entrancy: events published by event handlers during dispatch are queued and processed after the current batch completes (breadth-first, never depth-first). Qt adapter: a subscriber registered in the UI layer that marshals to Qt's event loop. Contract is `Callable[[DomainEvent], None]`. Core never knows Qt exists.

```python
@dataclass(frozen=True)
class DomainEvent:
    correlation_id: str
    timestamp: float
    source: str

@dataclass(frozen=True)
class BlockRenamed(DomainEvent):
    block_id: str
    old_name: str
    new_name: str
```

**Q2 — Persistence + Transaction Boundaries:**
Unit of Work middleware wraps every command automatically. Handlers do not manage transactions — they receive a `UnitOfWork` and call `uow.collect(event)` for side effects. The middleware calls `uow.commit()` and then dispatches collected events. Repositories are stateless — they receive a connection from the UoW. The STORE phase (D16) dispatches a `StoreResult` command through the pipeline like everything else. It does not manage its own transaction. No special paths. One pipeline, one transaction model.

```python
# Middleware pseudo-code
async def unit_of_work_middleware(command, next_handler):
    uow = UnitOfWork(get_connection())
    try:
        result = next_handler(command, uow)
        events = uow.commit()
        for event in events:
            event_bus.publish(event)
        return result
    except Exception:
        uow.rollback()
        raise
```

**Q3 — Schema & Migration:**
Versioned SQL scripts in a `migrations/` directory, numbered sequentially (`001_initial.sql`, `002_add_block_metadata.sql`). A `schema_version` table tracks current version. On open, run all unapplied migrations in order within a single transaction. No ORM migrations — SQLite's ALTER TABLE is limited and you need to control the SQL exactly. Read vs write connections: yes, use a separate read connection in WAL mode. WAL allows concurrent reads during writes. The write connection is exclusive to the pipeline (via UoW). Read connections are handed out freely for queries.

**Q4 — Integration:**
The middleware stack, in order:
1. **Logging/Correlation** — assigns `correlation_id` if absent, logs entry
2. **Validation** — calls `command.validate()`
3. **Lock acquisition** — per-block lock (D22) if command targets a block
4. **Unit of Work** — opens transaction, injects UoW into handler, commits on success, dispatches events post-commit
5. **Handler** — the actual command handler

Correlation ID flows by assignment at layer 1, passed into the UoW, stamped onto every event collected via `uow.collect()`. Subscribers receive it on the event. If a subscriber dispatches a new command, it passes the correlation ID forward as a parameter. This gives you a full causal chain for debugging without any implicit thread-local magic.

Queries skip layers 3-4 (no locks, no transactions needed for reads) and go directly to a read-only connection. This is the CQRS-lite split from D34.
