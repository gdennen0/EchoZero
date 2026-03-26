# Maya 🏗️ — The Architect

## Position

The fundamental architectural question hiding behind all four Qs is **dependency direction**. Every decision here must obey one rule: the core domain (event bus, commands, repositories) depends on *nothing* — not Qt, not SQLite, not threads. Adapters depend inward. This isn't abstract purity; it's what makes CLI mode, MCP, and testing possible without mocks-upon-mocks. The event bus is the spine of the system. If you get its boundaries wrong, every layer above it inherits the coupling. The legacy EventBus failed precisely because it married dispatch semantics to Qt's thread marshaling. Don't repeat that.

The persistence layer has a similar trap. SQLite is the real database now (D10), but that doesn't mean repositories should *know* they're talking to SQLite. The repository interface is a domain contract. SQLite is an implementation detail behind it. The moment you let `BaseRepository` own a `Database` reference and a lock, you've welded your domain to your storage engine. Instead, transactions should be a cross-cutting concern managed by middleware — a Unit of Work that wraps command execution, collects events, commits, *then* dispatches. This is the only ordering that guarantees consistency without distributed transaction nightmares.

The middleware pipeline is where everything composes. Think of it as an onion: correlation context → validation → Unit of Work (begin txn → execute handler → collect events → commit → flush events) → logging. Events dispatched *after* commit. Always. No exceptions. Pre-commit event dispatch is a consistency hole that will bite you exactly once before you rewrite it anyway.

## Key Insight

**The event bus and the Unit of Work are the same architectural boundary.** Everyone will treat them as separate systems. They're not. The UoW collects domain events during command handling, and the event bus dispatches them after the UoW commits. If you design them independently, you'll end up with two competing sources of "when things happen" — handlers firing events immediately (pre-commit, possibly rolled back) and the UoW flushing events post-commit. Pick one. The UoW *is* the event collection mechanism; the bus *is* the post-commit dispatch mechanism. One seam, two roles.

```python
class UnitOfWork:
    def __init__(self, conn_factory, event_bus):
        self._conn_factory = conn_factory
        self._event_bus = event_bus
        self._events: list[DomainEvent] = []

    def __enter__(self):
        self._conn = self._conn_factory()
        self._conn.execute("BEGIN")
        return self

    def collect(self, event: DomainEvent):
        self._events.append(event)

    def __exit__(self, exc_type, *_):
        if exc_type:
            self._conn.rollback()
            self._events.clear()
        else:
            self._conn.commit()
            for event in self._events:
                self._event_bus.dispatch(event)
            self._events.clear()
```

Handlers receive the UoW, call `uow.collect(event)`. Events only dispatch after successful commit. No leaks, no inconsistency.

## Risk

If you ignore the dependency direction and let the event bus couple to dispatch mechanisms (Qt signals, thread pools, whatever), you'll end up with a second rewrite in 6 months. The legacy system already proved this. Worse: if events fire pre-commit, you'll get phantom notifications — UI updates for data that was rolled back, subscribers acting on state that doesn't exist. In a single-user desktop app this manifests as subtle bugs ("why does the block show the old name after undo?") that are agonizing to trace because the event *did* fire, the handler *did* run, the data just isn't there anymore.

## Verdict

**Q1 — Event Bus Architecture:**
Typed event classes inheriting from a minimal `DomainEvent` base (with `timestamp`, `correlation_id`, `source`). No `ObservableEvent` mixin on components — that's coupling the emitter to the bus. Components return events; they don't dispatch them. Dispatch is synchronous on the caller's thread by default. A Qt adapter subscribes to domain events and posts to the Qt event loop — the adapter depends on the bus, not vice versa. Re-entrancy: queue. Events raised during dispatch go into a buffer and are dispatched after the current batch completes.

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

# Qt adapter (lives in UI layer, depends inward)
class QtEventBridge:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe(BlockRenamed, self._on_block_renamed)

    def _on_block_renamed(self, event: BlockRenamed):
        QCoreApplication.postEvent(target, BlockRenamedQEvent(event))
```

**Q2 — Persistence + Transaction Boundaries:**
Unit of Work middleware wraps every command. Handlers do NOT manage their own transactions. Repos are stateless — they receive a connection from the UoW. Events are collected during handling, flushed after commit (see Key Insight above). The STORE phase (D16) dispatches a command through the pipeline like everything else — no special path. It's just a `StoreExecutionResult` command that goes through the same UoW middleware. One path. One set of rules.

```python
class BlockRepository:
    """Stateless. Receives connection, returns data."""
    def get(self, conn: Connection, block_id: str) -> Block: ...
    def save(self, conn: Connection, block: Block) -> None: ...

# Middleware wires it:
def uow_middleware(command, next_handler, uow_factory):
    with uow_factory() as uow:
        result = next_handler(command, uow)
    return result  # events already flushed by UoW.__exit__
```

**Q3 — Schema & Migration:**
Versioned SQL scripts in a `migrations/` directory. `001_initial.sql`, `002_add_xyz.sql`. A `schema_version` table tracks applied migrations. No ORM migrations — you're using SQLite directly, keep it honest. Read vs write connections: yes, use a separate read connection in WAL mode. Writes go through the UoW's connection (single-writer). Reads can happen concurrently on a second connection. Two connections, one writer, one reader. Simple.

**Q4 — Integration / Middleware Stack:**

```
Command enters pipeline
  → CorrelationMiddleware (assign/propagate correlation_id)
    → ValidationMiddleware (command.validate())
      → UnitOfWorkMiddleware (BEGIN → hand conn to handler)
        → Handler executes, uses repos via conn, calls uow.collect(events)
      → COMMIT → dispatch collected events via EventBus
    → LoggingMiddleware (timing, errors)
```

Correlation ID is set once at the top, stamped onto the command context, and passed into every `DomainEvent` created during handling. Subscribers receive it on the event. If a subscriber triggers a *new* command, that command gets a *new* correlation ID with a `parent_correlation_id` link. Traceability without coupling.
