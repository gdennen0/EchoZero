# Sage 🧮 — Data Modeler Response

## Position

The core data modeling question here is **who owns the transaction boundary**, and the answer is unambiguous: the pipeline middleware does. Not the command handler, not the repository, not the event bus. A Unit of Work middleware wraps each command in a single transaction, passes the connection to repositories, and collects domain events. Events flush *after* commit. This is non-negotiable from a data integrity standpoint — if you let handlers manage their own transactions, you get partial writes, orphaned state, and events that announce things that never actually persisted. The command handler should see a connection, do its work, and never call `commit()`.

For the event bus, the data model question is: what is an event's identity? I say typed event classes with a shared `ObservableEvent` base that carries the universal schema fields (correlation_id, timestamp, layer, component). The typed classes are your schema — they define what payload shapes are legal. The base class is your wire format. Don't make developers choose; the typed class IS the observable event. Dispatch must be synchronous-then-queue: handlers on the publishing thread run synchronously, but a Qt adapter re-posts to the main thread via a bridge. Re-entrant events get queued, not dispatched immediately — immediate dispatch is a stack overflow waiting to happen and makes reasoning about state transitions impossible.

The STORE phase from D16 is the interesting edge case. It runs on a background thread after heavy computation, and it **must** dispatch a command through the pipeline — not manage its own transaction. Why? Because the pipeline is your single source of truth for "what happened." If the STORE phase silently writes to SQLite outside the pipeline, you've created a shadow write path. Every write goes through the pipeline. No exceptions. The command it dispatches (something like `StoreExecutionResult`) goes through the same middleware, gets the same transaction wrapping, fires the same events.

## Key Insight

**Repositories must be stateless — they receive a connection, they don't own one.** The current `BaseRepository` pattern where repos hold a `Database` reference and manage their own lock is a data ownership anti-pattern. It means every repository is its own transaction boundary, which means cross-repository consistency is impossible without external coordination (which is exactly what a UoW solves). Make repositories pure functions of `(connection, args) → result`. The UoW middleware creates the connection, passes it through, and commits. Repositories become stateless projections over a connection they don't control.

```python
# WRONG: Repository owns connection
class BlockRepository:
    def __init__(self, db: Database):
        self._db = db  # Who controls this? Who commits?
    
    def save(self, block: Block):
        with self._db.lock:
            self._db.execute("INSERT ...", block)
            self._db.commit()  # Commits HERE? What about the other repo?

# RIGHT: Repository receives connection
class BlockRepository:
    def save(self, conn: sqlite3.Connection, block: Block):
        conn.execute("INSERT INTO blocks ...", block.to_row())
        # No commit. UoW decides when.

# Middleware
class UnitOfWorkMiddleware:
    def __call__(self, command, next_handler):
        conn = self._db.connect()
        try:
            ctx = CommandContext(conn=conn, events=[])
            result = next_handler(command, ctx)
            conn.commit()
            for event in ctx.events:
                self._event_bus.publish(event)  # AFTER commit
            return result
        except Exception:
            conn.rollback()
            raise
```

## Risk

If you ignore stateless repositories and post-commit event dispatch, you will build a system where **events lie**. An event says "block renamed" but the transaction rolled back. A subscriber updates the UI based on an event whose underlying data doesn't exist yet. The MCP server projects stale state because it received an event before the write committed. In a single-user desktop app, these bugs are subtle — they manifest as "sometimes the UI shows the wrong thing after undo" or "the .ez export is missing a block that was definitely there." You'll chase these for months because the data model doesn't enforce consistency — it just hopes for it.

## Verdict

**Q1 — Event Bus Architecture:**
Typed event classes inheriting from `ObservableEvent` base. Synchronous dispatch on publisher's thread. Re-entrant events are queued (breadth-first, not depth-first). Qt adapter subscribes to `ObservableEvent` base and re-posts via `QMetaObject.invokeMethod` with `Qt.QueuedConnection` — this is the only Qt-touching code, lives in the adapter layer, and the core never imports Qt.

**Q2 — Persistence + Transaction Boundaries:**
Unit of Work middleware wraps every command. Events are collected during handler execution and flushed *after* successful commit. Repositories are stateless — receive `Connection` from the UoW context, never call commit/rollback. The STORE phase dispatches a `StoreExecutionResult` command through the pipeline like everything else.

**Q3 — Schema & Migration:**
Versioned SQL scripts in a `migrations/` directory, applied in order. Store a `schema_version` in a `_meta` table. No ORM migrations — you're using SQLite directly, keep it honest. One read connection (WAL mode, read-uncommitted) and one write connection. Read connection is cheap insurance against long writes blocking UI queries.

**Q4 — Integration:**
Middleware stack order: `Logging → Validation → UnitOfWork → Handler`. Correlation ID is born when the command enters the pipeline (generated by the first middleware), stamped on the `CommandContext`, and every event created during that handler execution inherits it from the context. The event bus doesn't generate correlation IDs — it receives them. This means you can trace `Command(correlation_id=X) → Event(correlation_id=X) → Event(correlation_id=X)` trivially.

```python
# Full flow
pipeline.execute(RenameBlock(block_id="b1", new_name="Reverb"))
# → LoggingMiddleware: assigns correlation_id, logs entry
# → ValidationMiddleware: RenameBlock.validate() 
# → UnitOfWorkMiddleware: opens connection, creates CommandContext
#   → RenameBlockHandler: uses ctx.conn, appends BlockRenamed to ctx.events
# → UnitOfWorkMiddleware: COMMIT, then publishes BlockRenamed events
# → Qt adapter receives BlockRenamed, posts to main thread, UI updates
```
