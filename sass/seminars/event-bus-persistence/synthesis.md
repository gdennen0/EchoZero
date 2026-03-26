# S.A.S.S. Synthesis — Event Bus + Persistence Architecture

**Panel:** Maya 🏗️, Kai ⚡, Dr. Voss 🔬, Rex 🔥, Sage 🧮  
**Date:** 2026-03-01

---

## Consensus (All 5 Agree)

These are unanimous — no debate needed:

1. **Typed event classes with a common base.** Frozen dataclasses. `BlockRenamed`, `ConnectionCreated`, etc. No stringly-typed universal blob. The base carries `timestamp`, `correlation_id`, and `source`.

2. **Synchronous dispatch on publisher's thread.** No async. No thread pools. Desktop app, ~30 event types, handful of subscribers. Sync is correct.

3. **Re-entrancy: queue, never recurse.** If a handler publishes an event during dispatch, it goes into a buffer. Dispatch breadth-first after the current batch completes. Prevents stack overflow and makes ordering predictable.

4. **Qt adapter is a subscriber in the UI layer. Core never imports Qt.** One class, ~30-50 lines, marshals events to the main thread via `QCoreApplication.postEvent` or `QTimer.singleShot`. The bus doesn't know Qt exists.

5. **Events fire AFTER transaction commits. Never before. No exceptions.** This is the strongest consensus. Maya and Dr. Voss independently called it the single most important architectural decision. Sage called pre-commit events "events that lie." Even Rex agrees.

6. **Versioned SQL scripts, no ORM.** `migrations/` directory, numbered files, `schema_version` table. Run unapplied scripts on startup.

7. **Correlation ID flows from command to events.** Generated at pipeline entry, stamped on command context, handlers copy it to events. No magic propagation — explicit assignment.

---

## Tensions

### T1: Unit of Work Middleware vs Handler-Managed Transactions

**For UoW middleware** (Maya, Voss, Sage): Handlers never call commit/rollback. The middleware wraps every command in a transaction, collects events, commits on success, discards on failure. Guarantees consistency. One pattern everywhere.

**Against** (Rex, partially Kai): "You have one user and one writer. `BEGIN`/`COMMIT` inside a handler is fine. UoW is ceremony for a desktop app."

**Resolution:** UoW middleware wins. The cost is ~40 lines of code. The benefit is that the "events after commit" rule (which everyone agrees on) is *enforced architecturally* rather than by developer discipline. Rex is right that this is a simple app — but the UoW is also simple, and it eliminates an entire class of bugs by construction. The alternative is trusting every future handler author to remember to commit before publishing events. That's a policy; the UoW is a mechanism.

### T2: Stateless Repos vs Repos Owning Database

**Stateless** (Maya, Voss, Sage, Rex): Repos receive a connection from the UoW. No internal locks. Pure data access helpers.

**Keep Database ref** (Kai): "Stateless repos are a refactor with no payoff."

**Resolution:** Stateless repos. This falls directly out of T1 — if the UoW owns the transaction, repos must receive the connection from outside. They can't commit because they don't own the transaction. Kai's concern about payoff is fair, but the payoff IS the UoW pattern working correctly. It's not a separate decision.

### T3: STORE Phase — Pipeline or Own Transaction?

**Through pipeline** (Maya, Voss, Sage, Rex): STORE dispatches a `StoreExecutionResult` command. One path for all writes.

**Own transaction** (Kai): "Routing background execution through the command pipeline is over-engineering that will bite us with deadlocks."

**Resolution:** Through the pipeline. Kai raises a real concern about deadlocks, but the risk is manageable: the background worker does computation, then dispatches a command to store results. The command goes through the pipeline on the calling thread (which is the background thread — that's fine, the pipeline isn't thread-restricted, just the Qt adapter is). The per-block lock (D22) is already held by the execution, so the `StoreExecutionResult` handler just writes. No contention. Having two write paths (pipeline + direct STORE) means two sets of bugs, two transaction models, two event emission patterns. Not worth it.

### T4: Read/Write Connection Split

**Yes** (Maya, Voss, Sage): Separate read connection in WAL mode. Writes go through UoW. Reads can happen during long writes.

**No** (Kai, Rex): "You're not running a web server. One connection, WAL mode, done."

**Resolution:** Start with one connection, add read connection if needed. Rex and Kai are right that this is premature for a desktop app. WAL mode already allows concurrent reads from the same connection. If profiling shows UI reads blocking during long execution writes, add a second read connection. Design repos so this is easy (they already receive connections from outside), but don't build it day one.

### T5: Event Base Class Fields

**Minimal** (Maya, Voss, Kai, Rex): `timestamp`, `correlation_id`, `source`. That's it.

**Full observable** (Sage): Add `layer`, `component` from the observability design.

**Resolution:** Minimal base. The observability `layer` and `component` can be derived from the event class itself (its module path, its type hierarchy) rather than stored as string fields. If the MCP adapter needs layer/component info, it can derive it from the event type. Don't store what you can compute.

---

## Decisions

### D35: Typed event classes with minimal DomainEvent base
Frozen dataclasses. Base carries `timestamp: float`, `correlation_id: str`, `source: str`. Each domain event (BlockAdded, ConnectionCreated, etc.) inherits from this. No universal ObservableEvent blob. Observability metadata (layer, component) derived from event type, not stored.

### D36: Collect-then-publish — events staged during handler, flushed after commit
The UoW collects events via `uow.collect(event)`. After successful commit, events are published to the bus. On rollback, events are discarded. No event escapes the transaction boundary.

### D37: Re-entrant publish uses breadth-first queue
Events published by handlers during dispatch are queued and processed after the current batch completes. Never recursive. Predictable ordering.

### D38: Qt adapter is one file in the UI layer — core never imports Qt
A subscriber that marshals events to the main thread via `QCoreApplication.postEvent`. ~30-50 lines. The event bus contract is `Callable[[DomainEvent], None]`. No Qt types in the core.

### D39: Unit of Work middleware wraps every command
Handlers receive a UoW context (connection + event collector). They never call commit/rollback. The middleware begins the transaction, calls the handler, commits on success (flushing events), rolls back on failure (discarding events). ~40 lines of infrastructure that eliminates an entire class of consistency bugs.

### D40: Repositories are stateless — receive connection from UoW
Repos are pure data access helpers. `save(conn, entity)`, `get(conn, id)`. No Database reference. No internal locks. Concurrency is the pipeline's job.

### D41: STORE phase dispatches through the pipeline
Background execution completes computation, then dispatches a `StoreExecutionResult` command through the pipeline. Same transaction model, same event emission, same middleware. No shadow write path.

### D42: Single connection to start, read connection added if needed
WAL mode handles concurrent reads. Start with one write connection owned by the UoW. If profiling shows read contention during long writes, add a separate read connection for queries. Repos already support this (they receive connections from outside).

### D43: Schema migrations via versioned SQL scripts
`migrations/` directory with numbered `.sql` files. `schema_version` table tracks current version. On startup, run unapplied scripts in order. No ORM. SQLite ALTER TABLE is limited — control the SQL exactly.

### D44: Middleware stack ordering
```
Command → Correlation (assign ID) → Validation → UoW (begin → handler → commit → flush events) → Result
Query → Handler (own read connection) → Result
```
Queries skip validation, UoW, and undo middleware. Direct to handler with a read connection.

---

## Open Questions (Need Human Input)

1. **Event base `source` field** — is this the component name (e.g., `"block_service"`) or the handler class name? Or do we skip it entirely and rely on event type for provenance?

2. **Migration transactionality** — run all pending migrations in one transaction (atomic upgrade), or each migration in its own transaction (partial upgrade on failure)? Single transaction is safer but means one bad migration rolls back everything.
