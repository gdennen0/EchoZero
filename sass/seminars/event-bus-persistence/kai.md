# Kai ⚡ — The Pragmatist

## Position

Stop building infrastructure and start building the app. Half of these questions are trying to solve problems we don't have yet. We have ~30 domain events, a single user, a single machine, and SQLite. This is not Kafka. This is not a distributed system. The event bus needs to do exactly one thing: decouple the core from Qt. That's it. A typed event base class, a dict of `{event_type: list[callable]}`, synchronous dispatch, done. We can make it async later if profiling tells us to — and it won't, because we're dispatching 30 event types to maybe 5 subscribers each on a desktop app.

For persistence, the Unit of Work middleware pattern is the obvious answer because D31 already decided commands are pure data with separate handlers. Wrap the handler call in a `with connection` block, collect events, commit, then dispatch events. This is ~40 lines of code. Don't make repositories stateless — that's a refactor that buys us nothing today. Repos holding a Database reference works. The STORE phase from D16 should just call the repository directly with its own connection; it's a background thread doing one atomic write, not a command flowing through the pipeline. Trying to route background execution results through the command pipeline is over-engineering that will bite us with deadlocks.

Migration strategy: a `schema_version` integer in a metadata table and a folder of numbered `.sql` files. `001_initial.sql`, `002_add_whatever.sql`. On open, run any scripts with version > current. This is solved technology. ORMs are overkill for a desktop app with maybe 15 tables. Read/write connection split? No. SQLite WAL mode handles concurrent reads fine. We're not running a web server with 500 concurrent requests. One connection per thread, WAL mode, move on.

## Key Insight

**The Qt adapter is the only hard problem here.** Everything else is vanilla Python that any mid-level dev could write in an afternoon. The real question nobody's asking is: what does the boundary between "core event dispatched on background thread" and "UI update on main thread" actually look like? That's where the bugs will live. My answer: the Qt adapter registers as a normal subscriber, but internally it posts a `QEvent` to marshal to the main thread. The core never knows Qt exists. One class, maybe 30 lines:

```python
class QtEventBridge:
    def __init__(self, app: QCoreApplication, bus: EventBus):
        self._app = app
        bus.subscribe_all(self._forward)
    
    def _forward(self, event: DomainEvent):
        QCoreApplication.postEvent(self._app, DomainEventWrapper(event))
```

That's the adapter. That's the whole thing. The rest is plumbing that writes itself.

## Risk

If you ignore the pragmatist: you'll spend 3 weeks building an event bus framework with async dispatch, re-entrancy guards, correlation ID propagation, and middleware chains — and then discover that the actual app behavior you need is "when user renames a block, update the sidebar." You'll have infrastructure nobody uses at full capacity, bugs in the infrastructure itself, and zero shipped features to show for it. The rewrite stalls because the foundation became the project.

## Verdict

- **Q1:** Typed event classes inheriting from a frozen `DomainEvent` base. Synchronous dispatch on publisher's thread. Re-entrancy: queue (finish current dispatch, then process new events). Simple FIFO. Qt adapter is a subscriber that marshals via `postEvent` — core never imports Qt.

- **Q2:** Unit of Work middleware wraps command handlers automatically. Events collected during handler execution, flushed after commit. Repos keep their Database reference — stateless repos are a refactor with no payoff. STORE phase (D16) manages its own transaction directly via repository, does NOT go through the command pipeline. It fires a `ComputationComplete` event after its commit so the UI can update.

- **Q3:** `schema_version` integer + numbered SQL scripts. No ORM. No read/write split. One connection per thread, WAL mode. Check version on open, run pending migrations. Ship it.

- **Q4:** Middleware stack, outermost to innermost: `Logging → Validation → UnitOfWork(begin txn → call handler → collect events → commit → dispatch events) → Handler`. Correlation ID: generate in the pipeline entry point, attach to the command context, handlers pass it to any events they create. Don't build a magic propagation system — explicit is better than implicit. That's Python 101.
