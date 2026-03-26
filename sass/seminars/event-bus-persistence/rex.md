# Rex 🔥 — Devil's Advocate

## Position

Have you considered that you're over-engineering a single-user desktop app? You've got ~30 domain events, one human clicking buttons, and SQLite on a local disk. Yet the brief reads like you're designing a distributed microservices platform. Correlation IDs? CQRS-lite? Universal event schemas with layers and components? This is an audio editor, not a stock exchange. The most dangerous thing in this brief isn't any technical choice — it's the sheer *volume* of abstraction being proposed before a single line of new code exists. You're pre-solving problems you haven't hit yet, and every abstraction you add now is one you'll fight later when reality disagrees with your whiteboard.

The event bus question is particularly telling. You're asking about re-entrancy policies, async dispatch models, adapter contracts — all before you know what the actual event flow looks like in practice. The legacy system has ~30 events and it *works*. The real problem is Qt coupling, which is a 50-line adapter problem, not an architecture problem. Same with persistence: you have one writer, one reader (the same user!), and SQLite handles transactions perfectly well with `BEGIN`/`COMMIT`. Why are we debating Unit of Work middleware when `connection.execute()` inside a `with` block does the job?

The one thing I'll defend in this brief is D31 (commands as pure data) and D16 (explicit execution flow). Those are genuinely good constraints. But the temptation to drape those clean decisions in layers of middleware, observable mixins, and "universal schemas" will kill them. Keep the good decisions. Kill the ceremony around them.

## Key Insight

**The Qt decoupling problem is a bridge, not an architecture.** Everyone will spend 90% of their energy designing the "clean" event bus and 10% on the actual bridge to Qt. In reality, you need: (1) a plain Python pub/sub — literally a `dict[type, list[callable]]` — and (2) a Qt adapter that does `QMetaObject.invokeMethod` or `QTimer.singleShot(0, callback)` to marshal onto the main thread. That's it. That's the whole thing. Everything else is premature abstraction dressed up as "good architecture."

```python
# This is your entire event bus
class EventBus:
    def __init__(self):
        self._handlers: dict[type, list[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: type, handler: Callable):
        self._handlers[event_type].append(handler)
    
    def publish(self, event):
        for handler in self._handlers[type(event)]:
            handler(event)

# This is your entire Qt adapter
class QtEventBridge:
    def __init__(self, bus: EventBus, qt_app):
        self._app = qt_app
    
    def subscribe_on_main_thread(self, event_type, handler):
        def marshal(event):
            QTimer.singleShot(0, lambda: handler(event))
        bus.subscribe(event_type, marshal)
```

Add complexity *when you hit a wall*, not before.

## Risk

If you ignore the "keep it simple" perspective, you'll spend 2-3 months building infrastructure (middleware chains, observable mixins, migration frameworks, correlation ID propagation) and 0 months shipping audio features. Then when you actually try to wire up real audio processing, you'll discover the abstractions don't fit and you'll hack around them — ending up with something *more* complex than if you'd started simple. I've seen this movie. The rewrite of the rewrite is always worse than the original.

Specific risk: the "universal event schema" with `layer, component, event_type, payload` is a stringly-typed system hiding behind structure. You'll lose all the type safety of your current 30 typed event classes and gain... what? The ability to log events uniformly? Use `logging.info(f"{event}")` and add `__str__` to your dataclasses.

## Verdict

**Q1 — Event Bus:** Typed event classes (you already have them), synchronous dispatch on publisher's thread, re-entrancy via simple queue (collect during dispatch, flush after). Qt adapter is a separate subscriber that marshals to main thread. No "universal schema." No "observable mixin." **Do the 50-line version first.** Add complexity only when a concrete use case demands it.

**Q2 — Persistence:** Command handlers manage their own transactions. No Unit of Work middleware — you have one user, one writer, and explicit command boundaries that already define your transaction scope. Events fire *after* commit (anything else is asking for pain). Repos receive a connection from outside (makes testing trivial, costs nothing). The STORE phase (D16) should dispatch a `StoreResult` command through the pipeline — don't let background threads bypass the pipeline or you'll have two persistence paths to debug forever.

**Q3 — Schema & Migration:** Versioned SQL scripts in a `migrations/` folder. A single `schema_version` table. A 30-line function that runs unapplied scripts in order. No ORM, no framework. Read/write connection split is unnecessary — SQLite WAL mode handles concurrent reads during writes already. Don't add connection pooling to a single-user app.

**Q4 — Integration:** The middleware stack should be: **validate → execute handler → commit → publish events**. That's it. Four steps. Correlation ID is a field on the command dataclass that gets copied to emitted events — no propagation framework needed, just `event = SomeEvent(..., correlation_id=command.correlation_id)`. If you can't explain the full middleware chain in one sentence, it's too complex.
