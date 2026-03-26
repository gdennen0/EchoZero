# EchoZero — Universal Observability Architecture

**Status:** Proposed  
**Date:** 2026-03-01  
**Decision:** Design universal cross-cutting observability into the rewrite from day one.

---

## Summary

Every layer of EchoZero should be instrumentable through a single, consistent mechanism. External consumers (MCP server, debug panel, test harness, log file) subscribe to the same event stream and project it however they need.

## Current State (Legacy Codebase)

The existing codebase already validates this pattern:

- **Event bus exists:** `src/application/events/event_bus.py` — thread-safe pub/sub with Qt main-thread dispatching. Clean `subscribe/publish/unsubscribe` API.
- **Typed domain events exist:** `src/application/events/events.py` — ~30 typed `DomainEvent` subclasses covering projects, blocks, connections, execution, setlists, MA3 comms, UI state, errors.
- **Facade publishes events:** `ApplicationFacade` publishes `ProjectLoaded`, `SubprocessProgress`, `UIStateChanged`, etc.
- **Pattern is proven:** The pub/sub pattern is already used for real decoupling between execution threads and UI.

### Gaps in Current Implementation

| Gap | Impact |
|-----|--------|
| No layer/component tagging | Can't filter events by origin |
| Inconsistent coverage | ~5-10 of 104 facade methods emit events |
| No infrastructure events | Persistence, file I/O, network calls are invisible |
| Coupled to Qt | Event bus uses `QEvent`/`QCoreApplication.postEvent` for threading |

## Proposed Architecture

### Universal Event Schema

```python
@dataclass
class ObservableEvent:
    """Base event for all layers."""
    layer: str          # "domain" | "application" | "infrastructure" | "ui"
    component: str      # "block_service" | "execution_engine" | "persistence" | ...
    event_type: str     # "BlockAdded" | "QueryExecuted" | "FileWritten" | ...
    payload: dict       # Event-specific data
    timestamp: float    # time.time()
    correlation_id: Optional[str] = None  # Trace operations across layers
```

### Layer Responsibilities

```
Layer 0 — Domain
  Entities emit pure domain events (no framework dependency)
  Examples: BlockAdded, ConnectionCreated, ProjectLoaded

Layer 1 — Application  
  Commands/queries emit operation events
  Examples: CommandExecuted, CommandUndone, QueryResolved

Layer 2 — Infrastructure
  Persistence/IO emit infrastructure events
  Examples: RowInserted, FileWritten, OscPacketSent

Layer 3 — Adapters (Consumers)
  Qt adapter, MCP server, debug tools — all just subscribers
  No events emitted, only consumed
```

### Event Bus Design

```python
class EventBus:
    """Framework-agnostic event bus. Core of the observation system."""
    
    def subscribe(self, 
                  handler: Callable[[ObservableEvent], None],
                  filter: Optional[EventFilter] = None) -> Subscription:
        """Subscribe with optional filtering by layer/component/event_type."""
        ...
    
    def publish(self, event: ObservableEvent) -> None:
        """Publish to all matching subscribers. Near-zero cost with 0 subscribers."""
        ...

@dataclass
class EventFilter:
    """Filter events by layer, component, or type."""
    layers: Optional[Set[str]] = None
    components: Optional[Set[str]] = None
    event_types: Optional[Set[str]] = None
```

- **Pure Python, no Qt dependency** — framework adapters wrap it for thread safety
- **Filter at subscription time** — subscribers only receive what they care about
- **Near-zero overhead** — publishing to 0 matching subscribers is a dict lookup + empty list check

### Observable Contract

Any component that emits events implements:

```python
class Observable:
    """Mixin for any component that emits events."""
    
    def __init__(self, event_bus: EventBus, layer: str, component: str):
        self._event_bus = event_bus
        self._layer = layer
        self._component = component
    
    def _emit(self, event_type: str, payload: dict, correlation_id: str = None):
        self._event_bus.publish(ObservableEvent(
            layer=self._layer,
            component=self._component,
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id,
        ))
```

### MCP Server (Consumer)

The MCP server is a thin projection of the event stream:

```
Event Bus  →  MCP Adapter  →  MCP Resources (read state)
                            →  MCP Tools (invoke commands via facades)
                            →  MCP Notifications (stream events)
```

- **Resources:** Project state, block graph, execution status — derived from event stream + current state queries
- **Tools:** Create block, connect, execute — thin wrappers around application layer commands
- **Notifications:** Real-time event stream, filterable by layer/component

The MCP server knows nothing about internals. It subscribes to the bus and projects into the MCP protocol. Replace it with a WebSocket server, a debug panel, or a test spy — same interface, different output.

### Phased Rollout

| Phase | Scope | Value |
|-------|-------|-------|
| **1** | Event bus + domain events + Observable mixin | Foundation |
| **2** | Application layer events (commands, queries) | Debugging |
| **3** | Infrastructure events (persistence, I/O) | Full tracing |
| **4** | MCP server (read-only resources) | AI observability |
| **5** | MCP tools (write operations) | AI interaction |
| **6** | Real-time streaming (SSE/WebSocket for execution monitoring) | Live dashboards |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **Event noise** | Filter at subscription time. Consider event levels (debug/info/warn) like logging. |
| **Performance during audio processing** | Publishing to 0 subscribers must be near-free. Profile early. Disable debug-level events in production. |
| **Circular event chains** | No re-entrant publish rule. Handler publishes → queued, not immediate. |
| **Serialization overhead** | Payload is a dict, not serialized until a consumer needs it (lazy serialization). |

## Key Principle

> The MCP surface is an API *contract* — design it independently from internal facades. The facades are the implementation; the MCP schema is the contract. They'll mostly align, but designing them separately catches the places where the internal structure doesn't match what an external consumer actually needs.

## References

- Current event bus: `src/application/events/event_bus.py`
- Current events: `src/application/events/events.py`
- Behavioral spec: `SPEC.md`
- Architecture: `docs/architecture/ARCHITECTURE.md`
