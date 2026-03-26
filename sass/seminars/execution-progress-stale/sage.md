# Sage 🧮 — Data Modeler Response

## Position

The core tension in this brief is **who owns what state and where does it live**. Right now staleness, progress, and execution status are smeared across multiple places — block entities, services, transient trackers. That's a data model problem. Let me be blunt: progress and staleness are fundamentally different kinds of data. Progress is **ephemeral stream data** — it exists only during execution and has no business in SQLite or the domain event bus. Staleness is **derived persistent state** — it's a function of the execution graph and should be computed from a single source of truth: the `execution` record and the graph topology.

The brief identifies the D36 tension (events fire after commit, but progress fires during execution). This isn't a tension to "resolve" — it's the data model telling you these are **two different systems**. Progress is an operational signal, not a domain event. Treat it like a log stream, not like `BlockExecutionCompleted`. Meanwhile, staleness should not be stored as a mutable field on block entities. That's denormalization that will rot. Staleness is a **query** — "does this block's last successful execution_id predate its upstream block's last successful execution_id?" If you can answer that from the execution table, you don't need a `data_state` column at all.

For execution decomposition: yes, decompose — but decompose along **data ownership boundaries**, not along arbitrary "component" lines. Each component should own exactly one table or state domain. The Graph Planner owns topology (read-only from connections). The Execution Dispatcher owns the `executions` table lifecycle. The Progress Aggregator owns an in-memory-only progress store. The Staleness Tracker is just a query service — it owns nothing, it reads executions + connections.

## Key Insight

**Staleness is not state — it's a query.** Everyone will want to store `data_state = STALE` on the block row and propagate it with events. That's redundant, mutable state that can drift from reality. Instead, staleness should be computed:

```python
def is_stale(block_id: BlockId, conn: Connection) -> bool:
    """A block is stale if any upstream block executed more recently."""
    return conn.execute("""
        SELECT EXISTS(
            SELECT 1 FROM connections c
            JOIN executions e_up ON e_up.block_id = c.source_block_id
            LEFT JOIN executions e_me ON e_me.block_id = c.target_block_id
                AND e_me.status = 'SUCCESS'
            WHERE c.target_block_id = ?
            AND e_up.status = 'SUCCESS'
            AND (e_me.completed_at IS NULL OR e_up.completed_at > e_me.completed_at)
        )
    """, (block_id,)).fetchone()[0]
```

Cache it in memory if you need performance. But the **source of truth** is the execution history, not a flag on a block. This eliminates an entire class of bugs where the flag disagrees with reality.

## Risk

If staleness is stored as mutable state on blocks, you will inevitably hit situations where:
1. A crash between "execution committed" and "downstream marked stale" leaves the graph in an inconsistent state.
2. Undo/redo of connections must manually recalculate stale flags — and someone will forget an edge case.
3. Two concurrent executions on independent branches propagate stale markers that stomp each other.

If progress events are shoved through the domain event bus (D35-D38), you'll flood the collect-then-publish queue with hundreds of events per second during ML inference, bloating memory and creating backpressure on the pipeline that was designed for low-frequency domain events.

## Verdict

### Q1: Progress Event Architecture

**Don't make progress a DomainEvent.** Use a separate, lightweight, in-memory-only channel. Concrete design:

```python
@dataclass(frozen=True)
class ProgressReport:
    execution_id: ExecutionId
    block_id: BlockId
    phase: str              # "pull", "execute", "store"
    fraction: float         # 0.0–1.0
    stage: str | None       # "Separating stems", "Loading model", etc.
    items_done: int | None  # e.g. samples processed
    items_total: int | None
    timestamp: float        # time.monotonic()

class ProgressChannel:
    """Thread-safe, in-memory, pub-sub for progress. Not persisted. Not a DomainEvent."""
    def __init__(self):
        self._subscribers: list[Callable[[ProgressReport], None]] = []
        self._lock = threading.Lock()
    
    def publish(self, report: ProgressReport) -> None:
        with self._lock:
            for sub in self._subscribers:
                sub(report)
    
    def subscribe(self, callback: Callable[[ProgressReport], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)
```

- Processors receive a `report_progress: Callable[[float, str | None], None]` callback. Required to call at least once (start). Recommended: throttle to max 10Hz at the publisher side.
- Composite/FULL progress: the **Progress Aggregator** holds the execution plan (ordered block list), assigns weight per block (default equal, overridable), and computes `overall = sum(block_fraction * block_weight)`.
- CancellationToken is checked inside the same callback — processor calls `report_progress(0.5, "Processing")`, which internally checks the token and raises `CancelledException` if cancelled. One touch point, not two.
- MCP/CLI subscribes to the same `ProgressChannel`. No Qt dependency.

### Q2: Stale Propagation

- **Staleness is computed, not stored.** Source of truth: `executions` table + `connections` table. Query shown above.
- **Propagation triggers:** No explicit propagation needed. When A executes successfully, `is_stale(B)` automatically returns True because A's `completed_at > B's completed_at`. Connection changes invalidate staleness implicitly (new connection = no execution covers it).
- **Cascade:** Yes, transitively. `is_stale` recurses or uses a CTE. If A→B→C and A re-executes, C is stale because its transitive upstream changed. This falls out naturally from the query.
- **UPSTREAM_ERROR:** Yes, distinguish it — but as a computed status, not a stored flag:

```python
class DataStatus(Enum):
    FRESH = "fresh"
    STALE = "stale"
    UPSTREAM_ERROR = "upstream_error"
    NO_DATA = "no_data"

def compute_status(block_id, conn) -> DataStatus:
    if has_upstream_failure(block_id, conn):
        return DataStatus.UPSTREAM_ERROR
    if is_stale(block_id, conn):
        return DataStatus.STALE
    if has_successful_output(block_id, conn):
        return DataStatus.FRESH
    return DataStatus.NO_DATA
```

- **Workspace boundaries:** Staleness stops at Workspace blocks. They are terminal — `is_stale` should not traverse past them. Workspace blocks are never stale from upstream; they're stale only if their own internal state says so.
- **Caching:** Wrap `compute_status` in an in-memory cache invalidated by `BlockExecutionCompleted` and `ConnectionChanged` domain events. This gives you O(1) reads with correctness guarantees.

### Q3: Execution Engine Decomposition

**Decompose along data ownership lines:**

| Component | Owns | Persists? |
|---|---|---|
| **GraphPlanner** | Topological sort, execution plan | No (reads connections) |
| **ExecutionDispatcher** | `executions` table rows, thread pool, per-block locks | Yes |
| **InputGatherer** | PULL phase, copy logic | No (reads upstream outputs) |
| **ProgressAggregator** | In-memory progress state | No |
| **StatusQuery** | Nothing — pure read service over executions + connections | No |

Kill "Output Persister" as a separate component — STORE is just the tail end of the execution handler, writing through the pipeline (D46). Don't extract it.

Kill "Staleness Tracker" as a component — it's `StatusQuery`, a stateless function.

**FULL orchestration** lives in `ExecutionDispatcher`. It receives the plan from `GraphPlanner`, iterates blocks in order, delegates PULL to `InputGatherer`, runs the processor, stores via pipeline, publishes `BlockExecutionCompleted`. On failure: stop, mark execution as failed, done. Downstream is automatically `UPSTREAM_ERROR` via query.

**Interaction:** Direct calls within the execution pipeline, not events between components. These are collaborators in a single use case, not independent services. Over-eventing internal orchestration is cargo-cult microservices in a desktop app.

**Thread vs subprocess for ML:** Thread (via ThreadPoolExecutor, per D47). Subprocess adds IPC complexity for progress reporting. If a specific block needs process isolation (e.g., GPU memory), make it opt-in per processor, with progress reported over a pipe that the dispatcher reads into the `ProgressChannel`.
