# Dr. Voss 🔬 — Response

## Position

The brief contains a critical unresolved tension that I suspect will be hand-waved by less rigorous panelists: **progress events are fundamentally different from domain events, yet the system has already committed to a collect-then-publish-after-commit model (D36).** Progress fires *during* execution, before STORE. Domain events fire *after* commit. These are irreconcilable within a single mechanism. Anyone who tries to shoehorn progress into the DomainEvent bus will either violate D36 or deliver progress updates minutes late — rendering them useless. Progress requires a separate, dedicated channel. Full stop.

On stale propagation, the answer is immediate full-depth cascade with distinct states. When A finishes, both B and C become stale *now* — not lazily. Lazy propagation is a correctness bug waiting to happen. A user looking at block C sees "FRESH" while its grandparent just invalidated everything. That's a lie in the UI. I don't ship lies. Furthermore, STALE and UPSTREAM_ERROR are categorically different states. STALE means "you can re-execute and it will probably work." UPSTREAM_ERROR means "don't even try — fix the ancestor first." Conflating them wastes the user's time with guaranteed-to-fail executions.

On decomposition: decompose, but surgically. The monolith disease isn't that the code is in one file — it's that concerns are entangled. The decomposition should follow the execution phases already defined in D16. Each phase is a unit with a clear contract. But the orchestrator (FULL execution) must remain a single, explicit component — not emergent behavior from events bouncing between services.

## Key Insight

**Progress events that fire after commit are not progress events — they're history.** The entire event system (D35-D38) is designed around transactional consistency: collect events during the unit of work, publish after commit. This is correct for domain events. It is *catastrophically wrong* for progress. Progress must be a side-channel — an `Observable[ProgressUpdate]` or callback that bypasses the event bus entirely. The event bus should receive exactly two progress-related domain events: `ExecutionStarted` and `ExecutionCompleted`. Everything in between is operational telemetry on a separate transport.

```python
# WRONG — progress as DomainEvent (violates D36 or arrives late)
class ProgressUpdated(DomainEvent):
    block_id: BlockId
    percent: float  # Published after commit... of what?

# RIGHT — progress as side-channel, separate from domain events
class ProgressChannel:
    """Thread-safe observable for in-flight progress. Not a DomainEvent."""
    def __init__(self):
        self._subscribers: list[Callable[[ProgressUpdate], None]] = []
        self._lock = threading.Lock()

    def report(self, update: ProgressUpdate) -> None:
        with self._lock:
            for sub in self._subscribers:
                sub(update)  # Fires immediately, no commit required

@dataclass(frozen=True)
class ProgressUpdate:
    block_id: BlockId
    execution_id: ExecutionId
    phase: str              # e.g. "demucs_inference", "writing_stems"
    fraction: float         # 0.0–1.0, NOT percentage — avoid int rounding traps
    items_completed: int | None = None
    items_total: int | None = None
    cancellation: CancellationToken | None = None  # So UI can cancel from progress display
```

## Risk

If progress is routed through the domain event bus, one of two things happens:

1. **D36 is violated** — you start publishing events mid-transaction, and now your event ordering guarantees are gone. Other handlers see progress events interleaved with uncommitted state. Debugging becomes archaeological.
2. **D36 is respected** — progress events queue up and publish after STORE commits. User sees 0% for 10 minutes, then a burst of 50 updates, then 100%. Congratulations, you've built a progress *historian*, not a progress *reporter*.

On staleness: if you do lazy propagation ("mark C stale only when B runs"), a user exports audio from C thinking it's fresh. It isn't. The output is wrong. They ship it. They blame you. This is a data integrity issue disguised as a UI concern.

## Verdict

**Q1 — Progress Event Architecture:**
- Progress is a **side-channel**, not a DomainEvent. Use a `ProgressChannel` per execution with thread-safe callbacks.
- Schema: `ProgressUpdate` with `block_id`, `execution_id`, `phase: str`, `fraction: float` (0.0–1.0), optional `items_completed`/`items_total`. No ETA — ETAs are lies (ML inference is non-linear). Let the UI extrapolate if it wants.
- Processors receive a `ProgressReporter` callback in their `process()` signature. Reporting is **mandatory** at minimum at phase boundaries. Processors that don't report get a synthetic 0%/100% wrapper — but this is the fallback, not the design.
- Composite progress: `CompositeProgressAggregator` holds a weighted plan (block weights based on historical execution time or equal weights as fallback). Each block's fraction contributes proportionally. The aggregator subscribes to individual `ProgressChannel`s.
- Throttle UI updates to **max 30Hz** at the Qt adapter layer. The channel itself is unthrottled — the adapter coalesces.
- CancellationToken is available via the `ProgressReporter` or passed alongside it. The cancel button in UI calls `token.cancel()`. The processor checks at breakpoints per D23.
- MCP/CLI subscribe to the same `ProgressChannel`. No Qt dependency.

**Q2 — Stale Propagation:**
- **Three states:** `FRESH`, `STALE`, `UPSTREAM_ERROR`. These are distinct and non-negotiable.
- **Triggers:** Execution completion (downstream → STALE), execution failure (downstream → UPSTREAM_ERROR), settings change on a block (self + downstream → STALE), connection added/removed (affected blocks → STALE).
- **Propagation depth:** Immediate, full-depth cascade via graph traversal. When A completes, walk all descendants and mark STALE. When A fails, walk all descendants and mark UPSTREAM_ERROR. This is a simple BFS on the adjacency list — it's O(n) and n is at most ~50 blocks. Don't over-engineer it.
- **Storage:** Column on the block row in SQLite: `data_state TEXT CHECK(data_state IN ('FRESH','STALE','UPSTREAM_ERROR'))`. Not computed on-demand — that's a trap that leads to inconsistent reads under concurrent execution.
- **Workspace boundaries:** Staleness **stops** at Workspace blocks, consistent with D20. A Workspace block is a terminal — upstream changes don't auto-invalidate its manually-managed data.
- **UI:** Stale blocks get a desaturated/dimmed visual treatment. UPSTREAM_ERROR blocks get a red indicator with tooltip showing which ancestor failed. Block context menu offers "Re-execute from here" for STALE, "Go to error" for UPSTREAM_ERROR.

**Q3 — Execution Engine Decomposition:**
- **Decompose into four components:** `GraphPlanner`, `ExecutionCoordinator`, `BlockExecutor`, `StalenessTracker`.
- `GraphPlanner`: Pure function. Takes graph + trigger block → returns ordered execution plan (list of block IDs). Handles topological sort, respects Workspace boundaries, validates no cycles. **Stateless, trivially testable.**
- `ExecutionCoordinator`: Owns the FULL execution lifecycle. Iterates the plan, delegates to `BlockExecutor` for each block, aggregates progress, implements fail-fast (D27). This is the one orchestrator — not event-driven, explicitly sequential with the plan.
- `BlockExecutor`: Handles one block's PULL → EXECUTE → STORE cycle. Acquires per-block lock (D22), copies inputs, runs processor with `ProgressReporter`, stores results through pipeline. Issues `StaleDownstreamCommand` after successful STORE.
- `StalenessTracker`: Handles `StaleDownstreamCommand` — walks the graph, updates `data_state` on affected blocks, publishes `BlocksBecameStale` domain event (this one IS a domain event — it fires after the staleness write commits).
- `InputGatherer` and `OutputPersister` are not separate components — they're phases within `BlockExecutor`. Extracting them adds indirection without reducing complexity.
- Interaction: `ExecutionCoordinator` calls `BlockExecutor` directly (not through pipeline — this is orchestration, not a command). `BlockExecutor` dispatches `StoreExecutionResult` through the pipeline per D46. `StalenessTracker` is a pipeline handler.
- Thread vs subprocess for ML: **Thread with ProcessPoolExecutor fallback** for blocks that need true isolation (GIL-heavy C extensions). Progress from subprocesses via `multiprocessing.Queue` bridged into `ProgressChannel`. But start with threads — only add subprocess isolation when a specific block proves it necessary.
