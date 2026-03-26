# Maya 🏗️ — Architect Response

## Position

The brief surfaces a fundamental architectural tension: progress events are **hot, high-frequency, in-flight signals** while the rest of the event system (D35-D38) is designed around **cold, post-commit domain events**. Trying to shove progress into the domain event bus would be like routing real-time audio through a message queue — wrong abstraction, wrong guarantees. Progress needs its own dedicated channel that is explicitly *not* a DomainEvent. Meanwhile, staleness is pure domain state that belongs squarely in the post-commit event flow. These are two completely different concerns with different lifecycles, and the architecture must reflect that separation.

On execution decomposition: the answer is an emphatic yes, decompose — but with discipline. The monolith's sin isn't that it's one class, it's that it conflates planning, dispatching, data movement, and progress tracking into a single blob where every concern can reach into every other concern. The decomposition should follow the existing TRIGGER → PULL → EXECUTE → STORE → NOTIFY pipeline (D16) — each phase becomes a component with a clean interface. But the orchestrator (what I'd call the `ExecutionCoordinator`) stays thin. It sequences phases. It does not *implement* phases. The coordinator knows the order of operations; the components know how to do the work.

Stale propagation must be **immediate and full-depth** on any trigger (execution completion, settings change, connection change). Half-hearted staleness is worse than no staleness — it gives users false confidence in stale data. Cascade everything, stop at Workspace boundaries (they're isolation points by D20), and split STALE from UPSTREAM_ERROR because they demand different user actions.

## Key Insight

Progress events and domain events have **opposite commit semantics**. Domain events fire *after* commit (D36). Progress fires *during* execution, *before* commit. Everyone will try to unify them. Don't. Progress is an **ephemeral observation channel** — a side-channel that exists outside the unit-of-work lifecycle. It should be a separate `ProgressChannel` that UI and MCP subscribe to independently. This isn't a hack, it's the correct architectural boundary. The moment you try to route progress through the domain event bus, you either break the "events after commit" invariant or you buffer progress until it's useless.

## Risk

If you unify progress with domain events or treat progress as "just another event," you'll end up with one of two failure modes: (1) you relax the post-commit guarantee for all events to accommodate progress, which breaks the transactional integrity of your entire event system, or (2) you create a "special case" flag on domain events meaning "this one doesn't wait for commit," which is a lie that will rot every event handler that assumes post-commit consistency. Either way, you've coupled two systems with fundamentally different temporal requirements, and every future change to one will risk breaking the other.

## Verdict

### Q1: Progress Event Architecture

**Schema:**
```python
@dataclass(frozen=True)
class ProgressReport:
    execution_id: str
    block_id: str
    phase: str              # "pull", "execute", "store"
    fraction: float         # 0.0–1.0, monotonically increasing
    stage: str | None       # e.g. "Loading model", "Processing chunk 3/10"
    items_completed: int | None
    items_total: int | None
    elapsed_ms: int
    estimated_remaining_ms: int | None
```

This is NOT a `DomainEvent`. It's a value object on a dedicated `ProgressChannel`.

**Reporting mechanism:** Processors receive a `ProgressReporter` protocol via their `execute()` call:

```python
class ProgressReporter(Protocol):
    def report(self, fraction: float, stage: str | None = None,
               items_completed: int | None = None, items_total: int | None = None) -> None: ...
    
    @property
    def cancellation_token(self) -> CancellationToken: ...
```

Combine progress and cancellation into one object the processor holds. Single dependency, clean interface. The reporter internally throttles to max ~10 updates/sec (UI doesn't need more), and the processor doesn't need to care about throttling.

**Composite progress:** An `ExecutionProgressAggregator` holds the block execution plan and weights each block by estimated duration (historical average, default to equal weight). It subscribes to per-block `ProgressReport`s and emits aggregate `ProgressReport`s with `block_id=None` for the overall execution. Simple weighted average.

**Integration:** `ProgressChannel` is a simple pub/sub with `subscribe(callback)` / `unsubscribe()`. Qt adapter subscribes and marshals to main thread via `QMetaObject.invokeMethod`. MCP server subscribes directly. No event bus involvement.

**Reporting is optional but expected.** Processors that don't report progress simply show indeterminate state in UI. No enforcement — it's a protocol, not a contract.

### Q2: Stale Propagation

**Triggers:** Three things cause staleness:
1. Upstream block execution completes (output changed)
2. Block settings change (needs re-execution)
3. Connection topology changes (inputs changed)

**Propagation depth:** Full cascade, immediately. A→B→C: when A completes, both B and C are marked stale in one pass. Don't wait for B to execute. The user needs to see the full blast radius *now*.

**Split the states:**
```python
class DataState(Enum):
    FRESH = "fresh"
    STALE = "stale"               # upstream output changed, re-execute to refresh
    UPSTREAM_ERROR = "upstream_error"  # upstream failed, can't execute until upstream is fixed
    ERROR = "error"               # this block's own execution failed
```

UPSTREAM_ERROR propagates the same way STALE does — full cascade. But UI shows it differently (red vs yellow).

**Storage:** `data_state` column on the `blocks` table. It's mutable block state, not derived. Computed-on-demand sounds clean but requires graph traversal on every read — O(n) per block render. Store it, update it transactionally when propagation triggers fire. Propagation is a single SQL update:

```sql
UPDATE blocks SET data_state = 'stale'
WHERE block_id IN (SELECT descendant_id FROM block_descendants WHERE ancestor_id = ?
                   AND descendant_id NOT IN (SELECT block_id FROM blocks WHERE block_type = 'workspace'))
```

Pre-compute the transitive closure in a `block_descendants` table, updated on connection changes. O(1) staleness propagation.

**Workspace boundaries:** Yes, staleness stops at Workspace blocks. They are isolation boundaries. This is consistent with D20.

**Stale marking is a domain operation** that goes through the pipeline and emits proper `DomainEvent`s (`BlocksMarkedStale`, `BlocksMarkedUpstreamError`) after commit. UI subscribes to these to update visual state.

### Q3: Execution Engine Decomposition

**Decompose into exactly four components:**

| Component | Responsibility | Interface |
|---|---|---|
| `GraphPlanner` | Topological sort, determines execution order, validates DAG | `plan(trigger_block_id, strategy) → ExecutionPlan` |
| `ExecutionCoordinator` | Orchestrates FULL execution: sequences blocks per plan, handles failure policy | `execute(plan) → Result[ExecutionSummary]` |
| `BlockExecutor` | Runs single block: PULL → EXECUTE → STORE. Owns the per-block lock. | `execute_block(block_id, execution_id, progress_reporter) → Result[BlockOutput]` |
| `StalenessService` | Owns propagation logic, maintains descendant table | `propagate(block_id, new_state) → list[block_id]` |

Don't create separate InputGatherer / OutputPersister classes. PULL and STORE are phases *within* `BlockExecutor` — they're sequential steps in a single unit of work, not independent services. Extracting them creates artificial boundaries that force data to hop between objects for no reason.

**Interaction pattern:** `ExecutionCoordinator` calls `BlockExecutor` directly (not through the pipeline — execution is infrastructure, not a command). After each block completes, `StoreExecutionResult` goes through the pipeline (D46-D49), which triggers `StalenessService` as a handler. Progress flows through `ProgressChannel`, completely orthogonal.

**FULL execution orchestration lives in `ExecutionCoordinator`:**
```python
class ExecutionCoordinator:
    async def execute(self, plan: ExecutionPlan) -> Result[ExecutionSummary]:
        aggregator = ProgressAggregator(plan, self._progress_channel)
        for block_id in plan.execution_order:
            if self._cancellation_token.is_cancelled:
                return Err(Cancelled())
            result = await self._block_executor.execute_block(
                block_id, plan.execution_id, aggregator.reporter_for(block_id))
            if isinstance(result, Err):
                self._staleness_service.propagate(block_id, DataState.UPSTREAM_ERROR)
                return result  # fail fast per D27
        return Ok(ExecutionSummary(...))
```

**Subprocess vs thread:** Thread (via ThreadPoolExecutor, per D46-D49). Subprocess isolation for ML blocks is premature complexity — you're single-user, single-machine. If a block crashes the process, that's a bug to fix, not an architecture to work around. Progress reporting across process boundaries adds serialization overhead and failure modes (pipe breaks, zombie processes). Keep it simple. If you need subprocess isolation later, `BlockExecutor` is the single point where that abstraction would live — clean boundary, easy to change.
