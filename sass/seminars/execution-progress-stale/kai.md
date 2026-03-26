# Kai ⚡ — The Pragmatist

## Position

The brief identifies a real tension: progress events fire *during* execution but D36 says events publish *after* commit. Stop trying to unify these. They're two different channels. Progress is a **live telemetry stream** — it's not a domain event, it doesn't go through the pipeline, and it shouldn't. A processor gets a `ProgressReporter` callback, calls `reporter(percent, stage_name)`, and that pushes directly to an in-memory pub/sub that the UI subscribes to. Done. No event bus, no persistence, no middleware. Progress is ephemeral by nature — nobody replays it.

For stale propagation and execution decomposition, the instinct to decompose into six named components is classic over-engineering. You have 17 blocks, one user, one machine. The execution engine is maybe 300 lines of real logic. Split it into two pieces max: a `GraphPlanner` that figures out what to run in what order, and an `ExecutionRunner` that does the PULL→EXECUTE→STORE→NOTIFY loop. Staleness is a flag on the block row in SQLite — not a separate tracker, not a service, not a component. When STORE commits, run one SQL statement: `UPDATE blocks SET data_state = 'STALE' WHERE id IN (SELECT downstream_id FROM connections WHERE upstream_id = ?)`. Cascade the full depth right there. It's a recursive CTE, it's fast, it's atomic with the commit.

The "UPSTREAM_ERROR" question is a good one, but keep it simple: add it as a third enum value (`FRESH | STALE | UPSTREAM_ERROR`). When a block fails, mark its downstreams `UPSTREAM_ERROR` instead of `STALE`. The UI shows a different color. That's the entire implementation — one `if` statement.

## Key Insight

Progress events and domain events are **fundamentally different things** and the biggest mistake this project can make is trying to route progress through the domain event system. Progress is high-frequency, ephemeral, fire-during-execution telemetry. Domain events are post-commit, durable, business-meaningful signals. Mixing them will either throttle your progress updates to be useless, or flood your event bus with noise. Two channels. Keep them separate.

## Risk

If you ignore this and try to "standardize" progress into the domain event system because Griff said "standardized through the application," you'll spend weeks building infrastructure for something that should be a callback and an `Observable`. You'll end up with progress events hitting middleware, going through UoW, getting queued breadth-first — all for a percentage number that's stale by the time it arrives. Meanwhile, the actual execution engine stays unbuilt. Standardized doesn't mean same-pipe. It means consistent interface.

## Verdict

**Q1 — Progress Event Architecture:**
- Simple `ProgressReport` dataclass: `block_id`, `percent` (0.0–1.0), `stage` (optional str), `timestamp`. No ETA — ETAs are always wrong and you'll waste time computing them. If you want one later, the UI can derive it from the percent history.
- Processors receive a `Callable[[float, str | None], None]` callback. Call it whenever you want. Optional — if you never call it, UI shows indeterminate spinner.
- Throttle at the **subscriber** side (UI updates at most every 100ms), not the producer side. Let processors fire as often as they want.
- Composite FULL progress: `overall = (completed_blocks + current_block_percent) / total_blocks`. Dead simple, good enough.
- Progress goes through a dedicated `ProgressBus` (literally a dict of `block_id → list[callback]`). Not the domain event bus. MCP/CLI subscribe the same way UI does.
- Cancel button calls `CancellationToken.cancel()` — it's already designed (D23). Progress display just shows/hides based on execution state.

```python
@dataclass
class ProgressReport:
    block_id: str
    percent: float  # 0.0 to 1.0, -1 for indeterminate
    stage: str | None = None

class ProgressBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, block_id: str, cb: Callable[[ProgressReport], None]):
        self._subs[block_id].append(cb)

    def report(self, block_id: str, percent: float, stage: str | None = None):
        r = ProgressReport(block_id, percent, stage)
        for cb in self._subs.get(block_id, []):
            cb(r)
```

**Q2 — Stale Propagation:**
- Triggers: execution completion, settings change, connection change. All three. If inputs changed, you're stale.
- Cascade immediately to full depth. Don't be clever. One recursive CTE, mark everything downstream. It's a graph with <50 nodes.
- `DataState = FRESH | STALE | UPSTREAM_ERROR`. Three values. UPSTREAM_ERROR when parent failed.
- Store it on the block row. `blocks.data_state TEXT NOT NULL DEFAULT 'STALE'`. No separate table.
- Staleness stops at Workspace boundaries — consistent with D20. Workspaces own their data.
- UI: colored dot on block. Green=fresh, yellow=stale, red=error. Tooltip explains why.

```sql
-- Mark all descendants stale in one shot
WITH RECURSIVE downstream(id) AS (
    SELECT downstream_block_id FROM connections WHERE upstream_block_id = ?
    UNION
    SELECT c.downstream_block_id FROM connections c
    JOIN downstream d ON c.upstream_block_id = d.id
    LEFT JOIN blocks b ON c.downstream_block_id = b.id
    WHERE b.block_type != 'workspace'  -- stop at workspace boundaries
)
UPDATE blocks SET data_state = 'STALE' WHERE id IN (SELECT id FROM downstream);
```

**Q3 — Execution Engine Decomposition:**
- Two pieces, not six: `GraphPlanner` and `ExecutionRunner`. That's it.
- `GraphPlanner.plan(trigger_block_id, strategy) → list[block_id]` — topological sort, skip fresh blocks, stop at workspaces for FULL.
- `ExecutionRunner.run(plan, progress_bus, cancellation_token)` — loops through the plan, does PULL→EXECUTE→STORE→mark-stale for each block.
- Direct calls between them. No pipeline routing for internal orchestration. The final `StoreExecutionResult` command goes through the pipeline per D46-D49, but the runner calling the planner is just a method call.
- Threads, not subprocesses. `ThreadPoolExecutor` is already decided (D49). If an ML block needs subprocess isolation, that's an implementation detail *inside* that processor's `execute()` method — the engine doesn't care.
- FULL orchestration lives in `ExecutionRunner.run_full()`. It's a for-loop with error handling. Don't overthink it.
