# S.A.S.S. Synthesis — Progress Events, Stale Propagation, Execution Decomposition

**Panel:** Maya 🏗️, Kai ⚡, Dr. Voss 🔬, Rex 🔥, Sage 🧮, Luna 🎨  
**Date:** 2026-03-01

---

## Consensus (All 6 Agree)

1. **Progress is NOT a domain event.** Strongest consensus of the entire distillation process. All 6 panelists independently identified the D36 tension (domain events fire after commit, progress fires during execution) and all said the same thing: progress needs its own dedicated side-channel, completely separate from the event bus. Domain events get exactly two execution signals: `ExecutionStarted` and `ExecutionCompleted`. Everything in between is operational telemetry on a separate transport.

2. **ProgressReport schema is a simple dataclass.** `block_id`, `fraction` (0.0–1.0), `stage` (optional string like "Separating vocals"), `execution_id`. Not persisted. Not serialized. Ephemeral.

3. **Processors receive a progress callback.** A `report_progress(fraction, stage)` callable passed into `process()`. Processors call it when they want. Simple contract.

4. **Throttle at the consumer, not the producer.** UI coalesces updates (10-30Hz). Processors fire as often as they want. The channel is unthrottled — adapters decide their refresh rate.

5. **Staleness stops at Workspace boundaries.** Consistent with D20 (Workspace blocks are terminal, manual-pull). Universal agreement.

6. **Threads, not subprocesses for ML blocks.** ThreadPoolExecutor (D47) for now. Subprocess isolation is premature. If needed later, it's a processor-internal detail — the engine doesn't need to know.

7. **Internal orchestration uses direct calls, not events.** Components within the execution system call each other directly. No pipeline routing or event-driven coordination for internal phases. Events are for external notification after the work is done.

---

## Tensions

### T1: Stale Propagation Depth — Full Cascade vs Single Level

**Full depth, immediate** (Maya, Voss, Kai, Luna, Sage — 5 of 6): When A executes, mark B AND C stale immediately. User sees the full blast radius right away. Prevents false confidence in stale data.

**Single level only** (Rex — 1 of 6): Only mark direct children stale. Grandchildren become stale when their parent re-executes. Argument: you don't *know* C's output is wrong until B runs — B might produce identical output.

**Resolution: Full depth.** Rex's argument is technically correct (B might produce identical output), but Luna and Voss make the stronger UX case: a user looking at block C and seeing "FRESH" when its grandparent just changed is a lie. Users should see the blast radius immediately. If C's output happens to still be valid after B re-runs, it gets marked FRESH again — no harm done. False "stale" is annoying. False "fresh" is dangerous.

### T2: Staleness Storage — Stored Column vs Computed Query

**Stored column on block row** (Maya, Voss, Kai, Rex, Luna — 5 of 6): `data_state TEXT` column. Updated transactionally when propagation triggers fire. Fast reads, explicit.

**Computed from execution history** (Sage — 1 of 6): Staleness is a query over `executions` + `connections` tables. "Is this block's last execution older than its upstream's?" No mutable flag. Eliminates drift between flag and reality.

**Resolution: Stored column.** Sage's computed approach is elegant and theoretically pure — it eliminates the possibility of stale flags disagreeing with reality. But it has practical problems: (1) computing staleness requires a graph traversal or recursive CTE for every block render, which is O(n) per block; (2) it requires an `executions` table with history, which we haven't designed yet and adds schema complexity; (3) staleness triggered by settings changes or connection changes wouldn't be captured by execution timestamps alone. A stored column is simpler, faster to read, and easier to update atomically with the trigger event. If the flag ever drifts, a "recompute staleness" operation can fix it — but with UoW-based transactional updates (D39), drift shouldn't happen.

### T3: UPSTREAM_ERROR — Split or Not?

**Split into three states** (Maya, Voss, Kai, Luna — 4 of 6): `FRESH`, `STALE`, `UPSTREAM_ERROR`. They're semantically different: STALE means "re-run and it'll probably work." UPSTREAM_ERROR means "don't bother — fix the ancestor first."

**Don't split** (Rex — 1 of 6): Just `STALE` with an optional reason. UI can check upstream status and render differently.

**Computed status, not stored flag** (Sage — 1 of 6): Derive UPSTREAM_ERROR from the execution history query.

**Resolution: Split.** Four states total:

```python
class DataState(Enum):
    FRESH = "fresh"              # Output is current
    STALE = "stale"              # Upstream changed, re-execute to update
    UPSTREAM_ERROR = "upstream_error"  # Upstream failed, can't execute until fixed
    ERROR = "error"              # This block's own execution failed
```

UPSTREAM_ERROR and ERROR are different user actions: ERROR means "look at this block's logs." UPSTREAM_ERROR means "find the red block upstream and fix that first." Luna: "collapsing these is lying to the user."

### T4: Execution Decomposition — How Many Components?

**Monolith with clean methods** (Rex — 1 of 6): One class, internal methods for plan/pull/execute/store/stale. Decomposition creates more glue than it saves.

**Two components** (Kai — 1 of 6): GraphPlanner + ExecutionRunner. That's it.

**Four components** (Maya, Voss, Luna — 3 of 6): GraphPlanner, ExecutionCoordinator, BlockExecutor, StalenessService.

**Five components** (Sage — 1 of 6): GraphPlanner, ExecutionDispatcher, InputGatherer, ProgressAggregator, StatusQuery.

**Resolution: Four components.** Rex is right that over-decomposition creates glue code. But a single class handling graph planning, thread management, data copying, result storage, AND stale propagation is what created the legacy monolith. Four components with clear boundaries:

| Component | Responsibility | Stateful? |
|---|---|---|
| **GraphPlanner** | Topological sort, execution order, DAG validation | No — pure function |
| **ExecutionCoordinator** | FULL orchestration, sequencing blocks, aggregate progress, fail-fast | Yes — owns the active execution lifecycle |
| **BlockExecutor** | Single block PULL → EXECUTE → STORE. Per-block lock. Progress reporting. | Yes — owns per-block locks |
| **StalenessService** | Propagation logic, updates `data_state` on affected blocks | No — pipeline handler triggered by events |

PULL and STORE are phases within BlockExecutor, not separate components. ProgressAggregator is a helper within ExecutionCoordinator, not a standalone service.

### T5: ETA in Progress Reports

**Include ETA** (Luna — 1 of 6): Users want to know "when will this be done?"

**No ETA** (Voss, Kai, Rex — 3 of 6): ETAs for ML inference are lies. Non-linear processing makes prediction unreliable.

**Resolution: No ETA in the schema.** The ProgressReport includes `fraction` and `timestamp`. The UI can compute a naive ETA from the rate of fraction change if it wants. But the progress system doesn't promise ETA accuracy — that's a UI presentation choice, not a data contract.

---

## Decisions

### D53: Progress is a side-channel, not a domain event
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** Progress uses a dedicated `ProgressChannel` — a thread-safe, in-memory pub/sub completely separate from the domain event bus (D35-D38). The domain event bus receives only `ExecutionStarted` and `ExecutionCompleted`/`ExecutionFailed`. All mid-execution progress goes through `ProgressChannel`. Not persisted. Not serialized.  
**Rationale:** Progress fires during execution (before commit). Domain events fire after commit (D36). These are irreconcilable in one system. Attempting to unify them either breaks D36 or makes progress useless.

### D54: ProgressReport schema
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:**
```python
@dataclass(frozen=True)
class ProgressReport:
    execution_id: str
    block_id: str
    phase: str              # "pull", "execute", "store"
    fraction: float         # 0.0–1.0
    stage: str | None       # "Separating vocals", "Loading model", etc.
    items_completed: int | None
    items_total: int | None
    timestamp: float        # time.monotonic()
```
No ETA field. UI can derive ETA from fraction rate if desired.  
**Rationale:** Minimal, sufficient, no promises the system can't keep.

### D55: Processors receive a ProgressReporter callback
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** Processors receive a `ProgressReporter` in their `process()` method:
```python
class ProgressReporter(Protocol):
    def report(self, fraction: float, stage: str | None = None,
               items_completed: int | None = None, items_total: int | None = None) -> None: ...
    @property
    def cancellation_token(self) -> CancellationToken: ...
```
Combines progress reporting and cancellation into one object. Reporting is optional — processors that don't call it show indeterminate in UI. Throttle at the consumer (UI at 10-30Hz), not the producer.  
**Rationale:** Single dependency for the processor. Clean protocol. No coupling to event bus or Qt.

### D56: Composite progress via weighted aggregation
**Date:** 2026-03-01  
**Status:** Decided  
**Decision:** For FULL execution, the ExecutionCoordinator holds an aggregator that weights each block's fraction by estimated duration (historical average, default to equal weight). Per-block progress maps into the block's slice of overall progress. One smooth bar for the user.  
**Rationale:** Users running FULL execution need one coherent progress view, not per-block fragments.

### D57: Four-state DataState enum
**Date:** 2026-03-01  
**Status:** Decided (5-1, Rex dissented)  
**Decision:**
```python
class DataState(Enum):
    FRESH = "fresh"
    STALE = "stale"
    UPSTREAM_ERROR = "upstream_error"
    ERROR = "error"
```
Stored as a column on the blocks table. STALE = upstream output changed. UPSTREAM_ERROR = upstream execution failed. ERROR = this block's own execution failed. FRESH = output is current.  
**Rationale:** Different states require different user actions. STALE → re-run. UPSTREAM_ERROR → fix ancestor first. ERROR → check this block's logs. Conflating them wastes user time.

### D58: Immediate full-depth stale cascade
**Date:** 2026-03-01  
**Status:** Decided (5-1, Rex dissented)  
**Decision:** Staleness propagates immediately to all descendants when triggered. Triggers: execution completion, settings change, connection change. A→B→C: when A executes, both B and C are marked STALE in one pass. When A fails, both B and C are marked UPSTREAM_ERROR. Cascade stops at Workspace boundaries (D20).  
**Rationale:** False "fresh" is dangerous (user exports stale audio thinking it's current). False "stale" is merely annoying (user re-runs and gets the same output). Full cascade is the safe default.

### D59: Staleness stored as column, not computed
**Date:** 2026-03-01  
**Status:** Decided (5-1, Sage dissented)  
**Decision:** `data_state` column on the `blocks` table. Updated transactionally when propagation triggers fire. Propagation is a single SQL update via recursive CTE or pre-computed descendant table. StalenessService is a pipeline handler triggered by `ExecutionCompleted`/`ExecutionFailed`/`SettingsChanged`/`ConnectionChanged` domain events.  
**Rationale:** Fast reads (O(1) per block), transactional updates via UoW, explicit state. Sage's computed approach is theoretically purer but requires graph traversal per read and doesn't capture settings/connection-triggered staleness.

### D60: Execution engine decomposed into four components
**Date:** 2026-03-01  
**Status:** Decided (4-1-1)  
**Decision:** Four components: GraphPlanner (pure function, topological sort), ExecutionCoordinator (FULL orchestration, sequencing, aggregate progress, fail-fast), BlockExecutor (single block PULL→EXECUTE→STORE, per-block lock, progress reporting), StalenessService (propagation logic, pipeline handler). PULL and STORE are phases within BlockExecutor, not separate components. Components interact via direct calls internally; StalenessService is triggered via pipeline events after STORE commits.  
**Rationale:** Clean boundaries without over-decomposition. Each component has a single responsibility and a clear interface. Rex's monolith-with-methods approach risks re-creating the legacy entanglement. Sage's 5-component split extracts phases that belong together.

### D61: Threads, not subprocesses for ML blocks
**Date:** 2026-03-01  
**Status:** Decided (panel unanimous)  
**Decision:** All block execution uses ThreadPoolExecutor (D47). No subprocess isolation. If a specific processor needs process isolation later (GPU memory, crash isolation), it's an internal implementation detail of that processor — the engine doesn't know or care.  
**Rationale:** Subprocess adds IPC complexity for progress reporting, error handling, and cancellation. Premature for a single-user desktop app.

---

## Open Questions

1. **Progress reporting during PULL phase:** PULL copies upstream data (D6, D21 — full copy including audio files). For large projects this could take seconds. Should BlockExecutor report progress during PULL, or just during EXECUTE? If PULL reports progress, the ProgressReport `phase` field distinguishes them.

2. **Historical execution time for weighted composite progress:** Where does this data come from? Do we need an `execution_history` table tracking past durations per block type? Or just use equal weights and let it be "good enough"?
