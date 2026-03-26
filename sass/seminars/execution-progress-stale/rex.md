# Rex 🔥 — Devil's Advocate

## Position

Everyone's going to walk in here and design a beautiful, decomposed, event-driven progress system with aggregate tracking and cascading staleness and six clean components. And it's going to be **over-engineered garbage** for a single-user desktop app. Let me remind everyone: this is one person processing audio on one machine. There is no distributed system. There is no multi-tenant concern. The "monolithic" execution engine isn't a problem — it's a *feature*. It's one class that does one job: run blocks in order. The fact that the legacy version exists and works should tell you something. The instinct to decompose comes from server-side PTSD, not from actual pain in this codebase.

Progress events are where I get genuinely concerned. The brief says Griff wants them "extremely solid, standardized through the application" — but the brief also says events fire after commit (D36), and progress fires *during* execution. That's not a "tension that needs resolution." That's the architecture screaming at you that **progress is not a domain event**. It's a UI concern. Cramming it through the domain event bus is going to pollute a clean event system with high-frequency, low-value noise. A 10-minute ML inference block spitting out progress every 100ms means thousands of events through a system designed for things like "block execution completed." Those are categorically different things.

On staleness: everyone will want to cascade. A→B→C, A executes, mark B *and* C stale. Sounds logical. It's also **wrong** for this app. If C is 5 blocks downstream and the user only cares about B's output right now, you just made the entire graph look broken for no reason. Stale should propagate exactly one level deep. When B re-executes, *then* C goes stale. Immediate staleness means "your direct input changed." Transitive staleness is a lie — you don't *know* C's output is wrong until B actually runs, because B might produce identical output from different input.

## Key Insight

**Progress is not a domain event — it's a side-channel.** The moment you route progress through the same event bus as `ExecutionCompleted` or `BlockStaleMarked`, you've coupled your UI refresh rate to your domain model. Progress needs a completely separate, direct callback mechanism — a simple `Callable[[ProgressUpdate], None]` passed into the processor. The domain event bus never sees it. The UI subscribes to it through its own adapter. MCP polls it through a separate query. This keeps the event bus clean and the progress system simple.

## Risk

If you ignore this and decompose everything into six components communicating through events and pipelines, you'll spend weeks wiring up a `GraphPlanner` → `ExecutionDispatcher` → `InputGatherer` → `OutputPersister` → `StalenessTracker` → `ProgressAggregator` Rube Goldberg machine. Each component will be 50-100 lines. The orchestration *between* them will be 500 lines of glue, error handling, and state synchronization. You'll have turned a 400-line class into a 1000-line distributed system that runs on one thread. And when something breaks, you'll debug across six files instead of one. The legacy monolith's real problem isn't that it's monolithic — it's that it has poor internal structure. **Refactor the internals, keep the boundary.**

## Verdict

**Q1 — Progress Event Architecture:**
Don't use the event bus. Pass a `progress_callback: Callable[[ProgressUpdate], None] | None` into every processor's `execute()` method. `ProgressUpdate` is a simple dataclass: `block_id`, `percent` (0.0-1.0), `stage` (optional str), `timestamp`. That's it. No ETA (you'll calculate it wrong). No bytes processed (not every block processes bytes). For FULL execution, the orchestrator wraps each block's callback to map per-block progress into an overall range (block 1 of 3 = 0.0-0.33, etc). The UI layer adapts this callback into Qt signals. MCP reads a `dict[BlockId, ProgressUpdate]` from the orchestrator. Progress reporting is optional — blocks that don't call the callback just show indeterminate. Throttle on the *consumer* side (UI updates at 10Hz max), not the producer side.

**Q2 — Stale Propagation:**
Single-level only. When A completes, mark A's *direct* children stale. Not grandchildren. Triggers: execution completion, settings change, connection change — all three. Don't split into `STALE` vs `UPSTREAM_ERROR` — just use `STALE` with an optional `reason` field. The UI can check if the upstream block is in ERROR state and render differently, but staleness itself is one concept: "your inputs are no longer what you computed from." Store it on the block entity directly — a `data_state` column. Computed-on-demand sounds elegant but means a graph traversal every time you render. Staleness stops at Workspace boundaries (they're manual-pull by definition per D20).

**Q3 — Execution Engine Decomposition:**
**Don't decompose into separate components.** Keep one `ExecutionEngine` class with clean internal methods: `_plan(block_ids) → List[BlockId]`, `_pull(block_id) → InputData`, `_execute(block_id, input, progress_cb)`, `_store(block_id, result)`, `_mark_stale(block_id)`. That's your decomposition — it's called *methods*. The FULL execution orchestration is a simple `for` loop in `run_full()`. Thread submission and lock management stay in the same class because they're inherently coupled. If you must extract something, extract only `GraphPlanner` (topological sort + cycle detection) because that's genuinely independent graph math. Everything else stays together.
