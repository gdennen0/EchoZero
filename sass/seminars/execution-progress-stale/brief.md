# S.A.S.S. Brief — Progress Events, Stale Propagation, Execution Decomposition

## Context

EchoZero is a desktop audio processing app (Python + Qt) being rewritten. Node-based architecture where blocks process audio data and connect via a directed graph. Single user, single machine.

### Already Decided (relevant subset)
- **D4:** Two block types — Processor (pure transform, background execution) and Workspace (manages own data, manual pull, no execution pipeline)
- **D5:** Non-blocking execution. Processors run on background thread, never touch Qt. UI stays responsive.
- **D6:** Copy-on-pull — each block is an island. Pull = full copy of upstream data.
- **D7:** Events are pure time data, audio bundled as snapshot copy.
- **D10 (revised):** SQLite is runtime engine. `.ez` file is zip archive (SQLite DB + audio assets).
- **D16:** Execution flow: TRIGGER → PULL (copy) → EXECUTE (background) → STORE (atomic) → NOTIFY
- **D20:** FULL strategy stops at Workspace boundaries. Workspace blocks are terminal inputs — engine reads current output without calling process().
- **D21:** Copy semantics = true full copy (metadata AND audio files). Each block owns its files.
- **D22:** Per-block execution lock (not global). Concurrent independent block execution allowed.
- **D23:** CancellationToken checked at processor breakpoints. Cancel before STORE = safe rollback.
- **D27:** Partial failure in FULL execution: fail fast, stop at first failure. Already-committed ancestors keep outputs. Downstream marked STALE.
- **D28:** DataItem garbage collection via execution_id tracking.
- **D30-D34:** Single pipeline, commands as data, CQRS-lite, global undo stack.
- **D35-D38:** Typed events with DomainEvent base, collect-then-publish (after commit), breadth-first queue, Qt adapter in UI layer.
- **D39-D44:** UoW middleware, stateless repos, STORE through pipeline, separate read/write connections, migrations on project load.
- **D46-D49:** Thread-agnostic pipeline, ThreadPoolExecutor, single shared write connection with lock, StoreExecutionResult through pipeline.
- **D50:** Result[T] for handler returns, exceptions for infrastructure only.
- **D52:** Standardized progress events required (Griff requirement — "extremely solid, standardized through the application").

### Legacy State
- Execution engine: monolithic `ExecutionEngine` class handles topological sort, execution scheduling, progress tracking
- Progress: `SubprocessProgress` event with percentage, `ProgressTracker` class per block type
- Stale tracking: `DataState` enum (FRESH, STALE, ERROR), stored on blocks. Propagation logic scattered across services.
- Block status: `BlockStatusService` computes status from data state + execution state + filter state

### Existing Block Types (17+)
Audio processing blocks ranging from simple transforms (EQ, reverb, normalize) to heavy ML inference (Demucs stem separation, Whisper transcription, RVC voice conversion). Execution times range from <1 second to 10+ minutes.

## Questions for the Panel

### Q1: Progress Event Architecture
Griff wants "extremely solid progress events standardized through the application." Design the complete progress system:

- **Schema:** What does a progress event look like? Percentage? Stage name? ETA? Bytes processed? Items processed?
- **Reporting mechanism:** How do processors report progress? Callback function? Shared progress object? Channel?
- **Granularity:** What's the minimum update frequency? Maximum? Should processors be required to report or is it optional?
- **Composite progress:** When running FULL execution (multiple blocks in sequence), how does aggregate progress work? Per-block progress within overall progress?
- **Integration:** How does progress relate to the event bus (D35-D38)? Are progress events DomainEvents or something lighter (they fire very frequently)?
- **Cancellation UI:** Progress and cancellation are linked — progress display shows a cancel button. How does CancellationToken (D23) integrate with progress reporting?
- **Observability:** How does the MCP server / external tooling consume progress?

### Q2: Stale Propagation
When a block finishes execution, its downstream blocks become "stale" (their inputs changed). How does staleness work?

- **Propagation trigger:** What triggers stale marking? Only execution completion? Also settings changes? Connection changes?
- **Propagation depth:** Does staleness cascade? If A→B→C and A re-executes, does C become stale immediately or only when B re-executes?
- **UPSTREAM_ERROR:** Should STALE be split into STALE (parent output changed) and UPSTREAM_ERROR (parent execution failed)? (D27 says fail fast, downstream marked STALE — but is STALE the right status when the upstream FAILED?)
- **Storage:** Where does stale state live? On the block entity? In a separate status table? Computed on-demand from execution history?
- **Workspace boundaries:** D20 says FULL stops at Workspace blocks. Does staleness propagation also stop at Workspace boundaries? (Workspace blocks manage their own data — upstream changes don't auto-propagate.)
- **UI implications:** How does the UI show stale state? Visual indicator on blocks? Warning before using stale data?

### Q3: Execution Engine Decomposition
The legacy execution engine is monolithic. Should the new one be decomposed? Previous panel recommended decomposition but no decision was made.

Potential components:
- **Graph Planner:** Determines execution order (topological sort), validates graph, identifies what needs re-execution
- **Execution Dispatcher:** Manages the ThreadPoolExecutor, submits work, handles per-block locks
- **Input Gatherer:** Implements PULL phase — follows connections, copies upstream data
- **Output Persister:** Implements STORE phase — writes results, updates manifest, marks downstream stale
- **Staleness Tracker:** Owns the stale state, handles propagation
- **Progress Aggregator:** Collects per-block progress, computes composite progress for FULL execution

Questions:
- Monolith or decompose? If decompose, which components?
- How do these components interact? Direct calls? Through the pipeline? Via events?
- Where does the "FULL execution" orchestration live? (Run A, then B, then C in order, aggregate progress, stop on failure)
- Subprocess isolation for ML blocks — thread vs subprocess? Affects how progress is reported back.

## Constraints
- Python desktop app, single user, single machine
- Progress must be smooth enough for a responsive UI (not just 0% → 100%)
- Some processors run for 10+ minutes (ML inference)
- FULL execution can chain 5-10 blocks
- Must work without Qt (CLI, MCP, testing)
- Events fire after commit (D36) — but progress events happen DURING execution, before STORE commits. This is a tension that needs resolution.
