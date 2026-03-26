# EchoZero 2 Architecture — March 22, 2026

## Overview

EchoZero 2 is a desktop audio analysis and event editing workstation built on a **first-principles architecture** derived from five axioms about information, execution, and user data.

- **216 tests** (all passing)
- **0.19 seconds** test run time
- **Zero external process dependencies** (pure Python, uses existing domain model)

---

## Core Axioms (First Principles)

1. **Information Separation**: A DAG has two kinds of information — structure (nodes, connections) and evaluation (what each node produces). These are categorically different.

2. **Derived = Cache**: Derived data can always be recomputed from its inputs. Therefore it is a cache, not truth. Losing it costs time, not information.

3. **User Edits = Truth**: A user edit to derived data (moving a detected event) creates NEW truth that cannot be recomputed. It is no longer derived — it has been promoted to user-authored data.

4. **Time Flows Forward**: A 30-second computation cannot deliver its results at t=0. Any system claiming to batch notifications until completion is lying to observers for 30 seconds. Execution progress and lifecycle are inherently streaming.

5. **Concurrency by Nature**: A DAG defines a partial order. Nodes with no dependency between them are concurrent by nature — serializing them is an artificial constraint, not a requirement.

---

## Architecture: 4 State Containers

The system maintains four distinct state containers, each with different durability and validity semantics:

### 1. **GraphStore** (`echozero.domain.graph.Graph`)
- **What**: Topology (blocks, connections) + block parameters
- **Durability**: Durable — must persist to disk on save
- **Authored by**: User via Pipeline commands
- **Axiom**: FP1 (structure)
- **Key methods**: `add_block()`, `add_connection()`, `set_block_state()`, `topological_sort()`, `downstream_of()`

### 2. **CacheStore** (`echozero.cache.ExecutionCache`)
- **What**: Evaluation outputs keyed by `(block_id, port_name)`
- **Durability**: Discardable — can be lost without data corruption (Axiom FP2)
- **Produced by**: ExecutionEngine after each block succeeds
- **Validity**: Invalidated when upstream inputs change via `propagate_stale()`
- **Key methods**: `store()`, `get()`, `get_all()`, `invalidate()`, `invalidate_downstream()`, `has_valid_output()`, `clear()`

### 3. **OverrideStore** (`echozero.overrides.OverrideStore`)
- **What**: User edits to derived data (Axiom FP3)
- **Durability**: Durable — must persist (user-authored truth)
- **Two types**:
  - **RelativeOverride**: Keyed to `(source_block_id, event_id)`. Carries forward on re-run if event exists; becomes orphaned if it doesn't.
  - **AbsoluteOverride**: User-created event. Always persists regardless of re-runs.
- **Reconciliation**: When a source block re-runs, `reconcile(block_id, current_event_ids)` reports which overrides carried forward and which became orphaned. Orphans are surfaced to the user but never silently dropped.
- **Key methods**: `add_relative()`, `add_absolute()`, `remove()`, `reconcile()`, `get_orphans()`, `clear()`

### 4. **RunTracker** (Coordinator + ExecutionContext)
- **What**: In-flight execution state (running tasks, cancellation tokens)
- **Durability**: Transient — never persisted, discarded on shutdown
- **Lifespan**: Exists only during `Coordinator.request_run()`
- **Key components**: 
  - `is_executing` property on Coordinator
  - `cancel_event` (threading.Event) on ExecutionContext
  - Cleared after execution completes

---

## Architecture: 3 Communication Channels

State changes flow through three separate channels, each with different delivery semantics:

### 1. **MutationChannel** (Synchronous, Reliable, Ordered)
- **Carries**: User mutations (graph edits, override edits)
- **Implementation**: Pipeline + EventBus (DocumentBus)
- **Semantics**: Commands dispatch → handlers mutate → events collect → atomic flush on success or discard on failure
- **Events published**: `BlockAddedEvent`, `ConnectionRemovedEvent`, `SettingsChangedEvent`, `BlockStateChangedEvent`
- **Guarantee**: All-or-nothing. Zero events escape if operation fails.

### 2. **ResultsChannel** (Asynchronous, Reliable, Per-Block)
- **Carries**: Block execution outputs
- **Implementation**: ExecutionEngine.run() → Coordinator.request_run() → ExecutionCache
- **Semantics**: Engine publishes outputs to cache immediately after each block succeeds. Coordinator cascades cache invalidation on failure.
- **Guarantee**: Exactly-once delivery per block per execution_id.

### 3. **ProgressChannel** (Asynchronous, Lossy, Streaming)
- **Carries**: Execution lifecycle reports and progress updates
- **Implementation**: RuntimeBus (was ProgressBus)
- **Reports**:
  - `ExecutionStartedReport(block_id, execution_id)`
  - `ProgressReport(block_id, phase, percent, message)`
  - `ExecutionCompletedReport(block_id, execution_id, success, error?)`
- **Semantics**: Fire-and-forget. Subscribers see updates in real-time. Dropped updates don't cause correctness issues.
- **Guarantee**: Best-effort. High-frequency updates may be lossy; terminal message (completion) is always delivered.

---

## Core Loop: 3 Operations (No Phases)

The system has **no modes** (Manual/Auto/Live). Instead, editing and evaluating are concurrent activities with an invalidation protocol. Three event handlers implement the entire reactive system:

### 1. **on_mutation(change) → Coordinator.propagate_stale()**
```python
# When user edits graph structure or settings:
1. Apply to GraphStore or OverrideStore
2. dirty = compute_invalidated_nodes(change)
3. cancel(dirty ∩ running)  # Invalidate in-flight work
4. mark_dirty(dirty)         # Set BlockState.STALE
5. launch(ready_nodes(...))  # Queue execution of ready blocks
```

Called by:
- `Pipeline` handlers for structural commands
- `SettingsChangedEvent` subscribers in auto mode
- `ConnectionAddedEvent`, `ConnectionRemovedEvent` subscribers in auto mode

### 2. **on_result(block_id, outputs) → Coordinator.request_run() completion**
```python
# After block execution succeeds:
1. store in CacheStore (per-port)
2. reconcile OverrideStore (carry forward or orphan)
3. mark_fresh(block_id)      # Set BlockState.FRESH
4. launch(ready_nodes(...))  # Queue execution of newly-ready blocks
```

### 3. **on_cancel(block_id) → Coordinator.cancel()**
```python
# User cancels or invalidation requires restart:
1. set(cancel_event)
2. discard partial results (outputs already cached stay)
3. downstream blocks stay dirty (not cancelled)
4. on next run, only stale blocks re-execute
```

---

## Scheduling Function: `ready_nodes()`

A pure function that computes which blocks are ready to execute:

```python
def ready_nodes(graph, dirty, running, cache) -> set[str]:
    """Return block IDs whose upstream dependencies are all satisfied."""
    # A block is ready when:
    # 1. It IS in the dirty set (needs execution)
    # 2. It is NOT currently running
    # 3. All upstream blocks are NOT dirty and NOT running
    # 4. All connected input ports have cached upstream outputs
```

This is the **entire scheduler**. No entity class, no state — just a function of (graph, dirty, running, cache). Called after every state change.

---

## Data Flow: ExecutionContext + Multi-Port Outputs

### Single-Port (Default)
```python
# Executor returns a single value
executor.execute(block_id, context) -> Result[AudioData]
# Stored to first output port name (fallback: 'out')
context.set_output(block_id, 'audio_out', audio_data)
```

### Multi-Port
```python
# Executor returns a dict mapping port names to values
executor.execute(block_id, context) -> Result[dict[str, EventData]]
# {
#    'drums_out': EventData(...),
#    'bass_out': EventData(...),
#    'vocals_out': EventData(...)
# }
# Each port stored separately:
context.set_output(block_id, 'drums_out', ...)
context.set_output(block_id, 'bass_out', ...)
context.set_output(block_id, 'vocals_out', ...)
```

### Type Checking at Boundaries
- **Consumer side**: `context.get_input(block_id, port_name, expected_type=int)` — raises `ExecutionError` on type mismatch
- **Producer side**: `PORT_TYPE_MAP` checks block's declared output port types after execution

---

## Auto-Evaluation (Optional)

When `Coordinator.auto_evaluate = True`:

1. `subscribe_to_document_bus(document_bus)` registers handlers for:
   - `SettingsChangedEvent` → `propagate_stale()` → `request_run()`
   - `ConnectionAddedEvent`, `ConnectionRemovedEvent` → `propagate_stale()` → `request_run()`
   - `BlockRemovedEvent` → cache invalidation

2. Changes are debounced (100ms window) to collapse rapid edits into single run

3. `unsubscribe_from_document_bus()` disables auto-evaluation

Manual mode (default): User calls `coordinator.request_run()` explicitly.

---

## File Structure

```
echozero/
├── errors.py              # Exception hierarchy
├── result.py              # Result[T] type
├── commands.py            # Command types
├── event_bus.py           # DocumentBus (structural events)
├── pipeline.py            # Pipeline (command dispatch)
├── progress.py            # RuntimeBus + Report types
├── execution.py           # ExecutionEngine + GraphPlanner
├── cache.py               # ExecutionCache (outputs)
├── coordinator.py         # Coordinator (request_run, cancel, propagate_stale)
├── overrides.py           # OverrideStore (user edits)
├── domain/
│   ├── enums.py           # PortType, Direction, BlockState, BlockCategory
│   ├── types.py           # Block, Port, Connection, Event, Layer, AudioData, EventData
│   ├── graph.py           # Graph (topology + set_block_state)
│   └── events.py          # DomainEvent types (BlockAddedEvent, etc.)
├── processors/
│   └── load_audio.py      # LoadAudioProcessor (first block implementation)
└── ui/
    └── FEEL.py            # UI constants
```

---

## Key Design Decisions

### Why No Scheduler Class?
Scheduling is `ready_nodes()`, a pure function. State (dirty/running sets) lives on Graph and Coordinator. Calling `ready_nodes()` after every state change is sufficient — no entity class needed.

### Why Two Buses (DocumentBus + RuntimeBus)?
- **DocumentBus**: Transactional semantics. Commands collect events → all-or-nothing flush.
- **RuntimeBus**: Fire-and-forget semantics. Execution starts → progress streams → completion reported.

These have contradictory delivery guarantees. Mixing them on one bus forces compromise. Two buses encode the distinction in the type system.

### Why No Modes?
"Manual" and "Auto" aren't phases — they're just different event subscriptions. Same `request_run()` code path. Coordinator.auto_evaluate flips a boolean; DocumentBus event handlers either call `request_run()` or don't. No state machines, no mode switching logic.

### Why Per-Port Caching?
Blocks like SeparateStems have multiple output ports. Keying by `(block_id, port_name)` lets the engine track which ports are current and which are stale. Future optimization: invalidate only affected ports, not entire block.

### Why OverrideStore Doesn't Delete Orphans?
Axiom FP3: user edits are truth. Silently discarding them violates that axiom. Instead, `reconcile()` reports orphans; UI surfaces them to the user with a choice: keep, delete, or reattach to a new referent.

---

## Test Coverage

- **test_domain.py** (39 tests): Block/port/connection/graph invariants
- **test_event_bus.py** (12 tests): Subscription, delivery, reentrancy
- **test_execution.py** (30 tests): Planning, dispatch, cancellation, data flow
- **test_progress.py** (26 tests): RuntimeBus delivery and error handling
- **test_pipeline.py** (15 tests): Command dispatch, event collection/flush
- **test_load_audio.py** (9 tests): First processor implementation
- **test_cache.py** (16 tests): Store/get/invalidate/cascade
- **test_coordinator.py** (28 tests): ready_nodes, request_run, cancel, propagate_stale, auto-eval
- **test_overrides.py** (19 tests): Relative/absolute overrides, reconciliation
- **test_helpers.py** (7 tests): Utilities

**Total: 216 tests | 0.19 seconds | 100% pass rate**

---

## Next Steps

1. **UI Layer** — Timeline view + node graph editor consuming both buses
2. **Persistence** — Save/load Graph + OverrideStore; cache is discardable
3. **DetectOnsets Processor** — First "real" block (ML inference)
4. **Multi-Block Chains** — Test complex graphs with 5+ blocks
5. **Parallel Execution** — Swap sequential loop for concurrent.futures.ThreadPoolExecutor
6. **Undo/Redo** — Log mutations on MutationChannel, replay in reverse
7. **Execution History** — Transient subscriber tracking all runs for comparison
8. **Live Mode** — Real-time audio engine continuously feeding analysis pipeline

---

## References

- **Axioms**: See First Principles panel discussion (2026-03-22)
- **Precedent**: Houdini (dirty propagation), DAWs (separation of document + engine), Elm (unidirectional data flow)
- **Code**: All 216 tests serve as executable specification
