# Luna 🎨 — UX Advocate

## Position

The user staring at a 10-minute Demucs separation doesn't care about your event bus topology. They care about one thing: **"Is it working, and when will it be done?"** Everything in this design must serve that feeling. Progress isn't a technical feature — it's the difference between "this app is frozen" and "this app respects my time." I want rich, honest progress: a stage name ("Separating vocals…"), a percentage, and an ETA that doesn't lie. No indeterminate spinners when we *can* know. No fake percentages that sit at 99% for three minutes.

For stale propagation, the UX goal is **zero surprise**. If I tweak an EQ block upstream, I need to *immediately* see that my downstream blocks are outdated — not discover it when I export and the audio sounds wrong. Staleness should cascade visually through the entire chain the moment it happens. The graph should light up like a trail of dominoes. And yes, UPSTREAM_ERROR must be visually distinct from STALE — "your input changed" is informational, "your input is broken" is a warning. Collapsing those into one state is lying to the user.

On decomposition: I don't care how many classes you split the engine into, as long as the *experience* of FULL execution feels like one coherent action. When I hit "Run All," I want a single progress panel that shows me where I am in the chain, what's currently processing, and a meaningful aggregate percentage. Not five separate toasts. Not a log dump. One calm, informative view.

## Key Insight

**Progress events and domain events are fundamentally different UX contracts.** Domain events say "something happened" (past tense, committed). Progress events say "something is happening" (present tense, ephemeral). Trying to force progress through the post-commit event bus (D36) will either make progress lag behind reality or force you to break commit-then-publish semantics. Progress needs its own lightweight channel — a direct callback or observable stream that bypasses the domain event queue entirely. The UI doesn't need progress events to be durable or ordered — it needs them to be *fast*.

## Risk

If you treat progress as an afterthought or squeeze it into the domain event system, you'll get one of two outcomes: (1) laggy, batched progress updates that make the UI feel unresponsive during long operations, or (2) hacked-in special cases that rot the event system's guarantees. Either way, users will alt-tab away during long processes because the app gives them no reason to trust it's still working. For an audio app where ML inference is the *core value proposition*, that's a death sentence for perceived quality.

## Verdict

### Q1: Progress Event Architecture

**Use a dedicated `ProgressChannel` — not the domain event bus.** Progress is ephemeral, high-frequency, and fires mid-execution (before commit). It doesn't belong in the DomainEvent system.

Schema:

```python
@dataclass(frozen=True)
class ProgressReport:
    block_id: str
    execution_id: str
    stage: str              # "Loading model", "Separating vocals", "Writing output"
    fraction: float         # 0.0–1.0, NOT percentage — lets aggregator do math
    eta_seconds: float | None
    items: tuple[int, int] | None  # (completed, total) — e.g. (3, 12) audio chunks
    timestamp: float        # time.monotonic() for ETA computation
```

Reporting: pass a `report_progress(stage, fraction)` callback into every processor's `process()` method. Processors SHOULD report but aren't forced — if they don't, the UI shows an indeterminate state (honest > fake). Throttle at the *consumer* side (UI repaints at 15fps max), not the producer side.

Composite progress for FULL execution: the `ProgressAggregator` holds a weighted plan (block A = 5% of total, block B = 80%, block C = 15% — weighted by historical execution time). Each block's `fraction` maps into its slice. The user sees one smooth bar, not jumps.

Cancellation: the cancel button lives on the progress display. Clicking it sets the `CancellationToken` (D23). The next progress report from the processor should check the token and bail. Progress UI immediately shows "Cancelling…" state — don't wait for confirmation.

MCP/CLI: `ProgressChannel` is an observable stream. MCP subscribes to it. CLI prints a `tqdm`-style bar. No Qt dependency.

### Q2: Stale Propagation

**Immediate full-depth cascade. No lazy computation.**

- **Triggers:** Execution completion, settings change, connection added/removed. Anything that invalidates output.
- **Depth:** Full cascade, immediately. A→B→C: if A re-executes, both B and C go stale *right now*. Don't wait for B to run. The user needs to see the full blast radius.
- **UPSTREAM_ERROR:** Yes, split it. Three visual states: FRESH (green/default), STALE (amber/yellow — "needs re-run"), ERROR (red — "upstream is broken, can't run"). The distinction matters for user decision-making.
- **Storage:** Computed column or lightweight status field on the block row. Not a separate table — staleness is *the block's* state, not a relationship.
- **Workspace boundaries:** Staleness stops at Workspace blocks. They're manual-pull by design (D20). But show a subtle indicator on the Workspace block itself: "upstream has changes available" — like a badge, not an alarm.
- **UI:** Stale blocks get a desaturated/dimmed appearance with a small refresh icon overlay. Error blocks get a red border. On hover, tooltip says "Output is outdated — upstream block [EQ] was re-executed. Run to update."

### Q3: Execution Engine Decomposition

**Decompose, but hide it behind one `ExecutionService` facade.** The user-facing API is simple:

- `execute(block_id)` — single block
- `execute_full(block_id)` — run everything upstream + this block
- `cancel(execution_id)` — stop it

Internally, split into: **GraphPlanner** (what to run, in what order), **ExecutionDispatcher** (thread management, locks), **ProgressAggregator** (composite progress from individual reports), and **StalenessTracker** (propagation logic). The InputGatherer and OutputPersister aren't separate services — they're phases within the block execution flow (part of the PULL→EXECUTE→STORE sequence that the dispatcher orchestrates).

FULL execution orchestration lives in the `ExecutionDispatcher` — it gets a plan from `GraphPlanner`, executes sequentially, feeds each block's progress callback into `ProgressAggregator`, and stops on first failure (D27). One class coordinates the sequence; separate classes own the sub-problems.

Subprocess vs thread for ML: use thread for now, subprocess later if needed. Progress reporting via callback works for both (callback in-process, pipe/queue for subprocess). Don't over-engineer isolation before you need it.
