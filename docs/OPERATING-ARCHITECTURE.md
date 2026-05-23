# EchoZero Operating Architecture

Status: active
Last verified: 2026-05-18


This is the operating architecture bible for EchoZero after the first major
feature buildout.

It answers one question:

How must app operations move through the system so that EchoZero stays correct,
responsive, debuggable, and provable as the codebase matures?

Use this document when designing or refactoring cross-cutting behavior,
especially playback, transport, timeline mutation, sync, import, review,
pipeline execution, and UI automation.

This document does not replace `docs/STATUS.md`, `docs/ARCHITECTURE.md`,
`docs/UNIFIED-IMPLEMENTATION-PLAN.md`, or subsystem READMEs. It defines the
operating rules those surfaces should converge toward.

## Source Of Truth

When this document conflicts with current implementation, the implementation is
the truth for debugging. For future work, this document is the target shape
unless a specific decision record says otherwise.

Authority order:

1. Current code and tests
2. `docs/STATUS.md`
3. `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
4. `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
5. This document
6. Subsystem plans and historical notes

## North Star

EchoZero is an app-first, timeline-first desktop system.

Every meaningful behavior must have:

- one truth owner
- one command path
- one observable state contract
- one bounded performance expectation
- one proof lane that exercises the real app boundary

If a feature cannot name those five things, it is not architecturally done.

## Operating Principles

### 1. UI Never Blocks On Runtime Work

The UI may render cached state and submit commands. It must not wait on audio
runtime, file IO, analysis, sync transfer, diagnostics, or long app assembly in
paint/input/timer hot paths.

Allowed:

- read an in-memory snapshot
- enqueue an operation
- optimistically update a local visual preview when the app contract allows it

Not allowed:

- synchronous IPC from a 16 ms UI timer
- audio decode on a click path
- full presentation assembly on every playback tick
- network or hardware transfer from widget event handlers

### 2. Commands Are Not State

A command says what the operator wants. State says what the system has accepted,
prepared, applied, or emitted.

Every long-running or runtime-adjacent command should expose state through a
small operation record:

- `queued`
- `preparing`
- `ready`
- `applying`
- `applied`
- `failed`
- `cancelled`

Widgets should render state. They should not infer state by poking runtime
objects.

### 3. Truth Lives Above Widgets

Widgets are surfaces, not authorities.

Application services own:

- timeline truth
- transport truth
- selection truth
- pipeline operation truth
- sync operation truth
- persistence truth

Widgets may keep local ephemeral state for pointer drag, hover, keyboard focus,
scroll position, and visual preview. They must commit durable behavior through
application contracts.

### 4. Runtime Lanes Are Explicit

Any operation that touches playback, external sync, pipeline execution, import,
or review must name its lane.

The default lanes are:

- `ui`: paint, input, visual preview, local cached snapshots
- `app`: command validation, truth mutation, presentation assembly
- `realtime`: audio callback and sample-time state only
- `transport`: play, pause, stop, seek, timing snapshots, external clock input
- `prepare`: decode, build, render, analyze, preflight, diagnostics
- `persist`: SQLite/archive writes and project runtime state
- `sync`: MA3 push, pull, receive, and protocol translation
- `automation`: app-boundary automation and proof harness control

Work may cross lanes only through a named command, queue, snapshot, or adapter.

### 5. Expensive Work Is Prepared, Not Discovered

An operator action should not discover expensive work at the moment the operator
expects feedback.

Preferred shape:

1. Detect desired state.
2. Prepare the expensive runtime artifact in the background.
3. Publish readiness with a generation id.
4. Apply atomically when the operator acts or when the app contract says it is
safe.

This is especially important for playback graphs, waveform assets, analysis
outputs, video/reference media, sync plans, and import batches.

### 6. Latest Intent Wins For Repeated Controls

High-frequency controls must coalesce.

Examples:

- scrub drag
- playhead drag
- wheel zoom
- gain slider
- routing changes
- selection navigation
- external transport updates

The system should preserve the latest intended value, not process every
intermediate value as a full app operation.

### 7. Real-Time Code Reads Immutable State

The audio callback and other real-time paths must not share mutable structures
with command threads.

Allowed:

- immutable graph snapshots
- atomic generation swaps
- callback-local smoothing state
- preallocated scratch buffers

Not allowed:

- file IO
- locks on the callback path
- mutating shared envelopes from both callback and command paths
- allocating large arrays during steady-state playback

### 8. Main Is Truth

Main remains the only live truth lane for playback, export, sync, freshness, and
operator-facing state.

Takes are subordinate history and candidates. They may be previewed, compared,
promoted, or merged. They must not become alternate live truth.

### 9. Proof Follows The Human Path

App-facing behavior is done only when it is proven through the canonical app
path.

Contract tests are necessary but not sufficient for operator-facing behavior.
Widget-only tests are useful but not sufficient for app-facing truth.

## Canonical Operation Shape

Every significant operation should fit this model:

```text
operator/input
  -> UI event or automation command
  -> typed application intent
  -> application validation and truth owner
  -> operation state record
  -> lane-specific executor or adapter
  -> generated snapshot/result
  -> atomic app truth update
  -> presentation/snapshot emission
  -> UI render from cached state
```

If a feature skips the application intent or truth owner, it is probably a
widget-local workflow bug.

If a feature skips operation state, it is probably hard to debug.

If a feature skips a proof lane, it is probably fragile.

## Playback And Transport Target Architecture

Playback must feel like hardware: immediate, stable, and boring.

### Ownership

- Application owns desired playback graph state.
- Playback runtime owns prepared graph generations and sample-time transport.
- UI owns only visual playback affordances and cached transport display.

### Target Lanes

```text
UI
  -> nonblocking transport command queue
  -> playback coordinator
       -> high-priority transport executor
       -> lower-priority graph preparation executor
       -> immutable playback graph store
       -> pushed transport snapshot stream
  -> UI transport snapshot cache
```

### Hard Rules

- Play, pause, stop, and seek are high-priority transport commands.
- UI ticks must not block on playback IPC.
- Timing display reads the latest cached snapshot.
- Scrub and drag commands coalesce; latest seek wins.
- Graph preparation is cancellable and generation-based.
- Timing reads must not apply pending structure rebuilds.
- Structure swaps must be atomic from the audio callback perspective.
- Preview decode and diagnostics writes must not block transport commands.

### Prepared Playback Contract

A song/version should expose a prepared playback generation:

- `generation_id`
- `source_signature`
- `mix_signature`
- `prepared_at_monotonic`
- `state`: queued, preparing, ready, stale, failed
- `diagnostics`: build time, decode time, source count, cache status

Play should start a ready generation. If no ready generation exists, the app
must choose explicitly:

- block with visible arming state
- play the last valid generation
- reject play with a clear unavailable reason

Silent blocking is not allowed.

## Timeline Mutation Target Architecture

Timeline mutation is application-owned.

The UI may preview:

- drag position
- resize position
- playhead hover/drag
- selection marquee
- video trim handles

The app must own:

- event create/update/delete
- layer/take selection identity
- stale/fresh/manual state
- section cues
- playback source selection
- sync-affecting event metadata

Hot-path timeline rules:

- pointer moves coalesce
- paint reads presentation snapshots only
- layout recomputation is explicit
- full presentation rebuilds do not happen on every visual tick
- follow-scroll must not fight manual scroll

## Sync Target Architecture

MA3 sync is main-only and adapter-bound.

Target shape:

```text
timeline app truth
  -> sync application contract
  -> sync plan / diff
  -> concrete MA3 adapter
  -> operation progress
  -> app-boundary result
```

Hard rules:

- UI does not decide sync truth.
- Non-main takes do not sync directly.
- Malformed required event metadata fails hard.
- Push, pull, and receive expose operation progress.
- Protocol fixtures cover current real MA3 payload shapes.

## Persistence Target Architecture

Persistence is a lane, not a side effect hidden inside widgets.

Hard rules:

- SQLite writes go through `ProjectStorage` and repositories.
- Runtime state persistence must be deferred or coalesced during live playback
  and external transport churn.
- Generated artifacts and local runtime state stay out of git.
- Autosave must not block UI input or playback hot paths.

## UI Target Architecture

The UI is a projection of app state plus ephemeral interaction state.

It may own:

- hover
- focus
- local drag preview
- viewport scroll/zoom
- transient menu/dialog state
- cached render geometry

It must not own:

- project truth
- playback truth
- sync truth
- pipeline result truth
- import truth
- review truth

UI controls should be wired to real commands before they are visible in the
operator path.

## Operation State Requirements

The public operation contract lives in `echozero/application/operations.py`.

Every long-running operation should have:

- stable operation id
- kind
- source command
- current phase
- progress when known
- started/updated timestamps
- user-visible status message
- cancellability
- final result
- diagnostics payload for failures

This applies to:

- import
- pipeline/object actions
- playback graph preparation
- diagnostics capture
- MA3 push/pull/receive
- export
- review batch operations
- packaging or release smoke helpers

## Performance Budgets

Budgets should become executable guardrails where possible.

Initial target budgets:

- UI input handler: under 4 ms typical, never waits on IPC or file IO
- active playback UI tick: under 4 ms typical
- cached transport display update: under 1 ms
- play/pause/stop command enqueue: under 10 ms
- prepared play apply: under 20 ms typical
- scrub update enqueue: under 4 ms, coalesced
- full timeline paint under dense data: guarded by benchmark thresholds
- playback graph preparation: asynchronous, measured, cancellable
- sync push/pull: progress-backed, never blocks paint/input

When a budget is violated, prefer changing the operation shape over raising the
threshold.

## Observability Requirements

Every runtime-adjacent subsystem should expose cheap diagnostics:

- latest command kind
- latest command latency
- queue depth
- dropped/coalesced command count
- active generation id
- stale generation reason
- last failure
- recent bounded event ring

Diagnostics must be cheap to sample. Expensive diagnostics capture is a
separate operation and must not block transport or UI hot paths.

## Refactor Rules

When refactoring an existing feature, ask:

1. Who owns the truth?
2. What is the command?
3. What state does the UI render?
4. Which lane does the work run in?
5. Can repeated inputs coalesce?
6. What is the performance budget?
7. What proof lane catches regressions?

Do not accept "the widget calls a helper" as an architecture answer.

Do not add another fallback path when the real problem is unclear ownership.

Prefer one small explicit operation contract over several implicit callbacks.

## Proof Matrix

Use the smallest proof that covers the changed boundary, then expand when the
change touches broader operator behavior.

| Change area | Required proof |
| --- | --- |
| Pure model/domain | targeted pytest |
| Timeline application truth | targeted application tests |
| Timeline UI behavior | targeted UI tests plus app-flow lane |
| Playback/transport | runtime audio tests, app-flow proof, perf/latency guardrail |
| Sync/MA3 | app-boundary sync tests plus sync/protocol lanes |
| Import/pipeline operation | operation progress tests plus app-flow proof |
| Release-affecting shell behavior | app-flow plus packaged smoke consideration |
| Human-path demo claim | real launcher, real runtime actions, real input assets |

## Red Flags

Treat these as architecture smells:

- widget-only workflow logic
- synchronous IPC in UI timers or event handlers
- timing poll that mutates runtime state
- audio callback reading command-thread mutable state
- command path that secretly decodes files
- operation with no id or progress state
- status text derived from UI guesses
- tests that prove only helper behavior for an app-facing claim
- broad fallback chains instead of one clear contract
- FEEL constants scattered outside `echozero/ui/FEEL.py`

## Immediate Convergence Targets

The current codebase should converge toward this architecture in small slices:

1. Move playback timing display to pushed/cached snapshots.
2. Coalesce seek and scrub commands.
3. Remove playback sync classification from pure transport intents.
4. Make prepared playback graph state explicit.
5. Stop timing reads from applying structure rebuilds.
6. Isolate mixer/callback state from command-thread mutation.
7. Move remaining widget-owned workflow truth into app contracts.
8. Add latency guardrails for transport controls and runtime ticks.
9. Promote operation progress to all long-running app-facing operations.
10. Keep hot-file decomposition aligned with responsibility boundaries.

This is intentionally incremental. The goal is not a heroic rewrite. The goal is
to make the correct operation shape the easiest shape to implement.
