# Audio Engine v2

Status: draft
Last verified: 2026-05-15

Audio Engine v2 is the target DAW-style playback backend for EchoZero. This
document defines the architecture that replaces the current split playback
state over time. Phase 1 adds foundation models only; it does not replace the
live v1 runtime.

## Problem

Current playback state is distributed across timeline presentation, sync
projection, decoded track plans, mutable engine tracks and mixer parameters,
clock/transport state, and IPC telemetry. That makes routing and transport
changes hard to reason about because the audio callback can observe a partial
state transition.

The v2 invariant is:

> Playback renders exactly one committed immutable `PlaybackSnapshotGeneration`.

Every audible block must be explainable by the generation that produced it.
Non-real-time code may build, validate, and prepare a successor generation, but
real-time code only swaps committed references at block or sample boundaries.

## Ownership

- Application playback owns snapshot submission and user-facing state.
- The non-real-time planner owns `PreparedGraph` construction and validation.
- The real-time engine owns callback-safe `RtGraph` execution.
- The transport lane owns explicit command ordering and state reduction.
- UI, editor, and MA3 semantics stay outside the engine.

## Core Model

`PreparedGraph` is immutable data:

- tracks: decoded or streamable sources with stable source keys
- buses: subgroup buses and a required master bus
- master: final mix bus before hardware output
- hardware outputs: explicit 1-based physical channel spans
- mix parameters: gain, pan, mute, solo, and future automation/ramp metadata
- routes: ordered route-target lists on tracks and buses

Route targets are explicit data:

- bus target: send downstream to a bus, including the master bus
- hardware target: send directly to one physical output span
- empty target list: no output

The default track route is `track -> master bus`. The master bus normally
targets hardware outputs. Direct physical output is explicit and bypasses the
master only when the graph says so. A route may target both master and hardware,
which preserves current route strings such as `master,outputs_3_3` as
`track -> master` plus `track -> output 3`.

Buses use the same route-target model as tracks. Subgroup routing is represented
as `track -> subgroup bus -> master bus -> hardware`. Bus validation rejects
missing bus targets, self-routes, and obvious bus route cycles.

`PlaybackSnapshotGeneration` binds:

- generation sequence
- prepared graph
- deterministic graph identity
- transport state
- reason/audit label

Future IPC v2 messages should be generation-aware:

- submitted
- prepared
- committed
- rendered-first-block
- rejected

## Graph Identity

Graph identity is split so planners can classify work:

- structural hash: tracks, sources, buses, channel layout
- route hash: track and bus route-target lists
- mix hash: gain/pan/mute/solo state
- full hash: complete render-relevant identity

Structural edits require a non-real-time graph rebuild. Route and mix edits may
be lowered into sample-boundary real-time commands when the active RT graph can
accept them safely. Identity must be deterministic across equivalent planner
inputs.

## Transport

Transport is an explicit command stream, not direct mutation:

- play
- pause
- stop
- seek
- set loop

Commands reduce immutable `TransportState` values. The RT engine may apply the
result at sample boundaries, but application code should not poke callback state
directly. Transport commands carry sequence numbers so IPC and telemetry can
prove ordering. Commands with a sequence less than or equal to the current
transport state's sequence are stale/replayed commands and are ignored. This
matches the RT command-stream requirement that late messages must not move
state backward.

## Real-Time Contract

The future callback contract is:

- never allocate in the callback
- never lock in the callback
- never inspect UI or timeline models
- atomically read the committed generation/RT graph reference
- apply queued RT-safe commands only at block or sample boundaries
- emit generation-aware telemetry without blocking audio

Mute, solo, gain, pan, route, and seek changes are command data. Audible
parameter changes should ramp where needed to avoid discontinuities.

## Compatibility Boundary

Phase 1 exposes mapping hooks from the current `PlaybackTrackPlan` shape into a
v2 `PreparedGraph`. Phase 2 adds a shadow parity harness from app/runtime
playback projections into those graph summaries. Phase 3 adds a non-live
`RtGraph` and offline renderer prototype. Phase 4 starts dev-gated live backend
wiring: `ECHOZERO_AUDIO_ENGINE=v2` or an explicit test parameter can select a
v2-backed runtime audio engine while v1 remains the default fallback. The first
v2 live adapter opens the same output backend contract as v1 and renders callback
blocks through `PreparedGraph -> RtGraph -> render_offline_block`; MA3, IPC
payload shape, and operator UI playback behavior stay out of scope.

Phase 4b hardens the selected live path without making v2 the default. Mix-only
runtime edits lower to RT track-mix commands and ramp in the renderer; route
edits still commit a prepared graph and use the existing graph crossfade.
Seek, pause, stop, preview overlay, device reconfiguration, and process-service
controller construction now have v2-selected tests with fake streams/backends so
CI does not require real speakers.

Phase 4c adds a real-project smoke path for local developer fixtures. It opens a
developer-supplied `.ez` through `ProjectStorage`, builds the runtime
`TimelinePresentation` through the same app composition used by the Qt shell,
syncs the selected playback plan into the v2 live adapter, and renders fake
output callbacks through play, seek, gain, mute, route, pause, stop, and preview
when an event-backed preview clip is available. The project file stays outside
git and CI; the archive is unpacked only into a temporary working root.

```bash
ECHOZERO_AUDIO_ENGINE=v2 \
ECHOZERO_REAL_PROJECT_SMOKE=/path/to/local-project.ez \
uv run python scripts/audio_engine_v2_real_project_smoke.py
```

Multiple projects can be passed as positional arguments:

```bash
ECHOZERO_AUDIO_ENGINE=v2 uv run python scripts/audio_engine_v2_real_project_smoke.py \
  /path/to/first.ez /path/to/second.ez
```

The pytest wrapper is intentionally skipped unless the private fixture path is
provided:

```bash
ECHOZERO_AUDIO_ENGINE=v2 \
ECHOZERO_REAL_PROJECT_SMOKE=/path/to/local-project.ez \
uv run pytest tests/ui/test_audio_engine_v2_real_project_smoke.py
```

This smoke uses a fake output backend and does not play speaker audio. For a
bounded human-path listen, keep monitor volume low and run the desktop app
manually with `ECHOZERO_AUDIO_ENGINE=v2 uv run python run_echozero.py`, then open
the local project and try play, seek, mute/gain/route changes, stop, and preview.
Do not use the fake-output smoke as proof of hardware-device behavior.

Manual bounded smoke for local listening:

```bash
ECHOZERO_AUDIO_ENGINE=v2 uv run --with pytest pytest tests/ui/test_runtime_audio_v2_live_backend.py
```

For a human-path app listen, launch the desktop app with
`ECHOZERO_AUDIO_ENGINE=v2 uv run python run_echozero.py`, open a small project or
import one audio file, play for a few seconds, try mute/gain/output-route
changes, seek, pause, stop, and preview an event clip. Keep monitor volume low;
automated tests use fake output and do not prove speaker hardware behavior.

## Phase 3 Prototype

The Phase 3 RT graph prototype lives under
`echozero/application/audio_engine_v2`:

- `rt_graph.py` lowers immutable `PreparedGraph` values into index-addressed
  `RtGraph` track and bus nodes with pre-resolved route targets.
- `rt_commands.py` defines bounded `RtCommandBatch` values and immutable
  `RtRuntimeState` reduction for graph commits, transport commands, and track
  mix edits. Commands apply at render block boundaries and stale/replayed
  sequences are reported without moving state backward.
- `offline_render.py` provides deterministic numpy block rendering against
  immutable `OfflineSourceBank` fixtures and preallocated `OfflineRenderMemory`.
  It supports track, subgroup bus, master, direct hardware, and master plus
  hardware sends across mono/stereo and explicit hardware channel spans.
- `TransitionPolicy` centralizes the v2 declick ramp foundation. The offline
  renderer uses it for block-boundary gain, mute, and transport stop
  transitions.

This is still non-live by design. Phase 3 does not import or replace
sounddevice, the live v1 `AudioEngine`, UI playback controls, MA3 sync, or IPC.
It proves render semantics in tests only.

## Non-Goals For Phase 1

- replacing the live v1 engine
- changing current routing behavior
- changing IPC payloads
- adding a real-time callback implementation
- changing MA3 sync behavior
- changing Foundry behavior

## Non-Goals For Phase 3

- driving hardware audio
- replacing or wrapping current v1 runtime playback
- adding UI, MA3, or IPC integration
- claiming callback no-allocation performance for Python/numpy internals
- porting current v1 tail-correction behavior into v2

## Required Proof

Any phase that touches live playback must prove behavior through the real app
path, not only helper tests. Sync-facing work also needs app-boundary guardrails.
Hot-path work needs performance guardrails before it can be considered done.
