# Audio Engine v2

Status: draft implementation spec
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
v2 `PreparedGraph`. These hooks are for tests and later migration only. They
must not be threaded into the live v1 engine until a later phase proves parity
through the app path.

## Non-Goals For Phase 1

- replacing the live v1 engine
- changing current routing behavior
- changing IPC payloads
- adding a real-time callback implementation
- changing MA3 sync behavior
- changing Foundry behavior

## Required Proof

Any phase that touches live playback must prove behavior through the real app
path, not only helper tests. Sync-facing work also needs app-boundary guardrails.
Hot-path work needs performance guardrails before it can be considered done.
