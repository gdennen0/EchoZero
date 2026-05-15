# Audio Engine v2 Execution Plan

Status: active phased plan
Last verified: 2026-05-15

This plan rebuilds EchoZero playback in small tested phases. The live v1 engine
stays unchanged until parity and migration gates are met.

## Phase 1: Foundation, Non-Live

Scope:

- document the DAW-style backend architecture
- add immutable snapshot/generation models
- add prepared graph models for tracks, buses, master, route targets, and
  hardware outputs
- add explicit monotonic transport command/state models
- add deterministic graph identity helpers
- add compatibility mapping from current playback track plans

Gates:

- no live v1 runtime behavior changes
- no app wiring to v2 graph execution
- focused tests for immutability, copy-on-write, identity, routing semantics,
  mapping, and transport command reduction
- routing tests prove master-only, no-output, direct hardware, master plus
  direct hardware, and bus-to-bus route behavior
- stale/replayed transport commands cannot mutate state or move sequence
  backward
- targeted tests plus one broader import/test slice pass

Audit:

- confirm no generated artifacts are tracked
- confirm user-local files remain untouched
- confirm imports do not make UI, MA3, or Foundry part of the engine foundation

## Phase 2: Planner Parity Harness

Scope:

- build v2 prepared graphs from real app/runtime playback projections and
  existing `PlaybackTrackPlan`/`PlaybackMixPlan` shaped inputs in a shadow lane
- compare v1 playback track plans with v2 graph identities, route summaries,
  mix summaries, and structure signatures
- classify graph identity edits as unchanged, mix, route, or structure for
  developer diagnostics
- make unsupported route tokens observable as parity planning failures instead
  of silently falling back to an incorrect route
- keep diagnostics visible to developers only, not operator-facing by default

Gates:

- shadow planner runs without changing audible output
- route summaries match existing v1 behavior for representative projects
- app-boundary tests cover selected song, selected version, event-slice, mute,
  solo, gain, no-output, master, master plus direct output, and direct output
  cases
- bus-to-bus route coverage remains in the immutable graph foundation until a
  current app projection exposes subgroup buses

Audit:

- verify no callback code depends on timeline/UI objects
- verify failures are observable and do not silently fall back to wrong routes

## Phase 3: RT Graph Prototype, Still Non-Live By Default

Scope:

- add `RtGraph` preparation from `PreparedGraph`
- add callback-safe command queues
- add atomic committed-generation reference
- add offline render harness for deterministic block rendering

Gates:

- offline block renders are deterministic
- callback path performs no allocation or locking in steady state
- route and mix commands apply only at block/sample boundaries
- seek and stop behavior is covered by click/discontinuity tests

Audit:

- run performance guardrails for graph preparation and block rendering
- inspect command queue overflow behavior and telemetry

## Phase 4: App Shadow Runtime

Scope:

- run v2 runtime in parallel with v1 for selected local developer sessions
- emit generation-aware IPC telemetry
- compare first-block timing, route summaries, and transport state against v1

Gates:

- no operator-visible behavior change unless an explicit developer flag is set
- generation telemetry proves submitted -> prepared -> committed ->
  rendered-first-block
- app human-path smoke covers import, select, play, pause, seek, stop, and route
  edits

Audit:

- verify shadow runtime cannot drive MA3 sync
- verify v2 failures cannot corrupt v1 playback state

## Phase 5: Opt-In Live Runtime

Scope:

- add a developer/operator opt-in switch for v2 playback
- keep v1 fallback available
- migrate selected runtime diagnostics to generation-aware reporting

Gates:

- app-boundary playback tests pass in v1 and v2 modes
- no-output, master, and direct hardware routes are proven on supported devices
- transport command ordering is proven through IPC and local runtime paths
- packaging smoke accounts for the new runtime modules

Audit:

- compare CPU, latency, glitch count, and route behavior against v1
- document known hardware limitations

## Phase 6: Default Runtime Migration

Scope:

- make v2 the default after opt-in soak
- keep v1 behind a temporary fallback flag
- update operator-facing diagnostics and troubleshooting docs

Gates:

- parity suite passes on supported platforms
- performance guardrails are no worse than accepted thresholds
- human-path demo proof uses real EZ runtime actions and real input assets
- rollback path is documented

Audit:

- release-affecting packaging and smoke checks pass
- stale v1-only diagnostics are removed or clearly marked

## Phase 7: v1 Retirement

Scope:

- remove v1 backend code only after migration confidence
- remove compatibility shims that no longer serve active tests
- simplify docs around the v2 architecture as current truth

Gates:

- no test or app path imports retired v1 modules
- all current playback docs point to v2 current truth
- packaging and app smoke pass after removal

Audit:

- search for dead route tokens and stale telemetry names
- confirm MA3 sync remains main-only and application-owned

## Migration Rules

- Additive first, replacement later.
- A phase cannot claim success without tests for its own invariants.
- Live runtime changes require app-path proof.
- Generated outputs and local runtime state stay out of git.
- Cross-lane findings become queued follow-up work unless they block playback
  correctness in the active phase.
