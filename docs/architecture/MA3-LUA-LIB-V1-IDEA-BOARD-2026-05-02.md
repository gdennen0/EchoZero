# MA3 Lua Lib V1 Idea Board

Status: draft
Last reviewed: 2026-05-02

## Intent

Capture clean-sheet concepts for a standardized MA3 Lua implementation.
This is an idea board, not a final spec.

The goal is to distill durable lessons from:

- `/Users/march/Documents/GitHub/MA3_Plugins`
- `MA3/` in this repo
- the live EZ plugin bundle on this machine

without inheriting old structure or ad hoc coupling by default.

## Framing

Treat the future MA3 implementation as a small Lua library with optional
services and extras, not as one growing monolithic plugin.

Core idea:

- `core` is tiny, boring, and reliable
- `services` expose a stable MA3-facing contract
- `extras` provide operator tools and UI sugar
- EchoZero should talk to stable services, not deep plugin internals

## Design Goals

- Keep the MA3 side as simple and light as possible
- Hide MA3 object-model weirdness behind a reusable library boundary
- Reduce wasted round trips and repeated structure queries
- Make disconnect/reconnect/reconcile behavior explicit
- Make async notifications an optimization, not a correctness requirement
- Support EchoZero without making the library EchoZero-specific
- Leave room for other clients, tools, and operator workflows

## MA3 Realities To Design Around

- `:Children()` is the real traversal API; direct child collection access is unreliable
- Marker tracks pollute internal indexing and must stay hidden from clients
- Hooks are useful but brittle
- Hook callbacks must persist
- Plugin reloads clear hook/runtime state
- MA3 and client process lifetimes are asymmetric
- Time storage/round-tripping is odd enough that exact matching is risky
- MA3 UI is powerful but expensive in complexity
- OSC is workable, but only with disciplined protocol and lifecycle rules

## Core Principles

### 1. Snapshot Is Truth

Snapshot queries are the correctness mechanism.

### 2. Invalidation Is A Hint

Async invalidation should improve freshness, but correctness must not depend on
never missing one.

### 3. Reconcile Is Mandatory

After disconnect, reconnect, plugin reload, or suspected drift, clients must be
able to re-read authoritative MA3 state and re-establish subscriptions.

### 4. Lua Owns MA3 Mechanics

Lua should own:

- object lookup
- traversal
- index normalization
- sequence assignment mechanics
- track preparation mechanics
- hook installation/removal

Clients should not need to understand MA3 internal object trivia.

### 5. Client Owns Product Policy

EchoZero or any other client should own:

- what to sync
- when to push/pull
- routing truth
- operator workflow
- persistence of app truth outside MA3

## Proposed Shape

### Core

Keep this layer tiny and reusable.

- safe `pcall` wrappers
- logging
- result/error envelope helpers
- object traversal helpers
- time conversion helpers
- pool/index normalization helpers
- hook registry/lifecycle helpers
- small cache primitives
- session/subscription bookkeeping

### Transport

Thin adapter layer only.

- OSC transport
- request/response envelope
- request id / session id
- heartbeat / health ping
- reconnect-safe semantics

### Services

Stable reusable modules.

- `health`
  - plugin status
  - active hooks
  - version/build/features
- `catalog`
  - timecodes
  - track groups
  - tracks
  - sequences
  - cues/events snapshots
- `mutation`
  - assign sequence
  - prepare track
  - write/delete/update events
  - create timecode/track/sequence where needed
- `subscription`
  - subscribe to narrow scopes
  - unsubscribe
  - emit invalidation hints

### Extras

Optional higher-level tooling.

- UI helpers
- selector popups
- progress dialogs
- SpeedOfLight-style operator tools
- recipe tools
- song/range conventions
- HitMaker-style sequence creation helpers

## Dependency Rules

- `core` must not depend on UI
- `services` may depend on `core`
- `transport` may depend on `core`
- `extras` may depend on `core`, `services`, and optional UI helpers
- app-specific logic should not live in `core`
- EchoZero should primarily use `services`

## Public API Direction

Keep the public API small.

Good candidates:

- `hello`
- `health.get`
- `catalog.list_timecodes`
- `catalog.list_track_groups`
- `catalog.list_tracks`
- `catalog.list_track_events`
- `catalog.list_sequences`
- `mutation.assign_track_sequence`
- `mutation.prepare_track`
- `mutation.write_events`
- `mutation.delete_events`
- `subscription.subscribe_track`
- `subscription.unsubscribe`
- `subscription.list`

Avoid a giant helper surface that leaks MA3 internals.

## Invalidations And Reconcile

MA3 does not magically know about all changes. It only knows what it is
explicitly watching through hooks or other narrow runtime state.

Implications:

- invalidation must be scoped
- subscriptions should be narrow and explicit
- subscriptions are session-scoped, not permanent truth
- reconnect must re-establish watched scopes
- plugin reload must be treated as subscription loss

Recommended rule:

`async notifications accelerate freshness; snapshot reconciliation guarantees correctness`

### Suggested Pattern

1. Client subscribes to a narrow scope like one track
2. Lua installs one hook on the relevant `CmdSubTrack` or similar object
3. Hook emits a lightweight invalidation signal
4. Client marks local cache stale
5. Client decides whether to refetch immediately, debounce, or wait until data is needed

Do not treat invalidation as a full diff stream.

## Standard Envelope Direction

All service traffic should converge on one consistent envelope shape.

Success shape:

```lua
{
  ok = true,
  request_id = "...",
  session_id = "...",
  kind = "catalog.track_events",
  data = { ... },
  error = nil
}
```

Failure shape:

```lua
{
  ok = false,
  request_id = "...",
  session_id = "...",
  kind = "mutation.prepare_track",
  error = {
    code = "no_sequence_assigned",
    message = "Track has no assigned sequence"
  }
}
```

## What To Distill From MA3_Plugins

Useful source material to convert, not preserve wholesale:

- traversal rules
- MA3 object-model learnings
- UI patterns
- sequence-range conventions where still useful
- selection/popup patterns
- command-building patterns
- proven utility helpers
- reusable operator workflows that belong in extras

## What Not To Preserve By Default

- historical folder structure
- legacy naming
- load-order-heavy architecture unless unavoidable
- EchoZero-specific assumptions in core
- giant multi-purpose plugin bundles
- UI-first design
- broad polling when narrow snapshot/query patterns will do

## Rough Package Sketch

```text
ma3lib/
  core.lua
  result.lua
  object.lua
  time.lua
  hooks.lua
  cache.lua
  session.lua
  transport/
    osc.lua
  services/
    health.lua
    catalog.lua
    mutation.lua
    subscription.lua
  extras/
    ui.lua
    sol_tools.lua
    hit_tools.lua
  service.lua
```

Possible facade direction:

```lua
MA3.health.get()
MA3.catalog.list_tracks(...)
MA3.catalog.list_track_events(...)
MA3.catalog.list_sequences(...)
MA3.mutation.prepare_track(...)
MA3.mutation.write_events(...)
MA3.subscription.subscribe_track(...)
```

## Open Questions

- How small can the initial service surface be while still being useful?
- Which operations should be standardized immediately versus left in extras?
- Should event writes be fully transactional or best-effort with verification?
- How much fingerprint/revision data should snapshots carry?
- Which subscription scopes are worth supporting in v1 besides track-level?
- How much of current SpeedOfLight/HitMaker logic belongs in extras versus nowhere?

## Best Next Artifacts

- `MA3 Lua Standard: Principles`
- `MA3 Lua Standard: Wire Protocol`
- `MA3 Lua Standard: Runtime and Subscription Model`
- migration notes mapping current EchoZero and SpeedOfLight concepts into the new layout
