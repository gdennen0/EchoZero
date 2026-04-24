# MA3 Push V1 Execution Plan

_Last updated: 2026-04-23_

## Goal

Build a clean MA3 push-first operator workflow around the real job:

1. Route an EZ layer to an MA3 track.
2. Push either all main events from that layer or only selected main events.
3. Choose merge or overwrite.
4. Leave pull, batch planning, and section routing for later milestones.

This milestone is intentionally not “generic sync.” It is a focused commit workflow from EchoZero main-take drum data to MA3.

## Scope Note

This is the active push-lane plan.

For current repo truth on pull workspace behavior, song-version MA3 timecode
pool defaults, and transfer routing defaults, use:

- `docs/STATUS.md`
- `MA3/README.md`

The older `docs/MA3-MANUAL-PUSH-PULL-IMPLEMENTATION-PLAN.md` is now historical
background, not the active execution doc.

## Locked Constraints

- Main is truth.
- Takes are subordinate and never become MA3 truth.
- MA3 sync is main-only.
- Persistent layer-to-track routing is required.
- One-shot reroute must not overwrite the saved route unless the operator explicitly routes the layer.
- UI must stay on the application contract and must not bypass app-layer routing or push semantics.

## V1 Operator Contract

Each eligible event layer should expose these actions:

- `Route Layer to MA3 Track`
- `Send Layer to MA3`
- `Send Selected Events to MA3`
- `Send to Different Track Once`

Each send uses exactly:

- scope:
  - `Send Layer Main`
  - `Send Selected Events`
- target:
  - `Use Saved Route`
  - `Send to Different Track Once`
- apply mode:
  - `Merge`
  - `Overwrite`

## Execution Order

### Slice 1 — Push command and saved route foundation

Status: implemented on 2026-04-22

- Add one typed app-layer push command with:
  - `layer_id`
  - `scope`
  - `target_mode`
  - `apply_mode`
  - `target_track_coord` only when target mode is one-shot
  - `selected_event_ids` only when scope is selected events
- Add one explicit saved-route intent for `layer -> ma3_track_coord`.
- Add one lightweight refresh intent for MA3 push track options.
- Enforce main-only push selection in the app layer.
- Keep this slice additive on top of the existing sync adapter.

### Slice 2 — Layer-local UI rewrite

Status: implemented on 2026-04-22

- Rework header and inspector actions around layer-local send actions.
- Stop using batch transfer planning as the primary operator path.
- Remove pull from the primary visible MA3 UX for this milestone.
- Keep the track picker application-backed by refreshing MA3 track options through the app boundary.

### Slice 3 — Overwrite confirmation

Status: implemented on 2026-04-22

- Add a compact overwrite confirmation with:
  - target track summary
  - selected event count
  - target event count when available
  - concise diff summary
- Keep merge fast.
- Require confirmation before overwrite apply.

### Slice 4 — Real hardware validation

Status: implemented on 2026-04-22

- Reconcile the production transport with the real EZ1 hardware path:
  - outbound OSC path is `"/cmd"`
  - payloads are wrapped as `Lua "EZ...."`
  - MA3 replies remain on `"/ez/message"`
- Verify hardware ping against real MA3 at `192.168.1.70` using the production-style transport.
- Update the production bridge, runtime config defaults, simulator, and dev probe to follow the real MA3 command path.
- Prove the operator flows on real MA3 hardware:
  - full-layer push to `tc2_tg1_tr1` with `215 -> 215`
  - selected-events push to `tc113_tg1_tr8` with `3 -> 3`
  - overwrite on `tc113_tg1_tr8` with `2 -> 2`
  - one-shot reroute to `tc2_tg2_tr1` with `4 -> 4`
  - saved route remains `tc2_tg1_tr1` after one-shot send
- Add bridge-side recovery for first-write `No CmdSubTrack` failures by issuing
  `EZ.CreateCmdSubTrack(...)` once and retrying the first event.

### Slice 5 — Push v1 stabilization

Status: in progress on 2026-04-22

- Keep MA3 push on the app/sync boundary. Do not add ad hoc alternate write paths.
- Manual sequence assignment/creation and MA3 track prep for unassigned targets are now implemented through the app/sync boundary.
- Keep MA3 sequence assignment as an explicit operator choice for empty targets.
- Let the bridge repair missing `CmdSubTrack` only after a sequence is already assigned.
- Add or tighten regression coverage whenever the real hardware uncovers a failure mode.
- Finish the remaining proof work through the canonical shell surface.

## What Completes The First Task

The first task is complete when all of this is true:

- A layer can persist a saved MA3 route by stable layer id.
- The app contract has a typed push command for main-only send.
- The app can refresh available MA3 push targets without entering batch push mode.
- The push command can send either all main events or selected main events.
- The push command can use either the saved route or a one-shot target.
- The sync adapter boundary remains the only execution path for MA3 push.

## Proven On Real MA3

The following is now proven on the real console at `192.168.1.70`:

1. Full-layer push from the live `Kick` main take to `tc2_tg1_tr1` lands `215` events and matches source count.
2. Selected-events push lands the exact selected count on a sacrificial target.
3. Overwrite replaces prior target content rather than appending.
4. One-shot reroute lands on the alternate target and leaves the saved route unchanged.
5. The production transport shape is correct for this hardware:
   `/cmd` with `Lua "EZ...."` outbound and `/ez/message` inbound.

## Current V1 Prerequisite

For an empty MA3 target track to be safely writable in v1:

1. the track must end up with an assigned MA3 sequence before write
2. EchoZero may satisfy that by:
   - assigning an existing sequence
   - creating a new next-available sequence
   - creating a new sequence in the current-song range
3. EchoZero may then create the missing `CmdSubTrack` automatically on first write

EchoZero still persists only `layer -> track coord`.
MA3 sequence numbers remain MA3-side preparation state, not EchoZero routing truth.

## What Still Remains Before Push V1 Is Stable

1. Run one canonical shell-path smoke on the real UI surface against MA3.
2. Re-run the real-MA3 proof lane against sequence-prepped empty targets.
3. Keep the bridge recovery path and simulator aligned with any new real-hardware failures.
4. Prune or quarantine the remaining legacy batch/pull entry code once the push-first path is trusted.

## Next Design Slice

The next implementation slice after push-v1 stabilization should tighten pull
selection, routing defaults, and song-version timecode-pool ergonomics around
the existing transfer foundation.

See:

- `docs/architecture/MA3-SEQUENCE-MANAGEMENT-DESIGN-2026-04-22.md`

That design note is now primarily a reference for the implemented sequence-prep
contract and future extensions of the same boundary.

## Next Milestone After Push V1

After push v1 is stable, add `Section` and `Recipe` on top of the existing
saved-route foundation so the current route model becomes the future global route
instead of being replaced.
