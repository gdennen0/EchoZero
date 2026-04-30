# MA3 Sequence Management Design — 2026-04-22

Status: reference
Last reviewed: 2026-04-30


## Status And Intent

This is the design note for the next MA3 push lane slice after transport and
push-path validation.

It defines how EchoZero should handle MA3 sequence assignment and creation for
push targets without inventing a second truth model beside the existing saved
route contract.

## Problem

MA3 push v1 is proven when the target track already has an assigned sequence.

The next operator problem is what to do when:

- a target track has no assigned sequence
- the operator wants to reuse an existing sequence
- the operator wants EchoZero to create a new sequence
- the operator wants creation constrained to the current song's MA3 range

The current saved route model only persists:

- `layer_id -> ma3_track_coord`

That is still correct.
The missing capability is not a new routing truth.
It is MA3-side target preparation.

## Design Rules

- Main remains truth.
- MA3 sync remains main-only.
- Saved route remains `layer -> track coord`.
- Do not use MA3 sequence names as EchoZero truth keys.
- MA3 sequence naming and current-song range discovery may be used as MA3-side
  helper logic, but EchoZero should persist stable identifiers, not names.
- MA3 track preparation belongs on the MA3 sync boundary, not in raw UI code.

## Operator Outcomes

When an operator routes or pushes to MA3 and the target track lacks a sequence,
EchoZero should offer exactly these preparation choices:

1. Use existing MA3 sequence
2. Create new MA3 sequence in next available slot
3. Create new MA3 sequence in current song range

After one of those completes successfully, EchoZero should prepare the track for
events and continue the original route or push action.

## Current Song Range Model

The SpeedOfLight MA3 plugin bundle uses this convention:

1. Read `GetVar(GlobalVars(), "song")`
2. Find the sequence whose `name` matches that song value
3. Treat that sequence number as the song-range start
4. Treat the range as `start .. start + 99`

That is a valid MA3-side helper model.
It should not become EchoZero truth.

EchoZero should treat "current song range" as a resolved MA3 helper result
returned by MA3 Lua, not as a locally derived rule based on EchoZero song names.

## Proposed Sync-Boundary Additions

Extend the MA3 adapter/bridge boundary with explicit sequence operations.

New snapshot types:

- `MA3SequenceSnapshot`
  - `number: int`
  - `name: str`
  - `cue_count: int | None = None`
- `MA3SequenceRangeSnapshot`
  - `song_label: str | None`
  - `start: int`
  - `end: int`

Extend `MA3TrackSnapshot` with:

- `sequence_no: int | None = None`

New adapter methods:

- `list_sequences(*, start_no: int | None = None, end_no: int | None = None) -> Sequence[MA3SequenceSnapshot]`
- `get_current_song_sequence_range() -> MA3SequenceRangeSnapshot | None`
- `assign_track_sequence(*, target_track_coord: str, sequence_no: int) -> None`
- `create_sequence_next_available(*, preferred_name: str | None = None) -> MA3SequenceSnapshot`
- `create_sequence_in_current_song_range(*, preferred_name: str | None = None) -> MA3SequenceSnapshot`
- `prepare_track_for_events(*, target_track_coord: str) -> None`

`prepare_track_for_events(...)` is the only prep helper EchoZero should call
after assignment.
Its job is to make the track write-ready using MA3 Lua.

That keeps the bridge simple:

- send typed MA3 commands
- normalize replies
- do not invent operator policy

## Proposed MA3 Lua Surface

The MA3 Lua side should own the concrete mechanics.

Needed helpers:

- `EZ.GetSequences(startNo, endNo)`
- `EZ.GetCurrentSongSequenceRange()`
- `EZ.CreateSequenceNextAvailable(name)`
- `EZ.CreateSequenceInCurrentSongRange(name)`
- `EZ.AssignTrackSequence(tcNo, tgNo, trackNo, seqNo)`
- `EZ.PrepareTrackForEvents(tcNo, tgNo, trackNo)`

`EZ.PrepareTrackForEvents(...)` should be the place where `CreateCmdSubTrack`
becomes deterministic after sequence assignment.

Desired behavior:

1. verify the track exists
2. verify a sequence is assigned
3. verify a `TimeRange` exists or create one if needed
4. verify a `CmdSubTrack` exists or create one if needed
5. return a structured success/failure result

That removes the need for bridge-side first-event repair once Lua prep is
trustworthy.

## Application-Layer Design

Keep the existing push contract intact.
Add a small sequence-preparation contract beside it.

New timeline intents:

- `RefreshMA3Sequences`
  - optional `range_mode`: `all` or `current_song`
- `AssignMA3TrackSequence`
  - `target_track_coord`
  - `sequence_no`
- `CreateMA3Sequence`
  - `creation_mode`: `next_available` or `current_song_range`
  - `preferred_name`
- `PrepareMA3TrackForPush`
  - `target_track_coord`

The orchestrator should compose them, not the widget:

1. resolve target track
2. if the track already has `sequence_no`, continue
3. otherwise perform the chosen sequence action
4. run `PrepareMA3TrackForPush`
5. continue the saved route or push action

This must work for both:

- saved-route sends
- one-shot sends

One-shot preparation must not mutate the saved route.

## Session And Presentation State

Extend manual-push state with only the data the UI needs to render choices.

Add:

- `available_sequences: list[ManualPushSequenceOption]`
- `current_song_sequence_range: ManualPushSequenceRange | None`

Extend `ManualPushTrackOption` with:

- `sequence_no: int | None = None`

Suggested UI-facing models:

- `ManualPushSequenceOption`
  - `number`
  - `name`
- `ManualPushSequenceRange`
  - `song_label`
  - `start`
  - `end`

Do not persist these in the timeline or layer model.
They are live MA3 state, not EchoZero truth.

## UI Flow

### Route Layer To MA3 Track

1. Operator chooses MA3 track
2. If the track already has a sequence:
   - save route immediately
3. If the track has no sequence:
   - show sequence preparation chooser:
     - use existing sequence
     - create next available
     - create in current song range
   - complete prep
   - save route

### Send Layer To MA3 / Send Selected Events To MA3

1. Resolve target as today
2. If target already has sequence:
   - continue
3. If target lacks sequence:
   - show the same sequence preparation chooser
   - prepare track
   - continue push

### Send To Different Track Once

1. Operator chooses temporary target
2. If target lacks sequence:
   - show sequence preparation chooser
   - prepare track
3. Continue push
4. Do not mutate the saved route

## Naming Rules

Sequence names should be treated as operator-facing labels only.

Recommended default name for new MA3 sequences:

- `<song title> - <layer name>`

But:

- the operator may override that later in MA3
- EchoZero should never depend on that name for routing truth

## Failure Handling

When a push target is missing sequence assignment:

- do not fall through to raw add-event failure if we can detect it first
- surface an explicit operator choice to assign or create a sequence

When current-song range resolution fails:

- explain why clearly:
  - no MA3 `song` global var
  - song anchor sequence not found
  - no free sequence slot in that range
- allow fallback to:
  - existing sequence
  - next available sequence

## Migration From The Current Bridge Repair

Current state:

- the bridge has a fallback path for first-write `No CmdSubTrack`

Target state:

1. MA3 Lua reliably prepares the track after sequence assignment
2. push flow explicitly prepares the track before sending events
3. bridge-side `CreateCmdSubTrack` retry path is removed

Until Lua prep is proven, keep the fallback only as a temporary guardrail.

## Proof Plan

This slice is complete when all of this is proven:

1. Track with no sequence can be prepared by selecting an existing sequence.
2. Track with no sequence can be prepared by creating a sequence in next available space.
3. Track with no sequence can be prepared by creating a sequence in current song range.
4. After prep, full-layer push succeeds.
5. After prep, selected-events push succeeds.
6. One-shot prep does not mutate the saved route.
7. The canonical shell path covers at least one full operator run on real MA3.
8. The bridge-side first-event repair path can be safely removed or explicitly
   retained with a reason.

## Immediate Next Slice

Implementation order:

1. Add sequence snapshots and `sequence_no` to the sync boundary.
2. Add MA3 Lua helpers for sequence listing, current-song range, creation, and
   deterministic track prep.
3. Add bridge/adapter support for those helpers.
4. Add app-layer intents and orchestrator support.
5. Add UI sequence-preparation chooser on route and push actions.
6. Validate all three operator prep flows on real hardware.
