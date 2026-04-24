# MA3 Sequence Management Helpers

This document defines the MA3 Lua helper contract used by EchoZero MA3 push
sequence preparation.

## Scope

Helpers live in `MA3/plugins/timecode.lua`.

They are MA3-side preparation helpers only. EchoZero route truth remains
`layer -> track coord`. Sequence names stay operator-facing labels and are not
truth keys.

## Current Song Range Rule

`EZ.GetCurrentSongSequenceRange()` follows the SpeedOfLight convention:

1. Read `GetVar(GlobalVars(), "song")`
2. Find the sequence whose `name` matches that value
3. Treat that sequence number as the start anchor
4. Treat the range as `start .. start + 99`

The resolved range is helper output, not EchoZero truth.

## Helper Surface

### `EZ.GetSequences(startNo, endNo)`

Lists MA3 sequences. When `startNo` and `endNo` are provided, the result is
filtered to that inclusive range.

Success reply:

- `sequences.list`
- Payload: `{count, sequences=[{no, name, cue_count}]}`

Failure reply:

- `sequences.error`
- Payload: `{error}`

### `EZ.GetCurrentSongSequenceRange()`

Resolves the MA3 current-song range from `GlobalVars().song`.

Success reply:

- `sequence_range.current_song`
- Payload: `{song_label, start, end}`

Failure reply:

- `sequence_range.error`
- Payload: `{error}`

Current error codes:

- `global_vars_unavailable`
- `song_global_lookup_failed`
- `song_global_missing`
- `sequence_enumeration_failed`
- `song_anchor_sequence_not_found`

### `EZ.CreateSequenceNextAvailable(name)`

Creates a sequence in the first free slot starting at sequence `1`.

Success reply:

- `sequence.created`
- Payload: `{no, name, mode="next_available"}`

Failure reply:

- `sequence.error`
- Payload: `{error, mode="next_available", ...}`

### `EZ.CreateSequenceInCurrentSongRange(name)`

Creates a sequence in the first free slot within the resolved current-song
range.

Success reply:

- `sequence.created`
- Payload: `{no, name, mode="current_song_range"}`

Failure replies:

- `sequence_range.error`
- Payload: `{error}`
- `sequence.error`
- Payload: `{error, mode="current_song_range", song_label, start, end}`

### `EZ.AssignTrackSequence(tcNo, tgNo, trackNo, seqNo)`

Assigns a sequence to the user-visible track number.

Success reply:

- `track.assigned`
- Payload: `{tc, tg, track, seq, changed}`

Failure reply:

- `track.error`
- Payload: `{tc, tg, track, error, ...}`

Current error codes:

- `track_not_found`
- `invalid_sequence_number`
- `sequence_not_found`
- `sequence_assignment_failed`
- `sequence_assignment_verification_failed`

### `EZ.PrepareTrackForEvents(tcNo, tgNo, trackNo)`

Deterministically prepares a track for command-event writes.

Behavior:

1. Verify the track exists
2. Verify a sequence is assigned
3. Ensure a `TimeRange` exists
4. Ensure a `CmdSubTrack` exists
5. Return structured success or failure

Success reply:

- `track.prepared`
- Payload: `{tc, tg, track, seq, time_range_idx, cmd_subtrack_ready=true}`

Additional success flags may be present:

- `time_range_created=true`
- `cmd_subtrack_created=true`

Failure reply:

- `track.error`
- Payload: `{tc, tg, track, error, ...}`

Current error codes:

- `track_not_found`
- `no_sequence_assigned`
- `time_range_create_failed`
- `cmd_subtrack_create_failed`

## Add Event Interaction

`EZ.AddEvent(...)` now reuses the same preparation logic before writing the
event. That keeps first-write behavior aligned with the new prep flow without
moving preparation into Python.

Direct `EZ.PrepareTrackForEvents(...)` calls emit `track.prepared` or
`track.error`. `EZ.AddEvent(...)` uses the prep logic silently and still emits
`event.error` if preparation fails.
