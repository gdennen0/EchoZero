# MA3 Hardware Handoff — 2026-04-22

## Status And Authority

This is a continuation note for the MA3 push lane.
It is not a new canonical architecture spec.

If anything here conflicts with canonical repo docs, the canonical docs win:

- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `docs/architecture/DECISIONS.md`
- `docs/MA3-PUSH-V1-EXECUTION-PLAN.md`
- `AGENTS.md`

## Executive Summary

The real MA3 transport path is confirmed and now matches production code:

- EchoZero -> MA3 command path is `"/cmd"`
- outbound payloads are wrapped as `Lua "EZ...."`
- MA3 -> EchoZero replies return on `"/ez/message"`
- tested MA3 command port is `8000`
- tested MA3 host is `192.168.1.70`

The real MA3 push workflow is now proven beyond transport:

- full-layer push succeeded on real hardware
- selected-events push succeeded on real hardware
- one-shot reroute succeeded on real hardware
- overwrite succeeded on real hardware
- one-shot reroute did not mutate the saved route

The original blocker was `No CmdSubTrack - Attempting to Aquire() CmdSubTrack`.
That is now partially automated in the bridge:

- if a track already has an assigned sequence but the first write hits `No CmdSubTrack`,
  the bridge now issues `EZ.CreateCmdSubTrack(...)` once and retries the first event
- if that retry still fails, the bridge surfaces the real prerequisite:
  assign a sequence to the MA3 track, then retry the push

For MA3 push v1, the remaining manual prep requirement is:

- the target track must already have an MA3 sequence assigned

EchoZero does not safely invent sequence numbers in v1.

## Repo Changes In This Slice

These files were intentionally changed during this stabilization pass:

- `echozero/infrastructure/sync/ma3_osc.py`
- `echozero/testing/ma3/simulator.py`
- `tests/testing/test_ma3_osc_bridge.py`

Relevant files inspected during this pass:

- `echozero/application/settings/models.py`
- `MA3/dev/ma3_ping_probe.py`
- `tests/application/test_ma3_push_v1_flow.py`
- `tests/ui/timeline_shell_transfer_cases.py`
- `tests/testing/test_simulated_ma3_bridge.py`
- `tests/application/test_sync_adapters.py`

## Proven Hardware Results

### Full-Layer Push

Source:

- DB: `/Users/march/.echozero/working/be668a6e32d34d9180be1bdbe6747d71/project.db`
- layer: `Kick`
- layer id: `1c3051a768ff43998842ff7dba6338d0`
- main take id: `ea3d5d84-7def-4384-883c-203dbc34be20`
- source event count: `215`

Target:

- MA3 coord: `tc2_tg1_tr1`
- assigned sequence: `2`
- apply mode used for the successful real push: `overwrite`

Preparation that was required at the time of the successful push:

1. `EZ.AssignTrackSequence(2, 1, 1, 2)`
2. `EZ.CreateCmdSubTrack(2, 1, 1, 1)`
3. one probe write to confirm the track was writable

Observed result:

- target event count before overwrite: `1` seed event
- target event count after overwrite: `215`
- counts matched: `yes`

The successful target state was re-checked later on the same hardware session:

- `EZ.GetTracks(2, 1)` still reported `event_count = 215` for `tc2_tg1_tr1`
- direct event query also returned `215`

### Selected-Events Push

Hardware validation used the real Kick main take through the app-layer push path.

Observed result:

- target: `tc113_tg1_tr8`
- selected event count sent: `3`
- final MA3 event count: `3`
- counts matched: `yes`

This target was used as a sacrificial ready track.

### Overwrite On A Sacrificial Track

After the 3-event selected push above, the same sacrificial target was overwritten.

Observed result:

- target: `tc113_tg1_tr8`
- overwrite event count sent: `2`
- final MA3 event count after overwrite: `2`
- counts matched: `yes`

This confirmed that overwrite replaced prior track content rather than appending.

### One-Shot Reroute

Hardware validation used the app-layer one-shot target flow with selected events.

Observed result:

- one-shot target: `tc2_tg2_tr1`
- selected event count sent: `4`
- final MA3 event count on one-shot target: `4`
- counts matched: `yes`

Saved-route behavior:

- saved route before one-shot send: `tc2_tg1_tr1`
- saved route after one-shot send: `tc2_tg1_tr1`
- saved-route target event count after one-shot send: `215`
- saved route mutated by one-shot send: `no`

Preparation required for this target:

- `EZ.AssignTrackSequence(2, 2, 1, 2)`

Once the sequence existed, the bridge-side retry path could handle `CmdSubTrack`
creation during the first write.

## The Original Blocker And The Current Recovery Path

The initial failing real MA3 error was:

- message key: `event.error`
- fields:
  - `tc = 2`
  - `tg = 1`
  - `track = 1`
  - `error = "No CmdSubTrack - Attempting to Aquire() CmdSubTrack"`

Current bridge behavior in `ma3_osc.py`:

1. try the first `EZ.AddEvent(...)`
2. watch briefly for `event.error`
3. if the error contains `No CmdSubTrack`, issue `EZ.CreateCmdSubTrack(...)`
4. retry the first event once
5. if the retry still fails, raise a clear operator-facing error that the track
   needs a sequence assigned in MA3 before retrying

This removes the need for the old “create one manual seed event first” workaround
when the target track already has a valid sequence assignment.

## Current MA3 Prep Requirement

Safe automation boundary for v1:

- EchoZero can recover a missing `CmdSubTrack`
- EchoZero does not yet auto-assign MA3 sequences

Minimal documented MA3 prerequisite:

1. choose or create the target MA3 sequence
2. assign that sequence to the target MA3 track
3. then run the EchoZero push

If the track is assigned but lacks the command subtrack, EchoZero should now
repair that case automatically during the first write attempt.

## Validation Completed

Focused automated validation completed after the bridge recovery change:

- `./.venv/bin/python -m pytest tests/testing/test_ma3_osc_bridge.py -q`
- `./.venv/bin/python -m pytest tests/testing/test_simulated_ma3_bridge.py tests/application/test_sync_adapters.py tests/application/test_ma3_push_v1_flow.py -q`
- `./.venv/bin/python -m pytest tests/ui/timeline_shell_transfer_cases.py -q`

Real hardware validation completed against `192.168.1.70`:

- transport ping over `/cmd`
- full-layer push to `tc2_tg1_tr1`
- selected-events push to `tc113_tg1_tr8`
- overwrite on `tc113_tg1_tr8`
- one-shot reroute to `tc2_tg2_tr1`
- saved-route non-mutation proof back to `tc2_tg1_tr1`

## Remaining Work Before MA3 Push V1 Is Fully Trustworthy

The lane is now beyond one-off debugging, but there are still two important
stabilization steps before calling push v1 fully stable:

1. run one canonical shell-path smoke from the real UI surface against MA3
2. make the MA3 sequence-assignment prerequisite explicit in operator-facing UX
   wherever a push can fail on an unassigned empty target

Neither of those changes should alter the core push contract.

## Recommended Next Milestone After Push V1

After push v1 is stable, the next MA3 milestone should be:

- `Section` and `Recipe` on top of the existing saved-route model

That keeps the current layer-to-track route as the durable foundation instead of
rewriting the push contract again.

## Worktree Warning

This repo/worktree is still very dirty with many unrelated tracked and untracked
changes. Do not clean or revert unrelated files while working this lane.
