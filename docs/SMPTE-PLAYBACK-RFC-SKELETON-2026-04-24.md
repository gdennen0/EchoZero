# SMPTE Playback Reliability RFC (Skeleton)

Status: Draft skeleton  
Date: 2026-04-24  
Owner: TBD  
Reviewers: TBD

## 1. Purpose

Define the canonical design contract for reliable SMPTE-aware playback in EchoZero, including:

- SMPTE display in the timeline/ruler
- deterministic frame/time conversions
- generation of outbound timecode signals (MTC/LTC)
- chase mode for inbound timecode
- acceptance criteria for production reliability

This RFC is a clean-sheet contract document. It is not an implementation PR.

## 2. Why Now

Playback has active architectural debt around clock ownership and transport truth. SMPTE reliability requires a stricter and more explicit contract than generic playhead rendering.

Existing product intent already calls for SMPTE + MTC/LTC + chase:

- `docs/architecture/DECISIONS.md` (D175)
- `docs/architecture/DECISIONS.md` (D100 per-project FPS/timebase)

## 3. Execution Kickoff Decisions (Provisional Freeze)

To begin execution now, this RFC adopts the following provisional decisions.
These are treated as active defaults for implementation unless superseded by explicit review.

### 3.1 v1 Supported Timebase Matrix

v1 includes:

1. 24.000 NDF
2. 25.000 NDF
3. 29.97 NDF
4. 29.97 DF
5. 30.000 NDF

Deferred from v1:

1. 23.976 NDF
2. 59.94 variants

### 3.2 Protocol Scope By Phase

1. Master mode generation: MTC and LTC in initial delivery.
2. Chase mode: MTC-first for initial chase reliability, LTC chase as the first follow-up extension after chase baseline acceptance.

### 3.3 Initial Reliability Targets

1. Locked long-run drift budget: less than 1 frame over 60 minutes.
2. Reacquire target after valid source resumes: less than or equal to 2.0 seconds.
3. Frame-boundary determinism: no off-by-one behavior in seek/stop/play boundary tests.
4. Correction safety bound: no unbounded correction jumps while reporting `LOCKED`.

## 4. Current State (Observed)

This section should be kept short and factual, with links to code paths that motivate this RFC.

Current highlights:

- timeline labels/ruler are currently second-based and MM:SS-focused in multiple UI paths
- export timecode path currently validates only `24/25/30` frame rates
- project-level `timecode_fps` exists in persistence but is not yet a runtime authority
- playback truth and presentation clock rendering are still being remediated

## 5. Goals

1. One canonical timebase authority across app, playback, UI, export, and sync.
2. Reliable SMPTE conversion (including 29.97 DF/NDF behavior).
3. Sample-accurate transport remains source of truth; frames/timecode are deterministic projections.
4. Support both generator mode (master) and chase mode (follower).
5. Provide measurable reliability gates before declaring done.

## 6. Non-Goals

1. Replacing the entire audio engine in this RFC.
2. Designing every UI affordance in detail.
3. Defining full MIDI/LTC device UX for all operating systems in this draft.
4. Bundling unrelated timeline/editor refactors.

## 7. Terminology

`TimebaseSpec`: Canonical project/session timebase configuration.  
`TransportSamples`: Monotonic sample position used as playback truth.  
`SMPTEFrame`: Integer frame index in selected SMPTE timebase.  
`DF`: Drop-frame numbering rules for nominal 29.97 family rates.  
`NDF`: Non-drop-frame numbering.  
`Master Mode`: EchoZero generates timecode from internal transport clock.  
`Chase Mode`: EchoZero follows external timecode.

## 8. Normative Requirements

The final RFC should use MUST/SHOULD/MAY language in this section.

### 8.1 Canonical Timebase

1. The system MUST define one canonical `TimebaseSpec` for active playback context.
2. `TimebaseSpec` MUST represent frame rate as rational values where applicable.
3. `TimebaseSpec` MUST include:
   - nominal fps descriptor
   - rational fps (`num/den`)
   - drop-frame flag
   - display format policy
   - start timecode offset

### 8.2 Truth Ownership

1. Playback truth MUST be sample-position-based and backend-owned.
2. UI MUST NOT own transport truth.
3. Any time label/ruler/export conversion MUST use the shared timecode codec path.

### 8.3 Supported Timebases (Initial)

Initial mandatory support target (finalize in review):

1. 24.000 NDF
2. 25.000 NDF
3. 29.97 NDF
4. 29.97 DF
5. 30.000 NDF

Optional later target:

1. 23.976 NDF
2. 59.94 variants

### 8.4 Conversion Semantics

1. Seconds/samples/frame/timecode conversions MUST be deterministic and reversible within defined rounding policy.
2. Rounding policy MUST be explicit and centralized (no per-call ad-hoc rounding).
3. 29.97 DF numbering rules MUST be implemented exactly and unit-tested on boundary minutes.

## 9. Proposed Architecture

### 9.1 Core Components

1. `PlaybackCore` (application boundary owner)
   - transport control (`play/pause/stop/seek`)
   - clock source selection (master/chase)
   - publishes transport snapshots
2. `TimecodeCodec`
   - parse/format SMPTE strings
   - sample <-> frame conversions
   - DF/NDF logic
3. `TimecodeIO` adapters
   - outbound generator: MTC, LTC
   - inbound chase: MTC and/or LTC decode
4. `ClockDiscipline` (chase control)
   - lock detection
   - correction and smoothing
   - holdover/reacquire behavior

### 9.2 State Machine (Chase)

Define and freeze transitions:

1. `UNLOCKED`
2. `LOCKING`
3. `LOCKED`
4. `HOLDOVER`

For each transition, define:

1. entry criteria
2. exit criteria
3. correction behavior
4. operator-visible status

## 10. Data Contracts (Draft)

Define typed contracts before implementation:

```text
TimebaseSpec
TransportSnapshot
TimecodeSnapshot
TimecodeLockState
```

Minimum required fields (to be finalized):

1. sample position
2. sample rate
3. projected frame index
4. formatted SMPTE label
5. is_playing
6. clock source
7. lock state and confidence
8. snapshot timestamp

## 11. UI Contract

1. Ruler/playhead labels use `TimecodeCodec` output only.
2. UI timer is repaint trigger only; never transport truth.
3. SMPTE mode and MM:SS mode can coexist as view modes over same underlying transport.
4. Drift/lock indicators must be visible when in chase mode.

## 12. Export and Sync Contract

1. Exporters and sync adapters must consume the same canonical `TimebaseSpec`.
2. MA3-related timing outputs must not introduce independent frame conversion rules.
3. Any legacy helper with duplicated conversion logic should be deprecated.

## 13. Reliability SLOs and Acceptance Criteria

Active baseline targets for execution:

1. Long-run drift in locked mode: less than 1 frame over 60 minutes.
2. Seek determinism: no off-by-one frame ambiguity at boundaries.
3. Reacquire time after source recovery: less than or equal to 2.0 seconds.
4. Correction safety: bounded correction behavior with no unbounded jump while `LOCKED`.
5. Audible integrity: no correction-induced artifact beyond existing playback baseline in acceptance smoke.

Mandatory test gates:

1. Conversion property tests (seconds/samples/frame/timecode roundtrip)
2. DF boundary tests (minute exceptions and hour rollover)
3. Playback integration tests for master and chase
4. Human-path proof run with real runtime actions and explicit lock telemetry

## 14. Observability

Define structured logs/metrics for:

1. clock source
2. lock state transitions
3. correction magnitude
4. drift estimates
5. signal loss/reacquire events

## 15. Rollout Plan

### Phase 0: Contract Freeze

1. finalize `TimebaseSpec` and conversion policy
2. freeze accepted frame-rate matrix
3. freeze chase state machine

### Phase 1: Shared Codec + UI/Export Unification

1. implement shared conversion path
2. remove duplicated UI/export formatting logic
3. pass conversion and boundary tests

### Phase 2: Master Mode Timecode IO

1. outbound MTC/LTC generation from canonical transport
2. verify continuity and frame-accurate progression

### Phase 3: Chase Mode

1. inbound decode path
2. discipline controller with lock states
3. dropout and reacquire behavior validation

### Phase 4: Production Hardening

1. app-boundary regression lanes
2. human-path demo proof package
3. release smoke and packaging checks

## 16. Migration and Backward Compatibility

1. preserve existing projects by defaulting missing timebase fields.
2. treat legacy `timecode_fps` as migration input to `TimebaseSpec`.
3. define explicit fallback behavior when unsupported/invalid settings are loaded.

## 17. Risks

1. hidden duplicate conversion logic surviving in edge surfaces
2. UI-level timing assumptions conflicting with backend truth
3. incorrect DF edge handling causing cumulative event alignment errors
4. brittle external-device behavior across host APIs

## 18. Open Questions

1. What are operator-facing defaults for start offset and drop-frame mode?
2. What correction profile should be preferred in chase (`slew-first` vs thresholded snap)?
3. Should LTC chase move into the same release train as MTC chase if stability proves strong?

## 19. Decision Checklist

Mark each item as `[ ]` pending or `[x]` decided during review.

1. [x] Supported frame-rate matrix finalized (provisional execution freeze)
2. [x] DF/NDF policy finalized for v1 matrix (provisional execution freeze)
3. [x] Clock-source mode behavior finalized (master + chase split by phase)
4. [ ] Chase lock-state semantics finalized
5. [x] Reliability SLO baseline finalized (provisional execution freeze)
6. [ ] Test gate matrix finalized

## 20. References

Internal:

1. `docs/architecture/DECISIONS.md` (D100, D175)
2. `docs/PLAYBACK-CLOCK-CLEAN-SHEET-DESIGN.md`
3. `docs/PLAYBACK-REMEDIATION-EXECUTION-PLAN.md`
4. `docs/STATUS.md`
5. `docs/SMPTE-PLAYBACK-TASK-BOARD-2026-04-24.md`
6. `docs/SMPTE-PLAYBACK-EXECUTION-PLAN-2026-04-24.md`

External background (non-normative):

1. PortAudio timing and latency guidance
2. sounddevice API docs
3. relevant SMPTE/MTC/LTC protocol references (to be listed)

## 21. Revision History

1. 2026-04-24: Initial skeleton created.
2. 2026-04-24: Added provisional execution freeze decisions and initial SLO baseline.
