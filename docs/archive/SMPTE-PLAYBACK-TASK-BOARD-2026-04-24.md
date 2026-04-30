# SMPTE Playback Task Board

Status: historical
Last reviewed: 2026-04-30


Originally updated: 2026-04-24

This board is retained as historical planning context.
For current SMPTE behavior, use `docs/STATUS.md` and `MA3/README.md`.

This board turns the SMPTE playback RFC skeleton into executable work with bounded scope and proof gates.

Primary source:
- [SMPTE-PLAYBACK-RFC-SKELETON-2026-04-24.md](SMPTE-PLAYBACK-RFC-SKELETON-2026-04-24.md)

## Goal

Ship reliable SMPTE-aware playback with:

- canonical timebase contract
- deterministic conversion logic
- unified UI/export/sync time semantics
- master and chase clock modes
- measurable reliability proof

## Hard Rules

1. Transport truth is backend-owned sample position.
2. UI timers are repaint triggers only.
3. No duplicate timecode conversion logic in UI/export/sync.
4. Human-path playback proof is mandatory for release-affecting claims.

## Execution Order

Work in this order:

1. SP-00 through SP-05
2. SP-10 through SP-15
3. SP-20 through SP-24
4. SP-30 through SP-34
5. SP-40 through SP-44
6. SP-50 final signoff

## Task Board

### Phase 0: Contract Freeze

#### SP-00: Freeze Supported Timebase Matrix
- Owner: `lead-dev`
- Depends on: none
- Scope:
  - freeze v1 supported modes:
    - 24 NDF
    - 25 NDF
    - 29.97 NDF
    - 29.97 DF
    - 30 NDF
  - explicitly mark deferred modes
- Proof:
  - RFC checklist updated
- Done when:
  - supported matrix is unambiguous and documented

#### SP-01: Freeze MTC/LTC Scope
- Owner: `lead-dev`
- Depends on: SP-00
- Scope:
  - freeze initial IO scope for generation and chase
  - define phased order if partial delivery is used
- Proof:
  - RFC section updates
- Done when:
  - implementation can proceed without protocol scope ambiguity

#### SP-02: Freeze SLO Targets
- Owner: `lead-dev`
- Depends on: SP-00
- Scope:
  - finalize numeric targets for drift/reacquire/correction behavior
- Proof:
  - reliability section no longer contains placeholder values
- Done when:
  - pass/fail criteria are measurable

#### SP-03: Freeze Chase State Machine
- Owner: `lead-dev`
- Depends on: SP-00, SP-01
- Scope:
  - define lock-state transitions and correction policy
- Proof:
  - state machine table approved in RFC
- Done when:
  - implementation can be tested against explicit transitions

#### SP-04: Freeze TimebaseSpec Contract
- Owner: `impl`
- Depends on: SP-00
- Scope:
  - define typed `TimebaseSpec` contract (fields and invariants)
- Proof:
  - contract doc section and targeted contract tests
- Done when:
  - all conversion consumers can share one contract

#### SP-05: Create Tracking Matrix
- Owner: `verify`
- Depends on: SP-00, SP-01, SP-02, SP-03, SP-04
- Scope:
  - map each requirement to proof lane and test IDs
- Proof:
  - matrix table committed to this board
- Done when:
  - no requirement lacks a proof lane

### Phase 1: Shared Codec and Contract Wiring

#### SP-10: Add Canonical Timecode Codec
- Owner: `impl`
- Depends on: SP-04
- Scope:
  - add one canonical conversion path for:
    - samples <-> seconds
    - samples/seconds <-> SMPTE frames
    - SMPTE string parse/format
  - centralize rounding policy
- Proof:
  - unit/property tests
- Done when:
  - no secondary conversion helper is required for core flows

#### SP-11: Unify UI Time Formatting
- Owner: `impl`
- Depends on: SP-10
- Scope:
  - route ruler/playhead labels through canonical codec
- Proof:
  - UI contract tests
- Done when:
  - MM:SS and SMPTE are view modes over one source conversion path

#### SP-12: Unify Export Time Formatting
- Owner: `impl`
- Depends on: SP-10
- Scope:
  - route export timecode formatting through canonical codec
  - remove local ad-hoc export conversions
- Proof:
  - export tests including 29.97 NDF/DF
- Done when:
  - exporter behavior follows same contract as runtime/UI

#### SP-13: Unify Sync Time Semantics
- Owner: `impl`
- Depends on: SP-10
- Scope:
  - ensure sync adapter surfaces consume canonical timebase semantics
- Proof:
  - sync adapter tests
- Done when:
  - no sync path has independent frame policy

#### SP-14: Migration of Legacy `timecode_fps`
- Owner: `impl`
- Depends on: SP-04
- Scope:
  - map legacy persisted `timecode_fps` into `TimebaseSpec`
  - define fallback rules for missing/invalid values
- Proof:
  - persistence migration tests
- Done when:
  - old projects load deterministically

#### SP-15: Phase 1 Gate
- Owner: `verify`
- Depends on: SP-11, SP-12, SP-13, SP-14
- Scope:
  - execute contract proof lanes for shared codec rollout
- Proof:
  - targeted pytest slices + app-boundary lane
- Done when:
  - contract convergence is verified end-to-end

### Phase 2: Master Mode Timecode IO

#### SP-20: Outbound MTC Generation
- Owner: `impl`
- Depends on: SP-15
- Scope:
  - generate MTC from canonical transport/timebase
- Proof:
  - transport continuity tests + protocol conformance checks
- Done when:
  - generated MTC progression is frame-consistent

#### SP-21: Outbound LTC Generation
- Owner: `impl`
- Depends on: SP-15
- Scope:
  - generate LTC from canonical transport/timebase
- Proof:
  - signal decode/roundtrip checks
- Done when:
  - generated LTC decodes to expected SMPTE frames

#### SP-22: Master Mode Device/IO Hardening
- Owner: `impl`
- Depends on: SP-20, SP-21
- Scope:
  - validate host/backend behavior across supported local device setups
- Proof:
  - real-runtime smoke notes and regression checks
- Done when:
  - master mode is stable on supported paths

#### SP-23: Master Mode Telemetry
- Owner: `impl`
- Depends on: SP-20, SP-21
- Scope:
  - log clock source, frame continuity, correction metrics (if any)
- Proof:
  - telemetry assertions in tests
- Done when:
  - operators can inspect master mode health

#### SP-24: Phase 2 Gate
- Owner: `verify`
- Depends on: SP-22, SP-23
- Scope:
  - pass master mode acceptance gates
- Proof:
  - automated + human-path smoke
- Done when:
  - master mode ready for chase integration

### Phase 3: Chase Mode

#### SP-30: Inbound Timecode Decode Path
- Owner: `impl`
- Depends on: SP-15
- Scope:
  - decode inbound timecode into canonical snapshots
- Proof:
  - decode conformance tests
- Done when:
  - inbound frames are parsed reliably

#### SP-31: Clock Discipline Controller
- Owner: `impl`
- Depends on: SP-03, SP-30
- Scope:
  - implement lock/unlock/holdover/reacquire behavior
  - enforce correction bounds
- Proof:
  - state transition tests and correction bound tests
- Done when:
  - chase transitions match frozen contract

#### SP-32: Drift and Holdover Behavior
- Owner: `impl`
- Depends on: SP-31
- Scope:
  - implement and validate drift/holdover behavior under signal loss
- Proof:
  - dropout simulation tests and integration checks
- Done when:
  - reacquire and holdover meet SLO targets

#### SP-33: Chase UI and Operator Status
- Owner: `impl`
- Depends on: SP-31
- Scope:
  - surface lock state and chase health without UI-owned truth
- Proof:
  - UI contract tests
- Done when:
  - operator can see authoritative chase state

#### SP-34: Phase 3 Gate
- Owner: `verify`
- Depends on: SP-32, SP-33
- Scope:
  - pass chase acceptance gates
- Proof:
  - automated + human-path run
- Done when:
  - chase mode meets reliability bar

### Phase 4: Hardening and Release Proof

#### SP-40: Long-Run Drift Gate
- Owner: `verify`
- Depends on: SP-24, SP-34
- Scope:
  - execute long-duration drift validation
- Proof:
  - measured drift report
- Done when:
  - drift remains within SLO

#### SP-41: Boundary Semantics Gate
- Owner: `verify`
- Depends on: SP-15
- Scope:
  - verify seek/stop/frame-boundary determinism
- Proof:
  - boundary regression suite
- Done when:
  - no off-by-one frame regressions

#### SP-42: Regression and App-Boundary Gate
- Owner: `verify`
- Depends on: SP-40, SP-41
- Scope:
  - run app-boundary playback and sync lanes
- Proof:
  - required lane pass set
- Done when:
  - no contract regressions on canonical path

#### SP-43: Human-Path Proof Package
- Owner: `verify`
- Depends on: SP-42
- Scope:
  - produce one human-path playback proof package for master and chase
- Proof:
  - operator run notes with explicit real vs synthetic labeling
- Done when:
  - proof artifacts satisfy demo policy

#### SP-44: Release Readiness Review
- Owner: `lead-dev`
- Depends on: SP-43
- Scope:
  - review risks, open defects, and rollout readiness
- Proof:
  - release review signoff note
- Done when:
  - SMPTE playback marked release-ready

### Phase 5: Final Signoff

#### SP-50: Baseline Acceptance
- Owner: `lead-dev`
- Depends on: SP-44
- Scope:
  - accept SMPTE playback reliability baseline
- Proof:
  - final signoff in docs
- Done when:
  - baseline declared canonical for future changes

## Tracking Matrix (Initial)

| Requirement | Task IDs | Proof lane |
| --- | --- | --- |
| Canonical timebase contract | SP-00, SP-04, SP-10 | unit/contract tests |
| Unified conversion semantics | SP-10, SP-11, SP-12, SP-13 | unit + app-boundary |
| Master mode reliability | SP-20 through SP-24 | integration + human-path |
| Chase mode reliability | SP-30 through SP-34 | integration + human-path |
| Release drift and determinism | SP-40, SP-41, SP-42 | long-run + regression |
