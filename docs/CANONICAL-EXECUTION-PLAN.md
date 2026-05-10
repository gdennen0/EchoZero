# Canonical Execution Plan

Status: active
Last reviewed: 2026-05-03



This is the repo-wide execution plan for the current EchoZero phase.
Use [docs/UNIFIED-IMPLEMENTATION-PLAN.md](UNIFIED-IMPLEMENTATION-PLAN.md) for
the canonical project plan and constraints.
Use [docs/STATUS.md](STATUS.md) for current repo truth.
Use [docs/TESTING.md](TESTING.md) for proof-lane rules.

This file is intentionally operational.
It answers: what should happen next, in what order, with what proof, and what
must stay blocked until earlier work is complete.

## Active Objective

Close alpha-signoff proof and remaining architecture debt without reopening
truth-model or sync-boundary drift.

## Non-Goals

- broad feature expansion
- speculative UX redesign
- large cross-cutting cleanup not tied to canonical risk
- alternate launcher or alternate truth-path work

## Queue Snapshot

1. `active` Keep proof lanes green on every change touching app, timeline, sync,
   or transfer paths.
2. `next` Complete packaged manual UX walkthrough and capture signoff notes.
3. `next` Run real MA3 hardware validation for push, pull, and receive.
4. `next` Capture one visible operator end-to-end proof sequence on the
   canonical app path.
5. `queued` Consolidate the sync boundary behind one app-contract seam.
6. `queued` Harden the sync receive protocol lane and edge-case fixtures.
7. `queued` Decompose canonical hot files without widening behavior scope.
8. `blocked` Reopen net-new feature growth.

## Wave 0
Proof Preservation

Goal:
- ensure every later slice starts from a known-good safety baseline

Required proof:

1. targeted pytest for touched contracts
2. `python -m echozero.testing.run --lane appflow`
3. `python -m echozero.testing.run --lane appflow-sync` for sync-affecting work
4. `python -m echozero.testing.run --lane appflow-protocol` for receive/protocol
   work
5. packaged smoke consideration for release-affecting changes

Stop conditions:

- any change that breaks the canonical app path
- any sync change that cannot be proven at the app boundary
- any cleanup slice that requires behavior redesign to finish

## Wave 1
Packaged UX Walkthrough

Goal:
- finish the manual packaged-app walkthrough that still blocks signoff

Scope:

- launch packaged app via canonical path
- create/open project sanity
- timeline interaction sanity
- transfer workflow sanity
- save/reopen persistence sanity

Deliverables:

- checklist or notes file with pass/fail observations
- explicit issues filed for any blocker found

Done when:

- packaged manual QA result exists and is linked from the active milestone work

## Wave 2
Real MA3 Hardware Validation

Goal:
- prove simulator and protocol coverage still match real operator conditions

Scope:

- connect from canonical app path
- validate push behavior
- validate manual pull and destination behavior
- validate receive-path updates from real MA3 traffic

Deliverables:

- hardware validation notes
- captured deltas between simulator assumptions and real payload behavior
- follow-up fixture/test updates if the real protocol differs

Done when:

- real MA3 validation result is captured and actionable gaps are either fixed or
  explicitly queued

## Wave 3
Visible Operator End-to-End Proof

Goal:
- produce one human-path proof sequence that demonstrates the release story

Scope:

- real launcher
- real shell path
- real timeline workflow
- real transfer/sync touchpoint where applicable

Rules:

- no injected widget presentation shortcuts
- no fake audio or fake app-state driving
- clearly label any synthetic portion if one is unavoidable

Done when:

- one operator-proof run is captured with notes on what was real versus
  synthetic

## Wave 4
Sync Boundary Consolidation

Goal:
- remove remaining ambiguity between app contract, orchestrator behavior, and UI
  routing

Priority file cluster:

- `echozero/application/sync/*`
- `echozero/application/timeline/*`
- `echozero/infrastructure/sync/ma3_adapter.py`

Execution notes:

1. Keep the seam explicit between app contract and concrete sync behavior.
2. Move assertions up to app-boundary tests where possible.
3. Do not mix this slice with large orchestrator decomposition unless a small
   compatibility seam is required.

Done when:

- one documented boundary exists and canonical app-path proof stays green

## Wave 5
Sync Receive Protocol Hardening

Goal:
- make protocol regressions cheap to catch

Priority file cluster:

- receive/protocol fixtures
- communication service receive path
- sync entrypoint handling

Execution notes:

1. Freeze known payload shapes in deterministic fixtures.
2. Add edge cases for delimiter-heavy and nested payloads.
3. Keep the receive lane required before sync-affecting merges.

Done when:

- the receive lane covers current plugin payload reality and stays green

## Wave 6
Canonical Hot-File Decomposition

Goal:
- reduce the regression surface in overloaded canonical files

Priority file cluster:

- `echozero/ui/qt/app_shell.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/presentation/inspector_contract.py`
- `echozero/application/timeline/object_action_settings_service.py`

Execution notes:

1. Split by responsibility boundary, not by arbitrary file size.
2. Preserve imports and app-path behavior incrementally.
3. Run proof after each slice instead of stacking large refactors.
4. Keep no more than two parallel writers on disjoint file clusters.

Done when:

- those files stop being the default dumping ground for unrelated changes

## Exit Gate For Feature Reopen

All of the following must be true:

1. Waves 1 through 6 are complete or explicitly downgraded by decision.
2. Contract, app-flow, sync, and receive proof lanes are green.
3. Packaged smoke remains green.
4. Manual and hardware signoff evidence exists.
5. No known truth-model or sync-boundary regression remains open.

## Suggested Goal Mapping

If Codex `/goals` becomes available in this environment, map this file into the
following goal set:

1. `alpha-signoff-proof`
2. `ma3-real-hardware-validation`
3. `operator-e2e-proof`
4. `sync-boundary-consolidation`
5. `sync-receive-hardening`
6. `canonical-hot-file-decomposition`

Until then, treat each wave above as a goal-shaped execution slice.
