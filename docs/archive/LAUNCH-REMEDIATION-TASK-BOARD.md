# Launch Remediation Task Board

Status: historical
Last reviewed: 2026-04-30


Originally updated: 2026-04-18

This document is retained as a historical execution board.
For current launcher/runtime truth, use `docs/STATUS.md`.

This board turns the current launcher/runtime regression into executable work.
It is intentionally concrete:
- each task has an ID
- each task has a bounded owner role
- each task has dependencies
- each task has a required proof lane
- each task has a done condition

This board is for near-term stabilization of the canonical desktop launch path.

## Goal

Restore `run_echozero.py` as the single real-app launcher for EchoZero so:
- the main app boots through the real EZ runtime only
- no `demo_app` or demo-mode plumbing remains in the runtime launch path
- stale timeline/demo imports do not block app startup
- launcher proof runs through the canonical app path, not demo scaffolding

## Hard Rules

- `run_echozero.py` is the canonical desktop entrypoint.
- Do not fix launch by leaning on `demo_app`.
- Do not preserve demo fallback behavior in the main runtime path.
- Remove or bypass demo-only launch plumbing where appropriate.
- The correct fix must align with the real app architecture, not obsolete demo scaffolding.

## Current Failure Snapshot

- Repo path reported by operator: `/Users/gdennen/Projects/EchoZero`
- Branch: `main`
- Commit: `e3fd1fa80`
- Worktree: clean
- Venv: `/Users/gdennen/Projects/EchoZero/.venv`
- Repro:
  - `cd /Users/gdennen/Projects/EchoZero`
  - `source .venv/bin/activate`
  - `python run_echozero.py`
- Observed import chain:
  - `run_echozero.py`
  - `echozero/ui/qt/app_shell.py`
  - `echozero/ui/qt/timeline/demo_app.py`
  - `echozero/ui/qt/timeline/widget.py`
- Observed error:
  - `ModuleNotFoundError: No module named 'echozero.ui.qt.timeline.object_model'`
- Symbols currently expected by `echozero/ui/qt/timeline/widget.py`:
  - `UiObjectHitRegion`
  - `UiObjectSpec`
  - `build_event_object`
  - `build_layer_object`
  - `build_take_object`

## Investigation Questions

1. Why is `run_echozero.py` still routing through `echozero/ui/qt/timeline/demo_app.py`?
2. Was `echozero/ui/qt/timeline/object_model.py` supposed to be committed and missed?
3. Was that module renamed or moved during refactor without import updates?
4. Is `demo_app.py` stale and keeping the launcher on an obsolete path?

## Execution Order

1. LR-00 through LR-02
2. LR-10 through LR-13
3. LR-20 through LR-23
4. LR-30 signoff

## Task Board

### Phase 0: Trace and Confirm

#### LR-00: Trace Canonical Launch Routing
- Owner: `research`
- Depends on: none
- Scope:
  - trace the live import/runtime path from `run_echozero.py`
  - identify why `app_shell.py` still reaches `timeline/demo_app.py`
  - confirm whether that route is intentional, stale, or accidental
- Files:
  - `run_echozero.py`
  - `echozero/ui/qt/app_shell.py`
  - `echozero/ui/qt/timeline/demo_app.py`
- Proof:
  - code-path notes with file references
  - targeted launcher import proof
- Done when:
  - the exact reason for demo-path routing is documented

#### LR-01: Resolve Missing `object_model` Provenance
- Owner: `research`
- Depends on: LR-00
- Scope:
  - determine whether `echozero/ui/qt/timeline/object_model.py` was deleted, renamed, moved, or omitted
  - map old symbol names to the current canonical module if one exists
- Files:
  - `echozero/ui/qt/timeline/widget.py`
  - related timeline modules
  - git history if required
- Proof:
  - code references or history references
- Done when:
  - the missing-module root cause is explicit

#### LR-02: Choose Architecture-Correct Fix
- Owner: `lead-dev`
- Depends on: LR-00, LR-01
- Scope:
  - choose the fix that preserves real-app launch only
  - reject fixes that keep `demo_app` as runtime glue
- Proof:
  - short decision note on chosen path
- Done when:
  - the chosen fix is documented and demo-path dependency is rejected as the launcher baseline

### Phase 1: Remove Demo-Only Launch Coupling

#### LR-10: Detach Main Launcher From `demo_app`
- Owner: `impl`
- Depends on: LR-02
- Scope:
  - remove or bypass `demo_app` from the `run_echozero.py` main runtime path
  - ensure the launcher boots the real EZ application only
- Files:
  - `run_echozero.py`
  - `echozero/ui/qt/app_shell.py`
  - related launch/runtime modules
- Proof:
  - targeted launcher tests
  - direct `python run_echozero.py` smoke run
- Done when:
  - canonical launch no longer imports `timeline/demo_app.py` during normal app startup

#### LR-11: Remove or Quarantine Stale Demo Runtime Plumbing
- Owner: `impl`
- Depends on: LR-10
- Scope:
  - isolate demo-only code so it cannot masquerade as the main app path
  - update imports/call sites to use canonical runtime modules only
- Files:
  - `echozero/ui/qt/timeline/demo_app.py`
  - related timeline/runtime files
- Proof:
  - targeted tests or import checks
- Done when:
  - demo-only runtime code is no longer on the canonical boot path

#### LR-12: Repair Timeline Object-Model References
- Owner: `impl`
- Depends on: LR-02
- Scope:
  - either restore the correct canonical module for:
    - `UiObjectHitRegion`
    - `UiObjectSpec`
    - `build_event_object`
    - `build_layer_object`
    - `build_take_object`
  - or update imports/callers to the current canonical implementation
- Files:
  - `echozero/ui/qt/timeline/widget.py`
  - canonical object-model module(s)
- Proof:
  - targeted widget/timeline tests
  - import smoke proof
- Done when:
  - `widget.py` resolves object-model symbols through maintained code only

#### LR-13: Keep Demo Behavior Out Of Runtime Baseline
- Owner: `review`
- Depends on: LR-10, LR-11, LR-12
- Scope:
  - audit for hidden fallback paths, demo-mode flags, or stale convenience imports in the runtime boot path
- Proof:
  - review findings
- Done when:
  - no remaining canonical-launch dependency on demo-mode plumbing is found

### Phase 2: Prove Real App Launch

#### LR-20: Launcher Contract Coverage
- Owner: `impl`
- Depends on: LR-10, LR-12
- Scope:
  - update or add tests proving `run_echozero.py` starts the real shell path only
- Files:
  - `tests/ui/test_run_echozero_launcher.py`
  - `tests/testing/test_app_shell_profiles.py`
  - related launcher/runtime tests
- Proof:
  - targeted pytest slices
- Done when:
  - tests fail if demo launch plumbing re-enters the canonical path

#### LR-21: App-Path Smoke Verification
- Owner: `verify`
- Depends on: LR-20
- Scope:
  - prove the app reaches a real launchable state through the canonical launcher
- Proof:
  - `python run_echozero.py`
  - relevant appflow or launcher test slice
- Done when:
  - the app no longer crashes before opening on the canonical launch path

#### LR-22: Timeline/UI Contract Guardrails
- Owner: `verify`
- Depends on: LR-12
- Scope:
  - prove the repaired timeline path still honors application/UI contracts
- Proof:
  - targeted timeline/widget tests
  - app-path UI contract slice if applicable
- Done when:
  - the timeline import fix does not regress app/runtime contracts

#### LR-23: Packaging/Smoke Consideration
- Owner: `review`
- Depends on: LR-21
- Scope:
  - confirm whether launcher-path changes affect packaging or release smoke obligations
- Proof:
  - review note
- Done when:
  - release-affecting implications are explicitly captured

### Phase 3: Signoff

#### LR-30: Launch Baseline Restored
- Owner: `lead-dev`
- Depends on: LR-13, LR-21, LR-22, LR-23
- Scope:
  - consolidate evidence
  - accept the restored launcher as the new baseline
- Proof:
  - launcher smoke result
  - targeted test results
  - review note
- Done when:
  - `run_echozero.py` is accepted as real-app-only launch baseline on `main`

## Decision Log

### D-LR-1
- Decision:
  - the canonical launcher must boot the real EZ application only

### D-LR-2
- Decision:
  - demo-only runtime code is not an acceptable dependency of the main launch path

### D-LR-3
- Decision:
  - timeline import repair must target maintained canonical modules, not revive obsolete demo scaffolding
