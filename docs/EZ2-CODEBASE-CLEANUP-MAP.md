# EZ2 Codebase Cleanup Map

_Updated: 2026-04-15_

This file is the working map for the EZ2-only branch state.

## Goal

Keep the EZ2 app lane, Foundry lane, and required sidecars.

Remove legacy EchoZero 1 code and prototype-only surfaces without breaking the active EZ2 runtime.

## Current Read

The repo now contains three different categories of code:

1. **Canonical EZ2**
   - `echozero/**`
   - `run_echozero.py`
   - `run_timeline_demo.py` as compatibility shim only
   - `tests/application/**`
   - `tests/ui/**`
   - `tests/testing/**`
   - `tests/foundry/**`
   - `tests/inference_eval/**`

2. **Removed legacy surfaces**
   - `src/**` removed
   - `ui/**` legacy surfaces removed
   - root `main.py` removed
   - root `main_qt.py` removed
   - `.cursor/**`, `AgentAssets/**`, and `sass/**` removed
   - vendored worker `node_modules/**` removed from `deploy/**`

3. **Sidecars / support**
   - `deploy/**`
   - `MA3/**`
   - `docs/**`
   - `packaging/**`
   - `scripts/**`
   - `foundry/tracking/**` remains local generated output and is no longer tracked

## Keep Now

Keep these as the branch source of truth:

- `echozero/application/**`
- `echozero/audio/**`
- `echozero/domain/**`
- `echozero/editor/**`
- `echozero/foundry/**`
- `echozero/inference_eval/**`
- `echozero/models/**`
- `echozero/persistence/**`
- `echozero/pipelines/**`
- `echozero/processors/**`
- `echozero/services/**`
- `echozero/testing/**`
- `echozero/ui/**`
- `run_echozero.py`
- `run_timeline_demo.py`
- EZ2-facing docs and release scripts

Keep generated outputs local only:

- `foundry/tracking/**`
- `artifacts/**`
- `data/echozero.db`

## Removed In This Branch

These legacy app surfaces are no longer present:

- `src/**`
- `ui/**`
- root `main.py`
- root `main_qt.py`
- legacy `tests/unit/**`
- legacy `tests/integration/**`
- `src`-backed application/benchmark tests
- legacy helper scripts tied to `src` or `ui/qt_gui`
- timeline screenshot/demo harness entrypoints not required by the canonical app shell
- generated tracking reports and local DB snapshots
- orphaned utility scripts with no remaining EZ2 ownership

## Test Split Snapshot

Current test posture after the legacy cut:

- EZ2-facing suites remain under `tests/ui/**`, `tests/testing/**`, `tests/foundry/**`, `tests/inference_eval/**`, and the surviving `tests/application/**`
- `src`-only tests have been removed with the legacy tree

The remaining test surface is intended to describe the EZ2 app/runtime only.

## Deletion Order

### Pass 1

Align branch truth:
- root docs
- install docs
- package metadata
- launcher messaging

### Pass 2

Remove or archive obvious non-runtime surfaces:
- `ui/timeline/**` completed
- top-level EZ1 launchers removed

### Pass 3

Delete the old app shell and legacy runtime:
- `ui/**` completed
- `main_qt.py` completed
- `main.py` completed
- `src/**` completed

### Pass 4

Sweep remaining historical docs/manifests so they stop implying EZ1 is runnable.
- legacy EZ1 reference docs removed

## Working Rule

If a file is not needed by:
- `run_echozero.py`
- `echozero/**`
- Foundry
- active EZ2 tests
- release/build sidecars

then it should default to removal or historical quarantine, not preservation.
