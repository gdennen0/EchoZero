# Sync Entrypoint Guardrail Execution Plan

Status: active focused remediation plan
Last updated: 2026-04-29

This plan documents the execution order for closing the canonical app-shell sync-entrypoint gap.
Use [docs/STATUS.md](/Users/march/Documents/GitHub/EchoZero/docs/STATUS.md:1) for current repo truth.
Use [docs/TESTING.md](/Users/march/Documents/GitHub/EchoZero/docs/TESTING.md:1) for proof-lane rules.

## Goal

Ensure the canonical app-shell sync enable and disable path uses the same guardrailed intent handling as the timeline orchestrator.

This work exists to remove the current split where:

- `TimelineOrchestrator.handle(EnableSync/DisableSync)` applies sync and live-sync guardrails
- `TimelineApplication.enable_sync()/disable_sync()` calls the sync service directly
- `AppShellRuntime.enable_sync()/disable_sync()` currently follows the direct app path

## Hard Rules

- Main remains the only live sync truth lane.
- The canonical app path must not bypass application guardrails.
- Proof must include the real app-facing sync path, not only orchestrator-only tests.
- Keep the first slice narrow because the worktree already contains unrelated in-flight changes in sync/runtime files.

## Execution Order

1. Wave 1
   Unify app-level sync routing behind the guarded intent path.
2. Wave 2
   Add regression coverage for the shell-facing app/runtime path.
3. Wave 3
   Run the sync proof lanes and capture any remaining gaps.
4. Wave 4
   Queue follow-on cleanup only after the guarded path is proven green.

## Wave 1
App-Level Routing Fix

Files:

- `echozero/application/timeline/app.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py` only if a small compatibility seam is required

Steps:

1. Route `TimelineApplication.enable_sync()` through `dispatch(EnableSync(...))` or an equivalent shared guarded path.
2. Route `TimelineApplication.disable_sync()` through `dispatch(DisableSync())` or an equivalent shared guarded path.
3. Preserve the current app-facing return contract of `SyncState`.
4. Avoid widening this slice into MA3 adapter cleanup or orchestrator decomposition.

Done when:

- the app-level sync API no longer calls `sync_service.connect()` or `disconnect()` directly
- reconnect and live-sync guardrails apply identically from the canonical shell path

## Wave 2
Shell-Path Regression Coverage

Files:

- `tests/application/test_timeline_runtime_sync_composition.py`
- additional shell/app-flow sync tests only if the existing runtime composition slice is insufficient

Steps:

1. Add regression coverage that proves the app-level sync path applies the same reconnect guardrails as direct orchestrator use.
2. Cover experimental live-sync disable/reset behavior through the app-facing surface if it is not already proven elsewhere.
3. Keep the tests app-boundary focused rather than widget-local.

Done when:

- the direct app/runtime sync API proves the same guardrailed behavior as the orchestrator lane

## Wave 3
Proof

Run:

1. targeted pytest for the modified sync composition and guardrail slices
2. `python -m echozero.testing.run --lane appflow-sync`
3. `python -m echozero.testing.run --lane appflow-protocol` if the sync contract or receive path is affected beyond routing

Report:

- commands run
- pass/fail result
- remaining risks or deferred follow-up

## Wave 4
Queued Follow-On Work

Do not start this until Waves 1-3 are complete.

Queue:

1. conservative startup sync state instead of optimistic connected defaults
2. stricter typed MA3 capability boundary
3. larger orchestrator command-family split

## Current Slice

Active now:

- document this plan
- implement Wave 1 routing
- add the smallest regression coverage needed to prove the canonical shell path no longer bypasses guardrails
