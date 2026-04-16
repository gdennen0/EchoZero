# EchoZero Agent Guide

Use this file as the fast-start orientation for coding agents in this repo.
It is a summary layer, not a competing architecture spec.
If this file conflicts with a canonical doc, the canonical doc wins.

## Read Order

1. `STYLE.md`
2. `GLOSSARY.md`
3. `docs/AGENT-CONTEXT.md`
4. `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
5. `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
6. `docs/architecture/DECISIONS.md`
7. `docs/APP-DELIVERY-PLAN.md`
8. `docs/WORKER-ROLES.md`
9. `docs/TESTING.md`

Use subsystem docs only when relevant:
- Foundry: `docs/FOUNDRY-TRAINING.md`
- MA3: `MA3/README.md`, `MA3/MA3_INTEGRATION_PITFALLS.md`
- Packaging: `docs/packaging/PACKAGING.md`
- Repo cleanup boundaries: `docs/EZ2-CODEBASE-CLEANUP-MAP.md`

## Canonical Repo Truth

- `run_echozero.py` is the canonical EZ2 desktop entrypoint.
- `run_timeline_demo.py` is a compatibility shim, not the primary app path.
- `echozero/` is the canonical EZ2 codebase.
- `echozero/ui/qt/app_shell.py` is the main Stage Zero shell surface.
- `echozero/application/timeline/*` holds the timeline app contract.
- `echozero/application/sync/*` and `echozero/infrastructure/sync/ma3_adapter.py` hold the sync boundary.
- `echozero/foundry/*` is the Foundry lane.
- EZ1 code was removed from the branch. If you need it, use git history, not new code paths.

## Locked Rules

- Main is truth. Takes are subordinate history/candidates, never alternate live truth.
- Staleness only changes when upstream main changes.
- MA3 sync is main-only.
- SongVersion starts blank; configs carry forward, processed results do not.
- Engine stays ignorant of UI/editor semantics.
- FEEL owns UI tuning constants. Do not spread magic numbers through timeline code.
- Generated outputs do not belong in git: `artifacts/**`, `foundry/tracking/**`, local DB snapshots.

## Proof Expectations

- App-facing work is not done until it is proven through the app path, not just demo helpers.
- Sync changes need app-boundary guardrail tests and the sync-receive lane.
- Timeline/UI changes need application and UI contract coverage; run perf guardrails for hot-path changes.
- Release-affecting changes need packaging and smoke consideration.
- Parallel worker use must stay bounded and follow `docs/WORKER-ROLES.md`.

## Do Not Reintroduce

- `src/**`, `ui/qt_gui/**`, `main.py`, `main_qt.py`
- active-take truth behavior
- widget-only workflow logic that bypasses application contracts
- tracked generated artifacts or local runtime state
- large sidecar agent frameworks when a small local doc will do

## Where To Look Next

- Deep context and preserved design intent: `docs/AGENT-CONTEXT.md`
- Architecture decisions and tradeoffs: `docs/architecture/DECISIONS.md`
- Delivery and release gates: `docs/APP-DELIVERY-PLAN.md`
- Worker orchestration rules: `docs/WORKER-ROLES.md`
- Verification command map: `docs/TESTING.md`
- Current cleanup/keep posture: `docs/EZ2-CODEBASE-CLEANUP-MAP.md`
