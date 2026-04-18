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
8. `docs/UI-STANDARD.md`
9. `docs/WORKER-ROLES.md`
10. `docs/AGENT-WORKFLOW.md`
11. `docs/TESTING.md`

Use subsystem docs only when relevant:
- Foundry: `docs/FOUNDRY-TRAINING.md`
- MA3: `MA3/README.md`, `MA3/MA3_INTEGRATION_PITFALLS.md`
- Packaging: `docs/packaging/PACKAGING.md`
- Repo cleanup boundaries: `docs/EZ2-CODEBASE-CLEANUP-MAP.md`
- UI direction: `docs/UI-STANDARD.md`, `docs/UI-CLEANUP-MAP.md`, `docs/UI-ENGINE-REDEVELOPMENT-PLAN.md`

## Canonical Repo Truth

- `run_echozero.py` is the canonical EZ2 desktop entrypoint.
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
- Long-running delegated work must emit visible status heartbeats and a stuck-agent escalation per `docs/AGENT-WORKFLOW.md`.

## Human-Path Demo Rule

- Demo videos and non-deterministic functional tests must use real human paths only.
- Do not use mock analysis services, fake audio streams, direct audio-callback driving, or widget-presentation injection for demo claims.
- Do not bypass launcher actions, widget interactions, runtime intents, dialog flows, or application contracts to fake user behavior.
- If a proof path uses simulation or state injection, it is not a human-path demo and must be labeled accordingly.
- For demo claims about playback or pipeline behavior, use real EZ runtime actions with real input assets and report what was real vs synthetic.

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
- Desktop UI North Star: `docs/UI-STANDARD.md`
- Worker orchestration rules: `docs/WORKER-ROLES.md`
- Default orchestration/delegation workflow: `docs/AGENT-WORKFLOW.md`
- Verification command map: `docs/TESTING.md`
- Current cleanup/keep posture: `docs/EZ2-CODEBASE-CLEANUP-MAP.md`
