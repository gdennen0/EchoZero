# EchoZero Agent Guide

Use this file as the minimal startup contract for coding agents in this repo.
Do not preload a long doc chain at session start. Read deeper docs only when the task needs them.
If this file conflicts with code or a canonical doc, code and the canonical doc win.

## Default Startup

- Stay in this file unless the task needs more context.
- Open `docs/STATUS.md` when you need current repo truth or are about to touch canonical app, timeline, sync, or Foundry paths.
- Open `STYLE.md` when editing Python or adding files.
- Open `GLOSSARY.md` when naming domain concepts, UI text, or docs.
- Do not open `docs/architecture/DECISIONS.md` by default. Use it only for a specific decision, topic, or decision ID.
- Open subsystem docs only for the subsystem you are changing.

## Canonical Repo Truth

- `run_echozero.py` is the canonical EZ2 desktop entrypoint.
- `echozero/` is the canonical EZ2 codebase.
- `echozero/ui/qt/app_shell.py` is the main Stage Zero shell surface.
- `echozero/application/timeline/*` is the timeline app contract.
- `echozero/application/sync/*` and `echozero/infrastructure/sync/ma3_adapter.py` are the sync boundary.
- `echozero/foundry/*` is the Foundry lane.
- EZ1 code was removed from this branch. Use git history if you need it.

## Locked Rules

- Main is truth. Takes are subordinate history and candidate lanes, never alternate live truth.
- Staleness changes only when upstream main changes.
- MA3 sync is main-only.
- `SongVersion` starts blank. Configs carry forward; processed results do not.
- The engine stays ignorant of UI, editor, and MA3 semantics.
- `echozero/ui/FEEL.py` owns UI tuning constants. Do not scatter magic numbers through timeline code.
- Generated outputs do not belong in git: `artifacts/**`, `foundry/tracking/**`, local DB snapshots, or local runtime state.

## Proof Rules

- App-facing work is not done until it is proven through the real app path, not only helpers.
- Sync changes need app-boundary guardrail tests and the sync-receive lane.
- Timeline or UI changes need application and UI contract coverage; run perf guardrails for hot-path work.
- Release-affecting changes need packaging and smoke consideration.
- Parallel worker use must stay bounded and follow `docs/WORKER-ROLES.md`.
- Long-running delegated work must emit status heartbeats and stuck-agent escalation per `docs/AGENT-WORKFLOW.md`.

## Human-Path Demo Rule

- Demo videos and non-deterministic functional tests must use real human paths only.
- Do not fake demo claims with mock analysis services, fake audio streams, direct audio-callback driving, widget-presentation injection, or bypassed launcher and dialog flows.
- If a proof path uses simulation or injected state, label it as non-human-path proof.
- For playback or pipeline demo claims, use real EZ runtime actions with real input assets and report what was real versus synthetic.

## Do Not Reintroduce

- `src/**`, `ui/qt_gui/**`, `main.py`, `main_qt.py`
- active-take truth behavior
- widget-only workflow logic that bypasses application contracts
- tracked generated artifacts or local runtime state
- large sidecar agent frameworks when a small local doc will do

## Read On Demand

- Current truth: `docs/STATUS.md`
- Deep orientation: `docs/AGENT-CONTEXT.md`
- First principles: `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- Delivery and implementation plans: `docs/UNIFIED-IMPLEMENTATION-PLAN.md`, `docs/APP-DELIVERY-PLAN.md`
- Decision log: `docs/architecture/DECISIONS.md`
- UI direction: `docs/UI-STANDARD.md`
- Worker orchestration: `docs/WORKER-ROLES.md`, `docs/AGENT-WORKFLOW.md`, `docs/OPENCLAW-CODEX-PROMPTING.md`
- Verification: `docs/TESTING.md`
- Timeline: `echozero/application/timeline/README.md`, `echozero/ui/qt/timeline/README.md`
- Presentation: `echozero/application/presentation/README.md`
- Foundry: `echozero/foundry/README.md`, `docs/FOUNDRY-TRAINING.md`
- MA3: `MA3/README.md`, `MA3/MA3_INTEGRATION_PITFALLS.md`
- Packaging: `docs/packaging/PACKAGING.md`
- Repo cleanup boundaries: `docs/EZ2-CODEBASE-CLEANUP-MAP.md`
