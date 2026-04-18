# Agent Context

This file is the compact knowledge base for code agents working in EchoZero 2.
It condenses the current canonical docs plus the most useful decisions and backlog intent from the planning and audit docs that were removed during cleanup.

This is an orientation layer, not a new competing authority.

## Source Of Truth Order

When documents disagree, use this order:

1. `STYLE.md`
2. `GLOSSARY.md`
3. `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
4. `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
5. `docs/architecture/DECISIONS.md`
6. `docs/APP-DELIVERY-PLAN.md`
7. This file
8. Other docs

## Core Philosophy

Preserved from the old agent-assets layer:

- Best part is no part.
- Simplicity and refinement are key.
- Prefer deletion over new abstraction.
- Prefer explicit, boring, testable code over flexible fallback-heavy code.
- If a behavior matters, make it application-backed and test-backed instead of painting a UI illusion around it.

## Canonical Surfaces

The repo is now centered on EZ2 only:

- `run_echozero.py`: canonical desktop launcher
- `echozero/project.py`: main application entry object
- `echozero/ui/qt/app_shell.py`: current app shell
- `echozero/application/timeline/*`: timeline application contract
- `echozero/application/sync/*`: sync contract, adapters, diff services
- `echozero/infrastructure/sync/ma3_adapter.py`: current MA3 bridge placeholder/infrastructure seam
- `echozero/foundry/*`: Foundry app lane
- `deploy/*`: worker sidecars

Treat removed EZ1 paths as history, not architecture.

## Locked Product Rules

These are the most important preserved rules across current and removed docs.

### Truth model

- Main is truth for playback, export, sync, and freshness comparisons.
- Takes are subordinate: history, rerun results, comparison candidates, promotion/merge inputs.
- There is no active-take truth model.

### Rerun and layer semantics

- Stable pipeline outputs map to stable layers.
- Reruns append takes by default.
- Downstream layers become stale only when upstream main changes.

### Provenance and edit state

Keep these concepts separate:

- provenance: where the data came from
- freshness: whether it still matches current upstream main
- manual modification: whether a user altered it after generation

Do not collapse those into one flag.

### Boundary rules

- Engine is graph execution only; it should not learn timeline/editor/MA3 UI semantics.
- Application/orchestrator maps typed outputs into layers, takes, persistence, and sync consequences.
- UI should depend on application contracts, not invent parallel truth.

### Song versions

- A new `SongVersion` starts as a blank editor state.
- Pipeline configs/settings carry forward.
- Prior processed results do not silently become truth in the new version.

### MA3 sync

- MA3 sync is main-only.
- Non-main takes do not sync directly.
- Pull/apply behavior must preserve confirm-vs-apply safety.

### UI / FEEL

- `echozero/ui/FEEL.py` owns tuning constants.
- Do not scatter timeline dimensions or interaction constants as magic numbers.

### Repo hygiene

- Generated outputs stay local: `artifacts/**`, `foundry/tracking/**`, local DB/runtime state.
- If a file is not needed by canonical EZ2 runtime, tests, packaging, or preserved design context, it should not stay in the repo.

## Architecture Snapshot

The current layered mental model is:

1. Domain: frozen types and graph invariants
2. Engine/editor core: pipeline, coordinator, execution, staleness
3. Persistence: SQLite working dir + `.ez` archive lifecycle
4. Services/application: orchestrator, waveform, sync contracts
5. Application shell: `Project`, app-shell runtime composition
6. UI: PyQt6 Stage Zero surfaces

The most important live file clusters are:

- `echozero/domain/*`
- `echozero/editor/*`
- `echozero/persistence/*`
- `echozero/services/*`
- `echozero/application/timeline/*`
- `echozero/application/sync/*`
- `echozero/ui/qt/*`
- `echozero/foundry/*`

## Preserved Lessons From Removed Docs

These came from deleted audits, trackers, and plans, but are still useful.

### Distillation / audit lessons

Historical audits repeatedly identified these as high-risk regression zones:

- active-take truth leakage
- FEEL drift and magic-number UI sizing
- rebuild/remap intent not being persisted
- main-only sync not being proven end-to-end
- terminology drift around takes/variations

Even when these issues are closed in current code, treat them as "must not regress" areas.

### UI planning lessons

Old UI planning docs were noisy, but their useful signal was:

- keep timeline work contract-first, not paint-first
- treat take rows as subordinate lanes
- keep selection identity rich enough to know layer, take, and event
- keep stale/provenance/manual-edit indicators application-backed
- manual scroll should not fight follow-mode behavior
- visible controls should only exist when they are wired for real

When touching timeline UX, check these hotspot areas first:

- expansion/take-row authority
- take-aware selection state
- header provenance/status chips
- follow-scroll interruption
- mute/solo wiring
- transport/ruler interaction
- zoom behavior
- event drag/resize/transfer flows

### Verification lessons

The deleted UI verification plan is worth preserving conceptually:

- prove behavior at the contract layer first
- prove app-level flows, not only isolated widgets
- use realistic data when a visible timeline behavior changes
- run perf guardrails for hot-path changes
- keep reviewer-visible evidence for release/signoff work

The helper scripts around that plan were removed, but the proof-lane idea remains good.

### Tracker / alpha lessons

The removed tracker and alpha docs converged on the same remaining release-signoff work:

- packaged manual QA evidence
- real MA3 hardware validation
- one clean visible operator end-to-end proof sequence

That is still the most durable "what remains" summary from the old planning layer.

### ShowManager redesign lesson

The older ShowManager sync redesign doc had one useful enduring principle:

- one entity model
- one orchestration point
- thin UI

Even if the concrete implementation has changed, that simplification bias is still the right direction when touching MA3 sync.

## Current Proof Lanes

These are the most useful live verification surfaces to remember:

- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane appflow-sync`
- `python -m echozero.testing.run --lane appflow-osc`
- `python -m echozero.testing.run --lane appflow-protocol`
- `python -m echozero.testing.run --lane appflow-all`
- `python -m echozero.testing.run --lane gui-lane-b`

Also important:

- CI sync-receive lane in `.github/workflows/test.yml`
- perf benchmark: `tests/benchmarks/benchmark_timeline_phase3.py`
- repo hygiene guard: `scripts/check_repo_hygiene.py`

## Foundry Context

Foundry is real product code, not just an experiment.
The durable workflow is:

1. create or ingest dataset
2. plan splits / balance
3. create run
4. train
5. export artifact
6. validate artifact compatibility

See:

- `docs/FOUNDRY-TRAINING.md`
- `echozero/foundry/contracts/*`
- `echozero/foundry/services/*`

## MA3 Context

MA3 knowledge worth preserving for agents:

- plugin docs live under `MA3/docs/*`
- critical gotchas live in `MA3/MA3_INTEGRATION_PITFALLS.md`
- current repo rule is still main-only sync
- if you change sync behavior, keep the UI thin and the app-layer contract explicit

## Current Likely Remaining Work

Confirmed high-value remaining work from current canonical docs:

- maintain app-first acceptance and release-smoke discipline
- complete packaged manual QA evidence
- complete real MA3 hardware validation
- preserve main-only sync guardrails while expanding transfer behavior

Potential future work implied by preserved design context:

- richer provenance/source inspection
- more complete timeline manipulation flows
- continued FEEL-backed UI hardening
- stronger operator-proof artifacts for release signoff

Treat these as backlog hints, not guaranteed missing features.

## Deep References

Use these when you need more than the summary:

- architecture overview: `docs/ARCHITECTURE.md`
- timeline truth model: `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- implementation order: `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
- architectural tradeoffs: `docs/architecture/DECISIONS.md`
- delivery/release gates: `docs/APP-DELIVERY-PLAN.md`
- project API shape: `docs/PROJECT-CLASS.md`
- Foundry workflow: `docs/FOUNDRY-TRAINING.md`
- MA3 references: `MA3/README.md`
- cleanup/keep posture: `docs/EZ2-CODEBASE-CLEANUP-MAP.md`
