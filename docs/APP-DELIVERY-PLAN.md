# EchoZero App Delivery Plan (App-First)

Status: active canonical delivery and release plan
Last verified: 2026-04-21

_Last updated: 2026-04-12_

This plan defines how EchoZero moves to a canonical app workflow with deterministic packaging and release validation.

## Current Status (2026-04-13)

Complete now:
- canonical launcher + compatibility delegate are in place
- app shell runtime lifecycle, launcher commands, and unsaved-close guard are implemented
- built-in appflow lanes now cover `appflow`, `appflow-sync`, `appflow-osc`, `appflow-protocol`, and `appflow-all`
- deterministic MA3 simulator and OSC loopback coverage are in place for Tier A/B app-flow validation
- protocol fixtures and receive-path integration tests are in place for Tier C validation
- release build, packaged smoke, and one-command local gate scripts are in place

Remaining:
- manual UX QA walkthrough on milestone checkpoints
- validation against real MA3 hardware before final release signoff

If this document conflicts with first-principles or sync safety contracts, those higher-order contracts win:
- `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`

---

## 1) Intent

Make EchoZero operate as a production app, not a demo surface:
- one canonical app entrypoint
- one canonical packaging path
- one canonical release-smoke path
- app-level flows as acceptance truth

---

## 2) Locked Rules

1. **App-first acceptance**
   - A feature is not complete unless it passes through main app flow.
   - Demo-only proof is insufficient.

2. **Main is truth**
   - Main take/lane is authoritative for playback/export/sync.
   - Takes remain subordinate candidates/history.

3. **MA3 sync safety**
   - MA3 writes remain main-only.
   - Confirm-vs-Apply semantics remain enforced for pull.

4. **Single release contract**
   - No test release unless packaged-build smoke lane passes.

5. **Single app launcher**
   - `run_echozero.py` is the only desktop app entrypoint.

---

## 3) Canonical Commands

## Dev run (main app)
- `C:/Users/griff/EchoZero/.venv/Scripts/python.exe run_echozero.py`

## Build test release
- `powershell -File scripts/build-test-release.ps1`

## Smoke packaged build
- `powershell -File scripts/smoke-test-release.ps1`
These are the canonical operator commands for local run, packaging, and packaged smoke validation.

---

## 4) Delivery Phases

## Phase A — App Entry + Lifecycle Baseline

Deliverables:
- `run_echozero.py` canonical launcher for the current Stage Zero shell flow
- Stable launcher contract for local run and packaged smoke

Exit criteria:
- App boots from `run_echozero.py`
- No alternate desktop launcher exists for demo/test app startup

## Phase B — App-Flow Test Harness

Deliverables:
- App-level test harness utilities
- Automated tests for high-value user flows in main app shell

Required initial app-flow tests:
- open/create project
- timeline interaction path initialization
- push flow: intent -> diff gate -> apply path wiring
- pull flow: intent -> target requirement -> diff gate -> apply path wiring
- save/reopen state persistence sanity

Exit criteria:
- App-flow lane exists and is green in local run and CI

## Phase C — Packaging Baseline

Deliverables:
- deterministic packaging script (`scripts/build-test-release.ps1`)
- stable output directory/versioning conventions under `artifacts/releases/test/`
- release artifact bundling (copied app folder, zip, metadata JSON)

Exit criteria:
- Repeatable test build artifact generation on dev machine

## Phase D — Packaged Smoke Lane

Deliverables:
- packaged app smoke script (`scripts/smoke-test-release.ps1`)
- pass/fail JSON summary in the release folder

Required smoke checks:
- launch packaged app
- auto-exit via `--smoke-exit-seconds`
- clean shutdown with exit code `0`

Exit criteria:
- Packaged smoke lane is green and documented

## Phase E — Demo Surface Demotion

Deliverables:
- docs updates marking timeline demos as non-primary
- CI/review checklist references app-first lanes

Exit criteria:
- Feature signoff no longer accepts demo-only evidence

---

## 5) Test Matrix (Source of Delivery Truth)

| Lane | Scope | Trigger | Required For | Owner | Pass Evidence |
|---|---|---|---|---|---|
| Unit/Contract | Intents, orchestrator, sync contracts | Every PR | Merge | Dev | `pytest` contract lanes green |
| App-Flow | Main app user paths | Every PR touching UX/app flow | Merge | Dev | App harness test report |
| Transfer Safety | Push/pull diff-gate + main-only boundaries | Every transfer/sync PR | Merge | Dev | Transfer suite + app-flow coverage |
| Packaging Build | Build distributable app | Release prep + weekly | Test release | Dev | Build log + artifact manifest |
| Packaged Smoke | Launch/use packaged app in real flow | Every test release candidate | Test release publish | Dev | Smoke JSON/log bundle |
| Manual UX QA | Human walkthrough in app | Milestone checkpoints | Product signoff | Griff + Dev | Checklist + notes/screens |

---

## 6) Acceptance Gates for Test Release

A test release can ship only if all are true:
1. Contract lanes green
2. App-flow lane green
3. Transfer safety lane green (including main-only constraints)
4. Packaging build lane green
5. Packaged smoke lane green
6. Manual app walkthrough complete (critical flow checklist)

---

## 7) Remaining Backlog

1. Complete manual UX QA walkthrough against the packaged app path
2. Validate transfer and receive flows against real MA3 hardware

---

## 8) Anti-Spaghetti Checklist (PR Gate)

For every app/sync/transfer PR, verify:
- [ ] Workflow logic is not trapped in widget-only code paths
- [ ] Behavior is reachable through app shell path
- [ ] First-principles truth model is preserved
- [ ] Main-only sync boundary remains enforced
- [ ] App-flow tests updated where behavior changed
- [ ] Packaging/smoke impact considered (if release-affecting)

If any item fails, PR is not ready.
