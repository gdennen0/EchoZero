# EchoZero Development Tracker (Canonical)

_Last updated: 2026-04-09_

This is the live execution status board.
Canonical implementation authority is: `docs/UNIFIED-IMPLEMENTATION-PLAN.md`.
If this tracker conflicts with the canonical implementation plan, update this tracker to match it immediately.

---

## 1) Current Direction Check (Go / No-Go)

**Direction:** ✅ Correct
- Main take truth model is preserved
- UI contract is becoming explicit and shared
- Real-data playback is part of verification (not mock-only)

**Readiness to add features:** ✅ Ready for next feature tranche
- P0 contract + hygiene gate is green. Continue with contract-first execution discipline.

---

## 2) Planning Decision Reconciliation (Reds/Yellows)

Reference baseline: `docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`

| ID | Topic | Prior | Current | Evidence | Action |
|---|---|---:|---:|---|---|
| A4 | Main-is-truth vs active-take truth leak | 🔴 | ✅ Closed | `echozero/application/timeline/orchestrator.py` (`SelectTake` selection-only), `echozero/application/timeline/assembler.py` (main take drives parent row), tests in `tests/application/test_timeline_assembler_contract.py` | none |
| A7 | FEEL contract drift / magic numbers | 🔴 | ✅ Closed (baseline) | `tests/ui/test_timeline_feel_contract.py` green | keep FEEL as required gate for UI changes |
| A9 | Legacy terminology drift (now standardizing on Take) | 🔴 | ✅ Closed | Documentation updated to use Take terminology for timeline variation model | keep this as an ongoing docs hygiene check |
| A5 | SongVersion rebuild_plan persistence | 🟡 | ✅ Closed | schema v4 + repo persistence + session update path + round-trip test (`tests/test_song_version_rebuild_plan.py`) | none |
| A6 | Sync boundary (main-only) proof | 🟡 | ✅ Closed | Sync harness covers apply/push path (`tests/unit/test_ma3_event_contract.py`), pull/reconnect divergence state handling (`tests/unit/test_multitrack_sync_coalesce.py`), and external MA3 fixture replay at app boundary (`tests/unit/test_ma3_fixture_replay.py`) | keep live-console soak as non-blocking operational verification |
| A10 | Real-data stems progression | 🟡 | ✅ Operational | real-data runs + visual proof loops active | continue during new feature work |

---

## 3) Phase-0 Closure Status (Pre-Feature Gate)

### P0 — Contract closure
- [x] Persist `SongVersionRecord.rebuild_plan` in SQLite and verify round-trip tests
- [x] Add explicit sync-boundary tests proving main-only sync semantics
- [x] Finish terminology sweep: replace legacy timeline variation terms with Take terminology

### P0 — Repo hygiene
- [x] Keep working tree clean (no generated artifacts/log churn in Git status)
- [x] Resolve local ACL-blocked temp folder warning (`.pytest-foundry-tmp/`) so `git status` is warning-free
- [x] Remove tracked runtime artifacts (`artifacts/*`) from source control

### P1 — Decision traceability
- [x] For each new feature, add one-line mapping: "which principle/decision does this implement?" (PR template added)
- [x] Keep this tracker as the only active backlog/status file
- [x] Add CI repo hygiene guard (`scripts/check_repo_hygiene.py` + workflow)

---

## 4) Active Phase (Phase 1 from unified plan): Sync Surface Consolidation

- [x] Add concrete app-layer `SyncService` adapters (`InMemorySyncService`, `MA3SyncAdapter`) in `echozero/application/sync/adapters.py`
- [x] Add adapter contract tests (`tests/application/test_sync_adapters.py`)
- [x] Remove duplicated demo-only sync service implementation from timeline demo app
- [x] Wire `MA3SyncAdapter` into timeline runtime composition path (`build_demo_app` / Stage Zero driver wiring)
- [ ] Lift remaining sync behavior assertions from manager-internals to app-boundary integration tests (legacy show-manager path still manager-heavy)

---

## 5) Cleanup Completed in This Pass
- Removed large untracked noise from repo root and generated runtime output directories (`artifacts/*`, `foundry/runs/*`, temp debug files, ad-hoc screenshots, abandoned local worktree folders under repo).
- Restored generated tracked files that were modified by local demo/training runs.
- Hardened `.gitignore` to prevent recurrence of local noise artifacts and run outputs.
- Added CI hygiene gate to block tracked generated/runtime outputs.
- Removed previously tracked timeline/foundry artifact files from Git history tip.
- Added MA3 payload normalization contract tests and legacy editor entity migration fix for flexible data intake.

---

## 6) Operating Rules (to stay clean)

1. No new feature implementation starts unless P0 items are green.
2. Any UI change must pass:
   - timeline shell tests
   - FEEL contract test
   - relevant application contract tests
3. Any sync-facing change must include a main-only boundary assertion.
4. Generated outputs live outside tracked source unless intentionally versioned.

---

## 7) Socratic "What If" Questions (Decision Check Before Feature Expansion)

1. **What if** we freeze all new features for 48 hours and only close P0 contract gaps — would that increase velocity more than shipping one more feature now?
2. **What if** sync ever reads a non-main take in one edge path — do we have a failing test today that would catch it immediately?
3. **What if** we require each PR to cite the principle/decision it implements — does that kill drift early?
4. **What if** we split Foundry runtime outputs to an external data root by default — do we reduce repo entropy permanently?
5. **What if** we define a hard acceptance gate: "no warning/noise in `git status`, no feature merge" — would this prevent spaghetti from re-accumulating?

---

## 8) Next Checkpoint

When P0 is complete, update this file and mark:
- **Direction:** ✅ Correct
- **Readiness:** ✅ Add next feature tranche
