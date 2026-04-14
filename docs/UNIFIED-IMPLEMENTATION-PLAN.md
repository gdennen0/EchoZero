# EchoZero Unified Implementation Plan (Canonical)

_Last updated: 2026-04-08_

This is the single source-of-truth implementation plan for current EchoZero development.

It unifies:
- original distillation intent
- current first principles
- application/API contract boundaries
- UX + FEEL constraints
- real-world DAW behavior references
- deprecations
- execution order

If any other plan doc conflicts with this file, this file wins.

---

## 1) Source-of-Truth Order (Highest → Lowest)

1. **Original distillation intent**
   - Upstream authority: `memory/echozero-distillation/DISTILLATION.md` (not vendored in this repo)
   - Local proxy: `docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`

2. **First principles**
   - `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`

3. **Application/API contract**
   - Timeline app contract:
     - `echozero/application/timeline/models.py`
     - `echozero/application/timeline/intents.py`
     - `echozero/application/timeline/orchestrator.py`
     - `echozero/application/timeline/assembler.py`
     - `echozero/application/timeline/queries.py`
     - `echozero/application/timeline/repository.py`
   - Sync contract + implementation boundary:
     - `echozero/application/sync/service.py` (contract)
     - `src/features/show_manager/application/sync_system_manager.py` (current concrete path)

4. **UX contract**
   - `docs/UI-INTERACTION-CLOSURE-PLAN-2026-04-05.md`
   - `docs/UI-VERIFICATION-AND-ACCEPTANCE-PLAN-2026-04-05.md`
   - `docs/OBJECT-INSPECTOR-MILESTONE-PLAN-2026-04-06.md`

5. **FEEL contract (visual + interaction constants)**
   - `echozero/ui/FEEL.py`

6. **Real-world DAW evidence (tie-breaker for behavior feel, not truth semantics)**
   - `docs/LD-WORKFLOW.md`
   - DAW precedent used in existing planning: Ableton / Bitwig / REAPER / Logic / Pro Tools

7. **Evidence/audits (informing, not authoritative)**
   - `docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`
   - `docs/UI-CONTRACT-AUDIT-2026-04-03.md`
   - `docs/architecture/UI-BLOCK-AUDIT-2026-04-02.md`

---

## 2) Locked Decisions (Non-Negotiable)

### Core model
- **Main is truth.**
- **Takes are subordinate** (candidates/history/comparison), never alternate live truth.
- **No active-take truth model** in app or UI.

### Pipeline/application boundary
- Engine is app-agnostic; returns typed outputs.
- Application maps outputs into stable layers/takes and persists provenance/freshness state.

### Sync boundary
- MA3 sync is **main-only**.
- Non-main/take-lane data cannot become MA3 truth directly.
- Sync writes now fail hard on malformed event metadata.

### Song versioning
- New SongVersion starts as blank slate.
- Configs copy forward.
- Rebuild plan is persisted and test-backed.

### FEEL ownership
- FEEL owns dimensions/spacing/timing/tuning constants.
- FEEL does not own truth semantics or persistence rules.

### Hygiene + traceability
- Generated artifacts are not tracked.
- Decision/principle mapping is required in PR flow.

---

## 3) Deprecation Register (Do Not Reintroduce)

1. **OverrideStore** → replaced by Take System.
2. **Legacy variation entity model** (branch-style variation primitive) → standardized on Take model.
3. **PromotedParam** → replaced by Knob system.
4. **PortProxy / legacy builder flow** → replaced by explicit `p.add()` construction.
5. **Pre-engine ingest shadow pipeline** → removed; Add Song = file copy + config.
6. **Active-take truth behavior** in timeline app/UI.
7. **Tracked runtime artifacts in Git** (`artifacts/*`, run outputs, transient training outputs).
8. **Parallel “source-of-truth” planning docs** as competing authorities.

---

## 4) Current State (Audit Reconciliation)

### Closed
- A4 main-truth leak (closed)
- A5 rebuild plan persistence (closed)
- A7 FEEL baseline integration for timeline shell (closed baseline)
- A9 terminology drift (closed)
- Repo hygiene + CI guardrails + PR traceability (closed)

### Closed (with release-signoff follow-through)
- **A6 main-only sync proof (automated end-to-end)**
  - Contract/unit/appflow/protocol evidence is green for main-only guardrails.
  - Remaining alpha claim work is operational signoff: packaged manual QA + real MA3 hardware validation + one visible operator E2E proof sequence.

### Operational but still maturing
- Real-data stems/classifier flow is running, but classifier-in-place behavior still needs stronger production-model parity evidence.

---

## 5) Execution Order (Strict)

## Phase 0 — Freeze + Contract Completion (NOW)
Goal: finish architecture debt before feature expansion.

Tasks:
1. Keep A6 automated end-to-end proof green:
   - main vs non-main fixture matrix
   - push, pull, divergence, reconnect, empty-state behaviors
   - assertion that only main-authorized events produce MA3 write commands
2. Maintain failure-mode tests for malformed sync payloads and mixed metadata quality.
3. Keep timeline/sync feature freeze active until release-signoff checklist items are closed.

Exit criteria:
- A6 remains closed in tracker and unified plan.
- Sync contract lane green and required in CI.
- Remaining alpha signoff work is explicitly tracked (packaged manual QA + real MA3 hardware + visible operator E2E proof).

## Phase 1 — Consolidate Sync Surface
Goal: reduce split-brain between app contract and legacy implementation.

Tasks:
1. Define explicit adapter seam from `echozero/application/sync/service.py` to current concrete implementation.
2. Move sync behavior assertions to contract tests at app boundary (not only manager-unit internals).
3. Remove duplicated/implicit sync rules from UI layer.

Exit criteria:
- Single documented sync boundary.
- App-layer contract tests enforce behavior independent of widget internals.

## Phase 2 — UX/FEEL Alignment Hardening
Goal: keep interaction quality while preserving contract safety.

Tasks:
1. Expand FEEL-backed tests beyond shell dimensions where magic values can drift.
2. Keep object inspector fully contract-driven (no widget-local semantic truth).
3. Ensure real-data walkthrough evidence accompanies visible behavior changes.

Exit criteria:
- FEEL/UI contract tests and walkthrough proof lanes are stable and repeatable.

## Phase 3 — Sync Receive Protocol Hardening
Goal: prove EZ2 reliably receives/parses MA3 OSC plugin payloads with simple, deterministic tests.

Tasks:
1. Encode MA3 plugin payload shapes as protocol tests (trackgroups, tracks, events, track.changed).
2. Validate parser robustness for embedded delimiters and nested payload structures.
3. Verify receive-path compatibility from communication service through event handler/sync manager entrypoints.
4. Keep a dedicated sync-receive lane green and required before sync-affecting merges.

Exit criteria:
- Sync receive protocol suite is green and stable.
- Known MA3 plugin payloads are covered by tests.
- No unresolved parser ambiguity for current plugin protocol.

## Phase 4 — Feature Expansion Re-open
Goal: re-enable net-new features without reintroducing drift.

Prerequisites:
- Phase 0-3 exit criteria all green.
- Tracker readiness flips to ✅.

Rule:
- Every new feature must map to explicit decision/principle and pass required proof lanes.

---

## 6) Acceptance Gates

### Gate A — Truth/contract integrity
- No active-take truth resurrection.
- No UI-only semantic state pretending to be truth.

### Gate B — Sync safety
- Main-only writes provable in tests.
- Missing/invalid metadata fails hard for MA3 write path.

### Gate C — FEEL + UX coherence
- FEEL constants own rendering/interactions where specified.
- No hardcoded magic regressions in covered timeline surfaces.

### Gate D — Evidence quality
- Contract/unit lane required.
- Real-data + walkthrough evidence required for visible behavior changes.
- Perf guardrails required for hot-path changes.

### Gate E — Repo hygiene
- No generated/runtime files tracked.
- Hygiene script and CI checks pass.

---

## 7) Decisions Needed (Only If Ambiguity Reappears)

1. If metadata schema expands for sync writes, should unknown fields be ignored or rejected?
   - Current direction: reject missing required metadata; tolerate additive known-safe extras.

2. If future v1 workflow toggles between push/pull-only and live-sync modes per layer, keep one explicit mode boundary per layer at all times.

---

## 8) Working Rule

Build order is:
1) truth model
2) app contract
3) sync safety
4) FEEL/UX polish
5) feature growth

Never reverse this order for convenience.
