# Distillation Conformance Audit v2 — 2026-04-04

## Scope
Audit current EchoZero2 code against distillation-level architecture intent (not just UI), focusing on:
- truth model (main vs takes)
- pipeline/application boundary
- output persistence contract
- freshness/staleness semantics
- song-version policy
- sync boundary assumptions
- FEEL/UI contract integration
- terminology consistency

## Source of Truth Reviewed
- `memory/echozero-distillation/DISTILLATION.md`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `echozero/ui/FEEL.py`
- current implementation in `echozero/services/*`, `echozero/application/timeline/*`, `echozero/persistence/*`, `echozero/ui/qt/timeline/*`

---

## Scoreboard
- ✅ Green: 4
- 🟡 Yellow: 3
- 🔴 Red: 3

---

## Detailed Findings

### A1) Pipeline/application boundary (engine ignorance)
**Status: ✅ Green**

Evidence:
- Distillation principle: pipeline is data and app maps outputs (`DISTILLATION.md:658`).
- `Orchestrator` resolves typed outputs and routes by target (`echozero/services/orchestrator.py:250,302-307`).
- Generated layer creation is app-layer persistence concern (`echozero/services/orchestrator.py:315,385,449`).

Assessment:
- Core execution remains separated from UI/editor semantics.

---

### A2) Stable output contract (EventData + AudioData persistence)
**Status: ✅ Green**

Evidence:
- Event outputs mapped to `layer_take`, audio outputs mapped to `song_version` target (`echozero/services/orchestrator.py:305-307`).
- Both event and audio paths create/append takes to stable layer names (`echozero/services/orchestrator.py:361+`, `421+`).

Assessment:
- This aligns with first-principles stable output mapping and rerun append behavior.

---

### A3) Staleness rule: only upstream main change
**Status: ✅ Green (service-level)**

Evidence:
- Explicit rule helper: stale only when main ID changes (`echozero/services/dependencies.py:29-37`).
- Dependents stale-marked only on main-change gate (`echozero/services/dependencies.py:39-57`).
- Promotion action applies stale propagation through helper (`echozero/services/take_actions.py:53`).

Assessment:
- Rule is correctly encoded in app services.

---

### A4) Truth model (main is truth; no active-take truth)
**Status: 🔴 Red**

Expected:
- First principles require no active-take truth model (`TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`, sections 1 + UI implication).

Current violation:
- `SelectTake` mutates `layer.active_take_id` (`echozero/application/timeline/orchestrator.py:52,123`).
- Assembler renders from `_get_active_take(layer)` (`echozero/application/timeline/assembler.py:53,138`) and even labels subtitle from active take (`:70`).

Impact:
- Reintroduces alternate-active truth semantics under application timeline layer.

Required fix:
- `SelectTake` should be selection-only.
- Assembler should always render main truth lane, takes as subordinate rows.

---

### A5) SongVersion policy (blank slate + config carry-forward)
**Status: 🟡 Yellow**

Expected:
- New SongVersion blank editor state; copy configs; preserve rebuild/remap intent (`TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`, section 7).

Current state:
- Config-copy path exists in `add_song_version` and rebuild plan is constructed (`echozero/persistence/session.py:583`).
- But rebuild plan is only assigned to local `version` (`:588`) and no `song_versions.update(version)` is called before commit.

Impact:
- Rebuild plan likely not persisted, weakening explicit remap/rerun intent tracking.

Required fix:
- Persist updated SongVersionRecord after rebuild_plan mutation.
- Add regression test for persisted `rebuild_plan` round-trip.

---

### A6) MA3 sync boundary (main only)
**Status: 🟡 Yellow / Incomplete**

Expected:
- MA3 sync reads/writes main only (`TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`, section 6).

Current state:
- Boundary is present conceptually, but concrete sync implementation is still abstract (`echozero/application/sync/service.py`).

Impact:
- Cannot fully verify end-to-end compliance yet.

Required fix:
- Add concrete sync adapter tests proving non-main takes are excluded.

---

### A7) FEEL contract integration (no magic numbers)
**Status: 🔴 Red**

Expected:
- FEEL claims all UI rendering/interaction should reference FEEL constants (`echozero/ui/FEEL.py:1-4`).

Current violations:
- Timeline UI hardcodes dimensions (`echozero/ui/qt/timeline/widget.py:141,143,144`).
- FEEL values diverge from runtime values (`echozero/ui/FEEL.py:82-83` vs widget constants).
- No FEEL imports/references in timeline UI modules (grep audit).

Impact:
- Design tuning cannot be centralized; risk of drift and repeated visual regressions.

Required fix:
- Introduce timeline FEEL adapter/config and replace magic constants.
- Add tests that layout dims are sourced from FEEL adapter values.

---

### A8) UX progressive disclosure + playback semantics
**Status: ✅ Green**

Evidence:
- Progressive disclosure principle in distillation (`DISTILLATION.md:1038`).
- Mute/solo playback-only principle (`DISTILLATION.md:1041`).
- Current UI keeps M/S on main row and metadata via hover-only disclosure.

Assessment:
- Current lane UI direction aligns with this principle.

---

### A9) Terminology consistency (Branch vs Take)
**Status: 🔴 Red (documentation drift)**

Conflict:
- Distillation glossary still defines **Branch** as user-facing timeline result primitive (`DISTILLATION.md:1055`).
- Current architecture memory states Branch entity killed; takes are the variation primitive (`MEMORY.md#L70-L84`).

Impact:
- Ambiguous guidance for contributors; can reintroduce dead model patterns.

Required fix:
- Update distillation glossary and any lingering branch-facing timeline semantics to take-system terms.

---

### A10) Real-data stems-first progression
**Status: 🟡 Yellow (implemented but partially provisional)**

State:
- Real stems path now runs and surfaces lanes; classifier-in-place currently preview heuristic layer set.

Assessment:
- Correct progression direction is in place, but classifier branch remains preview-mode until true model path is integrated.

---

## Priority Remediation Plan
1. **Kill active-take truth leak** (A4) — highest architecture risk.
2. **FEEL integration pass** (A7) — highest UX contract risk.
3. **Persist SongVersion rebuild plan + tests** (A5).
4. **Distillation glossary cleanup (Branch→Take)** (A9).
5. **Concrete sync-boundary tests for main-only behavior** (A6).

---

## Bottom Line
The codebase is materially aligned on boundary design, output mapping, staleness logic, and recent stems-first UX progress.  
However, three contract-level risks remain and must be fixed for full distillation compliance: **active-take truth leak, FEEL contract drift, and stale Branch terminology in distillation docs**.
