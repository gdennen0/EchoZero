# UI Contract Audit — 2026-04-03

## Scope
Audit Stage Zero timeline shell against:
1. `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
2. `docs/UX-MICRO-TESTS.md`
3. `echozero/ui/FEEL.py` contract ("craft lives in constants, no magic numbers")

## Result Summary
- **Green:** 5
- **Yellow:** 2
- **Red:** 2

---

## Checklist

### C1. Main is truth, takes are subordinate (First Principles §1)
**Status:** 🟡 Partial

**What passes:**
- UI now shows parent lane + child take rows.
- Take actions are explicit (overwrite/merge), matching intended model.

**What fails / risk:**
- App-layer timeline orchestrator still has `SelectTake` setting `layer.active_take_id = take_id`.
- Assembler still builds lane events from `_get_active_take(layer)`.

**Evidence:**
- `echozero/application/timeline/orchestrator.py:123`
- `echozero/application/timeline/assembler.py:53,138`

**Why this matters:**
- This reintroduces active-take semantics below the UI layer.

---

### C2. Mute/Solo only on main row (First Principles + UX)
**Status:** ✅ Green

**Evidence:**
- M/S controls rendered in `layer_header.py` (main row header path).
- Take row renderer has options/actions only, no M/S controls.

**Evidence refs:**
- `echozero/ui/qt/timeline/blocks/layer_header.py:35-36`
- `echozero/ui/qt/timeline/blocks/take_row.py`

---

### C3. Ruler remains pinned during vertical scroll (UX TR-2.2.4)
**Status:** ✅ Green

**Evidence:**
- Ruler is a dedicated widget above scroll area, not inside canvas rows.

**Evidence ref:**
- `echozero/ui/qt/timeline/widget.py:435,473`

---

### C4. Follow mode should not cause viewport jitter when paused
**Status:** ✅ Green

**Evidence:**
- Follow scroll logic is gated by playback state.

**Evidence ref:**
- `echozero/ui/qt/timeline/widget.py:72`

---

### C5. Metadata readability / progressive disclosure
**Status:** ✅ Green

**Current behavior:**
- Default row is minimal.
- Metadata is hover-only tooltip (floating, no layout reflow).

**Evidence:**
- `QToolTip.showText(...)` path in timeline canvas hover handling.

**Evidence refs:**
- `echozero/ui/qt/timeline/widget.py:214`

---

### C6. FEEL.py as single source of UI tuning constants
**Status:** 🔴 Red

**What fails:**
- Timeline UI currently has many hardcoded numeric/layout constants and no FEEL imports.

**Evidence:**
- `NO_FEEL_IMPORTS_OR_REFERENCES_IN_TIMELINE_UI` from grep audit.
- Header/row constants hardcoded in widget (`_header_width = 320`, `_main_row_height = 72`, `_take_row_height = 44`).

**Evidence refs:**
- `echozero/ui/qt/timeline/widget.py:141,143,144`
- `echozero/ui/FEEL.py:82-83` (contract values exist but not wired)

---

### C7. FEEL parity for canonical dimensions
**Status:** 🔴 Red

**What fails:**
- FEEL values diverge from active timeline layout values:
  - `LAYER_HEADER_WIDTH_PX=160` vs active 320
  - `LAYER_ROW_HEIGHT_PX=60` vs active 72

**Evidence refs:**
- `echozero/ui/FEEL.py:82-83`
- `echozero/ui/qt/timeline/widget.py:141,143`

---

### C8. Zoom/ruler tick behavior expected by micro tests
**Status:** 🟡 Partial

**What passes:**
- Ruler tick labels now track horizontal scroll correctly.

**What is incomplete:**
- Full FEEL-driven tick spacing policy and zoom anchor details from `UX-MICRO-TESTS` are not fully wired as explicit contract tests yet.

---

### C9. Real-data flow (stems-first) integrated in UI
**Status:** ✅ Green

**Evidence:**
- Real-data capture now runs stem separation first and renders stems + class preview lanes.

---

## Recommended Remediation Order (next)
1. **Kill active-take truth leak in app layer**
   - `SelectTake` should select/highlight only, not mutate `active_take_id`.
   - Assembler should render main truth lane by contract.
2. **FEEL integration pass**
   - Replace hardcoded layout constants with FEEL constants.
   - Add one timeline config shim so FEEL drives row/header/ruler dimensions.
3. **Contract tests pass**
   - Add explicit tests for: main-truth rendering, FEEL dimension wiring, no-take-activation-on-select.

---

## Bottom line
Current UI **looks** close to contract, but architecture-level contract is not fully satisfied yet due to active-take behavior and missing FEEL integration.
