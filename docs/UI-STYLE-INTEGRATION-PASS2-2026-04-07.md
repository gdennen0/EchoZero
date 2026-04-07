# UI Style Integration Pass 2 - 2026-04-07

Purpose: record the pass-2 shared shell style integration beyond the timeline-only surface, list the low-risk migrations completed in this run, and call out the remaining debt that still sits outside the new boundary.

## Scope

- Added a shared shell style package under `echozero/ui/style`.
- Kept the existing FEEL interaction constants and all timeline/Foundry widget structure intact.
- Migrated only low-risk shell literals:
  - timeline object palette shell QSS and layout spacing
  - timeline scroll-area background
  - Foundry window shell styling for root surfaces, group boxes, controls, tabs, lists, and status text
- Added focused tests that assert the shared token/QSS boundary is being used directly.

## Migrated Surfaces

### Shared Shell Boundary

New modules:
- `echozero/ui/style/tokens.py`
- `echozero/ui/style/scales.py`
- `echozero/ui/style/qt/qss.py`

What moved there:
- shell surface colors
- control-state colors
- shared spacing and radius scales
- reusable Qt stylesheet builders for timeline object-palette shell and Foundry shell surfaces

What did not move:
- FEEL interaction and rendering metrics
- timeline painter-local colors
- fixture palette literals

### Timeline

Migrated:
- `ObjectInfoPanel` now consumes `build_object_info_panel_qss()` instead of embedding a large inline stylesheet.
- object-palette spacing and section padding now come from shared shell scales.
- timeline scroll-area background now references the shared shell canvas token.

Intentionally unchanged:
- widget composition
- object names and selector contracts
- timeline row, ruler, transport, header, and take-lane painter logic

### Foundry

Migrated:
- `FoundryWindow` now applies shared shell QSS for the main window, root surfaces, group boxes, tabs, buttons, line edits, text areas, list widgets, and numeric controls.
- low-risk layout spacing now references shared shell scales.
- status line and root container received stable object names so the shared stylesheet has a narrow target surface and tests can validate the contract.

Intentionally unchanged:
- workflow tabs
- control wiring
- data loading and background-run behavior
- window size and general layout structure

## Tests Added

- `tests/ui/test_shared_shell_style.py`
  - verifies shared shell QSS builders resolve token values
  - verifies timeline object palette and Foundry window consume the shared builders
- `tests/foundry/test_ui_smoke.py`
  - adds a smoke assertion for Foundry shared stylesheet application

## Remaining Debt

Still outside the shared shell boundary:
- timeline painter-local colors in `layer_header.py`, `take_row.py`, `ruler.py`, and `transport_bar_block.py`
- event-lane and waveform fallback colors
- real-data fixture and benchmark palette literals
- screenshot-harness metric duplication
- overloaded visual token ownership still present in `echozero/ui/FEEL.py`

Recommended next style pass:
1. Add timeline-local style accessors for painter colors without changing painter math.
2. Align `real_data_fixture.py` with semantic palette helpers so screenshots stop drifting from runtime styling.
3. Derive harness capture sizing from shared metrics or FEEL-backed helpers.
