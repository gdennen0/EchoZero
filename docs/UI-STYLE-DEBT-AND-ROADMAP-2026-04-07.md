# UI Style Debt And Roadmap - 2026-04-07

Purpose: audit current EchoZero Qt UI style debt, classify the styling surface by ownership lane, and define the pass-2 migration plan beyond the timeline shell without refactoring production code in this run.

## Scope

- Audited paths:
  - `echozero/ui`
  - `echozero/ui/qt`
- Sampled for pass-2 planning context:
  - `echozero/foundry/ui/main_window.py`
- Focus:
  - style literals
  - duplicated colors and spacing
  - inline QSS
  - painter-local visual tokens
  - screenshot/test harness readiness

## Executive Read

- The active style debt is overwhelmingly concentrated in the Stage Zero timeline shell.
- `echozero/ui/FEEL.py` is the nearest thing to a shared style source, but it currently mixes interaction tuning, layout metrics, graph-editor colors, and timeline paint tokens in one flat module.
- `echozero/ui/qt/timeline/widget.py` is the largest hotspot because it owns both structural widget composition and a full inline QSS block for the object palette shell.
- The timeline block renderers each carry painter-local colors and chip/button states, which means visual consistency currently depends on file-by-file discipline rather than a token system.
- Fixture builders and screenshot harness code repeat style-adjacent values, which creates quiet drift risk even when runtime paint code is untouched.
- Foundry is not yet a major style-literal hotspot, which makes it the right pass-2 adopter once the shared shell layer exists.

## Inventory Of Style Literal Hotspots

Hotspots were counted using style-literal matches across `*.py` and `*.json` under the audited paths, excluding `__pycache__`.

| Rank | File | Hotspot count | Main debt shape | Classification |
|---|---|---:|---|---|
| 1 | `echozero/ui/qt/timeline/widget.py` | 64 | Inline QSS, QColor literals, repeated margins/padding, shell + theme coupling | timeline |
| 2 | `echozero/ui/qt/timeline/fixtures/realistic_timeline_fixture.json` | 41 | Hard-coded event colors in fixture data | legacy |
| 3 | `echozero/ui/FEEL.py` | 29 | Flat token dump spanning multiple UI domains | shared shell |
| 4 | `echozero/ui/qt/timeline/blocks/layer_header.py` | 14 | Header fills, text colors, chip colors, button states | timeline |
| 5 | `echozero/ui/qt/timeline/blocks/take_row.py` | 11 | Row backgrounds, option-button colors, action-chip colors | timeline |
| 6 | `echozero/ui/qt/timeline/blocks/ruler.py` | 9 | Ruler surface colors and playhead literal duplication | timeline |
| 7 | `echozero/ui/qt/timeline/blocks/transport_bar_block.py` | 8 | Transport surface/button colors and typography | timeline |
| 8 | `echozero/ui/qt/timeline/real_data_fixture.py` | 7 | Presentation-time lane/event colors duplicated outside shared tokens | legacy |
| 9 | `echozero/ui/qt/timeline/blocks/event_lane.py` | 3 | Event fallback fill/text colors | timeline |
| 10 | `echozero/ui/qt/timeline/blocks/waveform_lane.py` | 3 | Waveform inner padding and fallback rendering constants | timeline |
| 11 | `echozero/ui/qt/timeline/test_harness.py` | 2 | Screenshot sizing constants duplicated from runtime metrics | legacy |

### File Notes

`echozero/ui/FEEL.py`
- Owns live tuning constants for timeline dimensions and some colors.
- Also contains node-editor and classification palette values not yet expressed as separate token groups.
- Current structure is useful as a stopgap, but it is not a scalable app-wide style API.

`echozero/ui/qt/timeline/widget.py`
- Largest single hotspot.
- Mixes layout metrics, composition, object-palette QSS, scroll-area styling, canvas background, row fills, fallback waveform color defaults, and playhead colors.
- The file is currently both shell orchestrator and style host.

`echozero/ui/qt/timeline/blocks/layer_header.py`
- Carries its own surface, text, chip, and toggle button palette.
- Chip semantics like `STALE` and `EDITED` are visually important, but their colors live only in the painter.

`echozero/ui/qt/timeline/blocks/take_row.py`
- Repeats dark-surface variants that are visually adjacent to `layer_header.py` but implemented independently.
- Option-chip colors are hard-coded rather than derived from a shared action/take token set.

`echozero/ui/qt/timeline/blocks/ruler.py`
- Duplicates playhead red already represented in `FEEL.py`.
- Ruler surface and label colors are painter-local instead of timeline token references.

`echozero/ui/qt/timeline/blocks/transport_bar_block.py`
- Owns transport shell styling independently from `widget.py`, so top-level timeline chrome is split across files.

`echozero/ui/qt/timeline/real_data_fixture.py`
- Presentation construction assigns colors directly for song, stems, and classifier lanes.
- Classifier color ordering duplicates the concept already present in `FEEL.py::CLASSIFICATION_COLORS`.

`echozero/ui/qt/timeline/fixtures/realistic_timeline_fixture.json`
- Large volume of literal colors is acceptable for snapshot realism, but today it functions as an uncontrolled palette source.
- This is a drift risk when screenshots are treated as UI truth.

`echozero/ui/qt/timeline/test_harness.py`
- Screenshot height constants re-state runtime row/ruler/transport sizes instead of deriving from FEEL/runtime helpers.

## Classification

### Timeline

Timeline-owned styling should cover visual decisions that only exist because the timeline exists.

Current timeline files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/event_lane.py`
- `echozero/ui/qt/timeline/blocks/layer_header.py`
- `echozero/ui/qt/timeline/blocks/ruler.py`
- `echozero/ui/qt/timeline/blocks/take_row.py`
- `echozero/ui/qt/timeline/blocks/transport_bar_block.py`
- `echozero/ui/qt/timeline/blocks/waveform_lane.py`

Examples:
- playhead, ruler, lane-row fills, take-row affordances, waveform rendering tint, stale/edited badges.

Assessment:
- High debt concentration.
- Good candidate for first extraction because the files are already block-oriented.

### Foundry

Foundry-owned styling should cover desktop workflows unique to training, datasets, runs, evaluation, and artifacts.

Current files:
- `echozero/foundry/ui/main_window.py`

Assessment:
- Low current hotspot count in the sampled scan.
- More importantly, this means Foundry has not yet become deeply entangled in a parallel token system.
- It should consume the shared shell style layer after timeline extraction proves the boundary.

### Shared Shell

Shared shell styling should cover app-wide primitives that both timeline and Foundry will need.

Current closest source:
- `echozero/ui/FEEL.py`

Should eventually own:
- semantic color tokens
- spacing and radius scales
- typography roles
- shell surfaces
- control states
- status semantics
- helper APIs for QSS and `QColor` conversion

Assessment:
- Needed immediately as a boundary, but should be introduced incrementally instead of by giant rewrite.

### Legacy

Legacy here means styling data or helpers that are useful for demos, fixtures, or verification, but should not define production style truth.

Current files:
- `echozero/ui/qt/timeline/fixtures/realistic_timeline_fixture.json`
- `echozero/ui/qt/timeline/real_data_fixture.py`
- `echozero/ui/qt/timeline/test_harness.py`
- `echozero/ui/qt/timeline/demo_app.py`
- `echozero/ui/qt/timeline/demo_walkthrough.py`
- `echozero/ui/qt/timeline/drum_classifier_preview.py`

Assessment:
- Not the highest implementation risk, but these are where style drift becomes normalized and then re-enters production via screenshots and demos.

## Recommended Module Boundaries For An App-Wide Style System

Target shape:

1. `echozero/ui/style/tokens.py`
   - semantic token dataclasses or named maps
   - no Qt widget logic
   - examples: `surface.panel`, `surface.canvas`, `text.primary`, `accent.playhead`, `status.stale_bg`

2. `echozero/ui/style/scales.py`
   - spacing, radii, border widths, typography sizes, shared heights
   - migrate visual dimensions out of flat `FEEL.py` namespace gradually

3. `echozero/ui/style/qt/colors.py`
   - token to `QColor`
   - alpha helpers
   - lighten/darken wrappers to keep painter code declarative

4. `echozero/ui/style/qt/qss.py`
   - shared QSS fragment builder for shell panels, buttons, sliders, labels
   - object palette and future Foundry panels should consume this instead of embedding large QSS strings inline

5. `echozero/ui/style/semantic.py`
   - domain semantics independent of screen
   - status chips, classifier palette, lane kinds, severity/state colors

6. `echozero/ui/qt/timeline/style.py`
   - timeline-only mapping from shared tokens to timeline block usage
   - keeps playhead/ruler/lane specifics out of generic shell modules

7. `echozero/foundry/ui/style.py`
   - Foundry-only composition layer consuming shared shell tokens
   - should stay thin

### Boundary Rules

- `FEEL.py` should remain the source for interaction tuning during migration, but not remain the permanent home for all app-wide visual tokens.
- Shared shell modules should not import timeline widgets or presentation models.
- Timeline and Foundry style layers may translate semantic tokens into local rendering choices, but should not redefine base shell colors independently.
- Fixture builders should reference semantic color names or helper functions where possible, not literal hex values.

## Migration Order

No large refactor in this run. This is the recommended execution order for follow-on work.

| Order | Pass | Scope | Risk | Why this order |
|---|---|---|---|---|
| 1 | Shared shell seed | Introduce token/scales modules and Qt helpers beside existing `FEEL.py` | medium | Enables migration without breaking current FEEL consumers |
| 2 | Timeline shell chrome | Extract `ObjectInfoPanel` QSS and scroll/shell surface styling from `widget.py` | low | Highest hotspot, but largely isolated from painter math |
| 3 | Timeline semantic colors | Move playhead, ruler, row, badge, take-row, transport colors behind timeline style accessors | medium | Many files touched, but behavior stays visual-only |
| 4 | Fixture alignment | Replace duplicated classifier/stem colors in `real_data_fixture.py` and fixture JSON generation path with semantic palette mapping | medium | Reduces screenshot drift and keeps demos honest |
| 5 | Metric alignment | Derive screenshot harness sizes from FEEL/style scales instead of duplicated constants | low | Small change with immediate anti-regression value |
| 6 | FEEL split | Leave interaction tuning in `FEEL.py`; move visual tokens and scales into `echozero/ui/style/*` | high | Broadest import churn; do after timeline consumers exist |
| 7 | Foundry adoption | Add shared shell theme consumption in `echozero/foundry/ui/main_window.py` and any new Foundry widgets | low | Foundry currently has low style debt, so adoption should be cheap after the shared shell stabilizes |
| 8 | Legacy cleanup | Remove or quarantine obsolete demo-only literals once runtime and fixtures consume the new style API | medium | Prevents old demos from silently reintroducing literals |

## Risk Levels By Area

### Low Risk

- Object palette QSS extraction from `widget.py`
- Scroll area and panel shell styling centralization
- Screenshot harness derivation from shared metrics
- Foundry adoption after shared shell exists

### Medium Risk

- Timeline painter color tokenization across multiple block files
- Fixture and real-data presentation color normalization
- Shared semantic token introduction while FEEL remains active

### High Risk

- Splitting `FEEL.py` too early
- Mixing interaction constants and style tokens into one new abstraction without clear ownership
- Rewriting all timeline painter code in a single pass

## Anti-Regression Strategy

### Tests

- Add focused unit coverage for token helpers:
  - semantic token resolution
  - `QColor` conversion
  - alpha/lighten/darken wrappers
- Add timeline block paint smoke tests where practical:
  - selected vs unselected layer header states
  - stale/edited chip rendering contracts
  - object palette active/disabled button state selectors
- Add a test that `test_harness.py` uses shared metrics rather than duplicated literal sizes once migration begins.

### Screenshots

- Keep the existing screenshot harness and promote it into the primary visual gate for style changes.
- Capture at least these variants on every style-system pass:
  - default timeline
  - scrolled timeline
  - take lanes open
  - selected event with object palette active
  - zoomed in
  - zoomed out
  - real-data presentation
- Store approved screenshots under a stable artifact path and compare them in CI or a scripted local diff step.

### Process Guardrails

- Treat fixture JSON colors as presentation fixtures, not palette authority.
- Ban new inline `setStyleSheet("""...""")` blocks outside dedicated style modules.
- Ban new raw hex literals in timeline painter files unless they are first added to the local style accessor layer.
- Require any new screen to declare whether its styling is `shared shell`, `timeline`, `foundry`, or `legacy`.

## Pass-2 Recommendation Beyond Timeline

After pass 1 establishes shared shell tokens and timeline style accessors, pass 2 should target Foundry shell adoption, not deeper timeline ornamentation.

Reason:
- Foundry currently has low entanglement, so it is the cheapest place to prove the app-wide style system is truly app-wide.
- If pass 2 stays inside timeline, the result will likely be a better timeline theme but not a reusable EchoZero shell.
- A Foundry adoption pass will pressure-test whether the shared modules are semantic enough to serve multiple product lanes.

Concrete pass-2 targets:
- shell window surfaces
- panel/card styling
- button and status treatment
- typography roles
- shared spacing/radius scales

Do not make pass 2:
- a FEEL mega-split
- a cross-app restyle
- a screenshot refresh without tokenization underneath

## Top Findings

1. `echozero/ui/qt/timeline/widget.py` is the dominant style-debt hotspot and currently acts as both shell composer and theme container.
2. `echozero/ui/FEEL.py` is useful but overloaded; it is not yet an app-wide style system, only a flat constants module with mixed responsibilities.
3. Timeline painter blocks duplicate local color logic instead of consuming semantic tokens, especially in `layer_header.py`, `take_row.py`, `ruler.py`, and `transport_bar_block.py`.
4. Playhead color is duplicated between FEEL and runtime paint code, which is a direct drift vector.
5. Real-data and fixture builders carry literal lane/event colors outside the shared style surface, so screenshots can drift from runtime intent.
6. The screenshot harness repeats runtime dimensions, which weakens visual tests as a reliable guardrail.
7. Foundry is currently under-styled rather than over-entangled, making it the correct pass-2 consumer once a shared shell layer exists.
8. The safest migration path is incremental: shared shell seed, timeline shell extraction, timeline semantic colors, fixture alignment, then Foundry adoption.

## Recommended Next Action

Implement pass 1 as a narrow extraction:
- introduce shared shell token modules
- extract object-palette/shell QSS out of `widget.py`
- add timeline-local style accessors
- wire screenshot verification before broader token migration
