# UI Style Integration Pass 3

Date: 2026-04-07

## Final Boundary Contract

### FEEL (`echozero/ui/FEEL.py`)

- Owns geometry, sizing, hit-target, timing, and interaction-tuning constants.
- May be read by timeline layout/render code when a value changes interaction or spatial behavior.
- Must not own shell theming, semantic layer colors, action labels, stylesheet text, or other presentational tokens.

### Timeline Style (`echozero/ui/qt/timeline/style.py`)

- Owns timeline shell presentation tokens and token helpers.
- Covers block paint tokens, shell stylesheets, fixture/demo visual defaults, semantic timeline layer colors, and default take-action labels.
- Is the single discoverable style entry point for timeline shell surfaces via `TIMELINE_STYLE`, `fixture_color(...)`, and `fixture_take_action_label(...)`.

### Shared Shell Style Stack

- For the active Stage Zero timeline surface, the shared shell stack is:
  - `TIMELINE_STYLE`
  - `build_object_palette_stylesheet(...)`
  - `build_timeline_scroll_area_stylesheet(...)`
  - block defaults that bind to `TIMELINE_STYLE.*`
- Timeline widget/block code consumes this stack but does not redefine shell colors, fallback audio lane colors, fixture action labels, or fixture semantic colors locally.

## Migrated Files And Rationale

- `echozero/ui/qt/timeline/style.py`
  - Added fixture/demo-presentational tokens for semantic layer colors, default sync label, fallback audio lane color, and take-action labels.
- `echozero/ui/qt/timeline/fixture_loader.py`
  - Added token resolution so JSON fixtures can carry semantic style references instead of duplicating literal visual values.
- `echozero/ui/qt/timeline/fixtures/realistic_timeline_fixture.json`
  - Replaced duplicated layer color literals with `color_token` values and removed duplicated event colors and take-action labels.
- `echozero/ui/qt/timeline/real_data_fixture.py`
  - Moved real-data fallback colors, default sync label, and take-action labels onto timeline style tokens.
- `echozero/ui/qt/timeline/blocks/event_lane.py`
  - Added lane-level default fill plumbing so event visuals can inherit layer color without repeating event-local color literals.
- `echozero/ui/qt/timeline/widget.py`
  - Removed remaining local waveform fallback color seam and passed layer color into event-lane presentation.
- `echozero/ui/qt/timeline/blocks/take_row.py`
  - Removed duplicated default take-action labels in favor of style-token lookup.
- `tests/ui/test_timeline_style.py`
  - Added discoverability coverage for fixture token helpers.
- `tests/ui/test_timeline_fixture_loader.py`
  - Added coverage for token-backed layer/event color resolution and take-action label backfill.
- `tests/ui/test_real_data_fixture.py`
  - Added coverage for real-data fallback token plumbing.
- `tests/ui/test_event_lane_culling.py`
  - Added coverage for inherited event-lane fill behavior when events omit explicit colors.

## Remaining Debt

- Medium: `echozero/ui/qt/timeline/blocks/waveform_lane.py` still owns its own paint treatment instead of using a typed style token group. Runtime behavior is stable, but style ownership is not fully centralized there yet.
- Low: fixture/demo content copy such as subtitles, badge text, and object-info prose remains local content, not style. This is intentional for now and does not affect visual truth ownership.
- Low: legacy prototype modules under `ui/timeline/` still contain older visual constants, but they are not the active Stage Zero timeline shell used by the targeted test surface.

## Validation

Command:

```powershell
py -3.12 -m pytest tests\ui\test_timeline_style.py tests\ui\test_timeline_shell.py tests\ui\test_timeline_feel_contract.py tests\ui\test_timeline_fixture_loader.py tests\ui\test_real_data_fixture.py tests\ui\test_event_lane_culling.py
```

Result:

```text
56 passed in 1.72s
```
