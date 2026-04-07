# Playhead Event Alignment Audit

## Scope

This audit scaffold covers the Stage Zero timeline shell alignment path between:
- playhead rendering in the ruler/canvas
- event-bar rendering in main layer lanes and take lanes
- presentation-state inputs that drive both paths

This document is limited to measurement definitions, formulas, and data-path tracing. It does not include a full audit matrix, screenshots, or pass/fail judgments.

## Measurement Method + Formulas

Use a single `TimelinePresentation` snapshot and measure both playhead and event geometry from the same values for:
- `playhead`
- `pixels_per_second`
- `scroll_x`
- `header_width`
- `event.start`
- `event.end`

Render-space formulas used by the current Stage Zero shell:

```text
playhead_x = header_width + (playhead_s * pixels_per_second) - scroll_x
event_start_x = header_width + (event_start_s * pixels_per_second) - scroll_x
event_end_x = header_width + (event_end_s * pixels_per_second) - scroll_x
event_width_px = max(EVENT_MIN_VISIBLE_WIDTH_PX, (event_end_s - event_start_s) * pixels_per_second)
```

Primary alignment deltas:

```text
delta_start_px = event_start_x - playhead_x
delta_end_px = event_end_x - playhead_x
delta_start_s = delta_start_px / pixels_per_second
delta_end_s = delta_end_px / pixels_per_second
```

Useful simplification when both primitives share the same `header_width`, `scroll_x`, and `pixels_per_second`:

```text
delta_start_px = (event_start_s - playhead_s) * pixels_per_second
delta_end_px = (event_end_s - playhead_s) * pixels_per_second
```

Measurement procedure:
1. Load one fixture or real-data presentation without mutating timing fields during capture.
2. Record `playhead`, `scroll_x`, `pixels_per_second`, and active lane event times from the same presentation instance.
3. Compute expected screen-space `x` values using the formulas above.
4. Compare those computed `x` values to rendered playhead and event-bar edges in the same frame.
5. Report drift in both pixels and seconds, using `delta_t = delta_px / pixels_per_second`.

## Data-Path Tracing

### Presentation Sources

- Fixture JSON source: `echozero/ui/qt/timeline/fixtures/realistic_timeline_fixture.json`
- Fixture loader to presentation model: `echozero/ui/qt/timeline/fixture_loader.py`
- Real-data builder to presentation model: `echozero/ui/qt/timeline/real_data_fixture.py`

### Canonical UI State

- Shared presentation dataclasses: `echozero/application/presentation/models.py`
- Key fields for alignment: `TimelinePresentation.playhead`, `TimelinePresentation.pixels_per_second`, `TimelinePresentation.scroll_x`, `LayerPresentation.events`, `TakeLanePresentation.events`, `EventPresentation.start`, `EventPresentation.end`

### Intent and Update Flow

- Timeline intents, including seek/select operations: `echozero/application/timeline/intents.py`
- Demo dispatch that mutates transport/playhead state into presentation state: `echozero/ui/qt/timeline/demo_app.py`
- Widget dispatch, follow-scroll adjustment, runtime tick reconciliation, and child-widget propagation: `echozero/ui/qt/timeline/widget.py`

### Playhead Render Path

- `TimelineWidget` owns ruler/canvas presentation updates: `echozero/ui/qt/timeline/widget.py`
- Ruler playhead geometry helpers: `echozero/ui/qt/timeline/blocks/ruler.py`
- Canvas playhead draw and hit-test path: `echozero/ui/qt/timeline/widget.py`

### Event Render Path

- `TimelineCanvas` builds per-row lane presentations and forwards them to paint blocks: `echozero/ui/qt/timeline/widget.py`
- Event-lane screen-space geometry and label painting: `echozero/ui/qt/timeline/blocks/event_lane.py`
- Waveform/take-row adjacency that shares the same horizontal scroll basis: `echozero/ui/qt/timeline/blocks/waveform_lane.py`
- Take-row composition feeding alternate event lanes: `echozero/ui/qt/timeline/blocks/take_row.py`
