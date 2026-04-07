# Playhead Event Alignment Audit

## Scope

This audit is limited to the current Stage Zero horizontal alignment path that already exists in code and tests:
- playhead x-position from `echozero/ui/qt/timeline/blocks/ruler.py`
- event-bar x-position from `echozero/ui/qt/timeline/blocks/event_lane.py`
- waveform backdrop positioning from `echozero/ui/qt/timeline/blocks/waveform_lane.py`
- shared presentation plumbing in `echozero/ui/qt/timeline/widget.py`
- focused tests in `tests/ui/test_ruler_block.py`, `tests/ui/test_event_lane_culling.py`, `tests/ui/test_waveform_lane.py`, and `tests/ui/test_follow_scroll.py`

No broader repo scan, screenshot pass, or new instrumentation was added for this retry.

## Method

This pass replaces the scaffold with code-backed observations from the existing render formulas and tests only.

Shared horizontal basis confirmed in current code:

```text
playhead_x = header_width + (playhead_s * pixels_per_second) - scroll_x
event_x    = header_width + (event_start_s * pixels_per_second) - scroll_x
```

Observed verification points:
- `timeline_x_for_time()` and `seek_time_for_x()` are exact inverses in `tests/ui/test_ruler_block.py` for the covered case (`4.5s <-> 690px` at `scroll_x=80`, `pps=100`, `header_width=320`).
- `EventLaneBlock.paint()` uses the same `header_width + time * pps - scroll_x` basis as the ruler/playhead path.
- `compute_follow_scroll_x()` preserves the same presentation scroll basis during playback; follow-mode changes viewport placement, not the underlying time-to-x formula.
- Runtime reconciliation in `widget.py` snaps presentation playhead updates when drift exceeds `0.02s`, while redundant updates are skipped below `0.001s`.

## Current Observations

1. Playhead and event bars are on the same horizontal math path today.
   `ruler.py:timeline_x_for_time()` and `event_lane.py:EventLaneBlock.paint()` both anchor to `header_width` and subtract the same `scroll_x`. There is no code-level offset between those two primitives.

2. The current visible alignment risk is waveform-to-playhead/event mismatch, not event-to-playhead mismatch.
   `waveform_lane.py` uses `content_left = header_width + 8` in both cached and placeholder rendering paths. That creates an effective `+8px` inset relative to ruler ticks, playhead x, and event-bar starts.

3. That waveform inset is large in time terms at current tested zoom levels.
   Equivalent drift for the existing formulas:
- at `100 px/s`: `8px = 80ms`
- at `180 px/s`: `8px = 44.4ms`
- at `200 px/s`: `8px = 40ms`

4. Existing tests validate local geometry helpers, but they do not yet assert cross-panel coincidence for the same timestamp.
   Current coverage proves ruler invertibility, event-lane culling/min width, waveform column continuity, and follow-scroll behavior. It does not prove that playhead, event edge, and waveform transient land on the same screen x in one rendered frame.

5. Event width clamping can visually extend an event beyond its true duration, but it should not move the event start edge.
   `EVENT_MIN_VISIBLE_WIDTH_PX` affects width only. Start-edge alignment remains governed by `event.start`.

## Tolerance Targets (ms)

Use these targets for audit pass/fail until a pixel-diff harness exists:

| Comparison | Target | Notes |
| --- | ---: | --- |
| playhead vs event start edge | <= 5 ms | Equivalent to `<= 0.5px` at `100 px/s`; should be effectively exact with shared math |
| playhead vs event end edge | <= 5 ms | Same expectation; exclude min-width visual expansion cases from end-edge judgment |
| playhead/event vs waveform onset guide | <= 10 ms | Allows small rendering/inset cleanup margin, but current `+8px` path exceeds this at common zoom levels |
| runtime presentation drift before forced reconcile | 20 ms | Matches existing `widget.py` threshold |
| no-op runtime update suppression | 1 ms | Matches existing `widget.py` threshold |

## Next Fixes

1. Remove or centralize the waveform lane's `+8px` horizontal inset so waveform content shares the same origin as ruler ticks, playhead, and event bars.
2. Add one focused UI test that computes screen x for a single timestamp and asserts coincidence across `timeline_x_for_time()`, event-lane rect left edge, and waveform column origin within the targets above.
3. Add one real-data fixture assertion using the existing timeline fixture/build path so the same timestamp is checked after full presentation assembly, not only at helper level.
