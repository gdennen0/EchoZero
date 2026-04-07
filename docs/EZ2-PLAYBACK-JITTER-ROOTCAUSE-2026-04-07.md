# EZ2 Playback Jitter / Alignment Drift Root Cause Audit

Date: 2026-04-07

## Scope

Audit target: the current real-data playback path invoked by `run_timeline_real_data_playback.py`, `build_real_data_demo_app()`, `TimelineRuntimeAudioController`, and `TimelineWidget`.

Real source used for reproduction:

- `C:\Users\griff\.openclaw\workspace\tmp\doechii-audio\Doechii_NissanAltima_117bpm_SPMTE_v02_chan1.wav`

Observed real-data build summary from the active path:

- Layers: 8
- Takes: 4
- Main-lane events: 692
- Zoom: 180 px/s
- Audio engine buffer: 256 frames at 44.1 kHz

## A. Reproduction Protocol

1. Run the real-data entrypoint with a real WAV:

```powershell
C:\Users\griff\EchoZero\.venv\Scripts\python.exe run_timeline_real_data_playback.py --audio "C:\Users\griff\.openclaw\workspace\tmp\doechii-audio\Doechii_NissanAltima_117bpm_SPMTE_v02_chan1.wav" --working-root artifacts\timeline-real-data-runtime-audit
```

2. Confirm the app prints a `REALTIME_REAL_DATA_SUMMARY` block and loads the real-data timeline.
3. Press play and watch the playhead at the default `180 px/s` zoom.
4. Compare three things in the same viewport:
   - Playhead motion frame to frame
   - Ruler tick / event bar alignment
   - Waveform onset alignment against the ruler/playhead
5. For quantified measurements, drive the same `TimelineRuntimeAudioController` and `TimelineWidget` path with a fake output stream and the real WAV, then sample `_on_runtime_tick()` at the widget timer cadence.

## B. Measured Jitter / Drift Observations

### Pre-patch measurements on the current path

Using the real-data presentation, real waveform registration, current runtime controller, and current widget path:

- Audio clock step size: `256 / 44100 = 5.805 ms`
- Widget runtime timer: `16 ms`
- Measured playhead delta per UI update: `11.610 ms` or `17.415 ms`
- Mean delta: `16.006 ms`
- Std dev: `2.489 ms`
- At `180 px/s`, those jumps are `2.09 px` or `3.13 px`

Interpretation:

- The playhead is not drifting away from the audio clock.
- It is visibly stair-stepping because the UI samples a callback-driven clock on an unsynchronised 16 ms timer, while the clock itself advances in 5.805 ms chunks.

### Pre-patch alignment drift

- Waveform lane origin was `header_width + 8`
- Ruler, playhead, and event bars all used `header_width`
- At `180 px/s`, `8 px` equals `44.444 ms`

Interpretation:

- This is not subtle jitter. It is a deterministic left/right alignment error.
- Waveform content was consistently offset from the ruler, playhead, and event bars by about `44.4 ms` at the default real-data zoom.

### Post-patch spot-check

After the low-risk changes in this branch:

- Widget runtime timer: `8 ms`, `PreciseTimer`
- Measured playhead delta per UI update: `5.805 ms` or `11.610 ms`
- Mean delta: `7.991 ms`
- Ruler minus waveform x-offset: `0.0 px`

Interpretation:

- The underlying audio clock quantisation still exists.
- The UI now samples it twice as often, so visible playhead jump amplitude is reduced.
- The deterministic waveform alignment error is removed.

## C. Formula / Path Audit

### Playhead time source

- `TimelineRuntimeAudioController.current_time_seconds()` returns `engine.clock.position_seconds` in [runtime_audio.py](../echozero/ui/qt/timeline/runtime_audio.py).
- `Clock.position_seconds` is `_position / _sample_rate` in [clock.py](../echozero/audio/clock.py).
- `Clock.advance(frames)` increments `_position` once per audio callback in [clock.py](../echozero/audio/clock.py).

Formula:

```text
playhead_seconds = clock_position_samples / sample_rate
```

Granularity:

```text
clock_step_seconds = buffer_frames / sample_rate = 256 / 44100 = 0.005804988...
```

### Playhead x-position

`TimelineCanvas._draw_playhead()` calls `timeline_x_for_time()` in [widget.py](../echozero/ui/qt/timeline/widget.py) and [ruler.py](../echozero/ui/qt/timeline/blocks/ruler.py).

Formula:

```text
playhead_x = header_width + (playhead_seconds * pixels_per_second) - scroll_x
```

### Event bar x-position

`EventLaneBlock.paint()` in [event_lane.py](../echozero/ui/qt/timeline/blocks/event_lane.py) uses:

```text
event_x = header_width + (event.start * pixels_per_second) - scroll_x
event_width = max(EVENT_MIN_VISIBLE_WIDTH_PX, event.duration * pixels_per_second)
```

Result:

- Event bars are aligned with the ruler/playhead formula.

### Waveform x-position

`WaveformLaneBlock._paint_cached_waveform()` in [waveform_lane.py](../echozero/ui/qt/timeline/blocks/waveform_lane.py) now uses:

```text
waveform_x = header_width + (time_seconds * pixels_per_second) - scroll_x
```

Before this patch the code used `header_width + 8`, which introduced a constant horizontal bias.

### Ruler x-position

`RulerBlock.paint()` and `visible_ruler_seconds()` in [ruler.py](../echozero/ui/qt/timeline/blocks/ruler.py) use:

```text
ruler_x = header_width + (seconds * pixels_per_second) - scroll_x
```

### Follow-scroll path

`compute_follow_scroll_x()` in [widget.py](../echozero/ui/qt/timeline/widget.py) computes the scroll target from:

```text
timeline_x = playhead_seconds * pixels_per_second
```

Then `_update_horizontal_scroll_bounds()` rounds `scroll_x` through the integer `QScrollBar` value.

Implication:

- Follow scroll is tied to the same time base as the playhead.
- The scroll path can add up to about `0.5 px` of rounding error when a float target is collapsed into the integer scrollbar domain.

## D. Root-Cause Ranking

1. Waveform lane used a different horizontal origin than the ruler/playhead/event bars.
   Confidence: High
   Evidence: source audit showed `header_width + 8` only in the waveform lane; measured error was a deterministic `8 px = 44.444 ms`.

2. UI playhead polling cadence was too coarse relative to the callback-driven audio clock.
   Confidence: High
   Evidence: real-path measurements produced only `11.610 ms` and `17.415 ms` UI deltas from a `16 ms` timer sampling a `5.805 ms` clock step. That is the visible jitter signature.

3. `QScrollBar` integer quantisation can add a small follow-scroll snap on top of the sampled playhead motion.
   Confidence: Medium
   Evidence: `_update_horizontal_scroll_bounds()` rounds `scroll_x` to an int. This is bounded and much smaller than the waveform bias or timer/callback mismatch.

4. Painter integer casts add minor subpixel loss but are not the primary failure mode.
   Confidence: Low
   Evidence: the large observed issues were already explained by the timer/callback cadence and the former `+8 px` waveform bias.

## E. Immediate Patch Plan

Implemented in this branch:

1. Unify waveform x-mapping with ruler/playhead/event bars.
   - Remove the hardcoded `+8 px` waveform origin bias.
   - Add a shared `waveform_x_for_time()` mapping and test it against `timeline_x_for_time()`.

2. Reduce visible playhead jump size.
   - Switch the runtime UI timer to `Qt.PreciseTimer`.
   - Reduce the polling interval from `16 ms` to `8 ms`.
   - This does not change the audio clock source of truth. It reduces visible stair-step amplitude.

Next small diffs worth considering if more smoothing is still needed:

1. Decouple visual playhead interpolation from raw callback steps while keeping seeks and transport state locked to the audio clock.
2. Carry float follow-scroll state separately from the integer scrollbar value if follow-mode snap remains noticeable during long auto-scroll runs.
