# Playback And Clock Clean-Sheet Design

Status: historical
Last reviewed: 2026-04-30


Originally updated: 2026-04-18

This design note is retained as historical context.
For current playback behavior, use `docs/STATUS.md`.

## Goal

Ship a reliable near-term playback and clock system for EchoZero that:

- separates selection from playback target
- plays correctly on real hardware
- uses backend-owned timing rather than UI polling as transport truth
- supports human-path demo and non-deterministic playback proof
- leaves a clean boundary for a future bespoke DAW-grade engine

## First Principles

1. Selection is editorial state, not playback state.
2. Playback target must be explicit and preserved across app reloads when still valid.
3. Backend timing is truth. UI rendering is a view over that truth.
4. Real playback must negotiate to the output device instead of assuming a fixed format.
5. Human-path demos must use real UI and runtime actions, not state injection.

## What We Copy From Proven Systems

### Ardour

- Transport changes should be requested from the UI and completed by backend-owned threads.
- Realtime work and non-realtime work must be separated clearly.
- Cross-thread state delivery to the UI should be explicit and bounded.

References:

- https://ardour.org/transport_threading.html
- https://ardour.org/cross-thread.html

### Audacity

- Selection is its own editing concept and should not be conflated with playback routing.

Reference:

- https://manual.audacityteam.org/man/audacity_selection.html

### LMMS

- Threading boundaries must be explicit. UI and engine responsibilities should not bleed into each other.

Reference:

- https://docs.lmms.io/developer-guides/core/threading-and-synchronization

### PortAudio / sounddevice

- Playback timing should use backend timing data, including output-aligned latency information.
- Device and buffering choices should respect host/backend constraints instead of assuming a fixed universal format.

References:

- https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines
- https://portaudio.com/docs/v19-doxydocs/api_overview.html
- https://python-sounddevice.readthedocs.io/en/0.3.14/api.html

### Qt

- `QTimer` is not transport truth.
- High-level or device-level Qt audio primitives remain a viable fallback if the current backend proves too brittle.

References:

- https://doc.qt.io/qt-6/qtimer.html
- https://doc.qt.io/qt-6/qaudiosink.html
- https://doc.qt.io/qt-6/qmediaplayer.html

## Near-Term Architecture

### State Model

- `selected_layer_id` and `selected_take_id` remain editorial state.
- `active_playback_layer_id` and `active_playback_take_id` remain playback state.
- No runtime path may infer playback target from selection.

### Playback Backend

- Near-term backend remains the current `AudioEngine`, but only behind an explicit playback boundary.
- Real playback should use device-native output defaults unless a caller explicitly overrides them.
- On this machine, the immediate practical consequence is `48000 Hz` and `2` output channels rather than `44100 Hz` mono.

### Playback Service Boundary

The next service boundary should own:

- play / pause / stop / seek
- active playback target
- output device format negotiation
- reported timing snapshots
- future mute / solo / audition policy

Widget/runtime helpers should stop owning playback semantics directly.

### Clock / Playhead

- Backend publishes transport snapshots:
  - play state
  - sample position
  - sample rate
  - output latency or output-aligned timing
- UI renders a smooth playhead by interpolating between snapshots.
- UI timers remain repaint triggers only.

## Immediate Implementation Order

1. keep the explicit playback-target contract
2. stabilize real playback device negotiation
3. audit and fix any remaining static/noise path in the current backend
4. add a playback service boundary around backend semantics
5. move playhead rendering to backend-owned timing snapshots plus interpolation
6. rebuild playback demos on human paths only

## Explicit Avoids

- no fallback from selection to playback target
- no `QTimer`-driven transport truth
- no human-path demo claims on simulated callback-driven recorder helpers
- no further broadening of widget-owned playback logic
