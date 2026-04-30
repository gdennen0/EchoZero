# Playback Backend Bakeoff

Status: historical
Last reviewed: 2026-04-30


Originally updated: 2026-04-18

This memo captures a point-in-time backend decision during playback remediation.
For current playback behavior, use `docs/STATUS.md`.

This is the PB-20 decision memo for near-term playback stabilization.

## Decision

Keep the current `AudioEngine` / `sounddevice` path for the near term, but only behind an explicit playback runtime boundary.

Do not switch to a temporary `QAudioSink` backend for the current remediation slice.

## Recommendation

1. retain `echozero/audio/engine.py` as the near-term playback backend
2. move playback target resolution, backend timing snapshots, and output-session metadata behind `echozero/application/playback/*`
3. keep the UI on timing snapshots and repaint cadence only
4. continue treating Qt audio as fallback only if the current backend fails real hardware validation after the service-boundary cleanup

## Why This Wins

### Current engine strengths

- The current backend already owns the callback clock, mixer, and stream lifecycle in one place: [echozero/audio/engine.py](../../echozero/audio/engine.py)
- The branch already has working tests for:
  - default device format negotiation
  - reported output latency
  - backend timing snapshots
  - playhead extrapolation from backend timing
- The existing design direction in [docs/PLAYBACK-CLOCK-CLEAN-SHEET-DESIGN.md](PLAYBACK-CLOCK-CLEAN-SHEET-DESIGN.md) already points to "keep the current `AudioEngine`, but only behind an explicit playback boundary"

### Why not switch to `QAudioSink` now

- `QAudioSink` gives buffer/session control and processed elapsed time, but it does not replace the current application-side mixing/rendering problem by itself
- A backend swap now would combine two risks:
  - architectural cleanup
  - backend replacement
- The current failure mode under remediation is boundary ownership and device/session correctness, not proof that callback-driven playback is the wrong model

## Evidence

### Repo evidence

- Device-native output defaults are already implemented and tested in [echozero/audio/engine.py](../../echozero/audio/engine.py) and [tests/test_audio_engine.py](../../tests/test_audio_engine.py)
- Backend timing snapshot plumbing already exists in [echozero/application/playback/runtime.py](../../echozero/application/playback/runtime.py) and [tests/ui/test_runtime_audio.py](../../tests/ui/test_runtime_audio.py)
- The clean-sheet plan explicitly prefers current backend + explicit boundary in [docs/PLAYBACK-CLOCK-CLEAN-SHEET-DESIGN.md](PLAYBACK-CLOCK-CLEAN-SHEET-DESIGN.md)

### Primary-source references

- PortAudio API overview: timing information is exposed from stream callbacks and is intended for synchronization with GUI/audio state
  - https://portaudio.com/docs/v19-doxydocs/api_overview.html
- PortAudio timing/buffering guidance:
  - https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines
- `python-sounddevice` callback streams:
  - callback buffers expose timing information
  - `blocksize=0` is recommended for robust callback behavior when fixed-size callbacks are not required
  - https://python-sounddevice.readthedocs.io/en/latest/api/streams.html
- Qt `QAudioSink`:
  - useful for buffer/session inspection and as a fallback device sink
  - not selected here as the near-term default
  - https://doc.qt.io/qt-6/qaudiosink.html

## Tradeoffs

### Accepted

- We keep the callback backend complexity for now
- Real hardware validation is still required before final playback signoff

### Avoided

- Avoided rewriting the playback backend and the playback boundary in one step
- Avoided anchoring the system to a Qt-specific audio path before the app contract was clean

## Exit Criteria For This Decision

This decision remains valid if all are true:

- playback target stays explicit and separate from selection
- playback runtime metadata is published through the application boundary
- playhead rendering uses backend timing snapshots rather than timer truth
- real playback on target hardware is audibly correct

If real hardware validation still shows unrecoverable device/format brittleness after the service-boundary cleanup, reopen PB-20 and test a `QAudioSink` fallback branch.
