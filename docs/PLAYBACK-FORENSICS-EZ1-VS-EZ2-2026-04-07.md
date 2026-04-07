# Playback Forensics: EZ1 vs EZ2

Date: 2026-04-07
Scope: Minimal playback/timeline comparison only.

## EZ1 references

### Timing source and playhead updates
- `ui/qt_gui/widgets/timeline/playback/controller.py:48-58` uses a `QTimer` plus `QElapsedTimer` for 60 FPS playhead updates.
- `ui/qt_gui/widgets/timeline/playback/controller.py:201-244` treats backend position as ground truth when it changes, otherwise interpolates from the last known position.
- `ui/qt_gui/widgets/timeline/core/widget.py:2290-2293` wires controller playback signals into the widget.
- `ui/qt_gui/widgets/timeline/core/widget.py:2502-2507` mirrors controller position into scene, ruler, label, and follow logic on every tick.

### Follow behavior
- `ui/qt_gui/widgets/timeline/core/widget.py:2517-2523` converts follow target time to horizontal scroll pixels.
- `ui/qt_gui/widgets/timeline/core/widget.py:2524-2543` continuously lerps scroll toward `_target_scroll_x`.
- `ui/qt_gui/widgets/timeline/core/widget.py:2545-2590` splits follow into `PAGE`, `SMOOTH`, and `CENTER`.
- `ui/qt_gui/widgets/timeline/core/widget.py:2563-2573` keeps `PAGE` as threshold-based jumps near the viewport edges.
- `ui/qt_gui/widgets/timeline/core/widget.py:2575-2590` keeps `SMOOTH` at 75% and `CENTER` at 50%, both using continuous interpolation.

### Seek/playhead path
- `ui/qt_gui/widgets/timeline/core/widget.py:2267-2287` routes seek from view drag, scene playhead seek, and ruler click into `_on_seek`.
- `ui/qt_gui/widgets/timeline/core/widget.py:2466-2474` syncs controller position to the scene playhead before play starts.
- `ui/qt_gui/widgets/timeline/core/widget.py:2484-2500` snaps seek when grid snap is enabled, then updates controller, scene, ruler, and position label in one path.

## EZ2 references

### Current timing source
- `echozero/ui/qt/timeline/runtime_audio.py:91-107` delegates play/pause/stop/seek to `AudioEngine`, with current time read from `engine.clock.position_seconds`.
- `echozero/ui/qt/timeline/widget.py:816-819` polls runtime state every 16 ms with `_runtime_timer`.
- `echozero/ui/qt/timeline/widget.py:899-928` copies runtime time and playing state into presentation on each tick, then reapplies follow scroll.

### Current follow behavior
- `echozero/ui/qt/timeline/widget.py:111-140` computes follow scroll directly from presentation state.
- `echozero/ui/qt/timeline/widget.py:131-137` makes `PAGE` and `CENTER/SMOOTH` pixel-target calculations, but `CENTER` and `SMOOTH` currently share the same centering formula.
- `echozero/ui/qt/timeline/widget.py:823-845` reapplies follow during `set_presentation`.
- `echozero/ui/qt/timeline/widget.py:919-928` reapplies follow again during runtime ticks.

### Current seek/playhead path
- `echozero/ui/qt/timeline/widget.py:799-805` routes canvas drag and ruler seek into `_seek`.
- `echozero/ui/qt/timeline/widget.py:930-931` turns UI seek into a `Seek` intent only.
- `echozero/ui/qt/timeline/widget.py:883-897` overwrites intent-updated presentation playhead with runtime audio time when runtime audio is attached.

## Gap matrix

| Area | EZ1 | EZ2 | Gap |
|---|---|---|---|
| Jitter handling | Backend-grounded interpolation between backend updates (`playback/controller.py:201-244`) | Pure polling of runtime clock every 16 ms (`widget.py:816-819`, `899-928`) | EZ2 has no interpolation or smoothing layer between runtime clock samples and UI paint. |
| Alignment | Seek updates controller, scene, ruler, and label in one method (`core/widget.py:2484-2500`) | Seek becomes intent first; display is later corrected from runtime clock (`widget.py:930-931`, `883-897`) | EZ2 can show transient UI/runtime divergence after seek or transport changes. |
| Follow modes | Distinct `PAGE`, `SMOOTH`, `CENTER`; smooth modes use continuous lerp (`core/widget.py:2545-2590`) | `PAGE` is distinct, but `CENTER` and `SMOOTH` collapse to the same centering rule (`widget.py:131-137`) | EZ2 lost the behavioral difference between smooth trailing and hard centering. |
| Seek response | Optional snap plus immediate playhead/ruler redraw (`core/widget.py:2484-2500`) | Intent dispatch only; visual truth depends on subsequent presenter/runtime update (`widget.py:930-931`, `899-928`) | EZ2 seek feedback is less immediate and can lag one tick. |
| Playback start sync | Play action syncs controller to scene playhead before starting (`core/widget.py:2466-2474`) | No equivalent widget-side pre-play runtime sync in the cited path | EZ2 risks starting from stale runtime clock state unless handled elsewhere in intent/runtime code. |

## Top 5 migration fixes

1. Add a presentation-side interpolation layer in EZ2 so runtime clock polling does not directly drive visible playhead jumps.
2. Split EZ2 `SMOOTH` from `CENTER`; keep `SMOOTH` trailing near 75% and reserve 50% centering for `CENTER`.
3. Make seek optimistic in the widget/presenter path: update playhead, ruler, label, and scroll immediately before waiting for the next runtime tick.
4. Add pre-play synchronization so runtime audio is explicitly sought to the current presentation playhead before `Play`.
5. Collapse transport/seek state updates into one canonical path so follow, scroll clamping, and playhead display stay aligned across drag, ruler click, and transport actions.
