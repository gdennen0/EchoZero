# EZ2 Playback Churn Stress

Date: 2026-04-07
Scope: Real-path runtime playhead behavior under rapid seek/play/mute/solo churn in the EZ2 Qt timeline shell.

## Protocol

1. Use `TimelineWidget` with attached `TimelineRuntimeAudioController`-compatible fakes so the widget runs the same `_dispatch()` and `_on_runtime_tick()` paths used in runtime playback.
2. Drive the widget through rapid transport and mix churn inside `tests/ui/test_runtime_audio.py`.
3. Assert two invariants:
   - runtime ticks plus mix toggles never snap the visible playhead backward while playback is still advancing;
   - an explicit backward seek is allowed, but stale runtime samples after that seek cannot pull the playhead behind the seek target during subsequent mute/solo churn.
4. Keep the code patch minimal and local to widget runtime reconciliation.

## Code change under stress

- `echozero/ui/qt/timeline/widget.py`
- Added a small runtime playhead floor inside `TimelineWidget`.
- Runtime ticks now clamp stale backward samples while playback is active.
- Explicit `Seek` intents reset the floor to the requested seek position so intentional backward seeks still render immediately.
- `Stop` and paused runtime state clear the floor.

## Measured observations

### Churn pass A: runtime ticks plus mix toggles

Test: `test_widget_runtime_ticks_do_not_snap_backward_during_mix_toggle_churn`

Observed visible playhead samples:
- `4.000`
- `4.042`
- `4.042` after `ToggleMute` with stale runtime sample `4.018`
- `4.042` on the next runtime tick with the same stale sample
- `4.042` after `ToggleSolo` with stale runtime sample `4.019`
- `4.042` on the next runtime tick with the same stale sample
- `4.083` once the runtime clock advances again

Result: no backward snap during runtime ticks or mix churn.

### Churn pass B: play, backward seek, then mute/solo churn

Test: `test_widget_seek_churn_keeps_seek_anchor_through_stale_runtime_samples`

Observed visible playhead samples:
- `1.000` after `Play`
- `1.040` after a forward runtime tick
- `0.750` after explicit backward `Seek(0.75)`
- `0.750` on stale runtime tick `0.710`
- `0.750` after `ToggleMute` with stale runtime sample `0.720`
- `0.750` after `ToggleSolo` with stale runtime sample `0.710`
- `0.810` once runtime advances past the seek anchor

Result: explicit backward seek remains immediate, and stale post-seek churn does not pull the playhead behind the seek target.

## Focused test command

Command:

```powershell
py -3.12 -m pytest tests\ui\test_runtime_audio.py -q
```

Result:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\griff\EchoZero\.worktrees\panel-playback-core
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
plugins: cov-7.0.0
collected 9 items

tests\ui\test_runtime_audio.py .........                                 [100%]

============================== 9 passed in 0.89s ==============================
```

## Residual note

- This pass stabilizes visible runtime playhead updates against short-lived backward clock samples during churn.
- It does not add interpolation or change follow-mode behavior; those remain separate concerns.
