# EchoZero Audio Engine — Ship-Readiness Audit

**Date:** 2026-03-29  
**Auditor:** Chonch (AI Technical Audit)  
**Scope:** `echozero/audio/` (clock.py, transport.py, layer.py, mixer.py, engine.py, crossfade.py) + `tests/test_audio_engine.py`  
**Verdict:** ❌ **Not ship-ready.** Several 🔴 issues must be fixed before release.

---

## Rating Key

- 🔴 **Must fix before ship** — correctness bug, data corruption, or crash risk
- 🟡 **Should fix** — reliability, correctness at edge cases, or likely to surface in production
- 🟢 **Nice to have** — quality-of-life, polish, future-proofing

---

## 1. Correctness Issues

### 🔴 `Mixer.read_mix()` — Scratch Buffer Overflow / Overlap Corruption

**File:** `mixer.py:97–101`

```python
layer_buf = scratch[frames:frames + frames]
layer.read_into(layer_buf, position, frames)
```

The scratch buffer is allocated as `_MAX_SCRATCH_FRAMES = 8192`. The output occupies `scratch[0:frames]` and the per-layer temp region is carved from `scratch[frames:frames * 2]`. If `frames > 4096`, this writes out of bounds of the pre-allocated scratch (8192 samples total = 4096 + 4096). At `frames=4096` it just barely fits. At `frames=4097+` it silently corrupts memory or raises `IndexError`.

Worse: if `frames > _MAX_SCRATCH_FRAMES / 2`, the layer buffer region overlaps the output accumulator region in scratch, causing a write-while-read corruption.

**Fix:** Either assert `frames <= _MAX_SCRATCH_FRAMES // 2` at engine init, or use a second pre-allocated scratch buffer of fixed size `_MAX_SCRATCH_FRAMES`.

---

### 🔴 `engine._audio_callback()` — Third `read_mix()` Call Corrupts Split-Read Data

**File:** `engine.py:183–185`

```python
mixed = self._mixer.read_mix(position, frames)  # just to get the scratch view
mixed[:wrap_offset] = pre_audio
mixed[wrap_offset:] = post_audio[:remaining]
```

The comment says "just to get the scratch view," but `read_mix()` fully re-runs the mixer on the audio thread, overwriting the scratch buffer. Immediately after, `pre_audio` and `post_audio` are written back into that same scratch view. This works *only because* `pre_audio` and `post_audio` were `.copy()`-ed earlier (lines 174–179). However, it is a third full mixer pass (CPU waste), and it reads at `position` which has already wrapped — meaning this third pass reads the **wrong position** (the post-wrap position, not the original pre-wrap start). This third call is entirely unnecessary and semantically wrong.

**Fix:** Allocate a small fixed-size output buffer (same size as scratch) that is NOT the mixer's internal scratch. Write `pre_audio`/`post_audio` directly into it. Never make a third `read_mix()` call for this purpose.

---

### 🔴 `engine._audio_callback()` — Crossfade Applied AFTER Hard Clip

**File:** `engine.py` (crossfade path)

The flow is:
1. `pre_mix = self._mixer.read_mix(...)` → result is already hard-clipped to [-1, 1] by `np.clip()` inside `read_mix()`
2. `.copy()` into `pre_audio`
3. Same for `post_audio`
4. Crossfade blend: `tail * fade_out + head * fade_in`

The crossfade blends two signals that have each been independently hard-clipped. The blended result can produce values up to `1.0 * fade_out + 1.0 * fade_in`. At the equal-power midpoint: `cos(π/4) + sin(π/4) = √2 ≈ 1.414`. The crossfade output goes to `outdata[]` **without a final clip**, so the DAC receives values outside [-1, 1].

**Fix:** Apply `np.clip()` to the final assembled `mixed` buffer after the crossfade on the split-read path, mirroring the clip that already exists in the normal path via `read_mix()`.

---

### 🔴 `Clock.advance()` — Loop Multiple Wraps: Position Corruption

**File:** `clock.py:95–100`

```python
if new_pos >= snap.region.end:
    self._last_wrap_offset = snap.region.end - read_pos
    overshoot = new_pos - snap.region.end
    loop_len = snap.region.end - snap.region.start
    new_pos = snap.region.start + (overshoot % loop_len)
```

`last_wrap_offset` is set only to the **first** wrap point, which is correct. However, the engine's split-read logic in `_audio_callback` uses `last_wrap_offset` to assume the wrap happened at a single point within the buffer and reads `remaining = frames - wrap_offset` samples from `loop_start`. If the loop is shorter than `frames` (e.g. loop_len=100, frames=256), `overshoot % loop_len` correctly wraps the clock position, but `post_audio` is read as a contiguous block of `remaining=156` samples from `loop_start` — which crosses the loop end point again. The post-wrap audio is silently wrong (reads past loop end into undefined content), and there's no second split+crossfade.

**The test `test_loop_wraps_multiple_times` only checks the clock position, not the audio content.**

**Fix:** Engine must handle the case where `remaining > loop_len` by either (a) clamping `remaining` to `loop_len` and filling the rest with another read, or (b) asserting `buffer_size <= loop_length_samples` and documenting it as a hard constraint.

---

### 🟡 `Clock.advance()` — `last_wrap_offset = 0` Not Handled

**File:** `engine.py:164`

```python
if wrap_offset > 0 and loop_region is not None:
```

The condition checks `wrap_offset > 0`. But `wrap_offset == 0` means the wrap happens at the very first sample of the buffer — i.e. `read_pos == snap.region.end`. In this case `wrap_offset` is 0, the condition is false, and the normal path runs. The normal path reads from `position` (which after `advance()` is `loop_start + overshoot`) — which is correct for content, but no crossfade is applied. A click is possible.

**Fix:** Change condition to `wrap_offset >= 0` and handle `wrap_offset == 0` as a degenerate crossfade (just use head audio from position 0, no tail).

---

### 🟡 `Transport.seek()` — Stop Position Updated Only When Stopped

**File:** `transport.py:55–58`

```python
def seek(self, position_samples: int) -> None:
    self._clock.seek(position_samples)
    if self._state == TransportState.STOPPED:
        self._stop_position = max(0, position_samples)
```

If the user seeks while playing (e.g. scrubs to a new position), `_stop_position` is NOT updated. When stop() is later called, the transport returns to the *pre-seek* position, not the seek position. This is a surprising and almost certainly wrong behavior — the user scrubbed to a new location and then stopped; they expect stop to return to where they last seeked, not some prior location.

**Fix:** Update `_stop_position` regardless of state, or at minimum document this as a deliberate design choice with a rationale.

---

### 🟡 `AudioLayer.read_into()` — No Guard Against `frames > len(out)`

**File:** `layer.py:80`

```python
out[:frames] = 0.0
```

If `frames > len(out)`, this silently writes nothing (numpy slice beyond array end is a no-op on zeros, but assignment works up to the array length). The `out_end` calculation is also unchecked against `len(out)`. If the caller accidentally passes a `frames` larger than the pre-allocated buffer, the write is silently truncated. For DAW infrastructure, a silent data loss bug is unacceptable.

**Fix:** Add `assert len(out) >= frames` at the top of `read_into()`, or raise `ValueError`.

---

### 🟡 `resample_buffer()` — Empty Buffer Crashes

**File:** `layer.py:30–40`

If `buffer` has length 0: `new_len = 0`, `indices` is empty, but `idx_floor` and `idx_ceil` are also empty, and `idx_ceil = np.minimum(idx_floor + 1, len(buffer) - 1)` computes `np.minimum(..., -1)`, which is valid but semantically wrong. If `source_sr != target_sr` and `len(buffer) == 0`, you'll get an empty float32 array — probably fine, but untested.

**Fix:** Add early return for empty buffer: `if len(buffer) == 0: return buffer`.

---

## 2. What-If Scenarios

### 🔴 What if the user seeks during a crossfade?

The crossfade state (`CrossfadeBuffer`) is purely functional — `apply()` takes its inputs explicitly, and there's no persistent "currently crossfading" state. A seek mid-callback can't corrupt the crossfade object itself.

**However:** `seek()` writes to `_clock._position` from the main thread while the audio callback reads `_clock._position` (via `advance()`) on the RT thread. Python's GIL makes individual int reads/writes atomic, which is the stated assumption. The real problem is that a seek during the split-read path (between the first and second `read_mix()` calls in the loop-wrap branch) cannot be atomically detected — the position changes mid-buffer assembly. The split-read will use the pre-seek `position` for `pre_audio` but the crossfade placement is now wrong because `last_wrap_offset` was set based on the pre-seek position. Result: a partial-buffer glitch click. Not catastrophic, but audible.

**Additionally:** A seek while `_transport.is_playing` is True but before `advance()` runs causes `advance()` to use the new seek position as `read_pos` — correct behavior. A seek *after* `advance()` but before `outdata[:] =` assignment means we advance with the right position but then seek was wasted — also fine. The window is narrow enough that the GIL protects it in CPython, but this is a correctness assumption that must be documented.

**Fix:** Document the GIL dependency explicitly. Consider adding a `volatile_seek` flag that the callback checks and skips the current buffer (outputs silence) on a seek, preventing the torn-read scenario.

---

### 🔴 What if buffer size > loop length?

As documented in Issue #4 above (multiple-wrap corruption), this is a hard crash/corruption scenario. The engine makes no assertion and no documentation states this is prohibited.

**Fix:** In `_open_stream()` or `play()`, assert `buffer_size <= min(loop_length_samples)` when loop is enabled. Or handle multi-wrap within a single buffer in `_audio_callback`.

---

### 🟡 What if a layer's sample rate doesn't match after resample?

`AudioLayer.__init__` calls `resample_buffer()` which uses linear interpolation. After resampling, `self.sample_rate` is set to `target_sr`. If the caller checks `layer.sample_rate` and it equals `engine.sample_rate`, that's correct. But:

- `resample_buffer()` uses `int(len(buffer) * ratio)` which truncates. A 48000-sample buffer at 48kHz resampled to 44100 Hz gives `int(48000 * 44100/48000) = int(44100.0) = 44100`. This is correct here, but for non-integer ratios (e.g. 22050→44100: exact; 44100→48000: `int(44100 * 48000/44100) = int(48000.0) = 48000` — also exact). For arbitrary rates like 32000→44100: `int(32000 * 44100/32000) = int(44100.0) = 44100`. The `abs(len(result) - 44100) <= 1` tolerance in tests covers off-by-ones, but the buffer could be 1 sample short — meaning `duration_seconds` is very slightly wrong. Not a crash, but can cause subtle off-by-one in loop end detection.

- There is no post-resample validation that `len(buffer) > 0`. A zero-length buffer gets through silently.

**Fix:** After resample, verify `len(buffer) > 0`. Log a warning if off-by-more-than-1. Add a test for a non-standard sample rate like 32000→44100.

---

### 🔴 What if all layers are removed while playing?

`Mixer.read_mix()` snapshots `layers = self._layers` at the top. If `clear()` or `remove_layer()` is called from the main thread between callback invocations, the next callback sees an empty list and returns silence immediately. **This is safe.**

However: `_audio_callback` checks `duration = self._mixer.duration_samples` after advancing the clock. `duration_samples` on an empty mixer returns `0`. The condition is:

```python
if duration > 0 and not self._clock.loop_enabled:
    if position >= duration:
        ...
```

With `duration == 0`, the condition `duration > 0` is False — so the auto-stop check is skipped. The engine will keep playing (advancing the clock, outputting silence) indefinitely. It will never auto-stop. If the user removed all layers expecting the track to stop, it won't. Clock keeps running.

**Fix:** When `duration == 0` and not looping and there are no layers, either auto-pause or document that removing all layers does not auto-stop.

---

### 🟡 What if the crossfade length > half the loop region?

`xfade_len = min(xfade.length, wrap_offset, remaining)` in the engine clamps the crossfade to fit in the buffer. But if `xfade.length > loop_length / 2`, consecutive crossfades will overlap in the audio content — the fade-in from wrap N will still be playing when the fade-out for wrap N+1 begins. This produces a flanger/chorus artifact rather than a click, but the perceived loudness and timbre will be wrong.

At the default 4ms crossfade and 44100 Hz, the crossfade is 176 samples. Any loop shorter than 352 samples (~8ms) will exhibit this artifact. Users can set arbitrary loop regions including very short ones (the `LoopRegion` only validates `end > start` with no minimum).

**Fix:** Enforce a minimum loop length of `2 * crossfade_samples` in `set_loop()`. Document the constraint. Or dynamically reduce crossfade length when loop region is short.

---

### 🟡 What if two seeks happen in rapid succession?

Under the GIL, each `self._position = max(0, position_samples)` in `Clock.seek()` is atomic. Two seeks from the main thread in the same Python thread execute sequentially — the second wins. From two Python threads, the GIL serializes them; second wins. From a thread + callback: the callback calls `advance()` which reads `_position`, then seeks happen, then callback writes `new_pos` back. If a seek happens after `read_pos = self._position` but before `self._position = new_pos`, the advance will overwrite the seek's value with `new_pos = seek_value + frames` — effectively a seek-then-advance, which is correct behavior.

The real problem is `_stop_position` in Transport. Two seeks in STOPPED state both update `_stop_position`; whichever runs second wins. That's fine. Two seeks in PLAYING state — `_stop_position` is never updated (per Issue #5), so both seeks are equally "ignored" for stop-return-position. Consistent but wrong.

**No crash, but correctness degrades.** Document as a known limitation or fix Issue #5.

---

### 🟡 What if sounddevice reports a status error in the callback?

**File:** `engine.py:152`

```python
def _audio_callback(self, outdata, frames, time_info, status):
```

`status` is a `sounddevice.CallbackFlags` object. The engine completely ignores it. A non-zero status indicates buffer underrun (`output_underflow`), buffer overflow, or other device errors. These are the most actionable signals for detecting audio glitches.

The comment "No exceptions" in the docstring refers to not *raising* exceptions in the callback, which is correct. But silently discarding status means:
- UI team gets no feedback on glitches
- Debug builds can't detect underruns
- Production gets silent audio artifacts with no log entry

**Fix:** At minimum, log `status` to a lock-free ring buffer readable by the main thread (not `logging.warning()` — that allocates). Consider exposing a `glitch_count` counter (atomic int increment is GIL-safe) so the UI can display a glitch indicator.

---

### 🔴 What if a subscriber throws an exception?

**File:** `clock.py:104–106`

```python
subs = self._subscribers
for sub in subs:
    sub.on_clock_tick(read_pos, self._sample_rate)
```

No try/except. If any subscriber raises, the exception propagates out of `advance()`, back into `_audio_callback()`, and out of the callback. `sounddevice` will catch it (to prevent RT thread crash), silently abort the callback, and the stream may stop. This is essentially: **one bad UI subscriber kills audio for all subscribers and potentially the entire stream.**

**Fix:** Wrap each `sub.on_clock_tick()` call in a try/except. On exception, remove the offending subscriber from the list (copy-on-write), increment an error counter. Never let a subscriber crash the audio thread.

```python
bad_subs = []
for sub in subs:
    try:
        sub.on_clock_tick(read_pos, self._sample_rate)
    except Exception:
        bad_subs.append(sub)
if bad_subs:
    # schedule removal — can't acquire lock here
    self._pending_remove.extend(bad_subs)
```

Or simpler: bare `except Exception: pass` with an atomic error counter.

---

## 3. Performance Concerns

### 🔴 `engine._audio_callback()` — Three `read_mix()` Calls on Loop Wrap Path

**File:** `engine.py:171–185`

On every buffer that contains a loop wrap, the engine calls `read_mix()` three times:
1. `pre_mix = self._mixer.read_mix(position, wrap_offset)` — correct
2. `post_mix = self._mixer.read_mix(loop_region.start, remaining)` — correct
3. `mixed = self._mixer.read_mix(position, frames)` — **wrong and wasteful**

Each `read_mix()` iterates all layers, calls `read_into()` on each, accumulates, clips. At 8 layers, that's 24 layer reads per wrap event instead of 16. At 44100 Hz with a 256-sample buffer and a short loop, wraps happen every ~6ms = every callback. This is a constant 50% CPU overhead on the mix step for looping content.

**Fix:** Eliminate the third call as described in Issue #2.

---

### 🟡 `Mixer.read_mix()` — `any(l.solo for l in layers)` Every Callback

**File:** `mixer.py:104`

```python
any_solo = any(l.solo for l in layers)
```

This iterates all layers on every callback to check solo state. With 20 layers this is 20 attribute reads per ~5ms. For a simple flag, maintain a `_solo_count` integer that's incremented/decremented on `solo_exclusive()` / `unsolo_all()` and whenever `layer.solo` is mutated. This makes the hot-path check O(1) instead of O(n).

However, since `layer.solo` is a plain attribute that can be set directly on the `AudioLayer` object (bypassing Mixer entirely), `_solo_count` would be stale. Either make `solo` a property with a callback or document that solo must be set through Mixer methods.

---

### 🟡 `Clock.advance()` — Subscriber Notification Allocates on Exception Path

The subscriber loop itself is allocation-free when everything succeeds. But the Python interpreter's internal exception tracking machinery (if any subscriber raises) allocates. This is minor and only happens on error paths, but combine with Issue (subscriber exception) above.

---

### 🟢 `Mixer.read_mix()` — Hard Clip Uses Temporary View

```python
np.clip(out, -1.0, 1.0, out=out)
```

This is in-place and allocation-free. ✅ But it runs even when the summed signal is well within [-1, 1] (e.g. a single quiet layer). Consider a fast-path check: `if np.any((out > 1.0) | (out < -1.0))` before calling `np.clip`. However, `np.any` itself iterates the array, so this is only a win if clipping is rare. Leave as-is for now.

---

### 🟢 `resample_buffer()` — Linear Interpolation Quality

As noted in the source comment, linear interpolation introduces aliasing artifacts at high frequencies during downsampling. For a production DAW this will be audible on content above Nyquist/2. The comment acknowledges this and defers to libsamplerate. Fine for v1 preview builds, but must be replaced before final ship.

---

## 4. Missing Test Coverage

### 🔴 No Test: Buffer Size > Loop Length (Multiple Wraps)

The case where `frames > loop_length_samples` is never tested. This is the scenario that produces silent data corruption (see Issue #4). A test like:

```python
def test_callback_buffer_larger_than_loop():
    engine = AudioEngine(buffer_size=512, ...)
    engine.add_layer(...)
    engine.clock.set_loop(0, 100)  # 100 samples, buffer is 512
    engine.clock.loop_enabled = True
    engine.play()
    # This should either work correctly or raise a clear error
    outdata = np.zeros((512, 1), dtype=np.float32)
    engine._audio_callback(outdata, 512, None, None)
    # Verify output is valid (no NaN, no values outside [-1, 1])
    assert not np.any(np.isnan(outdata))
    assert np.max(np.abs(outdata)) <= 1.0
```

---

### 🔴 No Test: Subscriber Exception Isolation

No test verifies that a throwing subscriber doesn't kill the audio thread or stop other subscribers from receiving ticks.

```python
def test_subscriber_exception_does_not_kill_audio():
    clock = Clock(44100)
    bad_sub = BadSubscriber()  # raises on every tick
    good_sub = RecordingSubscriber()
    clock.add_subscriber(bad_sub)
    clock.add_subscriber(good_sub)
    clock.advance(256)
    assert len(good_sub.ticks) == 1  # good sub still fires
```

---

### 🔴 No Test: seek() During Active Playback + Stop Returns Wrong Position

The Transport.seek()-while-playing / stop-returns-wrong-position bug (Issue #5) has no test. Adding one would immediately expose the bug.

---

### 🟡 No Test: Crossfade Clipping

No test verifies that the crossfade output stays within [-1, 1] when tail and head are both at full scale (1.0). Given the equal-power issue identified in Issue #3, this test would fail:

```python
def test_crossfade_does_not_exceed_unity():
    xfade = CrossfadeBuffer(64)
    output = np.zeros(128, dtype=np.float32)
    tail = np.ones(64, dtype=np.float32)
    head = np.ones(64, dtype=np.float32)
    xfade.apply(output, tail, head, 32, 64)
    assert np.max(np.abs(output[32:96])) <= 1.0  # FAILS: peaks at √2 ≈ 1.414
```

---

### 🟡 No Test: add_layer() With Multichannel (2D) Buffer

The mixer and layer assume 1D float32 buffers. What happens if someone passes a stereo (N×2) array? `read_into()` does `out[:frames] = self.buffer[buf_start:buf_end]` where both sides are 2D/1D mismatched — numpy will broadcast or raise. There's no validation and no test.

---

### 🟡 No Test: Empty Mixer Auto-Stop Behavior

No test verifies what happens when all layers are removed during playback (see Issue #6). Expected behavior is ambiguous and untested.

---

### 🟡 No Test: `wrap_offset == 0` Edge Case

No test for a loop wrap occurring at exactly the first sample of the buffer. The condition `if wrap_offset > 0` silently mishandles this case.

---

### 🟡 No Test: `sounddevice` Status Error Handling

No test passes a non-None `status` to `_audio_callback` to verify it doesn't crash or corrupt state.

---

### 🟡 No Test: `Mixer._scratch` Overflow

No test exercises `read_mix()` with `frames > _MAX_SCRATCH_FRAMES // 2`. This would catch the buffer overlap corruption silently failing or raising.

---

### 🟢 No Test: Layer `volume=0.0` (Silence)

Edge case: a layer with volume=0.0 should produce silence without NaN/inf.

---

### 🟢 No Test: Non-Standard Sample Rate Resample (e.g. 32000→44100)

Only 22050↔44100 and 48000→44100 are tested. A 32000 Hz source has a non-integer ratio.

---

## 5. API Design Issues

### 🔴 `Mixer.read_mix()` Returns a View Into Internal Scratch Buffer

**File:** `mixer.py:131`

```python
return out  # view into self._scratch
```

The docstring says: *"Note: we return a view, so caller must consume before next read_mix call."* This is a ticking time bomb. The UI team will store this array, pass it around, and eventually read stale data or see it mutated. Python has no `const` reference mechanism.

The engine partially handles this with `.copy()` on the loop-wrap path, but on the **normal path** (no wrap), `outdata[:, 0] = mixed` is fine only because the assignment happens in the same callback. But `mixed` is returned to any caller of `read_mix()` — if a test or future code stores it, it's a silent bug.

**Fix:** Either document this extremely loudly (the current docstring buries it), return a copy always (costs one allocation per callback), or rename to `read_mix_into(output_buffer)` where the caller provides the destination.

---

### 🟡 `AudioEngine` Exposes `clock`, `transport`, `mixer` Directly

The engine properties `clock`, `transport`, and `mixer` return the raw internal objects. The UI team can call `engine.clock.advance()` or `engine.mixer._scratch` directly, bypassing all the engine's coordination logic. This has already caused one design issue: the test accesses `engine._end_of_content` directly.

**Fix:** For v1 this is acceptable — internal access is needed for testing. But document clearly which methods are "public API" vs "internal/test-only". Consider a `__all__` or explicit private naming.

---

### 🟡 `CrossfadeBuffer.apply()` Signature Is Confusing

```python
def apply(self, output, tail_audio, head_audio, xfade_start, xfade_len):
```

- `tail_audio` and `head_audio` must already be sliced to `xfade_len` or at least that long. The function silently truncates via `[:xfade_len]` slicing. If the caller passes shorter arrays, numpy silently produces a smaller crossfade region without error.
- `xfade_start` is an index into `output`, not into `tail_audio` or `head_audio`. This is non-obvious.
- The internal `_tail_buf` and `_head_buf` scratch buffers are allocated but never used in `apply()`. They appear to be vestigial from an earlier design where `apply()` did its own mixer reads.

**Fix:** Remove unused `_tail_buf`/`_head_buf` from `CrossfadeBuffer` (saves memory, reduces confusion). Add parameter validation. Add a docstring example showing exactly how to call it.

---

### 🟡 `Transport.return_to_start()` Has Inconsistent State Semantics

```python
def return_to_start(self) -> None:
    self._stop_position = 0
    self._clock.seek(0)
    if self._state != TransportState.PLAYING:
        self._state = TransportState.STOPPED
```

If called while PAUSED, this moves to position 0 AND transitions to STOPPED (kills the paused state). If called while PLAYING, it only seeks — transport stays PLAYING. This asymmetry is surprising: "return to start while paused" behaves differently than "seek to 0 while paused." There's also no test for `return_to_start()` while paused.

---

### 🟡 `Clock.sample_rate` Is Mutable After Init

```python
@sample_rate.setter
def sample_rate(self, value: int) -> None:
    self._sample_rate = value
```

Changing `sample_rate` after layers have been added invalidates all sample-position calculations without any warning. Layers store `original_sample_rate` and are already resampled; the clock's sample rate change does not trigger re-resampling. Any loop region set in samples is now measured in the wrong unit.

**Fix:** Remove the setter or add validation that no layers are loaded when sample rate is changed. At minimum, raise `RuntimeError` if the stream is active.

---

### 🟢 `LoopRegion` Has No Minimum Length Check

`LoopRegion(start=0, end=1)` is valid — a 1-sample loop at 44100 Hz. The crossfade is 176 samples long. Every callback will be a wrap event. Every wrap will have `xfade_len = min(176, 1, buffer_size-1)` → 1 sample. This is valid code but useless audio. A minimum loop length equal to `2 * crossfade_samples` would prevent obviously broken configurations.

---

### 🟢 No `__repr__` on `AudioLayer`

For debugging in the REPL or logs, `print(layer)` gives `<echozero.audio.layer.AudioLayer object at 0x...>`. Add a `__repr__` showing id, name, duration, and sample rate.

---

## Summary Table

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | 🔴 | Mixer | Scratch buffer overflow/overlap when `frames > 4096` |
| 2 | 🔴 | Engine | Third `read_mix()` call on loop-wrap path reads wrong position and corrupts output |
| 3 | 🔴 | Engine | Crossfade output can exceed [-1, 1] — no final clip |
| 4 | 🔴 | Clock/Engine | Buffer size > loop length causes multi-wrap audio corruption |
| 5 | 🟡 | Transport | `seek()` while playing doesn't update `_stop_position` → stop returns to wrong position |
| 6 | 🟡 | Layer | No bounds check in `read_into()` — silent data loss if `frames > len(out)` |
| 7 | 🟡 | Layer | `resample_buffer()` crashes on empty buffer |
| 8 | 🔴 | Clock | Subscriber exception kills audio thread |
| 9 | 🟡 | Engine | `sounddevice` status errors silently ignored |
| 10 | 🟡 | Engine | `wrap_offset == 0` silently skips crossfade |
| 11 | 🟡 | Clock | `last_wrap_offset` only set for first wrap — misleads engine on short loops |
| 12 | 🔴 | Mixer | Returns scratch buffer view — aliasing hazard for callers |
| 13 | 🟡 | Mixer | `any_solo` O(n) scan every callback |
| 14 | 🟡 | Transport | `return_to_start()` inconsistent state behavior while paused |
| 15 | 🟡 | Clock | `sample_rate` mutable post-init without invalidating layer data |
| 16 | 🟢 | Engine | Loop wrap CPU: 3 mixer passes instead of 2 |
| 17 | 🟢 | Layer | Linear interpolation resampling — aliasing on downsample |
| 18 | 🟢 | Loop | No minimum loop length enforcement vs crossfade duration |
| 19 | 🟢 | CrossfadeBuffer | Vestigial `_tail_buf`/`_head_buf` never used |

**🔴 Must-fix count: 6**  
**🟡 Should-fix count: 9**  
**🟢 Nice-to-have count: 4**

---

## Recommended Fix Priority

1. **Fix Issue #8 first** (subscriber exception) — this is a guaranteed crash in production the moment any UI code throws in a clock callback.
2. **Fix Issue #2 + #3 together** (wrong third `read_mix()` + missing final clip) — same code path, fix simultaneously.
3. **Fix Issue #1** (scratch buffer overflow) — add assertion or second scratch buffer.
4. **Fix Issue #4** (buffer > loop length) — add engine assertion and document hard constraint.
5. **Fix Issue #12** (scratch view aliasing) — add `.copy()` to `read_mix()` return, or redesign to take an output buffer.
6. Then address 🟡 items in order of user-visibility (seek/stop position, status handling).
