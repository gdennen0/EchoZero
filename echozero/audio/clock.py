"""
Clock: Sample-accurate master clock driven by the audio callback.

Exists because everything in a DAW needs a single source of truth for time.
The audio callback advances the clock. Subscribers (UI playhead, event triggering,
OSC output, MIDI sync) read position or receive callbacks.

Inspired by: Reaper's master timeline, Ableton's global transport clock,
Logic's SPL (Sample Position Locator).

The clock is THE arbiter of time. Nothing else keeps its own counter.

Thread safety model (lock-free on audio thread):
- _position is a Python int — reads/writes are atomic under GIL.
- _loop_snapshot is an immutable reference swapped atomically.
- advance() never acquires a lock. Main thread methods use a lock for
  compound operations (seek + update stop position, etc.) but the audio
  thread path is completely lock-free.
- Subscribers list uses copy-on-write: safe to add/remove while playing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol


class ClockSubscriber(Protocol):
    """Anything that wants to know the current playback position.

    Called from the audio callback thread — implementations MUST be fast
    (no allocations, no locks, no I/O). If you need to do heavy work,
    copy the position atomically and process it on another thread.
    """

    def on_clock_tick(self, position_samples: int, sample_rate: int) -> None:
        """Called every audio callback with the current position."""
        ...


@dataclass(frozen=True)
class LoopRegion:
    """Defines a loop range in samples. Immutable for atomic swap on audio thread."""
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Loop start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Loop end must be > start, got start={self.start} end={self.end}")


@dataclass(frozen=True)
class _LoopSnapshot:
    """Immutable snapshot read atomically by the audio thread."""
    enabled: bool
    region: LoopRegion | None


class Clock:
    """Sample-accurate master clock. Lock-free on the audio thread.

    The audio thread calls advance() which reads _position and _loop_snapshot
    without any lock. The main thread uses _lock for compound operations but
    never contends with the audio thread.

    Usage:
        clock = Clock(sample_rate=44100)
        clock.add_subscriber(my_playhead)
        # Audio callback calls clock.advance(frames) every buffer
    """

    __slots__ = (
        "_position", "_sample_rate", "_subscribers",
        "_loop_snapshot", "_lock",
    )

    def __init__(self, sample_rate: int = 44100) -> None:
        self._position: int = 0
        self._sample_rate: int = sample_rate
        self._subscribers: list[ClockSubscriber] = []
        self._loop_snapshot: _LoopSnapshot = _LoopSnapshot(enabled=False, region=None)
        self._lock = threading.Lock()  # main thread only, never held during advance()

    @property
    def position(self) -> int:
        """Current position in samples. Safe to read from any thread."""
        return self._position

    @property
    def position_seconds(self) -> float:
        """Current position in seconds."""
        return self._position / self._sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value

    def seek(self, position_samples: int) -> None:
        """Jump to a position. Safe from any thread."""
        self._position = max(0, position_samples)

    def seek_seconds(self, seconds: float) -> None:
        """Jump to a position in seconds."""
        self.seek(int(seconds * self._sample_rate))

    def advance(self, frames: int) -> int:
        """Advance the clock by `frames` samples. Called from audio callback.

        Returns the position BEFORE the advance (the position for this buffer).
        Handles loop wrapping if active.

        LOCK-FREE: reads _position (atomic int) and _loop_snapshot (atomic ref).
        No lock, no allocation, no I/O.
        """
        read_pos = self._position
        new_pos = read_pos + frames

        # Loop wrapping — reads immutable snapshot atomically
        snap = self._loop_snapshot
        if snap.enabled and snap.region is not None:
            if new_pos >= snap.region.end:
                overshoot = new_pos - snap.region.end
                loop_len = snap.region.end - snap.region.start
                new_pos = snap.region.start + (overshoot % loop_len)

        self._position = new_pos

        # Notify subscribers (snapshot reference, safe if list swapped mid-iteration)
        subs = self._subscribers
        for sub in subs:
            sub.on_clock_tick(read_pos, self._sample_rate)

        return read_pos

    def set_loop(self, start: int, end: int) -> None:
        """Set the loop region. Validates immediately."""
        region = LoopRegion(start, end)
        with self._lock:
            old = self._loop_snapshot
            self._loop_snapshot = _LoopSnapshot(enabled=old.enabled, region=region)

    def set_loop_seconds(self, start: float, end: float) -> None:
        """Set loop region in seconds."""
        self.set_loop(
            int(start * self._sample_rate),
            int(end * self._sample_rate),
        )

    @property
    def loop_enabled(self) -> bool:
        return self._loop_snapshot.enabled

    @loop_enabled.setter
    def loop_enabled(self, value: bool) -> None:
        with self._lock:
            old = self._loop_snapshot
            self._loop_snapshot = _LoopSnapshot(enabled=value, region=old.region)

    @property
    def loop_region(self) -> LoopRegion | None:
        return self._loop_snapshot.region

    def add_subscriber(self, sub: ClockSubscriber) -> None:
        """Add a clock subscriber. Safe to call while playing (copy-on-write)."""
        with self._lock:
            if sub not in self._subscribers:
                new_subs = list(self._subscribers)
                new_subs.append(sub)
                self._subscribers = new_subs

    def remove_subscriber(self, sub: ClockSubscriber) -> None:
        """Remove a subscriber. Safe to call while playing (copy-on-write)."""
        with self._lock:
            new_subs = [s for s in self._subscribers if s is not sub]
            self._subscribers = new_subs

    def reset(self) -> None:
        """Reset position to zero."""
        self._position = 0
