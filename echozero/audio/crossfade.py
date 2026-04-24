"""
Crossfade: Pre-computed fade curves for loop boundaries and seeks.

Exists because hard cuts at loop points or seek positions cause audible clicks.
A short crossfade (2-5ms) eliminates them. Pre-computing the fade curve avoids
per-callback math.

Standard practice from DAW engineering:
- Reaper uses ~5ms equal-power crossfade at loop points
- Ableton uses ~4ms linear crossfade
- Logic uses configurable crossfade (1-10ms)

We use equal-power (sine/cosine) because it maintains perceived loudness
through the transition. Linear crossfade dips ~3dB at the midpoint.

The crossfade buffer is pre-allocated at engine init and reused every loop wrap.
"""

from __future__ import annotations

import numpy as np


# Default crossfade duration in samples at 44.1kHz ≈ 4ms
DEFAULT_CROSSFADE_SAMPLES = 176


def build_equal_power_curves(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute equal-power fade-out and fade-in curves.

    Equal-power: fade_out = cos(t * π/2), fade_in = sin(t * π/2)
    where t goes from 0 to 1 over the crossfade length.

    At the midpoint: cos(π/4)² + sin(π/4)² = 1.0 — constant power.
    Linear would give 0.5 + 0.5 = 1.0 in amplitude but only 0.5 in power (−3dB dip).

    Args:
        length: Number of samples in the crossfade region.

    Returns:
        (fade_out, fade_in) — both float32 arrays of shape (length,).
    """
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    fade_out = np.cos(t * (np.pi / 2)).astype(np.float32)
    fade_in = np.sin(t * (np.pi / 2)).astype(np.float32)
    return fade_out, fade_in


class CrossfadeBuffer:
    """Pre-allocated crossfade infrastructure for the audio callback.

    Holds:
    - Equal-power fade curves (computed once)

    The engine calls apply() when a loop wrap is detected. The crossfade
    blends the tail of the outgoing region with the head of the incoming region
    in-place on the output buffer.

    Usage:
        xfade = CrossfadeBuffer(crossfade_samples=176)

        # In the audio callback, when loop wrap detected:
        xfade.apply(out_buffer, mixer, tail_pos, head_pos, frames)

    A14: removed vestigial _tail_buf and _head_buf scratch arrays — they were
    pre-allocated but never used (the engine builds tail/head slices directly
    from its own pre-allocated _output_scratch).
    """

    __slots__ = ("_length", "_fade_out", "_fade_in")

    def __init__(self, crossfade_samples: int = DEFAULT_CROSSFADE_SAMPLES) -> None:
        self._length = crossfade_samples
        self._fade_out, self._fade_in = build_equal_power_curves(crossfade_samples)

    @property
    def length(self) -> int:
        """Crossfade length in samples."""
        return self._length

    def apply(
        self,
        output: np.ndarray,
        tail_audio: np.ndarray,
        head_audio: np.ndarray,
        xfade_start: int,
        xfade_len: int,
    ) -> None:
        """Blend outgoing tail with incoming head in the output buffer.

        Called from the audio callback when a loop wrap is detected within
        the current buffer.

        The crossfade region in the output buffer (at xfade_start, length xfade_len)
        is replaced with: tail * fade_out + head * fade_in.

        Args:
            output: The output buffer being written (modified in-place).
            tail_audio: Audio from the end of the loop (outgoing). Shape (xfade_len,).
            head_audio: Audio from the start of the loop (incoming). Shape (xfade_len,).
            xfade_start: Index in output where crossfade begins.
            xfade_len: Actual crossfade length (may be shorter than self._length
                       if the wrap happens near the end of the buffer).
        """
        if xfade_len <= 0:
            return

        # Use pre-computed curves, truncated if needed
        fade_out = self._fade_out[:xfade_len]
        fade_in = self._fade_in[:xfade_len]

        region = output[xfade_start:xfade_start + xfade_len]
        if region.ndim > 1:
            fade_out = fade_out[:, None]
            fade_in = fade_in[:, None]
        region[:] = tail_audio[:xfade_len] * fade_out + head_audio[:xfade_len] * fade_in
