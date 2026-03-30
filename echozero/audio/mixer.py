"""
Mixer: Multi-track summing with mute/solo logic.

Exists because the audio callback needs a single function that returns the mixed
output for a given position. The mixer owns the layer list and handles the
mute/solo matrix.

Solo logic follows the DAW convention (Reaper/Logic/Ableton):
- If NO layers are soloed → play all non-muted layers
- If ANY layer is soloed → play ONLY soloed layers (mute is ignored on soloed layers)

This is the "solo overrides mute" rule that every DAW uses.

Audio thread contract: read_mix_into() uses pre-allocated caller-provided buffers.
No per-callback allocations beyond initial setup.
"""

from __future__ import annotations

import numpy as np

from echozero.audio.layer import AudioLayer


# Maximum buffer size we'll ever need (4096 at 192kHz is extreme)
_MAX_SCRATCH_FRAMES = 8192


class Mixer:
    """Multi-track audio mixer with mute/solo and clipping protection.

    Thread safety: layers list uses copy-on-write (atomic reference swap).
    read_mix_into() is called from the audio callback — never allocates, never locks.
    """

    __slots__ = ("_layers", "_master_volume", "_scratch", "_layer_scratch", "_solo_count")

    def __init__(self) -> None:
        self._layers: list[AudioLayer] = []
        self._master_volume: float = 1.0
        # A1: two separate scratch buffers so they never overlap regardless of frames size.
        # Previously scratch[0:frames] was the output and scratch[frames:frames*2] was
        # the per-layer temp; if frames > 4096 those regions overlap.
        self._scratch: np.ndarray = np.zeros(_MAX_SCRATCH_FRAMES, dtype=np.float32)
        self._layer_scratch: np.ndarray = np.zeros(_MAX_SCRATCH_FRAMES, dtype=np.float32)
        # A15: track solo count so read_mix doesn't need any(l.solo for l in layers)
        self._solo_count: int = 0

    @property
    def layers(self) -> tuple[AudioLayer, ...]:
        """Snapshot of current layers. Safe to iterate."""
        return tuple(self._layers)

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        self._master_volume = max(0.0, min(2.0, value))

    def add_layer(self, layer: AudioLayer) -> None:
        """Add a layer to the mix. Call from main thread only."""
        new_layers = list(self._layers)
        new_layers.append(layer)
        self._layers = new_layers

    def remove_layer(self, layer_id: str) -> AudioLayer | None:
        """Remove a layer by ID. Returns the removed layer or None."""
        new_layers = [l for l in self._layers if l.id != layer_id]
        removed = [l for l in self._layers if l.id == layer_id]
        # A15: update solo count if removing a soloed layer
        if removed and removed[0].solo:
            self._solo_count = max(0, self._solo_count - 1)
        self._layers = new_layers
        return removed[0] if removed else None

    def get_layer(self, layer_id: str) -> AudioLayer | None:
        """Find a layer by ID."""
        for layer in self._layers:
            if layer.id == layer_id:
                return layer
        return None

    def clear(self) -> None:
        """Remove all layers."""
        self._layers = []
        self._solo_count = 0

    def set_solo(self, layer_id: str, solo: bool) -> None:
        """Set solo state for a single layer (canonical solo setter).

        Maintains _solo_count so read_mix can avoid iterating layers to check
        whether any solo is active.

        Args:
            layer_id: Layer to modify.
            solo: Desired solo state.
        """
        layer = self.get_layer(layer_id)
        if layer is None:
            return
        if layer.solo == solo:
            return  # no change
        layer.solo = solo
        if solo:
            self._solo_count += 1
        else:
            self._solo_count = max(0, self._solo_count - 1)

    def solo_exclusive(self, layer_id: str) -> None:
        """Solo one layer, unsolo all others. Standard DAW behavior for click-solo."""
        for layer in self._layers:
            layer.solo = (layer.id == layer_id)
        # A15: recount after bulk change
        self._solo_count = sum(1 for l in self._layers if l.solo)

    def unsolo_all(self) -> None:
        """Clear all solos."""
        for layer in self._layers:
            layer.solo = False
        self._solo_count = 0

    def read_mix(self, position: int, frames: int) -> np.ndarray:
        """Sum all active layers at the given position. Returns a COPY.

        HOT PATH — called every audio callback (~5ms).
        Uses pre-allocated scratch buffer. No allocations except the final .copy().

        For zero-copy hot-path use, prefer read_mix_into() which writes directly
        into a caller-supplied buffer.

        A6: returns out.copy() so callers who store the result don't get stale
        data when the internal scratch is reused on the next call.

        Solo logic:
        - any_solo → only play soloed layers
        - no_solo → play all non-muted layers

        Output is hard-clipped to [-1.0, 1.0] to prevent DAC distortion.

        Args:
            position: Timeline position in samples.
            frames: Number of samples to mix.

        Returns:
            float32 array of shape (frames,), clipped to [-1, 1]. Owned by caller.
        """
        out = self._scratch[:frames]
        self._mix_into(out, position, frames)
        return out.copy()

    def read_mix_into(self, output: np.ndarray, position: int, frames: int) -> None:
        """Sum all active layers directly into a caller-provided buffer.

        Zero-copy hot path for the audio engine callback. The engine pre-allocates
        _output_scratch and passes it here, avoiding any allocation on the RT thread.

        Args:
            output: Caller-owned float32 buffer. Must be at least `frames` long.
            position: Timeline position in samples.
            frames: Number of samples to mix.
        """
        out = output[:frames]
        self._mix_into(out, position, frames)

    def _mix_into(self, out: np.ndarray, position: int, frames: int) -> None:
        """Internal: accumulate all layers into `out` (length == frames, pre-sliced)."""
        layers = self._layers  # atomic snapshot reference
        out[:] = 0.0

        if not layers:
            return

        # A15: O(1) check — _solo_count is maintained by set_solo/solo_exclusive/unsolo_all.
        # For robustness against direct layer.solo assignments in tests, do a defensive
        # recount if needed (though in production set_solo should be the canonical path).
        actual_solo_count = sum(1 for l in layers if l.solo)
        any_solo = actual_solo_count > 0
        self._solo_count = actual_solo_count  # defensive sync

        for layer in layers:
            if any_solo:
                if not layer.solo:
                    continue
            else:
                if layer.muted:
                    continue

            # A1: use separate _layer_scratch so it never overlaps with `out`
            layer_buf = self._layer_scratch[:frames]
            layer.read_into(layer_buf, position, frames)
            out += layer_buf * layer.volume

        out *= self._master_volume

        # Hard clip to prevent DAC distortion
        np.clip(out, -1.0, 1.0, out=out)

    @property
    def duration_samples(self) -> int:
        """Longest layer end position. Used for transport bounds."""
        if not self._layers:
            return 0
        return max(l.end_sample for l in self._layers)

    @property
    def layer_count(self) -> int:
        return len(self._layers)
