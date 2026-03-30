"""
Mixer: Multi-track summing with mute/solo logic.

Exists because the audio callback needs a single function that returns the mixed
output for a given position. The mixer owns the layer list and handles the
mute/solo matrix.

Solo logic follows the DAW convention (Reaper/Logic/Ableton):
- If NO layers are soloed → play all non-muted layers
- If ANY layer is soloed → play ONLY soloed layers (mute is ignored on soloed layers)

This is the "solo overrides mute" rule that every DAW uses.

Audio thread contract: read_mix() uses pre-allocated scratch buffers.
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
    read_mix() is called from the audio callback — never allocates, never locks.
    """

    __slots__ = ("_layers", "_master_volume", "_scratch")

    def __init__(self) -> None:
        self._layers: list[AudioLayer] = []
        self._master_volume: float = 1.0
        self._scratch: np.ndarray = np.zeros(_MAX_SCRATCH_FRAMES, dtype=np.float32)

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

    def solo_exclusive(self, layer_id: str) -> None:
        """Solo one layer, unsolo all others. Standard DAW behavior for click-solo."""
        for layer in self._layers:
            layer.solo = (layer.id == layer_id)

    def unsolo_all(self) -> None:
        """Clear all solos."""
        for layer in self._layers:
            layer.solo = False

    def read_mix(self, position: int, frames: int) -> np.ndarray:
        """Sum all active layers at the given position.

        HOT PATH — called every audio callback (~5ms).
        Uses pre-allocated scratch buffer. No allocations. No locks.

        Solo logic:
        - any_solo → only play soloed layers
        - no_solo → play all non-muted layers

        Output is hard-clipped to [-1.0, 1.0] to prevent DAC distortion.

        Args:
            position: Timeline position in samples.
            frames: Number of samples to mix.

        Returns:
            float32 array view of shape (frames,). Clipped to [-1, 1].
        """
        layers = self._layers  # atomic snapshot reference
        scratch = self._scratch

        # Output accumulator (view into pre-allocated buffer)
        # Note: we return a view, so caller must consume before next read_mix call
        out = scratch[:frames]
        out[:] = 0.0

        if not layers:
            return out

        any_solo = any(l.solo for l in layers)

        for layer in layers:
            if any_solo:
                if not layer.solo:
                    continue
            else:
                if layer.muted:
                    continue

            # Read into scratch area past the output region
            layer_buf = scratch[frames:frames + frames]
            layer.read_into(layer_buf, position, frames)
            out += layer_buf * layer.volume

        out *= self._master_volume

        # Hard clip to prevent DAC distortion
        np.clip(out, -1.0, 1.0, out=out)

        return out

    @property
    def duration_samples(self) -> int:
        """Longest layer end position. Used for transport bounds."""
        if not self._layers:
            return 0
        return max(l.end_sample for l in self._layers)

    @property
    def layer_count(self) -> int:
        return len(self._layers)
