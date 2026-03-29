"""
Mixer: Multi-track summing with mute/solo logic.

Exists because the audio callback needs a single function that returns the mixed
output for a given position. The mixer owns the layer list and handles the
mute/solo matrix.

Solo logic follows the DAW convention (Reaper/Logic/Ableton):
- If NO layers are soloed → play all non-muted layers
- If ANY layer is soloed → play ONLY soloed layers (mute is ignored on soloed layers)

This is the "solo overrides mute" rule that every DAW uses.
"""

from __future__ import annotations

import numpy as np

from echozero.audio.layer import AudioLayer


class Mixer:
    """Multi-track audio mixer with mute/solo.

    Thread safety: layers list is modified only from the main thread (add/remove).
    read_mix() is called from the audio callback thread. We use a snapshot pattern —
    the callback reads a reference to the list, which is safe because Python list
    reads are atomic (GIL). Modifications copy-on-write to avoid mutation during iteration.
    """

    __slots__ = ("_layers", "_master_volume")

    def __init__(self) -> None:
        self._layers: list[AudioLayer] = []
        self._master_volume: float = 1.0

    @property
    def layers(self) -> tuple[AudioLayer, ...]:
        """Snapshot of current layers. Safe to iterate."""
        return tuple(self._layers)

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        self._master_volume = max(0.0, min(1.0, value))

    def add_layer(self, layer: AudioLayer) -> None:
        """Add a layer to the mix. Call from main thread only."""
        # Copy-on-write: create new list so callback sees consistent state
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

        This is the HOT PATH — called every audio callback (~5ms).
        Must be fast: no allocations beyond the output buffer, no locks, no I/O.

        Solo logic:
        - any_solo → only play soloed layers
        - no_solo → play all non-muted layers

        Args:
            position: Timeline position in samples.
            frames: Number of samples to mix.

        Returns:
            float32 array of shape (frames,). Mixed and master-gained.
        """
        layers = self._layers  # snapshot reference
        if not layers:
            return np.zeros(frames, dtype=np.float32)

        any_solo = any(l.solo for l in layers)
        out = np.zeros(frames, dtype=np.float32)

        for layer in layers:
            # Solo logic
            if any_solo:
                if not layer.solo:
                    continue
            else:
                if layer.muted:
                    continue

            chunk = layer.read_samples(position, frames)
            out += chunk * layer.volume

        out *= self._master_volume
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
