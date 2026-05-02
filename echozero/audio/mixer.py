"""
Mixer: Multi-track summing with mute/solo logic.
Exists because the audio callback needs one place to sum active mono or stereo tracks.
Connects `AudioTrack` reads to engine-ready mixed buffers without UI semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echozero.audio.layer import AudioLayer, AudioTrack


# Leave headroom for host-chosen callback sizes when sounddevice runs with
# blocksize=0 on real hardware.
_MAX_SCRATCH_FRAMES = 32768
_MAX_OUTPUT_CHANNELS = 16
_GAIN_SMOOTHING_SECONDS = 0.02
_GAIN_SILENCE_EPSILON = 1e-5


def _resolve_output_bus_span(output_bus: str | None, output_channels: int) -> tuple[int, int]:
    """Resolve one zero-based output channel span from a layer output bus token."""

    if output_channels <= 1:
        return (0, 1)
    if output_bus is None:
        return (0, min(2, output_channels))

    token = output_bus.strip().lower()
    if not token.startswith("outputs_"):
        return (0, min(2, output_channels))
    parts = token.split("_")
    if len(parts) != 3 or (not parts[1].isdigit()) or (not parts[2].isdigit()):
        return (0, min(2, output_channels))

    start = max(1, int(parts[1])) - 1
    end = max(start + 1, int(parts[2])) - 1
    if start >= output_channels:
        return (-1, 0)
    resolved_end = min(end, output_channels - 1)
    width = max(0, resolved_end - start + 1)
    return (start, width)


@dataclass(slots=True)
class _GainEnvelope:
    current: float
    target: float
    remaining_samples: int


class Mixer:
    """Multi-track audio mixer with mute/solo and clipping protection.

    Thread safety: layers list uses copy-on-write (atomic reference swap).
    read_mix_into() is called from the audio callback — never allocates, never locks.
    """

    __slots__ = (
        "_layers",
        "_master_volume",
        "_scratch",
        "_layer_scratch",
        "_scratch_multichannel",
        "_layer_scratch_multichannel",
        "_solo_count",
        "_gain_envelopes",
        "_gain_ramp",
        "_gain_ramp_index",
        "_gain_smoothing_samples",
    )

    def __init__(self) -> None:
        self._layers: list[AudioLayer] = []
        self._master_volume: float = 1.0
        # A1: two separate scratch buffers so they never overlap regardless of frames size.
        # Previously scratch[0:frames] was the output and scratch[frames:frames*2] was
        # the per-layer temp; if frames > 4096 those regions overlap.
        self._scratch: np.ndarray = np.zeros(_MAX_SCRATCH_FRAMES, dtype=np.float32)
        self._layer_scratch: np.ndarray = np.zeros(_MAX_SCRATCH_FRAMES, dtype=np.float32)
        self._scratch_multichannel: np.ndarray = np.zeros(
            (_MAX_SCRATCH_FRAMES, _MAX_OUTPUT_CHANNELS),
            dtype=np.float32,
        )
        self._layer_scratch_multichannel: np.ndarray = np.zeros(
            (_MAX_SCRATCH_FRAMES, _MAX_OUTPUT_CHANNELS),
            dtype=np.float32,
        )
        # A15: track solo count so read_mix doesn't need any(l.solo for l in layers)
        self._solo_count: int = 0
        self._gain_envelopes: dict[str, _GainEnvelope] = {}
        self._gain_ramp: np.ndarray = np.zeros(_MAX_SCRATCH_FRAMES, dtype=np.float32)
        self._gain_ramp_index: np.ndarray = np.arange(_MAX_SCRATCH_FRAMES, dtype=np.float32)
        self._gain_smoothing_samples = max(8, int(round(44100 * _GAIN_SMOOTHING_SECONDS)))

    @property
    def tracks(self) -> tuple[AudioTrack, ...]:
        """Snapshot of current tracks. Safe to iterate."""
        return tuple(self._layers)

    @property
    def layers(self) -> tuple[AudioTrack, ...]:
        """Compatibility alias for callers that still say `layers`."""
        return self.tracks

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        self._master_volume = max(0.0, min(2.0, value))

    def add_track(self, track: AudioTrack) -> None:
        """Add one track to the mix. Call from the main thread only."""
        new_layers = list(self._layers)
        new_layers.append(track)
        self._layers = new_layers

    def add_layer(self, layer: AudioTrack) -> None:
        """Compatibility alias for callers that still say `add_layer`."""
        self.add_track(layer)

    def replace_tracks(self, tracks: list[AudioTrack]) -> None:
        """Atomically replace the current track set."""

        self._layers = list(tracks)
        self._solo_count = sum(1 for track in self._layers if track.solo)
        active_ids = {str(track.id) for track in self._layers}
        stale_ids = [track_id for track_id in self._gain_envelopes if track_id not in active_ids]
        for stale_id in stale_ids:
            self._gain_envelopes.pop(stale_id, None)

    def apply_track_mix_updates(
        self,
        updates: dict[str, tuple[bool, float, str | None]],
    ) -> tuple[bool, bool]:
        """Apply muted/volume/output_bus updates in place.

        Returns:
            (applied_changes, requires_declick)
        """

        if not updates:
            return (False, False)
        applied_change = False
        requires_declick = False
        for layer in self._layers:
            desired = updates.get(str(layer.id))
            if desired is None:
                continue
            muted, volume, output_bus = desired
            if bool(layer.muted) != bool(muted):
                layer.muted = bool(muted)
                applied_change = True
                requires_declick = True
            if abs(float(layer.volume) - float(volume)) > 1e-6:
                layer.volume = float(volume)
                applied_change = True
            if layer.output_bus != output_bus:
                layer.output_bus = output_bus
                applied_change = True
                requires_declick = True
        return (applied_change, requires_declick)

    def configure_gain_smoothing(
        self,
        *,
        sample_rate: int,
        seconds: float = _GAIN_SMOOTHING_SECONDS,
    ) -> None:
        """Tune gain smoothing window to the active output sample rate."""

        self._gain_smoothing_samples = max(8, int(round(float(sample_rate) * float(seconds))))

    def remove_track(self, track_id: str) -> AudioTrack | None:
        """Remove one track by ID. Returns the removed track or None."""
        new_layers = [track for track in self._layers if track.id != track_id]
        removed = [track for track in self._layers if track.id == track_id]
        # A15: update solo count if removing a soloed layer
        if removed and removed[0].solo:
            self._solo_count = max(0, self._solo_count - 1)
        self._layers = new_layers
        self._gain_envelopes.pop(str(track_id), None)
        return removed[0] if removed else None

    def remove_layer(self, layer_id: str) -> AudioTrack | None:
        """Compatibility alias for callers that still say `remove_layer`."""
        return self.remove_track(layer_id)

    def get_track(self, track_id: str) -> AudioTrack | None:
        """Find one track by ID."""
        for layer in self._layers:
            if layer.id == track_id:
                return layer
        return None

    def get_layer(self, layer_id: str) -> AudioTrack | None:
        """Compatibility alias for callers that still say `get_layer`."""
        return self.get_track(layer_id)

    def clear_tracks(self) -> None:
        """Remove all tracks."""
        self._layers = []
        self._solo_count = 0
        self._gain_envelopes.clear()

    def clear(self) -> None:
        """Compatibility alias for callers that still say `clear`."""
        self.clear_tracks()

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

    def read_mix(self, position: int, frames: int, *, channels: int = 1) -> np.ndarray:
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
            float32 array of shape `(frames,)` for mono or `(frames, channels)` for
            multi-channel output, clipped to [-1, 1]. Owned by caller.
        """
        if channels <= 1:
            out = self._scratch[:frames]
        else:
            if channels > self._scratch_multichannel.shape[1]:
                raise ValueError(
                    f"channels ({channels}) > supported output channels "
                    f"({self._scratch_multichannel.shape[1]})"
                )
            out = self._scratch_multichannel[:frames, :channels]
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
            layer_id = str(layer.id)
            if any_solo:
                audible = bool(layer.solo)
            else:
                audible = not bool(layer.muted)
            target_gain = float(layer.volume) if audible else 0.0
            envelope = self._gain_envelopes.get(layer_id)
            if envelope is None:
                envelope = _GainEnvelope(
                    current=float(target_gain),
                    target=float(target_gain),
                    remaining_samples=0,
                )
                self._gain_envelopes[layer_id] = envelope
            elif abs(float(target_gain) - float(envelope.target)) > _GAIN_SILENCE_EPSILON:
                envelope.target = float(target_gain)
                envelope.remaining_samples = max(1, int(self._gain_smoothing_samples))
            current_gain = float(envelope.current)
            if (
                abs(current_gain) <= _GAIN_SILENCE_EPSILON
                and abs(target_gain) <= _GAIN_SILENCE_EPSILON
                and envelope.remaining_samples <= 0
            ):
                envelope.current = 0.0
                envelope.target = 0.0
                continue

            if out.ndim == 1:
                target_start, target_width = _resolve_output_bus_span(layer.output_bus, 1)
                if target_start != 0 or target_width <= 0:
                    continue
                layer_buf = self._layer_scratch[:frames]
                # A1: use separate layer scratch so it never overlaps with `out`
                layer.read_into(layer_buf, position, frames)
                self._apply_gain_ramp(
                    layer_buf,
                    envelope=envelope,
                    frames=frames,
                )
                out += layer_buf
                continue

            output_channels = out.shape[1]
            target_start, target_width = _resolve_output_bus_span(
                layer.output_bus,
                output_channels,
            )
            if target_width <= 0:
                continue
            if target_width > self._layer_scratch_multichannel.shape[1]:
                raise ValueError(
                    f"output bus width ({target_width}) exceeds scratch channels "
                    f"({self._layer_scratch_multichannel.shape[1]})"
                )
            layer_buf = self._layer_scratch_multichannel[:frames, :target_width]
            # A1: use separate layer scratch so it never overlaps with `out`
            layer.read_into(layer_buf, position, frames)
            self._apply_gain_ramp(
                layer_buf,
                envelope=envelope,
                frames=frames,
            )
            out[:, target_start:target_start + target_width] += layer_buf

        out *= self._master_volume

        # Hard clip to prevent DAC distortion
        np.clip(out, -1.0, 1.0, out=out)

    def _apply_gain_ramp(
        self,
        buffer: np.ndarray,
        *,
        envelope: _GainEnvelope,
        frames: int,
    ) -> None:
        if frames <= 0:
            return
        current_gain = float(envelope.current)
        target_gain = float(envelope.target)
        remaining = max(0, int(envelope.remaining_samples))
        if abs(target_gain - current_gain) <= _GAIN_SILENCE_EPSILON:
            buffer[:frames] *= np.float32(target_gain)
            envelope.current = float(target_gain)
            envelope.remaining_samples = 0
            return
        if remaining <= 0:
            remaining = max(1, int(self._gain_smoothing_samples))
        ramp_samples = max(1, min(int(frames), remaining))
        ramp = self._gain_ramp[:frames]
        if ramp_samples <= 1:
            ramp[:frames] = np.float32(target_gain)
            end_gain = float(target_gain)
            remaining = 0
        else:
            phase = (self._gain_ramp_index[:ramp_samples] + 1.0) / float(remaining)
            phase = np.minimum(phase, np.float32(1.0))
            ramp[:ramp_samples] = np.float32(current_gain) + (
                np.float32(target_gain - current_gain) * phase
            )
            end_gain = float(ramp[ramp_samples - 1])
            remaining = max(0, remaining - ramp_samples)
            if ramp_samples < frames:
                if remaining <= 0:
                    ramp[ramp_samples:frames] = np.float32(target_gain)
                    end_gain = float(target_gain)
                else:
                    ramp[ramp_samples:frames] = np.float32(end_gain)
        if buffer.ndim == 1:
            np.multiply(buffer[:frames], ramp[:frames], out=buffer[:frames])
        else:
            np.multiply(buffer[:frames], ramp[:frames, None], out=buffer[:frames])
        envelope.current = float(end_gain)
        envelope.target = float(target_gain)
        envelope.remaining_samples = int(remaining)

    @property
    def duration_samples(self) -> int:
        """Longest track end position. Used for transport bounds."""
        if not self._layers:
            return 0
        return max(l.end_sample for l in self._layers)

    @property
    def track_count(self) -> int:
        return len(self._layers)

    @property
    def layer_count(self) -> int:
        return self.track_count
