"""
Audio Negate Block Processor

Negates/cancels audio at event time regions using three modes:
- silence: Zero out event regions with crossfade
- attenuate: Reduce volume at event regions by configurable dB
- subtract: Build a negative track from subtract_audio at event regions,
            then subtract it from the source. Outputs both the negative
            track and the subtracted result.

All modes apply configurable crossfades at region boundaries to avoid clicks.
"""

import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import AudioDataItem, EventDataItem, DataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


# Available negation modes with descriptions
NEGATE_MODES = {
    "silence": {
        "name": "Silence",
        "description": "Zero out audio at event time regions (fade to/from silence)",
        "needs_subtract_audio": False,
    },
    "attenuate": {
        "name": "Attenuate",
        "description": "Reduce volume at event time regions by a configurable dB amount",
        "needs_subtract_audio": False,
    },
    "subtract": {
        "name": "Subtract",
        "description": (
            "Extract subtract_audio samples at event regions into a negative track, "
            "then subtract that track from the source audio. "
            "Outputs both the negative track and the subtracted result. "
            "Requires the 'subtract_audio' input to be connected."
        ),
        "needs_subtract_audio": True,
    },
}


def _build_fade_envelope(region_length: int, fade_samples: int) -> np.ndarray:
    """
    Build an envelope array for a region: fade-in from 0 -> 1, hold at 1, fade-out 1 -> 0.

    The envelope is used to smoothly transition into and out of the negation region,
    avoiding audible clicks at boundaries.

    Args:
        region_length: Total number of samples in the region
        fade_samples: Number of samples for each fade (in and out)

    Returns:
        1D numpy array of shape (region_length,) with values in [0, 1]
    """
    if region_length <= 0:
        return np.array([], dtype=np.float64)

    envelope = np.ones(region_length, dtype=np.float64)

    # Clamp fade_samples so the two fades don't exceed the region
    fade_samples = min(fade_samples, region_length // 2)

    if fade_samples > 0:
        # Fade in: 0 -> 1 (cosine curve for smooth transition)
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
        envelope[:fade_samples] = fade_in

        # Fade out: 1 -> 0
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))
        envelope[-fade_samples:] = fade_out

    return envelope


def _normalise_audio_shape(audio: np.ndarray) -> np.ndarray:
    """
    Normalise audio to 2D array with shape (channels, samples).

    Handles:
    - 1D mono: (samples,) -> (1, samples)
    - 2D channels-first: (channels, samples) -> kept
    - 2D samples-first: (samples, channels) -> transposed when channels <= 16

    Returns:
        2D array of shape (channels, samples)
    """
    if audio.ndim == 1:
        return audio.reshape(1, -1)
    if audio.ndim == 2:
        # Heuristic: if axis-0 is much larger than axis-1, assume (samples, channels)
        if audio.shape[0] > audio.shape[1] and audio.shape[1] <= 16:
            return audio.T
        return audio
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def _align_subtract_audio(
    target: np.ndarray,
    subtract: np.ndarray,
    target_sr: int,
    subtract_sr: Optional[int] = None,
) -> np.ndarray:
    """
    Align subtract audio to match the target audio dimensions.

    Handles:
    - Shape normalisation to 2D (channels, samples)
    - Sample rate resampling if needed
    - Channel count matching (upmix/downmix)
    - Length matching (pad with zeros or truncate)

    Args:
        target: Target audio, already normalised to (channels, samples)
        subtract: Subtract audio, any supported shape
        target_sr: Target sample rate
        subtract_sr: Subtract sample rate (if None, assumed same as target)

    Returns:
        Subtract audio as (channels, samples) matching target dimensions
    """
    sub = _normalise_audio_shape(subtract).astype(np.float64)
    target_channels, target_samples = target.shape

    # Resample if sample rates differ
    if subtract_sr and subtract_sr != target_sr:
        try:
            import librosa
            resampled_channels = []
            for ch in range(sub.shape[0]):
                resampled_channels.append(
                    librosa.resample(sub[ch], orig_sr=subtract_sr, target_sr=target_sr)
                )
            sub = np.array(resampled_channels, dtype=np.float64)
            Log.info(
                f"apply_negate: Resampled subtract audio from {subtract_sr}Hz to {target_sr}Hz"
            )
        except Exception as e:
            Log.warning(f"apply_negate: Resample failed ({e}), proceeding without resample")

    # Match channel count
    if sub.shape[0] != target_channels:
        if sub.shape[0] == 1:
            # Mono subtract -> broadcast to all target channels
            sub = np.repeat(sub, target_channels, axis=0)
        elif target_channels == 1:
            # Multi-channel subtract -> mix down to mono
            sub = np.mean(sub, axis=0, keepdims=True)
        else:
            # Different channel counts -> take min and pad
            min_ch = min(sub.shape[0], target_channels)
            padded = np.zeros((target_channels, sub.shape[1]), dtype=np.float64)
            padded[:min_ch] = sub[:min_ch]
            sub = padded

    # Match length (pad with zeros or truncate)
    if sub.shape[1] < target_samples:
        padded = np.zeros((sub.shape[0], target_samples), dtype=np.float64)
        padded[:, :sub.shape[1]] = sub
        sub = padded
    elif sub.shape[1] > target_samples:
        sub = sub[:, :target_samples]

    return sub


def apply_negate(
    audio_data: np.ndarray,
    sample_rate: int,
    events: list,
    mode: str,
    fade_ms: float = 10.0,
    attenuation_db: float = -20.0,
) -> np.ndarray:
    """
    Apply silence or attenuate negation at event time regions using
    STFT-based spectral processing for clean, artifact-free results.

    Builds a per-sample gain mask (handling overlapping events correctly
    via np.minimum), then applies it in the frequency domain so that
    STFT windowing produces natural transitions at region boundaries.

    For subtract mode, use build_negative_track() instead.

    Args:
        audio_data: Target audio as numpy array (1D mono or 2D multi-channel)
        sample_rate: Sample rate in Hz
        events: List of Event objects with .time and .duration attributes
        mode: Negation mode ("silence" or "attenuate")
        fade_ms: Crossfade duration at region edges in milliseconds
        attenuation_db: Volume reduction in dB (for attenuate mode, should be negative)

    Returns:
        Negated audio data as numpy array (same shape as input)
    """
    orig_shape = audio_data.shape
    is_mono_1d = audio_data.ndim == 1

    audio_2d = _normalise_audio_shape(audio_data.copy()).astype(np.float64)
    n_channels, total_samples = audio_2d.shape

    fade_samples = max(0, int(fade_ms / 1000.0 * sample_rate))
    attenuation_gain = 10.0 ** (attenuation_db / 20.0) if mode == "attenuate" else 0.0

    # ---- Step 1: Build per-sample gain mask --------------------------------
    # Using np.minimum ensures overlapping events don't compound
    # (e.g. two overlapping -20 dB events stay at -20 dB, not -40 dB).
    gain_mask = np.ones(total_samples, dtype=np.float64)

    for event in events:
        start = max(0, int(event.time * sample_rate))
        end = min(total_samples, int((event.time + event.duration) * sample_rate))

        if start >= end or start >= total_samples:
            continue

        region_length = end - start
        envelope = _build_fade_envelope(region_length, fade_samples)

        if mode == "silence":
            event_gain = 1.0 - envelope  # fades to 0 at centre
        elif mode == "attenuate":
            event_gain = 1.0 - envelope * (1.0 - attenuation_gain)
        else:
            continue

        gain_mask[start:end] = np.minimum(gain_mask[start:end], event_gain)

    # ---- Step 2: Apply gain via STFT for spectral-quality processing -------
    n_fft = 2048
    hop_length = n_fft // 4
    window = np.hanning(n_fft)

    result = np.zeros_like(audio_2d)

    for ch in range(n_channels):
        # Forward STFT
        padded = np.pad(audio_2d[ch], (n_fft // 2, n_fft // 2), mode='reflect')
        frames_view = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
        stft = np.fft.rfft(frames_view * window, axis=-1)

        mag = np.abs(stft)
        phase = np.angle(stft)

        # Compute per-frame gain from the gain mask (sample at frame centre)
        n_frames = stft.shape[0]
        frame_centres = np.arange(n_frames) * hop_length
        frame_centres = np.clip(frame_centres, 0, total_samples - 1)
        frame_gains = gain_mask[frame_centres]

        # Apply gain to magnitude spectrum, preserve phase
        result_mag = mag * frame_gains[:, np.newaxis]

        # Inverse STFT via overlap-add
        result_stft = result_mag * np.exp(1j * phase)
        irfft_frames = np.fft.irfft(result_stft, n=n_fft, axis=-1)
        irfft_frames *= window  # synthesis window

        out_ch = np.zeros(total_samples + n_fft, dtype=np.float64)
        for i, frame in enumerate(irfft_frames):
            start_pos = i * hop_length
            end_pos = start_pos + n_fft
            if end_pos <= len(out_ch):
                out_ch[start_pos:end_pos] += frame

        result[ch] = out_ch[n_fft // 2 : n_fft // 2 + total_samples]

    # Normalise overlap-add gain
    ola_gain = np.sum(window ** 2) / hop_length
    if ola_gain > 0:
        result /= ola_gain

    np.clip(result, -1.0, 1.0, out=result)

    if is_mono_1d:
        return result[0]
    return result.reshape(orig_shape) if result.shape != orig_shape else result


def _build_onset_emphasis_envelope(
    region_length: int,
    onset_emphasis: float,
    sample_rate: int,
    decay_ms: float = 30.0,
) -> np.ndarray:
    """
    Build an onset emphasis envelope that starts at ``onset_emphasis`` and
    decays exponentially to 1.0 over the first ``decay_ms`` milliseconds.

    This front-loads the subtraction so transient/attack content is removed
    more aggressively than the sustained portion of the event.

    Args:
        region_length: Total samples in the event region.
        onset_emphasis: Peak multiplier at the very start (>= 1.0).
        sample_rate: Audio sample rate in Hz.
        decay_ms: Time constant for the exponential decay in milliseconds.

    Returns:
        1D array of shape (region_length,) with values decaying from
        ``onset_emphasis`` down to 1.0.
    """
    if onset_emphasis <= 1.0 or region_length <= 0:
        return np.ones(region_length, dtype=np.float64)

    decay_samples = max(1, int(decay_ms / 1000.0 * sample_rate))
    t = np.arange(region_length, dtype=np.float64)
    # Exponential decay from (onset_emphasis - 1) down to 0, offset by +1
    envelope = 1.0 + (onset_emphasis - 1.0) * np.exp(-t / decay_samples)
    return envelope


def build_negative_track(
    target_audio: np.ndarray,
    subtract_audio: np.ndarray,
    events: list,
    sample_rate: int,
    fade_ms: float = 10.0,
    subtract_sr: Optional[int] = None,
    subtract_gain: float = 1.0,
    onset_emphasis: float = 1.0,
) -> tuple:
    """
    Build a negative track and a subtracted result from event regions.

    Steps:
    1. Normalise and align subtract audio to target dimensions.
    2. Create a full-length zeros array (the "negative track").
    3. For each event, copy the subtract_audio samples at that time
       region into the negative track (with crossfade and onset emphasis).
    4. Spectrally subtract the negative track from the target audio,
       amplified by ``subtract_gain``.

    Args:
        target_audio: Source audio array (1D or 2D)
        subtract_audio: Audio to extract event regions from (1D or 2D)
        events: List of Event objects with .time and .duration
        sample_rate: Target sample rate
        fade_ms: Crossfade duration at region edges in milliseconds
        subtract_sr: Sample rate of subtract_audio (None = same as target)
        subtract_gain: Multiplier for spectral subtraction strength (1.0 = normal)
        onset_emphasis: Extra emphasis at onset/transient of each event (1.0 = none)

    Returns:
        Tuple of (negative_track, subtracted_result) as numpy arrays,
        both in the same shape as the original target_audio.
    """
    orig_shape = target_audio.shape
    is_mono_1d = target_audio.ndim == 1

    # Normalise both to 2D (channels, samples)
    target_2d = _normalise_audio_shape(target_audio).astype(np.float64)
    aligned_sub = _align_subtract_audio(target_2d, subtract_audio, sample_rate, subtract_sr)

    n_channels, total_samples = target_2d.shape
    fade_samples = max(0, int(fade_ms / 1000.0 * sample_rate))

    Log.info(
        f"build_negative_track: target shape={target_2d.shape}, "
        f"subtract shape={aligned_sub.shape}, "
        f"events={len(events)}, fade_ms={fade_ms}"
    )

    # Step 1-2: Build the negative track (zeros with subtract samples at event regions)
    negative = np.zeros_like(target_2d)
    regions_filled = 0
    skipped_events = 0

    # #region agent log
    try:
        import json, time as _t
        _first_events = [(e.time, e.duration) for e in events[:10]]
        with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
            _f.write(json.dumps({"hypothesisId": "H1", "location": "audio_negate_block.py:build_negative_track:pre_loop", "message": "event info before loop", "data": {"total_events": len(events), "first_10_events": _first_events, "sample_rate": sample_rate, "total_samples": total_samples, "total_duration_sec": total_samples / sample_rate, "aligned_sub_shape": list(aligned_sub.shape), "aligned_sub_nonzero": int(np.count_nonzero(aligned_sub))}, "timestamp": int(_t.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    for event in events:
        start = max(0, int(event.time * sample_rate))
        end = min(total_samples, int((event.time + event.duration) * sample_rate))

        if start >= end or start >= total_samples:
            skipped_events += 1
            continue

        region_length = end - start
        envelope = _build_fade_envelope(region_length, fade_samples)

        # Apply onset emphasis (front-loads subtraction at the transient)
        onset_env = _build_onset_emphasis_envelope(
            region_length, onset_emphasis, sample_rate
        )

        for ch in range(n_channels):
            negative[ch, start:end] = aligned_sub[ch, start:end] * envelope * onset_env

        regions_filled += 1

        # #region agent log
        if regions_filled <= 5:
            try:
                import json, time as _t2
                _sub_max = float(np.max(np.abs(aligned_sub[:, start:end]))) if end > start else 0.0
                _neg_max = float(np.max(np.abs(negative[:, start:end]))) if end > start else 0.0
                with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                    _f.write(json.dumps({"hypothesisId": "H1,H2,H3", "location": "audio_negate_block.py:build_negative_track:event_loop", "message": "event slice detail", "data": {"event_idx": regions_filled, "event_time": event.time, "event_duration": event.duration, "event_end_time": event.time + event.duration, "start_sample": start, "end_sample": end, "region_samples": region_length, "region_ms": region_length / sample_rate * 1000, "sub_region_max_amplitude": _sub_max, "neg_region_max_amplitude": _neg_max}, "timestamp": int(_t2.time() * 1000)}) + "\n")
            except Exception:
                pass
        # #endregion

    Log.info(
        f"build_negative_track: Filled {regions_filled} event regions into negative track. "
        f"Skipped {skipped_events} events (start>=end). "
        f"Negative track non-zero samples: {np.count_nonzero(negative)} / {negative.size}"
    )

    # #region agent log
    try:
        import json, time as _t3
        _neg_abs_max = float(np.max(np.abs(negative))) if negative.size > 0 else 0.0
        _target_abs_max = float(np.max(np.abs(target_2d))) if target_2d.size > 0 else 0.0
        with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
            _f.write(json.dumps({"hypothesisId": "H1,H2,H3", "location": "audio_negate_block.py:build_negative_track:post_loop", "message": "negative track summary", "data": {"regions_filled": regions_filled, "skipped_events": skipped_events, "negative_nonzero": int(np.count_nonzero(negative)), "negative_total": negative.size, "negative_max_amplitude": _neg_abs_max, "target_max_amplitude": _target_abs_max, "total_samples": total_samples}, "timestamp": int(_t3.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    # Step 3: Spectral subtraction -- subtract negative frequencies from target
    # Using STFT-based spectral subtraction for cleaner frequency removal
    # than time-domain subtraction (which requires perfect phase alignment).
    n_fft = 2048
    hop_length = n_fft // 4

    result = np.zeros_like(target_2d)
    for ch in range(n_channels):
        target_stft = np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(target_2d[ch], (n_fft // 2, n_fft // 2), mode='reflect'),
                n_fft,
            )[::hop_length]
            * np.hanning(n_fft),
            axis=-1,
        )
        neg_stft = np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(negative[ch], (n_fft // 2, n_fft // 2), mode='reflect'),
                n_fft,
            )[::hop_length]
            * np.hanning(n_fft),
            axis=-1,
        )

        target_mag = np.abs(target_stft)
        neg_mag = np.abs(neg_stft)
        target_phase = np.angle(target_stft)

        # Spectral subtraction: remove negative's magnitude (amplified by
        # subtract_gain), floor at zero
        result_mag = np.maximum(target_mag - neg_mag * subtract_gain, 0.0)

        # Reconstruct with original phase
        result_stft = result_mag * np.exp(1j * target_phase)

        # Inverse STFT via overlap-add
        frames = np.fft.irfft(result_stft, n=n_fft, axis=-1)
        frames *= np.hanning(n_fft)  # synthesis window

        out_ch = np.zeros(total_samples + n_fft, dtype=np.float64)
        for i, frame in enumerate(frames):
            start_pos = i * hop_length
            end_pos = start_pos + n_fft
            if end_pos <= len(out_ch):
                out_ch[start_pos:end_pos] += frame

        # Trim padding and normalise for overlap-add gain
        result[ch] = out_ch[n_fft // 2 : n_fft // 2 + total_samples]

    # Normalise overlap-add gain (sum of squared Hann window at hop=n_fft/4)
    ola_gain = np.sum(np.hanning(n_fft) ** 2) / hop_length
    if ola_gain > 0:
        result /= ola_gain

    np.clip(result, -1.0, 1.0, out=result)

    # #region agent log
    try:
        import json, time as _t_audit
        # Measure subtraction effectiveness at event regions
        _event_target_energy = 0.0
        _event_result_energy = 0.0
        _event_samples = 0
        for ev in events[:20]:  # sample first 20 events
            _s = max(0, int(ev.time * sample_rate))
            _e = min(total_samples, int((ev.time + ev.duration) * sample_rate))
            if _s < _e:
                _event_target_energy += float(np.sum(target_2d[:, _s:_e] ** 2))
                _event_result_energy += float(np.sum(result[:, _s:_e] ** 2))
                _event_samples += (_e - _s) * n_channels
        _reduction_pct = (1.0 - _event_result_energy / _event_target_energy) * 100 if _event_target_energy > 0 else 0.0
        _result_max = float(np.max(np.abs(result))) if result.size > 0 else 0.0
        _target_max = float(np.max(np.abs(target_2d))) if target_2d.size > 0 else 0.0
        # Also check a specific event region in detail
        _ev0 = events[0] if events else None
        _detail = {}
        if _ev0:
            _ds = max(0, int(_ev0.time * sample_rate))
            _de = min(total_samples, int((_ev0.time + _ev0.duration) * sample_rate))
            _detail = {
                "event0_time": _ev0.time,
                "target_region_max": float(np.max(np.abs(target_2d[:, _ds:_de]))),
                "negative_region_max": float(np.max(np.abs(negative[:, _ds:_de]))),
                "result_region_max": float(np.max(np.abs(result[:, _ds:_de]))),
                "target_region_rms": float(np.sqrt(np.mean(target_2d[:, _ds:_de] ** 2))),
                "result_region_rms": float(np.sqrt(np.mean(result[:, _ds:_de] ** 2))),
            }
        with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
            _f.write(json.dumps({"hypothesisId": "H-AUDIT", "location": "audio_negate_block.py:build_negative_track:post_subtraction", "message": "subtraction audit", "data": {"method": "spectral_subtraction", "event_target_energy": _event_target_energy, "event_result_energy": _event_result_energy, "energy_reduction_pct": round(_reduction_pct, 1), "event_samples_checked": _event_samples, "result_max": _result_max, "target_max": _target_max, "result_nonzero": int(np.count_nonzero(result)), "first_event_detail": _detail}, "timestamp": int(_t_audit.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    # Restore original shape
    if is_mono_1d:
        return negative[0], result[0]
    neg_out = negative.reshape(orig_shape) if negative.shape != orig_shape else negative
    res_out = result.reshape(orig_shape) if result.shape != orig_shape else result
    return neg_out, res_out


class AudioNegateBlockProcessor(BlockProcessor):
    """Processor for AudioNegate block type."""

    def can_process(self, block: Block) -> bool:
        """Check if this processor handles the given block."""
        return block.type == "AudioNegate"

    def get_block_type(self) -> str:
        """Get the block type identifier."""
        return "AudioNegate"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for AudioNegate block.

        Status levels:
        - Error (0): Required inputs not connected
        - Stale (1): Data is stale
        - Ready (2): All requirements met
        """
        from src.features.blocks.domain import BlockStatusLevel
        from src.shared.domain.data_state import DataState

        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio input is connected."""
            if not hasattr(f, "connection_service"):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [
                c
                for c in connections
                if c.target_block_id == blk.id and c.target_input_name == "audio"
            ]
            return len(incoming) > 0

        def check_events_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if events input is connected."""
            if not hasattr(f, "connection_service"):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [
                c
                for c in connections
                if c.target_block_id == blk.id and c.target_input_name == "events"
            ]
            return len(incoming) > 0

        def check_data_fresh(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if block data is fresh (not stale)."""
            if not hasattr(f, "data_state_service") or not f.data_state_service:
                return True
            try:
                project_id = (
                    getattr(f, "current_project_id", None)
                    if hasattr(f, "current_project_id")
                    else None
                )
                data_state = f.data_state_service.get_block_data_state(blk.id, project_id)
                return data_state != DataState.STALE
            except Exception:
                return True

        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_audio_input, check_events_input],
            ),
            BlockStatusLevel(
                priority=1,
                name="stale",
                display_name="Stale",
                color="#ffa94d",
                conditions=[check_data_fresh],
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[],
            ),
        ]

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process AudioNegate block: apply negation to audio at event time regions.

        Handles both single AudioDataItem and list of AudioDataItems on the audio
        input. The output structure mirrors the input.
        """
        from src.application.processing.output_name_helpers import make_output_name

        # -- Validate audio input -------------------------------------------
        audio_input = inputs.get("audio")
        if audio_input is None:
            raise ProcessingError(
                "AudioNegate block requires audio data from upstream block. "
                "Connect an audio source to the 'audio' input.",
                block_id=block.id,
                block_name=block.name,
            )

        # Normalise to list
        if isinstance(audio_input, list):
            audio_items = audio_input
            input_is_list = True
        else:
            audio_items = [audio_input]
            input_is_list = False

        for item in audio_items:
            if not isinstance(item, AudioDataItem):
                raise ProcessingError(
                    f"Input 'audio' contains non-audio item: {type(item).__name__}",
                    block_id=block.id,
                    block_name=block.name,
                )

        # -- Validate events input ------------------------------------------
        events_input = inputs.get("events")
        if events_input is None:
            raise ProcessingError(
                "AudioNegate block requires event data from upstream block. "
                "Connect an event source to the 'events' input.",
                block_id=block.id,
                block_name=block.name,
            )

        # Collect all events from EventDataItem(s)
        if isinstance(events_input, list):
            event_items = events_input
        else:
            event_items = [events_input]

        all_events = []
        for event_item in event_items:
            if isinstance(event_item, EventDataItem):
                all_events.extend(event_item.get_events())

        if not all_events:
            Log.warning(
                "AudioNegateBlockProcessor: No events found - passing audio through unchanged"
            )
            # Return audio unchanged
            if input_is_list:
                return {"audio": audio_items}
            else:
                return {"audio": audio_items[0]}

        # Filter events that have positive duration
        valid_events = [e for e in all_events if e.duration > 0]
        if not valid_events:
            Log.warning(
                "AudioNegateBlockProcessor: All events have zero duration - "
                "passing audio through unchanged. Events need duration > 0 to define regions."
            )
            if input_is_list:
                return {"audio": audio_items}
            else:
                return {"audio": audio_items[0]}

        # -- Read settings from block metadata ------------------------------
        mode = block.metadata.get("mode", "silence")
        fade_ms = float(block.metadata.get("fade_ms", 10.0))
        attenuation_db = float(block.metadata.get("attenuation_db", -20.0))
        subtract_gain = float(block.metadata.get("subtract_gain", 1.0))
        onset_emphasis = float(block.metadata.get("onset_emphasis", 1.0))

        if mode not in NEGATE_MODES:
            raise ProcessingError(
                f"Unknown negation mode: '{mode}'. "
                f"Valid modes: {', '.join(NEGATE_MODES.keys())}",
                block_id=block.id,
                block_name=block.name,
            )

        # -- Handle subtract_audio input (optional) -------------------------
        subtract_data_map: Dict[int, tuple] = {}  # idx -> (np.ndarray, sample_rate)
        if mode == "subtract":
            subtract_input = inputs.get("subtract_audio")
            if subtract_input is None:
                raise ProcessingError(
                    "Subtract mode requires the 'subtract_audio' input to be connected. "
                    "Connect an audio source to subtract from the target audio.",
                    block_id=block.id,
                    block_name=block.name,
                )

            if isinstance(subtract_input, list):
                subtract_items = subtract_input
            else:
                subtract_items = [subtract_input]

            for idx, sub_item in enumerate(subtract_items):
                if isinstance(sub_item, AudioDataItem):
                    sub_data = sub_item.get_audio_data()
                    sub_sr = sub_item.sample_rate or 44100
                    if sub_data is not None:
                        subtract_data_map[idx] = (sub_data, sub_sr)
                        Log.info(
                            f"AudioNegateBlockProcessor: subtract_audio[{idx}] "
                            f"shape={sub_data.shape}, sr={sub_sr}, "
                            f"dtype={sub_data.dtype}, "
                            f"min={sub_data.min():.4f}, max={sub_data.max():.4f}"
                        )

            if not subtract_data_map:
                raise ProcessingError(
                    "Could not load audio data from subtract_audio input.",
                    block_id=block.id,
                    block_name=block.name,
                )

        # -- Progress -------------------------------------------------------
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        total_items = len(audio_items)
        mode_info = NEGATE_MODES[mode]
        if progress_tracker:
            progress_tracker.start(
                f"Applying {mode_info['name']} negation to {total_items} item(s)...",
                total=total_items,
            )

        Log.info(
            f"AudioNegateBlockProcessor: Applying {mode} negation to "
            f"{total_items} audio item(s) using {len(valid_events)} events "
            f"(fade={fade_ms}ms, attenuation={attenuation_db}dB, "
            f"subtract_gain={subtract_gain}x, onset_emphasis={onset_emphasis}x)"
        )

        # -- Build output directory -----------------------------------------
        first_path = audio_items[0].file_path if audio_items[0].file_path else None
        if first_path:
            output_dir = Path(first_path).parent / f"{block.name}_negated"
        else:
            output_dir = Path(tempfile.gettempdir()) / "echozero_negate"
        output_dir.mkdir(parents=True, exist_ok=True)

        # -- Process each audio item ----------------------------------------
        output_items: List[AudioDataItem] = []

        for idx, audio_item in enumerate(audio_items):
            audio_data = audio_item.get_audio_data()
            if audio_data is None:
                raise ProcessingError(
                    f"Could not load audio data from item '{audio_item.name}'",
                    block_id=block.id,
                    block_name=block.name,
                )

            sample_rate = audio_item.sample_rate or 44100

            # Determine stem name for output files
            if audio_item.file_path:
                stem_name = Path(audio_item.file_path).stem
            else:
                stem_name = audio_item.name or f"item_{idx}"

            Log.info(
                f"AudioNegateBlockProcessor: target audio[{idx}] "
                f"shape={audio_data.shape}, sr={sample_rate}, "
                f"dtype={audio_data.dtype}, "
                f"min={audio_data.min():.4f}, max={audio_data.max():.4f}"
            )

            if mode == "subtract":
                # --- Subtract mode: build negative track, then subtract ---
                sub_entry = subtract_data_map.get(idx)
                if sub_entry is None:
                    sub_entry = next(iter(subtract_data_map.values()), None)
                if sub_entry is None:
                    raise ProcessingError(
                        f"No subtract audio available for item '{audio_item.name}'",
                        block_id=block.id,
                        block_name=block.name,
                    )
                subtract_data, subtract_sr = sub_entry

                try:
                    negative_track, subtracted_result = build_negative_track(
                        target_audio=audio_data,
                        subtract_audio=subtract_data,
                        events=valid_events,
                        sample_rate=sample_rate,
                        fade_ms=fade_ms,
                        subtract_sr=subtract_sr,
                        subtract_gain=subtract_gain,
                        onset_emphasis=onset_emphasis,
                    )
                except Exception as e:
                    raise ProcessingError(
                        f"Subtract processing failed on '{audio_item.name}': {e}",
                        block_id=block.id,
                        block_name=block.name,
                    ) from e

                # -- Save the negative track (extracted noise at event regions) --
                neg_path = output_dir / f"{stem_name}_negative.wav"
                neg_item = AudioDataItem(
                    id="",
                    block_id=block.id,
                    name=f"{block.name}_negative",
                    type="Audio",
                    created_at=datetime.utcnow(),
                    file_path=str(neg_path),
                )
                # #region agent log
                try:
                    import json, time as _t4
                    with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                        _f.write(json.dumps({"hypothesisId": "H-SAVE", "location": "audio_negate_block.py:process:before_save", "message": "data before save_audio", "data": {"neg_shape": list(negative_track.shape), "neg_nonzero": int(np.count_nonzero(negative_track)), "neg_total": negative_track.size, "neg_pct_nonzero": round(np.count_nonzero(negative_track) / negative_track.size * 100, 1), "neg_max": float(np.max(np.abs(negative_track))), "sub_shape": list(subtracted_result.shape), "sub_nonzero": int(np.count_nonzero(subtracted_result)), "sub_max": float(np.max(np.abs(subtracted_result))), "neg_path": str(neg_path), "sub_path": str(sub_path), "neg_first_1000_nonzero": int(np.count_nonzero(negative_track[..., :44100])), "neg_mid_1000_nonzero": int(np.count_nonzero(negative_track[..., 1810000:1820000]))}, "timestamp": int(_t4.time() * 1000)}) + "\n")
                except Exception:
                    pass
                # #endregion
                neg_item.set_audio_data(negative_track, sample_rate)
                neg_item.save_audio(str(neg_path))
                neg_item.metadata["output_name"] = make_output_name("audio", "negative")

                # -- Save the subtracted result (source minus negative) --
                sub_path = output_dir / f"{stem_name}_subtracted.wav"
                sub_item = AudioDataItem(
                    id="",
                    block_id=block.id,
                    name=f"{block.name}_subtracted",
                    type="Audio",
                    created_at=datetime.utcnow(),
                    file_path=str(sub_path),
                )
                sub_item.set_audio_data(subtracted_result, sample_rate)
                sub_item.save_audio(str(sub_path))
                sub_item.metadata["output_name"] = make_output_name("audio", "subtracted")

                # #region agent log
                try:
                    import json, time as _t5, os, soundfile as _sf
                    _neg_file_size = os.path.getsize(str(neg_path)) if os.path.exists(str(neg_path)) else 0
                    _sub_file_size = os.path.getsize(str(sub_path)) if os.path.exists(str(sub_path)) else 0
                    _neg_item_data_nonzero = int(np.count_nonzero(neg_item._audio_data)) if neg_item._audio_data is not None else -1
                    # Read the file BACK from disk and verify contents
                    _readback, _rb_sr = _sf.read(str(neg_path))
                    _rb_nonzero = int(np.count_nonzero(_readback))
                    _rb_shape = list(_readback.shape)
                    _rb_max = float(np.max(np.abs(_readback))) if _readback.size > 0 else 0
                    # Check first 44100 samples (1 second) - should be all zeros if events start at 41s
                    _first_sec_nonzero = int(np.count_nonzero(_readback[:44100]))
                    with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                        _f.write(json.dumps({"hypothesisId": "H-VERIFY", "location": "audio_negate_block.py:process:readback_verify", "message": "READBACK from disk", "data": {"neg_file_path": str(neg_path), "neg_file_size": _neg_file_size, "readback_shape": _rb_shape, "readback_sr": _rb_sr, "readback_nonzero": _rb_nonzero, "readback_total": _readback.size, "readback_pct_nonzero": round(_rb_nonzero / _readback.size * 100, 1), "readback_max": _rb_max, "first_sec_nonzero": _first_sec_nonzero, "in_memory_nonzero": _neg_item_data_nonzero}, "timestamp": int(_t5.time() * 1000)}) + "\n")
                except Exception as _e:
                    try:
                        with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                            _f.write(json.dumps({"hypothesisId": "H-VERIFY", "location": "audio_negate_block.py:process:readback_error", "message": f"readback failed: {_e}", "timestamp": int(time.time() * 1000)}) + "\n")
                    except Exception:
                        pass
                # #endregion

                # Generate waveforms for both
                try:
                    from src.shared.application.services.waveform_service import (
                        get_waveform_service,
                    )
                    waveform_service = get_waveform_service()
                    waveform_service.compute_and_store(neg_item)
                    waveform_service.compute_and_store(sub_item)
                except Exception as e:
                    Log.warning(
                        f"AudioNegateBlockProcessor: Failed to generate waveform: {e}"
                    )

                # -- Pass through the source subtract_audio for comparison --
                sub_src_item = subtract_items[idx] if idx < len(subtract_items) else subtract_items[0]
                src_path = output_dir / f"{stem_name}_source_subtract.wav"
                src_item = AudioDataItem(
                    id="",
                    block_id=block.id,
                    name=f"{block.name}_source_subtract",
                    type="Audio",
                    created_at=datetime.utcnow(),
                    file_path=str(src_path),
                )
                src_item.set_audio_data(subtract_data, subtract_sr or sample_rate)
                src_item.save_audio(str(src_path))
                src_item.metadata["output_name"] = make_output_name("audio", "source_subtract")

                try:
                    waveform_service.compute_and_store(src_item)
                except Exception:
                    pass

                # #region agent log
                try:
                    import json, time as _t6, os, soundfile as _sf2
                    _src_rb, _src_rb_sr = _sf2.read(str(src_path))
                    _src_rb_nonzero = int(np.count_nonzero(_src_rb))
                    _neg_rb2, _ = _sf2.read(str(neg_path))
                    _neg_rb2_nonzero = int(np.count_nonzero(_neg_rb2))
                    with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                        _f.write(json.dumps({"hypothesisId": "H-SRC-VERIFY", "location": "audio_negate_block.py:process:src_readback", "message": "COMPARE files on disk", "data": {"src_path": str(src_path), "neg_path": str(neg_path), "sub_path": str(sub_path), "src_readback_nonzero": _src_rb_nonzero, "src_readback_total": _src_rb.size, "src_readback_pct": round(_src_rb_nonzero / _src_rb.size * 100, 1), "neg_readback_nonzero": _neg_rb2_nonzero, "neg_readback_total": _neg_rb2.size, "neg_readback_pct": round(_neg_rb2_nonzero / _neg_rb2.size * 100, 1), "files_identical": bool(np.array_equal(_src_rb, _neg_rb2)), "src_first_sec_nonzero": int(np.count_nonzero(_src_rb[:44100])), "neg_first_sec_nonzero": int(np.count_nonzero(_neg_rb2[:44100]))}, "timestamp": int(_t6.time() * 1000)}) + "\n")
                except Exception as _e6:
                    try:
                        with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                            _f.write(json.dumps({"hypothesisId": "H-SRC-VERIFY", "location": "audio_negate_block.py:process:src_readback_error", "message": f"readback failed: {_e6}", "timestamp": int(_t6.time() * 1000)}) + "\n")
                    except Exception:
                        pass
                # #endregion

                output_items.append(src_item)
                output_items.append(neg_item)
                output_items.append(sub_item)

                Log.info(
                    f"AudioNegateBlockProcessor: Saved source subtract -> {src_path}"
                )
                Log.info(
                    f"AudioNegateBlockProcessor: Saved negative track -> {neg_path}"
                )
                Log.info(
                    f"AudioNegateBlockProcessor: Saved subtracted result -> {sub_path}"
                )

            else:
                # --- Silence / Attenuate mode ---
                try:
                    negated_data = apply_negate(
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        events=valid_events,
                        mode=mode,
                        fade_ms=fade_ms,
                        attenuation_db=attenuation_db,
                    )
                except Exception as e:
                    raise ProcessingError(
                        f"Negation processing failed on '{audio_item.name}': {e}",
                        block_id=block.id,
                        block_name=block.name,
                    ) from e

                output_path = output_dir / f"{stem_name}_{mode}.wav"
                output_item = AudioDataItem(
                    id="",
                    block_id=block.id,
                    name=f"{block.name}_{stem_name}",
                    type="Audio",
                    created_at=datetime.utcnow(),
                    file_path=str(output_path),
                )

                output_item.set_audio_data(negated_data, sample_rate)
                output_item.save_audio(str(output_path))

                upstream_output_name = audio_item.metadata.get("output_name")
                if upstream_output_name:
                    output_item.metadata["output_name"] = upstream_output_name
                else:
                    output_item.metadata["output_name"] = make_output_name(
                        "audio", stem_name
                    )

                try:
                    from src.shared.application.services.waveform_service import (
                        get_waveform_service,
                    )
                    waveform_service = get_waveform_service()
                    waveform_service.compute_and_store(output_item)
                except Exception as e:
                    Log.warning(
                        f"AudioNegateBlockProcessor: Failed to generate waveform "
                        f"for '{stem_name}': {e}"
                    )

                output_items.append(output_item)

                Log.info(
                    f"AudioNegateBlockProcessor: Saved negated '{stem_name}' -> {output_path}"
                )

            if progress_tracker:
                progress_tracker.update(
                    idx + 1,
                    total_items,
                    f"Negated {idx + 1}/{total_items}: {stem_name}",
                )

        # -- Complete -------------------------------------------------------
        if progress_tracker:
            progress_tracker.complete(
                f"Applied {mode_info['name']} negation to {total_items} item(s)"
            )

        Log.info(
            f"AudioNegateBlockProcessor: Returning {len(output_items)} "
            f"audio item(s) on 'audio' port"
        )

        return {"audio": output_items}

    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """Get expected output names for the audio negate block."""
        from src.application.processing.output_name_helpers import make_output_name

        mode = block.metadata.get("mode", "silence")
        if mode == "subtract":
            return {"audio": [
                make_output_name("audio", "source_subtract"),
                make_output_name("audio", "negative"),
                make_output_name("audio", "subtracted"),
            ]}
        return {"audio": [make_output_name("audio", "main")]}

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None,
    ) -> List[str]:
        """Validate AudioNegate block configuration."""
        errors = []
        mode = block.metadata.get("mode", "silence")
        if mode not in NEGATE_MODES:
            errors.append(f"Invalid negation mode: '{mode}'")

        fade_ms = block.metadata.get("fade_ms", 10.0)
        if fade_ms < 0 or fade_ms > 100:
            errors.append(f"Fade duration {fade_ms}ms is outside valid range (0-100 ms)")

        attenuation_db = block.metadata.get("attenuation_db", -20.0)
        if attenuation_db < -60 or attenuation_db > 0:
            errors.append(
                f"Attenuation {attenuation_db}dB is outside valid range (-60 to 0 dB)"
            )

        subtract_gain = block.metadata.get("subtract_gain", 1.0)
        if subtract_gain < 1.0 or subtract_gain > 10.0:
            errors.append(
                f"Subtract gain {subtract_gain}x is outside valid range (1.0-10.0)"
            )

        onset_emphasis = block.metadata.get("onset_emphasis", 1.0)
        if onset_emphasis < 1.0 or onset_emphasis > 5.0:
            errors.append(
                f"Onset emphasis {onset_emphasis}x is outside valid range (1.0-5.0)"
            )

        return errors


register_processor_class(AudioNegateBlockProcessor)
