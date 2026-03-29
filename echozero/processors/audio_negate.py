"""
AudioNegateProcessor: Silence, attenuate, or subtract audio at event regions.
Exists because hearing what's NOT at the events is essential for validation —
did the detector catch the right things? Also useful for creative isolation.
Used by ExecutionEngine when running blocks of type 'AudioNegate'.

Three modes:
- silence: Zero out audio at event regions with crossfade
- attenuate: Reduce volume at event regions by configurable dB
- subtract: Phase-cancel event regions using a reference audio source

The injectable function pattern allows testing without numpy/scipy.
"""

from __future__ import annotations

from typing import Any, Callable

from echozero.domain.types import AudioData, EventData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok

VALID_MODES = {"silence", "attenuate", "subtract"}


# ---------------------------------------------------------------------------
# Negate function signature for DI
# ---------------------------------------------------------------------------

NegateFn = Callable[
    [
        str,            # audio_file_path
        int,            # sample_rate
        list[tuple[float, float]],  # event regions: (start_time, end_time)
        str,            # mode
        float,          # fade_ms
        float,          # attenuation_db (for attenuate mode)
    ],
    tuple[str, int, float],  # (output_file_path, sample_rate, duration)
]


def _default_negate(
    audio_file: str,
    sample_rate: int,
    regions: list[tuple[float, float]],
    mode: str,
    fade_ms: float,
    attenuation_db: float,
) -> tuple[str, int, float]:
    """Apply silence/attenuate negation. Requires numpy + soundfile."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise ExecutionError(
            "numpy and soundfile are required. Install with: pip install numpy soundfile"
        )

    audio, sr = sf.read(audio_file)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    total_samples = audio.shape[0]
    fade_samples = max(0, int(fade_ms / 1000.0 * sr))

    result = audio.astype(np.float64).copy()

    for start_time, end_time in regions:
        start = max(0, int(start_time * sr))
        end = min(total_samples, int(end_time * sr))
        if start >= end:
            continue

        region_len = end - start
        fade = min(fade_samples, region_len // 2)

        # Build envelope: 1 at edges, 0 (or attenuated) at center
        if mode == "silence":
            gain_center = 0.0
        elif mode == "attenuate":
            gain_center = 10.0 ** (attenuation_db / 20.0)
        else:
            gain_center = 0.0

        envelope = np.full(region_len, gain_center, dtype=np.float64)

        if fade > 0:
            # Fade in: 1 → gain_center
            fade_in = np.linspace(1.0, gain_center, fade)
            envelope[:fade] = fade_in
            # Fade out: gain_center → 1
            fade_out = np.linspace(gain_center, 1.0, fade)
            envelope[-fade:] = fade_out

        for ch in range(result.shape[1]):
            result[start:end, ch] *= envelope

    np.clip(result, -1.0, 1.0, out=result)

    import tempfile
    import os

    fd, output_file = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(output_file, result, sr)

    duration = audio.shape[0] / sr
    return output_file, sr, duration


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class AudioNegateProcessor:
    """Negates audio at event time regions via silence, attenuation, or subtraction."""

    def __init__(self, negate_fn: NegateFn | None = None) -> None:
        self._negate_fn = negate_fn or _default_negate

    def execute(self, block_id: str, context: ExecutionContext) -> Result[AudioData]:
        """Read upstream audio + events, apply negation, return processed AudioData."""
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="audio_negate",
                percent=0.0,
                message="Starting audio negation",
            )
        )

        # Read audio input
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(ExecutionError(
                f"Block '{block_id}' has no audio input — "
                f"connect an audio source to 'audio_in'"
            ))

        # Read event input
        event_data = context.get_input(block_id, "events_in", EventData)
        if event_data is None:
            return err(ExecutionError(
                f"Block '{block_id}' has no event input — "
                f"connect an event source to 'events_in'"
            ))

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        mode = settings.get("mode", "silence")
        fade_ms = settings.get("fade_ms", 10.0)
        attenuation_db = settings.get("attenuation_db", -20.0)

        # Validate
        if mode not in VALID_MODES:
            return err(ValidationError(
                f"Invalid mode '{mode}'. Valid: {', '.join(VALID_MODES)}"
            ))
        if mode == "subtract":
            return err(ValidationError(
                "Subtract mode requires a subtract_audio input — not yet supported in V1. "
                "Use 'silence' or 'attenuate' mode."
            ))
        if not isinstance(fade_ms, (int, float)) or fade_ms < 0 or fade_ms > 100:
            return err(ValidationError(
                f"fade_ms must be 0-100, got {fade_ms}"
            ))
        if not isinstance(attenuation_db, (int, float)) or attenuation_db > 0:
            return err(ValidationError(
                f"attenuation_db must be <= 0, got {attenuation_db}"
            ))

        # Extract event regions (start_time, end_time) from all layers
        regions: list[tuple[float, float]] = []
        for layer in event_data.layers:
            for event in layer.events:
                if event.duration > 0:
                    regions.append((event.time, event.time + event.duration))

        if not regions:
            # No regions to negate — return audio unchanged
            context.progress_bus.publish(
                ProgressReport(
                    block_id=block_id,
                    phase="audio_negate",
                    percent=1.0,
                    message="No event regions with duration > 0 — audio unchanged",
                )
            )
            return ok(audio)

        regions.sort(key=lambda r: r[0])

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="audio_negate",
                percent=0.2,
                message=f"Applying {mode} to {len(regions)} event regions",
            )
        )

        # Apply negation
        try:
            output_file, sr, duration = self._negate_fn(
                audio_file=audio.file_path,
                sample_rate=audio.sample_rate,
                regions=regions,
                mode=mode,
                fade_ms=fade_ms,
                attenuation_db=attenuation_db,
            )
        except (ValidationError, ExecutionError) as exc:
            return err(exc)
        except Exception as exc:
            return err(ExecutionError(
                f"Audio negation failed for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="audio_negate",
                percent=1.0,
                message=f"Negation complete — {len(regions)} regions processed",
            )
        )

        return ok(AudioData(
            sample_rate=sr,
            duration=duration,
            file_path=output_file,
            channel_count=audio.channel_count,
        ))


