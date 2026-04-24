"""
DetectOnsetsProcessor: Detects onset times in audio and produces timeline events.
Exists because onset detection is the core analysis step — every event on the timeline starts here.
Used by ExecutionEngine when running blocks of type 'DetectOnsets'.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


def _default_onset_detect(
    file_path: str,
    sample_rate: int,
    threshold: float,
    min_gap: float,
    *,
    method: str = "default",
    backtrack: bool = True,
    timing_offset_ms: float = 0.0,
) -> list[float]:
    """Detect onsets using librosa. Production default — requires librosa installed."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise NotImplementedError(
            "Default onset detection requires librosa. "
            "Install with: pip install librosa"
        )

    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    if y.size == 0:
        return []

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-6:
        return []

    onset_envelope = _build_onset_envelope(y=y, sr=sr, method=method)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        backtrack=bool(backtrack),
        delta=threshold,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    if timing_offset_ms:
        offset_seconds = float(timing_offset_ms) / 1000.0
        onset_times = [max(0.0, t + offset_seconds) for t in onset_times]

    # Apply min_gap filter: remove onsets too close together
    if min_gap > 0 and len(onset_times) > 1:
        filtered = [onset_times[0]]
        for t in onset_times[1:]:
            if t - filtered[-1] >= min_gap:
                filtered.append(t)
        onset_times = filtered

    return onset_times


def _build_onset_envelope(*, y: Any, sr: int, method: str) -> Any:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise NotImplementedError(
            "Default onset detection requires librosa. Install with: pip install librosa"
        ) from exc

    normalized_method = str(method or "default").strip().lower()
    if normalized_method == "default":
        return librosa.onset.onset_strength(y=y, sr=sr)
    if normalized_method == "hfc":
        spectrum = np.abs(librosa.stft(y))
        freq_weights = np.linspace(1.0, 2.5, spectrum.shape[0], dtype=np.float32)[:, np.newaxis]
        weighted = spectrum * freq_weights
        return librosa.onset.onset_strength(sr=sr, S=weighted)
    if normalized_method == "complex":
        spectrum = np.abs(librosa.stft(y))
        return librosa.onset.onset_strength(sr=sr, S=spectrum, lag=1, max_size=1)
    raise ValueError(f"Unsupported onset detection method: {method}")


def _invoke_onset_detect_fn(
    onset_detect_fn: Callable[..., list[float]],
    *,
    file_path: str,
    sample_rate: int,
    threshold: float,
    min_gap: float,
    method: str,
    backtrack: bool,
    timing_offset_ms: float,
) -> list[float]:
    signature = inspect.signature(onset_detect_fn)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: dict[str, Any] = {}
    for key, value in (
        ("method", method),
        ("backtrack", backtrack),
        ("timing_offset_ms", timing_offset_ms),
    ):
        if accepts_kwargs or key in signature.parameters:
            kwargs[key] = value
    return onset_detect_fn(file_path, sample_rate, threshold, min_gap, **kwargs)


class DetectOnsetsProcessor:
    """Detects audio onsets and returns EventData with a single layer of marker events."""

    def __init__(
        self,
        onset_detect_fn: Callable[..., list[float]] | None = None,
    ) -> None:
        self._onset_detect_fn = onset_detect_fn or _default_onset_detect

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        """Read upstream audio, detect onsets, and return EventData."""
        # Report start
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_onsets",
                percent=0.0,
                message="Starting onset detection",
            )
        )

        # Read audio input from upstream
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — "
                    f"connect an audio source to 'audio_in'"
                )
            )

        # Read settings from the block
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        threshold = block.settings.get("threshold", 0.5)
        min_gap = block.settings.get("min_gap", 0.05)
        method = str(block.settings.get("method", "default"))
        backtrack = bool(block.settings.get("backtrack", True))
        timing_offset_ms = float(block.settings.get("timing_offset_ms", 0.0))

        # Run onset detection
        try:
            onset_times = _invoke_onset_detect_fn(
                self._onset_detect_fn,
                file_path=audio.file_path,
                sample_rate=audio.sample_rate,
                threshold=threshold,
                min_gap=min_gap,
                method=method,
                backtrack=backtrack,
                timing_offset_ms=timing_offset_ms,
            )
        except Exception as exc:
            return err(
                ExecutionError(f"Onset detection failed for block '{block_id}': {exc}")
            )

        # Report halfway
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_onsets",
                percent=0.5,
                message="Creating events from onsets",
            )
        )

        # Convert onset times to Event objects
        events: list[Event] = []
        for i, t in enumerate(onset_times):
            events.append(
                Event(
                    id=f"{block_id}_onset_{i}",
                    time=t,
                    duration=0.0,
                    classifications={},
                    metadata={
                        "threshold": threshold,
                        "min_gap": min_gap,
                        "method": method,
                        "backtrack": backtrack,
                        "timing_offset_ms": timing_offset_ms,
                        "index": i,
                    },
                    origin=block_id,
                )
            )

        layer = Layer(
            id=f"{block_id}_onsets",
            name="Detected Onsets",
            events=tuple(events),
        )

        # Report complete
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_onsets",
                percent=1.0,
                message="Onset detection complete",
            )
        )

        return ok(EventData(layers=(layer,)))
