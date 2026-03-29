"""
DetectOnsetsProcessor: Detects onset times in audio and produces timeline events.
Exists because onset detection is the core analysis step — every event on the timeline starts here.
Used by ExecutionEngine when running blocks of type 'DetectOnsets'.
"""

from __future__ import annotations

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
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=False, delta=threshold,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # Apply min_gap filter: remove onsets too close together
    if min_gap > 0 and len(onset_times) > 1:
        filtered = [onset_times[0]]
        for t in onset_times[1:]:
            if t - filtered[-1] >= min_gap:
                filtered.append(t)
        onset_times = filtered

    return onset_times


class DetectOnsetsProcessor:
    """Detects audio onsets and returns EventData with a single layer of marker events."""

    def __init__(
        self,
        onset_detect_fn: Callable[[str, int, float, float], list[float]] | None = None,
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

        # Run onset detection
        try:
            onset_times = self._onset_detect_fn(
                audio.file_path, audio.sample_rate, threshold, min_gap
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
                    metadata={"threshold": threshold, "min_gap": min_gap, "index": i},
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
