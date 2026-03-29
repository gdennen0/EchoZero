"""
LoadAudioProcessor: First real block implementation — loads audio file metadata.
Exists because every pipeline starts with audio ingestion; this block reads file info without loading samples.
Used by ExecutionEngine when running blocks of type 'LoadAudio'.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from echozero.domain.types import AudioData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.result import Result, err, ok


class AudioFileInfo(NamedTuple):
    """Lightweight container for audio file metadata returned by info functions."""

    sample_rate: int
    duration: float
    channels: int


def _default_audio_info(file_path: str) -> AudioFileInfo:
    """Read audio file metadata using soundfile. Production default."""
    import soundfile as sf

    info = sf.info(file_path)
    return AudioFileInfo(
        sample_rate=info.samplerate,
        duration=info.duration,
        channels=info.channels,
    )


class LoadAudioProcessor:
    """Loads audio file metadata and returns an AudioData value object."""

    def __init__(
        self,
        audio_info_fn: Callable[[str], AudioFileInfo] | None = None,
    ) -> None:
        self._audio_info_fn = audio_info_fn or _default_audio_info

    def execute(self, block_id: str, context: ExecutionContext) -> Result[AudioData]:
        """Read audio file info from block settings and return AudioData."""
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        file_path = block.settings.get("file_path")
        if file_path is None:
            return err(
                ValidationError(f"Block '{block_id}' is missing required setting 'file_path'")
            )

        if not os.path.isfile(file_path):
            return err(
                ExecutionError(f"Audio file not found: {file_path}")
            )

        try:
            info = self._audio_info_fn(file_path)
        except Exception as exc:
            return err(
                ExecutionError(f"Failed to read audio info from '{file_path}': {exc}")
            )

        return ok(
            AudioData(
                sample_rate=info.sample_rate,
                duration=info.duration,
                file_path=file_path,
                channel_count=info.channels,
            )
        )

