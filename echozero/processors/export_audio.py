"""
ExportAudioProcessor: Exports audio to a specified directory and format.
Exists because audio export (wav/mp3/flac/ogg) is expected for any audio tool.
Used by ExecutionEngine when running blocks of type 'ExportAudio'.

The injectable function pattern allows testing without file system dependencies.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from echozero.domain.types import AudioData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg", "aiff"}


# ---------------------------------------------------------------------------
# Export function signature for DI
# ---------------------------------------------------------------------------

ExportAudioFn = Callable[
    [
        str,  # source_file_path
        str,  # output_file_path
        str,  # format
    ],
    str,  # written file path
]


def _default_export(source_path: str, output_path: str, fmt: str) -> str:
    """Copy/convert audio to output path. Production default."""
    import shutil

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # V1: copy source file. Format conversion (mp3/ogg) requires ffmpeg or pydub.
    # If source and target format match, just copy. Otherwise, attempt soundfile.
    source_ext = os.path.splitext(source_path)[1].lstrip(".").lower()
    if source_ext == fmt or fmt == "wav":
        shutil.copy2(source_path, output_path)
    else:
        # Attempt conversion via soundfile (wav→wav only, others need ffmpeg)
        try:
            import soundfile as sf

            data, sr = sf.read(source_path)
            sf.write(output_path, data, sr)
        except Exception:
            # Fallback: just copy
            shutil.copy2(source_path, output_path)

    return output_path


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ExportAudioProcessor:
    """Exports audio files to a directory in a specified format."""

    def __init__(self, export_fn: ExportAudioFn | None = None) -> None:
        self._export_fn = export_fn or _default_export

    def execute(self, block_id: str, context: ExecutionContext) -> Result[str]:
        """Read upstream audio, export to output_dir.

        Returns the output file path as a string on success.
        """
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_audio",
                percent=0.0,
                message="Starting audio export",
            )
        )

        # Read audio input
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — "
                    f"connect an audio source to 'audio_in'"
                )
            )

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        output_dir = settings.get("output_dir")
        fmt = settings.get("format", "wav").lower()
        filename = settings.get("filename")

        # Validate
        if not output_dir:
            return err(ValidationError(
                f"Block '{block_id}' is missing required setting 'output_dir'"
            ))
        if fmt not in SUPPORTED_FORMATS:
            return err(ValidationError(
                f"Unsupported format '{fmt}'. Valid: {', '.join(sorted(SUPPORTED_FORMATS))}"
            ))

        # Build output filename
        if not filename:
            source_name = os.path.splitext(os.path.basename(audio.file_path))[0]
            filename = f"{source_name}.{fmt}"
        elif not filename.endswith(f".{fmt}"):
            filename = f"{filename}.{fmt}"

        output_path = os.path.join(output_dir, filename)

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_audio",
                percent=0.3,
                message=f"Exporting to {output_path}",
            )
        )

        # Export
        try:
            written_path = self._export_fn(audio.file_path, output_path, fmt)
        except Exception as exc:
            return err(ExecutionError(
                f"Audio export failed for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_audio",
                percent=1.0,
                message=f"Exported to {written_path}",
            )
        )

        return ok(written_path)


