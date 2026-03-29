"""
ExportAudioDatasetProcessor: Extracts audio clips at event times for ML training.
Exists because building training datasets from detected events is the flywheel —
detect → classify → correct → export → train → better classify.
Used by ExecutionEngine when running blocks of type 'ExportAudioDataset'.

For each event, extracts the audio between event.time and event.time + event.duration
from the source audio and writes it as an individual file. Organizes by classification
if available.

The injectable function pattern allows testing without audio libraries.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from echozero.domain.types import AudioData, EventData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# Export function signature for DI
# ---------------------------------------------------------------------------

ExportDatasetFn = Callable[
    [
        str,   # source_audio_path
        int,   # sample_rate
        str,   # output_dir
        str,   # format (wav, flac, etc.)
        list[dict[str, Any]],  # clips: [{start, end, label, filename}]
    ],
    int,  # number of clips written
]


def _default_export_dataset(
    source_audio: str,
    sample_rate: int,
    output_dir: str,
    fmt: str,
    clips: list[dict[str, Any]],
) -> int:
    """Extract audio clips and write to disk. Requires soundfile."""
    raise NotImplementedError(
        "Default dataset export requires soundfile. Provide a custom export_fn."
    )


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ExportAudioDatasetProcessor:
    """Extracts audio clips at event time regions for ML dataset creation."""

    def __init__(self, export_fn: ExportDatasetFn | None = None) -> None:
        self._export_fn = export_fn or _default_export_dataset

    def execute(self, block_id: str, context: ExecutionContext) -> Result[dict[str, Any]]:
        """Read upstream audio + events, extract clips, write to output_dir.

        Returns a dict with export stats: {clips_exported, output_dir, classes}.
        """
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_audio_dataset",
                percent=0.0,
                message="Starting audio dataset export",
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
        output_dir = settings.get("output_dir")
        fmt = settings.get("format", "wav").lower()
        organize_by_class = settings.get("organize_by_class", True)
        min_duration = settings.get("min_duration", 0.01)

        if not output_dir:
            return err(ValidationError(
                f"Block '{block_id}' is missing required setting 'output_dir'"
            ))

        # Build clip list from events
        clips: list[dict[str, Any]] = []
        classes_seen: set[str] = set()

        for layer in event_data.layers:
            for event in layer.events:
                if event.duration < min_duration:
                    continue

                # Determine classification label
                label = (
                    event.classifications.get("class")
                    or event.classifications.get("note")
                    or layer.name
                    or "unclassified"
                )
                classes_seen.add(label)

                # Build output path
                if organize_by_class:
                    clip_dir = os.path.join(output_dir, label)
                else:
                    clip_dir = output_dir

                filename = f"{event.id}.{fmt}"

                clips.append({
                    "start": event.time,
                    "end": event.time + event.duration,
                    "label": label,
                    "filename": filename,
                    "output_dir": clip_dir,
                })

        if not clips:
            return err(ExecutionError(
                f"No events with duration >= {min_duration}s for block '{block_id}'"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_audio_dataset",
                percent=0.2,
                message=f"Exporting {len(clips)} clips across {len(classes_seen)} classes",
            )
        )

        # Export clips
        try:
            clips_written = self._export_fn(
                source_audio=audio.file_path,
                sample_rate=audio.sample_rate,
                output_dir=output_dir,
                fmt=fmt,
                clips=clips,
            )
        except NotImplementedError:
            return err(ExecutionError(
                f"Dataset export backend not available for block '{block_id}'. "
                f"Install soundfile or provide a custom export_fn."
            ))
        except Exception as exc:
            return err(ExecutionError(
                f"Dataset export failed for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_audio_dataset",
                percent=1.0,
                message=f"Exported {clips_written} clips to {output_dir}",
            )
        )

        return ok({
            "clips_exported": clips_written,
            "output_dir": output_dir,
            "classes": sorted(classes_seen),
            "format": fmt,
        })


