"""
ExportMA2Processor: Exports event data to grandMA2 timecode format.
Exists because MA2 timecode export is how lighting designers get events
into their console — this is the primary output format for the LD market.
Used by ExecutionEngine when running blocks of type 'ExportMA2'.

MA2 timecode format:
- XML-based (.xml) with timecode events
- Each event has a timecode position (HH:MM:SS.FF) and a cue reference
- Frame rate configurable (25, 30, etc.)

The injectable function pattern allows testing without file system dependencies.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from echozero.domain.types import EventData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# MA2 timecode helpers
# ---------------------------------------------------------------------------

VALID_FRAME_RATES = {24, 25, 30}


def seconds_to_timecode(seconds: float, frame_rate: int = 30) -> str:
    """Convert seconds to timecode string HH:MM:SS.FF."""
    total_frames = int(round(seconds * frame_rate))
    ff = total_frames % frame_rate
    total_seconds = total_frames // frame_rate
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ff:02d}"


def build_ma2_xml(
    events: list[dict[str, Any]],
    frame_rate: int = 30,
    track_name: str = "EchoZero",
) -> str:
    """Build a grandMA2 timecode XML string from event dicts.

    Each event dict has: time (float), label (str), cue (str optional).

    Returns:
        XML string in MA2 timecode format.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<MA2Timecode frameRate="{frame_rate}" trackName="{track_name}">',
    ]

    for i, evt in enumerate(events):
        tc = seconds_to_timecode(evt["time"], frame_rate)
        label = evt.get("label", f"Event {i}")
        cue = evt.get("cue", str(i + 1))
        lines.append(
            f'  <Event index="{i}" timecode="{tc}" '
            f'label="{label}" cue="{cue}" />'
        )

    lines.append("</MA2Timecode>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export function signature for DI
# ---------------------------------------------------------------------------

ExportMA2Fn = Callable[
    [
        str,  # xml_content
        str,  # output_path
    ],
    str,  # written file path
]


def _default_export(xml_content: str, output_path: str) -> str:
    """Write MA2 XML to disk. Production default."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    return output_path


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ExportMA2Processor:
    """Exports event data to grandMA2 timecode XML format."""

    def __init__(self, export_fn: ExportMA2Fn | None = None) -> None:
        self._export_fn = export_fn or _default_export

    def execute(self, block_id: str, context: ExecutionContext) -> Result[str]:
        """Read upstream events, build MA2 XML, write to output_path.

        Returns the output file path as a string on success.
        """
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_ma2",
                percent=0.0,
                message="Starting MA2 timecode export",
            )
        )

        # Read event input
        event_data = context.get_input(block_id, "events_in", EventData)
        if event_data is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no event input — "
                    f"connect an event source to 'events_in'"
                )
            )

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        output_path = settings.get("output_path")
        frame_rate = settings.get("frame_rate", 30)
        track_name = settings.get("track_name", "EchoZero")

        # Validate
        if not output_path:
            return err(ValidationError(
                f"Block '{block_id}' is missing required setting 'output_path'"
            ))
        if frame_rate not in VALID_FRAME_RATES:
            return err(ValidationError(
                f"Invalid frame_rate {frame_rate}. Valid: {VALID_FRAME_RATES}"
            ))

        # Flatten all events from all layers, sorted by time
        all_events: list[dict[str, Any]] = []
        for layer in event_data.layers:
            for event in layer.events:
                label = event.classifications.get(
                    "class",
                    event.classifications.get("note", layer.name),
                )
                all_events.append({
                    "time": event.time,
                    "label": f"{layer.name}: {label}",
                    "cue": event.id,
                })

        all_events.sort(key=lambda e: e["time"])

        if not all_events:
            return err(ExecutionError(
                f"No events to export for block '{block_id}'"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_ma2",
                percent=0.5,
                message=f"Building MA2 XML with {len(all_events)} events",
            )
        )

        # Build XML
        xml_content = build_ma2_xml(
            events=all_events,
            frame_rate=frame_rate,
            track_name=track_name,
        )

        # Write to disk
        try:
            written_path = self._export_fn(xml_content, output_path)
        except Exception as exc:
            return err(ExecutionError(
                f"Failed to write MA2 file for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="export_ma2",
                percent=1.0,
                message=f"Exported {len(all_events)} events to {written_path}",
            )
        )

        return ok(written_path)


