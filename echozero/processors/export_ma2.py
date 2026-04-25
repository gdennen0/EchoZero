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

from echozero.application.playback.timecode import TimebaseSpec, TimecodeCodec
from echozero.domain.types import EventData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# MA2 timecode helpers
# ---------------------------------------------------------------------------

VALID_FRAME_RATES = (24.0, 25.0, 29.97, 30.0)


def _coerce_frame_rate(frame_rate: object) -> float:
    """Normalize one frame rate setting into a supported canonical value."""

    try:
        resolved = float(frame_rate)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid frame_rate {frame_rate!r}") from None
    for candidate in VALID_FRAME_RATES:
        if abs(resolved - candidate) <= 1e-6:
            return float(candidate)
    raise ValueError(
        f"Invalid frame_rate {frame_rate}. Valid: {VALID_FRAME_RATES}"
    )


def _timecode_codec_for_frame_rate(
    frame_rate: float | int,
    *,
    drop_frame: bool = False,
) -> TimecodeCodec:
    """Build the canonical timecode codec for one export frame-rate setting."""

    return TimecodeCodec(
        TimebaseSpec.from_legacy_fps(
            frame_rate,
            drop_frame=drop_frame,
        )
    )


def _format_frame_rate_for_xml(frame_rate: float | int) -> str:
    """Render frameRate attribute values without noisy trailing decimals."""

    resolved = float(frame_rate)
    if resolved.is_integer():
        return str(int(resolved))
    return str(resolved)


def seconds_to_timecode(
    seconds: float,
    frame_rate: float | int = 30,
    *,
    drop_frame: bool = False,
) -> str:
    """Convert seconds to timecode string HH:MM:SS.FF."""

    codec = _timecode_codec_for_frame_rate(frame_rate, drop_frame=drop_frame)
    total_frames = codec.seconds_to_frames(seconds)
    return codec.format_timecode_from_frames(total_frames, frame_separator=".")


def build_ma2_xml(
    events: list[dict[str, Any]],
    frame_rate: float | int = 30,
    track_name: str = "EchoZero",
    *,
    drop_frame: bool = False,
) -> str:
    """Build a grandMA2 timecode XML string from event dicts.

    Each event dict has: time (float), label (str), cue (str optional).

    Returns:
        XML string in MA2 timecode format.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<MA2Timecode frameRate="{_format_frame_rate_for_xml(frame_rate)}" trackName="{track_name}">',
    ]

    for i, evt in enumerate(events):
        tc = seconds_to_timecode(
            evt["time"],
            frame_rate,
            drop_frame=drop_frame,
        )
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
        drop_frame = bool(settings.get("drop_frame", False))
        track_name = settings.get("track_name", "EchoZero")

        # Validate
        if not output_path:
            return err(ValidationError(
                f"Block '{block_id}' is missing required setting 'output_path'"
            ))
        try:
            resolved_frame_rate = _coerce_frame_rate(frame_rate)
            _timecode_codec_for_frame_rate(
                resolved_frame_rate,
                drop_frame=drop_frame,
            )
        except ValueError as exc:
            return err(ValidationError(str(exc)))

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
            frame_rate=resolved_frame_rate,
            track_name=track_name,
            drop_frame=drop_frame,
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
