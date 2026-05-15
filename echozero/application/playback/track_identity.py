"""
Playback track identity helpers for runtime sync classification.
Exists because widget-side diffing and runtime track planning must share one notion of source identity.
Connects presentation/event fields to stable playback signatures without decoding audio.
"""

from __future__ import annotations

from collections.abc import Sequence

from echozero.output_routing import (
    NO_OUTPUT_BUS,
    canonical_layer_output_bus,
    parse_output_bus_token as _parse_output_bus_route,
)


def normalize_output_bus(value: object) -> str | None:
    """Normalize one output-bus token to the canonical lowercase form."""

    if not isinstance(value, str):
        return None
    output_bus = value.strip().lower()
    return output_bus or None


def parse_output_bus_token(output_bus: str | None) -> tuple[int, int] | None:
    """Parse one `outputs_X_Y` token into one inclusive channel span."""

    route = _parse_output_bus_route(output_bus)
    if route is None:
        return None
    return route.start_channel, route.end_channel


def sanitize_output_bus_for_channels(
    value: object,
    *,
    playback_output_channels: int,
) -> str | None:
    """Return one explicit layer route when it fits within the active output width."""

    output_bus = canonical_layer_output_bus(
        value,
        max_channel=max(1, int(playback_output_channels)),
        clamp_to_channels=True,
        reject_invalid=True,
    )
    if output_bus is None and normalize_output_bus(value) is not None:
        return NO_OUTPUT_BUS
    return output_bus


def event_slice_signature(events: Sequence[object]) -> str:
    """Build one stable event-slice signature from playback-relevant event fields."""

    tokens: list[str] = []
    for event in events:
        try:
            start_seconds = float(getattr(event, "start", 0.0))
        except (TypeError, ValueError):
            start_seconds = 0.0
        muted = int(bool(getattr(event, "muted", False)))
        badges = getattr(event, "badges", ())
        demoted = 0
        if isinstance(badges, (list, tuple, set)):
            demoted = int("demoted" in {str(badge) for badge in badges})
        tokens.append(f"{start_seconds:.6f}:{muted}:{demoted}")
    return ",".join(tokens)
