"""
Playback track identity helpers for runtime sync classification.
Exists because widget-side diffing and runtime track planning must share one notion of source identity.
Connects presentation/event fields to stable playback signatures without decoding audio.
"""

from __future__ import annotations

from collections.abc import Sequence


def normalize_output_bus(value: object) -> str | None:
    """Normalize one output-bus token to the canonical lowercase form."""

    if not isinstance(value, str):
        return None
    output_bus = value.strip().lower()
    return output_bus or None


def parse_output_bus_token(output_bus: str | None) -> tuple[int, int] | None:
    """Parse one `outputs_X_Y` token into one inclusive channel span."""

    token = normalize_output_bus(output_bus)
    if token is None or not token.startswith("outputs_"):
        return None
    parts = token.split("_")
    if len(parts) != 3 or (not parts[1].isdigit()) or (not parts[2].isdigit()):
        return None
    start_channel = int(parts[1])
    end_channel = int(parts[2])
    if start_channel < 1 or end_channel < start_channel:
        return None
    return start_channel, end_channel


def sanitize_output_bus_for_channels(
    value: object,
    *,
    playback_output_channels: int,
) -> str | None:
    """Return one route token only when it fits within the active output width."""

    parsed = parse_output_bus_token(normalize_output_bus(value))
    if parsed is None:
        return None
    start_channel, end_channel = parsed
    channel_count = max(1, int(playback_output_channels))
    if start_channel > channel_count or end_channel > channel_count:
        return None
    return f"outputs_{start_channel}_{end_channel}"


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

