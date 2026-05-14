"""
Audio output routing token helpers.
Exists because UI, settings, timeline intents, and mixer code must share one
canonical representation for physical output routes.
Connects user-facing output choices to engine channel spans without importing UI.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

MAX_OUTPUT_CHANNELS = 16
DEFAULT_STEREO_OUTPUT_BUS = "outputs_1_2"
DEFAULT_MASTER_OUTPUT_BUS = "outputs_1_1"

_OUTPUT_BUS_RE = re.compile(r"^outputs_(\d+)_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class OutputBusRoute:
    token: str
    label: str
    start_channel: int
    end_channel: int


def parse_output_bus_token(value: object) -> OutputBusRoute | None:
    """Parse one `outputs_<start>_<end>` token using 1-based inclusive channels."""

    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    match = _OUTPUT_BUS_RE.match(text)
    if match is None:
        return None
    start_channel = int(match.group(1))
    end_channel = int(match.group(2))
    if start_channel < 1 or end_channel < start_channel:
        return None
    token = f"outputs_{start_channel}_{end_channel}"
    return OutputBusRoute(
        token=token,
        label=_format_output_bus_label(start_channel, end_channel),
        start_channel=start_channel,
        end_channel=end_channel,
    )


def parse_output_bus_spans(value: object, *, reject_invalid: bool = False) -> tuple[tuple[int, int], ...]:
    """Return 1-based inclusive channel spans parsed from a comma-separated route list."""

    routes = _parse_routes(value, reject_invalid=reject_invalid)
    return tuple((route.start_channel, route.end_channel) for route in routes)


def canonical_layer_output_bus(
    value: object,
    *,
    default: str | None = DEFAULT_STEREO_OUTPUT_BUS,
    max_channel: int | None = None,
    clamp_to_channels: bool = False,
    reject_invalid: bool = False,
) -> str | None:
    """Canonicalize one explicit layer route token.

    Layer routes are intentionally single-token. If a comma-separated value is supplied,
    the first valid token is used so older multi-route payloads degrade safely.
    """

    routes = _parse_routes(value, reject_invalid=reject_invalid)
    for route in routes:
        if _route_fits(route, max_channel=max_channel):
            return route.token
        if reject_invalid and not clamp_to_channels:
            raise ValueError(f"Output bus exceeds available channels: {route.token}")
    if reject_invalid:
        return None
    return default


def canonical_master_output_buses(
    value: object,
    *,
    default: str | None = DEFAULT_MASTER_OUTPUT_BUS,
    max_channel: int | None = None,
    clamp_to_channels: bool = False,
    reject_invalid: bool = False,
) -> tuple[str, ...]:
    """Canonicalize a comma-separated set of master mirror output routes."""

    routes = _parse_routes(value, reject_invalid=reject_invalid)
    tokens: list[str] = []
    seen: set[str] = set()
    for route in routes:
        if not _route_fits(route, max_channel=max_channel):
            if clamp_to_channels:
                continue
            if reject_invalid:
                raise ValueError(f"Output bus exceeds available channels: {route.token}")
        if route.token not in seen:
            tokens.append(route.token)
            seen.add(route.token)
    if tokens:
        return tuple(tokens)
    if default is None:
        return ()
    fallback = _parse_routes(default, reject_invalid=False)
    return tuple(route.token for route in fallback) or (DEFAULT_MASTER_OUTPUT_BUS,)


def output_bus_options(channel_count: int) -> tuple[OutputBusRoute, ...]:
    """Return single-physical-output choices for the configured output count."""

    resolved_count = max(1, min(MAX_OUTPUT_CHANNELS, int(channel_count or 0)))
    return tuple(
        OutputBusRoute(
            token=f"outputs_{channel}_{channel}",
            label=f"Output {channel}",
            start_channel=channel,
            end_channel=channel,
        )
        for channel in range(1, resolved_count + 1)
    )


def output_bus_label(value: object) -> str:
    """Return a compact user-facing label for one or more output routes."""

    routes = _parse_routes(value, reject_invalid=False)
    if not routes:
        return "Master Output"
    labels = [
        _format_output_bus_label(route.start_channel, route.end_channel) for route in routes
    ]
    return ", ".join(labels)


def _format_output_bus_label(start_channel: int, end_channel: int) -> str:
    if start_channel == end_channel:
        return f"Output {start_channel}"
    return f"Outputs {start_channel}-{end_channel}"


def output_bus_channel_spans(
    value: object,
    output_channels: int,
    *,
    default_output_buses: Iterable[str] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Convert route tokens to zero-based `(start, width)` spans clipped to output width."""

    resolved_channels = max(1, int(output_channels or 0))
    routes = _parse_routes(value, reject_invalid=False)
    if not routes:
        default_tokens = tuple(default_output_buses or ())
        routes = _parse_routes(
            ",".join(default_tokens) if default_tokens else DEFAULT_STEREO_OUTPUT_BUS,
            reject_invalid=False,
        )
    spans: list[tuple[int, int]] = []
    for route in routes:
        if route.start_channel > resolved_channels:
            continue
        end_channel = min(route.end_channel, resolved_channels)
        width = max(0, end_channel - route.start_channel + 1)
        if width > 0:
            spans.append((route.start_channel - 1, width))
    return tuple(spans)


def _parse_routes(value: object, *, reject_invalid: bool) -> tuple[OutputBusRoute, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_tokens = value.split(",")
    elif isinstance(value, Iterable):
        raw_tokens = [str(item) for item in value]
    else:
        raw_tokens = [str(value)]
    routes: list[OutputBusRoute] = []
    for raw_token in raw_tokens:
        text = str(raw_token or "").strip()
        if not text:
            continue
        route = parse_output_bus_token(text)
        if route is None:
            if reject_invalid:
                raise ValueError(f"Invalid output bus token: {text}")
            continue
        routes.append(route)
    return tuple(routes)


def _route_fits(route: OutputBusRoute, *, max_channel: int | None) -> bool:
    if max_channel is None:
        return route.end_channel <= MAX_OUTPUT_CHANNELS
    return route.end_channel <= max(1, int(max_channel))


__all__ = [
    "DEFAULT_MASTER_OUTPUT_BUS",
    "DEFAULT_STEREO_OUTPUT_BUS",
    "MAX_OUTPUT_CHANNELS",
    "OutputBusRoute",
    "canonical_layer_output_bus",
    "canonical_master_output_buses",
    "output_bus_channel_spans",
    "output_bus_label",
    "output_bus_options",
    "parse_output_bus_spans",
    "parse_output_bus_token",
]
