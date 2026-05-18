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
DEFAULT_MASTER_OUTPUT_BUS = DEFAULT_STEREO_OUTPUT_BUS
MASTER_OUTPUT_BUS_TOKEN = "master"
NO_OUTPUT_BUS = "none"

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


def parse_output_bus_spans(
    value: object, *, reject_invalid: bool = False
) -> tuple[tuple[int, int], ...]:
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
    """Canonicalize explicit layer route tokens.

    ``None`` keeps the caller's default behavior. ``"master"`` mirrors to the
    current master output buses, while ``"none"`` is an explicit no-output route.
    Comma-separated route payloads are preserved so one layer can fan out to multiple
    physical outputs.
    """

    raw_tokens = _raw_tokens(value)
    master_requested = any(_is_master_output_bus(token) for token in raw_tokens)
    no_output_requested = any(_is_no_output_bus(token) for token in raw_tokens)
    route_tokens = [
        token
        for token in raw_tokens
        if not _is_master_output_bus(token) and not _is_no_output_bus(token)
    ]

    routes = _parse_routes(route_tokens, reject_invalid=reject_invalid)
    tokens: list[str] = []
    seen: set[str] = set()
    for route in routes:
        if _route_fits(route, max_channel=max_channel):
            if route.token not in seen:
                tokens.append(route.token)
                seen.add(route.token)
            continue
        if reject_invalid and not clamp_to_channels:
            raise ValueError(f"Output bus exceeds available channels: {route.token}")
    if master_requested and tokens:
        return ",".join((MASTER_OUTPUT_BUS_TOKEN, *tokens))
    if tokens:
        return ",".join(tokens)
    if no_output_requested and not master_requested:
        return NO_OUTPUT_BUS
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

    routes = _collapse_adjacent_single_master_routes(
        _parse_routes(value, reject_invalid=reject_invalid)
    )
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


def _collapse_adjacent_single_master_routes(
    routes: tuple[OutputBusRoute, ...],
) -> tuple[OutputBusRoute, ...]:
    by_token = {route.token: route for route in routes}
    collapsed: list[OutputBusRoute] = []
    consumed: set[str] = set()
    for route in routes:
        if route.token in consumed:
            continue
        if route.start_channel == route.end_channel and route.start_channel % 2 == 1:
            next_token = f"outputs_{route.start_channel + 1}_{route.start_channel + 1}"
            next_route = by_token.get(next_token)
            if next_route is not None:
                pair = parse_output_bus_token(
                    f"outputs_{route.start_channel}_{route.start_channel + 1}"
                )
                if pair is not None:
                    collapsed.append(pair)
                    consumed.add(route.token)
                    consumed.add(next_route.token)
                    continue
        collapsed.append(route)
        consumed.add(route.token)
    return tuple(collapsed)


def output_bus_options(
    channel_count: int,
    *,
    include_stereo_pairs: bool = False,
) -> tuple[OutputBusRoute, ...]:
    """Return physical output choices for the configured output count."""

    resolved_count = max(1, min(MAX_OUTPUT_CHANNELS, int(channel_count or 0)))
    routes = [
        OutputBusRoute(
            token=f"outputs_{channel}_{channel}",
            label=f"Output {channel}",
            start_channel=channel,
            end_channel=channel,
        )
        for channel in range(1, resolved_count + 1)
    ]
    if include_stereo_pairs:
        routes.extend(
            OutputBusRoute(
                token=f"outputs_{channel}_{channel + 1}",
                label=f"Outputs {channel}-{channel + 1}",
                start_channel=channel,
                end_channel=channel + 1,
            )
            for channel in range(1, resolved_count, 2)
        )
    return tuple(routes)


def output_bus_label(value: object) -> str:
    """Return a compact user-facing label for one or more output routes."""

    raw_tokens = _raw_tokens(value)
    if any(_is_no_output_bus(token) for token in raw_tokens):
        return "No Output"
    labels: list[str] = []
    if any(_is_master_output_bus(token) for token in raw_tokens):
        labels.append("Master Output")
    route_tokens = [token for token in raw_tokens if not _is_master_output_bus(token)]
    routes = _parse_routes(route_tokens, reject_invalid=False)
    if not routes and not labels:
        return "Master Output"
    labels.extend(
        _format_output_bus_label(route.start_channel, route.end_channel) for route in routes
    )
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
    raw_tokens = _raw_tokens(value)
    if any(_is_no_output_bus(token) for token in raw_tokens):
        return ()
    default_tokens = tuple(default_output_buses or ())
    include_default = not raw_tokens or any(_is_master_output_bus(token) for token in raw_tokens)
    route_tokens = [token for token in raw_tokens if not _is_master_output_bus(token)]
    routes = _parse_routes(route_tokens, reject_invalid=False)
    if include_default:
        routes = (
            *_parse_routes(
                ",".join(default_tokens) if default_tokens else DEFAULT_STEREO_OUTPUT_BUS,
                reject_invalid=False,
            ),
            *routes,
        )
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for route in routes:
        if route.start_channel > resolved_channels:
            continue
        end_channel = min(route.end_channel, resolved_channels)
        width = max(0, end_channel - route.start_channel + 1)
        span = (route.start_channel - 1, width)
        if width > 0 and span not in seen:
            spans.append(span)
            seen.add(span)
    return tuple(spans)


def _is_master_output_bus(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {MASTER_OUTPUT_BUS_TOKEN, "default"}


def _is_no_output_bus(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {NO_OUTPUT_BUS, "no_output", "off"}


def _raw_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [str(token or "").strip() for token in value.split(",") if str(token or "").strip()]
    if isinstance(value, Iterable):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value).strip()
    return [text] if text else []


def _parse_routes(value: object, *, reject_invalid: bool) -> tuple[OutputBusRoute, ...]:
    raw_tokens = _raw_tokens(value)
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
    "MASTER_OUTPUT_BUS_TOKEN",
    "MAX_OUTPUT_CHANNELS",
    "NO_OUTPUT_BUS",
    "OutputBusRoute",
    "canonical_layer_output_bus",
    "canonical_master_output_buses",
    "output_bus_channel_spans",
    "output_bus_label",
    "output_bus_options",
    "parse_output_bus_spans",
    "parse_output_bus_token",
]
