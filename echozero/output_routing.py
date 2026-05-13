"""Shared audio output-route parsing and presentation helpers.
Exists so app settings, layer actions, playback planning, and the mixer use one
vocabulary for `outputs_<start>_<end>` bus tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_OUTPUT_CHANNELS = 16
DEFAULT_MASTER_OUTPUT_BUS = "outputs_1_1"
DEFAULT_STEREO_OUTPUT_BUS = "outputs_1_2"


@dataclass(frozen=True, slots=True)
class OutputBusRoute:
    """One inclusive 1-based physical output-channel span."""

    start_channel: int
    end_channel: int

    @property
    def token(self) -> str:
        return f"outputs_{self.start_channel}_{self.end_channel}"

    @property
    def label(self) -> str:
        return output_bus_label(self.token)

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(range(self.start_channel, self.end_channel + 1))


def output_bus_label(output_bus: object) -> str:
    """Return the operator-facing label for one output bus token."""

    route = parse_output_bus_token(output_bus)
    if route is None:
        return str(output_bus or "").strip()
    if route.start_channel == route.end_channel:
        return f"Output {route.start_channel}"
    if route.end_channel == route.start_channel + 1:
        return f"Outputs {route.start_channel}/{route.end_channel}"
    return f"Outputs {route.start_channel}-{route.end_channel}"


def output_bus_options(channel_count: int) -> tuple[OutputBusRoute, ...]:
    """Return the standard single-physical-output route options."""

    bounded_channel_count = max(1, min(MAX_OUTPUT_CHANNELS, int(channel_count or 1)))
    return tuple(
        OutputBusRoute(channel, channel) for channel in range(1, bounded_channel_count + 1)
    )


def iter_output_bus_values(value: object) -> tuple[object, ...]:
    """Flatten comma-separated/list values into raw route token candidates."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split(","))
    if isinstance(value, (list, tuple, set)):
        raw_tokens: list[object] = []
        for item in value:
            if isinstance(item, str):
                raw_tokens.extend(item.split(","))
            else:
                raw_tokens.append(item)
        return tuple(raw_tokens)
    return (value,)


def parse_output_bus_token(value: object) -> OutputBusRoute | None:
    """Parse one `outputs_X_Y` token into an inclusive 1-based route span."""

    token = str(value or "").strip().lower()
    if not token.startswith("outputs_"):
        return None
    parts = token.split("_")
    if len(parts) != 3 or (not parts[1].isdigit()) or (not parts[2].isdigit()):
        return None
    start_channel = int(parts[1])
    end_channel = int(parts[2])
    if start_channel < 1 or end_channel < start_channel:
        return None
    return OutputBusRoute(start_channel, end_channel)


def canonical_output_bus_token(
    value: object,
    *,
    max_channel: int | None = MAX_OUTPUT_CHANNELS,
    clamp_to_channels: bool = False,
) -> str | None:
    """Canonicalize one route token, optionally bounding it to available channels."""

    route = parse_output_bus_token(value)
    if route is None:
        return None
    if max_channel is not None:
        bounded_max = max(1, min(MAX_OUTPUT_CHANNELS, int(max_channel or 1)))
        if route.start_channel > bounded_max:
            return None
        if route.end_channel > bounded_max:
            if not clamp_to_channels:
                return None
            route = OutputBusRoute(route.start_channel, bounded_max)
    elif route.end_channel > MAX_OUTPUT_CHANNELS:
        return None
    return route.token


def canonical_output_bus_tokens(
    value: object,
    *,
    max_channel: int | None = MAX_OUTPUT_CHANNELS,
    clamp_to_channels: bool = False,
    allow_multiple: bool = True,
    reject_invalid: bool = False,
) -> tuple[str, ...]:
    """Canonicalize, de-dupe, and bound route tokens from a scalar/list/comma value.

    When ``reject_invalid`` is true, any non-empty invalid token invalidates the whole
    value. When ``allow_multiple`` is false, the first valid token is returned; this
    preserves single-route layer compatibility if an older comma value is encountered.
    """

    resolved: list[str] = []
    seen: set[str] = set()
    saw_nonempty = False
    for raw_token in iter_output_bus_values(value):
        if not str(raw_token or "").strip():
            continue
        saw_nonempty = True
        canonical = canonical_output_bus_token(
            raw_token,
            max_channel=max_channel,
            clamp_to_channels=clamp_to_channels,
        )
        if canonical is None:
            if reject_invalid:
                return ()
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
        if not allow_multiple:
            break
    if reject_invalid and saw_nonempty and not resolved:
        return ()
    return tuple(resolved)


def canonical_master_output_buses(
    value: object,
    *,
    default: str | None = DEFAULT_MASTER_OUTPUT_BUS,
    max_channel: int | None = MAX_OUTPUT_CHANNELS,
    clamp_to_channels: bool = False,
    reject_invalid: bool = False,
) -> tuple[str, ...]:
    """Return canonical master mirror bus tokens.

    Master routing intentionally supports multiple mirrored buses, including the
    persisted comma-separated representation used by settings.
    """

    source = value if value is not None else default
    tokens = canonical_output_bus_tokens(
        source,
        max_channel=max_channel,
        clamp_to_channels=clamp_to_channels,
        allow_multiple=True,
        reject_invalid=reject_invalid,
    )
    if tokens or default is None:
        return tokens
    return canonical_output_bus_tokens(default, max_channel=max_channel, allow_multiple=True)


def canonical_layer_output_bus(
    value: object,
    *,
    max_channel: int | None = MAX_OUTPUT_CHANNELS,
    clamp_to_channels: bool = False,
    reject_invalid: bool = False,
) -> str | None:
    """Return the canonical single explicit layer route, or ``None`` for default.

    Layers/timecode routes are a single explicit bus. If a legacy comma value is
    seen, the first valid route is used so old transient state does not fan out.
    """

    tokens = canonical_output_bus_tokens(
        value,
        max_channel=max_channel,
        clamp_to_channels=clamp_to_channels,
        allow_multiple=False,
        reject_invalid=reject_invalid,
    )
    return tokens[0] if tokens else None


def parse_output_bus_spans(
    value: object,
    *,
    reject_invalid: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Parse route token(s) into ``(start_channel, end_channel)`` spans."""

    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_token in iter_output_bus_values(value):
        if not str(raw_token or "").strip():
            continue
        route = parse_output_bus_token(raw_token)
        if route is None:
            if reject_invalid:
                return ()
            continue
        span = (route.start_channel, route.end_channel)
        if span in seen:
            continue
        seen.add(span)
        spans.append(span)
    return tuple(spans)


def output_bus_channel_span(
    output_bus: object,
    output_channels: int,
) -> tuple[int, int] | None:
    """Resolve one route token to a zero-based ``(start, width)`` channel span.

    The route is clipped to the active output width. A route that starts beyond the
    active device width returns ``(-1, 0)`` so callers can distinguish an explicit
    out-of-range route from an unparsable token.
    """

    try:
        channel_count = int(output_channels)
    except (TypeError, ValueError):
        channel_count = 1
    if channel_count <= 1:
        return (0, 1)
    route = parse_output_bus_token(output_bus)
    if route is None:
        return None
    start = route.start_channel - 1
    if start >= channel_count:
        return (-1, 0)
    resolved_end = min(route.end_channel - 1, channel_count - 1)
    return (start, max(0, resolved_end - start + 1))


def output_bus_channel_spans(
    output_bus: object,
    output_channels: int,
    *,
    default_output_buses: object = None,
    allow_explicit_multiple: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Resolve explicit layer or default master routes to zero-based channel spans.

    Explicit layer/timecode routes are single-bus by default; pass
    ``allow_explicit_multiple=True`` only for callers that intentionally mirror an
    explicit value. A missing explicit value resolves all default/master buses.
    """

    try:
        channel_count = int(output_channels)
    except (TypeError, ValueError):
        channel_count = 1
    if channel_count <= 1:
        return ((0, 1),)

    if output_bus is not None:
        tokens = (
            canonical_output_bus_tokens(output_bus)
            if allow_explicit_multiple
            else tuple(
                token
                for token in (canonical_layer_output_bus(output_bus),)
                if token is not None
            )
        )
        if not tokens:
            tokens = (DEFAULT_STEREO_OUTPUT_BUS,)
        return _resolved_channel_spans_for_tokens(
            tokens,
            channel_count,
            empty_fallback=((-1, 0),),
        )

    default_tokens = canonical_master_output_buses(
        default_output_buses,
        default=DEFAULT_STEREO_OUTPUT_BUS,
    )
    return _resolved_channel_spans_for_tokens(
        default_tokens,
        channel_count,
        empty_fallback=((0, min(2, channel_count)),),
    )


def _resolved_channel_spans_for_tokens(
    tokens: tuple[str, ...],
    channel_count: int,
    *,
    empty_fallback: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in tokens:
        span = output_bus_channel_span(token, channel_count)
        if span is None or span[1] <= 0 or span in seen:
            continue
        seen.add(span)
        spans.append(span)
    return tuple(spans) if spans else empty_fallback
