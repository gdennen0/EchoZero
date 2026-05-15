"""
Compatibility mapping hooks for audio engine v2 graph planning.
Exists to translate current playback track plans into v2 prepared graphs without live wiring.
Connects v1 projection outputs to the additive v2 foundation for tests and future migration.
"""

from __future__ import annotations

from collections.abc import Iterable

from echozero.application.audio_engine_v2.graph import (
    MASTER_BUS_ID,
    HardwareOutputRoute,
    MixParameters,
    PreparedBus,
    PreparedGraph,
    PreparedTrack,
    TrackRoute,
)
from echozero.output_routing import (
    DEFAULT_STEREO_OUTPUT_BUS,
    MASTER_OUTPUT_BUS_TOKEN,
    NO_OUTPUT_BUS,
    parse_output_bus_spans,
)


def build_prepared_graph_from_playback_plan(
    playback_plan: object,
    *,
    graph_id: str,
    master_output_bus: object = DEFAULT_STEREO_OUTPUT_BUS,
) -> PreparedGraph:
    """Map a current PlaybackTrackPlan-like object into an additive v2 graph."""

    master_routes = _hardware_routes_from_tokens(master_output_bus)
    tracks = tuple(
        _prepared_track_from_playback_track(playback_track)
        for playback_track in tuple(getattr(playback_plan, "tracks", ()))
    )
    master_bus = PreparedBus(
        bus_id=MASTER_BUS_ID,
        name="Master",
        route=TrackRoute.to_hardware(master_routes)
        if master_routes
        else TrackRoute.no_output(),
    )
    return PreparedGraph(graph_id=graph_id, tracks=tracks, buses=(master_bus,))


def _prepared_track_from_playback_track(playback_track: object) -> PreparedTrack:
    output_bus = getattr(playback_track, "output_bus", None)
    return PreparedTrack(
        track_id=str(getattr(playback_track, "track_id")),
        name=str(getattr(playback_track, "name", getattr(playback_track, "track_id"))),
        source_key=str(getattr(playback_track, "source_key")),
        route=_track_route_from_output_bus(output_bus),
        mix=MixParameters(
            gain_db=float(getattr(playback_track, "gain_db", 0.0)),
            muted=bool(getattr(playback_track, "muted", False)),
        ),
        channels=_channels_from_buffer(playback_track),
        source_sample_rate=max(0, int(getattr(playback_track, "sample_rate", 0) or 0)),
    )


def _track_route_from_output_bus(output_bus: object) -> TrackRoute:
    if output_bus is None:
        return TrackRoute.to_master()
    tokens = _route_tokens(output_bus)
    if any(token in {NO_OUTPUT_BUS, "no_output", "off"} for token in tokens):
        return TrackRoute.no_output()
    master_requested = any(
        token in {MASTER_OUTPUT_BUS_TOKEN, "default"} for token in tokens
    )
    routes = _hardware_routes_from_tokens(output_bus)
    if master_requested and routes:
        return TrackRoute.to_master_and_hardware(routes)
    if master_requested:
        return TrackRoute.to_master()
    if routes:
        return TrackRoute.to_hardware(routes)
    return TrackRoute.to_master()


def _hardware_routes_from_tokens(value: object) -> tuple[HardwareOutputRoute, ...]:
    spans = parse_output_bus_spans(value, reject_invalid=False)
    return tuple(
        HardwareOutputRoute(first_channel=start_channel, last_channel=end_channel)
        for start_channel, end_channel in spans
    )


def _route_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(
            token.strip().lower()
            for token in value.split(",")
            if token.strip()
        )
    return (str(value).strip().lower(),)


def _channels_from_buffer(playback_track: object) -> int:
    buffer = getattr(playback_track, "buffer", None)
    shape = getattr(buffer, "shape", None)
    if isinstance(shape, Iterable):
        shape_tuple = tuple(shape)
        if len(shape_tuple) >= 2:
            return max(1, int(shape_tuple[1]))
    return 1
