"""
Callback-shaped graph values for audio engine v2.
Exists because immutable PreparedGraph data must be lowered before RT rendering.
Connects non-live planning to the offline renderer without touching v1 playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from echozero.application.audio_engine_v2.graph import (
    HardwareOutputRoute,
    MixParameters,
    PreparedGraph,
    RouteTarget,
    RouteTargetKind,
)


class RtNodeKind(Enum):
    """Render node kinds in a prepared RT graph."""

    TRACK = "track"
    BUS = "bus"


@dataclass(frozen=True, slots=True)
class RtRouteTarget:
    """One prevalidated route target addressed by RT graph indices."""

    bus_index: int | None = None
    hardware_output: HardwareOutputRoute | None = None

    @property
    def is_bus(self) -> bool:
        """Return whether this target routes to another render bus."""

        return self.bus_index is not None


@dataclass(frozen=True, slots=True)
class RtTrackNode:
    """One renderable track node with pre-resolved route targets."""

    track_id: str
    source_key: str
    channels: int
    mix: MixParameters
    route_targets: tuple[RtRouteTarget, ...]


@dataclass(frozen=True, slots=True)
class RtBusNode:
    """One renderable bus node with pre-resolved route targets."""

    bus_id: str
    mix: MixParameters
    route_targets: tuple[RtRouteTarget, ...]


@dataclass(frozen=True, slots=True)
class RtGraph:
    """A PreparedGraph lowered to index-addressed render nodes."""

    graph_id: str
    identity_full_hash: str
    tracks: tuple[RtTrackNode, ...]
    buses: tuple[RtBusNode, ...]
    bus_render_order: tuple[int, ...]
    master_bus_index: int


def prepare_rt_graph(graph: PreparedGraph) -> RtGraph:
    """Lower an immutable PreparedGraph into callback-shaped render data."""

    bus_indices = {bus.bus_id: index for index, bus in enumerate(graph.buses)}
    return RtGraph(
        graph_id=graph.graph_id,
        identity_full_hash=graph.identity.full_hash,
        tracks=tuple(
            RtTrackNode(
                track_id=track.track_id,
                source_key=track.source_key,
                channels=track.channels,
                mix=track.mix,
                route_targets=_prepare_route_targets(track.route.targets, bus_indices),
            )
            for track in graph.tracks
        ),
        buses=tuple(
            RtBusNode(
                bus_id=bus.bus_id,
                mix=bus.mix,
                route_targets=_prepare_route_targets(bus.route.targets, bus_indices),
            )
            for bus in graph.buses
        ),
        bus_render_order=_compute_bus_render_order(graph),
        master_bus_index=bus_indices[graph.master_bus_id],
    )


def _prepare_route_targets(
    targets: tuple[RouteTarget, ...],
    bus_indices: dict[str, int],
) -> tuple[RtRouteTarget, ...]:
    prepared: list[RtRouteTarget] = []
    for target in targets:
        kind = getattr(target, "kind")
        if kind is RouteTargetKind.BUS:
            prepared.append(RtRouteTarget(bus_index=bus_indices[str(target.bus_id)]))
        else:
            prepared.append(RtRouteTarget(hardware_output=target.hardware_output))
    return tuple(prepared)


def _compute_bus_render_order(graph: PreparedGraph) -> tuple[int, ...]:
    bus_indices = {bus.bus_id: index for index, bus in enumerate(graph.buses)}
    downstream = {
        bus.bus_id: tuple(bus_id for bus_id in bus.route.bus_ids if bus_id in bus_indices)
        for bus in graph.buses
    }
    depths = {
        bus.bus_id: _compute_downstream_depth(bus.bus_id, downstream=downstream)
        for bus in graph.buses
    }
    ordered_bus_ids = sorted(
        (bus.bus_id for bus in graph.buses),
        key=lambda bus_id: (-depths[bus_id], bus_indices[bus_id]),
    )
    return tuple(bus_indices[bus_id] for bus_id in ordered_bus_ids)


def _compute_downstream_depth(
    bus_id: str,
    *,
    downstream: dict[str, tuple[str, ...]],
) -> int:
    child_depths = [
        _compute_downstream_depth(next_bus_id, downstream=downstream)
        for next_bus_id in downstream[bus_id]
    ]
    return 0 if not child_depths else 1 + max(child_depths)
