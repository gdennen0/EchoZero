"""
Prepared audio graph models for engine v2.
Exists because DAW-style playback needs explicit tracks, buses, master, and hardware routes.
Connects non-real-time planning to future real-time graph commits through immutable values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json

MASTER_BUS_ID = "master"


class RouteTargetKind(Enum):
    """Kinds of downstream route targets supported by v2 graph planning."""

    BUS = "bus"
    HARDWARE = "hardware"


@dataclass(frozen=True, slots=True)
class HardwareOutputRoute:
    """One explicit 1-based inclusive physical output channel span."""

    first_channel: int
    last_channel: int

    def __post_init__(self) -> None:
        if self.first_channel < 1:
            raise ValueError("Hardware output routes must start at channel 1 or greater.")
        if self.last_channel < self.first_channel:
            raise ValueError("Hardware output routes must end at or after the first channel.")

    @property
    def token(self) -> str:
        """Return the canonical `outputs_X_Y` route token."""

        return f"outputs_{self.first_channel}_{self.last_channel}"


@dataclass(frozen=True, slots=True)
class MixParameters:
    """Sample-boundary mix values owned by a track or bus."""

    gain_db: float = 0.0
    pan: float = 0.0
    muted: bool = False
    soloed: bool = False


@dataclass(frozen=True, slots=True)
class RouteTarget:
    """One downstream target in a DAW-style multi-target route."""

    kind: RouteTargetKind
    bus_id: str | None = None
    hardware_output: HardwareOutputRoute | None = None

    def __post_init__(self) -> None:
        if self.kind is RouteTargetKind.BUS:
            if not str(self.bus_id or "").strip():
                raise ValueError("Bus route target id cannot be blank.")
            if self.hardware_output is not None:
                raise ValueError("Bus route targets cannot carry hardware output spans.")
        if self.kind is RouteTargetKind.HARDWARE:
            if self.hardware_output is None:
                raise ValueError("Hardware route targets require an output span.")
            if self.bus_id is not None:
                raise ValueError("Hardware route targets cannot carry bus ids.")

    @classmethod
    def bus(cls, bus_id: str) -> RouteTarget:
        """Build one bus route target."""

        return cls(kind=RouteTargetKind.BUS, bus_id=str(bus_id).strip())

    @classmethod
    def hardware(cls, route: HardwareOutputRoute) -> RouteTarget:
        """Build one hardware route target."""

        return cls(kind=RouteTargetKind.HARDWARE, hardware_output=route)

    def identity_payload(self) -> dict[str, object]:
        """Return deterministic route-target identity data."""

        if self.kind is RouteTargetKind.BUS:
            return {"kind": self.kind.value, "bus_id": self.bus_id}
        if self.hardware_output is None:
            raise ValueError("Hardware route target is missing an output span.")
        return {"kind": self.kind.value, "hardware_output": self.hardware_output.token}


@dataclass(frozen=True, slots=True)
class SignalRoute:
    """A multi-target route to buses, hardware outputs, or no output."""

    targets: tuple[RouteTarget, ...] = field(
        default_factory=lambda: (RouteTarget.bus(MASTER_BUS_ID),)
    )

    def __post_init__(self) -> None:
        deduped: list[RouteTarget] = []
        seen: set[tuple[str, str]] = set()
        for target in self.targets:
            key = _route_target_key(target)
            if key in seen:
                continue
            deduped.append(target)
            seen.add(key)
        if tuple(deduped) != self.targets:
            object.__setattr__(self, "targets", tuple(deduped))

    @classmethod
    def to_master(cls) -> SignalRoute:
        """Route through the master bus."""

        return cls(targets=(RouteTarget.bus(MASTER_BUS_ID),))

    @classmethod
    def to_bus(cls, bus_id: str) -> SignalRoute:
        """Route to one named bus."""

        return cls(targets=(RouteTarget.bus(bus_id),))

    @classmethod
    def to_hardware(cls, routes: tuple[HardwareOutputRoute, ...]) -> SignalRoute:
        """Route directly to physical outputs."""

        if not routes:
            raise ValueError("Direct hardware output routes must be explicit.")
        return cls(targets=tuple(RouteTarget.hardware(route) for route in routes))

    @classmethod
    def to_master_and_hardware(
        cls,
        routes: tuple[HardwareOutputRoute, ...],
    ) -> SignalRoute:
        """Route through master and also directly to physical outputs."""

        if not routes:
            raise ValueError("Master plus hardware routes must include hardware outputs.")
        return cls(
            targets=(
                RouteTarget.bus(MASTER_BUS_ID),
                *(RouteTarget.hardware(route) for route in routes),
            )
        )

    @classmethod
    def no_output(cls) -> SignalRoute:
        """Route to silence through an empty target list."""

        return cls(targets=())

    @property
    def bus_ids(self) -> tuple[str, ...]:
        """Return downstream bus ids in route order."""

        return tuple(
            str(target.bus_id)
            for target in self.targets
            if target.kind is RouteTargetKind.BUS and target.bus_id is not None
        )

    @property
    def hardware_outputs(self) -> tuple[HardwareOutputRoute, ...]:
        """Return downstream hardware output spans in route order."""

        return tuple(
            target.hardware_output
            for target in self.targets
            if target.kind is RouteTargetKind.HARDWARE and target.hardware_output is not None
        )

    def identity_payload(self) -> tuple[dict[str, object], ...]:
        """Return deterministic route identity data."""

        return tuple(target.identity_payload() for target in self.targets)


TrackRoute = SignalRoute


@dataclass(frozen=True, slots=True)
class PreparedTrack:
    """One immutable playback track prepared for graph planning."""

    track_id: str
    name: str
    source_key: str
    route: SignalRoute = field(default_factory=SignalRoute.to_master)
    mix: MixParameters = field(default_factory=MixParameters)
    channels: int = 2
    source_sample_rate: int = 0


@dataclass(frozen=True, slots=True)
class PreparedBus:
    """One immutable mix bus with explicit downstream route semantics."""

    bus_id: str
    name: str
    route: SignalRoute
    mix: MixParameters = field(default_factory=MixParameters)


@dataclass(frozen=True, slots=True)
class GraphIdentity:
    """Deterministic identity tokens for prepared graph comparison."""

    structural_hash: str
    route_hash: str
    mix_hash: str
    full_hash: str


@dataclass(frozen=True, slots=True)
class PreparedGraph:
    """One immutable non-real-time graph generation ready for RT preparation."""

    graph_id: str
    tracks: tuple[PreparedTrack, ...]
    buses: tuple[PreparedBus, ...]
    master_bus_id: str = MASTER_BUS_ID

    def __post_init__(self) -> None:
        bus_ids = {bus.bus_id for bus in self.buses}
        if self.master_bus_id not in bus_ids:
            raise ValueError("PreparedGraph must include the configured master bus.")
        for track in self.tracks:
            _validate_route_targets(
                track.route,
                bus_ids=bus_ids,
                owner_id=track.track_id,
                owner_kind="Track",
            )
        for bus in self.buses:
            _validate_route_targets(
                bus.route,
                bus_ids=bus_ids,
                owner_id=bus.bus_id,
                owner_kind="Bus",
                reject_bus_id=bus.bus_id,
            )
        _validate_bus_routes_are_acyclic(self.buses)

    @property
    def identity(self) -> GraphIdentity:
        """Return deterministic hashes for structure, route, mix, and full graph state."""

        return compute_graph_identity(self)


def replace_track_mix(
    graph: PreparedGraph,
    *,
    track_id: str,
    mix: MixParameters,
) -> PreparedGraph:
    """Return a new graph with one track's mix parameters replaced."""

    matched = False
    replaced_tracks: list[PreparedTrack] = []
    for track in graph.tracks:
        if track.track_id == track_id:
            matched = True
            replaced_tracks.append(replace(track, mix=mix))
        else:
            replaced_tracks.append(track)
    if not matched:
        raise ValueError(f"Track not found: {track_id}")
    return replace(graph, tracks=tuple(replaced_tracks))


def replace_track_route(
    graph: PreparedGraph,
    *,
    track_id: str,
    route: SignalRoute,
) -> PreparedGraph:
    """Return a new graph with one track's route replaced."""

    matched = False
    replaced_tracks: list[PreparedTrack] = []
    for track in graph.tracks:
        if track.track_id == track_id:
            matched = True
            replaced_tracks.append(replace(track, route=route))
        else:
            replaced_tracks.append(track)
    if not matched:
        raise ValueError(f"Track not found: {track_id}")
    return replace(graph, tracks=tuple(replaced_tracks))


def compute_graph_identity(graph: PreparedGraph) -> GraphIdentity:
    """Compute deterministic graph identity hashes from canonical JSON payloads."""

    structural_payload = {
        "graph_id": graph.graph_id,
        "tracks": [
            {
                "track_id": track.track_id,
                "name": track.name,
                "source_key": track.source_key,
                "channels": track.channels,
                "source_sample_rate": track.source_sample_rate,
            }
            for track in graph.tracks
        ],
        "buses": [{"bus_id": bus.bus_id, "name": bus.name} for bus in graph.buses],
        "master_bus_id": graph.master_bus_id,
    }
    route_payload = {
        "tracks": [
            {
                "track_id": track.track_id,
                "targets": track.route.identity_payload(),
            }
            for track in graph.tracks
        ],
        "buses": [
            {
                "bus_id": bus.bus_id,
                "targets": bus.route.identity_payload(),
            }
            for bus in graph.buses
        ],
    }
    mix_payload = {
        "tracks": [
            {"track_id": track.track_id, "mix": _mix_payload(track.mix)}
            for track in graph.tracks
        ],
        "buses": [
            {"bus_id": bus.bus_id, "mix": _mix_payload(bus.mix)} for bus in graph.buses
        ],
    }
    full_payload = {
        "structural": structural_payload,
        "route": route_payload,
        "mix": mix_payload,
    }
    return GraphIdentity(
        structural_hash=_hash_payload(structural_payload),
        route_hash=_hash_payload(route_payload),
        mix_hash=_hash_payload(mix_payload),
        full_hash=_hash_payload(full_payload),
    )


def _mix_payload(mix: MixParameters) -> dict[str, object]:
    return {
        "gain_db": round(float(mix.gain_db), 9),
        "pan": round(float(mix.pan), 9),
        "muted": bool(mix.muted),
        "soloed": bool(mix.soloed),
    }


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _route_target_key(target: RouteTarget) -> tuple[str, str]:
    if target.kind is RouteTargetKind.BUS:
        return target.kind.value, str(target.bus_id)
    if target.hardware_output is None:
        raise ValueError("Hardware route target is missing an output span.")
    return target.kind.value, target.hardware_output.token


def _validate_route_targets(
    route: SignalRoute,
    *,
    bus_ids: set[str],
    owner_id: str,
    owner_kind: str,
    reject_bus_id: str | None = None,
) -> None:
    for bus_id in route.bus_ids:
        if bus_id not in bus_ids:
            raise ValueError(f"{owner_kind} '{owner_id}' routes to missing bus.")
        if reject_bus_id is not None and bus_id == reject_bus_id:
            raise ValueError(f"Bus '{owner_id}' cannot route to itself.")


def _validate_bus_routes_are_acyclic(buses: tuple[PreparedBus, ...]) -> None:
    downstream = {bus.bus_id: set(bus.route.bus_ids) for bus in buses}
    for bus_id in downstream:
        _visit_bus_route(bus_id, downstream=downstream, active=(), visited=set())


def _visit_bus_route(
    bus_id: str,
    *,
    downstream: dict[str, set[str]],
    active: tuple[str, ...],
    visited: set[str],
) -> None:
    if bus_id in active:
        cycle = " -> ".join((*active, bus_id))
        raise ValueError(f"Bus route cycle detected: {cycle}")
    if bus_id in visited:
        return
    visited.add(bus_id)
    for next_bus_id in downstream.get(bus_id, set()):
        _visit_bus_route(
            next_bus_id,
            downstream=downstream,
            active=(*active, bus_id),
            visited=visited,
        )
