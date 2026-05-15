"""
Prepared audio graph models for engine v2.
Exists because DAW-style playback needs explicit tracks, buses, master, and hardware routes.
Connects non-real-time planning to future real-time graph commits through immutable values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json

MASTER_BUS_ID = "master"


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
class TrackRoute:
    """A track output route to one bus, no output, or explicit hardware outputs."""

    bus_id: str | None = MASTER_BUS_ID
    hardware_outputs: tuple[HardwareOutputRoute, ...] = ()

    def __post_init__(self) -> None:
        if self.bus_id is not None and self.hardware_outputs:
            raise ValueError("A track route cannot target a bus and hardware outputs together.")

    @classmethod
    def to_master(cls) -> TrackRoute:
        """Route the track through the master bus."""

        return cls(bus_id=MASTER_BUS_ID)

    @classmethod
    def to_bus(cls, bus_id: str) -> TrackRoute:
        """Route the track through one named bus."""

        resolved_bus_id = str(bus_id).strip()
        if not resolved_bus_id:
            raise ValueError("Bus route id cannot be blank.")
        return cls(bus_id=resolved_bus_id)

    @classmethod
    def to_hardware(cls, routes: tuple[HardwareOutputRoute, ...]) -> TrackRoute:
        """Route the track directly to physical outputs."""

        if not routes:
            raise ValueError("Direct hardware output routes must be explicit.")
        return cls(bus_id=None, hardware_outputs=tuple(routes))

    @classmethod
    def no_output(cls) -> TrackRoute:
        """Route the track to silence through an empty output route."""

        return cls(bus_id=None, hardware_outputs=())


@dataclass(frozen=True, slots=True)
class PreparedTrack:
    """One immutable playback track prepared for graph planning."""

    track_id: str
    name: str
    source_key: str
    route: TrackRoute = field(default_factory=TrackRoute.to_master)
    mix: MixParameters = field(default_factory=MixParameters)
    channels: int = 2
    source_sample_rate: int = 0


@dataclass(frozen=True, slots=True)
class PreparedBus:
    """One immutable mix bus with explicit downstream route semantics."""

    bus_id: str
    name: str
    output_routes: tuple[HardwareOutputRoute, ...]
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
            if track.route.bus_id is not None and track.route.bus_id not in bus_ids:
                raise ValueError(f"Track '{track.track_id}' routes to missing bus.")

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

    tracks = tuple(
        replace(track, mix=mix) if track.track_id == track_id else track
        for track in graph.tracks
    )
    if tracks == graph.tracks:
        raise ValueError(f"Track not found: {track_id}")
    return replace(graph, tracks=tracks)


def replace_track_route(
    graph: PreparedGraph,
    *,
    track_id: str,
    route: TrackRoute,
) -> PreparedGraph:
    """Return a new graph with one track's route replaced."""

    tracks = tuple(
        replace(track, route=route) if track.track_id == track_id else track
        for track in graph.tracks
    )
    if tracks == graph.tracks:
        raise ValueError(f"Track not found: {track_id}")
    return replace(graph, tracks=tracks)


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
                "bus_id": track.route.bus_id,
                "hardware_outputs": [
                    route.token for route in track.route.hardware_outputs
                ],
            }
            for track in graph.tracks
        ],
        "buses": [
            {
                "bus_id": bus.bus_id,
                "output_routes": [route.token for route in bus.output_routes],
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
