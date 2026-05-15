"""
Audio engine v2 foundation package.
Exists to model the next DAW-style backend without replacing the live v1 runtime.
Connects playback planning to future real-time graph preparation through immutable data.
"""

from echozero.application.audio_engine_v2.graph import (
    MASTER_BUS_ID,
    GraphIdentity,
    HardwareOutputRoute,
    MixParameters,
    PreparedBus,
    PreparedGraph,
    PreparedTrack,
    RouteTarget,
    RouteTargetKind,
    SignalRoute,
    TrackRoute,
    compute_graph_identity,
    replace_track_mix,
    replace_track_route,
)
from echozero.application.audio_engine_v2.snapshot import (
    PlaybackSnapshotGeneration,
    create_snapshot_generation,
)
from echozero.application.audio_engine_v2.transport import (
    LoopRegion,
    TransportCommand,
    TransportCommandKind,
    TransportPlayState,
    TransportState,
    apply_transport_command,
)

__all__ = [
    "MASTER_BUS_ID",
    "GraphIdentity",
    "HardwareOutputRoute",
    "LoopRegion",
    "MixParameters",
    "PlaybackSnapshotGeneration",
    "PreparedBus",
    "PreparedGraph",
    "PreparedTrack",
    "RouteTarget",
    "RouteTargetKind",
    "SignalRoute",
    "TrackRoute",
    "TransportCommand",
    "TransportCommandKind",
    "TransportPlayState",
    "TransportState",
    "apply_transport_command",
    "compute_graph_identity",
    "create_snapshot_generation",
    "replace_track_mix",
    "replace_track_route",
]
