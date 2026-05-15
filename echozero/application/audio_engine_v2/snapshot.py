"""
Playback snapshot generation models for audio engine v2.
Exists because future playback must render exactly one committed immutable generation.
Connects prepared graphs and transport state to generation-aware runtime commits.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.audio_engine_v2.graph import GraphIdentity, PreparedGraph
from echozero.application.audio_engine_v2.transport import TransportState


@dataclass(frozen=True, slots=True)
class PlaybackSnapshotGeneration:
    """One immutable playback snapshot submitted, prepared, and later committed."""

    generation: int
    graph: PreparedGraph
    graph_identity: GraphIdentity
    transport: TransportState
    reason: str


def create_snapshot_generation(
    *,
    graph: PreparedGraph,
    transport: TransportState,
    previous: PlaybackSnapshotGeneration | None = None,
    reason: str = "initial",
) -> PlaybackSnapshotGeneration:
    """Create a new immutable snapshot generation from graph and transport values."""

    generation = 1 if previous is None else previous.generation + 1
    return PlaybackSnapshotGeneration(
        generation=generation,
        graph=graph,
        graph_identity=graph.identity,
        transport=transport,
        reason=reason,
    )
