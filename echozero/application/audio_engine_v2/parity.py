"""
Planner parity harness for audio engine v2 shadow graphs.
Exists because v2 must prove it can mirror current app playback planning before live use.
Connects PlaybackTrackPlan projections to deterministic PreparedGraph diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np

from echozero.application.audio_engine_v2.graph import (
    GraphIdentity,
    HardwareOutputRoute,
    PreparedGraph,
    PreparedTrack,
    RouteTargetKind,
)
from echozero.application.audio_engine_v2.mapping import (
    build_prepared_graph_from_playback_plan,
)
from echozero.output_routing import (
    DEFAULT_STEREO_OUTPUT_BUS,
    MASTER_OUTPUT_BUS_TOKEN,
    NO_OUTPUT_BUS,
    parse_output_bus_spans,
)


class GraphEditKind(Enum):
    """Coarse planner edit classes derived from v2 identity hashes."""

    UNCHANGED = "unchanged"
    MIX = "mix"
    ROUTE = "route"
    STRUCTURE = "structure"


class ParityPlanningError(ValueError):
    """Raised when a shadow parity graph cannot represent current playback planning."""


@dataclass(frozen=True, slots=True)
class GraphEditClassification:
    """Deterministic identity comparison between two prepared graphs."""

    kind: GraphEditKind
    structure_changed: bool
    route_changed: bool
    mix_changed: bool
    before: GraphIdentity
    after: GraphIdentity


@dataclass(frozen=True, slots=True)
class TrackRouteSummary:
    """Developer-facing summary for one planned playback track route and mix state."""

    track_id: str
    name: str
    source_key: str
    v1_output_bus: str | None
    v2_targets: tuple[str, ...]
    gain_db: float
    muted: bool
    soloed: bool
    mode: str


@dataclass(frozen=True, slots=True)
class GraphParitySummary:
    """Deterministic developer-facing summary of one shadow v2 graph."""

    graph_id: str
    identity: GraphIdentity
    tracks: tuple[TrackRouteSummary, ...]
    buses: tuple[tuple[str, tuple[str, ...]], ...]
    structure_signature: tuple[tuple[str, str], ...]
    route_signature: tuple[tuple[str, tuple[str, ...]], ...]
    mix_signature: tuple[tuple[str, float, bool, bool], ...]


@dataclass(frozen=True, slots=True)
class PlannerParityReport:
    """Prepared graph plus deterministic diagnostics for shadow parity comparisons."""

    graph: PreparedGraph
    summary: GraphParitySummary
    diagnostics: tuple[str, ...]


AudioLoader = Callable[[str | Path], tuple[np.ndarray, int]]


def build_shadow_graph_from_playback_projection(
    projection: object,
    *,
    audio_loader: AudioLoader,
    graph_id: str,
    master_output_bus: object = DEFAULT_STEREO_OUTPUT_BUS,
) -> PlannerParityReport:
    """Build a shadow v2 graph from the current app/runtime playback projection."""

    from echozero.application.playback.tracks import PlaybackTrackBuilder

    builder = PlaybackTrackBuilder(audio_loader)
    mix_plan = builder.build_mix_plan(cast(Any, projection))
    return build_shadow_graph_from_track_plan(
        mix_plan,
        graph_id=graph_id,
        master_output_bus=master_output_bus,
    )


def build_shadow_graph_from_track_plan(
    track_plan: object,
    *,
    graph_id: str,
    master_output_bus: object = DEFAULT_STEREO_OUTPUT_BUS,
) -> PlannerParityReport:
    """Build a shadow v2 graph from a current PlaybackTrackPlan-shaped object."""

    playback_tracks = tuple(getattr(track_plan, "tracks", ()))
    diagnostics = _diagnose_playback_plan(playback_tracks, master_output_bus=master_output_bus)
    graph = build_prepared_graph_from_playback_plan(
        track_plan,
        graph_id=graph_id,
        master_output_bus=master_output_bus,
    )
    summary = summarize_graph_parity(
        graph,
        playback_tracks=playback_tracks,
        graph_id=graph_id,
    )
    return PlannerParityReport(graph=graph, summary=summary, diagnostics=diagnostics)


def summarize_graph_parity(
    graph: PreparedGraph,
    *,
    playback_tracks: tuple[object, ...],
    graph_id: str | None = None,
) -> GraphParitySummary:
    """Summarize a shadow v2 graph in a deterministic comparison-friendly form."""

    playback_by_track_id = {
        str(getattr(playback_track, "track_id")): playback_track
        for playback_track in playback_tracks
    }
    track_summaries = tuple(
        _summarize_track(track, playback_by_track_id.get(track.track_id)) for track in graph.tracks
    )
    bus_summaries = tuple(
        (bus.bus_id, _route_target_tokens(bus.route.targets)) for bus in graph.buses
    )
    return GraphParitySummary(
        graph_id=graph_id or graph.graph_id,
        identity=graph.identity,
        tracks=track_summaries,
        buses=bus_summaries,
        structure_signature=tuple((track.track_id, track.source_key) for track in graph.tracks),
        route_signature=tuple(
            (track.track_id, summary.v2_targets)
            for track, summary in zip(graph.tracks, track_summaries, strict=True)
        ),
        mix_signature=tuple(
            (
                track.track_id,
                round(float(track.mix.gain_db), 9),
                bool(track.mix.muted),
                bool(track.mix.soloed),
            )
            for track in graph.tracks
        ),
    )


def classify_graph_edit(
    before: PreparedGraph | GraphIdentity,
    after: PreparedGraph | GraphIdentity,
) -> GraphEditClassification:
    """Classify graph identity changes as structure, route, mix, or unchanged."""

    before_identity = before.identity if isinstance(before, PreparedGraph) else before
    after_identity = after.identity if isinstance(after, PreparedGraph) else after
    structure_changed = before_identity.structural_hash != after_identity.structural_hash
    route_changed = before_identity.route_hash != after_identity.route_hash
    mix_changed = before_identity.mix_hash != after_identity.mix_hash
    if structure_changed:
        kind = GraphEditKind.STRUCTURE
    elif route_changed:
        kind = GraphEditKind.ROUTE
    elif mix_changed:
        kind = GraphEditKind.MIX
    else:
        kind = GraphEditKind.UNCHANGED
    return GraphEditClassification(
        kind=kind,
        structure_changed=structure_changed,
        route_changed=route_changed,
        mix_changed=mix_changed,
        before=before_identity,
        after=after_identity,
    )


def _summarize_track(
    track: PreparedTrack,
    playback_track: object | None,
) -> TrackRouteSummary:
    mode = getattr(playback_track, "mode", None)
    return TrackRouteSummary(
        track_id=track.track_id,
        name=track.name,
        source_key=track.source_key,
        v1_output_bus=(
            str(getattr(playback_track, "output_bus"))
            if playback_track is not None and getattr(playback_track, "output_bus", None)
            else None
        ),
        v2_targets=_route_target_tokens(track.route.targets),
        gain_db=round(float(track.mix.gain_db), 9),
        muted=bool(track.mix.muted),
        soloed=bool(track.mix.soloed),
        mode=str(getattr(mode, "value", mode or "")),
    )


def _route_target_tokens(targets: tuple[object, ...]) -> tuple[str, ...]:
    tokens: list[str] = []
    for target in targets:
        if getattr(target, "kind", None) is RouteTargetKind.BUS:
            tokens.append(str(getattr(target, "bus_id")))
            continue
        hardware_output = getattr(target, "hardware_output", None)
        if isinstance(hardware_output, HardwareOutputRoute):
            tokens.append(hardware_output.token)
    return tuple(tokens)


def _diagnose_playback_plan(
    playback_tracks: tuple[object, ...],
    *,
    master_output_bus: object,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if not playback_tracks:
        diagnostics.append("empty-track-plan")
    _require_known_route_tokens(master_output_bus, owner="master")
    for playback_track in playback_tracks:
        output_bus = getattr(playback_track, "output_bus", None)
        _require_known_route_tokens(
            output_bus,
            owner=f"track:{getattr(playback_track, 'track_id', '<unknown>')}",
        )
    return tuple(diagnostics)


def _require_known_route_tokens(value: object, *, owner: str) -> None:
    tokens = _route_tokens(value)
    unknown_tokens = [
        token
        for token in tokens
        if token not in {MASTER_OUTPUT_BUS_TOKEN, "default", NO_OUTPUT_BUS, "no_output", "off"}
        and not parse_output_bus_spans(token, reject_invalid=False)
    ]
    if unknown_tokens:
        joined_tokens = ",".join(unknown_tokens)
        raise ParityPlanningError(f"{owner} has unsupported output route token: {joined_tokens}")


def _route_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(token.strip().lower() for token in value.split(",") if token.strip())
    if isinstance(value, Iterable):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return (str(value).strip().lower(),)


__all__ = [
    "GraphEditClassification",
    "GraphEditKind",
    "GraphParitySummary",
    "ParityPlanningError",
    "PlannerParityReport",
    "TrackRouteSummary",
    "build_shadow_graph_from_playback_projection",
    "build_shadow_graph_from_track_plan",
    "classify_graph_edit",
    "summarize_graph_parity",
]
