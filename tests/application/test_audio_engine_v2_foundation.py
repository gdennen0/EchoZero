"""Audio engine v2 foundation invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from echozero.application.audio_engine_v2.graph import (
    MASTER_BUS_ID,
    HardwareOutputRoute,
    MixParameters,
    PreparedBus,
    PreparedGraph,
    PreparedTrack,
    TrackRoute,
    replace_track_mix,
    replace_track_route,
)
from echozero.application.audio_engine_v2.mapping import (
    build_prepared_graph_from_playback_plan,
)
from echozero.application.audio_engine_v2.snapshot import create_snapshot_generation
from echozero.application.audio_engine_v2.transport import (
    LoopRegion,
    TransportCommand,
    TransportPlayState,
    TransportState,
    apply_transport_command,
)


def _graph() -> PreparedGraph:
    master = PreparedBus(
        bus_id=MASTER_BUS_ID,
        name="Master",
        output_routes=(HardwareOutputRoute(1, 2),),
    )
    track = PreparedTrack(track_id="drums", name="Drums", source_key="audio:drums.wav")
    return PreparedGraph(graph_id="song-a", tracks=(track,), buses=(master,))


def test_snapshot_generation_is_immutable_and_copy_on_write() -> None:
    graph = _graph()
    first = create_snapshot_generation(graph=graph, transport=TransportState())
    changed = replace_track_mix(
        graph,
        track_id="drums",
        mix=MixParameters(gain_db=-6.0),
    )
    second = create_snapshot_generation(
        graph=changed,
        transport=first.transport,
        previous=first,
        reason="gain edit",
    )

    with pytest.raises(FrozenInstanceError):
        first.graph.tracks[0].mix = MixParameters(gain_db=-12.0)  # type: ignore[misc]
    assert first.generation == 1
    assert second.generation == 2
    assert first.graph.tracks[0].mix.gain_db == 0.0
    assert second.graph.tracks[0].mix.gain_db == -6.0
    assert first.graph is not second.graph


def test_graph_identity_changes_by_structural_route_and_mix_edits() -> None:
    graph = _graph()
    structural = PreparedGraph(
        graph_id="song-a",
        tracks=(
            *graph.tracks,
            PreparedTrack(track_id="bass", name="Bass", source_key="audio:bass.wav"),
        ),
        buses=graph.buses,
    )
    routed = replace_track_route(
        graph,
        track_id="drums",
        route=TrackRoute.to_hardware((HardwareOutputRoute(3, 4),)),
    )
    mixed = replace_track_mix(
        graph,
        track_id="drums",
        mix=MixParameters(gain_db=-3.0),
    )

    assert structural.identity.structural_hash != graph.identity.structural_hash
    assert structural.identity.full_hash != graph.identity.full_hash
    assert routed.identity.route_hash != graph.identity.route_hash
    assert routed.identity.structural_hash == graph.identity.structural_hash
    assert routed.identity.full_hash != graph.identity.full_hash
    assert mixed.identity.mix_hash != graph.identity.mix_hash
    assert mixed.identity.structural_hash == graph.identity.structural_hash
    assert mixed.identity.route_hash == graph.identity.route_hash
    assert mixed.identity.full_hash != graph.identity.full_hash


def test_master_no_output_and_direct_hardware_routes_are_explicit() -> None:
    graph = _graph()
    no_output = replace_track_route(
        graph,
        track_id="drums",
        route=TrackRoute.no_output(),
    )
    direct = replace_track_route(
        graph,
        track_id="drums",
        route=TrackRoute.to_hardware((HardwareOutputRoute(5, 6),)),
    )

    assert graph.tracks[0].route.bus_id == MASTER_BUS_ID
    assert graph.buses[0].output_routes == (HardwareOutputRoute(1, 2),)
    assert no_output.tracks[0].route.bus_id is None
    assert no_output.tracks[0].route.hardware_outputs == ()
    assert direct.tracks[0].route.bus_id is None
    assert direct.tracks[0].route.hardware_outputs == (HardwareOutputRoute(5, 6),)


def test_playback_plan_mapping_preserves_route_semantics() -> None:
    playback_plan = SimpleNamespace(
        tracks=(
            SimpleNamespace(
                track_id="song",
                name="Song",
                source_key="audio:song.wav",
                gain_db=-1.0,
                muted=False,
                output_bus=None,
                sample_rate=44100,
            ),
            SimpleNamespace(
                track_id="click",
                name="Click",
                source_key="audio:click.wav",
                gain_db=0.0,
                muted=True,
                output_bus="none",
                sample_rate=44100,
            ),
            SimpleNamespace(
                track_id="ltc",
                name="LTC",
                source_key="audio:ltc.wav",
                gain_db=0.0,
                muted=False,
                output_bus="outputs_3_3",
                sample_rate=44100,
            ),
        )
    )

    graph = build_prepared_graph_from_playback_plan(
        playback_plan,
        graph_id="mapped",
        master_output_bus="outputs_1_2",
    )

    assert graph.tracks[0].route == TrackRoute.to_master()
    assert graph.tracks[1].route == TrackRoute.no_output()
    assert graph.tracks[1].mix.muted is True
    assert graph.tracks[2].route == TrackRoute.to_hardware((HardwareOutputRoute(3, 3),))
    assert graph.buses[0].output_routes == (HardwareOutputRoute(1, 2),)


def test_transport_commands_return_new_state_values() -> None:
    initial = TransportState()
    playing = apply_transport_command(initial, TransportCommand.play(1))
    seeked = apply_transport_command(playing, TransportCommand.seek(2, 12.5))
    looped = apply_transport_command(
        seeked,
        TransportCommand.set_loop(
            3,
            loop_region=LoopRegion(10.0, 20.0),
            enabled=True,
        ),
    )
    stopped = apply_transport_command(looped, TransportCommand.stop(4))

    assert initial.play_state is TransportPlayState.STOPPED
    assert playing.play_state is TransportPlayState.PLAYING
    assert seeked.position_seconds == 12.5
    assert looped.loop_enabled is True
    assert looped.loop_region == LoopRegion(10.0, 20.0)
    assert stopped.play_state is TransportPlayState.STOPPED
    assert stopped.position_seconds == 0.0
    assert stopped.command_sequence == 4
