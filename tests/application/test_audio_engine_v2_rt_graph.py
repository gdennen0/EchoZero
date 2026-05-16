"""Audio engine v2 non-live RT graph prototype coverage."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from echozero.application.audio_engine_v2.graph import (
    MASTER_BUS_ID,
    HardwareOutputRoute,
    MixParameters,
    PreparedBus,
    PreparedGraph,
    PreparedTrack,
    TrackRoute,
)
from echozero.application.audio_engine_v2.offline_render import (
    OfflineRenderMemory,
    OfflineRenderState,
    OfflineSourceBank,
    TransitionPolicy,
    render_offline_block,
)
from echozero.application.audio_engine_v2.rt_commands import (
    RtCommand,
    RtCommandBatch,
    RtRuntimeState,
)
from echozero.application.audio_engine_v2.rt_graph import prepare_rt_graph
from echozero.application.audio_engine_v2.transport import (
    TransportCommand,
    TransportPlayState,
    TransportState,
)


def _master(output: HardwareOutputRoute = HardwareOutputRoute(1, 2)) -> PreparedBus:
    return PreparedBus(MASTER_BUS_ID, "Master", TrackRoute.to_hardware((output,)))


def _graph(
    *,
    track_route: TrackRoute | None = None,
    track_mix: MixParameters = MixParameters(),
    buses: tuple[PreparedBus, ...] | None = None,
    channels: int = 2,
) -> PreparedGraph:
    return PreparedGraph(
        graph_id="rt-test",
        tracks=(
            PreparedTrack(
                "track",
                "Track",
                "source",
                route=track_route or TrackRoute.to_master(),
                mix=track_mix,
                channels=channels,
            ),
        ),
        buses=buses or (_master(),),
    )


def _renderer(graph: PreparedGraph, *, hardware_channels: int = 4) -> OfflineRenderState:
    return OfflineRenderState(
        runtime=RtRuntimeState(
            graph=prepare_rt_graph(graph),
            transport=TransportState(play_state=TransportPlayState.PLAYING),
        )
    )


def _memory(graph: PreparedGraph, *, frames: int = 4) -> OfflineRenderMemory:
    return OfflineRenderMemory.create(
        bus_count=len(graph.buses),
        block_frames=frames,
        max_bus_channels=2,
        hardware_channels=4,
    )


def _sources(buffer: NDArray[np.float32] | None = None) -> OfflineSourceBank:
    source = buffer if buffer is not None else np.ones((8, 2), dtype=np.float32)
    return OfflineSourceBank({"source": np.asarray(source, dtype=np.float32)})


def _render(
    graph: PreparedGraph,
    *,
    state: OfflineRenderState | None = None,
    sources: OfflineSourceBank | None = None,
    commands: RtCommandBatch | None = None,
    ramp_frames: int = 0,
) -> tuple[NDArray[np.float32], OfflineRenderState]:
    result = render_offline_block(
        state or _renderer(graph),
        sources=sources or _sources(),
        memory=_memory(graph),
        policy=TransitionPolicy(ramp_frames=ramp_frames),
        commands=commands,
    )
    return result.block, result.state


def test_offline_render_is_deterministic_for_equivalent_graphs() -> None:
    graph = _graph()
    first, _ = _render(graph)
    second, _ = _render(replace(graph, tracks=tuple(graph.tracks)))

    np.testing.assert_allclose(first, second)


def test_track_to_master_to_hardware_renders_expected_channels() -> None:
    block, _ = _render(_graph())

    np.testing.assert_allclose(block[:, :2], np.ones((4, 2), dtype=np.float32))
    np.testing.assert_allclose(block[:, 2:], np.zeros((4, 2), dtype=np.float32))


def test_track_to_subgroup_to_master_to_hardware_renders() -> None:
    buses = (
        PreparedBus("drums", "Drums", TrackRoute.to_master()),
        _master(),
    )
    block, _ = _render(_graph(track_route=TrackRoute.to_bus("drums"), buses=buses))

    np.testing.assert_allclose(block[:, :2], np.ones((4, 2), dtype=np.float32))
    np.testing.assert_allclose(block[:, 2:], 0.0)


def test_direct_hardware_and_master_plus_hardware_sends_render_spans() -> None:
    direct, _ = _render(_graph(track_route=TrackRoute.to_hardware((HardwareOutputRoute(3, 4),))))
    mirrored, _ = _render(
        _graph(track_route=TrackRoute.to_master_and_hardware((HardwareOutputRoute(3, 3),)))
    )

    np.testing.assert_allclose(direct[:, :2], 0.0)
    np.testing.assert_allclose(direct[:, 2:4], 1.0)
    np.testing.assert_allclose(mirrored[:, :2], 1.0)
    np.testing.assert_allclose(mirrored[:, 2], 1.0)


def test_no_output_route_renders_silence() -> None:
    block, _ = _render(_graph(track_route=TrackRoute.no_output()))

    np.testing.assert_allclose(block, np.zeros((4, 4), dtype=np.float32))


def test_mute_solo_and_gain_interpretation_matches_v2_mix_model() -> None:
    muted, _ = _render(_graph(track_mix=MixParameters(muted=True)))
    soloed = _graph(track_mix=MixParameters(gain_db=-6.0, soloed=True))
    non_solo = PreparedTrack("other", "Other", "source", mix=MixParameters())
    solo_graph = replace(soloed, tracks=(*soloed.tracks, non_solo))

    solo_block, _ = _render(solo_graph)
    np.testing.assert_allclose(muted, 0.0)
    np.testing.assert_allclose(solo_block[:, :2], np.float32(10.0 ** (-6.0 / 20.0)))


def test_mono_source_duplicates_to_stereo_and_stereo_folds_to_mono_span() -> None:
    mono_graph = _graph(channels=1)
    mono_source: NDArray[np.float32] = np.ones((8,), dtype=np.float32)
    direct_mono = _graph(track_route=TrackRoute.to_hardware((HardwareOutputRoute(3, 3),)))

    stereo_block, _ = _render(mono_graph, sources=_sources(mono_source))
    mono_block, _ = _render(direct_mono)
    np.testing.assert_allclose(stereo_block[:, :2], 1.0)
    np.testing.assert_allclose(mono_block[:, 2], 1.0)


def test_commands_apply_at_block_boundary_and_stale_sequences_reported() -> None:
    graph = _graph()
    state = _renderer(graph)
    batch = RtCommandBatch(
        (
            RtCommand.track_mix(2, track_id="track", mix=MixParameters(muted=True)),
            RtCommand.track_mix(1, track_id="track", mix=MixParameters(gain_db=-6.0)),
        )
    )

    result = render_offline_block(
        state,
        sources=_sources(),
        memory=_memory(graph),
        policy=TransitionPolicy(ramp_frames=0),
        commands=batch,
    )
    stale_result = render_offline_block(
        result.state,
        sources=_sources(),
        memory=_memory(graph),
        policy=TransitionPolicy(ramp_frames=0),
        commands=RtCommandBatch((RtCommand.track_mix(2, track_id="track", mix=MixParameters()),)),
    )
    np.testing.assert_allclose(result.block, 0.0)
    assert result.applied_sequences == (1, 2)
    assert stale_result.stale_sequences == (2,)


def test_transition_ramp_prevents_full_scale_mute_and_stop_discontinuities() -> None:
    graph = _graph()
    _, state = _render(graph, ramp_frames=0)
    muted = RtCommandBatch(
        (RtCommand.track_mix(1, track_id="track", mix=MixParameters(muted=True)),)
    )
    mute_block, muted_state = _render(graph, state=state, commands=muted, ramp_frames=4)
    stopped = RtCommandBatch((RtCommand.transport(TransportCommand.stop(2)),))
    stop_block, _ = _render(graph, state=state, commands=stopped, ramp_frames=4)

    assert 0.0 < float(mute_block[0, 0]) < 1.0
    assert muted_state.runtime.command_sequence == 1
    assert 0.0 < float(stop_block[0, 0]) < 1.0
    assert np.max(np.abs(np.diff(mute_block[:, 0]))) < 1.0
    assert np.max(np.abs(np.diff(stop_block[:, 0]))) < 1.0
