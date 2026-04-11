from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import (
    ManualPullDiffPreview,
    ManualPullEventOption,
    ManualPullTargetOption,
    ManualPullTrackOption,
    Session,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, ProjectId, SessionId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.diff_service import SyncDiffRow, SyncDiffSummary
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ConfirmPullFromMA3,
    OpenPullFromMA3Dialog,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
    SetPullSourceEvents,
    SetPullTrackOptions,
)
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService


class _SessionService(SessionService):
    def __init__(self, session: Session):
        self._session = session

    def get_session(self) -> Session:
        return self._session

    def set_active_song(self, song_id):
        self._session.active_song_id = song_id
        return self._session

    def set_active_song_version(self, song_version_id):
        self._session.active_song_version_id = song_version_id
        return self._session

    def set_active_timeline(self, timeline_id):
        self._session.active_timeline_id = timeline_id
        return self._session


class _TransportService(TransportService):
    def __init__(self):
        self._state = TransportState()

    def get_state(self) -> TransportState:
        return self._state

    def play(self) -> TransportState:
        self._state.is_playing = True
        return self._state

    def pause(self) -> TransportState:
        self._state.is_playing = False
        return self._state

    def stop(self) -> TransportState:
        self._state.is_playing = False
        self._state.playhead = 0.0
        return self._state

    def seek(self, position: float) -> TransportState:
        self._state.playhead = max(0.0, position)
        return self._state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._state.loop_region = loop_region
        self._state.loop_enabled = enabled
        return self._state


class _MixerService(MixerService):
    def __init__(self):
        self._state = MixerState()

    def get_state(self) -> MixerState:
        return self._state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._state.layer_states[layer_id] = state
        return self._state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        return self._state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        return self._state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        return self._state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        return self._state

    def resolve_audibility(self, layers: list) -> list[AudibilityState]:
        return []


class _PlaybackService(PlaybackService):
    def __init__(self):
        self._state = PlaybackState()
        self.update_runtime_calls = 0

    def get_state(self) -> PlaybackState:
        return self._state

    def prepare(self, timeline: Timeline) -> PlaybackState:
        return self._state

    def update_runtime(self, timeline, transport, audibility, sync) -> PlaybackState:
        self.update_runtime_calls += 1
        return self._state

    def stop(self) -> PlaybackState:
        return self._state


class _SyncService(SyncService):
    def __init__(self, tracks=None, events_by_track=None):
        self._state = SyncState()
        self._tracks = list(tracks or [])
        self._events_by_track = dict(events_by_track or {})

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode):
        self._state.mode = mode
        return self._state

    def connect(self):
        self._state.connected = True
        return self._state

    def disconnect(self):
        self._state.connected = False
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport

    def list_pull_track_options(self):
        return list(self._tracks)

    def list_pull_source_events(self, source_track_coord: str):
        return list(self._events_by_track.get(source_track_coord, []))


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator(sync_service: SyncService | None = None):
    session = Session(
        id=SessionId("session_manual_pull_flow"),
        project_id=ProjectId("project_manual_pull_flow"),
        active_timeline_id=TimelineId("timeline_manual_pull_flow"),
    )
    visible_event_layer = Layer(
        id=LayerId("layer_target"),
        timeline_id=TimelineId("timeline_manual_pull_flow"),
        name="Target Layer",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[Take(id=TakeId("take_target"), layer_id=LayerId("layer_target"), name="Main")],
    )
    hidden_event_layer = Layer(
        id=LayerId("layer_hidden"),
        timeline_id=TimelineId("timeline_manual_pull_flow"),
        name="Hidden Layer",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[Take(id=TakeId("take_hidden"), layer_id=LayerId("layer_hidden"), name="Main")],
    )
    hidden_event_layer.presentation_hints.visible = False
    audio_layer = Layer(
        id=LayerId("layer_audio"),
        timeline_id=TimelineId("timeline_manual_pull_flow"),
        name="Audio Layer",
        kind=LayerKind.AUDIO,
        order_index=2,
        takes=[Take(id=TakeId("take_audio"), layer_id=LayerId("layer_audio"), name="Main")],
    )
    timeline = Timeline(
        id=TimelineId("timeline_manual_pull_flow"),
        song_version_id=SongVersionId("song_version_manual_pull_flow"),
        layers=[visible_event_layer, hidden_event_layer, audio_layer],
    )
    playback_service = _PlaybackService()
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(),
        mixer_service=_MixerService(),
        playback_service=playback_service,
        sync_service=sync_service or _SyncService(),
        assembler=_Assembler(),
    )
    return orchestrator, timeline, session, playback_service


def test_open_pull_intent_resets_flow_and_hydrates_tracks_and_target_layers():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[
                ManualPullTrackOption(
                    coord="tc1_tg1_tr1",
                    name="Track 1",
                    note="Main",
                    event_count=4,
                )
            ]
        )
    )
    session.manual_pull_flow.source_track_coord = "stale_coord"
    session.manual_pull_flow.available_events = [ManualPullEventOption(event_id="old", label="Old")]
    session.manual_pull_flow.selected_ma3_event_ids = ["old"]
    session.manual_pull_flow.available_target_layers = [
        ManualPullTargetOption(layer_id=LayerId("old_layer"), name="Old Layer")
    ]
    session.manual_pull_flow.target_layer_id = LayerId("old_layer")
    session.manual_pull_flow.diff_gate_open = True
    session.manual_pull_flow.diff_preview = ManualPullDiffPreview(
        selected_count=2,
        source_track_coord="stale_coord",
        source_track_name="Stale Track",
        target_layer_id=LayerId("old_layer"),
        target_layer_name="Old Layer",
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    assert session.manual_pull_flow.dialog_open is True
    assert session.manual_pull_flow.available_tracks == [
        ManualPullTrackOption(coord="tc1_tg1_tr1", name="Track 1", note="Main", event_count=4)
    ]
    assert session.manual_pull_flow.source_track_coord is None
    assert session.manual_pull_flow.available_events == []
    assert session.manual_pull_flow.selected_ma3_event_ids == []
    assert session.manual_pull_flow.available_target_layers == [
        ManualPullTargetOption(layer_id=LayerId("layer_target"), name="Target Layer")
    ]
    assert session.manual_pull_flow.target_layer_id is None
    assert session.manual_pull_flow.diff_gate_open is False
    assert session.manual_pull_flow.diff_preview is None


def test_pull_selection_intents_update_source_track_events_and_target_layer_state():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track")],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                    ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
                ]
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(
        timeline,
        SetPullTrackOptions(tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track")]),
    )
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(
        timeline,
        SetPullSourceEvents(
            events=[
                ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
            ]
        ),
    )
    orchestrator.handle(
        timeline,
        SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"]),
    )
    orchestrator.handle(
        timeline,
        SelectPullTargetLayer(target_layer_id=LayerId("layer_target")),
    )

    assert session.manual_pull_flow.available_tracks == [
        ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track")
    ]
    assert session.manual_pull_flow.source_track_coord == "tc1_tg2_tr3"
    assert session.manual_pull_flow.available_events == [
        ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
        ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
    ]
    assert session.manual_pull_flow.selected_ma3_event_ids == ["ma3_evt_1", "ma3_evt_2"]
    assert session.manual_pull_flow.target_layer_id == LayerId("layer_target")
    assert session.manual_pull_flow.diff_gate_open is False
    assert session.manual_pull_flow.diff_preview is None


def test_confirm_pull_intent_stages_diff_gate_without_immediate_transfer():
    orchestrator, timeline, session, playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[
                ManualPullTrackOption(
                    coord="tc1_tg2_tr3",
                    name="MA3 Track",
                    note="Lead",
                    event_count=8,
                )
            ],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1"),
                    ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2"),
                ]
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(
        timeline,
        SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"]),
    )
    orchestrator.handle(
        timeline,
        SelectPullTargetLayer(target_layer_id=LayerId("layer_target")),
    )

    orchestrator.handle(
        timeline,
        ConfirmPullFromMA3(
            source_track_coord="tc1_tg2_tr3",
            selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"],
            target_layer_id=LayerId("layer_target"),
        ),
    )

    assert session.manual_pull_flow.dialog_open is False
    assert session.manual_pull_flow.source_track_coord == "tc1_tg2_tr3"
    assert session.manual_pull_flow.selected_ma3_event_ids == ["ma3_evt_1", "ma3_evt_2"]
    assert session.manual_pull_flow.target_layer_id == LayerId("layer_target")
    assert session.manual_pull_flow.diff_gate_open is True
    assert session.manual_pull_flow.diff_preview == ManualPullDiffPreview(
        selected_count=2,
        source_track_coord="tc1_tg2_tr3",
        source_track_name="MA3 Track",
        source_track_note="Lead",
        source_track_event_count=8,
        target_layer_id=LayerId("layer_target"),
        target_layer_name="Target Layer",
        import_mode="new_take",
        diff_summary=SyncDiffSummary(
            added_count=2,
            removed_count=0,
            modified_count=0,
            unchanged_count=0,
            row_count=2,
        ),
        diff_rows=[
            SyncDiffRow(
                row_id="ma3_evt_1",
                action="add",
                start=0.0,
                end=0.25,
                label="Cue 1",
                before="Not present in EZ target layer",
                after="Target Layer",
            ),
            SyncDiffRow(
                row_id="ma3_evt_2",
                action="add",
                start=0.25,
                end=0.5,
                label="Cue 2",
                before="Not present in EZ target layer",
                after="Target Layer",
            ),
        ],
    )
    assert playback_service.update_runtime_calls == 5


def test_apply_pull_import_creates_new_take_selects_imported_events_and_clears_diff_gate():
    orchestrator, timeline, session, playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[
                ManualPullTrackOption(
                    coord="tc1_tg2_tr3",
                    name="MA3 Track",
                    note="Lead",
                    event_count=8,
                )
            ],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                    ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=3.0),
                    ManualPullEventOption(event_id="ma3_evt_3", label="Cue 3"),
                ]
            },
        )
    )
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_target"))
    target_layer.takes[0].events = [
        Event(
            id="existing_evt",
            take_id=TakeId("take_target"),
            start=0.25,
            end=0.5,
            label="Existing",
        )
    ]

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(
        timeline,
        SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2", "ma3_evt_3"]),
    )
    orchestrator.handle(
        timeline,
        SelectPullTargetLayer(target_layer_id=LayerId("layer_target")),
    )
    orchestrator.handle(
        timeline,
        ConfirmPullFromMA3(
            source_track_coord="tc1_tg2_tr3",
            selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2", "ma3_evt_3"],
            target_layer_id=LayerId("layer_target"),
        ),
    )

    orchestrator.handle(timeline, ApplyPullFromMA3())

    assert len(target_layer.takes) == 2
    imported_take = target_layer.takes[-1]
    assert imported_take.id == TakeId("layer_target:ma3_pull:1")
    assert imported_take.name == "MA3 Pull - MA3 Track"
    assert imported_take.source_ref == "tc1_tg2_tr3"
    assert imported_take.events == [
        Event(
            id="layer_target:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_3:3",
            take_id=TakeId("layer_target:ma3_pull:1"),
            start=0.5,
            end=0.75,
            label="Cue 3",
            payload_ref="ma3_evt_3",
        ),
        Event(
            id="layer_target:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_1:1",
            take_id=TakeId("layer_target:ma3_pull:1"),
            start=1.0,
            end=1.5,
            label="Cue 1",
            payload_ref="ma3_evt_1",
        ),
        Event(
            id="layer_target:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_2:2",
            take_id=TakeId("layer_target:ma3_pull:1"),
            start=3.0,
            end=3.25,
            label="Cue 2",
            payload_ref="ma3_evt_2",
        ),
    ]
    assert timeline.selection.selected_layer_id == LayerId("layer_target")
    assert timeline.selection.selected_take_id == TakeId("layer_target:ma3_pull:1")
    assert timeline.selection.selected_event_ids == [
        "layer_target:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_3:3",
        "layer_target:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_1:1",
        "layer_target:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_2:2",
    ]
    assert session.manual_pull_flow.diff_gate_open is False
    assert session.manual_pull_flow.diff_preview is None
    assert session.manual_pull_flow.selected_ma3_event_ids == ["ma3_evt_1", "ma3_evt_2", "ma3_evt_3"]
    assert playback_service.update_runtime_calls == 6


def test_pull_flow_validation_errors_reject_unknown_source_events_and_targets():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track")],
            events_by_track={
                "tc1_tg2_tr3": [ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1")]
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="ConfirmPullFromMA3 requires a non-empty source_track_coord",
    ):
        ConfirmPullFromMA3(
            source_track_coord="",
            selected_ma3_event_ids=["ma3_evt_1"],
            target_layer_id=LayerId("layer_target"),
        )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    with pytest.raises(
        ValueError,
        match="SelectPullSourceTrack source_track_coord not found in available_tracks: tc9_tg9_tr9",
    ):
        orchestrator.handle(
            timeline,
            SelectPullSourceTrack(source_track_coord="tc9_tg9_tr9"),
        )

    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))

    with pytest.raises(
        ValueError,
        match="SelectPullSourceEvents selected_ma3_event_ids not found in available_events: missing_evt",
    ):
        orchestrator.handle(
            timeline,
            SelectPullSourceEvents(selected_ma3_event_ids=["missing_evt"]),
        )

    with pytest.raises(
        ValueError,
        match="SelectPullTargetLayer target_layer_id not found in available_target_layers: missing_layer",
    ):
        orchestrator.handle(
            timeline,
            SelectPullTargetLayer(target_layer_id=LayerId("missing_layer")),
        )


def test_apply_pull_requires_open_diff_preview():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator()

    with pytest.raises(
        ValueError,
        match="ApplyPullFromMA3 requires an open diff preview",
    ):
        orchestrator.handle(timeline, ApplyPullFromMA3())
