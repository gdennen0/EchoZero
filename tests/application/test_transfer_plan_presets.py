from __future__ import annotations

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTrackOption,
    ManualPushTrackOption,
    Session,
    TransferPresetState,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ApplyTransferPreset,
    DeleteTransferPreset,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    SaveTransferPreset,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullSourceTracks,
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

    def get_state(self) -> PlaybackState:
        return self._state

    def prepare(self, timeline: Timeline) -> PlaybackState:
        return self._state

    def update_runtime(self, timeline, transport, audibility, sync) -> PlaybackState:
        return self._state

    def stop(self) -> PlaybackState:
        return self._state


class _SyncService(SyncService):
    def __init__(self, *, push_tracks=None, pull_tracks=None, events_by_track=None):
        self._state = SyncState()
        self._push_tracks = list(push_tracks or [])
        self._pull_tracks = list(pull_tracks or [])
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

    def list_push_track_options(self, *, timecode_no: int | None = None):
        if timecode_no is None:
            return list(self._push_tracks)
        return [
            track
            for track in self._push_tracks
            if track.coord.startswith(f"tc{int(timecode_no)}_")
        ]

    def list_pull_track_options(self):
        return list(self._pull_tracks)

    def list_pull_source_events(self, source_track_coord: str):
        return list(self._events_by_track.get(source_track_coord, []))


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator(sync_service: SyncService):
    session = Session(
        id=SessionId("session_transfer_presets"),
        project_id=ProjectId("project_transfer_presets"),
        active_song_version_ma3_timecode_pool_no=1,
        active_timeline_id=TimelineId("timeline_transfer_presets"),
    )
    kick_layer = Layer(
        id=LayerId("layer_kick"),
        timeline_id=TimelineId("timeline_transfer_presets"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[
            Take(
                id=TakeId("take_kick"),
                layer_id=LayerId("layer_kick"),
                name="Main",
                events=[Event(id=EventId("kick_evt"), take_id=TakeId("take_kick"), start=1.0, end=1.5, label="Kick")],
            )
        ],
    )
    snare_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=TimelineId("timeline_transfer_presets"),
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[
            Take(
                id=TakeId("take_snare"),
                layer_id=LayerId("layer_snare"),
                name="Main",
                events=[Event(id=EventId("snare_evt"), take_id=TakeId("take_snare"), start=2.0, end=2.5, label="Snare")],
            )
        ],
    )
    timeline = Timeline(
        id=TimelineId("timeline_transfer_presets"),
        song_version_id=SongVersionId("song_version_transfer_presets"),
        layers=[kick_layer, snare_layer],
    )
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=sync_service,
        assembler=_Assembler(),
    )
    return orchestrator, timeline, session


def test_save_transfer_preset_captures_current_mappings_in_creation_order():
    sync_service = _SyncService(
        push_tracks=[
            ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
            ManualPushTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
        ]
    )
    orchestrator, timeline, session = _build_orchestrator(sync_service)
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt"), EventId("snare_evt")]
    timeline.layers[0].sync.ma3_track_coord = "tc1_tg2_tr3"
    timeline.layers[1].sync.ma3_track_coord = "tc1_tg2_tr4"

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt"), EventId("snare_evt")]),
    )
    orchestrator.handle(timeline, SaveTransferPreset(name="Drum Mapping"))
    orchestrator.handle(timeline, SaveTransferPreset(name="Drum Mapping"))

    assert [preset.preset_id for preset in session.transfer_presets] == ["drum-mapping", "drum-mapping-2"]
    assert [preset.name for preset in session.transfer_presets] == ["Drum Mapping", "Drum Mapping"]
    assert session.transfer_presets[0].push_target_mapping_by_layer_id == {
        LayerId("layer_kick"): "tc1_tg2_tr3",
        LayerId("layer_snare"): "tc1_tg2_tr4",
    }
    assert session.transfer_presets[0].pull_target_mapping_by_source_track == {}


def test_apply_transfer_preset_updates_active_push_context_and_rebuilds_rows():
    sync_service = _SyncService(
        push_tracks=[
            ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
            ManualPushTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
        ]
    )
    orchestrator, timeline, session = _build_orchestrator(sync_service)
    session.transfer_presets.append(
        TransferPresetState(
            preset_id="drums",
            name="Drums",
            push_target_mapping_by_layer_id={
                LayerId("layer_kick"): "tc1_tg2_tr3",
                LayerId("layer_snare"): "tc1_tg2_tr4",
            },
        )
    )
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt"), EventId("snare_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt"), EventId("snare_evt")]),
    )
    orchestrator.handle(timeline, ApplyTransferPreset(preset_id="drums"))

    assert timeline.layers[0].sync.ma3_track_coord == "tc1_tg2_tr3"
    assert timeline.layers[1].sync.ma3_track_coord == "tc1_tg2_tr4"
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.batch_transfer_plan is not None
    assert [row.row_id for row in session.batch_transfer_plan.rows] == ["push:layer_kick", "push:layer_snare"]
    assert [row.status for row in session.batch_transfer_plan.rows] == ["ready", "ready"]
    assert [row.target_track_coord for row in session.batch_transfer_plan.rows] == ["tc1_tg2_tr3", "tc1_tg2_tr4"]


def test_apply_and_delete_transfer_preset_updates_active_pull_context():
    tracks = [
        ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
        ManualPullTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
    ]
    events_by_track = {
        "tc1_tg2_tr3": [ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1")],
        "tc1_tg2_tr4": [ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2")],
    }
    orchestrator, timeline, session = _build_orchestrator(
        _SyncService(pull_tracks=tracks, events_by_track=events_by_track)
    )
    session.transfer_presets.append(
        TransferPresetState(
            preset_id="import-routing",
            name="Import Routing",
            pull_target_mapping_by_source_track={
                "tc1_tg2_tr3": LayerId("layer_kick"),
                "tc1_tg2_tr4": LayerId("layer_snare"),
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SetPullTrackOptions(tracks=tracks))
    orchestrator.handle(timeline, SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3", "tc1_tg2_tr4"]))
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(timeline, SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1"]))
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr4"))
    orchestrator.handle(timeline, SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_2"]))

    orchestrator.handle(timeline, ApplyTransferPreset(preset_id="import-routing"))

    assert session.manual_pull_flow.target_layer_id_by_source_track == {
        "tc1_tg2_tr3": LayerId("layer_kick"),
        "tc1_tg2_tr4": LayerId("layer_snare"),
    }
    assert session.manual_pull_flow.target_layer_id == LayerId("layer_snare")
    assert session.batch_transfer_plan is not None
    assert [row.status for row in session.batch_transfer_plan.rows] == ["ready", "ready"]
    assert [row.target_layer_id for row in session.batch_transfer_plan.rows] == [
        LayerId("layer_kick"),
        LayerId("layer_snare"),
    ]

    orchestrator.handle(timeline, DeleteTransferPreset(preset_id="import-routing"))

    assert session.transfer_presets == []
