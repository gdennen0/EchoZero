from __future__ import annotations

from echozero.application.mixer.models import AudibilityState, MixerState, LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import ManualPushDiffPreview, ManualPushTrackOption, Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, ProjectId, SessionId, SongVersionId, TimelineId
from echozero.application.sync.diff_service import SyncDiffRow, SyncDiffSummary
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import ConfirmPushToMA3, OpenPushToMA3Dialog
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
    def __init__(self):
        self._state = SyncState()

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


class _PushTrackListingSyncService(_SyncService):
    def __init__(self, tracks):
        super().__init__()
        self._tracks = tracks

    def list_push_track_options(self):
        return list(self._tracks)


class _AvailableTracksSyncService(_SyncService):
    def __init__(self, tracks):
        super().__init__()
        self._tracks = tracks

    def get_available_ma3_tracks(self):
        return list(self._tracks)


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator(sync_service: SyncService | None = None):
    session = Session(
        id=SessionId("session_manual_push_flow"),
        project_id=ProjectId("project_manual_push_flow"),
        active_timeline_id=TimelineId("timeline_manual_push_flow"),
    )
    timeline = Timeline(
        id=TimelineId("timeline_manual_push_flow"),
        song_version_id=SongVersionId("song_version_manual_push_flow"),
        layers=[
            Layer(
                id="layer_push",
                timeline_id=TimelineId("timeline_manual_push_flow"),
                name="Push Layer",
                kind=LayerKind.EVENT,
                order_index=0,
                takes=[
                    Take(
                        id="take_push",
                        layer_id="layer_push",
                        name="Main",
                        events=[
                            Event(
                                id=EventId("evt_1"),
                                take_id="take_push",
                                start=1.0,
                                end=1.5,
                                label="Cue 1",
                            ),
                            Event(
                                id=EventId("evt_2"),
                                take_id="take_push",
                                start=2.0,
                                end=2.5,
                                label="Cue 2",
                            ),
                        ],
                    )
                ],
            )
        ],
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


def test_open_push_intent_sets_manual_push_dialog_state():
    orchestrator, timeline, session, _playback_service = _build_orchestrator()
    session.manual_push_flow.diff_preview = ManualPushDiffPreview(
        selected_count=3,
        target_track_coord="tc9_tg9_tr9",
        target_track_name="Old Preview",
    )

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("evt_1"), EventId("evt_2")]),
    )

    assert session.manual_push_flow.dialog_open is True
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1"), EventId("evt_2")]
    assert session.manual_push_flow.target_track_coord is None
    assert session.manual_push_flow.diff_gate_open is False
    assert session.manual_push_flow.diff_preview is None


def test_open_push_intent_hydrates_available_tracks_from_sync_service_provider():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_PushTrackListingSyncService(
            tracks=[
                ManualPushTrackOption(
                    coord="tc1_tg1_tr1",
                    name="Track 1",
                    note="Main",
                    event_count=12,
                ),
                ManualPushTrackOption(coord="tc1_tg1_tr2", name="Track 2"),
            ]
        )
    )

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("evt_1")]),
    )

    assert session.manual_push_flow.available_tracks == [
        ManualPushTrackOption(
            coord="tc1_tg1_tr1",
            name="Track 1",
            note="Main",
            event_count=12,
        ),
        ManualPushTrackOption(coord="tc1_tg1_tr2", name="Track 2"),
    ]


def test_open_push_intent_maps_legacy_provider_dict_payload_note_and_event_count():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_AvailableTracksSyncService(
            tracks=[
                {
                    "coord": "tc2_tg3_tr4",
                    "name": "Drums",
                    "note": "ez:Drums",
                    "event_count": 7,
                }
            ]
        )
    )

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("evt_1")]),
    )

    assert session.manual_push_flow.available_tracks == [
        ManualPushTrackOption(
            coord="tc2_tg3_tr4",
            name="Drums",
            note="ez:Drums",
            event_count=7,
        )
    ]


def test_confirm_push_intent_stages_diff_gate_without_immediate_transfer():
    orchestrator, timeline, session, playback_service = _build_orchestrator(
        sync_service=_PushTrackListingSyncService(
            tracks=[
                ManualPushTrackOption(
                    coord="tc1_tg2_tr3",
                    name="Track 3",
                    note="Bass",
                    event_count=8,
                )
            ]
        )
    )

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("evt_1")]),
    )

    orchestrator.handle(
        timeline,
        ConfirmPushToMA3(
            target_track_coord="tc1_tg2_tr3",
            selected_event_ids=[EventId("evt_1")],
        ),
    )

    assert session.manual_push_flow.dialog_open is False
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1")]
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.manual_push_flow.diff_gate_open is True
    assert session.manual_push_flow.diff_preview == ManualPushDiffPreview(
        selected_count=1,
        target_track_coord="tc1_tg2_tr3",
        target_track_name="Track 3",
        target_track_note="Bass",
        target_track_event_count=8,
        diff_summary=SyncDiffSummary(
            added_count=1,
            removed_count=0,
            modified_count=0,
            unchanged_count=0,
            row_count=1,
        ),
        diff_rows=[
            SyncDiffRow(
                row_id="evt_1",
                action="add",
                start=1.0,
                end=1.5,
                label="Cue 1",
                before="Not present in MA3 target",
                after="Track 3 (tc1_tg2_tr3)",
            )
        ],
    )
    assert playback_service.update_runtime_calls == 2
