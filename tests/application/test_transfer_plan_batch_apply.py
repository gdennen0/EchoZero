from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTrackOption,
    ManualPushTrackOption,
    Session,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ApplyTransferPlan,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    PreviewTransferPlan,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    SetPullImportMode,
    SetPushTransferMode,
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
    def __init__(self, *, push_tracks=None, pull_tracks=None, events_by_track=None, fail_track: str | None = None):
        self._state = SyncState()
        self._push_tracks = list(push_tracks or [])
        self._pull_tracks = list(pull_tracks or [])
        self._events_by_track = dict(events_by_track or {})
        self._fail_track = fail_track
        self.push_calls: list[tuple[str | None, list[str]]] = []
        self.push_mode_calls: list[str] = []

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

    def apply_push_transfer(self, *, target_track_coord, selected_events):
        event_ids = [str(event.id) for event in selected_events]
        self.push_calls.append((target_track_coord, event_ids))
        if self._fail_track is not None and target_track_coord == self._fail_track:
            raise RuntimeError(f"Push apply failed for {target_track_coord}")


class _TransferModeSyncService(_SyncService):
    def apply_push_transfer(self, *, target_track_coord, selected_events, transfer_mode):
        self.push_mode_calls.append(transfer_mode)
        super().apply_push_transfer(target_track_coord=target_track_coord, selected_events=selected_events)


class _ModeAliasSyncService(_SyncService):
    def apply_push_transfer(self, *, target_track_coord, selected_events, mode):
        self.push_mode_calls.append(mode)
        super().apply_push_transfer(target_track_coord=target_track_coord, selected_events=selected_events)


class _NoPushExecutionSyncService(_SyncService):
    apply_push_transfer = None


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator(sync_service: SyncService):
    session = Session(
        id=SessionId("session_transfer_plan_batch"),
        project_id=ProjectId("project_transfer_plan_batch"),
        active_song_version_ma3_timecode_pool_no=1,
        active_timeline_id=TimelineId("timeline_transfer_plan_batch"),
    )
    kick_layer = Layer(
        id=LayerId("layer_kick"),
        timeline_id=TimelineId("timeline_transfer_plan_batch"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[
            Take(
                id=TakeId("take_kick"),
                layer_id=LayerId("layer_kick"),
                name="Main",
                events=[
                    Event(id=EventId("kick_evt"), take_id=TakeId("take_kick"), start=1.0, end=1.5, label="Kick"),
                ],
            )
        ],
    )
    snare_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=TimelineId("timeline_transfer_plan_batch"),
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[
            Take(
                id=TakeId("take_snare"),
                layer_id=LayerId("layer_snare"),
                name="Main",
                events=[
                    Event(id=EventId("snare_evt"), take_id=TakeId("take_snare"), start=2.0, end=2.5, label="Snare"),
                ],
            )
        ],
    )
    timeline = Timeline(
        id=TimelineId("timeline_transfer_plan_batch"),
        song_version_id=SongVersionId("song_version_transfer_plan_batch"),
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


def _pull_target_option_id(session: Session, name: str) -> LayerId:
    for target in session.manual_pull_flow.available_target_layers:
        if target.name == name:
            return target.layer_id
    raise AssertionError(f"Pull target option not found: {name}")


def _stage_pull_plan(orchestrator: TimelineOrchestrator, timeline: Timeline, *, tracks, events_by_track) -> str:
    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SetPullTrackOptions(tracks=tracks))
    orchestrator.handle(
        timeline,
        SelectPullSourceTracks(source_track_coords=[track.coord for track in tracks]),
    )
    for track in tracks:
        orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord=track.coord))
        selected_event_ids = [event.event_id for event in events_by_track.get(track.coord, [])]
        if selected_event_ids:
            orchestrator.handle(
                timeline,
                SelectPullSourceEvents(selected_ma3_event_ids=selected_event_ids),
            )
        target_layer_id = LayerId("layer_kick") if track.coord.endswith("tr3") else None
        if target_layer_id is not None:
            orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=target_layer_id))
    return f"pull:{timeline.id}"


def test_preview_transfer_plan_keeps_mixed_ready_and_blocked_rows_deterministic():
    tracks = [
        ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
        ManualPullTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
    ]
    events_by_track = {
        "tc1_tg2_tr3": [ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5)],
        "tc1_tg2_tr4": [],
    }
    orchestrator, timeline, session = _build_orchestrator(
        _SyncService(pull_tracks=tracks, events_by_track=events_by_track)
    )
    plan_id = _stage_pull_plan(orchestrator, timeline, tracks=tracks, events_by_track=events_by_track)

    orchestrator.handle(timeline, PreviewTransferPlan(plan_id=plan_id))

    assert session.batch_transfer_plan is not None
    assert [row.row_id for row in session.batch_transfer_plan.rows] == ["pull:tc1_tg2_tr3", "pull:tc1_tg2_tr4"]
    assert [row.status for row in session.batch_transfer_plan.rows] == ["ready", "blocked"]
    assert session.batch_transfer_plan.ready_count == 1
    assert session.batch_transfer_plan.blocked_count == 1
    assert session.batch_transfer_plan.applied_count == 0
    assert session.batch_transfer_plan.failed_count == 0


def test_apply_transfer_plan_imports_pull_rows_and_marks_applied():
    tracks = [ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    events_by_track = {
        "tc1_tg2_tr3": [
            ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
            ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
        ],
    }
    orchestrator, timeline, session = _build_orchestrator(
        _SyncService(pull_tracks=tracks, events_by_track=events_by_track)
    )
    plan_id = _stage_pull_plan(orchestrator, timeline, tracks=tracks, events_by_track=events_by_track)

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=plan_id))

    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_kick"))
    assert len(target_layer.takes) == 2
    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "applied"
    assert session.batch_transfer_plan.applied_count == 1
    assert session.batch_transfer_plan.failed_count == 0
    assert session.batch_transfer_plan.ready_count == 0
    assert session.batch_transfer_plan.blocked_count == 0


def test_apply_transfer_plan_existing_layer_pull_still_creates_new_take_when_mode_is_forced():
    tracks = [ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    events_by_track = {
        "tc1_tg2_tr3": [
            ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
            ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=0.5, end=0.75),
        ],
    }
    orchestrator, timeline, session = _build_orchestrator(
        _SyncService(pull_tracks=tracks, events_by_track=events_by_track)
    )
    plan_id = _stage_pull_plan(orchestrator, timeline, tracks=tracks, events_by_track=events_by_track)
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_kick"))
    target_layer.takes[0].events = [
        Event(
            id=EventId("take_kick:ma3:tc1_tg2_tr3:ma3_evt_1:1"),
            take_id=TakeId("take_kick"),
            start=0.25,
            end=0.5,
            label="Existing",
        )
    ]
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(timeline, SetPullImportMode(import_mode="main"))

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=plan_id))

    assert len(target_layer.takes) == 2
    assert [str(event.id) for event in target_layer.takes[0].events] == [
        "take_kick:ma3:tc1_tg2_tr3:ma3_evt_1:1",
    ]
    assert [str(event.id) for event in target_layer.takes[-1].events] == [
        "layer_kick:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_2:2",
        "layer_kick:ma3_pull:1:ma3:tc1_tg2_tr3:ma3_evt_1:1",
    ]
    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "applied"


def test_apply_transfer_plan_stops_after_row_failure_and_leaves_later_ready_rows_untouched():
    sync_service = _SyncService(
        push_tracks=[
            ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
            ManualPushTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
        ],
        fail_track="tc1_tg2_tr4",
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
    plan_id = f"push:{timeline.id}"

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=plan_id))

    assert session.batch_transfer_plan is not None
    assert [row.row_id for row in session.batch_transfer_plan.rows] == ["push:layer_kick", "push:layer_snare"]
    assert [row.status for row in session.batch_transfer_plan.rows] == ["applied", "failed"]
    assert session.batch_transfer_plan.rows[1].issue == "Push apply failed for tc1_tg2_tr4"
    assert session.batch_transfer_plan.applied_count == 1
    assert session.batch_transfer_plan.failed_count == 1
    assert session.batch_transfer_plan.ready_count == 0
    assert sync_service.push_calls == [
        ("tc1_tg2_tr3", ["kick_evt"]),
        ("tc1_tg2_tr4", ["snare_evt"]),
    ]


def test_apply_transfer_plan_fail_fast_leaves_remaining_ready_rows_and_blocked_rows_untouched():
    sync_service = _SyncService(
        push_tracks=[
            ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
            ManualPushTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
            ManualPushTrackOption(coord="tc1_tg2_tr5", name="Track 5"),
        ],
        fail_track="tc1_tg2_tr4",
    )
    orchestrator, timeline, session = _build_orchestrator(sync_service)
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt"), EventId("snare_evt")]
    timeline.layers[0].sync.ma3_track_coord = "tc1_tg2_tr3"
    timeline.layers[1].sync.ma3_track_coord = "tc1_tg2_tr4"

    third_layer = Layer(
        id=LayerId("layer_yam"),
        timeline_id=timeline.id,
        name="Yam",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[
            Take(
                id=TakeId("take_yam"),
                layer_id=LayerId("layer_yam"),
                name="Main",
                events=[Event(id=EventId("yam_evt"), take_id=TakeId("take_yam"), start=3.0, end=3.5, label="Yam")],
            )
        ],
    )
    fourth_layer = Layer(
        id=LayerId("layer_zed"),
        timeline_id=timeline.id,
        name="Zed",
        kind=LayerKind.EVENT,
        order_index=3,
        takes=[
            Take(
                id=TakeId("take_zed"),
                layer_id=LayerId("layer_zed"),
                name="Main",
                events=[Event(id=EventId("zed_evt"), take_id=TakeId("take_zed"), start=4.0, end=4.5, label="Zed")],
            )
        ],
    )
    timeline.layers.extend([third_layer, fourth_layer])
    timeline.selection.selected_event_ids.extend([EventId("yam_evt"), EventId("zed_evt")])

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(
            selection_event_ids=[EventId("kick_evt"), EventId("snare_evt"), EventId("yam_evt"), EventId("zed_evt")]
        ),
    )
    orchestrator.handle(
        timeline,
        SelectPushTargetTrack(target_track_coord="tc1_tg2_tr5", layer_id=LayerId("layer_zed")),
    )
    orchestrator.handle(
        timeline,
        SelectPushTargetTrack(target_track_coord="tc1_tg2_tr4", layer_id=LayerId("layer_snare")),
    )

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=f"push:{timeline.id}"))

    assert session.batch_transfer_plan is not None
    assert [row.row_id for row in session.batch_transfer_plan.rows] == [
        "push:layer_kick",
        "push:layer_snare",
        "push:layer_yam",
        "push:layer_zed",
    ]
    assert [row.status for row in session.batch_transfer_plan.rows] == ["applied", "failed", "blocked", "ready"]
    assert sync_service.push_calls == [
        ("tc1_tg2_tr3", ["kick_evt"]),
        ("tc1_tg2_tr4", ["snare_evt"]),
    ]
    assert session.batch_transfer_plan.ready_count == 1
    assert session.batch_transfer_plan.blocked_count == 1
    assert session.batch_transfer_plan.applied_count == 1
    assert session.batch_transfer_plan.failed_count == 1


def test_apply_transfer_plan_create_new_layer_per_source_track_creates_distinct_layers():
    tracks = [
        ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3"),
        ManualPullTrackOption(coord="tc1_tg2_tr4", name="Track 4"),
    ]
    events_by_track = {
        "tc1_tg2_tr3": [ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5)],
        "tc1_tg2_tr4": [ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5)],
    }
    orchestrator, timeline, session = _build_orchestrator(
        _SyncService(pull_tracks=tracks, events_by_track=events_by_track)
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SetPullTrackOptions(tracks=tracks))
    orchestrator.handle(
        timeline,
        SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3", "tc1_tg2_tr4"]),
    )
    create_per_source_target_id = _pull_target_option_id(session, "+ Create New Layer Per Source Track...")
    for track in tracks:
        orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord=track.coord))
        orchestrator.handle(
            timeline,
            SelectPullSourceEvents(
                selected_ma3_event_ids=[event.event_id for event in events_by_track[track.coord]]
            ),
        )
        if track.coord == "tc1_tg2_tr3":
            orchestrator.handle(
                timeline,
                SelectPullTargetLayer(target_layer_id=create_per_source_target_id),
            )

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=f"pull:{timeline.id}"))

    created_layers = timeline.layers[2:]
    assert [layer.name for layer in created_layers] == ["Track 3", "Track 4"]
    assert len(created_layers) == 2
    assert created_layers[0].id != created_layers[1].id
    assert [len(layer.takes) for layer in created_layers] == [1, 1]
    assert [layer.takes[0].source_ref for layer in created_layers] == [None, None]
    assert [
        [event.payload_ref for event in layer.takes[0].events]
        for layer in created_layers
    ] == [["ma3_evt_1"], ["ma3_evt_2"]]
    assert session.batch_transfer_plan is not None
    assert [row.status for row in session.batch_transfer_plan.rows] == ["applied", "applied"]
    assert session.batch_transfer_plan.applied_count == 2


def test_apply_transfer_plan_fails_push_row_when_execution_endpoint_unavailable():
    orchestrator, timeline, session = _build_orchestrator(
        _NoPushExecutionSyncService(
            push_tracks=[ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
        )
    )
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt")]),
    )
    orchestrator.handle(
        timeline,
        SelectPushTargetTrack(target_track_coord="tc1_tg2_tr3", layer_id=LayerId("layer_kick")),
    )
    plan_id = f"push:{timeline.id}"

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=plan_id))

    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "failed"
    assert session.batch_transfer_plan.rows[0].issue == "Push execution endpoint unavailable"
    assert session.batch_transfer_plan.failed_count == 1


def test_transfer_plan_plan_id_mismatch_validation_is_strict():
    orchestrator, timeline, _session = _build_orchestrator(_SyncService())
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt")]
    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt")]),
    )

    with pytest.raises(
        ValueError,
        match="ApplyTransferPlan plan_id does not match active batch transfer plan: expected push:timeline_transfer_plan_batch, got wrong_plan",
    ):
        orchestrator.handle(timeline, ApplyTransferPlan(plan_id="wrong_plan"))


def test_apply_transfer_plan_passes_push_transfer_mode_when_endpoint_supports_transfer_mode():
    sync_service = _TransferModeSyncService(
        push_tracks=[ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    )
    orchestrator, timeline, _session = _build_orchestrator(sync_service)
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt")]),
    )
    orchestrator.handle(
        timeline,
        SelectPushTargetTrack(target_track_coord="tc1_tg2_tr3", layer_id=LayerId("layer_kick")),
    )
    orchestrator.handle(timeline, SetPushTransferMode(mode="overwrite"))
    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=f"push:{timeline.id}"))

    assert sync_service.push_calls == [("tc1_tg2_tr3", ["kick_evt"])]
    assert sync_service.push_mode_calls == ["overwrite"]


def test_apply_transfer_plan_passes_push_transfer_mode_when_endpoint_uses_mode_alias():
    sync_service = _ModeAliasSyncService(
        push_tracks=[ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    )
    orchestrator, timeline, _session = _build_orchestrator(sync_service)
    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick")
    timeline.selection.selected_event_ids = [EventId("kick_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt")]),
    )
    orchestrator.handle(
        timeline,
        SelectPushTargetTrack(target_track_coord="tc1_tg2_tr3", layer_id=LayerId("layer_kick")),
    )
    orchestrator.handle(timeline, SetPushTransferMode(mode="overwrite"))
    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=f"push:{timeline.id}"))

    assert sync_service.push_calls == [("tc1_tg2_tr3", ["kick_evt"])]
    assert sync_service.push_mode_calls == ["overwrite"]



def test_open_push_plan_blocks_rows_when_selection_is_non_main_only():
    sync_service = _SyncService(
        push_tracks=[ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    )
    orchestrator, timeline, session = _build_orchestrator(sync_service)
    kick_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_kick"))
    kick_layer.sync.ma3_track_coord = "tc1_tg2_tr3"
    kick_layer.takes.append(
        Take(
            id=TakeId("take_kick_alt"),
            layer_id=LayerId("layer_kick"),
            name="Alt",
            events=[
                Event(
                    id=EventId("kick_alt_evt"),
                    take_id=TakeId("take_kick_alt"),
                    start=1.25,
                    end=1.75,
                    label="Kick Alt",
                )
            ],
        )
    )

    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick_alt")
    timeline.selection.selected_event_ids = [EventId("kick_alt_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_alt_evt")]),
    )

    assert session.batch_transfer_plan is not None
    row = session.batch_transfer_plan.rows[0]
    assert row.status == "blocked"
    assert row.issue == "Select main-take events to push"
    assert row.selected_count == 0
    assert row.selected_event_ids == []


def test_apply_transfer_plan_does_not_push_non_main_only_rows():
    sync_service = _SyncService(
        push_tracks=[ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    )
    orchestrator, timeline, session = _build_orchestrator(sync_service)
    kick_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_kick"))
    kick_layer.sync.ma3_track_coord = "tc1_tg2_tr3"
    kick_layer.takes.append(
        Take(
            id=TakeId("take_kick_alt"),
            layer_id=LayerId("layer_kick"),
            name="Alt",
            events=[
                Event(
                    id=EventId("kick_alt_evt"),
                    take_id=TakeId("take_kick_alt"),
                    start=1.25,
                    end=1.75,
                    label="Kick Alt",
                )
            ],
        )
    )

    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick_alt")
    timeline.selection.selected_event_ids = [EventId("kick_alt_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_alt_evt")]),
    )
    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=f"push:{timeline.id}"))

    assert sync_service.push_calls == []
    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "blocked"
    assert session.batch_transfer_plan.blocked_count == 1


def test_apply_transfer_plan_pushes_only_main_events_when_selection_is_mixed():
    sync_service = _SyncService(
        push_tracks=[ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3")]
    )
    orchestrator, timeline, session = _build_orchestrator(sync_service)
    kick_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_kick"))
    kick_layer.sync.ma3_track_coord = "tc1_tg2_tr3"
    kick_layer.takes.append(
        Take(
            id=TakeId("take_kick_alt"),
            layer_id=LayerId("layer_kick"),
            name="Alt",
            events=[
                Event(
                    id=EventId("kick_alt_evt"),
                    take_id=TakeId("take_kick_alt"),
                    start=1.25,
                    end=1.75,
                    label="Kick Alt",
                )
            ],
        )
    )

    timeline.selection.selected_layer_id = LayerId("layer_kick")
    timeline.selection.selected_take_id = TakeId("take_kick_alt")
    timeline.selection.selected_event_ids = [EventId("kick_evt"), EventId("kick_alt_evt")]

    orchestrator.handle(
        timeline,
        OpenPushToMA3Dialog(selection_event_ids=[EventId("kick_evt"), EventId("kick_alt_evt")]),
    )

    assert session.batch_transfer_plan is not None
    row = session.batch_transfer_plan.rows[0]
    assert row.status == "ready"
    assert [str(event_id) for event_id in row.selected_event_ids] == ["kick_evt"]

    orchestrator.handle(timeline, ApplyTransferPlan(plan_id=f"push:{timeline.id}"))

    assert sync_service.push_calls == [("tc1_tg2_tr3", ["kick_evt"])]
    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "applied"
