from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTargetOption,
    ManualPullTrackOption,
    Session,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, ProjectId, SessionId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    OpenPullFromMA3Dialog,
    ReplaceSectionCues,
    SectionCueEdit,
    SelectPullSourceEvents,
    SelectPullSourceTimecode,
    SelectPullSourceTrack,
    SelectPullSourceTrackGroup,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SetPullImportMode,
)
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.timeline.orchestrator_transfer_lookup_mixin import (
    _PULL_TARGET_CREATE_NEW_LAYER_ID,
    _PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
)
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

    def list_pull_track_options(self, *, timecode_no: int | None = None, track_group_no: int | None = None):
        tracks = list(self._tracks)
        if timecode_no is not None:
            tracks = [
                track for track in tracks if str(track.coord).startswith(f"tc{int(timecode_no)}_")
            ]
        if track_group_no is not None:
            tracks = [
                track for track in tracks if f"_tg{int(track_group_no)}_" in str(track.coord)
            ]
        return tracks

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
        active_song_version_ma3_timecode_pool_no=1,
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


def _pull_target_option_id(session: Session, name: str) -> LayerId:
    for target in session.manual_pull_flow.available_target_layers:
        if target.name == name:
            return target.layer_id
    raise AssertionError(f"Pull target option not found: {name}")


def test_open_pull_intent_hydrates_hierarchy_and_defaults_to_new_layer_when_unlinked():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[
                ManualPullTrackOption(
                    coord="tc1_tg1_tr1",
                    name="Track 1",
                    number=1,
                    event_count=4,
                )
            ]
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    assert session.manual_pull_flow.workspace_active is True
    assert [(timecode.number, timecode.name) for timecode in session.manual_pull_flow.available_timecodes] == [
        (1, None)
    ]
    assert session.manual_pull_flow.selected_timecode_no == 1
    assert [(group.number, group.track_count) for group in session.manual_pull_flow.available_track_groups] == [
        (1, 1)
    ]
    assert session.manual_pull_flow.selected_track_group_no == 1
    assert session.manual_pull_flow.available_tracks == [
        ManualPullTrackOption(
            coord="tc1_tg1_tr1",
            name="Track 1",
            number=1,
            timecode_name=None,
            note=None,
            event_count=4,
        )
    ]
    assert session.manual_pull_flow.selected_source_track_coords == ["tc1_tg1_tr1"]
    assert session.manual_pull_flow.active_source_track_coord == "tc1_tg1_tr1"
    assert session.manual_pull_flow.source_track_coord == "tc1_tg1_tr1"
    assert session.manual_pull_flow.available_events == []
    assert session.manual_pull_flow.selected_ma3_event_ids == []
    assert session.manual_pull_flow.available_target_layers == [
        ManualPullTargetOption(
            layer_id=LayerId("layer_target"),
            name="Target Layer",
            kind=LayerKind.EVENT,
        ),
        ManualPullTargetOption(
            layer_id=_PULL_TARGET_CREATE_NEW_LAYER_ID,
            name="+ Create New Layer...",
            kind=LayerKind.EVENT,
        ),
        ManualPullTargetOption(
            layer_id=_PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
            name="+ Create Section Layer...",
            kind=LayerKind.SECTION,
        ),
    ]
    assert session.manual_pull_flow.target_layer_id == _PULL_TARGET_CREATE_NEW_LAYER_ID
    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "blocked"
    assert session.batch_transfer_plan.rows[0].issue == "Select source events"


def test_open_pull_defaults_to_linked_layer_when_selected_layer_route_matches_source_track():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [ManualPullEventOption(event_id="evt_1", label="Cue 1", start=1.0)]
            },
        )
    )
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_target"))
    target_layer.sync.ma3_track_coord = "tc1_tg2_tr3"
    timeline.selection.selected_layer_id = target_layer.id

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    assert session.manual_pull_flow.selected_track_group_no == 2
    assert session.manual_pull_flow.source_track_coord == "tc1_tg2_tr3"
    assert session.manual_pull_flow.selected_ma3_event_ids == ["evt_1"]
    assert session.manual_pull_flow.target_layer_id == LayerId("layer_target")
    assert session.batch_transfer_plan is not None
    assert session.batch_transfer_plan.rows[0].status == "ready"
    assert session.batch_transfer_plan.rows[0].target_layer_id == LayerId("layer_target")


def test_select_pull_timecode_and_track_group_reloads_scoped_tracks():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[
                ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3),
                ManualPullTrackOption(coord="tc2_tg1_tr1", name="Track 1", number=1),
                ManualPullTrackOption(coord="tc2_tg4_tr7", name="Track 7", number=7),
            ],
            events_by_track={
                "tc2_tg4_tr7": [ManualPullEventOption(event_id="evt_7", label="Cue 7", start=7.0)]
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SelectPullSourceTimecode(timecode_no=2))
    orchestrator.handle(timeline, SelectPullSourceTrackGroup(track_group_no=4))

    assert session.manual_pull_flow.selected_timecode_no == 2
    assert session.manual_pull_flow.selected_track_group_no == 4
    assert [track.coord for track in session.manual_pull_flow.available_tracks] == [
        "tc2_tg1_tr1",
        "tc2_tg4_tr7",
    ]
    assert session.manual_pull_flow.source_track_coord == "tc2_tg4_tr7"
    assert session.manual_pull_flow.selected_ma3_event_ids == ["evt_7"]


def test_apply_pull_import_creates_new_take_and_preserves_start_times():
    orchestrator, timeline, session, playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5, cue_number=11),
                    ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=3.0, cue_number=12),
                    ManualPullEventOption(event_id="ma3_evt_3", label="Cue 3", cue_number=13),
                ]
            },
        )
    )
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_target"))

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(
        timeline,
        SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2", "ma3_evt_3"]),
    )
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=LayerId("layer_target")))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    imported_take = target_layer.takes[-1]
    assert imported_take.id == TakeId("layer_target:ma3_pull:1")
    assert [event.start for event in imported_take.events] == [0.6, 1.0, 3.0]
    assert [event.end for event in imported_take.events] == pytest.approx([0.9, 1.5, 3.3])
    assert [event.cue_number for event in imported_take.events] == [13, 11, 12]
    assert [event.payload_ref for event in imported_take.events] == [
        "ma3_evt_3",
        "ma3_evt_1",
        "ma3_evt_2",
    ]
    assert timeline.selection.selected_layer_id == LayerId("layer_target")
    assert timeline.selection.selected_take_id == imported_take.id
    assert playback_service.update_runtime_calls >= 4


def test_apply_pull_import_expands_zero_width_one_shots_to_default_duration():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Go+", start=41.0, end=41.0),
                ]
            },
        )
    )
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_target"))

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=LayerId("layer_target")))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    imported_take = target_layer.takes[-1]
    assert [event.start for event in imported_take.events] == [41.0]
    assert [event.end for event in imported_take.events] == pytest.approx([41.3])


def test_open_pull_defaults_to_selected_marker_layer_and_imports_marker_events():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Marker Track", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(
                        event_id="ma3_evt_1",
                        label="Verse",
                        start=1.0,
                        cue_number=11,
                    ),
                    ManualPullEventOption(
                        event_id="ma3_evt_2",
                        label="Chorus",
                        start=3.0,
                        cue_number=12,
                    ),
                ]
            },
        )
    )
    marker_layer = Layer(
        id=LayerId("layer_marker"),
        timeline_id=timeline.id,
        name="Marker Layer",
        kind=LayerKind.MARKER,
        order_index=3,
        takes=[Take(id=TakeId("take_marker"), layer_id=LayerId("layer_marker"), name="Main")],
    )
    timeline.layers.append(marker_layer)
    timeline.selection.selected_layer_id = marker_layer.id

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    assert any(
        target.layer_id == LayerId("layer_marker")
        for target in session.manual_pull_flow.available_target_layers
    )
    assert session.manual_pull_flow.target_layer_id == LayerId("layer_marker")
    assert session.manual_pull_flow.import_mode == "new_take"

    orchestrator.handle(timeline, ApplyPullFromMA3())

    imported_take = marker_layer.takes[-1]
    assert marker_layer.kind is LayerKind.MARKER
    assert imported_take.name == "MA3 Pull - Marker Track"
    assert [event.cue_number for event in imported_take.events] == [11, 12]
    assert [event.label for event in imported_take.events] == ["Verse", "Chorus"]
    assert marker_layer.sync.ma3_track_coord == "tc1_tg2_tr3"


def test_open_pull_defaults_to_selected_section_layer_and_refreshes_canonical_sections():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Section Track", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(
                        event_id="ma3_evt_1",
                        label="Verse",
                        start=1.0,
                        cue_number=11,
                    ),
                    ManualPullEventOption(
                        event_id="ma3_evt_2",
                        label="Chorus",
                        start=3.0,
                        cue_number=3,
                        cue_ref="Q3",
                    ),
                ]
            },
        )
    )
    section_layer = Layer(
        id=LayerId("layer_sections"),
        timeline_id=timeline.id,
        name="Sections",
        kind=LayerKind.SECTION,
        order_index=3,
        takes=[Take(id=TakeId("take_sections"), layer_id=LayerId("layer_sections"), name="Main")],
    )
    timeline.layers.append(section_layer)
    timeline.selection.selected_layer_id = section_layer.id

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    assert any(
        target.layer_id == LayerId("layer_sections")
        for target in session.manual_pull_flow.available_target_layers
    )
    assert session.manual_pull_flow.target_layer_id == LayerId("layer_sections")
    assert session.manual_pull_flow.import_mode == "main"

    orchestrator.handle(timeline, ApplyPullFromMA3())

    main_take = section_layer.takes[0]
    assert len(section_layer.takes) == 1
    assert section_layer.kind is LayerKind.SECTION
    assert [event.cue_ref for event in main_take.events] == ["11", "Q3"]
    assert [event.label for event in main_take.events] == ["Verse", "Chorus"]
    assert [(cue.cue_ref, cue.start, cue.name) for cue in timeline.section_cues] == [
        ("11", 1.0, "Verse"),
        ("Q3", 3.0, "Chorus"),
    ]
    assert section_layer.sync.ma3_track_coord == "tc1_tg2_tr3"


def test_apply_pull_import_preserves_shared_cue_metadata_fields():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(
                        event_id="ma3_evt_1",
                        label="Verse",
                        start=1.0,
                        cue_number=11,
                        cue_ref="Q11A",
                        color="#ffaa00",
                        notes="Imported exactly",
                        payload_ref="payload://ma3_evt_1",
                    ),
                ]
            },
        )
    )
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_target"))

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=LayerId("layer_target")))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    imported_event = target_layer.takes[-1].events[0]
    assert imported_event.cue_ref == "Q11A"
    assert imported_event.color == "#ffaa00"
    assert imported_event.notes == "Imported exactly"
    assert imported_event.payload_ref == "payload://ma3_evt_1"


def test_apply_pull_create_new_target_creates_event_layer_and_imports_events():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                    ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
                ]
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    create_new_target_id = _pull_target_option_id(session, "+ Create New Layer...")
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=create_new_target_id))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    created_layer = timeline.layers[-1]
    assert created_layer.kind is LayerKind.EVENT
    assert created_layer.name == "Track 3"
    assert created_layer.sync.ma3_track_coord == "tc1_tg2_tr3"
    assert len(created_layer.takes) == 1
    assert created_layer.takes[0].name == "Main"
    assert [event.payload_ref for event in created_layer.takes[0].events] == [
        "ma3_evt_1",
        "ma3_evt_2",
    ]
    assert timeline.selection.selected_layer_id == created_layer.id
    assert timeline.selection.selected_take_id == created_layer.takes[0].id
    assert session.manual_pull_flow.target_layer_id_by_source_track["tc1_tg2_tr3"] == created_layer.id


def test_apply_pull_create_section_target_creates_section_layer_and_imports_main():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Song Parts", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(
                        event_id="ma3_evt_1",
                        label="Verse",
                        start=12.0,
                        cue_number=7,
                    ),
                    ManualPullEventOption(
                        event_id="ma3_evt_2",
                        label="Chorus",
                        start=41.0,
                        cue_number=3,
                        cue_ref="Q3",
                    ),
                ]
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    create_section_target_id = _pull_target_option_id(session, "+ Create Section Layer...")
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=create_section_target_id))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    created_layer = next(layer for layer in timeline.layers if layer.kind is LayerKind.SECTION)
    assert created_layer.id == LayerId("layer_sections")
    assert created_layer.name == "Sections"
    assert created_layer.order_index == min(layer.order_index for layer in timeline.layers)
    assert created_layer.sync.ma3_track_coord == "tc1_tg2_tr3"
    assert len(created_layer.takes) == 1
    assert created_layer.takes[0].name == "Main"
    assert [event.cue_ref for event in created_layer.takes[0].events] == ["7", "Q3"]
    assert [event.label for event in created_layer.takes[0].events] == ["Verse", "Chorus"]
    assert [(cue.cue_ref, cue.start, cue.name) for cue in timeline.section_cues] == [
        ("7", 12.0, "Verse"),
        ("Q3", 41.0, "Chorus"),
    ]
    assert timeline.selection.selected_layer_id == created_layer.id
    assert timeline.selection.selected_take_id == created_layer.takes[0].id
    assert session.manual_pull_flow.target_layer_id_by_source_track["tc1_tg2_tr3"] == created_layer.id


def test_apply_pull_create_section_target_preserves_float_cue_numbers():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Song Parts", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(
                        event_id="ma3_evt_1",
                        label="Break",
                        start=12.0,
                        cue_number=7.5,
                    ),
                ]
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    create_section_target_id = _pull_target_option_id(session, "+ Create Section Layer...")
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=create_section_target_id))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    created_layer = next(layer for layer in timeline.layers if layer.kind is LayerKind.SECTION)
    created_event = created_layer.takes[0].events[0]

    assert created_event.cue_number == 7.5
    assert created_event.cue_ref == "7.5"
    assert [(cue.cue_ref, cue.start, cue.name) for cue in timeline.section_cues] == [
        ("7.5", 12.0, "Break"),
    ]


def test_apply_pull_import_to_existing_layer_still_creates_new_take():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3)],
            events_by_track={
                "tc1_tg2_tr3": [
                    ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                    ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
                ]
            },
        )
    )
    target_layer = next(layer for layer in timeline.layers if layer.id == LayerId("layer_target"))
    target_main_take = target_layer.takes[0]

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(
        timeline,
        SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"]),
    )
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=LayerId("layer_target")))
    orchestrator.handle(timeline, SetPullImportMode(import_mode="main"))
    orchestrator.handle(timeline, ApplyPullFromMA3())

    assert len(target_layer.takes) == 2
    assert target_layer.sync.ma3_track_coord == "tc1_tg2_tr3"
    assert target_main_take.events == []
    assert [event.payload_ref for event in target_layer.takes[-1].events] == [
        "ma3_evt_1",
        "ma3_evt_2",
    ]
    assert timeline.selection.selected_take_id == target_layer.takes[-1].id


def test_select_pull_source_track_preserves_multi_track_selection_and_per_source_target():
    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[
                ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3),
                ManualPullTrackOption(coord="tc1_tg2_tr4", name="Track 4", number=4),
            ],
            events_by_track={
                "tc1_tg2_tr3": [ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0)],
                "tc1_tg2_tr4": [ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0)],
            },
        )
    )

    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    orchestrator.handle(
        timeline,
        SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3", "tc1_tg2_tr4"]),
    )
    create_per_source_target_id = _pull_target_option_id(
        session,
        "+ Create New Layer Per Source Track...",
    )
    orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc1_tg2_tr4"))
    orchestrator.handle(timeline, SelectPullTargetLayer(target_layer_id=create_per_source_target_id))

    assert session.manual_pull_flow.selected_source_track_coords == [
        "tc1_tg2_tr3",
        "tc1_tg2_tr4",
    ]
    assert session.manual_pull_flow.active_source_track_coord == "tc1_tg2_tr4"
    assert session.manual_pull_flow.target_layer_id == create_per_source_target_id
    assert session.manual_pull_flow.target_layer_id_by_source_track == {
        "tc1_tg2_tr3": create_per_source_target_id,
        "tc1_tg2_tr4": create_per_source_target_id,
    }


def test_apply_pull_requires_selected_source_events_and_target():
    orchestrator, timeline, session, _playback_service = _build_orchestrator()

    with pytest.raises(ValueError, match="ApplyPullFromMA3 requires a selected source_track_coord"):
        orchestrator.handle(timeline, ApplyPullFromMA3())

    orchestrator, timeline, session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", number=3)],
            events_by_track={"tc1_tg2_tr3": [ManualPullEventOption(event_id="evt_1", label="Cue 1")]},
        )
    )
    orchestrator.handle(timeline, OpenPullFromMA3Dialog())
    session.manual_pull_flow.selected_ma3_event_ids = []
    with pytest.raises(ValueError, match="ApplyPullFromMA3 requires selected MA3 events"):
        orchestrator.handle(timeline, ApplyPullFromMA3())


def test_set_pull_import_mode_accepts_main_and_rejects_unknown_modes():
    assert SetPullImportMode(import_mode="main").import_mode == "main"
    with pytest.raises(
        ValueError,
        match="SetPullImportMode requires import_mode 'new_take' or 'main'",
    ):
        SetPullImportMode(import_mode="overwrite_main")


def test_pull_flow_validation_errors_reject_unknown_scoped_values():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator(
        sync_service=_SyncService(
            tracks=[ManualPullTrackOption(coord="tc1_tg2_tr3", name="MA3 Track", number=3)],
            events_by_track={"tc1_tg2_tr3": [ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1")]},
        )
    )
    orchestrator.handle(timeline, OpenPullFromMA3Dialog())

    with pytest.raises(
        ValueError,
        match="SelectPullSourceTimecode timecode_no not found in available_timecodes: 9",
    ):
        orchestrator.handle(timeline, SelectPullSourceTimecode(timecode_no=9))

    with pytest.raises(
        ValueError,
        match="SelectPullSourceTrackGroup track_group_no not found in available_track_groups: 9",
    ):
        orchestrator.handle(timeline, SelectPullSourceTrackGroup(track_group_no=9))

    with pytest.raises(
        ValueError,
        match="SelectPullSourceTrack source_track_coord not found in available_tracks: tc9_tg9_tr9",
    ):
        orchestrator.handle(timeline, SelectPullSourceTrack(source_track_coord="tc9_tg9_tr9"))

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


def test_replace_section_cues_creates_section_layer_and_refreshes_timeline_sections():
    orchestrator, timeline, _session, _playback_service = _build_orchestrator()

    orchestrator.handle(
        timeline,
        ReplaceSectionCues(
            cues=[
                SectionCueEdit(cue_id=None, start=12.0, cue_ref="Q7", name="Verse"),
                SectionCueEdit(cue_id=None, start=41.0, cue_ref="Q3", name="Chorus"),
            ]
        ),
    )

    section_layer = next(layer for layer in timeline.layers if layer.kind is LayerKind.SECTION)
    main_take = section_layer.takes[0]

    assert section_layer.name == "Sections"
    assert section_layer.order_index == min(layer.order_index for layer in timeline.layers)
    assert [(cue.cue_ref, cue.start, cue.name) for cue in timeline.section_cues] == [
        ("Q7", 12.0, "Verse"),
        ("Q3", 41.0, "Chorus"),
    ]
    assert [(event.cue_ref, event.label) for event in main_take.events] == [
        ("Q7", "Verse"),
        ("Q3", "Chorus"),
    ]
