from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import (
    ManualPushSequenceOption,
    ManualPushSequenceRange,
    ManualPushTimecodeOption,
    ManualPushTrackGroupOption,
    ManualPushTrackOption,
    Session,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind, SyncMode
from echozero.application.shared.ids import EventId, ProjectId, SessionId, SongVersionId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.ma3_push_intents import (
    AssignMA3TrackSequence,
    CreateMA3Timecode,
    CreateMA3Track,
    CreateMA3TrackGroup,
    CreateMA3Sequence,
    MA3PushApplyMode,
    MA3PushScope,
    MA3PushTargetMode,
    MA3SequenceCreationMode,
    PushLayerToMA3,
    RefreshMA3Sequences,
    RefreshMA3PushTracks,
    SetLayerMA3Route,
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
        self._state.playhead = position
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


class _PushSyncService(SyncService):
    def __init__(self, track_options: list[ManualPushTrackOption] | None = None):
        self._state = SyncState(mode=SyncMode.MA3)
        self._track_options = list(track_options or [])
        self._timecodes_by_number: dict[int, str | None] = {}
        self._track_group_names_by_timecode: dict[int, dict[int, str]] = {}
        for track in self._track_options:
            tc_no, tg_no, _track_no = self._coord_parts(track.coord)
            if tc_no is None or tg_no is None:
                continue
            self._timecodes_by_number.setdefault(tc_no, track.timecode_name)
            self._track_group_names_by_timecode.setdefault(tc_no, {}).setdefault(
                tg_no,
                f"Group {tg_no}",
            )
        if not self._timecodes_by_number:
            self._timecodes_by_number = {1: None}
        if not self._track_group_names_by_timecode:
            self._track_group_names_by_timecode = {1: {1: "Group 1"}}
        self._sequences = [
            ManualPushSequenceOption(number=215, name="Song Kick"),
            ManualPushSequenceOption(number=216, name="Song Snare"),
            ManualPushSequenceOption(number=301, name="Spare"),
        ]
        self._current_song_range = ManualPushSequenceRange(
            song_label="Song Kick",
            start=215,
            end=314,
        )
        self.push_calls: list[dict[str, object]] = []
        self.assign_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []
        self.create_timecode_calls: list[dict[str, object]] = []
        self.create_track_group_calls: list[dict[str, object]] = []
        self.create_track_calls: list[dict[str, object]] = []
        self.prepare_calls: list[str] = []
        self.refresh_push_track_options_calls: list[dict[str, object | None]] = []

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        self._state.connected = True
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport

    def list_push_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[ManualPushTrackOption]:
        tracks = list(self._track_options)
        if timecode_no is not None:
            tracks = [
                track
                for track in tracks
                if track.coord.startswith(f"tc{int(timecode_no)}_")
            ]
        if track_group_no is not None:
            tracks = [
                track
                for track in tracks
                if f"_tg{int(track_group_no)}_" in track.coord
            ]
        return tracks

    def list_timecodes(self) -> list[ManualPushTimecodeOption]:
        return [
            ManualPushTimecodeOption(
                number=number,
                name=name,
            )
            for number, name in sorted(self._timecodes_by_number.items())
        ]

    def list_track_groups(self, *, timecode_no: int) -> list[ManualPushTrackGroupOption]:
        requested_timecode_no = int(timecode_no)
        names = dict(self._track_group_names_by_timecode.get(requested_timecode_no, {}))
        counts: dict[int, int] = {}
        for track in self._track_options:
            tc_no, tg_no, _track_no = self._coord_parts(track.coord)
            if tc_no != requested_timecode_no or tg_no is None:
                continue
            counts[tg_no] = counts.get(tg_no, 0) + 1
            names.setdefault(tg_no, f"Group {tg_no}")
        return [
            ManualPushTrackGroupOption(
                number=group_no,
                name=names[group_no],
                track_count=counts.get(group_no, 0),
            )
            for group_no in sorted(names)
        ]

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[ManualPushSequenceOption]:
        sequences = list(self._sequences)
        if start_no is not None:
            sequences = [sequence for sequence in sequences if sequence.number >= start_no]
        if end_no is not None:
            sequences = [sequence for sequence in sequences if sequence.number <= end_no]
        return sequences

    def get_current_song_sequence_range(self) -> ManualPushSequenceRange:
        return self._current_song_range

    def assign_track_sequence(self, *, target_track_coord: str, sequence_no: int) -> None:
        self.assign_calls.append(
            {
                "target_track_coord": target_track_coord,
                "sequence_no": sequence_no,
            }
        )
        for index, track in enumerate(self._track_options):
            if track.coord == target_track_coord:
                self._track_options[index] = ManualPushTrackOption(
                    coord=track.coord,
                    name=track.name,
                    number=track.number,
                    timecode_name=track.timecode_name,
                    note=track.note,
                    event_count=track.event_count,
                    sequence_no=sequence_no,
                )
                return

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> ManualPushSequenceOption:
        created = ManualPushSequenceOption(
            number=max(sequence.number for sequence in self._sequences) + 1,
            name=preferred_name or "New Sequence",
        )
        self._sequences.append(created)
        self.create_calls.append(
            {
                "creation_mode": "next_available",
                "preferred_name": preferred_name,
                "number": created.number,
            }
        )
        return created

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> ManualPushSequenceOption:
        current_numbers = {
            sequence.number
            for sequence in self._sequences
            if self._current_song_range.start <= sequence.number <= self._current_song_range.end
        }
        created_number = next(
            number
            for number in range(self._current_song_range.start, self._current_song_range.end + 1)
            if number not in current_numbers
        )
        created = ManualPushSequenceOption(
            number=created_number,
            name=preferred_name or "Current Song Sequence",
        )
        self._sequences.append(created)
        self.create_calls.append(
            {
                "creation_mode": "current_song_range",
                "preferred_name": preferred_name,
                "number": created.number,
            }
        )
        return created

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> ManualPushTimecodeOption:
        created_no = max(self._timecodes_by_number, default=0) + 1
        created_name = preferred_name or f"Timecode {created_no}"
        self._timecodes_by_number[created_no] = created_name
        self._track_group_names_by_timecode.setdefault(created_no, {})
        self.create_timecode_calls.append(
            {
                "preferred_name": preferred_name,
                "number": created_no,
            }
        )
        return ManualPushTimecodeOption(number=created_no, name=created_name)

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> ManualPushTrackGroupOption:
        requested_timecode_no = int(timecode_no)
        if requested_timecode_no not in self._timecodes_by_number:
            self._timecodes_by_number[requested_timecode_no] = None
        names = self._track_group_names_by_timecode.setdefault(requested_timecode_no, {})
        created_no = max(names, default=0) + 1
        created_name = preferred_name or f"Group {created_no}"
        names[created_no] = created_name
        self.create_track_group_calls.append(
            {
                "timecode_no": requested_timecode_no,
                "preferred_name": preferred_name,
                "number": created_no,
            }
        )
        return ManualPushTrackGroupOption(number=created_no, name=created_name, track_count=0)

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> ManualPushTrackOption:
        requested_timecode_no = int(timecode_no)
        requested_track_group_no = int(track_group_no)
        if requested_timecode_no not in self._timecodes_by_number:
            self._timecodes_by_number[requested_timecode_no] = None
        self._track_group_names_by_timecode.setdefault(requested_timecode_no, {}).setdefault(
            requested_track_group_no,
            f"Group {requested_track_group_no}",
        )
        next_track_no = 1
        for track in self._track_options:
            tc_no, tg_no, track_no = self._coord_parts(track.coord)
            if (
                tc_no == requested_timecode_no
                and tg_no == requested_track_group_no
                and track_no is not None
            ):
                next_track_no = max(next_track_no, track_no + 1)
        coord = f"tc{requested_timecode_no}_tg{requested_track_group_no}_tr{next_track_no}"
        created = ManualPushTrackOption(
            coord=coord,
            name=preferred_name or f"Track {next_track_no}",
            number=next_track_no,
            event_count=0,
            sequence_no=None,
        )
        self._track_options.append(created)
        self.create_track_calls.append(
            {
                "timecode_no": requested_timecode_no,
                "track_group_no": requested_track_group_no,
                "preferred_name": preferred_name,
                "coord": coord,
            }
        )
        return created

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        self.prepare_calls.append(target_track_coord)

    def apply_push_transfer(
        self,
        *,
        target_track_coord,
        selected_events,
        transfer_mode: str = "merge",
    ) -> None:
        self.push_calls.append(
            {
                "target_track_coord": target_track_coord,
                "selected_event_ids": [event.id for event in selected_events],
                "transfer_mode": transfer_mode,
            }
        )

    def refresh_push_track_options(
        self,
        *,
        target_track_coord: str | None = None,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> None:
        self.refresh_push_track_options_calls.append(
            {
                "target_track_coord": target_track_coord,
                "timecode_no": timecode_no,
                "track_group_no": track_group_no,
            }
        )

    @staticmethod
    def _coord_parts(coord: str) -> tuple[int | None, int | None, int | None]:
        text = str(coord or "").strip().lower()
        if not text.startswith("tc"):
            return None, None, None
        try:
            tc_text = text[2:].split("_", 1)[0]
            tg_text = text.split("_tg", 1)[1].split("_", 1)[0]
            tr_text = text.split("_tr", 1)[1]
            return int(tc_text), int(tg_text), int(tr_text)
        except (IndexError, ValueError):
            return None, None, None


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator(
    *,
    include_alt_take: bool = False,
    saved_route: str | None = None,
    track_options: list[ManualPushTrackOption] | None = None,
    sync_service_override: _PushSyncService | None = None,
):
    session = Session(
        id=SessionId("session_ma3_push_v1"),
        project_id=ProjectId("project_ma3_push_v1"),
        active_song_version_ma3_timecode_pool_no=1,
        active_timeline_id=TimelineId("timeline_ma3_push_v1"),
    )
    layer = Layer(
        id="layer_kick",
        timeline_id=TimelineId("timeline_ma3_push_v1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[
            Take(
                id="take_main",
                layer_id="layer_kick",
                name="Main",
                events=[
                    Event(
                        id=EventId("evt_1"),
                        take_id="take_main",
                        start=1.0,
                        end=1.5,
                        label="Kick 1",
                    ),
                    Event(
                        id=EventId("evt_2"),
                        take_id="take_main",
                        start=2.0,
                        end=2.5,
                        label="Kick 2",
                    ),
                ],
            )
        ],
    )
    if include_alt_take:
        layer.takes.append(
            Take(
                id="take_alt",
                layer_id="layer_kick",
                name="Alt",
                events=[
                    Event(
                        id=EventId("evt_alt"),
                        take_id="take_alt",
                        start=4.0,
                        end=4.5,
                        label="Alt Kick",
                    )
                ],
            )
        )
    layer.sync.ma3_track_coord = saved_route
    timeline = Timeline(
        id=TimelineId("timeline_ma3_push_v1"),
        song_version_id=SongVersionId("song_version_ma3_push_v1"),
        layers=[layer],
    )
    sync_service = sync_service_override or _PushSyncService(
        track_options=(
            list(track_options)
            if track_options is not None
            else [
                ManualPushTrackOption(
                    coord="tc1_tg2_tr3",
                    name="Track 3",
                    event_count=8,
                    sequence_no=215,
                ),
                ManualPushTrackOption(
                    coord="tc1_tg2_tr5",
                    name="Track 5",
                    event_count=4,
                    sequence_no=216,
                ),
                ManualPushTrackOption(coord="tc1_tg2_tr9", name="Track 9", event_count=2),
            ]
        )
    )
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=sync_service,
        assembler=_Assembler(),
    )
    return orchestrator, timeline, session, sync_service


def test_refresh_ma3_push_tracks_hydrates_manual_push_flow_options():
    orchestrator, timeline, session, sync_service = _build_orchestrator()

    orchestrator.handle(timeline, RefreshMA3PushTracks())

    assert sync_service.refresh_push_track_options_calls == [
        {
            "target_track_coord": None,
            "timecode_no": None,
            "track_group_no": None,
        }
    ]
    assert session.manual_push_flow.available_tracks == [
        ManualPushTrackOption(
            coord="tc1_tg2_tr3",
            name="Track 3",
            event_count=8,
            sequence_no=215,
        ),
        ManualPushTrackOption(
            coord="tc1_tg2_tr5",
            name="Track 5",
            event_count=4,
            sequence_no=216,
        ),
        ManualPushTrackOption(coord="tc1_tg2_tr9", name="Track 9", event_count=2),
    ]


def test_refresh_ma3_push_tracks_scopes_timecode_track_group_and_tracks():
    orchestrator, timeline, session, sync_service = _build_orchestrator(
        track_options=[
            ManualPushTrackOption(
                coord="tc1_tg2_tr3",
                name="Track 3",
                event_count=8,
                sequence_no=215,
            ),
            ManualPushTrackOption(
                coord="tc2_tg1_tr1",
                name="Track 1",
                event_count=2,
                sequence_no=401,
            ),
            ManualPushTrackOption(
                coord="tc2_tg4_tr8",
                name="Track 8",
                event_count=1,
                sequence_no=402,
            ),
        ]
    )

    orchestrator.handle(
        timeline,
        RefreshMA3PushTracks(timecode_no=2, track_group_no=4),
    )

    assert sync_service.refresh_push_track_options_calls == [
        {
            "target_track_coord": None,
            "timecode_no": 2,
            "track_group_no": 4,
        }
    ]
    assert [(timecode.number, timecode.name) for timecode in session.manual_push_flow.available_timecodes] == [
        (1, None),
        (2, None),
    ]
    assert session.manual_push_flow.selected_timecode_no == 2
    assert [(group.number, group.track_count) for group in session.manual_push_flow.available_track_groups] == [
        (1, 1),
        (4, 1),
    ]
    assert session.manual_push_flow.selected_track_group_no == 4
    assert [track.coord for track in session.manual_push_flow.available_tracks] == [
        "tc2_tg4_tr8"
    ]


def test_create_ma3_timecode_creates_and_selects_new_pool():
    orchestrator, timeline, session, sync_service = _build_orchestrator()

    orchestrator.handle(
        timeline,
        CreateMA3Timecode(preferred_name="Song B"),
    )

    assert sync_service.create_timecode_calls == [
        {"preferred_name": "Song B", "number": 2}
    ]
    assert [(timecode.number, timecode.name) for timecode in session.manual_push_flow.available_timecodes] == [
        (1, None),
        (2, "Song B"),
    ]
    assert session.manual_push_flow.selected_timecode_no == 2
    assert session.manual_push_flow.available_track_groups == []
    assert session.manual_push_flow.available_tracks == []


def test_create_ma3_track_group_creates_and_selects_new_group():
    orchestrator, timeline, session, sync_service = _build_orchestrator()

    orchestrator.handle(
        timeline,
        CreateMA3TrackGroup(timecode_no=1, preferred_name="FX"),
    )

    assert sync_service.create_track_group_calls == [
        {"timecode_no": 1, "preferred_name": "FX", "number": 3}
    ]
    assert session.manual_push_flow.selected_timecode_no == 1
    assert [(group.number, group.name, group.track_count) for group in session.manual_push_flow.available_track_groups] == [
        (2, "Group 2", 3),
        (3, "FX", 0),
    ]
    assert session.manual_push_flow.selected_track_group_no == 3
    assert session.manual_push_flow.available_tracks == []


def test_create_ma3_track_creates_track_and_sets_flow_target():
    orchestrator, timeline, session, sync_service = _build_orchestrator()

    orchestrator.handle(
        timeline,
        CreateMA3Track(
            timecode_no=1,
            track_group_no=2,
            preferred_name="Laser",
        ),
    )

    assert sync_service.create_track_calls == [
        {
            "timecode_no": 1,
            "track_group_no": 2,
            "preferred_name": "Laser",
            "coord": "tc1_tg2_tr10",
        }
    ]
    assert session.manual_push_flow.selected_timecode_no == 1
    assert session.manual_push_flow.selected_track_group_no == 2
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr10"
    assert [track.coord for track in session.manual_push_flow.available_tracks] == [
        "tc1_tg2_tr3",
        "tc1_tg2_tr5",
        "tc1_tg2_tr9",
        "tc1_tg2_tr10",
    ]


def test_refresh_ma3_sequences_hydrates_manual_push_sequence_state():
    orchestrator, timeline, session, _sync_service = _build_orchestrator()

    orchestrator.handle(timeline, RefreshMA3Sequences(range_mode="current_song"))

    assert session.manual_push_flow.available_sequences == [
        ManualPushSequenceOption(number=215, name="Song Kick"),
        ManualPushSequenceOption(number=216, name="Song Snare"),
        ManualPushSequenceOption(number=301, name="Spare"),
    ]
    assert session.manual_push_flow.current_song_sequence_range == ManualPushSequenceRange(
        song_label="Song Kick",
        start=215,
        end=314,
    )


def test_set_layer_ma3_route_persists_saved_route_on_layer():
    orchestrator, timeline, _session, _sync_service = _build_orchestrator()

    orchestrator.handle(
        timeline,
        SetLayerMA3Route(layer_id="layer_kick", target_track_coord="tc1_tg2_tr3"),
    )

    assert timeline.layers[0].sync.ma3_track_coord == "tc1_tg2_tr3"


def test_set_layer_ma3_route_prepares_unassigned_track_before_saving_route():
    orchestrator, timeline, session, sync_service = _build_orchestrator()

    orchestrator.handle(
        timeline,
        SetLayerMA3Route(
            layer_id="layer_kick",
            target_track_coord="tc1_tg2_tr9",
            sequence_action=CreateMA3Sequence(
                creation_mode=MA3SequenceCreationMode.NEXT_AVAILABLE,
                preferred_name="Kick - Route",
            ),
        ),
    )

    assert timeline.layers[0].sync.ma3_track_coord == "tc1_tg2_tr9"
    assert sync_service.create_calls == [
        {
            "creation_mode": "next_available",
            "preferred_name": "Kick - Route",
            "number": 302,
        }
    ]
    assert sync_service.assign_calls == [
        {
            "target_track_coord": "tc1_tg2_tr9",
            "sequence_no": 302,
        }
    ]
    assert sync_service.prepare_calls == ["tc1_tg2_tr9"]
    assert session.manual_push_flow.available_tracks == [
        ManualPushTrackOption(
            coord="tc1_tg2_tr3",
            name="Track 3",
            event_count=8,
            sequence_no=215,
        ),
        ManualPushTrackOption(
            coord="tc1_tg2_tr5",
            name="Track 5",
            event_count=4,
            sequence_no=216,
        ),
        ManualPushTrackOption(
            coord="tc1_tg2_tr9",
            name="Track 9",
            event_count=2,
            sequence_no=302,
        ),
    ]


def test_push_layer_to_ma3_uses_saved_route_and_layer_main_events():
    orchestrator, timeline, session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.LAYER_MAIN,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
        ),
    )

    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr3",
            "selected_event_ids": [EventId("evt_1"), EventId("evt_2")],
            "transfer_mode": "merge",
        }
    ]
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1"), EventId("evt_2")]


def test_push_layer_to_ma3_passes_project_push_offset_to_sync_service_when_supported():
    class _OffsetAwarePushSyncService(_PushSyncService):
        def __init__(self):
            super().__init__()
            self.start_offsets: list[float] = []

        def apply_push_transfer(
            self,
            *,
            target_track_coord,
            selected_events,
            transfer_mode: str = "merge",
            start_offset_seconds: float = 0.0,
        ) -> None:
            self.start_offsets.append(float(start_offset_seconds))
            super().apply_push_transfer(
                target_track_coord=target_track_coord,
                selected_events=selected_events,
                transfer_mode=transfer_mode,
            )

    sync_service = _OffsetAwarePushSyncService()
    orchestrator, timeline, session, _ = _build_orchestrator(
        saved_route="tc1_tg2_tr3",
        sync_service_override=sync_service,
    )
    session.project_ma3_push_offset_seconds = -0.75

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.LAYER_MAIN,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
        ),
    )

    assert sync_service.start_offsets == pytest.approx([-0.75])


def test_push_layer_to_ma3_refreshes_manual_push_track_catalog_after_send():
    orchestrator, timeline, session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.LAYER_MAIN,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
        ),
    )

    assert sync_service.refresh_push_track_options_calls == [
        {
            "target_track_coord": "tc1_tg2_tr3",
            "timecode_no": None,
            "track_group_no": None,
        }
    ]
    assert [(timecode.number, timecode.name) for timecode in session.manual_push_flow.available_timecodes] == [
        (1, None),
    ]
    assert session.manual_push_flow.selected_timecode_no == 1
    assert [(group.number, group.track_count) for group in session.manual_push_flow.available_track_groups] == [
        (2, 3),
    ]
    assert session.manual_push_flow.selected_track_group_no == 2
    assert [track.coord for track in session.manual_push_flow.available_tracks] == [
        "tc1_tg2_tr3",
        "tc1_tg2_tr5",
        "tc1_tg2_tr9",
    ]


def test_push_layer_to_ma3_layer_main_excludes_demoted_events():
    orchestrator, timeline, session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")
    timeline.layers[0].takes[0].events[1].metadata = {"review": {"promotion_state": "demoted"}}

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.LAYER_MAIN,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
        ),
    )

    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr3",
            "selected_event_ids": [EventId("evt_1")],
            "transfer_mode": "merge",
        }
    ]
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1")]


def test_push_layer_to_ma3_selected_scope_excludes_demoted_events():
    orchestrator, timeline, session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")
    timeline.layers[0].takes[0].events[1].metadata = {"review": {"promotion_state": "demoted"}}

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.SELECTED_EVENTS,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
            selected_event_ids=[EventId("evt_1"), EventId("evt_2")],
        ),
    )

    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr3",
            "selected_event_ids": [EventId("evt_1")],
            "transfer_mode": "merge",
        }
    ]
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1")]


def test_push_layer_to_ma3_selected_scope_rejects_demoted_only_selection():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")
    timeline.layers[0].takes[0].events[1].metadata = {"review": {"promotion_state": "demoted"}}

    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 requires selected promoted main events to push",
    ):
        orchestrator.handle(
            timeline,
            PushLayerToMA3(
                layer_id="layer_kick",
                scope=MA3PushScope.SELECTED_EVENTS,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
                selected_event_ids=[EventId("evt_2")],
            ),
        )

    assert sync_service.push_calls == []


def test_push_layer_to_ma3_layer_main_rejects_when_all_main_events_are_demoted():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")
    for event in timeline.layers[0].takes[0].events:
        event.metadata = {"review": {"promotion_state": "demoted"}}

    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 requires at least one promoted main event on the layer",
    ):
        orchestrator.handle(
            timeline,
            PushLayerToMA3(
                layer_id="layer_kick",
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        )

    assert sync_service.push_calls == []


def test_push_layer_to_ma3_accepts_marker_layers():
    orchestrator, timeline, session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")
    timeline.layers[0].kind = LayerKind.MARKER

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.LAYER_MAIN,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
        ),
    )

    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr3",
            "selected_event_ids": [EventId("evt_1"), EventId("evt_2")],
            "transfer_mode": "merge",
        }
    ]
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1"), EventId("evt_2")]


def test_push_layer_to_ma3_accepts_section_layers():
    orchestrator, timeline, session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")
    timeline.layers[0].kind = LayerKind.SECTION
    timeline.layers[0].takes[0].events[0].cue_number = 11
    timeline.layers[0].takes[0].events[0].cue_ref = "Q11A"
    timeline.layers[0].takes[0].events[0].label = "Verse"

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.LAYER_MAIN,
            target_mode=MA3PushTargetMode.SAVED_ROUTE,
            apply_mode=MA3PushApplyMode.MERGE,
        ),
    )

    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr3",
            "selected_event_ids": [EventId("evt_1"), EventId("evt_2")],
            "transfer_mode": "merge",
        }
    ]
    assert session.manual_push_flow.target_track_coord == "tc1_tg2_tr3"
    assert session.manual_push_flow.selected_event_ids == [EventId("evt_1"), EventId("evt_2")]


def test_push_layer_to_ma3_one_shot_selected_events_does_not_mutate_saved_route():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.SELECTED_EVENTS,
            target_mode=MA3PushTargetMode.DIFFERENT_TRACK_ONCE,
            apply_mode=MA3PushApplyMode.OVERWRITE,
            target_track_coord="tc1_tg2_tr5",
            selected_event_ids=[EventId("evt_2")],
        ),
    )

    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr5",
            "selected_event_ids": [EventId("evt_2")],
            "transfer_mode": "overwrite",
        }
    ]
    assert timeline.layers[0].sync.ma3_track_coord == "tc1_tg2_tr3"


def test_push_layer_to_ma3_prepares_one_shot_target_before_push_without_mutating_saved_route():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr3")

    orchestrator.handle(
        timeline,
        PushLayerToMA3(
            layer_id="layer_kick",
            scope=MA3PushScope.SELECTED_EVENTS,
            target_mode=MA3PushTargetMode.DIFFERENT_TRACK_ONCE,
            apply_mode=MA3PushApplyMode.OVERWRITE,
            target_track_coord="tc1_tg2_tr9",
            selected_event_ids=[EventId("evt_2")],
            sequence_action=AssignMA3TrackSequence(
                target_track_coord="tc1_tg2_tr9",
                sequence_no=216,
            ),
        ),
    )

    assert sync_service.assign_calls == [
        {
            "target_track_coord": "tc1_tg2_tr9",
            "sequence_no": 216,
        }
    ]
    assert sync_service.prepare_calls == ["tc1_tg2_tr9"]
    assert sync_service.push_calls == [
        {
            "target_track_coord": "tc1_tg2_tr9",
            "selected_event_ids": [EventId("evt_2")],
            "transfer_mode": "overwrite",
        }
    ]
    assert timeline.layers[0].sync.ma3_track_coord == "tc1_tg2_tr3"


def test_push_layer_to_ma3_rejects_known_unassigned_target_without_sequence_action():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(saved_route="tc1_tg2_tr9")

    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 target track tc1_tg2_tr9 has no assigned MA3 sequence",
    ):
        orchestrator.handle(
            timeline,
            PushLayerToMA3(
                layer_id="layer_kick",
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
            ),
        )

    assert sync_service.assign_calls == []
    assert sync_service.prepare_calls == []
    assert sync_service.push_calls == []


def test_push_layer_to_ma3_rejects_non_main_take_selected_events():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(
        include_alt_take=True,
        saved_route="tc1_tg2_tr3",
    )

    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 selected_event_ids must belong to the layer main take: evt_alt",
    ):
        orchestrator.handle(
            timeline,
            PushLayerToMA3(
                layer_id="layer_kick",
                scope=MA3PushScope.SELECTED_EVENTS,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
                selected_event_ids=[EventId("evt_alt")],
            ),
        )

    assert sync_service.push_calls == []


def test_push_layer_to_ma3_requires_saved_route_when_requested():
    orchestrator, timeline, _session, sync_service = _build_orchestrator(saved_route=None)

    with pytest.raises(
        ValueError,
        match="PushLayerToMA3 requires a saved MA3 route for the layer",
    ):
        orchestrator.handle(
            timeline,
            PushLayerToMA3(
                layer_id="layer_kick",
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
            ),
        )

    assert sync_service.push_calls == []
