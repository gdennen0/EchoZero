from __future__ import annotations

import json
import sqlite3
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, LayerKind
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ObjectContentId,
    ObjectRevisionId,
    ProjectId,
    SessionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
    TimelineObjectId,
)
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.event_similarity_audio import (
    align_shape_to_reference,
    compare_shape_similarity,
)
from echozero.application.timeline.event_comparison_service import (
    TimbreFingerprintSettings,
    build_timbre_fingerprint_preview,
    compare_timbre_fingerprint_similarity,
)
from echozero.application.timeline.event_batch_scope import EventBatchScope
from echozero.application.timeline.intents import (
    ClearSelection,
    CopiedEventClip,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    EventCueMappingEdit,
    MoveSelectedEventsToAdjacentLayer,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    PasteCopiedEvents,
    ReorderLayer,
    RenumberEventCueNumbers,
    SelectAllEvents,
    SelectAdjacentEventInSelectedLayer,
    SelectAdjacentLayer,
    SelectEveryOtherEvents,
    SelectSimilarEvents,
    SelectSimilarSoundingEvents,
    SelectEvent,
    SelectTake,
    SetFollowCursorEnabled,
    SetGain,
    SetLayerMute,
    SetLayerOutputBus,
    SetLayerSolo,
    SetSelectedEvents,
    Stop,
    ToggleLayerExpanded,
    TriggerTakeAction,
    UpdateEventCueMappings,
)
from echozero.application.timeline.object_content import SourceRef
from echozero.application.timeline.models import Event, EventRef, Layer, Take, Timeline
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
    def __init__(self, state: TransportState):
        self._state = state

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

    def set_layer_state(self, layer_id, state):
        self._state.layer_states[layer_id] = state
        return self._state

    def set_mute(self, layer_id, muted: bool):
        state = self._state.layer_states.setdefault(layer_id, LayerMixerState())
        state.mute = bool(muted)
        return self._state

    def set_solo(self, layer_id, soloed: bool):
        state = self._state.layer_states.setdefault(layer_id, LayerMixerState())
        state.solo = bool(soloed)
        return self._state

    def set_gain(self, layer_id, gain_db: float):
        state = self._state.layer_states.setdefault(layer_id, LayerMixerState())
        state.gain_db = float(gain_db)
        return self._state

    def set_pan(self, layer_id, pan: float):
        state = self._state.layer_states.setdefault(layer_id, LayerMixerState())
        state.pan = float(pan)
        return self._state

    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        return [
            AudibilityState(layer_id=layer.id, is_audible=True, reason="default")
            for layer in layers
        ]


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
    def __init__(self):
        self._state = SyncState()

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode):
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


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _event(
    event_id: str,
    take_id: str,
    start: float,
    *,
    metadata: dict[str, object] | None = None,
) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=start,
        end=start + 0.2,
        label=event_id,
        metadata=dict(metadata or {}),
    )


def _build_orchestrator_and_timeline() -> tuple[TimelineOrchestrator, Timeline, Layer, Take, Take]:
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_kick"),
        name="Main",
        events=[_event("main_1", "take_main", 1.0), _event("main_2", "take_main", 2.0)],
    )
    alt_take = Take(
        id=TakeId("take_alt"),
        layer_id=LayerId("layer_kick"),
        name="Take 2",
        events=[_event("alt_1", "take_alt", 1.25), _event("alt_2", "take_alt", 2.25)],
    )
    layer = Layer(
        id=LayerId("layer_kick"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take, alt_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=TimelineId("timeline_1"),
    )

    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(session.transport_state),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=_SyncService(),
        assembler=_Assembler(),
    )
    return orchestrator, timeline, layer, main_take, alt_take


def _write_mono_wav(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
    normalized = np.clip(samples, -1.0, 1.0)
    pcm = (normalized * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _sine_burst(
    *,
    frequency_hz: float,
    sample_rate: int = 22050,
    duration_seconds: float = 0.18,
) -> np.ndarray:
    times = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    envelope = np.hanning(len(times))
    return (0.7 * np.sin(2.0 * np.pi * frequency_hz * times) * envelope).astype(np.float32)


def _shaped_burst(
    *,
    shape: str,
    frequency_hz: float = 220.0,
    sample_rate: int = 22050,
    duration_seconds: float = 0.18,
) -> np.ndarray:
    sample_count = int(sample_rate * duration_seconds)
    times = np.linspace(0.0, duration_seconds, sample_count, endpoint=False)
    if shape == "tight":
        envelope = np.power(np.hanning(sample_count), 1.8)
    elif shape == "wide":
        envelope = np.power(np.hanning(sample_count), 0.8)
    elif shape == "double":
        pulse = np.hanning(max(8, sample_count // 2))
        gap = np.zeros(max(4, sample_count // 10), dtype=np.float32)
        combined = np.concatenate((pulse, gap, pulse))
        envelope = np.interp(
            np.linspace(0.0, 1.0, sample_count, endpoint=False),
            np.linspace(0.0, 1.0, combined.size, endpoint=False),
            combined,
        )
    elif shape == "front_loaded":
        envelope = np.linspace(1.0, 0.0, sample_count, dtype=np.float32)
        envelope *= np.hanning(sample_count)
    else:
        raise ValueError(f"Unknown burst shape: {shape}")
    return (0.7 * np.sin(2.0 * np.pi * frequency_hz * times) * envelope).astype(np.float32)


def _build_audio_similarity_timeline(
    audio_path: Path,
) -> tuple[TimelineOrchestrator, Timeline, Layer, Take]:
    take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_events"),
        name="Main",
        source_content_ref=SourceRef(
            object_id=TimelineObjectId("object_audio"),
            content_id=ObjectContentId("content_audio"),
            revision_id=ObjectRevisionId("revision_audio"),
            locator=str(audio_path),
        ),
        events=[
            Event(id=EventId("evt_low_1"), take_id=TakeId("take_main"), start=0.0, end=0.18),
            Event(id=EventId("evt_low_2"), take_id=TakeId("take_main"), start=0.5, end=0.68),
            Event(id=EventId("evt_high_1"), take_id=TakeId("take_main"), start=1.0, end=1.18),
            Event(id=EventId("evt_high_2"), take_id=TakeId("take_main"), start=1.5, end=1.68),
        ],
    )
    layer = Layer(
        id=LayerId("layer_events"),
        timeline_id=TimelineId("timeline_audio"),
        name="Events",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_audio"),
        song_version_id=SongVersionId("version_audio"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_audio"),
        project_id=ProjectId("project_audio"),
        active_song_id=SongId("song_audio"),
        active_song_version_id=SongVersionId("version_audio"),
        active_timeline_id=TimelineId("timeline_audio"),
    )
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(session.transport_state),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=_SyncService(),
        assembler=_Assembler(),
    )
    return orchestrator, timeline, layer, take


def _build_selected_layers_audio_similarity_timeline(
    audio_path: Path,
) -> tuple[TimelineOrchestrator, Timeline, Layer, Take, Layer]:
    kick_take = Take(
        id=TakeId("take_kick_main"),
        layer_id=LayerId("layer_kick"),
        name="Main",
        source_content_ref=SourceRef(
            object_id=TimelineObjectId("object_audio"),
            content_id=ObjectContentId("content_audio"),
            revision_id=ObjectRevisionId("revision_audio"),
            locator=str(audio_path),
        ),
        events=[
            Event(id=EventId("kick_1"), take_id=TakeId("take_kick_main"), start=0.0, end=0.18),
            Event(id=EventId("kick_2"), take_id=TakeId("take_kick_main"), start=0.5, end=0.68),
        ],
    )
    snare_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        source_content_ref=SourceRef(
            object_id=TimelineObjectId("object_audio"),
            content_id=ObjectContentId("content_audio"),
            revision_id=ObjectRevisionId("revision_audio"),
            locator=str(audio_path),
        ),
        events=[
            Event(id=EventId("snare_1"), take_id=TakeId("take_snare_main"), start=1.0, end=1.18),
            Event(id=EventId("snare_2"), take_id=TakeId("take_snare_main"), start=1.5, end=1.68),
        ],
    )
    kick_layer = Layer(
        id=LayerId("layer_kick"),
        timeline_id=TimelineId("timeline_audio_layers"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[kick_take],
    )
    snare_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=TimelineId("timeline_audio_layers"),
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[snare_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_audio_layers"),
        song_version_id=SongVersionId("version_audio_layers"),
        layers=[kick_layer, snare_layer],
    )
    session = Session(
        id=SessionId("session_audio_layers"),
        project_id=ProjectId("project_audio_layers"),
        active_song_id=SongId("song_audio_layers"),
        active_song_version_id=SongVersionId("version_audio_layers"),
        active_timeline_id=TimelineId("timeline_audio_layers"),
    )
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(session.transport_state),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=_SyncService(),
        assembler=_Assembler(),
    )
    return orchestrator, timeline, kick_layer, kick_take, snare_layer


def _load_real_project_timeline() -> tuple[TimelineOrchestrator, Timeline, Layer, Take, Layer]:
    project_ez = Path(__file__).resolve().parents[2] / "project.ez"
    if not project_ez.exists():
        pytest.skip(f"Missing project.ez fixture at {project_ez}")

    with tempfile.TemporaryDirectory() as tmp_root:
        with zipfile.ZipFile(project_ez) as archive:
            archive.extract("project.db", tmp_root)

        connection = sqlite3.connect(Path(tmp_root) / "project.db")
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        timeline_layers: list[Layer] = []
        for layer_row in cursor.execute(
            'select id, name, "order" from layers order by "order"'
        ).fetchall():
            layer_id = LayerId(str(layer_row["id"]))
            takes: list[Take] = []
            for take_row in cursor.execute(
                "select id, label, data_json from takes where layer_id = ? order by created_at",
                (str(layer_id),),
            ).fetchall():
                events: list[Event] = []
                data_json = take_row["data_json"]
                if data_json is not None:
                    data = json.loads(data_json)
                    if data.get("type") == "EventData":
                        projected_events: list[tuple[float, float, int, str, dict]] = []
                        for layer in data.get("layers", []):
                            domain_layer_id = str(layer.get("id"))
                            for event_index, event_data in enumerate(layer.get("events", [])):
                                projected_events.append(
                                    (
                                        float(event_data["time"]),
                                        float(event_data["duration"]),
                                        event_index,
                                        domain_layer_id,
                                        event_data,
                                    )
                                )

                        projected_events.sort(
                            key=lambda item: (
                                item[0],
                                item[1],
                                str(item[4].get("id")),
                                item[3],
                                item[2],
                            )
                        )
                        for _, _, event_index, domain_layer_id, event_data in projected_events:
                            event_id = str(event_data.get("id", ""))
                            source_event_id = event_data.get("source_event_id")
                            parent_event_id = event_data.get("parent_event_id")
                            if source_event_id is not None or parent_event_id is not None:
                                timeline_event_id = event_id
                            else:
                                timeline_event_id = (
                                    f"take:{take_row['id']}|layer:{domain_layer_id}"
                                    f"|event:{event_id}|index:{event_index}"
                                )
                            classifications = event_data.get("classifications")
                            metadata = event_data.get("metadata")
                            events.append(
                                Event(
                                    id=EventId(timeline_event_id),
                                    take_id=TakeId(str(take_row["id"])),
                                    start=float(event_data["time"]),
                                    end=float(event_data["time"])
                                    + max(float(event_data["duration"]), 0.08),
                                    label=str(event_data.get("label") or "Onset"),
                                    classifications=dict(classifications or {}),
                                    metadata=dict(metadata or {}),
                                    origin=str(event_data.get("origin", "model")),
                                    source_event_id=(source_event_id or event_id or None),
                                    parent_event_id=(
                                        parent_event_id if parent_event_id is not None else None
                                    ),
                                )
                            )

                takes.append(
                    Take(
                        id=TakeId(str(take_row["id"])),
                        layer_id=layer_id,
                        name=str(take_row["label"]),
                        events=events,
                    )
                )
            timeline_layers.append(
                Layer(
                    id=layer_id,
                    timeline_id=TimelineId("timeline_project_ez"),
                    name=str(layer_row["name"]),
                    kind=LayerKind.EVENT,
                    order_index=int(layer_row["order"]) + 1,
                    takes=takes,
                )
            )

        connection.close()

    timeline = Timeline(
        id=TimelineId("timeline_project_ez"),
        song_version_id=SongVersionId("song_version_project_ez"),
        layers=timeline_layers,
        end=0.0,
    )
    orchestrator, _template_timeline, _template_layer, _main_take, _alt_take = (
        _build_orchestrator_and_timeline()
    )
    timeline.selection.selected_layer_id = timeline_layers[0].id if timeline_layers else None
    timeline.selection.selected_layer_ids = [layer.id for layer in timeline_layers]
    timeline.selection.selected_take_id = (
        timeline_layers[0].takes[0].id if timeline_layers and timeline_layers[0].takes else None
    )
    return (
        orchestrator,
        timeline,
        timeline_layers[0],
        timeline_layers[0].takes[0],
        timeline_layers[1],
    )


def test_select_take_is_selection_only_and_does_not_change_main_truth():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    original_main_event_ids = [event.id for event in main_take.events]
    orchestrator.handle(
        timeline,
        SelectTake(layer_id=layer.id, take_id=alt_take.id),
    )

    assert timeline.selection.selected_take_id == alt_take.id
    assert [event.id for event in main_take.events] == original_main_event_ids


def test_toggle_layer_expanded_round_trips_through_assembled_presentation():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    orchestrator.assembler = TimelineAssembler()

    expanded = orchestrator.handle(
        timeline,
        ToggleLayerExpanded(layer_id=layer.id),
    )

    assert timeline.layers[0].presentation_hints.expanded is True
    assert expanded.layers[0].is_expanded is True

    collapsed = orchestrator.handle(
        timeline,
        ToggleLayerExpanded(layer_id=layer.id),
    )

    assert timeline.layers[0].presentation_hints.expanded is False
    assert collapsed.layers[0].is_expanded is False


def test_select_event_updates_selected_take_for_main_and_take_events():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=main_take.id, event_id=main_take.events[0].id),
    )
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [main_take.events[0].id]

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id),
    )
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_event_additive_and_toggle_preserve_deterministic_take_context():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id,
            take_id=main_take.id,
            event_id=main_take.events[0].id,
            mode="replace",
        ),
    )
    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id, mode="additive"
        ),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id, alt_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id

    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id, mode="toggle"
        ),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id

    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id, take_id=alt_take.id, event_id=main_take.events[0].id, mode="toggle"
        ),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id


def test_select_adjacent_layer_moves_selection_between_visible_layers():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    snare_take = Take(
        id=TakeId("take_snare"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare", 3.0)],
    )
    hat_take = Take(
        id=TakeId("take_hat"),
        layer_id=LayerId("layer_hat"),
        name="Main",
        events=[_event("hat_1", "take_hat", 4.0)],
    )
    snare_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[snare_take],
    )
    hat_layer = Layer(
        id=LayerId("layer_hat"),
        timeline_id=timeline.id,
        name="Hat",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[hat_take],
    )
    timeline.layers.extend([snare_layer, hat_layer])
    timeline.selection.selected_layer_id = snare_layer.id
    timeline.selection.selected_layer_ids = [snare_layer.id]
    timeline.selection.selected_take_id = snare_take.id
    timeline.selection.selected_event_ids = [snare_take.events[0].id]

    orchestrator.handle(timeline, SelectAdjacentLayer(direction=-1))

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_layer_ids == [layer.id]
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []

    orchestrator.handle(timeline, SelectAdjacentLayer(direction=1))

    assert timeline.selection.selected_layer_id == snare_layer.id
    assert timeline.selection.selected_layer_ids == [snare_layer.id]


def test_select_adjacent_event_in_selected_layer_uses_current_take_context():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=1))

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]

    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=-1))

    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_adjacent_event_without_selection_uses_playhead_position():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = []
    timeline.selection.selected_event_refs = []

    orchestrator.transport_service.seek(1.5)
    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=1))

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]

    timeline.selection.selected_event_ids = []
    timeline.selection.selected_event_refs = []
    orchestrator.transport_service.seek(2.1)
    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=-1))

    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_adjacent_event_skips_demoted_when_demoted_navigation_disabled():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    alt_take.events = [
        _event("alt_1", "take_alt", 1.25),
        _event(
            "alt_2",
            "take_alt",
            2.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event("alt_3", "take_alt", 3.25),
    ]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=False),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[2].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=-1, include_demoted=False),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_adjacent_event_can_include_demoted_when_enabled():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    alt_take.events = [
        _event("alt_1", "take_alt", 1.25),
        _event(
            "alt_2",
            "take_alt",
            2.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event("alt_3", "take_alt", 3.25),
    ]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[2].id]


def test_select_adjacent_event_anchors_to_most_recent_selected_ref_order() -> None:
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    alt_take.events = [
        _event(
            "alt_1",
            "take_alt",
            1.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event(
            "alt_2",
            "take_alt",
            2.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event(
            "alt_3",
            "take_alt",
            3.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
    ]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_refs = [
        EventRef(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[1].id),
        EventRef(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id),
    ]
    timeline.selection.selected_event_ids = [
        alt_take.events[1].id,
        alt_take.events[0].id,
    ]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]


def test_clear_selection_clears_events_and_take_without_dropping_selected_layer():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [EventId("alt_1")]

    orchestrator.handle(timeline, ClearSelection())

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []


def test_select_all_events_uses_selected_layer_when_present_and_skips_locked_layers():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    layer.presentation_hints.visible = True
    layer.presentation_hints.locked = False
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    other_layer.presentation_hints.locked = True
    timeline.layers.append(other_layer)
    timeline.selection.selected_layer_id = layer.id

    orchestrator.handle(timeline, SelectAllEvents())

    assert timeline.selection.selected_event_ids == [
        main_take.events[0].id,
        main_take.events[1].id,
        alt_take.events[0].id,
        alt_take.events[1].id,
    ]
    assert timeline.selection.selected_take_id == main_take.id


def test_select_all_events_without_selected_layer_uses_visible_unlocked_layers_only():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    layer.presentation_hints.visible = True
    layer.presentation_hints.locked = False

    visible_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    visible_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[visible_take],
    )
    hidden_take = Take(
        id=TakeId("take_hat_main"),
        layer_id=LayerId("layer_hat"),
        name="Main",
        events=[_event("hat_1", "take_hat_main", 4.0)],
    )
    hidden_layer = Layer(
        id=LayerId("layer_hat"),
        timeline_id=timeline.id,
        name="Hat",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[hidden_take],
    )
    hidden_layer.presentation_hints.visible = False
    timeline.layers.extend([visible_layer, hidden_layer])
    timeline.selection.selected_layer_id = None

    orchestrator.handle(timeline, SelectAllEvents())

    assert timeline.selection.selected_event_ids == [
        main_take.events[0].id,
        main_take.events[1].id,
        alt_take.events[0].id,
        alt_take.events[1].id,
        visible_take.events[0].id,
    ]
    assert timeline.selection.selected_take_id == main_take.id


def test_set_selected_events_preserves_cross_layer_batch_selection_context():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    timeline.layers.append(other_layer)

    orchestrator.handle(
        timeline,
        SetSelectedEvents(
            event_ids=[main_take.events[0].id, other_take.events[0].id],
            anchor_layer_id=other_layer.id,
            anchor_take_id=other_take.id,
            selected_layer_ids=[layer.id, other_layer.id],
        ),
    )

    assert timeline.selection.selected_layer_id == other_layer.id
    assert timeline.selection.selected_layer_ids == [layer.id, other_layer.id]
    assert timeline.selection.selected_take_id == other_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("snare_1")]


def test_select_every_other_events_uses_current_selected_event_scope():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SetSelectedEvents(
            event_ids=[main_take.events[0].id, alt_take.events[0].id, alt_take.events[1].id],
            event_refs=[],
            anchor_layer_id=layer.id,
            anchor_take_id=alt_take.id,
            selected_layer_ids=[layer.id],
        ),
    )

    orchestrator.handle(
        timeline,
        SelectEveryOtherEvents(scope=EventBatchScope(mode="selected_events")),
    )

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_layer_ids == [layer.id]
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("alt_2")]


def test_select_similar_sounding_events_with_missing_audio_keeps_anchor_only():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    main_take.events = [
        Event(
            id=EventId("main_1"),
            take_id=main_take.id,
            start=1.0,
            end=1.2,
            label="Event",
            classifications={},
        ),
        Event(
            id=EventId("main_2"),
            take_id=main_take.id,
            start=2.0,
            end=2.2,
            label="Event",
            classifications={},
        ),
        Event(
            id=EventId("main_3"),
            take_id=main_take.id,
            start=3.0,
            end=3.2,
            label="Event",
            classifications={},
        ),
    ]

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=main_take.id,
            event_id=EventId("main_1"),
        ),
    )

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_layer_ids == [layer.id]
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1")]


def test_compare_shape_similarity_aligns_shifted_curves():
    anchor = (0.0, 0.08, 0.42, 0.9, 1.0, 0.46, 0.15, 0.02)
    shifted = (0.0, 0.0, 0.08, 0.42, 0.9, 1.0, 0.46, 0.15)

    aligned = align_shape_to_reference(anchor, shifted)

    assert np.argmax(np.asarray(aligned, dtype=np.float32)) == np.argmax(
        np.asarray(anchor, dtype=np.float32)
    )
    assert compare_shape_similarity(anchor, shifted) >= 0.95


def test_compare_timbre_fingerprint_similarity_prefers_matching_one_shots(tmp_path: Path):
    kick_a = _shaped_burst(shape="tight", frequency_hz=120.0)
    kick_b = _shaped_burst(shape="tight", frequency_hz=120.0)
    hat = _shaped_burst(shape="tight", frequency_hz=2200.0, duration_seconds=0.12)

    audio_a_path = tmp_path / "kick_a.wav"
    audio_b_path = tmp_path / "kick_b.wav"
    audio_c_path = tmp_path / "hat.wav"
    _write_mono_wav(audio_a_path, kick_a)
    _write_mono_wav(audio_b_path, kick_b)
    _write_mono_wav(audio_c_path, hat)

    cache: dict[str, tuple[np.ndarray, int]] = {}
    settings = TimbreFingerprintSettings(sample_count=64, padding_ms=20.0)
    anchor = build_timbre_fingerprint_preview(
        audio_path=str(audio_a_path),
        start_seconds=0.0,
        end_seconds=0.18,
        settings=settings,
        audio_cache=cache,
    )
    matching = build_timbre_fingerprint_preview(
        audio_path=str(audio_b_path),
        start_seconds=0.0,
        end_seconds=0.18,
        settings=settings,
        audio_cache=cache,
    )
    different = build_timbre_fingerprint_preview(
        audio_path=str(audio_c_path),
        start_seconds=0.0,
        end_seconds=0.12,
        settings=settings,
        audio_cache=cache,
    )

    assert anchor is not None
    assert matching is not None
    assert different is not None
    assert compare_timbre_fingerprint_similarity(anchor, matching) >= 0.99
    assert compare_timbre_fingerprint_similarity(anchor, different) < 0.9


def test_select_similar_sounding_events_same_take_uses_normalized_shape(tmp_path: Path):
    tight_a = _shaped_burst(shape="tight")
    tight_b = _shaped_burst(shape="tight")
    wide = _shaped_burst(shape="wide")
    double_a = _shaped_burst(shape="double")
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate((tight_a, silence, tight_b, silence, wide, silence, double_a))
    audio_path = tmp_path / "similarity.wav"
    _write_mono_wav(audio_path, audio)
    orchestrator, timeline, layer, take = _build_audio_similarity_timeline(audio_path)

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="balanced",
            similarity_threshold_override=0.85,
        ),
    )

    assert timeline.selection.selected_event_ids == [
        EventId("evt_low_1"),
        EventId("evt_low_2"),
        EventId("evt_high_1"),
    ]


def test_select_similar_events_shape_envelope_mode_matches_legacy_behavior(tmp_path: Path):
    tight_a = _shaped_burst(shape="tight")
    tight_b = _shaped_burst(shape="tight")
    wide = _shaped_burst(shape="wide")
    double_a = _shaped_burst(shape="double")
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate((tight_a, silence, tight_b, silence, wide, silence, double_a))
    audio_path = tmp_path / "comparison_mode.wav"
    _write_mono_wav(audio_path, audio)
    orchestrator, timeline, layer, take = _build_audio_similarity_timeline(audio_path)

    orchestrator.handle(
        timeline,
        SelectSimilarEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            comparison_mode="shape_envelope",
            match_strength="balanced",
            similarity_threshold_override=0.85,
        ),
    )

    assert timeline.selection.selected_event_ids == [
        EventId("evt_low_1"),
        EventId("evt_low_2"),
        EventId("evt_high_1"),
    ]


def test_select_similar_sounding_events_strength_thresholds_widen_selection_by_score(
    tmp_path: Path,
):
    tight = _shaped_burst(shape="tight")
    tight_copy = _shaped_burst(shape="tight")
    wide = _shaped_burst(shape="wide")
    double = _shaped_burst(shape="double")
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate((tight, silence, tight_copy, silence, wide, silence, double))
    audio_path = tmp_path / "shape_thresholds.wav"
    _write_mono_wav(audio_path, audio)
    orchestrator, timeline, layer, take = _build_audio_similarity_timeline(audio_path)

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="very_strict",
        ),
    )
    very_strict_ids = list(timeline.selection.selected_event_ids)

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="strict",
        ),
    )
    strict_ids = list(timeline.selection.selected_event_ids)

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="balanced",
        ),
    )
    balanced_ids = list(timeline.selection.selected_event_ids)

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="loose",
        ),
    )
    loose_ids = list(timeline.selection.selected_event_ids)

    assert very_strict_ids == [EventId("evt_low_1"), EventId("evt_low_2")]
    assert strict_ids == [EventId("evt_low_1"), EventId("evt_low_2")]
    assert balanced_ids == [EventId("evt_low_1"), EventId("evt_low_2"), EventId("evt_high_1")]
    assert loose_ids == [EventId("evt_low_1"), EventId("evt_low_2"), EventId("evt_high_1")]


def test_select_similar_sounding_events_threshold_override_is_applied(tmp_path: Path):
    tight = _shaped_burst(shape="tight")
    tight_copy = _shaped_burst(shape="tight")
    wide = _shaped_burst(shape="wide")
    double = _shaped_burst(shape="double")
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate((tight, silence, tight_copy, silence, wide, silence, double))
    audio_path = tmp_path / "shape_override.wav"
    _write_mono_wav(audio_path, audio)
    orchestrator, timeline, layer, take = _build_audio_similarity_timeline(audio_path)

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="loose",
        ),
    )
    assert timeline.selection.selected_event_ids == [
        EventId("evt_low_1"),
        EventId("evt_low_2"),
        EventId("evt_high_1"),
    ]

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=take.id,
            event_id=EventId("evt_low_1"),
            match_strength="loose",
            similarity_threshold_override=0.85,
        ),
    )
    assert timeline.selection.selected_event_ids == [
        EventId("evt_low_1"),
        EventId("evt_low_2"),
        EventId("evt_high_1"),
    ]


def test_select_similar_sounding_events_layer_scope_includes_other_takes(tmp_path: Path):
    tight_a = _shaped_burst(shape="tight")
    tight_b = _shaped_burst(shape="tight")
    double_a = _shaped_burst(shape="double")
    front_loaded = _shaped_burst(shape="front_loaded")
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate((tight_a, silence, front_loaded, silence, tight_b, silence, double_a))
    audio_path = tmp_path / "layer_scope.wav"
    _write_mono_wav(audio_path, audio)

    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    source_ref = SourceRef(
        object_id=TimelineObjectId("object_audio_layer_scope"),
        content_id=ObjectContentId("content_audio_layer_scope"),
        revision_id=ObjectRevisionId("revision_audio_layer_scope"),
        locator=str(audio_path),
    )
    main_take.source_content_ref = source_ref
    alt_take.source_content_ref = source_ref
    main_take.events = [
        Event(id=EventId("main_1"), take_id=main_take.id, start=0.0, end=0.18),
        Event(id=EventId("main_2"), take_id=main_take.id, start=0.5, end=0.68),
    ]
    alt_take.events = [
        Event(id=EventId("alt_1"), take_id=alt_take.id, start=1.0, end=1.18),
        Event(id=EventId("alt_2"), take_id=alt_take.id, start=1.5, end=1.68),
    ]

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=layer.id,
            take_id=alt_take.id,
            event_id=EventId("alt_1"),
            scope_mode="layer",
            match_strength="balanced",
            similarity_threshold_override=0.85,
        ),
    )

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_layer_ids == [layer.id]
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [
        EventId("main_1"),
        EventId("main_2"),
        EventId("alt_1"),
    ]


def test_select_similar_sounding_events_selected_layers_scope_filters_other_layers(
    tmp_path: Path,
):
    tight_a = _shaped_burst(shape="tight")
    tight_b = _shaped_burst(shape="tight")
    double_a = _shaped_burst(shape="double")
    double_b = _shaped_burst(shape="double", frequency_hz=320.0)
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate((tight_a, silence, tight_b, silence, double_a, silence, double_b))
    audio_path = tmp_path / "selected_layers.wav"
    _write_mono_wav(audio_path, audio)
    orchestrator, timeline, kick_layer, kick_take, snare_layer = (
        _build_selected_layers_audio_similarity_timeline(audio_path)
    )
    timeline.selection.selected_layer_ids = [kick_layer.id, snare_layer.id]
    timeline.selection.selected_layer_id = kick_layer.id
    timeline.selection.selected_take_id = kick_take.id

    orchestrator.handle(
        timeline,
        SelectSimilarSoundingEvents(
            layer_id=kick_layer.id,
            take_id=kick_take.id,
            event_id=EventId("kick_1"),
            scope_mode="selected_layers_main",
            match_strength="balanced",
            similarity_threshold_override=0.85,
        ),
    )

    assert timeline.selection.selected_layer_ids == [kick_layer.id, snare_layer.id]
    assert timeline.selection.selected_take_id == kick_take.id
    assert timeline.selection.selected_event_ids == [EventId("kick_1"), EventId("kick_2")]


def test_select_every_other_events_restarts_per_selected_layer_main_scope():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[
            _event("snare_1", "take_snare_main", 3.0),
            _event("snare_2", "take_snare_main", 4.0),
        ],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    timeline.layers.append(other_layer)
    timeline.selection.selected_layer_id = other_layer.id
    timeline.selection.selected_layer_ids = [layer.id, other_layer.id]
    timeline.selection.selected_take_id = other_take.id
    timeline.selection.selected_event_ids = []

    orchestrator.handle(
        timeline,
        SelectEveryOtherEvents(scope=EventBatchScope(mode="selected_layers_main")),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_1"), EventId("alt_2")]
    assert timeline.selection.selected_layer_id == other_layer.id
    assert timeline.selection.selected_layer_ids == [layer.id, other_layer.id]
    assert timeline.selection.selected_take_id == other_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("snare_1")]


def test_renumber_event_cue_numbers_restarts_per_selected_layer_main_scope():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[
            _event("snare_1", "take_snare_main", 3.0),
            _event("snare_2", "take_snare_main", 4.0),
        ],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    timeline.layers.append(other_layer)
    timeline.selection.selected_layer_id = other_layer.id
    timeline.selection.selected_layer_ids = [layer.id, other_layer.id]
    timeline.selection.selected_take_id = other_take.id

    orchestrator.handle(
        timeline,
        RenumberEventCueNumbers(
            scope=EventBatchScope(mode="selected_layers_main"),
            start_at=1,
            step=1,
        ),
    )

    assert [event.cue_number for event in main_take.events] == [1, 2]
    assert [event.cue_number for event in other_take.events] == [1, 2]
    assert [event.cue_number for event in alt_take.events] == [1, 1]
    assert timeline.selection.selected_layer_id == other_layer.id
    assert timeline.selection.selected_layer_ids == [layer.id, other_layer.id]
    assert timeline.selection.selected_take_id == other_take.id
    assert timeline.selection.selected_event_ids == [
        EventId("main_1"),
        EventId("main_2"),
        EventId("snare_1"),
        EventId("snare_2"),
    ]


def test_update_event_cue_mappings_updates_target_events_on_main_take():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    main_take.events[0].cue_number = 1
    main_take.events[0].cue_ref = "1"
    main_take.events[1].cue_number = 2
    main_take.events[1].cue_ref = "2"

    orchestrator.handle(
        timeline,
        UpdateEventCueMappings(
            layer_id=layer.id,
            take_id=main_take.id,
            edits=[
                EventCueMappingEdit(
                    event_id=main_take.events[0].id,
                    cue_number=80.101,
                    cue_ref="80.101",
                ),
                EventCueMappingEdit(
                    event_id=main_take.events[1].id,
                    cue_number=80.102,
                    cue_ref="80.102",
                ),
            ],
        ),
    )

    assert [event.cue_number for event in main_take.events] == [80.101, 80.102]
    assert [event.cue_ref for event in main_take.events] == ["80.101", "80.102"]


def test_create_event_appends_sorted_event_and_selects_it():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        CreateEvent(
            layer_id=layer.id,
            take_id=main_take.id,
            time_range=TimeRange(start=1.4, end=1.7),
            label="Inserted",
        ),
    )

    inserted = next(event for event in main_take.events if event.label == "Inserted")
    assert inserted.id == EventId("take_main:event:1")
    assert inserted.cue_number == 1
    assert [event.start for event in main_take.events] == [1.0, 1.4, 2.0]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [inserted.id]


def test_create_event_accepts_float_cue_numbers():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        CreateEvent(
            layer_id=layer.id,
            take_id=main_take.id,
            time_range=TimeRange(start=1.4, end=1.7),
            label="Inserted",
            cue_number=1.5,
        ),
    )

    inserted = next(event for event in main_take.events if event.label == "Inserted")
    assert inserted.cue_number == 1.5


def test_create_event_on_section_layer_creates_section_start_on_main_take():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    section_main_take = Take(
        id=TakeId("take_sections_main"),
        layer_id=LayerId("layer_sections"),
        name="Main",
        events=[
            Event(
                id=EventId("section_1"),
                take_id=TakeId("take_sections_main"),
                start=0.5,
                end=0.58,
                cue_number=7.5,
                label="Verse",
                cue_ref="Q7.5",
            )
        ],
    )
    section_alt_take = Take(
        id=TakeId("take_sections_alt"),
        layer_id=LayerId("layer_sections"),
        name="Take 2",
        events=[],
    )
    section_layer = Layer(
        id=LayerId("layer_sections"),
        timeline_id=timeline.id,
        name="Sections",
        kind=LayerKind.SECTION,
        order_index=1,
        takes=[section_main_take, section_alt_take],
    )
    timeline.layers.append(section_layer)

    orchestrator.handle(
        timeline,
        CreateEvent(
            layer_id=section_layer.id,
            take_id=section_alt_take.id,
            time_range=TimeRange(start=1.6, end=2.2),
        ),
    )

    created = next(
        event
        for event in section_main_take.events
        if event.id == EventId("take_sections_main:event:1")
    )
    assert created.start == pytest.approx(1.6)
    assert created.end == pytest.approx(1.68)
    assert created.cue_number == 8
    assert created.cue_ref == "Cue 8"
    assert created.label == "Section 8"
    assert section_alt_take.events == []
    assert timeline.selection.selected_layer_id == section_layer.id
    assert timeline.selection.selected_layer_ids == [section_layer.id]
    assert timeline.selection.selected_take_id == section_main_take.id
    assert timeline.selection.selected_event_ids == [created.id]


def test_delete_events_removes_records_and_clears_selected_take_when_selection_is_empty():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id, alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        DeleteEvents(event_ids=[main_take.events[0].id, alt_take.events[0].id]),
    )

    assert [event.id for event in main_take.events] == [EventId("main_2")]
    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []


def test_stop_resets_transport_playhead_and_playing_state():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    session = orchestrator.session_service.get_session()
    session.transport_state.is_playing = True
    session.transport_state.playhead = 3.5

    orchestrator.handle(timeline, Stop())

    assert session.transport_state.is_playing is False
    assert session.transport_state.playhead == 0.0


def test_set_gain_updates_layer_mixer_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    layer.kind = LayerKind.AUDIO

    orchestrator.handle(timeline, SetGain(layer.id, -6.0))
    assert layer.mixer.gain_db == -6.0


def test_set_layer_mute_updates_layer_and_session_mixer_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    layer.kind = LayerKind.AUDIO

    orchestrator.handle(timeline, SetLayerMute(layer.id, True))

    assert layer.mixer.mute is True
    mixer_state = orchestrator.mixer_service.get_state()
    assert mixer_state.layer_states[layer.id].mute is True

    orchestrator.handle(timeline, SetLayerMute(layer.id, False))

    assert layer.mixer.mute is False
    assert mixer_state.layer_states[layer.id].mute is False


def test_set_layer_solo_updates_layer_and_session_mixer_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    layer.kind = LayerKind.AUDIO

    orchestrator.handle(timeline, SetLayerSolo(layer.id, True))

    assert layer.mixer.solo is True
    mixer_state = orchestrator.mixer_service.get_state()
    assert mixer_state.layer_states[layer.id].solo is True

    orchestrator.handle(timeline, SetLayerSolo(layer.id, False))

    assert layer.mixer.solo is False
    assert mixer_state.layer_states[layer.id].solo is False


def test_set_layer_output_bus_updates_layer_and_mixer_session_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    layer.kind = LayerKind.AUDIO

    orchestrator.handle(timeline, SetLayerOutputBus(layer.id, "outputs_3_4"))

    assert layer.mixer.output_bus == "outputs_3_4"
    mixer_state = orchestrator.mixer_service.get_state()
    assert mixer_state.layer_states[layer.id].output_bus == "outputs_3_4"

    orchestrator.handle(timeline, SetLayerOutputBus(layer.id, None))

    assert layer.mixer.output_bus is None
    assert mixer_state.layer_states[layer.id].output_bus is None


def test_set_layer_output_bus_collapses_legacy_comma_value_to_one_route():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    layer.kind = LayerKind.AUDIO

    orchestrator.handle(timeline, SetLayerOutputBus(layer.id, "outputs_1_1,outputs_3_3"))

    assert layer.mixer.output_bus == "outputs_1_1"
    mixer_state = orchestrator.mixer_service.get_state()
    assert mixer_state.layer_states[layer.id].output_bus == "outputs_1_1"


def test_set_gain_is_stubbed_for_event_layers():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    assert layer.kind is LayerKind.EVENT

    orchestrator.handle(timeline, SetGain(layer.id, -6.0))

    assert layer.mixer.gain_db == 0.0
    mixer_state = orchestrator.mixer_service.get_state()
    assert layer.id not in mixer_state.layer_states


def test_set_layer_mute_is_stubbed_for_event_layers():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    assert layer.kind is LayerKind.EVENT

    orchestrator.handle(timeline, SetLayerMute(layer.id, True))

    assert layer.mixer.mute is False
    mixer_state = orchestrator.mixer_service.get_state()
    assert layer.id not in mixer_state.layer_states


def test_set_layer_solo_is_stubbed_for_event_layers():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    assert layer.kind is LayerKind.EVENT

    orchestrator.handle(timeline, SetLayerSolo(layer.id, True))

    assert layer.mixer.solo is False
    mixer_state = orchestrator.mixer_service.get_state()
    assert layer.id not in mixer_state.layer_states


def test_set_layer_output_bus_is_stubbed_for_event_layers():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    assert layer.kind is LayerKind.EVENT

    orchestrator.handle(timeline, SetLayerOutputBus(layer.id, "outputs_3_4"))

    assert layer.mixer.output_bus is None
    mixer_state = orchestrator.mixer_service.get_state()
    assert layer.id not in mixer_state.layer_states


def test_set_follow_cursor_enabled_updates_transport_follow_mode():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    session = orchestrator.session_service.get_session()
    session.transport_state.follow_mode = FollowMode.CENTER

    orchestrator.handle(timeline, SetFollowCursorEnabled(enabled=False))
    assert session.transport_state.follow_mode == FollowMode.OFF

    orchestrator.handle(timeline, SetFollowCursorEnabled(enabled=True))
    assert session.transport_state.follow_mode == FollowMode.CENTER


def test_trigger_take_action_overwrite_main_replaces_events_from_source_take():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="overwrite_main"),
    )

    assert len(main_take.events) == len(alt_take.events)
    assert all(event.take_id == main_take.id for event in main_take.events)
    assert all(str(event.id).startswith("take_main:from:") for event in main_take.events)
    assert timeline.selection.selected_take_id == main_take.id


def test_trigger_take_action_merge_main_appends_sorted_events():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    before_count = len(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="merge_main"),
    )

    assert len(main_take.events) == before_count + len(alt_take.events)
    starts = [event.start for event in main_take.events]
    assert starts == sorted(starts)
    assert timeline.selection.selected_layer_id == layer.id


def test_trigger_take_action_add_selection_to_main_only_clones_selected_take_events():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    selected_event = alt_take.events[1]
    orchestrator.handle(
        timeline,
        SetSelectedEvents(
            event_ids=[selected_event.id],
            event_refs=[],
            anchor_layer_id=layer.id,
            anchor_take_id=alt_take.id,
            selected_layer_ids=[layer.id],
        ),
    )

    before_count = len(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(
            layer_id=layer.id,
            take_id=alt_take.id,
            action_id="add_selection_to_main",
        ),
    )

    assert len(main_take.events) == before_count + 1
    cloned_event = next(event for event in main_take.events if event.start == selected_event.start)
    assert cloned_event.take_id == main_take.id
    assert cloned_event.parent_event_id == str(selected_event.id)
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [cloned_event.id]


def test_trigger_take_action_delete_take_removes_non_main_take_and_falls_back_to_main():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="delete_take"),
    )

    assert [take.id for take in layer.takes] == [main_take.id]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == []


def test_trigger_take_action_unknown_action_is_noop():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    original = list(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="future_action"),
    )

    assert [event.id for event in main_take.events] == [event.id for event in original]


def test_move_selected_events_shifts_selected_events_and_preserves_deterministic_take_context():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id, alt_take.events[0].id]

    orchestrator.handle(timeline, MoveSelectedEvents(delta_seconds=0.5))

    assert main_take.events[0].start == 1.5
    assert main_take.events[0].end == 1.7
    assert alt_take.events[0].start == 1.75
    assert alt_take.events[0].end == 1.95
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("alt_1")]


def test_move_selected_events_clamps_at_time_zero():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id]

    orchestrator.handle(timeline, MoveSelectedEvents(delta_seconds=-5.0))

    assert main_take.events[0].start == 0.0
    assert main_take.events[0].end == pytest.approx(0.2)
    assert timeline.selection.selected_take_id == main_take.id


def test_move_selected_events_transfers_to_target_main_take():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    target_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    target_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[target_take],
    )
    timeline.layers.append(target_layer)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.25, target_layer_id=target_layer.id),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert [event.id for event in target_take.events] == [EventId("alt_1"), EventId("snare_1")]
    assert target_take.events[0].take_id == target_take.id
    assert target_take.events[0].start == 1.5
    assert target_take.events[0].end == 1.7
    assert timeline.selection.selected_layer_id == target_layer.id
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("alt_1")]


def test_move_selected_events_transfers_to_explicit_existing_take():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    source_ref = SourceRef(
        object_id=TimelineObjectId("object_source"),
        content_id=ObjectContentId("content_source"),
        revision_id=ObjectRevisionId("revision_source"),
        locator="/tmp/source-drums.wav",
    )
    alt_take.source_content_ref = source_ref
    layer.source_content_ref = source_ref
    layer.playback.armed_source_ref = "/tmp/source-drums.wav"
    target_take = Take(
        id=TakeId("take_target"),
        layer_id=layer.id,
        name="Take 3",
        events=[_event("target_1", "take_target", 4.0)],
    )
    layer.takes.append(target_take)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(
            delta_seconds=0.25,
            target_layer_id=layer.id,
            target_take_id=target_take.id,
        ),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert [event.id for event in target_take.events] == [EventId("alt_1"), EventId("target_1")]
    assert target_take.events[0].take_id == target_take.id
    assert target_take.events[0].start == 1.5
    assert target_take.events[0].end == 1.7
    assert target_take.source_content_ref == source_ref
    assert layer.playback.armed_source_ref == "/tmp/source-drums.wav"
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("alt_1")]


def test_move_selected_events_with_copy_selected_duplicates_in_place():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.5, copy_selected=True),
    )

    assert [event.id for event in main_take.events] == [
        EventId("main_1"),
        EventId("take_main:dup:main_1:1"),
        EventId("main_2"),
    ]
    duplicate = next(
        event for event in main_take.events if event.id == EventId("take_main:dup:main_1:1")
    )
    original = next(event for event in main_take.events if event.id == EventId("main_1"))
    assert original.start == 1.0
    assert original.end == 1.2
    assert duplicate.start == 1.5
    assert duplicate.end == 1.7
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [EventId("take_main:dup:main_1:1")]


def test_move_selected_events_with_copy_selected_to_target_layer_keeps_originals():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    target_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    target_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[target_take],
    )
    timeline.layers.append(target_layer)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(
            delta_seconds=0.25,
            target_layer_id=target_layer.id,
            copy_selected=True,
        ),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_1"), EventId("alt_2")]
    assert [event.id for event in target_take.events] == [
        EventId("take_snare_main:dup:alt_1:1"),
        EventId("snare_1"),
    ]
    assert target_take.events[0].start == 1.5
    assert target_take.events[0].end == 1.7
    assert timeline.selection.selected_layer_id == target_layer.id
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("take_snare_main:dup:alt_1:1")]


def test_move_selected_events_can_create_new_destination_layer():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    source_ref = SourceRef(
        object_id=TimelineObjectId("object_source"),
        content_id=ObjectContentId("content_source"),
        revision_id=ObjectRevisionId("revision_source"),
        locator="/tmp/source-drums.wav",
    )
    alt_take.source_content_ref = source_ref
    layer.source_content_ref = source_ref
    layer.playback.armed_source_ref = "/tmp/source-drums.wav"
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    original_layer_count = len(timeline.layers)

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(
            delta_seconds=0.0,
            create_layer_title="Percussion Details",
        ),
    )

    assert len(timeline.layers) == original_layer_count + 1
    created_layer = timeline.layers[-1]
    assert created_layer.name == "Percussion Details"
    assert created_layer.kind is LayerKind.EVENT
    assert len(created_layer.takes) == 1
    assert created_layer.takes[0].name == "Main"
    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert [event.id for event in created_layer.takes[0].events] == [EventId("alt_1")]
    assert created_layer.takes[0].events[0].take_id == created_layer.takes[0].id
    assert created_layer.takes[0].source_content_ref == source_ref
    assert created_layer.source_content_ref == source_ref
    assert created_layer.playback.armed_source_ref == "/tmp/source-drums.wav"
    assert created_layer.provenance.source_layer_id == layer.id
    assert timeline.selection.selected_layer_id == created_layer.id
    assert timeline.selection.selected_take_id == created_layer.takes[0].id
    assert timeline.selection.selected_event_ids == [EventId("alt_1")]


def test_paste_copied_events_into_selected_target_take_at_playhead():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    source_ref = SourceRef(
        object_id=TimelineObjectId("object_source"),
        content_id=ObjectContentId("content_source"),
        revision_id=ObjectRevisionId("revision_source"),
        locator="/tmp/source-drums.wav",
    )
    alt_take.source_content_ref = source_ref
    layer.source_content_ref = source_ref
    layer.playback.armed_source_ref = "/tmp/source-drums.wav"
    target_take = Take(
        id=TakeId("take_target"),
        layer_id=layer.id,
        name="Take 3",
        events=[],
    )
    layer.takes.append(target_take)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = target_take.id

    orchestrator.handle(
        timeline,
        PasteCopiedEvents(
            clips=[
                CopiedEventClip(
                    source_layer_id=layer.id,
                    source_take_id=alt_take.id,
                    source_layer_kind=layer.kind,
                    event=_event("clip_1", "take_alt", 1.25),
                ),
                CopiedEventClip(
                    source_layer_id=layer.id,
                    source_take_id=alt_take.id,
                    source_layer_kind=layer.kind,
                    event=_event("clip_2", "take_alt", 2.25),
                ),
            ],
            target_layer_id=layer.id,
            target_take_id=target_take.id,
            insert_at_seconds=10.0,
        ),
    )

    assert [event.id for event in target_take.events] == [
        EventId("take_target:dup:clip_1:1"),
        EventId("take_target:dup:clip_2:1"),
    ]
    assert target_take.events[0].start == 10.0
    assert target_take.events[0].end == 10.2
    assert target_take.events[1].start == 11.0
    assert target_take.events[1].end == 11.2
    assert target_take.source_content_ref == source_ref
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [
        EventId("take_target:dup:clip_1:1"),
        EventId("take_target:dup:clip_2:1"),
    ]


def test_move_selected_events_rejects_locked_or_hidden_transfer_targets():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    target_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[],
    )
    target_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[target_take],
    )
    timeline.layers.append(target_layer)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]
    original_start = alt_take.events[0].start
    original_take_ids = [event.id for event in alt_take.events]

    target_layer.presentation_hints.locked = True
    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.5, target_layer_id=target_layer.id),
    )
    assert [event.id for event in alt_take.events] == original_take_ids
    assert alt_take.events[0].start == original_start
    assert target_take.events == []

    target_layer.presentation_hints.locked = False
    target_layer.presentation_hints.visible = False
    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.5, target_layer_id=target_layer.id),
    )
    assert [event.id for event in alt_take.events] == original_take_ids
    assert alt_take.events[0].start == original_start
    assert target_take.events == []


def test_move_selected_events_to_adjacent_layer_skips_locked_layers():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    locked_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[],
    )
    target_take = Take(
        id=TakeId("take_hat_main"),
        layer_id=LayerId("layer_hat"),
        name="Main",
        events=[],
    )
    locked_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[locked_take],
    )
    locked_layer.presentation_hints.locked = True
    target_layer = Layer(
        id=LayerId("layer_hat"),
        timeline_id=timeline.id,
        name="Hat",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[target_take],
    )
    timeline.layers.extend([locked_layer, target_layer])
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(timeline, MoveSelectedEventsToAdjacentLayer(direction=1))

    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert [event.id for event in target_take.events] == [EventId("alt_1")]
    assert timeline.selection.selected_layer_id == target_layer.id
    assert timeline.selection.selected_layer_ids == [target_layer.id]
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("alt_1")]


def test_reorder_layer_allows_layers_to_move_above_source_audio():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    source_audio = Layer(
        id=LayerId("source_audio"),
        timeline_id=timeline.id,
        name="Song",
        kind=LayerKind.AUDIO,
        order_index=0,
        takes=[],
    )
    drums = Layer(
        id=LayerId("layer_drums"),
        timeline_id=timeline.id,
        name="Drums",
        kind=LayerKind.AUDIO,
        order_index=1,
        takes=[],
    )
    bass = Layer(
        id=LayerId("layer_bass"),
        timeline_id=timeline.id,
        name="Bass",
        kind=LayerKind.AUDIO,
        order_index=2,
        takes=[],
    )
    timeline.layers = [source_audio, drums, bass]

    orchestrator.handle(
        timeline,
        ReorderLayer(
            source_layer_id=bass.id,
            target_after_layer_id=None,
            insert_at_start=True,
        ),
    )

    assert [layer.id for layer in timeline.layers] == [
        LayerId("layer_bass"),
        LayerId("source_audio"),
        LayerId("layer_drums"),
    ]
    assert [layer.order_index for layer in timeline.layers] == [0, 1, 2]


def test_nudge_selected_events_moves_selection_by_one_frame_and_preserves_identity():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    event = main_take.events[0]
    original_end = event.end
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [event.id]

    orchestrator.handle(timeline, NudgeSelectedEvents(direction=1))

    assert event.id == EventId("main_1")
    assert event.start == pytest.approx(1.0 + (1.0 / 30.0))
    assert event.end == pytest.approx(original_end + (1.0 / 30.0))
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [event.id]


def test_nudge_selected_events_clamps_at_zero():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    event = main_take.events[0]
    event.start = 0.01
    event.end = 0.21
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [event.id]

    orchestrator.handle(timeline, NudgeSelectedEvents(direction=-1))

    assert event.start == pytest.approx(0.0)
    assert event.end == pytest.approx(0.2)


def test_duplicate_selected_events_creates_deterministic_ids_and_selects_new_copies():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    selected_ids = [main_take.events[0].id, alt_take.events[0].id]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = selected_ids

    orchestrator.handle(timeline, DuplicateSelectedEvents())

    assert timeline.selection.selected_event_ids == [
        EventId("take_main:dup:main_1:1"),
        EventId("take_alt:dup:alt_1:1"),
    ]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert any(event.id == EventId("take_main:dup:main_1:1") for event in main_take.events)
    assert any(event.id == EventId("take_alt:dup:alt_1:1") for event in alt_take.events)


def test_duplicate_selected_events_offsets_copies_and_is_repeatable():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    event = main_take.events[0]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [event.id]

    orchestrator.handle(timeline, DuplicateSelectedEvents())
    first_duplicate = next(
        candidate
        for candidate in main_take.events
        if str(candidate.id) == "take_main:dup:main_1:1"
    )

    timeline.selection.selected_event_ids = [event.id]
    orchestrator.handle(timeline, DuplicateSelectedEvents())
    second_duplicate = next(
        candidate
        for candidate in main_take.events
        if str(candidate.id) == "take_main:dup:main_1:2"
    )

    assert first_duplicate.start == pytest.approx(event.start + (1.0 / 30.0))
    assert second_duplicate.start == pytest.approx(event.start + (1.0 / 30.0))
