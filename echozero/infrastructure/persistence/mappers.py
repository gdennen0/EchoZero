"""Mappers between application models and persistence records.

These stay dumb on purpose: convert shapes, do not interpret business meaning.
"""

from echozero.application.shared.ids import (
    ProjectId,
    SessionId,
    SongId,
    SongVersionId,
    TimelineId,
    LayerId,
    TakeId,
    EventId,
)
from echozero.application.shared.enums import LayerKind, PlaybackMode, FollowMode, PlaybackStatus, SyncMode
from echozero.application.shared.ranges import TimeRange
from echozero.application.project.models import Project
from echozero.application.song.models import Song, SongVersion
from echozero.application.session.models import Session
from echozero.application.timeline.models import (
    Timeline,
    Layer,
    Take,
    Event,
    TimelineSelection,
    TimelineViewport,
    LayerSyncState,
    LayerPresentationHints,
)
from echozero.application.transport.models import TransportState
from echozero.application.mixer.models import MixerState, LayerMixerState
from echozero.application.playback.models import PlaybackState, PlaybackSource, LayerPlaybackState
from echozero.application.sync.models import SyncState
from echozero.infrastructure.persistence.records import (
    ProjectRecord,
    SongRecord,
    SongVersionRecord,
    SessionRecord,
    TimelineRecord,
    LayerRecord,
    TakeRecord,
    EventRecord,
)


def to_project(record: ProjectRecord) -> Project:
    return Project(
        id=ProjectId(record.id),
        name=record.name,
        songs=[SongId(song_id) for song_id in record.songs],
        active_song_id=SongId(record.active_song_id) if record.active_song_id else None,
        session_id=SessionId(record.session_id) if record.session_id else None,
    )


def to_project_record(project: Project) -> ProjectRecord:
    return ProjectRecord(
        id=str(project.id),
        name=project.name,
        songs=[str(song_id) for song_id in project.songs],
        active_song_id=str(project.active_song_id) if project.active_song_id else None,
        session_id=str(project.session_id) if project.session_id else None,
    )


def to_song(record: SongRecord) -> Song:
    return Song(
        id=SongId(record.id),
        project_id=ProjectId(record.project_id),
        title=record.title,
        versions=[SongVersionId(version_id) for version_id in record.versions],
        active_version_id=SongVersionId(record.active_version_id) if record.active_version_id else None,
    )


def to_song_record(song: Song) -> SongRecord:
    return SongRecord(
        id=str(song.id),
        project_id=str(song.project_id),
        title=song.title,
        versions=[str(version_id) for version_id in song.versions],
        active_version_id=str(song.active_version_id) if song.active_version_id else None,
    )


def to_song_version(record: SongVersionRecord) -> SongVersion:
    return SongVersion(
        id=SongVersionId(record.id),
        song_id=SongId(record.song_id),
        name=record.name,
        timeline_id=TimelineId(record.timeline_id),
        layer_order=[LayerId(layer_id) for layer_id in record.layer_order],
    )


def to_song_version_record(song_version: SongVersion) -> SongVersionRecord:
    return SongVersionRecord(
        id=str(song_version.id),
        song_id=str(song_version.song_id),
        name=song_version.name,
        timeline_id=str(song_version.timeline_id),
        layer_order=[str(layer_id) for layer_id in song_version.layer_order],
    )


def _to_event(record: EventRecord) -> Event:
    return Event(
        id=EventId(record.id),
        take_id=TakeId(record.take_id),
        start=record.start,
        end=record.end,
        payload_ref=record.payload_ref,
        label=record.label,
        color=record.color,
        muted=record.muted,
    )


def _to_event_record(event: Event) -> EventRecord:
    return EventRecord(
        id=str(event.id),
        take_id=str(event.take_id),
        start=event.start,
        end=event.end,
        payload_ref=event.payload_ref,
        label=event.label,
        color=event.color,
        muted=event.muted,
    )


def _to_take(record: TakeRecord) -> Take:
    return Take(
        id=TakeId(record.id),
        layer_id=LayerId(record.layer_id),
        name=record.name,
        version_label=record.version_label,
        events=[_to_event(event) for event in record.events],
        source_ref=record.source_ref,
        available=record.available,
        is_comped=record.is_comped,
    )


def _to_take_record(take: Take) -> TakeRecord:
    return TakeRecord(
        id=str(take.id),
        layer_id=str(take.layer_id),
        name=take.name,
        version_label=take.version_label,
        events=[_to_event_record(event) for event in take.events],
        source_ref=take.source_ref,
        available=take.available,
        is_comped=take.is_comped,
    )


def _to_layer(record: LayerRecord) -> Layer:
    mixer_data = record.mixer
    playback_data = record.playback
    sync_data = record.sync
    hints_data = record.presentation_hints
    return Layer(
        id=LayerId(record.id),
        timeline_id=TimelineId(record.timeline_id),
        name=record.name,
        kind=LayerKind(record.kind),
        order_index=record.order_index,
        takes=[_to_take(take) for take in record.takes],
        active_take_id=TakeId(record.active_take_id) if record.active_take_id else None,
        mixer=LayerMixerState(
            mute=mixer_data.get("mute", False),
            solo=mixer_data.get("solo", False),
            gain_db=mixer_data.get("gain_db", 0.0),
            pan=mixer_data.get("pan", 0.0),
            output_bus=mixer_data.get("output_bus"),
        ),
        playback=LayerPlaybackState(
            mode=PlaybackMode(playback_data.get("mode", PlaybackMode.NONE.value)),
            enabled=playback_data.get("enabled", False),
            armed_source_ref=playback_data.get("armed_source_ref"),
            preloaded=playback_data.get("preloaded", False),
            supports_scrub=playback_data.get("supports_scrub", False),
            supports_loop=playback_data.get("supports_loop", True),
        ),
        sync=LayerSyncState(
            mode=sync_data.get("mode", SyncMode.NONE.value),
            connected=sync_data.get("connected", False),
            offset_ms=sync_data.get("offset_ms", 0.0),
            target_ref=sync_data.get("target_ref"),
            show_manager_block_id=sync_data.get("show_manager_block_id"),
            ma3_track_coord=sync_data.get("ma3_track_coord"),
            derived_from_source=sync_data.get("derived_from_source", False),
        ),
        presentation_hints=LayerPresentationHints(
            visible=hints_data.get("visible", True),
            locked=hints_data.get("locked", False),
            expanded=hints_data.get(
                "expanded",
                hints_data.get("take_selector_expanded", not hints_data.get("collapsed", False)),
            ),
            height=hints_data.get("height", 40.0),
            color=hints_data.get("color"),
            group_id=hints_data.get("group_id"),
            group_name=hints_data.get("group_name"),
            group_index=hints_data.get("group_index"),
            show_take_selector=hints_data.get("show_take_selector", True),
            show_take_lane=hints_data.get("show_take_lane", False),
        ),
    )


def _to_layer_record(layer: Layer) -> LayerRecord:
    return LayerRecord(
        id=str(layer.id),
        timeline_id=str(layer.timeline_id),
        name=layer.name,
        kind=layer.kind.value,
        order_index=layer.order_index,
        takes=[_to_take_record(take) for take in layer.takes],
        active_take_id=str(layer.active_take_id) if layer.active_take_id else None,
        mixer={
            "mute": layer.mixer.mute,
            "solo": layer.mixer.solo,
            "gain_db": layer.mixer.gain_db,
            "pan": layer.mixer.pan,
            "output_bus": layer.mixer.output_bus,
        },
        playback={
            "mode": layer.playback.mode.value,
            "enabled": layer.playback.enabled,
            "armed_source_ref": layer.playback.armed_source_ref,
            "preloaded": layer.playback.preloaded,
            "supports_scrub": layer.playback.supports_scrub,
            "supports_loop": layer.playback.supports_loop,
        },
        sync={
            "mode": layer.sync.mode,
            "connected": layer.sync.connected,
            "offset_ms": layer.sync.offset_ms,
            "target_ref": layer.sync.target_ref,
            "show_manager_block_id": layer.sync.show_manager_block_id,
            "ma3_track_coord": layer.sync.ma3_track_coord,
            "derived_from_source": layer.sync.derived_from_source,
        },
        presentation_hints={
            "visible": layer.presentation_hints.visible,
            "locked": layer.presentation_hints.locked,
            "expanded": layer.presentation_hints.expanded,
            "height": layer.presentation_hints.height,
            "color": layer.presentation_hints.color,
            "group_id": layer.presentation_hints.group_id,
            "group_name": layer.presentation_hints.group_name,
            "group_index": layer.presentation_hints.group_index,
            "show_take_selector": layer.presentation_hints.show_take_selector,
            "show_take_lane": layer.presentation_hints.show_take_lane,
        },
    )


def to_timeline(record: TimelineRecord) -> Timeline:
    selection = record.selection
    viewport = record.viewport
    loop_region = record.loop_region
    return Timeline(
        id=TimelineId(record.id),
        song_version_id=SongVersionId(record.song_version_id),
        start=record.start,
        end=record.end,
        layers=[_to_layer(layer) for layer in record.layers],
        loop_region=TimeRange(loop_region["start"], loop_region["end"]) if loop_region else None,
        selection=TimelineSelection(
            selected_layer_id=LayerId(selection["selected_layer_id"]) if selection.get("selected_layer_id") else None,
            selected_take_id=TakeId(selection["selected_take_id"]) if selection.get("selected_take_id") else None,
            selected_event_ids=[EventId(event_id) for event_id in selection.get("selected_event_ids", [])],
        ),
        viewport=TimelineViewport(
            pixels_per_second=viewport.get("pixels_per_second", 100.0),
            scroll_x=viewport.get("scroll_x", 0.0),
            scroll_y=viewport.get("scroll_y", 0.0),
        ),
    )


def to_timeline_record(timeline: Timeline) -> TimelineRecord:
    return TimelineRecord(
        id=str(timeline.id),
        song_version_id=str(timeline.song_version_id),
        start=timeline.start,
        end=timeline.end,
        layers=[_to_layer_record(layer) for layer in timeline.layers],
        loop_region=(
            {"start": timeline.loop_region.start, "end": timeline.loop_region.end}
            if timeline.loop_region else None
        ),
        selection={
            "selected_layer_id": str(timeline.selection.selected_layer_id) if timeline.selection.selected_layer_id else None,
            "selected_take_id": str(timeline.selection.selected_take_id) if timeline.selection.selected_take_id else None,
            "selected_event_ids": [str(event_id) for event_id in timeline.selection.selected_event_ids],
        },
        viewport={
            "pixels_per_second": timeline.viewport.pixels_per_second,
            "scroll_x": timeline.viewport.scroll_x,
            "scroll_y": timeline.viewport.scroll_y,
        },
    )


def to_session(record: SessionRecord) -> Session:
    transport = record.transport
    mixer = record.mixer
    playback = record.playback
    sync = record.sync
    return Session(
        id=SessionId(record.id),
        project_id=ProjectId(record.project_id),
        active_song_id=SongId(record.active_song_id) if record.active_song_id else None,
        active_song_version_id=SongVersionId(record.active_song_version_id) if record.active_song_version_id else None,
        active_timeline_id=TimelineId(record.active_timeline_id) if record.active_timeline_id else None,
        transport_state=TransportState(
            is_playing=transport.get("is_playing", False),
            playhead=transport.get("playhead", 0.0),
            loop_enabled=transport.get("loop_enabled", False),
            loop_region=(
                TimeRange(transport["loop_region"]["start"], transport["loop_region"]["end"])
                if transport.get("loop_region") else None
            ),
            preroll_enabled=transport.get("preroll_enabled", False),
            follow_mode=FollowMode(transport.get("follow_mode", FollowMode.PAGE.value)),
        ),
        mixer_state=MixerState(
            master_gain_db=mixer.get("master_gain_db", 0.0),
            layer_states={
                LayerId(layer_id): LayerMixerState(
                    mute=state.get("mute", False),
                    solo=state.get("solo", False),
                    gain_db=state.get("gain_db", 0.0),
                    pan=state.get("pan", 0.0),
                    output_bus=state.get("output_bus"),
                )
                for layer_id, state in mixer.get("layer_states", {}).items()
            },
        ),
        playback_state=PlaybackState(
            status=PlaybackStatus(playback.get("status", PlaybackStatus.STOPPED.value)),
            active_sources=[
                PlaybackSource(
                    layer_id=LayerId(source["layer_id"]),
                    take_id=TakeId(source["take_id"]) if source.get("take_id") else None,
                    source_ref=source.get("source_ref"),
                    mode=PlaybackMode(source.get("mode", PlaybackMode.NONE.value)),
                )
                for source in playback.get("active_sources", [])
            ],
            latency_ms=playback.get("latency_ms", 0.0),
            backend_name=playback.get("backend_name", "unconfigured"),
        ),
        sync_state=SyncState(
            mode=SyncMode(sync.get("mode", SyncMode.NONE.value)),
            connected=sync.get("connected", False),
            leader_follower_state=sync.get("leader_follower_state", "standalone"),
            offset_ms=sync.get("offset_ms", 0.0),
            target_ref=sync.get("target_ref"),
            health=sync.get("health", "unknown"),
        ),
        ui_prefs_ref=record.ui_prefs_ref,
    )


def to_session_record(session: Session) -> SessionRecord:
    return SessionRecord(
        id=str(session.id),
        project_id=str(session.project_id),
        active_song_id=str(session.active_song_id) if session.active_song_id else None,
        active_song_version_id=str(session.active_song_version_id) if session.active_song_version_id else None,
        active_timeline_id=str(session.active_timeline_id) if session.active_timeline_id else None,
        ui_prefs_ref=session.ui_prefs_ref,
        transport={
            "is_playing": session.transport_state.is_playing,
            "playhead": session.transport_state.playhead,
            "loop_enabled": session.transport_state.loop_enabled,
            "loop_region": (
                {
                    "start": session.transport_state.loop_region.start,
                    "end": session.transport_state.loop_region.end,
                }
                if session.transport_state.loop_region else None
            ),
            "preroll_enabled": session.transport_state.preroll_enabled,
            "follow_mode": session.transport_state.follow_mode.value,
        },
        mixer={
            "master_gain_db": session.mixer_state.master_gain_db,
            "layer_states": {
                str(layer_id): {
                    "mute": state.mute,
                    "solo": state.solo,
                    "gain_db": state.gain_db,
                    "pan": state.pan,
                    "output_bus": state.output_bus,
                }
                for layer_id, state in session.mixer_state.layer_states.items()
            },
        },
        playback={
            "status": session.playback_state.status.value,
            "active_sources": [
                {
                    "layer_id": str(source.layer_id),
                    "take_id": str(source.take_id) if source.take_id else None,
                    "source_ref": source.source_ref,
                    "mode": source.mode.value,
                }
                for source in session.playback_state.active_sources
            ],
            "latency_ms": session.playback_state.latency_ms,
            "backend_name": session.playback_state.backend_name,
        },
        sync={
            "mode": session.sync_state.mode.value,
            "connected": session.sync_state.connected,
            "leader_follower_state": session.sync_state.leader_follower_state,
            "offset_ms": session.sync_state.offset_ms,
            "target_ref": session.sync_state.target_ref,
            "health": session.sync_state.health,
        },
    )
