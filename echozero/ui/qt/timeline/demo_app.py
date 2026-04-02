"""Runnable demo/testing loop for the new read-only timeline shell."""

from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication

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
from echozero.application.project.models import Project
from echozero.application.song.models import Song, SongVersion
from echozero.application.session.models import Session
from echozero.application.timeline.models import Timeline, Layer, Take, Event, TimelineSelection, TimelineViewport, LayerSyncState, LayerPresentationHints
from echozero.application.transport.models import TransportState
from echozero.application.mixer.models import MixerState, LayerMixerState
from echozero.application.playback.models import PlaybackState, LayerPlaybackState
from echozero.application.sync.models import SyncState
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.ui.qt.timeline.widget import TimelineWidget


def build_demo_timeline():
    timeline_id = TimelineId("timeline_demo")
    layer_a = Layer(
        id=LayerId("layer_drums"),
        timeline_id=timeline_id,
        name="Drums",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[
            Take(
                id=TakeId("take_drums_main"),
                layer_id=LayerId("layer_drums"),
                name="Main",
                events=[
                    Event(id=EventId("e1"), take_id=TakeId("take_drums_main"), start=0.5, end=0.8, label="Kick", color="#66a3ff"),
                    Event(id=EventId("e2"), take_id=TakeId("take_drums_main"), start=1.2, end=1.45, label="Snare", color="#7fd1ae"),
                    Event(id=EventId("e3"), take_id=TakeId("take_drums_main"), start=2.0, end=2.35, label="Crash", color="#f8c555"),
                ],
            ),
            Take(
                id=TakeId("take_drums_alt"),
                layer_id=LayerId("layer_drums"),
                name="Alt",
                events=[],
            ),
        ],
        active_take_id=TakeId("take_drums_main"),
        mixer=LayerMixerState(gain_db=-1.5),
        playback=LayerPlaybackState(mode=PlaybackMode.EVENT_SLICE, enabled=True),
        presentation_hints=LayerPresentationHints(color="#66a3ff", take_selector_expanded=True),
    )

    layer_b = Layer(
        id=LayerId("layer_bass"),
        timeline_id=timeline_id,
        name="Bass",
        kind=LayerKind.AUDIO,
        order_index=1,
        takes=[
            Take(
                id=TakeId("take_bass_1"),
                layer_id=LayerId("layer_bass"),
                name="Stem A",
                events=[
                    Event(id=EventId("e4"), take_id=TakeId("take_bass_1"), start=0.0, end=3.5, label="Bass Stem", color="#d68cff"),
                ],
                source_ref="bass_stem.wav",
            )
        ],
        active_take_id=TakeId("take_bass_1"),
        mixer=LayerMixerState(solo=True, gain_db=-3.0),
        playback=LayerPlaybackState(mode=PlaybackMode.CONTINUOUS_AUDIO, enabled=True),
        presentation_hints=LayerPresentationHints(color="#d68cff"),
    )

    layer_c = Layer(
        id=LayerId("layer_sync"),
        timeline_id=timeline_id,
        name="MA3 Sync",
        kind=LayerKind.SYNC,
        order_index=2,
        takes=[
            Take(
                id=TakeId("take_sync_1"),
                layer_id=LayerId("layer_sync"),
                name="Main",
                events=[
                    Event(id=EventId("e5"), take_id=TakeId("take_sync_1"), start=0.75, end=1.05, label="Cue 101", color="#ff8c78"),
                    Event(id=EventId("e6"), take_id=TakeId("take_sync_1"), start=2.1, end=2.4, label="Cue 102", color="#ff8c78"),
                ],
            )
        ],
        active_take_id=TakeId("take_sync_1"),
        mixer=LayerMixerState(mute=True),
        playback=LayerPlaybackState(mode=PlaybackMode.NONE, enabled=False),
        sync=LayerSyncState(mode=SyncMode.MA3.value, connected=True, target_ref="show_manager", ma3_track_coord="tc101"),
        presentation_hints=LayerPresentationHints(color="#ff8c78", locked=True),
    )

    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId("song_version_demo"),
        start=0.0,
        end=8.0,
        layers=[layer_a, layer_b, layer_c],
        selection=TimelineSelection(
            selected_layer_id=LayerId("layer_drums"),
            selected_take_id=TakeId("take_drums_main"),
            selected_event_ids=[EventId("e2")],
        ),
        viewport=TimelineViewport(pixels_per_second=180.0, scroll_x=0.0, scroll_y=0.0),
    )

    session = Session(
        id=SessionId("session_demo"),
        project_id=ProjectId("project_demo"),
        active_song_id=SongId("song_demo"),
        active_song_version_id=SongVersionId("song_version_demo"),
        active_timeline_id=timeline_id,
        transport_state=TransportState(is_playing=True, playhead=1.35, follow_mode=FollowMode.PAGE),
        mixer_state=MixerState(),
        playback_state=PlaybackState(status=PlaybackStatus.PLAYING, backend_name="demo"),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref="show_manager"),
    )

    return timeline, session


def main() -> int:
    app = QApplication(sys.argv)
    timeline, session = build_demo_timeline()
    presentation = TimelineAssembler().assemble(timeline, session)
    widget = TimelineWidget(presentation)
    widget.resize(1400, 420)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
