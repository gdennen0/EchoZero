"""Visual Lab current-state factories.
Exists to build lab preview data through the current timeline application contract.
The lab owns sample values, while production models and assemblers own presentation shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.presentation.models import (
    LayerPresentation,
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TimelinePresentation,
)
from echozero.application.mixer.models import LayerMixerState
from echozero.application.playback.models import LayerPlaybackState
from echozero.application.session.models import Session
from echozero.application.shared.enums import FollowMode, LayerKind, PlaybackMode
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ProjectId,
    SectionCueId,
    SessionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import (
    Event,
    EventRef,
    Layer,
    LayerPresentationHints,
    LayerProvenance,
    LayerStatus,
    LayerSyncState,
    SectionCue,
    Take,
    Timeline,
    TimelineSelection,
    TimelineViewport,
)
from echozero.application.transport.models import TransportState

from dev.visual_lab.waveforms import (
    FUN_WAVEFORM_KEY,
    FUN_WAVEFORM_SOURCE,
    register_fun_waveform_preview,
)


SOURCE_AUDIO_LAYER_ID = LayerId("current_source_audio")
VOCALS_LAYER_ID = LayerId("current_stem_vocals")
BASS_LAYER_ID = LayerId("current_stem_bass")
DRUMS_LAYER_ID = LayerId("current_drum_events")
CUES_LAYER_ID = LayerId("current_section_cues")
SELECTED_EVENT_ID = EventId("current_snare_1")


@dataclass(frozen=True, slots=True)
class CurrentVisualLabState:
    """Canonical timeline/session pair used to assemble current Visual Lab previews."""

    timeline: Timeline
    session: Session


def build_current_visual_lab_state() -> CurrentVisualLabState:
    """Build a compact current timeline model with representative UI states."""

    timeline_id = TimelineId("current_visual_lab_timeline")
    song_version_id = SongVersionId("current_visual_lab_song_version")
    selected_take_id = TakeId("current_drums_main")
    timeline = Timeline(
        id=timeline_id,
        song_version_id=song_version_id,
        end=48.0,
        layers=[
            _source_audio_layer(timeline_id),
            _stem_layer(
                timeline_id=timeline_id,
                layer_id=VOCALS_LAYER_ID,
                take_id=TakeId("current_vocals_main"),
                name="Vocals stem",
                color="#d86bd6",
                events=[
                    _event("current_vocal_phrase_a", "current_vocals_main", 8.0, 13.2, "phrase"),
                    _event("current_vocal_phrase_b", "current_vocals_main", 24.0, 31.0, "hook"),
                ],
                order_index=1,
            ),
            _stem_layer(
                timeline_id=timeline_id,
                layer_id=BASS_LAYER_ID,
                take_id=TakeId("current_bass_main"),
                name="Bass stem",
                color="#7f7cff",
                events=[
                    _event("current_bass_a", "current_bass_main", 4.0, 18.0, "bass", muted=True),
                    _event("current_bass_b", "current_bass_main", 22.0, 38.0, "bass", muted=True),
                ],
                order_index=2,
                muted=True,
            ),
            _drum_event_layer(timeline_id),
            _section_cue_layer(timeline_id),
        ],
        section_cues=[
            SectionCue(
                id=SectionCueId("current_section_intro"),
                start=0.0,
                cue_ref="1",
                name="Intro",
                color="#58c4dd",
            ),
            SectionCue(
                id=SectionCueId("current_section_drop"),
                start=32.0,
                cue_ref="2",
                name="Drop",
                color="#f0b74f",
            ),
        ],
        selection=TimelineSelection(
            selected_layer_id=DRUMS_LAYER_ID,
            selected_layer_ids=[DRUMS_LAYER_ID],
            selected_take_id=selected_take_id,
            selected_event_refs=[
                EventRef(
                    layer_id=DRUMS_LAYER_ID,
                    take_id=selected_take_id,
                    event_id=SELECTED_EVENT_ID,
                )
            ],
            selected_event_ids=[SELECTED_EVENT_ID],
        ),
        viewport=TimelineViewport(pixels_per_second=16.0),
    )
    session = Session(
        id=SessionId("current_visual_lab_session"),
        project_id=ProjectId("current_visual_lab_project"),
        active_song_id=SongId("current_visual_lab_song"),
        active_song_version_id=song_version_id,
        active_song_version_ma3_timecode_pool_no=2,
        active_timeline_id=timeline_id,
        transport_state=TransportState(
            is_playing=True,
            playhead=18.5,
            follow_mode=FollowMode.CENTER,
        ),
    )
    return CurrentVisualLabState(timeline=timeline, session=session)


def build_current_visual_lab_presentation() -> TimelinePresentation:
    """Assemble the current Visual Lab presentation through TimelineAssembler."""

    state = build_current_visual_lab_state()
    presentation = TimelineAssembler().assemble(state.timeline, state.session)
    presentation.title = "EchoZero current visual preview"
    presentation.active_song_id = str(state.session.active_song_id or "")
    presentation.active_song_title = "Current app preview"
    presentation.active_song_version_id = str(state.session.active_song_version_id or "")
    presentation.active_song_version_label = "main"
    presentation.active_song_version_ma3_timecode_pool_no = (
        state.session.active_song_version_ma3_timecode_pool_no
    )
    presentation.available_songs = [
        SongOptionPresentation(
            song_id=str(state.session.active_song_id),
            title="Current app preview",
            is_active=True,
            active_version_id=str(state.session.active_song_version_id),
            active_version_label="main",
            version_count=2,
            versions=[
                SongVersionOptionPresentation(
                    song_version_id=str(state.session.active_song_version_id),
                    label="main",
                    is_active=True,
                    ma3_timecode_pool_no=2,
                ),
                SongVersionOptionPresentation(
                    song_version_id="current_visual_lab_song_version_alt",
                    label="lighting notes",
                    ma3_timecode_pool_no=3,
                ),
            ],
        ),
        SongOptionPresentation(
            song_id="current_visual_lab_song_2",
            title="Tiny weird encore",
            active_version_id="current_visual_lab_song_2_main",
            active_version_label="main",
            version_count=1,
        ),
        SongOptionPresentation(
            song_id="current_visual_lab_song_3",
            title="Sine wave confetti",
            active_version_id="current_visual_lab_song_3_main",
            active_version_label="main",
            version_count=1,
        ),
    ]
    presentation.available_song_versions = presentation.available_songs[0].versions
    presentation.current_time_label = "00:18.50"
    presentation.end_time_label = "00:48.00"
    _attach_lab_waveform_sources(presentation)
    return presentation


def current_layer_by_id(layer_id: LayerId | str) -> LayerPresentation:
    """Return one assembled current Visual Lab layer by id."""

    presentation = build_current_visual_lab_presentation()
    for layer in presentation.layers:
        if str(layer.layer_id) == str(layer_id):
            return layer
    raise KeyError(f"unknown current Visual Lab layer id: {layer_id}")


def _source_audio_layer(timeline_id: TimelineId) -> Layer:
    take_id = TakeId("current_source_main")
    return Layer(
        id=SOURCE_AUDIO_LAYER_ID,
        timeline_id=timeline_id,
        name="Main audio",
        kind=LayerKind.AUDIO,
        order_index=0,
        takes=[
            Take(
                id=take_id,
                layer_id=SOURCE_AUDIO_LAYER_ID,
                name="Main",
                source_ref="local audio",
                events=[
                    _event("current_section_a", take_id, 0.0, 16.0, "A"),
                    _event("current_section_b", take_id, 16.0, 32.0, "B"),
                    _event("current_section_c", take_id, 32.0, 48.0, "C"),
                ],
            )
        ],
        playback=LayerPlaybackState(
            mode=PlaybackMode.CONTINUOUS_AUDIO,
            enabled=True,
            armed_source_ref="local audio",
        ),
        sync=LayerSyncState(mode="none", connected=False),
        presentation_hints=LayerPresentationHints(
            expanded=True,
            color="#58c4dd",
        ),
    )


def _stem_layer(
    *,
    timeline_id: TimelineId,
    layer_id: LayerId,
    take_id: TakeId,
    name: str,
    color: str,
    events: list[Event],
    order_index: int,
    muted: bool = False,
) -> Layer:
    return Layer(
        id=layer_id,
        timeline_id=timeline_id,
        name=name,
        kind=LayerKind.AUDIO,
        order_index=order_index,
        parent_layer_id=SOURCE_AUDIO_LAYER_ID,
        takes=[
            Take(
                id=take_id,
                layer_id=layer_id,
                name="Main",
                source_ref="stem split",
                events=events,
            )
        ],
        mixer=LayerMixerState(mute=muted),
        playback=LayerPlaybackState(
            mode=PlaybackMode.CONTINUOUS_AUDIO,
            enabled=not muted,
            armed_source_ref="stem split",
        ),
        provenance=LayerProvenance(
            source_layer_id=SOURCE_AUDIO_LAYER_ID,
            source_song_version_id=SongVersionId("current_visual_lab_song_version"),
            pipeline_id="timeline.extract_stems",
            output_name=name.split()[0].lower(),
        ),
        presentation_hints=LayerPresentationHints(color=color),
    )


def _drum_event_layer(timeline_id: TimelineId) -> Layer:
    take_id = TakeId("current_drums_main")
    return Layer(
        id=DRUMS_LAYER_ID,
        timeline_id=timeline_id,
        name="Drum hits",
        kind=LayerKind.EVENT,
        order_index=3,
        takes=[
            Take(
                id=take_id,
                layer_id=DRUMS_LAYER_ID,
                name="Main",
                events=[
                    _event("current_kick_1", take_id, 2.0, 2.24, "kick"),
                    _event(SELECTED_EVENT_ID, take_id, 6.0, 6.22, "snare"),
                    _event("current_hat_1", take_id, 9.0, 9.12, "hat"),
                    _event("current_fill", take_id, 34.0, 36.0, "fill"),
                ],
            )
        ],
        playback=LayerPlaybackState(mode=PlaybackMode.EVENT_TONE, enabled=True),
        sync=LayerSyncState(
            mode="ma3",
            connected=True,
            ma3_track_coord="TC2/TG1/T14",
        ),
        provenance=LayerProvenance(
            source_layer_id=SOURCE_AUDIO_LAYER_ID,
            source_song_version_id=SongVersionId("current_visual_lab_song_version"),
            pipeline_id="timeline.extract_drum_events",
            output_name="drums",
        ),
        status=LayerStatus(manually_modified=True),
        presentation_hints=LayerPresentationHints(color="#ff6b68"),
    )


def _section_cue_layer(timeline_id: TimelineId) -> Layer:
    take_id = TakeId("current_cues_main")
    return Layer(
        id=CUES_LAYER_ID,
        timeline_id=timeline_id,
        name="Section cues",
        kind=LayerKind.SECTION,
        order_index=4,
        takes=[
            Take(
                id=take_id,
                layer_id=CUES_LAYER_ID,
                name="Main",
                events=[
                    _event("current_cue_intro", take_id, 0.0, 0.45, "Intro", cue_ref="1"),
                    _event("current_cue_drop", take_id, 32.0, 32.45, "Drop", cue_ref="2"),
                ],
            )
        ],
        sync=LayerSyncState(
            mode="ma3",
            connected=True,
            ma3_track_coord="TC2/TG1/T15",
        ),
        status=LayerStatus(stale=True, stale_reason="source layer changed"),
        presentation_hints=LayerPresentationHints(color="#f0b74f"),
    )


def _event(
    event_id: EventId | str,
    take_id: TakeId | str,
    start: float,
    end: float,
    label: str,
    *,
    muted: bool = False,
    cue_ref: str | None = None,
) -> Event:
    return Event(
        id=EventId(str(event_id)),
        take_id=TakeId(str(take_id)),
        start=start,
        end=end,
        label=label,
        muted=muted,
        cue_ref=cue_ref,
    )


def _attach_lab_waveform_sources(presentation: TimelinePresentation) -> None:
    register_fun_waveform_preview()
    for layer in presentation.layers:
        if str(layer.layer_id) == str(SOURCE_AUDIO_LAYER_ID):
            layer.source_audio_path = FUN_WAVEFORM_SOURCE
            layer.waveform_key = FUN_WAVEFORM_KEY
            layer.playback_source_ref = FUN_WAVEFORM_SOURCE
        if layer.kind is LayerKind.AUDIO and layer.playback_source_ref:
            layer.source_audio_path = layer.source_audio_path or FUN_WAVEFORM_SOURCE
            layer.waveform_key = layer.waveform_key or FUN_WAVEFORM_KEY
