"""Runtime timeline app composition for the Qt shell.
Exists to isolate in-memory service adapters from AppShellRuntime behavior.
Connects ProjectStorage baseline timelines to TimelineApplication for local app use.
"""

from __future__ import annotations

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.application.shared.enums import PlaybackStatus, SyncMode
from echozero.application.shared.ids import ProjectId, SessionId
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter, MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Layer
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.timeline.queries import TimelineQueries
from echozero.audio.engine import AudioEngine
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline import (
    apply_timeline_presentation_overlay,
    build_project_native_baseline_timeline,
)
from echozero.ui.qt.app_shell_project_runtime_state import (
    load_project_runtime_state,
)
from echozero.ui.qt.timeline.runtime_audio import (
    TimelinePlaybackController,
    TimelineRuntimeAudioController,
)


class RuntimeSessionService(SessionService):
    """Session adapter backed directly by the current in-memory app session."""

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


class RuntimeTransportService(TransportService):
    """Transport adapter that mutates the current session transport state."""

    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> TransportState:
        return self._session.transport_state

    def play(self) -> TransportState:
        self._session.transport_state.is_playing = True
        return self._session.transport_state

    def pause(self) -> TransportState:
        self._session.transport_state.is_playing = False
        return self._session.transport_state

    def stop(self) -> TransportState:
        self._session.transport_state.is_playing = False
        self._session.transport_state.playhead = 0.0
        return self._session.transport_state

    def seek(self, position: float) -> TransportState:
        self._session.transport_state.playhead = max(0.0, position)
        return self._session.transport_state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._session.transport_state.loop_region = loop_region if enabled else None
        return self._session.transport_state


class RuntimeMixerService(MixerService):
    """Mixer adapter backed by the current session mixer state."""

    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> MixerState:
        return self._session.mixer_state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._session.mixer_state.layer_states[layer_id] = state
        return self._session.mixer_state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.mute = muted
        return self._session.mixer_state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.solo = soloed
        return self._session.mixer_state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.gain_db = gain_db
        return self._session.mixer_state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.pan = pan
        return self._session.mixer_state

    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        return [AudibilityState(layer_id=layer.id, is_audible=True, reason="normal") for layer in layers]


class RuntimePlaybackService(PlaybackService):
    """Playback adapter that mirrors runtime state into the session."""

    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(self, timeline, transport: TransportState, audibility, sync: SyncState) -> PlaybackState:
        self._session.playback_state.status = (
            PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        )
        return self._session.playback_state

    def stop(self) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.STOPPED
        return self._session.playback_state


def build_playback_controller(
    audio_output_config: AudioOutputRuntimeConfig | None = None,
) -> TimelinePlaybackController:
    """Build the canonical playback controller for one optional audio output config."""

    if audio_output_config is None:
        return TimelinePlaybackController()

    def _engine_factory() -> AudioEngine:
        return AudioEngine(
            sample_rate=audio_output_config.sample_rate,
            channels=audio_output_config.channels,
            stream_latency=audio_output_config.stream_latency,
            stream_blocksize=audio_output_config.stream_blocksize,
            prime_output_buffers_using_stream_callback=(
                audio_output_config.prime_output_buffers_using_stream_callback
            ),
            output_device=audio_output_config.output_device,
        )

    return TimelinePlaybackController(
        engine_factory=_engine_factory,
        preview_engine_factory=_engine_factory,
    )


def build_runtime_audio_controller(
    audio_output_config: AudioOutputRuntimeConfig | None = None,
) -> TimelineRuntimeAudioController:
    """Compatibility alias for callers that still say `build_runtime_audio_controller`."""

    return build_playback_controller(audio_output_config)


def _coerce_project_ma3_push_offset_seconds(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def build_runtime_timeline_application(
    *,
    project_storage: ProjectStorage,
    sync_bridge: MA3SyncBridge | None,
    sync_service: SyncService | None,
    runtime_audio=None,
    audio_output_config: AudioOutputRuntimeConfig | None = None,
) -> TimelineApplication:
    """Build the in-memory timeline application for the app shell runtime."""
    if runtime_audio is None:
        runtime_audio = build_playback_controller(audio_output_config)
    runtime_state = load_project_runtime_state(project_storage)
    timeline, overlay, active_song_id, active_song_version_id = build_project_native_baseline_timeline(
        project_storage,
        active_song_id=runtime_state.active_song_id,
        active_song_version_id=runtime_state.active_song_version_id,
    )
    timeline.viewport.pixels_per_second = runtime_state.pixels_per_second
    timeline.viewport.scroll_x = runtime_state.scroll_x
    timeline.viewport.scroll_y = runtime_state.scroll_y
    max_playhead = max(0.0, float(timeline.end))
    playhead = (
        min(runtime_state.playhead, max_playhead)
        if max_playhead > 0.0
        else runtime_state.playhead
    )
    active_version = (
        project_storage.song_versions.get(str(active_song_version_id))
        if active_song_version_id is not None
        else None
    )
    session = Session(
        id=SessionId(f"session_{project_storage.project.id}"),
        project_id=ProjectId(project_storage.project.id),
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
        active_song_version_ma3_timecode_pool_no=(
            active_version.ma3_timecode_pool_no if active_version is not None else None
        ),
        project_ma3_push_offset_seconds=_coerce_project_ma3_push_offset_seconds(
            project_storage.project.settings.ma3_push_offset_seconds
        ),
        active_timeline_id=timeline.id,
        transport_state=TransportState(
            is_playing=False,
            playhead=max(0.0, float(playhead)),
        ),
        mixer_state=MixerState(),
        playback_state=PlaybackState(
            status=PlaybackStatus.STOPPED,
            backend_name="sounddevice",
        ),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref="show_manager"),
    )

    runtime_sync_service: SyncService
    if sync_service is not None:
        runtime_sync_service = sync_service
    elif sync_bridge is not None:
        runtime_sync_service = MA3SyncAdapter(sync_bridge, state=session.sync_state, target_ref="show_manager")
    else:
        runtime_sync_service = InMemorySyncService(session.sync_state)

    assembler = TimelineAssembler()
    return TimelineApplication(
        timeline=timeline,
        session=session,
        orchestrator=TimelineOrchestrator(
            session_service=RuntimeSessionService(session),
            transport_service=RuntimeTransportService(session),
            mixer_service=RuntimeMixerService(session),
            playback_service=RuntimePlaybackService(session),
            sync_service=runtime_sync_service,
            assembler=assembler,
        ),
        queries=TimelineQueries(assembler=assembler),
        sync_service=runtime_sync_service,
        runtime_audio=runtime_audio,
        presentation_enricher=lambda presentation: apply_timeline_presentation_overlay(
            presentation,
            overlay=overlay,
        ),
    )
