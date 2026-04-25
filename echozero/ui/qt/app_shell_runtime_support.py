"""Runtime support helpers for the Qt app shell.
Exists to isolate service wiring, playback sync, and preview/layer lookup helpers.
Connects app-shell orchestration to runtime audio, object actions, and shutdown.
"""

from __future__ import annotations

from typing import Protocol

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.settings import AudioOutputRuntimeConfig, AppSettingsService
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.sync.adapters import MA3SyncBridge
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.object_actions import ObjectActionService
from echozero.application.timeline.pipeline_run_service import PipelineRunService
from echozero.infrastructure.osc import OscUdpSendTransport
from echozero.persistence.session import ProjectStorage
from echozero.services.orchestrator import Orchestrator
from echozero.ui.qt.app_shell_runtime_services import build_runtime_audio_controller
from echozero.ui.qt.app_shell_timeline_state import clear_selected_events, resolve_event_clip_preview


class RuntimeAudioController(Protocol):
    def build_for_presentation(self, presentation: TimelinePresentation) -> None: ...

    def apply_mix_state(self, presentation: TimelinePresentation) -> None: ...

    def play(self) -> None: ...

    def pause(self) -> None: ...

    def stop(self) -> None: ...

    def seek(self, position_seconds: float) -> None: ...

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool: ...

    def current_time_seconds(self) -> float: ...

    def is_playing(self) -> bool: ...

    def shutdown(self) -> None: ...


class RuntimeSupportShell(Protocol):
    _app: TimelineApplication
    _analysis_service: Orchestrator
    _object_action_settings: ObjectActionService
    _pipeline_runs: PipelineRunService
    _sync_bridge: MA3SyncBridge | None
    _app_settings_service: AppSettingsService | None
    project_storage: ProjectStorage

    @property
    def session(self) -> Session: ...

    @property
    def runtime_audio(self) -> RuntimeAudioController | None: ...

    @runtime_audio.setter
    def runtime_audio(self, value: RuntimeAudioController | None) -> None: ...

    def presentation(self) -> TimelinePresentation: ...


def build_object_action_services(shell: RuntimeSupportShell) -> None:
    shell._object_action_settings = ObjectActionService(
        project_storage_getter=lambda: shell.project_storage,
        session_getter=lambda: shell.session,
        presentation_getter=shell.presentation,
        require_layer=lambda layer_id: require_layer(shell, LayerId(str(layer_id))),
        analysis_service=shell._analysis_service,
        active_run_lookup=lambda action_id, object_id, object_type: shell._pipeline_runs.visible_run_for(
            action_id=action_id,
            object_id=object_id,
            object_type=object_type,
            song_id=shell.session.active_song_id,
            song_version_id=shell.session.active_song_version_id,
        ),
    )
    shell._pipeline_runs = PipelineRunService(
        project_storage_getter=lambda: shell.project_storage,
        session_getter=lambda: shell.session,
        analysis_service=shell._analysis_service,
        prepare_run=lambda action_id, params, object_id, object_type, persist_scope: shell._object_action_settings.prepare_run(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            persist_scope=persist_scope,
        ),
        persist_generated_source_layer_id=shell._object_action_settings.persist_generated_source_layer_id,
    )


def select_active_source_layer(shell: RuntimeSupportShell) -> None:
    source_layer_id = LayerId("source_audio")
    if not any(layer.id == source_layer_id for layer in shell._app.timeline.layers):
        return
    timeline = shell._app.timeline
    timeline.selection.selected_layer_id = source_layer_id
    timeline.selection.selected_layer_ids = [source_layer_id]
    timeline.selection.selected_take_id = None
    clear_selected_events(timeline)
    timeline.playback_target.layer_id = source_layer_id
    timeline.playback_target.take_id = None
    sync_runtime_audio_from_presentation(shell, shell.presentation())


def sync_runtime_audio_from_presentation(
    shell: RuntimeSupportShell,
    presentation: TimelinePresentation,
) -> None:
    runtime_audio = shell.runtime_audio
    if runtime_audio is None:
        return
    runtime_audio.build_for_presentation(presentation)
    snapshot_state = getattr(runtime_audio, "snapshot_state", None)
    if callable(snapshot_state):
        shell.session.playback_state = snapshot_state(presentation)


def preview_event_clip(
    shell: RuntimeSupportShell,
    *,
    layer_id: LayerId,
    take_id: TakeId | None = None,
    event_id: EventId,
) -> None:
    runtime_audio = shell.runtime_audio
    if runtime_audio is None:
        raise RuntimeError("This runtime does not support event clip preview.")

    clip = resolve_event_clip_preview(
        shell.presentation(),
        layer_id=layer_id,
        take_id=take_id,
        event_id=event_id,
    )
    runtime_audio.preview_clip(
        clip.source_ref,
        start_seconds=clip.start_seconds,
        end_seconds=clip.end_seconds,
        gain_db=clip.gain_db,
    )


def apply_audio_output_config(
    shell: RuntimeSupportShell,
    config: AudioOutputRuntimeConfig | None,
) -> None:
    """Rebuild runtime audio using one updated machine-local output configuration."""

    previous_runtime_audio = shell.runtime_audio
    presentation = shell.presentation()
    current_time_seconds = 0.0
    was_playing = False

    if previous_runtime_audio is not None:
        current_time_seconds = float(previous_runtime_audio.current_time_seconds())
        was_playing = bool(previous_runtime_audio.is_playing())
        if was_playing:
            previous_runtime_audio.pause()

    next_runtime_audio = build_runtime_audio_controller(config)
    try:
        next_runtime_audio.build_for_presentation(presentation)
        if current_time_seconds > 0.0:
            next_runtime_audio.seek(current_time_seconds)
        if was_playing:
            next_runtime_audio.play()
    except Exception:
        next_runtime_audio.shutdown()
        if previous_runtime_audio is not None and was_playing:
            previous_runtime_audio.play()
        raise

    shell.runtime_audio = next_runtime_audio
    if previous_runtime_audio is not None:
        previous_runtime_audio.shutdown()
    snapshot_state = getattr(next_runtime_audio, "snapshot_state", None)
    if callable(snapshot_state):
        shell.session.playback_state = snapshot_state(presentation)
    shell.session.transport_state.is_playing = bool(next_runtime_audio.is_playing())
    shell.session.transport_state.playhead = float(next_runtime_audio.current_time_seconds())


def apply_ma3_osc_runtime_config(shell: RuntimeSupportShell) -> bool:
    """Reconfigure the active MA3 bridge from current AppSettings values."""

    bridge = getattr(shell, "_sync_bridge", None)
    if bridge is None or not hasattr(bridge, "reconfigure"):
        return False

    settings_service = getattr(shell, "_app_settings_service", None)
    if not isinstance(settings_service, AppSettingsService):
        return False

    runtime_config = settings_service.resolve_ma3_osc_runtime_config()
    command_transport = None
    if runtime_config.send.enabled and runtime_config.send.port is not None:
        command_transport = OscUdpSendTransport(
            runtime_config.send.host,
            runtime_config.send.port,
            path=runtime_config.send.path,
        )
    try:
        bridge.reconfigure(
            listen_host=runtime_config.receive.host,
            listen_port=runtime_config.receive.port,
            listen_path=runtime_config.receive.path,
            command_transport=command_transport,
        )
    except Exception:
        if command_transport is not None:
            command_transport.close()
        raise
    return True


def require_layer(shell: RuntimeSupportShell, layer_id: LayerId) -> LayerPresentation:
    for layer in shell.presentation().layers:
        if layer.layer_id == layer_id:
            return layer
    raise ValueError(f"Unknown layer_id: {layer_id}")


def shutdown(shell: RuntimeSupportShell) -> None:
    shell._pipeline_runs.shutdown()
    runtime_audio = shell.runtime_audio
    if runtime_audio is not None:
        runtime_audio.shutdown()
    if shell._sync_bridge is not None and hasattr(shell._sync_bridge, "shutdown"):
        shell._sync_bridge.shutdown()
    shell.project_storage.close()
