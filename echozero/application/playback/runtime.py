"""Application playback runtime boundary over the concrete audio backend.

This owns presentation-to-backend playback mapping and publishes the narrow
runtime/timing surface the UI needs without exposing backend internals as
timeline semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from echozero.audio.file_cache import load_audio_file
from echozero.application.playback.models import (
    PlaybackSource,
    PlaybackState,
    PlaybackTimingSnapshot,
)
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import PlaybackMode, PlaybackStatus
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.audio.engine import AudioEngine

_SOUNDDEVICE_BACKEND = "sounddevice"
_QT_MULTIMEDIA_BACKEND = "qt_multimedia"


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (float(gain_db) / 20.0))


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_output_bus(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    output_bus = value.strip()
    return output_bus or None


def _load_runtime_audio(path: str | Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = load_audio_file(path)
    return samples.astype(np.float32, copy=False), int(sample_rate)


@dataclass(slots=True)
class RuntimeAudioLayer:
    layer_id: str
    name: str
    gain_db: float
    output_bus: str | None
    source_key: str
    cache_keys: tuple[str, ...]
    buffer: np.ndarray | None = None
    sample_rate: int = 0
    source_ref: str | None = None


class PresentationPlaybackRuntime:
    """Playback runtime facade for the current presentation-backed app shell."""

    _MONITOR_LAYER_ID = "__ez_monitor__"
    _PREVIEW_LAYER_ID = "__ez_preview__"

    def __init__(
        self,
        engine: AudioEngine | None = None,
        *,
        engine_factory: Callable[[], AudioEngine] | None = None,
        preview_engine: AudioEngine | None = None,
        preview_engine_factory: Callable[[], AudioEngine] | None = None,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] = _load_runtime_audio,
        use_qt_player: bool = True,
        prefer_qt_for_continuous_audio: bool = True,
        force_qt_for_continuous_audio: bool = False,
        qt_output_device: int | str | None = None,
    ) -> None:
        self._engine = engine or (
            engine_factory() if engine_factory is not None else AudioEngine()
        )
        resolved_preview_factory = preview_engine_factory or engine_factory
        self._preview_engine = preview_engine or (
            resolved_preview_factory() if resolved_preview_factory is not None else AudioEngine()
        )
        self._audio_loader = audio_loader
        self._loaded_source_key: str | None = None
        self._loaded_routed_source_keys: dict[str, str] = {}
        self._buffer_cache: dict[str, tuple[np.ndarray, int]] = {}
        # Use Qt Multimedia as the default file-playback path in app runtime.
        # Explicit injected engines keep the legacy backend for deterministic tests.
        # prefer_qt_for_continuous_audio chooses whether plain "audio:*" sources
        # should route through Qt at all. force_qt_for_continuous_audio only
        # controls whether a Qt init failure is allowed to fall back to the
        # callback engine.
        self._qt_enabled = bool(use_qt_player and engine is None)
        self._prefer_qt_for_continuous_audio = bool(prefer_qt_for_continuous_audio)
        self._force_qt_for_continuous_audio = bool(force_qt_for_continuous_audio)
        self._qt_output_device = qt_output_device
        self._qt_player: Any | None = None
        self._qt_audio_output: Any | None = None
        self._qt_source_key: str | None = None
        self._qt_source_ref: str | None = None
        self._qt_pending_source_key: str | None = None
        self._qt_pending_source_ref: str | None = None
        self._qt_pending_resume_seconds: float = 0.0
        self._qt_pending_resume_playing: bool = False
        self._qt_pending_target_volume: float = 1.0
        self._qt_transition_timer: Any | None = None
        self._qt_transition_values: list[float] = []
        self._qt_transition_complete: Callable[[], None] | None = None
        self._qt_transition_interval_ms = 2
        self._qt_transition_steps = 5
        self._active_backend: str = _SOUNDDEVICE_BACKEND
        self._preview_active = False

    @property
    def engine(self) -> AudioEngine:
        return self._engine

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        self._sync_active_layer(presentation)

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self._sync_active_layer(presentation)

    def play(self) -> None:
        self._sync_preview_state()
        if self._preview_active:
            self.stop_preview()
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            if self._qt_player is not None:
                self._qt_player.play()
            return
        self._engine.play()

    def pause(self) -> None:
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            if self._qt_player is not None:
                self._qt_player.pause()
            return
        self._engine.pause()

    def stop(self) -> None:
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            if self._qt_player is not None:
                self._qt_player.stop()
                self._qt_seek_seconds(0.0)
            return
        self._engine.stop()

    def seek(self, position_seconds: float) -> None:
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            if self._qt_player is not None:
                self._qt_seek_seconds(position_seconds)
            return
        self._engine.seek_seconds(position_seconds)

    def current_time_seconds(self) -> float:
        self._sync_preview_state()
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            if self._qt_player is None:
                return 0.0
            try:
                return max(0.0, float(self._qt_player.position()) / 1000.0)
            except Exception:
                return 0.0
        return float(self._engine.audible_time_seconds)

    def timing_snapshot(self) -> PlaybackTimingSnapshot:
        self._sync_preview_state()
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            current_time = self.current_time_seconds()
            return PlaybackTimingSnapshot(
                audible_time_seconds=current_time,
                clock_time_seconds=current_time,
                snapshot_monotonic_seconds=None,
                is_playing=self.is_playing(),
            )

        snapshot_time = getattr(self._engine, "_last_audible_time_seconds", None)
        snapshot_monotonic = getattr(self._engine, "_last_audible_monotonic_seconds", None)
        clock_time = float(self._engine.clock.position_seconds)
        is_playing = bool(self._engine.transport.is_playing)
        if snapshot_time is None:
            snapshot_time = float(self._engine.audible_time_seconds)
            snapshot_monotonic = None
        return PlaybackTimingSnapshot(
            audible_time_seconds=max(0.0, float(snapshot_time)),
            clock_time_seconds=max(0.0, clock_time),
            snapshot_monotonic_seconds=(
                max(0.0, float(snapshot_monotonic)) if snapshot_monotonic is not None else None
            ),
            is_playing=is_playing,
        )

    def is_playing(self) -> bool:
        self._sync_preview_state()
        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            if self._qt_player is None:
                return False
            from PyQt6.QtMultimedia import QMediaPlayer

            return bool(
                self._qt_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            )
        return bool(self._engine.transport.is_playing)

    def shutdown(self) -> None:
        self._deactivate_qt(reset_source=True)
        self._clear_engine_routed_layers()
        self._engine.shutdown()
        self.stop_preview()
        self._preview_engine.shutdown()

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        source_path = str(source_ref).strip()
        if not source_path:
            return False

        source_key = f"preview:{source_path}"
        source_buffer, sample_rate = self._buffer_cache.get(source_key) or self._audio_loader(
            source_path
        )
        self._buffer_cache[source_key] = (source_buffer, sample_rate)
        if source_buffer.size == 0:
            return False

        start_sample = max(0, int(round(float(start_seconds) * sample_rate)))
        end_sample = max(start_sample, int(round(float(end_seconds) * sample_rate)))
        end_sample = min(end_sample, int(source_buffer.shape[0]))
        if end_sample <= start_sample:
            return False

        preview_buffer = np.asarray(source_buffer[start_sample:end_sample], dtype=np.float32)
        if preview_buffer.size == 0:
            return False

        self.stop_preview()
        self._preview_engine.add_layer(
            self._PREVIEW_LAYER_ID,
            preview_buffer,
            sample_rate,
            name="Event Preview",
            volume=_db_to_linear(gain_db),
        )
        self._preview_engine.seek_seconds(0.0)
        self._preview_engine.play()
        self._preview_active = True
        return True

    def stop_preview(self) -> None:
        self._preview_active = False
        self._preview_engine.stop()
        self._preview_engine.remove_layer(self._PREVIEW_LAYER_ID)
        if self._preview_engine.is_active:
            self._preview_engine.shutdown()

    def _sync_preview_state(self) -> None:
        if not self._preview_active:
            return
        if self._preview_engine.reached_end:
            self.stop_preview()

    def presentation_signature(
        self, presentation: TimelinePresentation
    ) -> tuple[tuple[str, str], ...]:
        routed_signature = self._routed_source_signature(presentation)
        if routed_signature:
            return routed_signature
        source_signature = self._active_source_signature(presentation)
        if source_signature is None:
            return ()
        return (source_signature,)

    def snapshot_state(self, presentation: TimelinePresentation) -> PlaybackState:
        active_sources: list[PlaybackSource] = []
        routed_layers = self._select_routed_runtime_layers(presentation)
        if routed_layers:
            for routed_layer in routed_layers:
                active_sources.append(
                    PlaybackSource(
                        layer_id=routed_layer.layer_id,
                        source_ref=routed_layer.source_ref,
                        mode=PlaybackMode.CONTINUOUS_AUDIO,
                    )
                )
        else:
            active_layer = self._select_active_runtime_layer(presentation)
            if active_layer is not None:
                active_sources.append(
                    PlaybackSource(
                        layer_id=presentation.active_playback_layer_id,
                        take_id=presentation.active_playback_take_id,
                        source_ref=active_layer.source_ref,
                        mode=(
                            PlaybackMode.EVENT_SLICE
                            if active_layer.source_ref and active_layer.source_ref.startswith("event:")
                            else PlaybackMode.CONTINUOUS_AUDIO
                        ),
                    )
                )

        if self._active_backend == _QT_MULTIMEDIA_BACKEND:
            output_sample_rate, output_channels = self._qt_output_format()
            return PlaybackState(
                status=PlaybackStatus.PLAYING if self.is_playing() else PlaybackStatus.STOPPED,
                active_sources=active_sources,
                latency_ms=0.0,
                backend_name=_QT_MULTIMEDIA_BACKEND,
                active_layer_id=presentation.active_playback_layer_id,
                active_take_id=presentation.active_playback_take_id,
                output_sample_rate=output_sample_rate,
                output_channels=output_channels,
            )

        return PlaybackState(
            status=PlaybackStatus.PLAYING if self.is_playing() else PlaybackStatus.STOPPED,
            active_sources=active_sources,
            latency_ms=float(self._engine.reported_output_latency_seconds) * 1000.0,
            backend_name=_SOUNDDEVICE_BACKEND,
            active_layer_id=presentation.active_playback_layer_id,
            active_take_id=presentation.active_playback_take_id,
            output_sample_rate=int(self._engine.sample_rate),
            output_channels=int(getattr(self._engine, "_channels", 1)),
        )

    def _sync_active_layer(self, presentation: TimelinePresentation) -> None:
        routed_layers = self._select_routed_runtime_layers(presentation)
        active_layer = self._select_active_runtime_layer(presentation)
        desired_cache_keys: set[str] = set()
        if routed_layers:
            for routed_layer in routed_layers:
                desired_cache_keys.update(routed_layer.cache_keys)
        elif active_layer is not None:
            desired_cache_keys.update(active_layer.cache_keys)
        stale_cache_keys = [key for key in self._buffer_cache if key not in desired_cache_keys]
        for stale_key in stale_cache_keys:
            self._buffer_cache.pop(stale_key, None)

        resume_seconds = self.current_time_seconds()
        resume_playing = self.is_playing()

        if routed_layers:
            self._sync_engine_routed_layers(
                routed_layers,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
            )
            if self._loaded_source_key is not None:
                self._engine.remove_layer(self._MONITOR_LAYER_ID)
                self._loaded_source_key = None
            self._deactivate_qt(reset_source=False)
            self._active_backend = _SOUNDDEVICE_BACKEND
            return

        if active_layer is None:
            if self._loaded_source_key is not None:
                self._engine.remove_layer(self._MONITOR_LAYER_ID)
                self._loaded_source_key = None
            self._clear_engine_routed_layers()
            self._deactivate_qt(reset_source=True)
            self._active_backend = _SOUNDDEVICE_BACKEND
            return

        if self._can_use_qt_backend_for_layer(active_layer):
            self._sync_qt_layer(
                active_layer,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
            )
            if self._loaded_source_key is not None:
                self._engine.remove_layer(self._MONITOR_LAYER_ID)
                self._loaded_source_key = None
            self._clear_engine_routed_layers()
            # For continuous-audio layers, app runtime should remain on the Qt path.
            self._active_backend = _QT_MULTIMEDIA_BACKEND
            return

        self._sync_engine_layer(
            active_layer,
            resume_seconds=resume_seconds,
            resume_playing=resume_playing,
        )
        self._clear_engine_routed_layers()
        self._deactivate_qt(reset_source=False)
        self._active_backend = _SOUNDDEVICE_BACKEND

    def _sync_engine_layer(
        self,
        active_layer: RuntimeAudioLayer,
        *,
        resume_seconds: float,
        resume_playing: bool,
    ) -> None:
        if self._loaded_source_key != active_layer.source_key:
            buffer, sample_rate = self._resolve_runtime_layer_audio(active_layer)
            if self._loaded_source_key is not None:
                self._engine.remove_layer(self._MONITOR_LAYER_ID)
            self._engine.add_layer(
                self._MONITOR_LAYER_ID,
                buffer,
                sample_rate,
                name=active_layer.name,
                volume=_db_to_linear(active_layer.gain_db),
                output_bus=active_layer.output_bus,
            )
            self._loaded_source_key = active_layer.source_key
            if resume_seconds > 0.0:
                self._engine.seek_seconds(resume_seconds)
            if resume_playing:
                self._engine.play()

        engine_layer = self._engine.mixer.get_layer(self._MONITOR_LAYER_ID)
        if engine_layer is not None:
            engine_layer.muted = False
            engine_layer.volume = _db_to_linear(active_layer.gain_db)
            engine_layer.output_bus = active_layer.output_bus

    def _sync_engine_routed_layers(
        self,
        routed_layers: list[RuntimeAudioLayer],
        *,
        resume_seconds: float,
        resume_playing: bool,
    ) -> None:
        desired_ids = {
            self._routed_engine_layer_id(runtime_layer.layer_id)
            for runtime_layer in routed_layers
        }
        stale_ids = [
            layer_id
            for layer_id in list(self._loaded_routed_source_keys)
            if layer_id not in desired_ids
        ]
        for stale_id in stale_ids:
            self._engine.remove_layer(stale_id)
            self._loaded_routed_source_keys.pop(stale_id, None)

        any_reloaded = False
        for runtime_layer in routed_layers:
            engine_layer_id = self._routed_engine_layer_id(runtime_layer.layer_id)
            loaded_source_key = self._loaded_routed_source_keys.get(engine_layer_id)
            if loaded_source_key != runtime_layer.source_key:
                buffer, sample_rate = self._resolve_runtime_layer_audio(runtime_layer)
                if loaded_source_key is not None:
                    self._engine.remove_layer(engine_layer_id)
                self._engine.add_layer(
                    engine_layer_id,
                    buffer,
                    sample_rate,
                    name=runtime_layer.name,
                    volume=_db_to_linear(runtime_layer.gain_db),
                    output_bus=runtime_layer.output_bus,
                )
                self._loaded_routed_source_keys[engine_layer_id] = runtime_layer.source_key
                any_reloaded = True

            engine_layer = self._engine.mixer.get_layer(engine_layer_id)
            if engine_layer is not None:
                engine_layer.muted = False
                engine_layer.volume = _db_to_linear(runtime_layer.gain_db)
                engine_layer.output_bus = runtime_layer.output_bus

        if any_reloaded and resume_seconds > 0.0:
            self._engine.seek_seconds(resume_seconds)
        if any_reloaded and resume_playing:
            self._engine.play()

    def _clear_engine_routed_layers(self) -> None:
        if not self._loaded_routed_source_keys:
            return
        for layer_id in list(self._loaded_routed_source_keys):
            self._engine.remove_layer(layer_id)
        self._loaded_routed_source_keys.clear()

    @staticmethod
    def _routed_engine_layer_id(layer_id: str) -> str:
        return f"__ez_route__{layer_id}"

    def _resolve_runtime_layer_audio(
        self, runtime_layer: RuntimeAudioLayer
    ) -> tuple[np.ndarray, int]:
        if runtime_layer.buffer is not None and runtime_layer.sample_rate > 0:
            return runtime_layer.buffer, runtime_layer.sample_rate

        source_ref = str(runtime_layer.source_ref or "").strip()
        if not source_ref:
            raise ValueError(f"Runtime layer '{runtime_layer.layer_id}' has no source audio ref.")

        cached = self._buffer_cache.get(runtime_layer.source_key)
        if cached is None:
            cached = self._audio_loader(source_ref)
            self._buffer_cache[runtime_layer.source_key] = cached
        buffer, sample_rate = cached
        runtime_layer.buffer = buffer
        runtime_layer.sample_rate = sample_rate
        return buffer, sample_rate

    def _can_use_qt_backend_for_layer(self, active_layer: RuntimeAudioLayer) -> bool:
        if not self._qt_enabled or not self._prefer_qt_for_continuous_audio:
            return False
        source_ref = str(active_layer.source_ref or "").strip()
        if not source_ref:
            return False
        return bool(active_layer.source_key.startswith("audio:"))

    def _sync_qt_layer(
        self,
        active_layer: RuntimeAudioLayer,
        *,
        resume_seconds: float,
        resume_playing: bool,
    ) -> None:
        if not self._ensure_qt_player():
            if self._force_qt_for_continuous_audio:
                return
            self._sync_engine_layer(
                active_layer,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
            )
            return

        source_ref = str(active_layer.source_ref or "").strip()
        if not source_ref:
            return

        target_volume = _clamp_unit_interval(_db_to_linear(active_layer.gain_db))
        if self._qt_pending_source_key is not None:
            if active_layer.source_key == self._qt_source_key:
                self._clear_qt_pending_source_switch()
                self._stop_qt_volume_ramp()
                self._qt_audio_output.setVolume(target_volume)
                if resume_playing:
                    self._qt_player.play()
                return
            if active_layer.source_key == self._qt_pending_source_key:
                self._qt_pending_resume_seconds = max(0.0, float(resume_seconds))
                self._qt_pending_resume_playing = bool(resume_playing)
                self._qt_pending_target_volume = target_volume
                return

        source_changed = self._qt_source_key != active_layer.source_key
        if source_changed and resume_playing and self._active_backend == _QT_MULTIMEDIA_BACKEND:
            self._queue_qt_source_switch(
                source_key=active_layer.source_key,
                source_ref=source_ref,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
                target_volume=target_volume,
            )
            return

        if source_changed:
            from PyQt6.QtCore import QUrl

            self._qt_player.setSource(QUrl.fromLocalFile(source_ref))
            self._qt_source_key = active_layer.source_key
            self._qt_source_ref = source_ref

        if resume_seconds > 0.0 and (
            source_changed or self._active_backend != _QT_MULTIMEDIA_BACKEND
        ):
            self._qt_seek_seconds(resume_seconds)

        self._qt_audio_output.setVolume(target_volume)
        if resume_playing:
            self._qt_player.play()

    def _queue_qt_source_switch(
        self,
        *,
        source_key: str,
        source_ref: str,
        resume_seconds: float,
        resume_playing: bool,
        target_volume: float,
    ) -> None:
        self._qt_pending_source_key = source_key
        self._qt_pending_source_ref = source_ref
        self._qt_pending_resume_seconds = max(0.0, float(resume_seconds))
        self._qt_pending_resume_playing = bool(resume_playing)
        self._qt_pending_target_volume = float(target_volume)

        if self._qt_audio_output is None:
            return

        current_volume = self._qt_output_volume(default=target_volume)
        if current_volume <= 0.0:
            self._commit_qt_source_switch()
            return

        if self._qt_transition_timer is not None and self._qt_transition_timer.isActive():
            return

        self._start_qt_volume_ramp(
            current_volume,
            0.0,
            self._commit_qt_source_switch,
        )

    def _commit_qt_source_switch(self) -> None:
        if self._qt_player is None or self._qt_audio_output is None:
            self._clear_qt_pending_source_switch()
            return

        source_key = self._qt_pending_source_key
        source_ref = str(self._qt_pending_source_ref or "").strip()
        resume_seconds = self._qt_pending_resume_seconds
        resume_playing = self._qt_pending_resume_playing
        target_volume = self._qt_pending_target_volume
        self._clear_qt_pending_source_switch()
        if source_key is None or not source_ref:
            return

        from PyQt6.QtCore import QUrl

        self._qt_player.stop()
        self._qt_audio_output.setVolume(0.0)
        self._qt_player.setSource(QUrl.fromLocalFile(source_ref))
        self._qt_source_key = source_key
        self._qt_source_ref = source_ref
        if resume_seconds > 0.0:
            self._qt_seek_seconds(resume_seconds)
        if resume_playing:
            self._qt_player.play()
            self._start_qt_volume_ramp(0.0, target_volume, None)
            return
        self._qt_audio_output.setVolume(target_volume)

    def _clear_qt_pending_source_switch(self) -> None:
        self._qt_pending_source_key = None
        self._qt_pending_source_ref = None
        self._qt_pending_resume_seconds = 0.0
        self._qt_pending_resume_playing = False
        self._qt_pending_target_volume = 1.0

    def _start_qt_volume_ramp(
        self,
        start_volume: float,
        end_volume: float,
        on_complete: Callable[[], None] | None,
    ) -> None:
        if self._qt_audio_output is None:
            if on_complete is not None:
                on_complete()
            return

        start = _clamp_unit_interval(start_volume)
        end = _clamp_unit_interval(end_volume)
        if self._qt_transition_steps <= 1 or abs(start - end) < 1e-6:
            self._qt_audio_output.setVolume(end)
            if on_complete is not None:
                on_complete()
            return

        from PyQt6.QtCore import QCoreApplication, QTimer

        if QCoreApplication.instance() is None:
            self._qt_audio_output.setVolume(end)
            if on_complete is not None:
                on_complete()
            return

        if self._qt_transition_timer is None:
            self._qt_transition_timer = QTimer()
            self._qt_transition_timer.setSingleShot(False)
            self._qt_transition_timer.timeout.connect(self._advance_qt_volume_ramp)

        self._qt_transition_values = [
            start + ((end - start) * step / self._qt_transition_steps)
            for step in range(1, self._qt_transition_steps + 1)
        ]
        self._qt_transition_complete = on_complete
        self._qt_transition_timer.start(self._qt_transition_interval_ms)

    def _advance_qt_volume_ramp(self) -> None:
        if self._qt_audio_output is None:
            self._stop_qt_volume_ramp()
            return
        if not self._qt_transition_values:
            complete = self._qt_transition_complete
            self._stop_qt_volume_ramp()
            if complete is not None:
                complete()
            return

        next_volume = _clamp_unit_interval(self._qt_transition_values.pop(0))
        self._qt_audio_output.setVolume(next_volume)
        if self._qt_transition_values:
            return

        complete = self._qt_transition_complete
        self._stop_qt_volume_ramp()
        if complete is not None:
            complete()

    def _stop_qt_volume_ramp(self) -> None:
        if self._qt_transition_timer is not None:
            self._qt_transition_timer.stop()
        self._qt_transition_values = []
        self._qt_transition_complete = None

    def _qt_output_volume(self, *, default: float) -> float:
        if self._qt_audio_output is None:
            return _clamp_unit_interval(default)
        volume = getattr(self._qt_audio_output, "volume", None)
        if not callable(volume):
            return _clamp_unit_interval(default)
        try:
            return _clamp_unit_interval(volume())
        except Exception:
            return _clamp_unit_interval(default)

    def _ensure_qt_player(self) -> bool:
        if not self._qt_enabled:
            return False
        if self._qt_player is not None and self._qt_audio_output is not None:
            return True

        from PyQt6.QtCore import QCoreApplication

        if QCoreApplication.instance() is None:
            return False

        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

        self._qt_player = QMediaPlayer()
        self._qt_audio_output = QAudioOutput()
        preferred_device = self._resolve_qt_output_device()
        if preferred_device is not None:
            self._qt_audio_output.setDevice(preferred_device)
        self._qt_player.setAudioOutput(self._qt_audio_output)
        return True

    def _resolve_qt_output_device(self) -> Any | None:
        from PyQt6.QtMultimedia import QMediaDevices

        outputs = tuple(QMediaDevices.audioOutputs())
        if not outputs:
            return None

        preference = self._qt_output_device
        if preference is None:
            return QMediaDevices.defaultAudioOutput()

        if isinstance(preference, int) and 0 <= preference < len(outputs):
            return outputs[preference]

        text = str(preference).strip()
        if text.isdigit():
            index = int(text)
            if 0 <= index < len(outputs):
                return outputs[index]

        lowered = text.lower()
        for candidate in outputs:
            candidate_id = bytes(candidate.id()).decode(errors="ignore")
            if candidate_id == text:
                return candidate
            if lowered and lowered in candidate.description().lower():
                return candidate

        return QMediaDevices.defaultAudioOutput()

    def _qt_seek_seconds(self, position_seconds: float) -> None:
        if self._qt_player is None:
            return
        target_ms = max(0, int(round(float(position_seconds) * 1000.0)))
        self._qt_player.setPosition(target_ms)

    def _deactivate_qt(self, *, reset_source: bool) -> None:
        self._stop_qt_volume_ramp()
        self._clear_qt_pending_source_switch()
        if self._qt_player is None:
            return
        self._qt_player.stop()
        if not reset_source:
            return

        from PyQt6.QtCore import QUrl

        self._qt_player.setSource(QUrl())
        self._qt_source_key = None
        self._qt_source_ref = None

    def _qt_output_format(self) -> tuple[int, int]:
        if self._qt_audio_output is None:
            return 0, 0
        device = self._qt_audio_output.device()
        if device is None:
            return 0, 0
        preferred = device.preferredFormat()
        return max(0, int(preferred.sampleRate())), max(0, int(preferred.channelCount()))

    def _select_active_runtime_layer(
        self, presentation: TimelinePresentation
    ) -> RuntimeAudioLayer | None:
        active_layer_id = presentation.active_playback_layer_id
        active_take_id = presentation.active_playback_take_id
        if active_layer_id is not None:
            runtime_layer = self._runtime_layer_for_target(
                presentation,
                layer_id=str(active_layer_id),
                take_id=str(active_take_id) if active_take_id is not None else None,
            )
            if runtime_layer is not None:
                return runtime_layer
        return None

    def _runtime_layer_for_target(
        self,
        presentation: TimelinePresentation,
        *,
        layer_id: str,
        take_id: str | None,
    ) -> RuntimeAudioLayer | None:
        for layer in presentation.layers:
            if str(layer.layer_id) != layer_id:
                continue
            if take_id is not None:
                for take in layer.takes:
                    if str(take.take_id) == take_id:
                        runtime_take = self._runtime_layer_from_take(layer, take)
                        if runtime_take is not None:
                            return runtime_take
            return self._runtime_layer_from_layer(layer)
        return None

    def _runtime_layer_from_layer(self, layer: object) -> RuntimeAudioLayer | None:
        layer_id = str(getattr(layer, "layer_id"))
        title = str(getattr(layer, "title"))
        gain_db = float(getattr(layer, "gain_db", 0.0))
        output_bus = _normalize_output_bus(getattr(layer, "output_bus", None))
        source_audio_path = getattr(layer, "source_audio_path", None)
        if source_audio_path:
            return self._build_audio_runtime_layer(
                layer_id=layer_id,
                title=title,
                gain_db=gain_db,
                output_bus=output_bus,
                source_audio_path=source_audio_path,
            )
        if not self._is_event_slice_layer(layer):
            return None
        return self._build_event_runtime_layer(
            layer_id=layer_id,
            title=title,
            gain_db=gain_db,
            output_bus=output_bus,
            playback_source_ref=getattr(layer, "playback_source_ref"),
            presentation_events=list(getattr(layer, "events")),
        )

    def _runtime_layer_from_take(self, layer: object, take: object) -> RuntimeAudioLayer | None:
        layer_id = str(getattr(layer, "layer_id"))
        take_id = str(getattr(take, "take_id"))
        title = f"{getattr(layer, 'title')} · {getattr(take, 'name')}"
        gain_db = float(getattr(layer, "gain_db", 0.0))
        output_bus = _normalize_output_bus(getattr(layer, "output_bus", None))
        source_audio_path = getattr(take, "source_audio_path", None)
        if source_audio_path:
            return self._build_audio_runtime_layer(
                layer_id=f"{layer_id}:{take_id}",
                title=title,
                gain_db=gain_db,
                output_bus=output_bus,
                source_audio_path=source_audio_path,
            )
        if not self._is_event_slice_layer(take):
            return None
        return self._build_event_runtime_layer(
            layer_id=f"{layer_id}:{take_id}",
            title=title,
            gain_db=gain_db,
            output_bus=output_bus,
            playback_source_ref=getattr(take, "playback_source_ref"),
            presentation_events=list(getattr(take, "events")),
        )

    def _build_audio_runtime_layer(
        self,
        *,
        layer_id: str,
        title: str,
        gain_db: float,
        output_bus: str | None,
        source_audio_path: str,
    ) -> RuntimeAudioLayer:
        source_key = f"audio:{source_audio_path}"
        cached = self._buffer_cache.get(source_key)
        buffer: np.ndarray | None
        sample_rate: int
        if cached is not None:
            buffer, sample_rate = cached
        elif self._qt_enabled:
            # Defer decode while Qt Multimedia can stream directly from file.
            buffer = None
            sample_rate = 0
        else:
            buffer, sample_rate = self._audio_loader(source_audio_path)
            self._buffer_cache[source_key] = (buffer, sample_rate)
        return RuntimeAudioLayer(
            layer_id=layer_id,
            name=title,
            gain_db=gain_db,
            output_bus=output_bus,
            source_key=source_key,
            cache_keys=(source_key,),
            buffer=buffer,
            sample_rate=sample_rate,
            source_ref=source_audio_path,
        )

    def _active_source_signature(
        self, presentation: TimelinePresentation
    ) -> tuple[str, str] | None:
        active_layer_id = presentation.active_playback_layer_id
        if active_layer_id is None:
            return None
        active_take_id = presentation.active_playback_take_id
        return self._source_signature_for_target(
            presentation,
            layer_id=str(active_layer_id),
            take_id=str(active_take_id) if active_take_id is not None else None,
        )

    def _routed_source_signature(
        self,
        presentation: TimelinePresentation,
    ) -> tuple[tuple[str, str], ...]:
        routed_layers = self._select_routed_runtime_layers(presentation)
        return tuple(
            (
                runtime_layer.layer_id,
                f"{runtime_layer.source_key}|{runtime_layer.output_bus or 'outputs_1_2'}",
            )
            for runtime_layer in routed_layers
        )

    def _select_routed_runtime_layers(
        self,
        presentation: TimelinePresentation,
    ) -> list[RuntimeAudioLayer]:
        layer_candidates = [
            layer
            for layer in presentation.layers
            if bool(getattr(layer, "source_audio_path", None))
        ]
        if not layer_candidates:
            return []

        has_explicit_route = any(
            _normalize_output_bus(getattr(layer, "output_bus", None)) is not None
            for layer in layer_candidates
        )
        if not has_explicit_route:
            return []

        active_layer_id = (
            str(presentation.active_playback_layer_id)
            if presentation.active_playback_layer_id is not None
            else None
        )
        active_take_id = (
            str(presentation.active_playback_take_id)
            if presentation.active_playback_take_id is not None
            else None
        )
        routed_layers: list[RuntimeAudioLayer] = []
        seen_layer_ids: set[str] = set()
        for layer in layer_candidates:
            layer_id = str(getattr(layer, "layer_id"))
            output_bus = _normalize_output_bus(getattr(layer, "output_bus", None))
            if output_bus is None and layer_id != active_layer_id:
                continue
            if layer_id == active_layer_id:
                runtime_layer = self._runtime_layer_for_target(
                    presentation,
                    layer_id=layer_id,
                    take_id=active_take_id,
                )
            else:
                runtime_layer = self._runtime_layer_from_layer(layer)
            if runtime_layer is None or runtime_layer.layer_id in seen_layer_ids:
                continue
            routed_layers.append(runtime_layer)
            seen_layer_ids.add(runtime_layer.layer_id)
        return routed_layers

    def _source_signature_for_target(
        self,
        presentation: TimelinePresentation,
        *,
        layer_id: str,
        take_id: str | None,
    ) -> tuple[str, str] | None:
        for layer in presentation.layers:
            if str(layer.layer_id) != layer_id:
                continue
            if take_id is not None:
                for take in layer.takes:
                    if str(take.take_id) == take_id:
                        source_key = self._source_key_for_layer_or_take(take)
                        if source_key is None:
                            return None
                        return (f"{layer_id}:{take_id}", source_key)
            source_key = self._source_key_for_layer_or_take(layer)
            if source_key is None:
                return None
            return (layer_id, source_key)
        return None

    @staticmethod
    def _source_key_for_layer_or_take(layer: object) -> str | None:
        source_audio_path = getattr(layer, "source_audio_path", None)
        if source_audio_path:
            return f"audio:{source_audio_path}"
        if not PresentationPlaybackRuntime._is_event_slice_layer(layer):
            return None
        playback_source_ref = str(getattr(layer, "playback_source_ref"))
        presentation_events = list(getattr(layer, "events"))
        event_signature = ",".join(
            f"{event.start:.6f}:{int(event.muted)}:{int('demoted' in getattr(event, 'badges', []))}"
            for event in presentation_events
        )
        return f"event:{playback_source_ref}:{event_signature}"

    def _build_event_runtime_layer(
        self,
        *,
        layer_id: str,
        title: str,
        gain_db: float,
        output_bus: str | None,
        playback_source_ref: str,
        presentation_events: list,
    ) -> RuntimeAudioLayer | None:
        sample_source_key = f"event-sample:{playback_source_ref}"
        event_buffer, sample_rate = self._buffer_cache.get(
            sample_source_key
        ) or self._audio_loader(playback_source_ref)
        self._buffer_cache[sample_source_key] = (event_buffer, sample_rate)
        event_signature = ",".join(
            f"{event.start:.6f}:{int(event.muted)}:{int('demoted' in getattr(event, 'badges', []))}"
            for event in presentation_events
        )
        rendered_source_key = f"event:{playback_source_ref}:{event_signature}"
        cached_render = self._buffer_cache.get(rendered_source_key)
        if cached_render is None:
            rendered = PresentationPlaybackRuntime._render_event_slice_buffer(
                event_buffer,
                sample_rate,
                presentation_events=presentation_events,
            )
            if rendered.size == 0:
                return None
            self._buffer_cache[rendered_source_key] = (rendered, sample_rate)
        else:
            rendered, sample_rate = cached_render
        self._buffer_cache[rendered_source_key] = (rendered, sample_rate)
        return RuntimeAudioLayer(
            layer_id=layer_id,
            name=title,
            gain_db=gain_db,
            output_bus=output_bus,
            source_key=rendered_source_key,
            cache_keys=(sample_source_key, rendered_source_key),
            buffer=rendered,
            sample_rate=sample_rate,
            source_ref=playback_source_ref,
        )

    @staticmethod
    def _is_event_slice_layer(layer: object) -> bool:
        return bool(
            is_event_like_layer_kind(getattr(layer, "kind", None))
            and getattr(layer, "playback_enabled", False)
            and getattr(layer, "playback_mode", None) == PlaybackMode.EVENT_SLICE
            and getattr(layer, "playback_source_ref", None)
        )

    @staticmethod
    def _render_event_slice_buffer(
        event_buffer: np.ndarray,
        sample_rate: int,
        *,
        presentation_events: list,
    ) -> np.ndarray:
        if event_buffer.size == 0:
            return np.zeros(0, dtype=np.float32)

        active_events = [
            event
            for event in presentation_events
            if not event.muted and "demoted" not in getattr(event, "badges", [])
        ]
        if not active_events:
            return np.zeros(0, dtype=np.float32)

        start_samples = [
            max(0, int(round(float(event.start) * sample_rate))) for event in active_events
        ]
        total_samples = max(start_samples) + int(event_buffer.shape[0])
        if event_buffer.ndim == 1:
            rendered = np.zeros(total_samples, dtype=np.float32)
        else:
            rendered = np.zeros((total_samples, event_buffer.shape[1]), dtype=np.float32)

        for start_sample in start_samples:
            end_sample = start_sample + int(event_buffer.shape[0])
            rendered[start_sample:end_sample] += event_buffer

        np.clip(rendered, -1.0, 1.0, out=rendered)
        return rendered
