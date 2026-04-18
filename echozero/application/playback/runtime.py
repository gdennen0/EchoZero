"""Application playback runtime boundary over the concrete audio backend.

This owns presentation-to-backend playback mapping and publishes the narrow
runtime/timing surface the UI needs without exposing backend internals as
timeline semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from echozero.application.playback.models import PlaybackSource, PlaybackState, PlaybackTimingSnapshot
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import LayerKind, PlaybackMode, PlaybackStatus
from echozero.audio.engine import AudioEngine


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (float(gain_db) / 20.0))


def _load_mono_audio(path: str | Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False), int(sample_rate)


@dataclass(slots=True)
class RuntimeAudioLayer:
    layer_id: str
    name: str
    gain_db: float
    source_key: str
    cache_keys: tuple[str, ...]
    buffer: np.ndarray
    sample_rate: int
    source_ref: str | None = None


class PresentationPlaybackRuntime:
    """Playback runtime facade for the current presentation-backed app shell."""

    _MONITOR_LAYER_ID = "__ez_monitor__"

    def __init__(
        self,
        engine: AudioEngine | None = None,
        *,
        engine_factory: Callable[[], AudioEngine] | None = None,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] = _load_mono_audio,
    ) -> None:
        self._engine = engine or (engine_factory() if engine_factory is not None else AudioEngine())
        self._audio_loader = audio_loader
        self._loaded_source_key: str | None = None
        self._buffer_cache: dict[str, tuple[np.ndarray, int]] = {}

    @property
    def engine(self) -> AudioEngine:
        return self._engine

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        self._sync_active_layer(presentation)

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self._sync_active_layer(presentation)

    def play(self) -> None:
        self._engine.play()

    def pause(self) -> None:
        self._engine.pause()

    def stop(self) -> None:
        self._engine.stop()

    def seek(self, position_seconds: float) -> None:
        self._engine.seek_seconds(position_seconds)

    def current_time_seconds(self) -> float:
        return float(self._engine.audible_time_seconds)

    def timing_snapshot(self) -> PlaybackTimingSnapshot:
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
                max(0.0, float(snapshot_monotonic))
                if snapshot_monotonic is not None
                else None
            ),
            is_playing=is_playing,
        )

    def is_playing(self) -> bool:
        return bool(self._engine.transport.is_playing)

    def shutdown(self) -> None:
        self._engine.shutdown()

    def presentation_signature(self, presentation: TimelinePresentation) -> tuple[tuple[str, str], ...]:
        active_layer = self._select_active_runtime_layer(presentation)
        if active_layer is None:
            return ()
        return ((active_layer.layer_id, active_layer.source_key),)

    def snapshot_state(self, presentation: TimelinePresentation) -> PlaybackState:
        active_layer = self._select_active_runtime_layer(presentation)
        active_sources: list[PlaybackSource] = []
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
        return PlaybackState(
            status=PlaybackStatus.PLAYING if self.is_playing() else PlaybackStatus.STOPPED,
            active_sources=active_sources,
            latency_ms=float(self._engine.reported_output_latency_seconds) * 1000.0,
            backend_name="sounddevice",
            active_layer_id=presentation.active_playback_layer_id,
            active_take_id=presentation.active_playback_take_id,
            output_sample_rate=int(self._engine.sample_rate),
            output_channels=int(getattr(self._engine, "_channels", 1)),
        )

    def _sync_active_layer(self, presentation: TimelinePresentation) -> None:
        active_layer = self._select_active_runtime_layer(presentation)
        desired_cache_keys = set(active_layer.cache_keys) if active_layer is not None else set()
        stale_cache_keys = [key for key in self._buffer_cache if key not in desired_cache_keys]
        for stale_key in stale_cache_keys:
            self._buffer_cache.pop(stale_key, None)

        if active_layer is None:
            if self._loaded_source_key is not None:
                self._engine.remove_layer(self._MONITOR_LAYER_ID)
                self._loaded_source_key = None
            return
        if self._loaded_source_key != active_layer.source_key:
            if self._loaded_source_key is not None:
                self._engine.remove_layer(self._MONITOR_LAYER_ID)
            self._engine.add_layer(
                self._MONITOR_LAYER_ID,
                active_layer.buffer,
                active_layer.sample_rate,
                name=active_layer.name,
                volume=_db_to_linear(active_layer.gain_db),
            )
            self._loaded_source_key = active_layer.source_key

        engine_layer = self._engine.mixer.get_layer(self._MONITOR_LAYER_ID)
        if engine_layer is not None:
            engine_layer.muted = False
            engine_layer.volume = _db_to_linear(active_layer.gain_db)

    def _select_active_runtime_layer(self, presentation: TimelinePresentation) -> RuntimeAudioLayer | None:
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
        source_audio_path = getattr(layer, "source_audio_path", None)
        if source_audio_path:
            return self._build_audio_runtime_layer(
                layer_id=layer_id,
                title=title,
                gain_db=gain_db,
                source_audio_path=source_audio_path,
            )
        if not self._is_event_slice_layer(layer):
            return None
        return self._build_event_runtime_layer(
            layer_id=layer_id,
            title=title,
            gain_db=gain_db,
            playback_source_ref=getattr(layer, "playback_source_ref"),
            presentation_events=list(getattr(layer, "events")),
        )

    def _runtime_layer_from_take(self, layer: object, take: object) -> RuntimeAudioLayer | None:
        layer_id = str(getattr(layer, "layer_id"))
        take_id = str(getattr(take, "take_id"))
        title = f"{getattr(layer, 'title')} · {getattr(take, 'name')}"
        gain_db = float(getattr(layer, "gain_db", 0.0))
        source_audio_path = getattr(take, "source_audio_path", None)
        if source_audio_path:
            return self._build_audio_runtime_layer(
                layer_id=f"{layer_id}:{take_id}",
                title=title,
                gain_db=gain_db,
                source_audio_path=source_audio_path,
            )
        if not self._is_event_slice_layer(take):
            return None
        return self._build_event_runtime_layer(
            layer_id=f"{layer_id}:{take_id}",
            title=title,
            gain_db=gain_db,
            playback_source_ref=getattr(take, "playback_source_ref"),
            presentation_events=list(getattr(take, "events")),
        )

    def _build_audio_runtime_layer(
        self,
        *,
        layer_id: str,
        title: str,
        gain_db: float,
        source_audio_path: str,
    ) -> RuntimeAudioLayer:
        source_key = f"audio:{source_audio_path}"
        buffer, sample_rate = self._buffer_cache.get(source_key) or self._audio_loader(source_audio_path)
        self._buffer_cache[source_key] = (buffer, sample_rate)
        return RuntimeAudioLayer(
            layer_id=layer_id,
            name=title,
            gain_db=gain_db,
            source_key=source_key,
            cache_keys=(source_key,),
            buffer=buffer,
            sample_rate=sample_rate,
            source_ref=source_audio_path,
        )

    def _build_event_runtime_layer(
        self,
        *,
        layer_id: str,
        title: str,
        gain_db: float,
        playback_source_ref: str,
        presentation_events: list,
    ) -> RuntimeAudioLayer | None:
        sample_source_key = f"event-sample:{playback_source_ref}"
        event_buffer, sample_rate = self._buffer_cache.get(sample_source_key) or self._audio_loader(playback_source_ref)
        self._buffer_cache[sample_source_key] = (event_buffer, sample_rate)
        event_signature = ",".join(
            f"{event.start:.6f}:{int(event.muted)}"
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
            source_key=rendered_source_key,
            cache_keys=(sample_source_key, rendered_source_key),
            buffer=rendered,
            sample_rate=sample_rate,
            source_ref=playback_source_ref,
        )

    @staticmethod
    def _is_event_slice_layer(layer: object) -> bool:
        return bool(
            getattr(layer, "kind", None) == LayerKind.EVENT
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

        active_events = [event for event in presentation_events if not event.muted]
        if not active_events:
            return np.zeros(0, dtype=np.float32)

        start_samples = [max(0, int(round(float(event.start) * sample_rate))) for event in active_events]
        total_samples = max(start_samples) + int(event_buffer.size)
        rendered = np.zeros(total_samples, dtype=np.float32)

        for start_sample in start_samples:
            end_sample = start_sample + int(event_buffer.size)
            rendered[start_sample:end_sample] += event_buffer

        np.clip(rendered, -1.0, 1.0, out=rendered)
        return rendered
