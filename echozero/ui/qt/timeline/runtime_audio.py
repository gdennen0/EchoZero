from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import LayerKind, PlaybackMode
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
    muted: bool
    soloed: bool
    source_key: str
    buffer: np.ndarray
    sample_rate: int


class TimelineRuntimeAudioController:
    def __init__(
        self,
        engine: AudioEngine | None = None,
        *,
        engine_factory: Callable[[], AudioEngine] | None = None,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] = _load_mono_audio,
    ) -> None:
        self._engine = engine or (engine_factory() if engine_factory is not None else AudioEngine())
        self._audio_loader = audio_loader
        self._loaded_paths: dict[str, str] = {}

    @property
    def engine(self) -> AudioEngine:
        return self._engine

    def source_signature(self, presentation: TimelinePresentation) -> tuple[tuple[str, str], ...]:
        """Stable signature describing each playable layer's backing audio source."""

        return tuple(
            (layer.layer_id, layer.source_key)
            for layer in self._runtime_layers(presentation)
        )

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        runtime_layers = self._runtime_layers(presentation)
        desired_ids = {layer.layer_id for layer in runtime_layers}

        for existing in list(self._loaded_paths):
            if existing not in desired_ids:
                self._engine.remove_layer(existing)
                self._loaded_paths.pop(existing, None)

        for layer in runtime_layers:
            previous_source = self._loaded_paths.get(layer.layer_id)
            if previous_source != layer.source_key:
                if previous_source is not None:
                    self._engine.remove_layer(layer.layer_id)
                self._engine.add_layer(
                    layer.layer_id,
                    layer.buffer,
                    layer.sample_rate,
                    name=layer.name,
                    volume=_db_to_linear(layer.gain_db),
                )
                self._loaded_paths[layer.layer_id] = layer.source_key

        self.apply_mix_state(presentation)

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        desired_ids = {
            str(layer.layer_id)
            for layer in presentation.layers
            if layer.source_audio_path or self._is_event_slice_layer(layer)
        }
        for layer in presentation.layers:
            layer_id = str(layer.layer_id)
            if layer_id not in desired_ids:
                continue
            engine_layer = self._engine.mixer.get_layer(layer_id)
            if engine_layer is None:
                continue
            engine_layer.muted = bool(layer.muted)
            engine_layer.volume = _db_to_linear(layer.gain_db)
            self._engine.mixer.set_solo(layer_id, bool(layer.soloed))

    def play(self) -> None:
        self._engine.play()

    def pause(self) -> None:
        self._engine.pause()

    def stop(self) -> None:
        self._engine.stop()

    def seek(self, position_seconds: float) -> None:
        self._engine.seek_seconds(position_seconds)

    def current_time_seconds(self) -> float:
        return float(self._engine.clock.position_seconds)

    def is_playing(self) -> bool:
        return bool(self._engine.transport.is_playing)

    def shutdown(self) -> None:
        self._engine.shutdown()

    def _runtime_layers(self, presentation: TimelinePresentation) -> list[RuntimeAudioLayer]:
        layers: list[RuntimeAudioLayer] = []
        for layer in presentation.layers:
            layer_id = str(layer.layer_id)
            if layer.source_audio_path:
                buffer, sample_rate = self._audio_loader(layer.source_audio_path)
                layers.append(
                    RuntimeAudioLayer(
                        layer_id=layer_id,
                        name=layer.title,
                        gain_db=layer.gain_db,
                        muted=layer.muted,
                        soloed=layer.soloed,
                        source_key=f"audio:{layer.source_audio_path}",
                        buffer=buffer,
                        sample_rate=sample_rate,
                    )
                )
                continue

            if not self._is_event_slice_layer(layer):
                continue

            event_buffer, sample_rate = self._audio_loader(layer.playback_source_ref)
            rendered = TimelineRuntimeAudioController._render_event_slice_buffer(
                event_buffer,
                sample_rate,
                presentation_events=layer.events,
            )
            if rendered.size == 0:
                continue
            event_signature = ",".join(
                f"{event.start:.6f}:{int(event.muted)}"
                for event in layer.events
            )
            layers.append(
                RuntimeAudioLayer(
                    layer_id=layer_id,
                    name=layer.title,
                    gain_db=layer.gain_db,
                    muted=layer.muted,
                    soloed=layer.soloed,
                    source_key=f"event:{layer.playback_source_ref}:{event_signature}",
                    buffer=rendered,
                    sample_rate=sample_rate,
                )
            )
        return layers

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
