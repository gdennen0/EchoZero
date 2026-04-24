"""Shared helpers for runtime-audio support cases.
Exists to keep fake runtime-audio backends and presentation builders out of the compatibility wrapper.
Connects the behavior-owned runtime-audio support modules to one stable shared seam.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind, PlaybackMode
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.intents import (
    Pause,
    Play,
    Seek,
    SelectLayer,
    SetActivePlaybackTarget,
    Stop,
)
from echozero.audio.engine import AudioEngine
from echozero.ui.qt.timeline.blocks.ruler import seek_time_for_x
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.runtime_audio import (
    RuntimeAudioTimingSnapshot,
    TimelineRuntimeAudioController,
)
from echozero.ui.qt.timeline.widget import TimelineWidget


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.blocksize = kwargs.get("blocksize")
        self.device = kwargs.get("device")
        self.latency = kwargs.get("latency", 0.0)
        self.prime_output_buffers_using_stream_callback = kwargs.get(
            "prime_output_buffers_using_stream_callback"
        )
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _fake_stream_factory(**kwargs):
    return FakeStream(**kwargs)


class FakeRuntimeAudio:
    def __init__(self):
        self.playing = False
        self.current_time = 0.0
        self.mix_states: list[TimelinePresentation] = []
        self.snapshot: RuntimeAudioTimingSnapshot | None = None

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        return None

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self.mix_states.append(presentation)

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def stop(self) -> None:
        self.playing = False
        self.current_time = 0.0

    def seek(self, position_seconds: float) -> None:
        self.current_time = position_seconds

    def current_time_seconds(self) -> float:
        return self.current_time

    def timing_snapshot(self) -> RuntimeAudioTimingSnapshot:
        if self.snapshot is not None:
            return self.snapshot
        return RuntimeAudioTimingSnapshot(
            audible_time_seconds=self.current_time,
            clock_time_seconds=self.current_time,
            snapshot_monotonic_seconds=None,
            is_playing=self.playing,
        )

    def is_playing(self) -> bool:
        return self.playing

    def presentation_signature(self, presentation: TimelinePresentation):
        return tuple(
            (
                str(layer.layer_id),
                layer.source_audio_path or layer.playback_source_ref or "",
            )
            for layer in presentation.layers
            if layer.source_audio_path or layer.playback_source_ref
        )


class RecordingRuntimeAudio:
    def __init__(self):
        self.calls: list[tuple[str, float | None]] = []
        self.playing = False
        self.current_time = 0.0

    def play(self) -> None:
        self.playing = True
        self.calls.append(("play", None))

    def pause(self) -> None:
        self.playing = False
        self.calls.append(("pause", None))

    def stop(self) -> None:
        self.playing = False
        self.current_time = 0.0
        self.calls.append(("stop", None))

    def seek(self, position_seconds: float) -> None:
        self.current_time = float(position_seconds)
        self.calls.append(("seek", position_seconds))

    def current_time_seconds(self) -> float:
        return self.current_time

    def is_playing(self) -> bool:
        return self.playing

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        return None

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self.calls.append(("mix", None))

    def presentation_signature(self, presentation: TimelinePresentation):
        return tuple(
            (
                str(layer.layer_id),
                layer.source_audio_path or layer.playback_source_ref or "",
            )
            for layer in presentation.layers
            if layer.source_audio_path or layer.playback_source_ref
        )


class CountingRuntimeAudio(FakeRuntimeAudio):
    def __init__(self):
        super().__init__()
        self.build_calls = 0
        self.mix_calls = 0

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        self.build_calls += 1

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self.mix_calls += 1


class SignatureAwareRuntimeAudio(CountingRuntimeAudio):
    @staticmethod
    def _is_event_slice_layer(layer: LayerPresentation) -> bool:
        return (
            layer.kind == LayerKind.EVENT
            and layer.playback_enabled
            and layer.playback_mode == PlaybackMode.EVENT_SLICE
            and bool(layer.playback_source_ref)
        )

    def presentation_signature(
        self, presentation: TimelinePresentation
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                str(layer.layer_id),
                (
                    layer.source_audio_path
                    if layer.source_audio_path is not None
                    else f"event:{layer.playback_source_ref}:{','.join(f'{event.start:.6f}:{int(event.muted)}' for event in layer.events)}"
                ),
            )
            for layer in presentation.layers
            if layer.source_audio_path or self._is_event_slice_layer(layer)
        )


def _audio_presentation() -> TimelinePresentation:
    base = build_demo_app().presentation()
    audio_layer = LayerPresentation(
        layer_id=LayerId("runtime_audio"),
        title="Runtime Audio",
        kind=LayerKind.AUDIO,
        source_audio_path="demo.wav",
    )
    return replace(
        base,
        layers=[audio_layer],
        active_playback_layer_id=audio_layer.layer_id,
        playhead=0.0,
        is_playing=False,
        current_time_label="00:00.00",
    )


def _event_slice_presentation() -> TimelinePresentation:
    base = build_demo_app().presentation()
    audio_layer = LayerPresentation(
        layer_id=LayerId("bed"),
        title="Bed",
        kind=LayerKind.AUDIO,
        source_audio_path="bed.wav",
    )
    kick_layer = LayerPresentation(
        layer_id=LayerId("kick_lane"),
        title="Kick",
        kind=LayerKind.EVENT,
        playback_enabled=True,
        playback_mode=PlaybackMode.EVENT_SLICE,
        playback_source_ref="kick.wav",
        events=[
            EventPresentation(
                event_id=EventId("kick_1"),
                start=0.5,
                end=0.6,
                label="Kick",
            ),
            EventPresentation(
                event_id=EventId("kick_2"),
                start=1.0,
                end=1.1,
                label="Kick",
            ),
        ],
    )
    return replace(
        base,
        layers=[audio_layer, kick_layer],
        playhead=0.0,
        is_playing=False,
        current_time_label="00:00.00",
    )



__all__ = [name for name in globals() if not name.startswith("__")]
