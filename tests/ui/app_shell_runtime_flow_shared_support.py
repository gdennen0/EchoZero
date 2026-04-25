"""Shared helpers for app-shell runtime-flow support cases.
Exists to keep runtime harnesses, helper executors, and assertions out of the compatibility wrapper.
Connects behavior-owned runtime-flow support modules to one stable shared support seam.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.intents import (
    Play,
    Seek,
    SelectEvent,
    SetActivePlaybackTarget,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.application.timeline.object_actions import (
    ApplyCopySource,
    ChangeSessionScope,
    PreviewCopySource,
    ResetSessionDefaults,
    ReplaceSessionValues,
    RunSession,
    SaveAndRunSession,
    SaveSessionToDefaults,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.domain.types import AudioData, Event as DomainEvent
from echozero.domain.types import EventData, Layer as DomainLayer
from echozero.pipelines.registry import get_registry
from echozero.result import ok
from echozero.testing.analysis_mocks import (
    build_mock_analysis_service,
    write_test_model,
    write_test_wav,
)
from echozero.testing.app_flow import AppFlowHarness
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.waveform_cache import clear_waveform_cache, get_cached_waveform


class _CountedRuntimeAudio:
    def __init__(self):
        self.build_calls = 0
        self.play_calls = 0
        self.preview_calls: list[tuple[str, float, float, float]] = []
        self.is_playing_state = False

    def build_for_presentation(self, _presentation) -> None:
        self.build_calls += 1

    def apply_mix_state(self, _presentation) -> None:
        return None

    def play(self) -> None:
        self.play_calls += 1
        self.is_playing_state = True

    def pause(self) -> None:
        self.is_playing_state = False

    def stop(self) -> None:
        self.is_playing_state = False

    def seek(self, _position_seconds: float) -> None:
        return None

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        self.preview_calls.append(
            (source_ref, float(start_seconds), float(end_seconds), float(gain_db))
        )
        return True

    def current_time_seconds(self) -> float:
        return 0.0

    def is_playing(self) -> bool:
        return self.is_playing_state

    def shutdown(self) -> None:
        return None


def _wait_until(predicate, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _CollidingBinaryDrumClassifyExecutor:
    def execute(self, block_id: str, context):
        block = context.graph.blocks[block_id]
        target_class = str(block.settings.get("target_class", "")).strip().lower()
        input_events = _merged_binary_drum_input_events(block_id, context)
        kick_events: list[DomainEvent] = []
        snare_events: list[DomainEvent] = []
        for event in input_events:
            kick_events.append(
                DomainEvent(
                    id=event.id,
                    time=event.time,
                    duration=event.duration,
                    classifications={"class": "kick", "confidence": "0.99"},
                    metadata={**event.metadata, "classified": True},
                    origin="binary_classify:kick",
                )
            )
            snare_events.append(
                DomainEvent(
                    id=event.id,
                    time=event.time + 0.1,
                    duration=event.duration,
                    classifications={"class": "snare", "confidence": "0.97"},
                    metadata={**event.metadata, "classified": True},
                    origin="binary_classify:snare",
                )
            )
        if target_class == "kick":
            return ok(EventData(layers=(DomainLayer(id="kick", name="kick", events=tuple(kick_events)),)))
        if target_class == "snare":
            return ok(
                EventData(layers=(DomainLayer(id="snare", name="snare", events=tuple(snare_events)),))
            )
        return ok(
            EventData(
                layers=(
                    DomainLayer(id="kick", name="kick", events=tuple(kick_events)),
                    DomainLayer(id="snare", name="snare", events=tuple(snare_events)),
                )
            )
        )


class _CaptureDetectOnsetsAudioExecutor:
    def __init__(self) -> None:
        self.audio_paths: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def execute(self, block_id: str, context):
        audio = context.get_input(block_id, "audio_in", AudioData)
        assert audio is not None
        audio_path = str(audio.file_path)
        self.audio_paths.append(audio_path)
        self.calls.append((block_id, audio_path))
        event = DomainEvent(
            id="evt_1",
            time=0.25,
            duration=0.05,
            classifications={"namespace:onset": "hit"},
            metadata={},
            origin="detect_onsets",
        )
        return ok(
            EventData(
                layers=(
                    DomainLayer(
                        id="layer_onsets",
                        name="Onsets",
                        events=(event,),
                    ),
                )
            )
        )


class _ThresholdAwareDetectOnsetsExecutor:
    def __init__(self) -> None:
        self.thresholds: list[float] = []

    def execute(self, block_id: str, context):
        block = context.graph.blocks[block_id]
        threshold = float(block.settings.get("threshold", 0.3))
        self.thresholds.append(threshold)
        event_count = max(1, int(round((1.0 - threshold) * 10)))
        events = tuple(
            DomainEvent(
                id=f"evt_{index}",
                time=0.1 * index,
                duration=0.05,
                classifications={"namespace:onset": "hit"},
                metadata={},
                origin="detect_onsets",
            )
            for index in range(event_count)
        )
        return ok(
            EventData(
                layers=(
                    DomainLayer(
                        id="layer_onsets",
                        name="Onsets",
                        events=events,
                    ),
                )
            )
        )


class _CaptureBinaryDrumClassifyAudioExecutor:
    def __init__(self) -> None:
        self.audio_paths: list[str] = []
        self.calls: list[tuple[str, str, str]] = []

    def execute(self, block_id: str, context):
        audio = context.get_input(block_id, "audio_in", AudioData)
        assert audio is not None
        block = context.graph.blocks[block_id]
        target_class = str(block.settings.get("target_class", "")).strip().lower()
        audio_path = str(audio.file_path)
        self.audio_paths.append(audio_path)
        self.calls.append((block_id, target_class, audio_path))
        input_events = _merged_binary_drum_input_events(block_id, context)
        kick_events: list[DomainEvent] = []
        snare_events: list[DomainEvent] = []
        for event in input_events:
            kick_events.append(
                DomainEvent(
                    id=f"{event.id}_kick",
                    time=event.time,
                    duration=event.duration,
                    classifications={"class": "kick", "confidence": "0.99"},
                    metadata={**event.metadata, "classified": True},
                    origin="binary_classify:kick",
                )
            )
            snare_events.append(
                DomainEvent(
                    id=f"{event.id}_snare",
                    time=event.time + 0.1,
                    duration=event.duration,
                    classifications={"class": "snare", "confidence": "0.97"},
                    metadata={**event.metadata, "classified": True},
                    origin="binary_classify:snare",
                )
            )
        if target_class == "kick":
            return ok(EventData(layers=(DomainLayer(id="kick", name="kick", events=tuple(kick_events)),)))
        if target_class == "snare":
            return ok(
                EventData(layers=(DomainLayer(id="snare", name="snare", events=tuple(snare_events)),))
            )
        return ok(
            EventData(
                layers=(
                    DomainLayer(id="kick", name="kick", events=tuple(kick_events)),
                    DomainLayer(id="snare", name="snare", events=tuple(snare_events)),
                )
            )
        )


class _CapturePyTorchAudioClassifyAudioExecutor:
    def __init__(self) -> None:
        self.audio_paths: list[str] = []

    def execute(self, block_id: str, context):
        audio = context.get_input(block_id, "audio_in", AudioData)
        assert audio is not None
        self.audio_paths.append(str(audio.file_path))
        event_data = context.get_input(block_id, "events_in", EventData)
        assert event_data is not None
        classified_layers: list[DomainLayer] = []
        for layer in event_data.layers:
            classified_events: list[DomainEvent] = []
            for event in layer.events:
                classified_events.append(
                    DomainEvent(
                        id=event.id,
                        time=event.time,
                        duration=event.duration,
                        classifications={"class": "kick", "confidence": "0.99"},
                        metadata={**event.metadata, "classified": True},
                        origin="classify",
                    )
                )
            classified_layers.append(
                DomainLayer(id=layer.id, name="Kick", events=tuple(classified_events))
            )
        return ok(EventData(layers=tuple(classified_layers)))


def _repo_local_temp_root() -> Path:
    root = Path("C:/Users/griff/.codex/memories/test_app_shell_runtime_flow") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _assert_waveform_registered(waveform_key: str | None) -> None:
    assert waveform_key is not None
    cached = get_cached_waveform(waveform_key)
    assert cached is not None
    assert cached.peaks.size > 0


def _merged_binary_drum_input_events(
    block_id: str,
    context,
) -> tuple[DomainEvent, ...]:
    event_batches: list[EventData] = []
    for port_name in ("events_in", "kick_events_in", "snare_events_in"):
        event_data = context.get_input(block_id, port_name, EventData)
        if event_data is not None:
            event_batches.append(event_data)

    assert event_batches
    merged: list[DomainEvent] = []
    seen: set[tuple[str, float, float]] = set()
    for event_data in event_batches:
        for layer in event_data.layers:
            for event in layer.events:
                key = (str(event.id), float(event.time), float(event.duration))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(event)
    return tuple(merged)



__all__ = [name for name in globals() if not name.startswith("__")]
