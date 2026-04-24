"""Shared helpers for session support cases.
Exists to keep storage fixtures and graph builders out of the compatibility wrapper.
Connects the behavior-owned session support modules to one stable shared seam.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from echozero.domain.enums import (
    BlockCategory,
    BlockState,
    Direction,
    PortType,
)
from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    BlockStateChangedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    SettingsChangedEvent,
    create_event_id,
)
from echozero.domain.graph import Graph
from echozero.domain.types import (
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.event_bus import EventBus
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import (
    LayerRecord,
    ProjectRecord,
    ProjectSettingsRecord,
    SongRecord,
    SongVersionRecord,
)
from echozero.persistence.repositories import PipelineConfigRepository
from echozero.persistence.session import ProjectStorage
from echozero.takes import Take, TakeSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(correlation_id: str = "") -> dict:
    """Common fields for constructing a DomainEvent."""
    return {
        "event_id": create_event_id(),
        "timestamp": time.time(),
        "correlation_id": correlation_id or _uid(),
    }


def _make_graph() -> Graph:
    """Build a small two-block graph for round-trip testing."""
    graph = Graph()
    b1 = Block(
        id="b1",
        name="Source",
        block_type="audio_source",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(
            Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings({"file": "test.wav"}),
    )
    b2 = Block(
        id="b2",
        name="Detector",
        block_type="onset_detector",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        ),
        output_ports=(
            Port(name="events_out", port_type=PortType.EVENT, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings({"threshold": 0.3}),
    )
    graph.add_block(b1)
    graph.add_block(b2)
    graph.add_connection(Connection(
        source_block_id="b1",
        source_output_name="audio_out",
        target_block_id="b2",
        target_input_name="audio_in",
    ))
    return graph


def _make_song(project_id: str, **kw) -> SongRecord:
    defaults = dict(
        id=_uid(), project_id=project_id, title="SongRecord A",
        artist="Artist", order=0, active_version_id=None,
    )
    defaults.update(kw)
    return SongRecord(**defaults)


def _make_version(song_id: str, **kw) -> SongVersionRecord:
    defaults = dict(
        id=_uid(), song_id=song_id, label="Studio Mix",
        audio_file="audio/song.wav", duration_seconds=180.0,
        original_sample_rate=44100, audio_hash="abc123", created_at=_now(),
    )
    defaults.update(kw)
    return SongVersionRecord(**defaults)


def _make_layer(song_version_id: str, **kw) -> LayerRecord:
    defaults = dict(
        id=_uid(), song_version_id=song_version_id, name="Drums",
        layer_type="analysis", color="#FF0000", order=0,
        visible=True, locked=False, parent_layer_id=None,
        source_pipeline=None, created_at=_now(),
    )
    defaults.update(kw)
    return LayerRecord(**defaults)


def _make_event_data() -> EventData:
    return EventData(layers=(
        Layer(id=_uid(), name="onsets", events=(
            Event(
                id=_uid(), time=1.0, duration=0.1,
                classifications={"type": "kick"}, metadata={}, origin="pipeline",
            ),
        )),
    ))


def _make_take(is_main: bool = False, **kw) -> Take:
    defaults = dict(
        id=_uid(), label="Take 1", data=_make_event_data(),
        origin="pipeline", source=None, created_at=_now(),
        is_main=is_main, notes="",
    )
    defaults.update(kw)
    return Take(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_root(tmp_path):
    """Provide a temporary working dir root for each test."""
    return tmp_path / "working"


@pytest.fixture
def event_bus():
    bus = EventBus()
    yield bus
    bus.clear()


# ---------------------------------------------------------------------------
# DirtyTracker unit tests
# ---------------------------------------------------------------------------



__all__ = [name for name in globals() if not name.startswith("__")]
