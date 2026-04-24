"""Shared helpers for persistence support cases.
Exists to keep repository factories and the shared database fixture out of the compatibility wrapper.
Connects the behavior-owned persistence support modules to one stable shared seam.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import PersistenceError
from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import (
    LayerRecord,
    TimelineRegionRecord,
    ProjectRecord,
    ProjectSettingsRecord,
    SongRecord,
    PipelineConfigRecord,
    SongVersionRecord,
)
from echozero.persistence.repositories import (
    LayerRepository,
    PipelineConfigRepository,
    ProjectRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
    TimelineRegionRepository,
)
from echozero.persistence.schema import (
    SCHEMA_VERSION,
    get_schema_version,
    init_db,
    set_schema_version,
)
from echozero.takes import Take, TakeSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_project(**kw) -> ProjectRecord:
    defaults = dict(
        id=_uid(), name="Test ProjectRecord", settings=ProjectSettingsRecord(),
        created_at=_now(), updated_at=_now(),
    )
    defaults.update(kw)
    return ProjectRecord(**defaults)


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


def _make_audio_data() -> AudioData:
    return AudioData(
        sample_rate=44100, duration=120.5,
        file_path="audio/test.wav", channel_count=2,
    )


def _make_take(is_main: bool = False, **kw) -> Take:
    defaults = dict(
        id=_uid(), label="Take 1", data=_make_event_data(),
        origin="pipeline", source=None, created_at=_now(),
        is_main=is_main, notes="",
    )
    defaults.update(kw)
    return Take(**defaults)


def _make_pipeline_config(song_version_id: str, **kw) -> PipelineConfigRecord:
    defaults = dict(
        id=_uid(), song_version_id=song_version_id,
        template_id="onset_detection",
        name="Onset Detection",
        graph_json='{"blocks": [], "connections": []}',
        outputs_json='[]',
        knob_values={"threshold": 0.3},
        created_at=_now(),
        updated_at=_now(),
    )
    defaults.update(kw)
    return PipelineConfigRecord(**defaults)


def _make_timeline_region(song_version_id: str, **kw) -> TimelineRegionRecord:
    defaults = dict(
        id=_uid(),
        song_version_id=song_version_id,
        label="Region 1",
        start_seconds=0.0,
        end_seconds=1.0,
        color=None,
        order_index=0,
        kind="custom",
        created_at=_now(),
    )
    defaults.update(kw)
    return TimelineRegionRecord(**defaults)


# ---------------------------------------------------------------------------
# Fixture: fresh in-memory database for every test
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db(c)
    return c


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------



__all__ = [name for name in globals() if not name.startswith("__")]
