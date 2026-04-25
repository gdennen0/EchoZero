from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echozero.application.shared.ids import TimelineId
from echozero.domain.types import EventData
from echozero.persistence.entities import LayerRecord
from echozero.takes import Take as PersistedTake, TakeSource
from echozero.ui.qt.app_shell_project_timeline_storage import build_storage_layer


class _StubTakeRepository:
    def __init__(self, takes_by_layer: dict[str, list[PersistedTake]]) -> None:
        self._takes_by_layer = takes_by_layer

    def list_by_layer(self, layer_id: str) -> list[PersistedTake]:
        return list(self._takes_by_layer.get(layer_id, []))


class _StubProjectStorage:
    def __init__(self, takes_by_layer: dict[str, list[PersistedTake]]) -> None:
        self.working_dir = Path.cwd()
        self.takes = _StubTakeRepository(takes_by_layer)


def test_build_storage_layer_defaults_take_lanes_collapsed():
    takes = _layer_takes()
    storage = _StubProjectStorage({"layer_kick": takes})

    layer, _, _ = build_storage_layer(
        storage,
        TimelineId("timeline_runtime"),
        _layer_record(layer_id="layer_kick", state_flags={}),
    )

    assert layer is not None
    assert layer.presentation_hints.expanded is False


def test_build_storage_layer_restores_saved_take_lane_expansion():
    takes = _layer_takes()
    storage = _StubProjectStorage({"layer_snare": takes})

    layer, _, _ = build_storage_layer(
        storage,
        TimelineId("timeline_runtime"),
        _layer_record(
            layer_id="layer_snare",
            state_flags={"take_lanes_expanded": True},
        ),
    )

    assert layer is not None
    assert layer.presentation_hints.expanded is True


def test_build_storage_layer_resolves_event_take_source_audio_path_from_snapshot():
    take = PersistedTake.create(
        data=EventData(layers=()),
        label="Take 1",
        origin="pipeline",
        source=TakeSource(
            block_id="classify_drums",
            block_type="BinaryDrumClassify",
            settings_snapshot={
                "pipeline_id": "extract_song_drum_events",
                "output_name": "classified_drums",
                "source_audio_path": "stems/drums.wav",
            },
            run_id="run_1",
        ),
        is_main=True,
    )
    storage = _StubProjectStorage({"layer_kick": [take]})

    layer, layer_audio, take_audio = build_storage_layer(
        storage,
        TimelineId("timeline_runtime"),
        _layer_record(layer_id="layer_kick", state_flags={}),
    )

    assert layer is not None
    assert layer_audio.source_audio_path is not None
    assert Path(layer_audio.source_audio_path).as_posix().endswith("stems/drums.wav")
    take_audio_fields = next(iter(take_audio.values()))
    assert take_audio_fields.source_audio_path is not None
    assert Path(take_audio_fields.source_audio_path).as_posix().endswith("stems/drums.wav")


def _layer_record(*, layer_id: str, state_flags: dict[str, Any]) -> LayerRecord:
    return LayerRecord(
        id=layer_id,
        song_version_id="song_version_runtime",
        name="drums",
        layer_type="manual",
        color="#00ff00",
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline=None,
        created_at=datetime.now(timezone.utc),
        state_flags=state_flags,
        provenance={},
    )


def _layer_takes() -> list[PersistedTake]:
    return [
        PersistedTake.create(
            data=EventData(layers=()),
            label="Take 1",
            origin="user",
            is_main=True,
        ),
        PersistedTake.create(
            data=EventData(layers=()),
            label="Take 2",
            origin="user",
            is_main=False,
        ),
    ]
