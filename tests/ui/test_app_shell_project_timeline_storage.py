from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echozero.application.shared.ids import TimelineId
from echozero.domain.types import EventData
from echozero.application.timeline.object_content_persistence import (
    content_id_for_take,
    object_id_for_layer,
    revision_id_for_take,
)
from echozero.persistence.entities import LayerRecord, ObjectContentRecord, TimelineObjectRecord
from echozero.takes import Take as PersistedTake, TakeSource
from echozero.ui.qt.app_shell_project_timeline_storage import build_storage_layer


class _StubTakeRepository:
    def __init__(self, takes_by_layer: dict[str, list[PersistedTake]]) -> None:
        self._takes_by_layer = takes_by_layer

    def list_by_layer(self, layer_id: str) -> list[PersistedTake]:
        return list(self._takes_by_layer.get(layer_id, []))


class _StubTimelineObjectRepository:
    def __init__(self, records: dict[str, TimelineObjectRecord]) -> None:
        self._records = records

    def get(self, object_id: str) -> TimelineObjectRecord | None:
        return self._records.get(object_id)


class _StubObjectContentRepository:
    def __init__(self, records: dict[str, list[ObjectContentRecord]]) -> None:
        self._records = records

    def list_by_object(self, object_id: str) -> list[ObjectContentRecord]:
        return list(self._records.get(object_id, []))


class _StubProjectStorage:
    def __init__(self, takes_by_layer: dict[str, list[PersistedTake]]) -> None:
        self.working_dir = Path.cwd()
        self.takes = _StubTakeRepository(takes_by_layer)
        timeline_objects: dict[str, TimelineObjectRecord] = {}
        object_contents: dict[str, list[ObjectContentRecord]] = {}
        for layer_id, takes in takes_by_layer.items():
            object_id = object_id_for_layer(layer_id)
            main_take = next((take for take in takes if take.is_main), takes[0])
            timeline_objects[object_id] = TimelineObjectRecord(
                id=object_id,
                song_version_id="song_version_runtime",
                name="drums",
                object_kind="event_layer",
                main_content_id=content_id_for_take(main_take.id),
                created_at=main_take.created_at,
            )
            object_contents[object_id] = [
                ObjectContentRecord(
                    id=content_id_for_take(take.id),
                    object_id=object_id,
                    revision_id=revision_id_for_take(take.id),
                    content_kind="event_layer",
                    payload={"take_id": take.id},
                    source_ref=None,
                    analysis_build=None,
                    created_at=take.created_at,
                )
                for take in takes
            ]
        self.timeline_objects = _StubTimelineObjectRepository(timeline_objects)
        self.object_contents = _StubObjectContentRepository(object_contents)


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


def test_build_storage_layer_restores_layer_output_bus_state_flag():
    takes = _layer_takes()
    storage = _StubProjectStorage({"layer_timecode": takes})

    layer, _, _ = build_storage_layer(
        storage,
        TimelineId("timeline_runtime"),
        _layer_record(
            layer_id="layer_timecode",
            state_flags={"output_bus": "outputs_3_4"},
        ),
    )

    assert layer is not None
    assert layer.mixer.output_bus == "outputs_3_4"


def test_build_storage_layer_restores_layer_mute_and_solo_state_flags():
    takes = _layer_takes()
    storage = _StubProjectStorage({"layer_timecode": takes})

    layer, _, _ = build_storage_layer(
        storage,
        TimelineId("timeline_runtime"),
        _layer_record(
            layer_id="layer_timecode",
            state_flags={"mute": True, "solo": True},
        ),
    )

    assert layer is not None
    assert layer.mixer.mute is True
    assert layer.mixer.solo is True


def test_build_storage_layer_resolves_event_take_playback_source_ref_from_snapshot():
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
    assert layer_audio.source_audio_path is None
    assert layer_audio.playback_source_ref is not None
    assert Path(layer_audio.playback_source_ref).as_posix().endswith("stems/drums.wav")
    take_audio_fields = next(iter(take_audio.values()))
    assert take_audio_fields.source_audio_path is None
    assert take_audio_fields.playback_source_ref is not None
    assert Path(take_audio_fields.playback_source_ref).as_posix().endswith("stems/drums.wav")


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
