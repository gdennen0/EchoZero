from __future__ import annotations

from datetime import datetime, timezone

from echozero.domain.types import AudioData
from echozero.persistence.entities import LayerRecord
from echozero.services.dependencies import (
    capture_main_lineage,
    mark_dependents_stale_on_upstream_main_change,
    upstream_main_change_requires_stale,
)
from echozero.takes import Take, TakeSource


def _layer(layer_id: str, *, source_layer_id: str | None = None) -> LayerRecord:
    return LayerRecord(
        id=layer_id,
        song_version_id='version1',
        name=layer_id,
        layer_type='analysis',
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'stem_separation'},
        created_at=datetime.now(timezone.utc),
        provenance={'source_layer_id': source_layer_id} if source_layer_id else {},
    )


def _take(take_id: str, *, run_id: str) -> Take:
    return Take(
        id=take_id,
        label=take_id,
        data=AudioData(sample_rate=44100, duration=1.0, file_path='x.wav', channel_count=2),
        origin='pipeline',
        source=TakeSource(block_id='b1', block_type='SeparateAudio', settings_snapshot={}, run_id=run_id),
        created_at=datetime.now(timezone.utc),
        is_main=True,
    )


def test_capture_main_lineage_records_take_and_run_ids():
    layer = _layer('drums')
    take = _take('take1', run_id='run1')
    updated = capture_main_lineage(layer, take)
    assert updated.provenance['current_main_take_id'] == 'take1'
    assert updated.provenance['current_main_run_id'] == 'run1'


def test_non_main_take_existence_does_not_imply_stale_without_main_change():
    main_take = _take('take1', run_id='run1')
    assert upstream_main_change_requires_stale(previous_main_take=main_take, new_main_take=main_take) is False


def test_dependents_mark_stale_only_when_upstream_main_changes():
    upstream_old = _take('take1', run_id='run1')
    upstream_new = _take('take2', run_id='run2')
    parent = _layer('drums')
    child = _layer('kick', source_layer_id='drums')
    unrelated = _layer('vox', source_layer_id='vocals')

    updated = mark_dependents_stale_on_upstream_main_change(
        layers=[parent, child, unrelated],
        upstream_layer_id='drums',
        previous_main_take=upstream_old,
        new_main_take=upstream_new,
    )

    kick = next(layer for layer in updated if layer.id == 'kick')
    vox = next(layer for layer in updated if layer.id == 'vox')

    assert kick.state_flags['stale'] is True
    assert kick.state_flags['source_main_changed'] is True
    assert vox.state_flags == {}
