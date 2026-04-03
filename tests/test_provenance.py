from __future__ import annotations

from datetime import datetime, timezone

from echozero.persistence.entities import LayerRecord
from echozero.services.provenance import (
    build_song_version_rebuild_plan,
    clear_layer_stale,
    initialize_generated_layer_state,
    mark_layer_manually_modified,
    mark_layer_stale,
)


def _layer() -> LayerRecord:
    return LayerRecord(
        id='layer1',
        song_version_id='version1',
        name='Drums',
        layer_type='analysis',
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline=None,
        created_at=datetime.now(timezone.utc),
    )


def test_initialize_generated_layer_state_sets_expected_flags_and_provenance():
    layer = initialize_generated_layer_state(
        _layer(),
        pipeline_id='stem_separation',
        output_name='drums',
        block_id='separate',
        data_type='audio',
        source_song_version_id='version1',
        source_layer_id='main-song',
        source_run_id='run123',
    )

    assert layer.state_flags['derived'] is True
    assert layer.state_flags['stale'] is False
    assert layer.state_flags['manually_modified'] is False
    assert layer.provenance['output_name'] == 'drums'
    assert layer.provenance['source_layer_id'] == 'main-song'


def test_mark_stale_and_clear_stale_round_trip():
    layer = initialize_generated_layer_state(
        _layer(),
        pipeline_id='stem_separation',
        output_name='drums',
        block_id='separate',
        data_type='audio',
        source_song_version_id='version1',
    )
    stale = mark_layer_stale(layer, reason='Upstream main changed', upstream_layer_id='main-song')
    assert stale.state_flags['stale'] is True
    assert stale.state_flags['source_main_changed'] is True
    assert stale.state_flags['stale_upstream_layer_id'] == 'main-song'

    fresh = clear_layer_stale(stale)
    assert fresh.state_flags['stale'] is False
    assert fresh.state_flags['source_main_changed'] is False


def test_mark_layer_manually_modified_sets_flag():
    layer = initialize_generated_layer_state(
        _layer(),
        pipeline_id='classify_drums',
        output_name='kick',
        block_id='classify',
        data_type='event',
        source_song_version_id='version1',
    )
    modified = mark_layer_manually_modified(layer)
    assert modified.state_flags['manually_modified'] is True


def test_song_version_rebuild_plan_is_blank_slate_with_rerun():
    plan = build_song_version_rebuild_plan(
        previous_version_id='v1',
        new_version_id='v2',
        pipeline_config_ids=['cfg1', 'cfg2'],
    )
    assert plan['mode'] == 'blank_slate_with_rerun'
    assert plan['pipeline_config_ids'] == ['cfg1', 'cfg2']
    assert plan['remap']['status'] == 'deferred'
