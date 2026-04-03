from __future__ import annotations

from datetime import datetime, timezone
import uuid

from echozero.domain.types import AudioData
from echozero.persistence.entities import LayerRecord, SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.services.take_actions import promote_take_to_main
from echozero.takes import Take, TakeSource


def _take(take_id: str, *, run_id: str, is_main: bool) -> Take:
    return Take(
        id=take_id,
        label=take_id,
        data=AudioData(sample_rate=44100, duration=5.0, file_path='x.wav', channel_count=2),
        origin='pipeline',
        source=TakeSource(block_id='b1', block_type='SeparateAudio', settings_snapshot={}, run_id=run_id),
        created_at=datetime.now(timezone.utc),
        is_main=is_main,
    )


def test_promote_take_to_main_marks_dependents_stale(tmp_path):
    session = ProjectStorage.create_new('Test', working_dir_root=tmp_path)
    now = datetime.now(timezone.utc)
    song = SongRecord(id=uuid.uuid4().hex, project_id=session.project.id, title='Song', artist='Artist', order=0)
    session.songs.create(song)
    version = SongVersionRecord(
        id=uuid.uuid4().hex,
        song_id=song.id,
        label='Original',
        audio_file='song.wav',
        duration_seconds=120.0,
        original_sample_rate=44100,
        audio_hash='abc',
        created_at=now,
    )
    session.song_versions.create(version)

    parent = LayerRecord(
        id='drums',
        song_version_id=version.id,
        name='drums',
        layer_type='analysis',
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'stem_separation'},
        created_at=now,
        provenance={'current_main_take_id': 'take1'},
    )
    child = LayerRecord(
        id='kick',
        song_version_id=version.id,
        name='kick',
        layer_type='analysis',
        color=None,
        order=1,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'classify_drums'},
        created_at=now,
        provenance={'source_layer_id': 'drums'},
    )
    session.layers.create(parent)
    session.layers.create(child)
    session.takes.create('drums', _take('take1', run_id='run1', is_main=True))
    session.takes.create('drums', _take('take2', run_id='run2', is_main=False))
    session.commit()

    updated_parent, updated_layers = promote_take_to_main(session, layer_id='drums', take_id='take2')

    assert updated_parent.provenance['current_main_take_id'] == 'take2'
    kick = next(layer for layer in updated_layers if layer.id == 'kick')
    assert kick.state_flags['stale'] is True
    assert kick.state_flags['source_main_changed'] is True

    session.close()


def test_promote_same_main_take_is_noop_for_stale(tmp_path):
    session = ProjectStorage.create_new('Test', working_dir_root=tmp_path)
    now = datetime.now(timezone.utc)
    song = SongRecord(id=uuid.uuid4().hex, project_id=session.project.id, title='Song', artist='Artist', order=0)
    session.songs.create(song)
    version = SongVersionRecord(
        id=uuid.uuid4().hex,
        song_id=song.id,
        label='Original',
        audio_file='song.wav',
        duration_seconds=120.0,
        original_sample_rate=44100,
        audio_hash='abc',
        created_at=now,
    )
    session.song_versions.create(version)

    parent = LayerRecord(
        id='drums',
        song_version_id=version.id,
        name='drums',
        layer_type='analysis',
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'stem_separation'},
        created_at=now,
        provenance={'current_main_take_id': 'take1'},
    )
    child = LayerRecord(
        id='kick',
        song_version_id=version.id,
        name='kick',
        layer_type='analysis',
        color=None,
        order=1,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'classify_drums'},
        created_at=now,
        provenance={'source_layer_id': 'drums'},
    )
    session.layers.create(parent)
    session.layers.create(child)
    session.takes.create('drums', _take('take1', run_id='run1', is_main=True))
    session.commit()

    _, updated_layers = promote_take_to_main(session, layer_id='drums', take_id='take1')

    kick = next(layer for layer in updated_layers if layer.id == 'kick')
    assert kick.state_flags == {}

    session.close()
