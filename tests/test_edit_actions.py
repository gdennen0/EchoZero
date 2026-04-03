from __future__ import annotations

from datetime import datetime, timezone
import uuid

from echozero.persistence.entities import LayerRecord, SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.services.edit_actions import mark_layer_as_manually_modified


def test_mark_layer_as_manually_modified_updates_persisted_flags(tmp_path):
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
    layer = LayerRecord(
        id='kick',
        song_version_id=version.id,
        name='kick',
        layer_type='analysis',
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'classify_drums'},
        created_at=now,
    )
    session.layers.create(layer)
    session.commit()

    updated = mark_layer_as_manually_modified(session, layer_id='kick')

    assert updated.state_flags['manually_modified'] is True
    reloaded = session.layers.get('kick')
    assert reloaded is not None
    assert reloaded.state_flags['manually_modified'] is True
    session.close()
