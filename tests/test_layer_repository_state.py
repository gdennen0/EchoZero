from __future__ import annotations

from datetime import datetime, timezone

from echozero.persistence.entities import LayerRecord
from echozero.persistence.repositories.layer import LayerRepository
from echozero.persistence.session import ProjectStorage


def test_layer_repository_round_trips_state_flags_and_provenance(tmp_path):
    session = ProjectStorage.create_new('Test', working_dir_root=tmp_path)

    song = session.songs.list_by_project(session.project.id)
    if not song:
        from echozero.persistence.entities import SongRecord, SongVersionRecord
        import uuid
        now = datetime.now(timezone.utc)
        song_record = SongRecord(id=uuid.uuid4().hex, project_id=session.project.id, title='Song', artist='', order=0)
        session.songs.create(song_record)
        version = SongVersionRecord(
            id=uuid.uuid4().hex,
            song_id=song_record.id,
            label='Original',
            audio_file='song.wav',
            duration_seconds=120.0,
            original_sample_rate=44100,
            audio_hash='abc',
            created_at=now,
        )
        session.song_versions.create(version)
        session.commit()
    else:
        version = session.song_versions.list_by_song(song[0].id)[0]

    layer = LayerRecord(
        id='layer-1',
        song_version_id=version.id,
        name='Drums',
        layer_type='analysis',
        color=None,
        order=0,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={'pipeline_id': 'stem_separation'},
        created_at=datetime.now(timezone.utc),
        state_flags={'derived': True, 'stale': False, 'manually_modified': True},
        provenance={'output_name': 'drums', 'source_song_version_id': version.id},
    )

    repo = LayerRepository(session.db)
    repo.create(layer)
    session.commit()

    loaded = repo.get('layer-1')
    assert loaded is not None
    assert loaded.state_flags['manually_modified'] is True
    assert loaded.provenance['output_name'] == 'drums'

    session.close()
