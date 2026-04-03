from __future__ import annotations

from datetime import datetime, timezone
import uuid

import echozero.pipelines.templates  # noqa: F401
from echozero.domain.types import AudioData
from echozero.persistence.entities import SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.result import Ok, ok
from echozero.services.orchestrator import Orchestrator


class MockLoadAudio:
    def execute(self, block_id, context):
        return ok(AudioData(sample_rate=44100, duration=180.0, file_path='mix.wav', channel_count=2))


class MockSeparator:
    def execute(self, block_id, context):
        return ok({
            'drums_out': AudioData(sample_rate=44100, duration=180.0, file_path='drums.wav', channel_count=2),
            'bass_out': AudioData(sample_rate=44100, duration=180.0, file_path='bass.wav', channel_count=2),
            'vocals_out': AudioData(sample_rate=44100, duration=180.0, file_path='vocals.wav', channel_count=2),
            'other_out': AudioData(sample_rate=44100, duration=180.0, file_path='other.wav', channel_count=2),
        })


def test_generated_layers_include_initialized_provenance_and_state(tmp_path):
    session = ProjectStorage.create_new('Test', working_dir_root=tmp_path)
    now = datetime.now(timezone.utc)
    song = SongRecord(id=uuid.uuid4().hex, project_id=session.project.id, title='Song', artist='Artist', order=0)
    session.songs.create(song)
    version = SongVersionRecord(
        id=uuid.uuid4().hex,
        song_id=song.id,
        label='Studio Mix',
        audio_file='mix.wav',
        duration_seconds=180.0,
        original_sample_rate=44100,
        audio_hash='abc123',
        created_at=now,
    )
    session.song_versions.create(version)
    session.commit()

    orch = Orchestrator(
        registry=get_registry(),
        executors={
            'LoadAudio': MockLoadAudio(),
            'SeparateAudio': MockSeparator(),
        },
    )

    result = orch.analyze(session, version.id, 'stem_separation')
    assert isinstance(result, Ok)

    layers = session.layers.list_by_version(version.id)
    drums = next(layer for layer in layers if layer.name == 'drums')
    assert drums.state_flags['derived'] is True
    assert drums.state_flags['stale'] is False
    assert drums.state_flags['manually_modified'] is False
    assert drums.provenance['output_name'] == 'drums'
    assert drums.provenance['source_song_version_id'] == version.id

    session.close()
