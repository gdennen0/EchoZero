from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

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


class MockSeparatorWithTempFiles:
    def __init__(self, stems_root: Path) -> None:
        self._stems_root = stems_root

    def execute(self, block_id, context):
        _ = block_id, context
        self._stems_root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, AudioData] = {}
        for stem_name in ("drums", "bass", "vocals", "other"):
            stem_path = self._stems_root / f"{stem_name}.wav"
            stem_path.write_bytes(b"RIFF0000WAVEfmt ")
            outputs[f"{stem_name}_out"] = AudioData(
                sample_rate=44100,
                duration=180.0,
                file_path=str(stem_path),
                channel_count=2,
            )
        return ok(outputs)


def _create_session(tmp_path):
    session = ProjectStorage.create_new('Test ProjectRecord', working_dir_root=tmp_path)
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
    return session, version


def test_stem_separation_persists_audio_layers_and_takes(tmp_path):
    import echozero.pipelines.templates  # noqa: F401

    session, version = _create_session(tmp_path)
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
    assert sorted(layer.name for layer in layers) == ['bass', 'drums', 'other', 'vocals']

    for layer in layers:
        takes = session.takes.list_by_layer(layer.id)
        assert len(takes) == 1
        assert takes[0].is_main is True
        assert isinstance(takes[0].data, AudioData)

    session.close()


def test_stem_audio_outputs_are_materialized_into_project_audio_storage(tmp_path):
    import echozero.pipelines.templates  # noqa: F401

    session, version = _create_session(tmp_path)
    stems_root = tmp_path / "temp-stems"
    orch = Orchestrator(
        registry=get_registry(),
        executors={
            'LoadAudio': MockLoadAudio(),
            'SeparateAudio': MockSeparatorWithTempFiles(stems_root),
        },
    )

    result = orch.analyze(session, version.id, 'stem_separation')
    assert isinstance(result, Ok)

    for layer in session.layers.list_by_version(version.id):
        take = session.takes.list_by_layer(layer.id)[0]
        assert isinstance(take.data, AudioData)
        assert str(take.data.file_path).startswith("audio/generated/")
        persisted_audio = session.working_dir / str(take.data.file_path)
        assert persisted_audio.exists()
        assert persisted_audio.name in {"drums.wav", "bass.wav", "vocals.wav", "other.wav"}

    session.close()
