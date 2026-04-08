from __future__ import annotations

from pathlib import Path

from echozero.persistence.session import ProjectStorage


def test_add_song_version_copies_configs_and_records_blank_slate_rebuild_plan(tmp_path):
    session = ProjectStorage.create_new('Test', working_dir_root=tmp_path / 'working')

    audio1 = tmp_path / 'song1.wav'
    audio1.write_bytes(b'fake-audio-1')
    audio2 = tmp_path / 'song2.wav'
    audio2.write_bytes(b'fake-audio-2')

    class Meta:
        def __init__(self, duration_seconds=120.0, sample_rate=44100):
            self.duration_seconds = duration_seconds
            self.sample_rate = sample_rate

    def fake_scan(_path: Path, scan_fn=None):
        return Meta()

    song, version1 = session.import_song(
        title='Song',
        audio_source=audio1,
        default_templates=['stem_separation'],
        scan_fn=fake_scan,
    )

    version2 = session.add_song_version(
        song_id=song.id,
        audio_source=audio2,
        label='Festival Edit',
        activate=True,
        scan_fn=fake_scan,
    )

    cfgs_v1 = session.pipeline_configs.list_by_version(version1.id)
    cfgs_v2 = session.pipeline_configs.list_by_version(version2.id)
    assert len(cfgs_v1) == len(cfgs_v2)

    plan = getattr(version2, 'rebuild_plan', None)
    assert plan is not None
    assert plan['mode'] == 'blank_slate_with_rerun'
    assert plan['previous_version_id'] == version1.id
    assert plan['new_version_id'] == version2.id

    # Must round-trip through persistence, not only in-memory return value.
    persisted = session.song_versions.get(version2.id)
    assert persisted is not None
    assert persisted.rebuild_plan['mode'] == 'blank_slate_with_rerun'
    assert persisted.rebuild_plan['previous_version_id'] == version1.id
    assert persisted.rebuild_plan['new_version_id'] == version2.id

    session.close()
