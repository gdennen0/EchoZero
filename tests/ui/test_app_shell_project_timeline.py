"""Project timeline baseline tests for version-scoped musical timing fields.
Exists to lock active-version overlay behavior when the timeline is rebuilt from storage.
Connects persistence song-version tempo truth to canonical app-shell presentation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from echozero.persistence.audio import AudioMetadata
from echozero.persistence.entities import ProjectSettingsRecord
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline import build_project_native_baseline_timeline


def test_build_project_native_baseline_timeline_prefers_active_version_tempo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = ProjectStorage.create_new(
        name="Tempo Project",
        settings=ProjectSettingsRecord(bpm=90.0, bpm_confidence=0.25),
        working_dir_root=tmp_path / "working",
    )
    audio = tmp_path / "tempo-song.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 128)

    song, version = session.import_song(
        "Tempo Song",
        audio,
        scan_fn=lambda _path: AudioMetadata(
            duration_seconds=180.0,
            sample_rate=44100,
            channel_count=2,
        ),
    )
    session.song_versions.update(
        replace(
            version,
            bpm=128.0,
            bpm_confidence=0.91,
            beat_anchor_seconds=0.42,
        )
    )
    session.commit()
    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_timeline.ensure_registered_waveform",
        lambda key, _audio_path: key,
    )

    _timeline, overlay, active_song_id, active_version_id = build_project_native_baseline_timeline(
        session
    )

    assert str(active_song_id) == song.id
    assert str(active_version_id) == version.id
    assert overlay.bpm == 128.0
    assert overlay.bpm_confidence == 0.91
    assert overlay.beat_anchor_seconds == 0.42
    session.close()
