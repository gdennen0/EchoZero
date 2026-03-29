"""
SongVersion tests: Verify unified audio import, metadata scanning, config copy (D278),
and default template application on first song add.
Exists because song version management is a critical vertical — audio in, metadata out,
configs copied or created, all through one clean path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from echozero.persistence.audio import AudioMetadata
from echozero.persistence.entities import PipelineConfig, Song, SongVersion
from echozero.persistence.session import ProjectSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_scan(path) -> AudioMetadata:
    """Mock audio scanner that returns realistic metadata without reading real audio."""
    return AudioMetadata(duration_seconds=180.5, sample_rate=44100, channel_count=2)


def _mock_scan_48k(path) -> AudioMetadata:
    """Mock scanner returning 48kHz mono metadata."""
    return AudioMetadata(duration_seconds=240.0, sample_rate=48000, channel_count=1)


def _create_session(tmp_path: Path) -> ProjectSession:
    """Create a fresh ProjectSession in a temp directory."""
    return ProjectSession.create_new(
        name="Test Project",
        working_dir_root=tmp_path / "working",
    )


def _create_audio_file(tmp_path: Path, name: str = "test.wav") -> Path:
    """Create a fake audio file for import."""
    audio = tmp_path / name
    audio.write_bytes(b"RIFF" + b"\x00" * 100)
    return audio


def _make_pipeline_config(
    song_version_id: str,
    template_id: str = "onset_detection",
    name: str = "Onset Detection",
    knob_values: dict[str, Any] | None = None,
    block_overrides: dict[str, list[str]] | None = None,
) -> PipelineConfig:
    """Create a PipelineConfig for testing."""
    now = datetime.now(timezone.utc)
    graph = {
        "blocks": [
            {
                "id": "load1",
                "name": "Load Audio",
                "block_type": "LoadAudio",
                "category": "PROCESSOR",
                "input_ports": [],
                "output_ports": [{"name": "audio_out", "port_type": "AUDIO", "direction": "OUTPUT"}],
                "settings": {"file_path": ""},
            },
            {
                "id": "detect1",
                "name": "Detect Onsets",
                "block_type": "DetectOnsets",
                "category": "PROCESSOR",
                "input_ports": [{"name": "audio_in", "port_type": "AUDIO", "direction": "INPUT"}],
                "output_ports": [{"name": "events_out", "port_type": "EVENT", "direction": "OUTPUT"}],
                "settings": {"threshold": 0.3, "min_gap": 0.05},
            },
        ],
        "connections": [
            {
                "source_block_id": "load1",
                "source_output_name": "audio_out",
                "target_block_id": "detect1",
                "target_input_name": "audio_in",
            }
        ],
    }
    return PipelineConfig(
        id=uuid.uuid4().hex,
        song_version_id=song_version_id,
        template_id=template_id,
        name=name,
        graph_json=json.dumps(graph),
        outputs_json=json.dumps([]),
        knob_values=knob_values or {"threshold": 0.3, "min_gap": 0.05},
        created_at=now,
        updated_at=now,
        block_overrides=block_overrides or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    """Verify that _create_version populates real metadata from scanned audio."""

    def test_version_has_real_duration(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan)

        assert v1.duration_seconds == 180.5
        session.close()

    def test_version_has_real_sample_rate(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan)

        assert v1.original_sample_rate == 44100
        session.close()

    def test_version_metadata_from_48k_file(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan_48k)

        assert v1.original_sample_rate == 48000
        assert v1.duration_seconds == 240.0
        session.close()

    def test_add_version_also_scans_metadata(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test", audio1, scan_fn=_mock_scan)
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan_48k)

        assert v2.duration_seconds == 240.0
        assert v2.original_sample_rate == 48000
        session.close()


class TestDefaultTemplates:
    """Verify that import_song creates default pipeline configs from templates."""

    def test_import_creates_configs_from_registered_templates(self, tmp_path: Path) -> None:
        # Ensure templates are registered
        import echozero.pipelines.templates  # noqa: F401
        from echozero.pipelines.registry import get_registry

        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan)

        configs = session.pipeline_configs.list_by_version(v1.id)
        registered = get_registry().list()

        assert len(configs) == len(registered)
        assert len(configs) > 0  # at least onset_detection exists
        session.close()

    def test_import_with_specific_templates(self, tmp_path: Path) -> None:
        import echozero.pipelines.templates  # noqa: F401

        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song(
            "Test", audio, scan_fn=_mock_scan,
            default_templates=["onset_detection"],
        )

        configs = session.pipeline_configs.list_by_version(v1.id)
        assert len(configs) == 1
        assert configs[0].template_id == "onset_detection"
        session.close()

    def test_import_with_empty_templates_creates_none(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song(
            "Test", audio, scan_fn=_mock_scan,
            default_templates=[],
        )

        configs = session.pipeline_configs.list_by_version(v1.id)
        assert len(configs) == 0
        session.close()

    def test_add_version_copies_configs_not_templates(self, tmp_path: Path) -> None:
        """add_song_version copies from source version, not from templates."""
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        # Import with no default templates
        song, v1 = session.import_song(
            "Test", audio1, scan_fn=_mock_scan, default_templates=[],
        )
        # Manually add one config
        config = _make_pipeline_config(v1.id, "custom", "My Custom Pipeline")
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        # Should have copied the one manual config, not applied templates
        assert len(v2_configs) == 1
        assert v2_configs[0].template_id == "custom"
        session.close()


class TestAddSongVersion:
    """Verify the D278 add_song_version flow."""

    def test_creates_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        v2 = session.add_song_version(song.id, audio2, label="Festival Edit", scan_fn=_mock_scan)

        assert v2.song_id == song.id
        assert v2.label == "Festival Edit"
        assert v2.id != v1.id
        session.close()

    def test_new_version_has_different_audio_file(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = tmp_path / "v2.wav"
        audio2.write_bytes(b"RIFF" + b"\xff" * 100)

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        assert v2.audio_file != v1.audio_file
        assert v2.audio_hash != v1.audio_hash
        session.close()

    def test_activates_new_version_by_default(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        updated_song = session.songs.get(song.id)
        assert updated_song.active_version_id == v2.id
        session.close()

    def test_does_not_activate_when_flag_false(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        v2 = session.add_song_version(song.id, audio2, activate=False, scan_fn=_mock_scan)

        updated_song = session.songs.get(song.id)
        assert updated_song.active_version_id == v1.id
        session.close()

    def test_copies_pipeline_configs(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        config1 = _make_pipeline_config(v1.id, "onset", "Onset Detection")
        config2 = _make_pipeline_config(v1.id, "classify", "Classification")
        session.pipeline_configs.create(config1)
        session.pipeline_configs.create(config2)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)
        assert len(v2_configs) == 2
        session.close()

    def test_copied_configs_have_new_ids(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].id != config.id
        session.close()

    def test_copied_configs_preserve_knob_values(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        config = _make_pipeline_config(v1.id, knob_values={"threshold": 0.7, "min_gap": 0.02})
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].knob_values == {"threshold": 0.7, "min_gap": 0.02}
        session.close()

    def test_copied_configs_preserve_block_overrides(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        config = _make_pipeline_config(v1.id, block_overrides={"detect1": ["threshold"]})
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].block_overrides == {"detect1": ["threshold"]}
        session.close()

    def test_copied_configs_point_to_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].song_version_id == v2.id
        session.close()

    def test_original_configs_unchanged(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v1_configs = session.pipeline_configs.list_by_version(v1.id)

        assert len(v1_configs) == 1
        assert v1_configs[0].id == config.id
        session.close()

    def test_auto_label_when_none_provided(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        assert v2.label == "v2"
        session.close()

    def test_version_listed_in_song_versions(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1, scan_fn=_mock_scan, default_templates=[])
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        versions = session.song_versions.list_by_song(song.id)
        assert len(versions) == 2
        session.close()

    def test_nonexistent_song_raises(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)

        with pytest.raises(ValueError, match="Song not found"):
            session.add_song_version("ghost_id", audio, scan_fn=_mock_scan)
        session.close()

    def test_song_without_active_version_raises(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)

        from dataclasses import replace
        song, v1 = session.import_song("Test Song", audio, scan_fn=_mock_scan, default_templates=[])
        session.songs.update(replace(song, active_version_id=None))
        session.commit()

        audio2 = _create_audio_file(tmp_path, "v2.wav")
        with pytest.raises(ValueError, match="no active version"):
            session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        session.close()
