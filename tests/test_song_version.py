"""
SongVersion config-copy tests: Verify D278 "Update Track" flow.
Exists because song updates must copy all pipeline configs to the new version
while preserving knob values, graph structure, and block overrides.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from echozero.persistence.entities import PipelineConfig, Song, SongVersion
from echozero.persistence.session import ProjectSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class TestAddSongVersion:
    """Verify the D278 add_song_version flow."""

    def test_creates_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2, label="Festival Edit")

        assert v2.song_id == song.id
        assert v2.label == "Festival Edit"
        assert v2.id != v1.id
        session.close()

    def test_new_version_has_different_audio_file(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = tmp_path / "v2.wav"
        audio2.write_bytes(b"RIFF" + b"\xff" * 100)  # Different content

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2)

        assert v2.audio_file != v1.audio_file
        assert v2.audio_hash != v1.audio_hash
        session.close()

    def test_activates_new_version_by_default(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2)

        updated_song = session.songs.get(song.id)
        assert updated_song.active_version_id == v2.id
        session.close()

    def test_does_not_activate_when_flag_false(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2, activate=False)

        updated_song = session.songs.get(song.id)
        assert updated_song.active_version_id == v1.id
        session.close()

    def test_copies_pipeline_configs(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)

        # Add pipeline configs to v1
        config1 = _make_pipeline_config(v1.id, "onset", "Onset Detection")
        config2 = _make_pipeline_config(v1.id, "classify", "Classification")
        session.pipeline_configs.create(config1)
        session.pipeline_configs.create(config2)
        session.commit()

        # Add new version
        v2 = session.add_song_version(song.id, audio2)

        # Check configs were copied
        v2_configs = session.pipeline_configs.list_by_version(v2.id)
        assert len(v2_configs) == 2
        session.close()

    def test_copied_configs_have_new_ids(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert len(v2_configs) == 1
        assert v2_configs[0].id != config.id
        session.close()

    def test_copied_configs_preserve_knob_values(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        config = _make_pipeline_config(
            v1.id,
            knob_values={"threshold": 0.7, "min_gap": 0.02},
        )
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].knob_values == {"threshold": 0.7, "min_gap": 0.02}
        session.close()

    def test_copied_configs_preserve_graph_json(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].graph_json == config.graph_json
        session.close()

    def test_copied_configs_preserve_block_overrides(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        config = _make_pipeline_config(
            v1.id,
            block_overrides={"detect1": ["threshold"]},
        )
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].block_overrides == {"detect1": ["threshold"]}
        session.close()

    def test_copied_configs_point_to_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].song_version_id == v2.id
        session.close()

    def test_original_configs_unchanged(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        config = _make_pipeline_config(v1.id)
        session.pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2)

        # v1 configs should be untouched
        v1_configs = session.pipeline_configs.list_by_version(v1.id)
        assert len(v1_configs) == 1
        assert v1_configs[0].id == config.id
        assert v1_configs[0].song_version_id == v1.id
        session.close()

    def test_no_configs_to_copy_produces_empty(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2)

        v2_configs = session.pipeline_configs.list_by_version(v2.id)
        assert len(v2_configs) == 0
        session.close()

    def test_auto_label_when_none_provided(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2)

        assert v2.label == "v2"
        session.close()

    def test_version_listed_in_song_versions(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test Song", audio1)
        v2 = session.add_song_version(song.id, audio2)

        versions = session.song_versions.list_by_song(song.id)
        version_ids = [v.id for v in versions]
        assert v1.id in version_ids
        assert v2.id in version_ids
        assert len(versions) == 2
        session.close()

    def test_nonexistent_song_raises(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)

        with pytest.raises(ValueError, match="Song not found"):
            session.add_song_version("ghost_id", audio)
        session.close()

    def test_song_without_active_version_raises(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)

        # Create song manually with no active version
        from dataclasses import replace
        song, v1 = session.import_song("Test Song", audio)
        updated_song = replace(song, active_version_id=None)
        session.songs.update(updated_song)
        session.commit()

        audio2 = _create_audio_file(tmp_path, "v2.wav")
        with pytest.raises(ValueError, match="no active version"):
            session.add_song_version(song.id, audio2)
        session.close()
