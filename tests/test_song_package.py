"""
Song package tests for .ezsong import/export.
Exercises package integrity, media copy, and ID remapping through ProjectStorage.
"""

from __future__ import annotations

import json
import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from echozero.domain.types import AudioData, EventData, Layer
from echozero.errors import PersistenceError
from echozero.persistence.audio import AudioMetadata
from echozero.persistence.entities import (
    LayerRecord,
    ObjectCandidateRecord,
    ObjectContentRecord,
    PipelineConfigRecord,
    SongVideoAttachmentRecord,
    SongVideoPlacementRecord,
    TimelineObjectRecord,
)
from echozero.persistence.session import ProjectStorage
from echozero.persistence.song_package import (
    export_song_package,
    import_song_package,
    inspect_song_package,
)
from echozero.takes import Take
from echozero.ui.qt.app_shell_project_timeline import build_project_native_baseline_timeline


def _mock_scan(_path: Path) -> AudioMetadata:
    return AudioMetadata(duration_seconds=120.0, sample_rate=44100, channel_count=2)


def _write_source_audio(tmp_path: Path, name: str = "song.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"RIFF" + bytes(name, "utf-8") + (b"\0" * 64))
    return path


def _create_storage(tmp_path: Path, name: str) -> ProjectStorage:
    return ProjectStorage.create_new(name, working_dir_root=tmp_path / "working")


def _add_package_fixture(storage: ProjectStorage, tmp_path: Path):
    song, version = storage.import_song(
        "Package Song",
        _write_source_audio(tmp_path),
        artist="Tester",
        default_templates=[],
        scan_fn=_mock_scan,
    )
    now = datetime.now(timezone.utc)
    take_audio_rel = "audio/take_ref.wav"
    take_audio_path = storage.working_dir / take_audio_rel
    take_audio_path.parent.mkdir(parents=True, exist_ok=True)
    take_audio_path.write_bytes(b"take audio")

    layer = LayerRecord(
        id="layer_events",
        song_version_id=version.id,
        name="Events",
        layer_type="manual",
        color=None,
        order=1,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline=None,
        created_at=now,
        state_flags={},
        provenance={},
    )
    storage.layers.create(layer)
    storage.takes.create(
        layer.id,
        Take(
            id="take_events",
            label="Main",
            data=EventData(layers=(Layer(id="domain", name="Events", events=()),)),
            origin="user",
            source=None,
            created_at=now,
            is_main=True,
            is_archived=False,
            notes="",
        ),
    )
    storage.takes.create(
        layer.id,
        Take(
            id="take_audio",
            label="Audio Ref",
            data=AudioData(
                sample_rate=44100,
                duration=10.0,
                file_path=take_audio_rel,
                channel_count=1,
            ),
            origin="user",
            source=None,
            created_at=now,
            is_main=False,
            is_archived=False,
            notes="",
        ),
    )
    storage.pipeline_configs.create(
        PipelineConfigRecord(
            id="config_version",
            song_version_id=version.id,
            template_id="template",
            name="Template",
            graph_json="{}",
            outputs_json="[]",
            knob_values={},
            created_at=now,
            updated_at=now,
            block_overrides={},
        )
    )
    storage.timeline_objects.create(
        TimelineObjectRecord(
            id="object_a",
            song_version_id=version.id,
            name="A",
            object_kind="event_set",
            main_content_id="content_a",
            created_at=now,
        )
    )
    storage.object_contents.create(
        ObjectContentRecord(
            id="content_a",
            object_id="object_a",
            revision_id="revision_a",
            content_kind="event_set",
            payload={},
            source_ref=None,
            analysis_build=None,
            created_at=now,
        )
    )
    storage.timeline_objects.create(
        TimelineObjectRecord(
            id="object_b",
            song_version_id=version.id,
            name="B",
            object_kind="event_set",
            main_content_id="content_b",
            created_at=now,
        )
    )
    storage.object_contents.create(
        ObjectContentRecord(
            id="content_b",
            object_id="object_b",
            revision_id="revision_b",
            content_kind="event_set",
            payload={"audio_file": take_audio_rel},
            source_ref={
                "object_id": "object_a",
                "content_id": "content_a",
                "revision_id": "revision_a",
            },
            analysis_build=None,
            created_at=now,
        )
    )
    storage.object_candidates.create(
        ObjectCandidateRecord(
            id="candidate_b",
            object_id="object_b",
            content_id="content_b",
            label="Candidate",
            created_at=now,
        )
    )
    storage.db.commit()
    return song, version, take_audio_rel


def _rewrite_package(
    source_path: Path,
    dest_path: Path,
    *,
    manifest_update=None,
    payload_update=None,
    drop_members: set[str] | None = None,
    add_members: dict[str, bytes] | None = None,
) -> None:
    drop_members = drop_members or set()
    add_members = add_members or {}
    with zipfile.ZipFile(source_path) as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if name not in drop_members
        }
    manifest = json.loads(members["manifest.json"])
    payload = json.loads(members["payload/song.json"])
    if manifest_update is not None:
        manifest_update(manifest)
    if payload_update is not None:
        payload_update(payload)
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    members["payload/song.json"] = json.dumps(payload).encode("utf-8")
    members.update(add_members)
    with zipfile.ZipFile(dest_path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _row_count(storage: ProjectStorage, table: str) -> int:
    return int(storage.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_export_song_package_contains_manifest_payload_and_referenced_media(tmp_path: Path) -> None:
    storage = _create_storage(tmp_path, "Export")
    try:
        _song, version, take_audio_rel = _add_package_fixture(storage, tmp_path)
        package_path = tmp_path / "song.ezsong"

        manifest = export_song_package(storage, version.id, package_path)

        assert manifest.title == "Package Song"
        with zipfile.ZipFile(package_path) as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names
            assert "payload/song.json" in names
            assert f"media/{version.audio_file}" in names
            assert f"media/{take_audio_rel}" in names
    finally:
        storage.close()


def test_import_song_package_into_empty_project_recreates_song_version_and_media(
    tmp_path: Path,
) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)

        result = import_song_package(target, package_path, target_song_id=None)

        assert result.created_song is True
        songs = target.songs.list_by_project(target.project.id)
        assert [song.title for song in songs] == ["Package Song"]
        imported_version = target.song_versions.get(result.song_version_id)
        assert imported_version is not None
        assert (target.working_dir / imported_version.audio_file).exists()
        assert (target.working_dir / take_audio_rel).exists()
        assert target.timeline_objects.get(f"object_song_{result.song_version_id}") is not None
        assert len(target.layers.list_by_version(result.song_version_id)) == 1
    finally:
        source.close()
        target.close()


def test_import_song_package_into_existing_song_adds_inactive_version_by_default(
    tmp_path: Path,
) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)
        target_song, target_version = target.import_song(
            "Local Song",
            _write_source_audio(tmp_path, "local.wav"),
            default_templates=[],
            scan_fn=_mock_scan,
        )

        result = import_song_package(
            target,
            package_path,
            target_song_id=target_song.id,
            activate_import=False,
        )

        persisted_song = target.songs.get(target_song.id)
        assert persisted_song is not None
        assert persisted_song.active_version_id == target_version.id
        versions = target.song_versions.list_by_song(target_song.id)
        assert [item.label for item in versions] == ["Original", "Original (2)"]
        assert result.song_version_id != target_version.id
    finally:
        source.close()
        target.close()


def test_import_song_package_remaps_object_source_refs_and_candidates(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)

        result = import_song_package(target, package_path)

        objects = target.timeline_objects.list_by_version(result.song_version_id)
        object_ids = {record.id for record in objects}
        assert "object_a" not in object_ids
        imported_object_b = next(record for record in objects if record.name == "B")
        imported_content = target.object_contents.get(imported_object_b.main_content_id)
        assert imported_content is not None
        assert imported_content.source_ref is not None
        assert imported_content.source_ref["object_id"] != "object_a"
        assert target.timeline_objects.get(imported_content.source_ref["object_id"]) is not None
        candidates = target.object_candidates.list_by_object(imported_object_b.id)
        assert len(candidates) == 1
        assert candidates[0].content_id == imported_content.id
    finally:
        source.close()
        target.close()


def test_export_includes_cross_version_source_ref_targets(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        _other_song, other_version = source.import_song(
            "Source Audio",
            _write_source_audio(tmp_path, "source_ref.wav"),
            default_templates=[],
            scan_fn=_mock_scan,
        )
        now = datetime.now(timezone.utc)
        source_audio_rel = "audio/source_ref_payload.wav"
        source_audio_path = source.working_dir / source_audio_rel
        source_audio_path.parent.mkdir(parents=True, exist_ok=True)
        source_audio_path.write_bytes(b"source ref audio")
        source.timeline_objects.create(
            TimelineObjectRecord(
                id="object_external_audio",
                song_version_id=other_version.id,
                name="External Audio",
                object_kind="audio_clip",
                main_content_id="content_external_audio",
                created_at=now,
            )
        )
        source.object_contents.create(
            ObjectContentRecord(
                id="content_external_audio",
                object_id="object_external_audio",
                revision_id="revision_external_audio",
                content_kind="audio_clip",
                payload={"audio_file": source_audio_rel},
                source_ref=None,
                analysis_build=None,
                created_at=now,
            )
        )
        source.timeline_objects.create(
            TimelineObjectRecord(
                id="object_cross_ref",
                song_version_id=version.id,
                name="Cross Ref",
                object_kind="event_set",
                main_content_id="content_cross_ref",
                created_at=now,
            )
        )
        source.object_contents.create(
            ObjectContentRecord(
                id="content_cross_ref",
                object_id="object_cross_ref",
                revision_id="revision_cross_ref",
                content_kind="event_set",
                payload={},
                source_ref={
                    "object_id": "object_external_audio",
                    "content_id": "content_external_audio",
                    "revision_id": "revision_external_audio",
                },
                analysis_build=None,
                created_at=now,
            )
        )
        source.db.commit()
        package_path = tmp_path / "cross_ref.ezsong"

        export_song_package(source, version.id, package_path)
        result = import_song_package(target, package_path)

        imported_objects = target.timeline_objects.list_by_version(result.song_version_id)
        imported_external = next(
            record for record in imported_objects if record.name == "External Audio"
        )
        imported_cross = next(record for record in imported_objects if record.name == "Cross Ref")
        imported_cross_content = target.object_contents.get(imported_cross.main_content_id)
        assert imported_cross_content is not None
        assert imported_cross_content.source_ref is not None
        assert imported_cross_content.source_ref["object_id"] == imported_external.id
        assert (
            target.object_contents.get(imported_cross_content.source_ref["content_id"])
            is not None
        )
    finally:
        source.close()
        target.close()


def test_inspect_song_package_rejects_newer_format_version(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)
        with zipfile.ZipFile(package_path, "a") as archive:
            manifest = json.loads(archive.read("manifest.json"))
            manifest["format_version"] = 999
            archive.writestr("manifest.json", json.dumps(manifest))

        with pytest.raises(PersistenceError, match="newer than supported"):
            inspect_song_package(package_path)
    finally:
        source.close()


def test_import_rejects_missing_manifest_media_entry_before_mutation(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        broken_path = tmp_path / "broken.ezsong"
        export_song_package(source, version.id, package_path)

        def drop_take_audio(manifest: dict) -> None:
            manifest["media"] = [
                entry
                for entry in manifest["media"]
                if entry["rel_path"] != "audio/take_ref.wav"
            ]

        _rewrite_package(package_path, broken_path, manifest_update=drop_take_audio)

        with pytest.raises(
            PersistenceError,
            match="Unsupported song package member|missing media entries",
        ):
            import_song_package(target, broken_path)
        assert _row_count(target, "songs") == 0
        assert not (target.working_dir / "audio" / "take_ref.wav").exists()
    finally:
        source.close()
        target.close()


def test_import_rejects_path_traversal_and_unsupported_members(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        broken_path = tmp_path / "evil.ezsong"
        export_song_package(source, version.id, package_path)
        _rewrite_package(
            package_path,
            broken_path,
            add_members={"media/../escape.wav": b"bad"},
        )

        with pytest.raises(PersistenceError, match="Invalid package path"):
            import_song_package(target, broken_path)
        assert _row_count(target, "songs") == 0
    finally:
        source.close()
        target.close()


def test_import_rejects_hash_mismatch_before_mutation(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        broken_path = tmp_path / "hash.ezsong"
        export_song_package(source, version.id, package_path)
        _rewrite_package(
            package_path,
            broken_path,
            drop_members={"media/audio/take_ref.wav"},
            add_members={"media/audio/take_ref.wav": b"wrong"},
        )

        with pytest.raises(PersistenceError, match="size mismatch|hash mismatch"):
            import_song_package(target, broken_path)
        assert _row_count(target, "songs") == 0
    finally:
        source.close()
        target.close()


def test_import_rejects_dangling_source_ref_before_mutation(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        broken_path = tmp_path / "dangling.ezsong"
        export_song_package(source, version.id, package_path)

        def break_source_ref(payload: dict) -> None:
            for row in payload["object_contents"]:
                if row["id"] == "content_b":
                    row["source_ref_json"] = json.dumps(
                        {"object_id": "missing", "content_id": "missing"}
                    )

        _rewrite_package(package_path, broken_path, payload_update=break_source_ref)

        with pytest.raises(PersistenceError, match="dangling source_ref"):
            import_song_package(target, broken_path)
        assert _row_count(target, "songs") == 0
    finally:
        source.close()
        target.close()


def test_failed_import_rolls_back_rows_and_promoted_media(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        broken_path = tmp_path / "invalid_take.ezsong"
        export_song_package(source, version.id, package_path)

        def break_take_origin(payload: dict) -> None:
            payload["takes"][0]["origin"] = "package"

        _rewrite_package(package_path, broken_path, payload_update=break_take_origin)

        with pytest.raises(Exception):
            import_song_package(target, broken_path)
        assert _row_count(target, "songs") == 0
        assert _row_count(target, "song_versions") == 0
        assert _row_count(target, "layers") == 0
        assert not (target.working_dir / "audio" / "take_ref.wav").exists()
    finally:
        source.close()
        target.close()


def test_media_collision_uses_full_hash_path_and_rewrites_payload(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)
        collision = target.working_dir / "audio" / "take_ref.wav"
        collision.parent.mkdir(parents=True, exist_ok=True)
        collision.write_bytes(b"local different")
        expected_hash = _sha256(b"take audio")

        result = import_song_package(target, package_path)

        imported_object = next(
            record
            for record in target.timeline_objects.list_by_version(result.song_version_id)
            if record.name == "B"
        )
        imported_content = target.object_contents.get(imported_object.main_content_id)
        assert imported_content is not None
        remapped_path = imported_content.payload["audio_file"]
        package_id = inspect_song_package(package_path).package_id
        assert remapped_path == f"audio/song_packages/{package_id}/{expected_hash}.wav"
        assert (target.working_dir / remapped_path).read_bytes() == b"take audio"
    finally:
        source.close()
        target.close()


def test_failed_import_does_not_delete_reused_existing_media(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)
        manifest = inspect_song_package(package_path)
        reused_entry = next(
            entry for entry in manifest.media if entry.rel_path == version.audio_file
        )
        reused_path = target.working_dir / reused_entry.rel_path
        reused_path.parent.mkdir(parents=True, exist_ok=True)
        reused_path.write_bytes((source.working_dir / reused_entry.rel_path).read_bytes())
        collision = target.working_dir / "audio" / "take_ref.wav"
        collision.write_bytes(b"different")
        fallback_parent = target.working_dir / "audio" / "song_packages" / manifest.package_id
        fallback_parent.parent.mkdir(parents=True, exist_ok=True)
        fallback_parent.write_bytes(b"not a directory")

        with pytest.raises(PersistenceError, match="Could not promote imported media"):
            import_song_package(target, package_path)

        assert reused_path.exists()
        assert reused_path.read_bytes() == (source.working_dir / reused_entry.rel_path).read_bytes()
        assert _row_count(target, "songs") == 0
    finally:
        source.close()
        target.close()


def test_import_strips_ma3_layer_routes_but_keeps_package_provenance(tmp_path: Path) -> None:
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        layer = source.layers.get("layer_events")
        assert layer is not None
        source.layers.update(
            LayerRecord(
                **{
                    **layer.__dict__,
                    "state_flags": {
                        "ma3_track_coord": "1.2",
                        "ma3_channel_no": 10,
                        "manual_kind": "event",
                    },
                }
            )
        )
        source.db.commit()
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)

        result = import_song_package(target, package_path)

        imported_layer = next(
            record for record in target.layers.list_by_version(result.song_version_id)
            if record.name == "Events"
        )
        assert "ma3_track_coord" not in imported_layer.state_flags
        assert "ma3_channel_no" not in imported_layer.state_flags
        assert imported_layer.provenance["song_package"]["source_entity_id"] == "layer_events"
    finally:
        source.close()
        target.close()


def test_imported_legacy_video_becomes_version_layer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_timeline.ensure_registered_waveform",
        lambda _key, _path: None,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_timeline_storage.ensure_registered_waveform",
        lambda _key, _path: None,
    )
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        _song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        video_rel = "video/ref.mov"
        video_path = source.working_dir / video_rel
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video one")
        now = datetime.now(timezone.utc)
        source.song_video_attachments.upsert(
            SongVideoAttachmentRecord(
                id="video_attachment",
                song_id=_song.id,
                video_file=video_rel,
                video_hash=_sha256(b"video one"),
                duration_seconds=12.0,
                extracted_audio_file=None,
                extracted_audio_hash=None,
                width=1920,
                height=1080,
                fps=24.0,
                created_at=now,
                updated_at=now,
            )
        )
        source.song_video_placements.upsert(
            SongVideoPlacementRecord(
                song_version_id=version.id,
                video_start_seconds=2.0,
                video_trim_start_seconds=1.0,
                video_visible_duration_seconds=6.0,
                video_loop_enabled=True,
            )
        )
        source.db.commit()
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)

        result = import_song_package(target, package_path)

        assert target.song_video_attachments.get_by_song(result.song_id) is None
        video_layers = [
            layer for layer in target.layers.list_by_version(result.song_version_id)
            if layer.state_flags.get("reference_kind") == "video"
        ]
        assert len(video_layers) == 1
        video_object = target.timeline_objects.get(f"object_{video_layers[0].id}")
        assert video_object is not None
        video_content = target.object_contents.get(video_object.main_content_id)
        assert video_content is not None
        assert video_content.content_kind == "video_clip"
        assert video_content.payload["video_file"] == video_rel
        timeline, overlay, _song_id, _version_id = build_project_native_baseline_timeline(
            target,
            active_song_id=result.song_id,
            active_song_version_id=result.song_version_id,
        )
        assert timeline.layers
        assert len(overlay.layer_video) == 1
        video_fields = next(iter(overlay.layer_video.values()))
        assert video_fields.video_start_seconds == 2.0
        assert video_fields.video_loop_enabled is True
    finally:
        source.close()
        target.close()


def test_package_export_omits_legacy_video_when_version_has_video_layer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_timeline.ensure_registered_waveform",
        lambda _key, _path: None,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_timeline_storage.ensure_registered_waveform",
        lambda _key, _path: None,
    )
    source = _create_storage(tmp_path / "source", "Source")
    target = _create_storage(tmp_path / "target", "Target")
    try:
        song, version, _take_audio_rel = _add_package_fixture(source, tmp_path)
        video_rel = "video/ref.mov"
        video_path = source.working_dir / video_rel
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video one")
        now = datetime.now(timezone.utc)
        source.db.execute(
            "INSERT INTO layers "
            '(id, song_version_id, name, layer_type, color, "order", visible, locked, '
            "parent_layer_id, source_pipeline, state_flags_json, provenance_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "layer_video_existing",
                version.id,
                "Video Reference",
                "manual",
                None,
                2,
                1,
                0,
                None,
                "{}",
                json.dumps({"manual_kind": "reference", "reference_kind": "video"}),
                "{}",
                now.isoformat(),
            ),
        )
        source.db.execute(
            "INSERT INTO timeline_objects "
            "(id, song_version_id, name, object_kind, main_content_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "object_layer_video_existing",
                version.id,
                "Video Reference",
                "video_clip",
                "content_video_existing",
                now.isoformat(),
            ),
        )
        source.db.execute(
            "INSERT INTO object_contents "
            "(id, object_id, revision_id, content_kind, payload_json, "
            "source_ref_json, analysis_build_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "content_video_existing",
                "object_layer_video_existing",
                "revision_video_existing",
                "video_clip",
                json.dumps({"video_file": video_rel, "duration_seconds": 12.0}),
                None,
                None,
                now.isoformat(),
            ),
        )
        source.song_video_attachments.upsert(
            SongVideoAttachmentRecord(
                id="stale_attachment",
                song_id=song.id,
                video_file=video_rel,
                video_hash=_sha256(b"video one"),
                duration_seconds=12.0,
                extracted_audio_file=None,
                extracted_audio_hash=None,
                width=None,
                height=None,
                fps=None,
                created_at=now,
                updated_at=now,
            )
        )
        source.db.commit()
        package_path = tmp_path / "song.ezsong"
        export_song_package(source, version.id, package_path)

        result = import_song_package(target, package_path)

        video_layers = [
            layer for layer in target.layers.list_by_version(result.song_version_id)
            if layer.state_flags.get("reference_kind") == "video"
        ]
        assert len(video_layers) == 1
    finally:
        source.close()
        target.close()
