"""Song import and versioning helpers for project storage.
Exists to keep audio import, default-config propagation, and version rebuild planning out of the ProjectStorage root.
Connects project-level song/version workflows to persistence repositories.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import (
    LayerRecord,
    PipelineConfigRecord,
    ProjectRecord,
    SongDefaultPipelineConfigRecord,
    SongRecord,
    SongVersionRecord,
)
from echozero.persistence.repositories import (
    LayerRepository,
    PipelineConfigRepository,
    SongDefaultPipelineConfigRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
)


class _ProjectStorageVersioningHost(Protocol):
    project: ProjectRecord
    working_dir: Path
    db: sqlite3.Connection
    dirty_tracker: DirtyTracker
    _lock: RLock

    @property
    def songs(self) -> SongRepository: ...

    @property
    def song_versions(self) -> SongVersionRepository: ...

    @property
    def pipeline_configs(self) -> PipelineConfigRepository: ...

    @property
    def song_default_pipeline_configs(self) -> SongDefaultPipelineConfigRepository: ...

    @property
    def layers(self) -> LayerRepository: ...

    @property
    def takes(self) -> TakeRepository: ...

    def _check_closed(self) -> None: ...


class ProjectStorageVersioningMixin:
    def _next_default_ma3_timecode_pool_no(self) -> int:
        """Return the next unused positive MA3 timecode pool number for this project."""

        host = cast(_ProjectStorageVersioningHost, self)
        used_pool_numbers: set[int] = set()
        for song in host.songs.list_by_project(host.project.id):
            for version in host.song_versions.list_by_song(song.id):
                if version.ma3_timecode_pool_no is not None:
                    used_pool_numbers.add(int(version.ma3_timecode_pool_no))

        candidate = 1
        while candidate in used_pool_numbers:
            candidate += 1
        return candidate

    def _create_version(
        self,
        song_id: str,
        audio_source: Path,
        label: str,
        *,
        ma3_timecode_pool_no: int | None = None,
        audio_import_options: object | None = None,
        scan_fn: object | None = None,
    ) -> SongVersionRecord:
        """Shared version factory for audio-backed song versions."""

        host = cast(_ProjectStorageVersioningHost, self)
        from echozero.errors import ValidationError
        from echozero.persistence.audio import (
            AudioImportOptions,
            cleanup_prepared_audio,
            import_audio,
            prepare_audio_for_import,
            scan_audio_metadata,
        )

        import_options = (
            cast(AudioImportOptions, audio_import_options)
            if isinstance(audio_import_options, AudioImportOptions)
            else AudioImportOptions()
        )
        prepared_source = prepare_audio_for_import(
            audio_source,
            host.working_dir,
            options=import_options,
            scan_fn=scan_fn,
        )
        try:
            metadata = scan_audio_metadata(prepared_source.source_path, scan_fn=scan_fn)
        except Exception as exc:
            raise ValidationError(f"Invalid audio file '{audio_source.name}': {exc}") from exc

        try:
            audio_rel_path, audio_hash = import_audio(prepared_source.source_path, host.working_dir)
        finally:
            cleanup_prepared_audio(prepared_source)

        version = SongVersionRecord(
            id=uuid.uuid4().hex,
            song_id=song_id,
            label=label,
            audio_file=audio_rel_path,
            duration_seconds=metadata.duration_seconds,
            original_sample_rate=metadata.sample_rate,
            audio_hash=audio_hash,
            created_at=datetime.now(timezone.utc),
            ma3_timecode_pool_no=ma3_timecode_pool_no,
        )
        host.song_versions.create(version)
        if prepared_source.ltc_artifact_path is not None:
            self._create_import_timecode_layer(
                song_version_id=version.id,
                ltc_artifact_path=prepared_source.ltc_artifact_path,
                scan_fn=scan_fn,
            )
        return version

    def _create_import_timecode_layer(
        self,
        *,
        song_version_id: str,
        ltc_artifact_path: Path,
        scan_fn: object | None = None,
    ) -> None:
        """Create one persisted timecode audio layer for an imported LTC artifact."""

        host = cast(_ProjectStorageVersioningHost, self)
        from echozero.domain.types import AudioData
        from echozero.persistence.audio import scan_audio_metadata
        from echozero.takes import Take

        metadata = scan_audio_metadata(ltc_artifact_path, scan_fn=scan_fn)
        now = datetime.now(timezone.utc)
        existing_layers = host.layers.list_by_version(song_version_id)
        next_order = max((int(layer.order) for layer in existing_layers), default=-1) + 1

        layer_id = uuid.uuid4().hex
        host.layers.create(
            LayerRecord(
                id=layer_id,
                song_version_id=song_version_id,
                name="Timecode",
                layer_type="manual",
                color=None,
                order=next_order,
                visible=True,
                locked=False,
                parent_layer_id=None,
                source_pipeline=None,
                created_at=now,
                state_flags={
                    "manual_kind": "audio",
                    "take_lanes_expanded": False,
                    "import_timecode_layer": True,
                    "output_bus": "outputs_3_4",
                },
                provenance={},
            )
        )
        host.takes.create(
            layer_id,
            Take(
                id=uuid.uuid4().hex,
                label="Main",
                data=AudioData(
                    sample_rate=int(metadata.sample_rate),
                    duration=float(metadata.duration_seconds),
                    file_path=self._relative_project_audio_path(ltc_artifact_path),
                    channel_count=max(1, int(metadata.channel_count)),
                ),
                origin="user",
                source=None,
                created_at=now,
                is_main=True,
                is_archived=False,
                notes="",
            ),
        )

    def _relative_project_audio_path(self, path: Path) -> str:
        """Resolve a best-effort project-relative audio path for persisted take data."""

        host = cast(_ProjectStorageVersioningHost, self)
        try:
            return path.resolve().relative_to(host.working_dir.resolve()).as_posix()
        except ValueError:
            return str(path)

    def import_song(
        self,
        title: str,
        audio_source: Path,
        artist: str = "",
        label: str = "Original",
        default_templates: list[str] | None = None,
        audio_import_options: object | None = None,
        scan_fn: object | None = None,
    ) -> tuple[SongRecord, SongVersionRecord]:
        """Import an audio file as a new song with default pipeline configs."""

        host = cast(_ProjectStorageVersioningHost, self)
        host._check_closed()

        with host._lock:
            song_id = uuid.uuid4().hex
            song = SongRecord(
                id=song_id,
                project_id=host.project.id,
                title=title,
                artist=artist,
                order=len(host.songs.list_by_project(host.project.id)),
                active_version_id=None,
            )
            host.songs.create(song)

            version = self._create_version(
                song_id,
                audio_source,
                label,
                ma3_timecode_pool_no=self._next_default_ma3_timecode_pool_no(),
                audio_import_options=audio_import_options,
                scan_fn=scan_fn,
            )
            updated_song = dataclass_replace(song, active_version_id=version.id)
            host.songs.update(updated_song)

            self._apply_default_templates_to_song(song_id, default_templates)
            self._copy_song_default_configs_to_version(song_id, version.id)

            host.db.commit()
            host.dirty_tracker.mark_dirty(song_id)
            return updated_song, version

    def _apply_default_templates_to_song(
        self,
        song_id: str,
        template_ids: list[str] | None = None,
    ) -> None:
        """Create song default configs from registered templates."""

        host = cast(_ProjectStorageVersioningHost, self)
        from echozero.pipelines.registry import get_registry

        registry = get_registry()
        if template_ids is None:
            templates = registry.list()
        else:
            templates = [template for template_id in template_ids if (template := registry.get(template_id)) is not None]

        for template in templates:
            pipeline = template.build_pipeline()
            config = PipelineConfigRecord.from_pipeline(
                pipeline,
                template_id=template.id,
                song_version_id="",
                knob_values={key: knob.default for key, knob in template.knobs.items()},
                name=template.name,
            )
            host.song_default_pipeline_configs.create(
                SongDefaultPipelineConfigRecord.from_version_config(config, song_id=song_id)
            )

    def _copy_song_default_configs_to_version(
        self, song_id: str, song_version_id: str
    ) -> list[str]:
        """Copy song-default configs into a concrete version scope."""

        host = cast(_ProjectStorageVersioningHost, self)
        new_config_ids: list[str] = []
        now = datetime.now(timezone.utc)
        for config in host.song_default_pipeline_configs.list_by_song(song_id):
            new_config = config.to_version_config(song_version_id=song_version_id)
            materialized = PipelineConfigRecord(
                id=uuid.uuid4().hex,
                song_version_id=new_config.song_version_id,
                template_id=new_config.template_id,
                name=new_config.name,
                graph_json=new_config.graph_json,
                outputs_json=new_config.outputs_json,
                knob_values=dict(new_config.knob_values),
                created_at=now,
                updated_at=now,
                block_overrides={key: list(values) for key, values in new_config.block_overrides.items()},
            )
            host.pipeline_configs.create(materialized)
            new_config_ids.append(materialized.id)
        return new_config_ids

    def _selected_source_layers_for_transfer(
        self,
        source_layers: list[LayerRecord],
        layer_ids_to_transfer: list[str] | None,
    ) -> list[LayerRecord]:
        if layer_ids_to_transfer is None:
            return list(source_layers)
        selected_ids = {
            layer_id.strip()
            for layer_id in layer_ids_to_transfer
            if isinstance(layer_id, str) and layer_id.strip()
        }
        if not selected_ids:
            return []
        return [layer for layer in source_layers if layer.id in selected_ids]

    def _copy_layer_takes(
        self,
        *,
        source_layer_id: str,
        target_layer_id: str,
        created_at: datetime,
    ) -> None:
        host = cast(_ProjectStorageVersioningHost, self)
        for source_take in host.takes.list_by_layer(source_layer_id):
            host.takes.create(
                target_layer_id,
                dataclass_replace(
                    source_take,
                    id=uuid.uuid4().hex,
                    created_at=created_at,
                ),
            )

    def _copy_version_layers(
        self,
        *,
        source_song_version_id: str,
        target_song_version_id: str,
        layer_ids_to_transfer: list[str] | None,
    ) -> list[str]:
        host = cast(_ProjectStorageVersioningHost, self)
        source_layers = host.layers.list_by_version(source_song_version_id)
        selected_source_layers = self._selected_source_layers_for_transfer(
            source_layers,
            layer_ids_to_transfer,
        )
        if not selected_source_layers:
            return []

        now = datetime.now(timezone.utc)
        copied_layer_ids: list[str] = []
        layer_id_map = {
            source_layer.id: uuid.uuid4().hex
            for source_layer in selected_source_layers
        }
        for order_index, source_layer in enumerate(selected_source_layers):
            new_layer_id = layer_id_map[source_layer.id]
            copied_layer = LayerRecord(
                id=new_layer_id,
                song_version_id=target_song_version_id,
                name=source_layer.name,
                layer_type=source_layer.layer_type,
                color=source_layer.color,
                order=order_index,
                visible=source_layer.visible,
                locked=source_layer.locked,
                parent_layer_id=(
                    layer_id_map.get(source_layer.parent_layer_id)
                    if source_layer.parent_layer_id is not None
                    else None
                ),
                source_pipeline=(
                    dict(source_layer.source_pipeline)
                    if source_layer.source_pipeline is not None
                    else None
                ),
                created_at=now,
                state_flags=dict(source_layer.state_flags),
                provenance=dict(source_layer.provenance),
            )
            host.layers.create(copied_layer)
            self._copy_layer_takes(
                source_layer_id=source_layer.id,
                target_layer_id=new_layer_id,
                created_at=now,
            )
            copied_layer_ids.append(new_layer_id)
        return copied_layer_ids

    def add_song_version(
        self,
        song_id: str,
        audio_source: Path,
        label: str | None = None,
        activate: bool = True,
        transfer_layers: bool = False,
        transfer_layer_ids: list[str] | None = None,
        audio_import_options: object | None = None,
        scan_fn: object | None = None,
    ) -> SongVersionRecord:
        """Add a new version of an existing song and copy song defaults into it.

        Optionally clones persisted layers and takes from the source version.
        """

        host = cast(_ProjectStorageVersioningHost, self)
        host._check_closed()

        with host._lock:
            song = host.songs.get(song_id)
            if song is None:
                raise ValueError(f"SongRecord not found: {song_id}")

            source_version_id = song.active_version_id
            if source_version_id is None:
                raise ValueError(f"SongRecord '{song_id}' has no active version")

            if label is None:
                existing = host.song_versions.list_by_song(song_id)
                label = f"v{len(existing) + 1}"

            source_version = host.song_versions.get(source_version_id)
            if source_version is None:
                raise ValueError(f"SongVersionRecord not found: {source_version_id}")

            version = self._create_version(
                song_id,
                audio_source,
                label,
                ma3_timecode_pool_no=source_version.ma3_timecode_pool_no,
                audio_import_options=audio_import_options,
                scan_fn=scan_fn,
            )
            new_config_ids = self._copy_song_default_configs_to_version(song_id, version.id)
            if transfer_layers:
                self._copy_version_layers(
                    source_song_version_id=source_version_id,
                    target_song_version_id=version.id,
                    layer_ids_to_transfer=transfer_layer_ids,
                )

            from echozero.services.provenance import build_song_version_rebuild_plan

            rebuild_plan = build_song_version_rebuild_plan(
                previous_version_id=source_version_id,
                new_version_id=version.id,
                pipeline_config_ids=new_config_ids,
            )
            version = dataclass_replace(version, rebuild_plan=rebuild_plan)
            host.song_versions.update(version)

            if activate:
                host.songs.update(dataclass_replace(song, active_version_id=version.id))

            host.db.commit()
            host.dirty_tracker.mark_dirty(song_id)
            return version

    def reorder_songs(self, song_ids: list[str]) -> None:
        """Persist one full setlist ordering for the active project."""

        host = cast(_ProjectStorageVersioningHost, self)
        host._check_closed()

        with host._lock:
            existing_song_ids = [
                song.id for song in host.songs.list_by_project(host.project.id)
            ]
            if len(song_ids) != len(existing_song_ids):
                raise ValueError(
                    "reorder_songs requires one ID for every song in the setlist."
                )
            if set(song_ids) != set(existing_song_ids):
                raise ValueError(
                    "reorder_songs requires the same song IDs currently in the setlist."
                )

            host.songs.reorder(host.project.id, song_ids)
            host.db.commit()
            for song_id in song_ids:
                host.dirty_tracker.mark_dirty(song_id)

    def move_song(self, song_id: str, direction: int) -> None:
        """Move one song up/down in the setlist by one slot."""

        host = cast(_ProjectStorageVersioningHost, self)
        host._check_closed()

        if direction not in {-1, 1}:
            raise ValueError("move_song direction must be -1 or 1.")

        with host._lock:
            songs = list(host.songs.list_by_project(host.project.id))
            current_index = next(
                (index for index, song in enumerate(songs) if song.id == song_id),
                None,
            )
            if current_index is None:
                raise ValueError(f"SongRecord not found: {song_id}")

            target_index = max(0, min(len(songs) - 1, current_index + direction))
            if target_index == current_index:
                return

            moved = songs.pop(current_index)
            songs.insert(target_index, moved)
            ordered_song_ids = [song.id for song in songs]
            host.songs.reorder(host.project.id, ordered_song_ids)
            host.db.commit()
            for changed_song_id in ordered_song_ids:
                host.dirty_tracker.mark_dirty(changed_song_id)

    def delete_song(self, song_id: str) -> None:
        """Delete one song and reorder the remaining setlist entries."""

        host = cast(_ProjectStorageVersioningHost, self)
        host._check_closed()

        with host._lock:
            song = host.songs.get(song_id)
            if song is None:
                raise ValueError(f"SongRecord not found: {song_id}")

            host.songs.delete(song_id)
            remaining_song_ids = [
                remaining_song.id
                for remaining_song in host.songs.list_by_project(host.project.id)
            ]
            host.songs.reorder(host.project.id, remaining_song_ids)
            host.db.commit()
            host.dirty_tracker.mark_dirty(song_id)

    def delete_song_version(self, song_version_id: str) -> None:
        """Delete one version and retarget the owning song if needed."""

        host = cast(_ProjectStorageVersioningHost, self)
        host._check_closed()

        with host._lock:
            version = host.song_versions.get(song_version_id)
            if version is None:
                raise ValueError(f"SongVersionRecord not found: {song_version_id}")

            song = host.songs.get(version.song_id)
            if song is None:
                raise RuntimeError(
                    f"SongRecord not found for SongVersionRecord '{song_version_id}'"
                )

            versions = host.song_versions.list_by_song(song.id)
            next_active_version = _adjacent_version_id(versions, song_version_id)

            host.song_versions.delete(song_version_id)

            if next_active_version is None:
                host.songs.delete(song.id)
                remaining_song_ids = [
                    remaining_song.id
                    for remaining_song in host.songs.list_by_project(host.project.id)
                ]
                host.songs.reorder(host.project.id, remaining_song_ids)
            elif song.active_version_id == song_version_id:
                host.songs.update(
                    dataclass_replace(song, active_version_id=next_active_version)
                )

            host.db.commit()
            host.dirty_tracker.mark_dirty(song.id)


def _adjacent_version_id(versions: list[SongVersionRecord], deleted_version_id: str) -> str | None:
    remaining_versions = [
        version for version in versions if version.id != deleted_version_id
    ]
    if not remaining_versions:
        return None

    deleted_index = next(
        (
            index
            for index, version in enumerate(versions)
            if version.id == deleted_version_id
        ),
        len(versions) - 1,
    )
    if deleted_index < len(remaining_versions):
        return remaining_versions[deleted_index].id
    return remaining_versions[-1].id


__all__ = ["ProjectStorageVersioningMixin"]
