"""
Song package import/export for portable .ezsong files.
Exists so one song version can move between projects without overwriting local truth.
Connects ProjectStorage records, project media, and ZIP package manifests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
from contextlib import suppress
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echozero.errors import PersistenceError
from echozero.persistence.archive import APP_VERSION
from echozero.persistence.schema import SCHEMA_VERSION

SONG_PACKAGE_FORMAT_VERSION = 1
SONG_PACKAGE_EXTENSION = ".ezsong"
_PACKAGE_MEDIA_ROOTS = ("audio/", "video/")
_PACKAGE_ALLOWED_FIXED_MEMBERS = frozenset({"manifest.json", "payload/song.json"})
_LIVE_MA3_ROUTE_KEYS = frozenset(
    {
        "ma3_track_coord",
        "ma3_channel_no",
        "ma3_track_group_no",
        "ma3_track_no",
        "ma3_route",
        "ma3_channel_mapping",
        "ma3_track_mapping",
        "live_sync_state",
        "sync_target_channel_no",
    }
)


@dataclass(frozen=True, slots=True)
class SongPackageMediaEntry:
    """One media file carried inside a song package."""

    rel_path: str
    package_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SongPackageManifest:
    """User-facing metadata and integrity index for a .ezsong package."""

    format_version: int
    app_version: str
    schema_version: int
    package_id: str
    source_project_id: str
    source_song_id: str
    source_song_version_id: str
    title: str
    artist: str
    version_label: str
    duration_seconds: float
    audio_hash: str
    created_at: str
    media: tuple[SongPackageMediaEntry, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SongPackageManifest:
        """Create a manifest from decoded JSON."""

        media_entries = tuple(
            SongPackageMediaEntry(
                rel_path=str(entry["rel_path"]),
                package_path=str(entry["package_path"]),
                sha256=str(entry["sha256"]),
                size_bytes=int(entry["size_bytes"]),
            )
            for entry in data.get("media", [])
        )
        return cls(
            format_version=int(data["format_version"]),
            app_version=str(data["app_version"]),
            schema_version=int(data["schema_version"]),
            package_id=str(data["package_id"]),
            source_project_id=str(data["source_project_id"]),
            source_song_id=str(data["source_song_id"]),
            source_song_version_id=str(data["source_song_version_id"]),
            title=str(data["title"]),
            artist=str(data.get("artist", "")),
            version_label=str(data["version_label"]),
            duration_seconds=float(data["duration_seconds"]),
            audio_hash=str(data["audio_hash"]),
            created_at=str(data["created_at"]),
            media=media_entries,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""

        return {
            "format_version": self.format_version,
            "app_version": self.app_version,
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "source_project_id": self.source_project_id,
            "source_song_id": self.source_song_id,
            "source_song_version_id": self.source_song_version_id,
            "title": self.title,
            "artist": self.artist,
            "version_label": self.version_label,
            "duration_seconds": self.duration_seconds,
            "audio_hash": self.audio_hash,
            "created_at": self.created_at,
            "media": [
                {
                    "rel_path": entry.rel_path,
                    "package_path": entry.package_path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for entry in self.media
            ],
        }


@dataclass(frozen=True, slots=True)
class SongPackageImportResult:
    """Result of importing one song package into a project."""

    song_id: str
    song_version_id: str
    created_song: bool
    activated: bool
    copied_media: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class _ImportContext:
    package_id: str
    source_project_id: str
    source_song_id: str
    source_song_version_id: str
    created_at: str
    id_map: dict[str, str] = field(default_factory=dict)
    media_path_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _StagedPackageMedia:
    staging_dir: Path
    destination_rel_paths: tuple[str, ...]


def inspect_song_package(path: Path) -> SongPackageManifest:
    """Read and validate package metadata without importing it."""

    with zipfile.ZipFile(path, "r") as archive:
        _validate_package_members(archive)
        if "manifest.json" not in archive.namelist():
            raise PersistenceError(f"Invalid song package: missing manifest.json in {path}")
        try:
            manifest = SongPackageManifest.from_dict(json.loads(archive.read("manifest.json")))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError(f"Invalid song package manifest: {exc}") from exc
        _validate_manifest(manifest)
        return manifest


def export_song_package(storage: Any, song_version_id: str, dest_path: Path) -> SongPackageManifest:
    """Export one song version and its referenced media to a .ezsong package."""

    storage._check_closed()
    version = storage.song_versions.get(song_version_id)
    if version is None:
        raise ValueError(f"SongVersionRecord not found: {song_version_id}")
    song = storage.songs.get(version.song_id)
    if song is None:
        raise RuntimeError(f"SongRecord not found for SongVersionRecord '{song_version_id}'")

    payload = _build_payload(storage, song_id=song.id, song_version_id=version.id)
    media_paths = _collect_media_paths(payload)
    media_entries = _build_media_entries(storage.working_dir, media_paths)
    manifest = SongPackageManifest(
        format_version=SONG_PACKAGE_FORMAT_VERSION,
        app_version=APP_VERSION,
        schema_version=SCHEMA_VERSION,
        package_id=uuid.uuid4().hex,
        source_project_id=storage.project.id,
        source_song_id=song.id,
        source_song_version_id=version.id,
        title=song.title,
        artist=song.artist,
        version_label=version.label,
        duration_seconds=float(version.duration_seconds),
        audio_hash=version.audio_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        media=tuple(media_entries),
    )

    destination = dest_path if dest_path.suffix else dest_path.with_suffix(SONG_PACKAGE_EXTENSION)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp_path, "w") as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest.to_dict(), indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.writestr(
                "payload/song.json",
                json.dumps(payload, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for entry in media_entries:
                source = storage.working_dir / entry.rel_path
                archive.write(
                    source,
                    entry.package_path,
                    compress_type=zipfile.ZIP_STORED,
                )
        tmp_path.replace(destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return manifest


def import_song_package(
    storage: Any,
    package_path: Path,
    *,
    target_song_id: str | None = None,
    activate_import: bool = False,
) -> SongPackageImportResult:
    """Import a .ezsong package as a new song or as a new version on an existing song."""

    storage._check_closed()
    with zipfile.ZipFile(package_path, "r") as archive:
        _validate_package_members(archive)
        manifest = inspect_song_package(package_path)
        payload = _read_payload(archive)
        _validate_package_member_allowlist(archive, manifest)
        _validate_payload_media_index(payload, manifest)
        _validate_source_ref_closure(payload)
        _verify_media_entries(archive, manifest)

        context = _ImportContext(
            package_id=manifest.package_id,
            source_project_id=manifest.source_project_id,
            source_song_id=manifest.source_song_id,
            source_song_version_id=manifest.source_song_version_id,
            created_at=manifest.created_at,
        )
        staged_media = _stage_package_media(
            archive=archive,
            manifest=manifest,
            working_dir=storage.working_dir,
            context=context,
        )
        promoted_media: list[str] = []
        try:
            with storage.transaction():
                created_song = target_song_id is None
                if created_song:
                    song_id = uuid.uuid4().hex
                    song_order = len(storage.songs.list_by_project(storage.project.id))
                else:
                    song = storage.songs.get(str(target_song_id))
                    if song is None:
                        raise ValueError(f"SongRecord not found: {target_song_id}")
                    song_id = song.id
                    song_order = int(song.order)

                version_id = uuid.uuid4().hex
                context.id_map[str(payload["version"]["id"])] = version_id

                now = datetime.now(timezone.utc).isoformat()
                if created_song:
                    storage.db.execute(
                        "INSERT INTO songs "
                        '(id, project_id, title, artist, "order", active_version_id) '
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            song_id,
                            storage.project.id,
                            manifest.title,
                            manifest.artist,
                            song_order,
                            version_id,
                        ),
                    )

                ma3_pool = _resolve_import_ma3_pool(storage, target_song_id=song_id)
                version_label = _unique_version_label(
                    storage,
                    song_id=song_id,
                    requested_label=manifest.version_label,
                )
                version_row = dict(payload["version"])
                version_row.update(
                    {
                        "id": version_id,
                        "song_id": song_id,
                        "label": version_label,
                        "audio_file": _rewrite_media_path(
                            str(version_row["audio_file"]),
                            context.media_path_map,
                        ),
                        "ma3_timecode_pool_no": ma3_pool,
                        "rebuild_plan_json": json.dumps(
                            _package_provenance(context, entity_id=str(payload["version"]["id"]))
                        ),
                        "created_at": now,
                    }
                )
                _insert_song_version(storage, version_row)
                _import_payload_rows(
                    storage,
                    payload=payload,
                    song_id=song_id,
                    version_id=version_id,
                    created_song=created_song,
                    context=context,
                    now=now,
                )

                if not created_song and activate_import:
                    song = storage.songs.get(song_id)
                    if song is None:
                        raise RuntimeError(f"SongRecord not found after import: {song_id}")
                    storage.songs.update(dataclass_replace(song, active_version_id=version_id))

                promoted_media = _promote_staged_media(
                    staged_media=staged_media,
                    working_dir=storage.working_dir,
                )
            storage.dirty_tracker.mark_dirty(song_id)
        except Exception:
            _cleanup_imported_media(storage.working_dir, promoted_media)
            raise
        finally:
            shutil.rmtree(staged_media.staging_dir, ignore_errors=True)

    return SongPackageImportResult(
        song_id=song_id,
        song_version_id=version_id,
        created_song=created_song,
        activated=created_song or activate_import,
        copied_media=tuple(staged_media.destination_rel_paths),
        warnings=(),
    )


def _build_payload(storage: Any, *, song_id: str, song_version_id: str) -> dict[str, Any]:
    song_row = _fetch_required_row(storage, "songs", "id", song_id)
    version_row = _fetch_required_row(storage, "song_versions", "id", song_version_id)
    layers = _fetch_rows(
        storage,
        "SELECT * FROM layers WHERE song_version_id = ? ORDER BY \"order\"",
        (song_version_id,),
    )
    layer_ids = [row["id"] for row in layers]
    takes = _fetch_rows_for_ids(storage, "takes", "layer_id", layer_ids, "created_at")
    objects = _fetch_rows(
        storage,
        "SELECT * FROM timeline_objects WHERE song_version_id = ? ORDER BY created_at",
        (song_version_id,),
    )
    object_ids = [row["id"] for row in objects]
    contents = _fetch_rows_for_ids(
        storage,
        "object_contents",
        "object_id",
        object_ids,
        "created_at",
    )
    objects, contents = _expand_source_ref_rows(storage, objects=objects, contents=contents)
    object_ids = [row["id"] for row in objects]
    candidates = _fetch_rows_for_ids(
        storage,
        "object_candidates",
        "object_id",
        object_ids,
        "created_at",
    )
    payload = {
        "song": song_row,
        "version": version_row,
        "song_default_pipeline_configs": _fetch_rows(
            storage,
            "SELECT * FROM song_default_pipeline_configs WHERE song_id = ? ORDER BY created_at",
            (song_id,),
        ),
        "pipeline_configs": _fetch_rows(
            storage,
            "SELECT * FROM pipeline_configs WHERE song_version_id = ? ORDER BY created_at",
            (song_version_id,),
        ),
        "layers": layers,
        "takes": takes,
        "timeline_objects": objects,
        "object_contents": contents,
        "object_candidates": candidates,
        "video_attachment": _fetch_optional_row(
            storage,
            "SELECT * FROM song_video_attachments WHERE song_id = ?",
            (song_id,),
        ),
        "video_placement": _fetch_optional_row(
            storage,
            "SELECT * FROM song_video_placements WHERE song_version_id = ?",
            (song_version_id,),
        ),
    }
    if _payload_has_video_clip(payload):
        payload["video_attachment"] = None
        payload["video_placement"] = None
    return payload


def _expand_source_ref_rows(
    storage: Any,
    *,
    objects: list[dict[str, Any]],
    contents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects_by_id = {str(row["id"]): dict(row) for row in objects}
    contents_by_id = {str(row["id"]): dict(row) for row in contents}
    pending = list(contents_by_id.values())
    while pending:
        row = pending.pop()
        source_ref = _json_loads(row.get("source_ref_json"), {})
        if not source_ref:
            continue
        object_id = str(source_ref.get("object_id") or "")
        content_id = str(source_ref.get("content_id") or "")
        revision_id = str(source_ref.get("revision_id") or "")
        if not object_id or not content_id:
            continue
        if content_id in contents_by_id and object_id in objects_by_id:
            continue
        object_row = _fetch_optional_row(
            storage,
            "SELECT * FROM timeline_objects WHERE id = ?",
            (object_id,),
        )
        content_row = _fetch_optional_row(
            storage,
            "SELECT * FROM object_contents WHERE id = ?",
            (content_id,),
        )
        if object_row is None or content_row is None:
            raise PersistenceError(
                "Cannot export song package with unresolved source_ref: "
                f"object_id={object_id!r}, content_id={content_id!r}."
            )
        if str(content_row.get("object_id") or "") != object_id:
            raise PersistenceError(
                "Cannot export song package with inconsistent source_ref: "
                f"content_id={content_id!r}."
            )
        if revision_id and str(content_row.get("revision_id") or "") != revision_id:
            raise PersistenceError(
                "Cannot export song package with stale source_ref revision: "
                f"content_id={content_id!r}."
            )
        if object_id not in objects_by_id:
            objects_by_id[object_id] = object_row
        if content_id not in contents_by_id:
            contents_by_id[content_id] = content_row
            pending.append(content_row)
    return list(objects_by_id.values()), list(contents_by_id.values())


def _import_payload_rows(
    storage: Any,
    *,
    payload: dict[str, Any],
    song_id: str,
    version_id: str,
    created_song: bool,
    context: _ImportContext,
    now: str,
) -> None:
    source_version_id = str(payload["version"]["id"])
    context.id_map.setdefault(f"object_song_{source_version_id}", f"object_song_{version_id}")
    context.id_map.setdefault(
        f"content_song_audio_{source_version_id}",
        f"content_song_audio_{version_id}",
    )

    if created_song:
        for row in payload.get("song_default_pipeline_configs", []):
            new_row = dict(row)
            context.id_map[str(row["id"])] = uuid.uuid4().hex
            new_row.update({"id": context.id_map[str(row["id"])], "song_id": song_id})
            _insert_song_default_config(storage, new_row, now=now)

    for row in payload.get("pipeline_configs", []):
        new_row = dict(row)
        context.id_map[str(row["id"])] = uuid.uuid4().hex
        new_row.update({"id": context.id_map[str(row["id"])], "song_version_id": version_id})
        _insert_pipeline_config(storage, new_row, now=now)

    _allocate_ids(context, payload.get("layers", []))
    _allocate_ids(context, payload.get("takes", []))
    _allocate_ids(context, payload.get("timeline_objects", []))
    _allocate_ids(context, payload.get("object_contents", []))
    _allocate_ids(context, payload.get("object_candidates", []))

    for row in payload.get("layers", []):
        new_row = dict(row)
        source_provenance = _json_loads(new_row.get("provenance_json"), {})
        source_provenance["song_package"] = _package_provenance(
            context,
            entity_id=str(row["id"]),
        )
        new_row.update(
            {
                "id": context.id_map[str(row["id"])],
                "song_version_id": version_id,
                "parent_layer_id": _mapped_optional(context, row.get("parent_layer_id")),
                "source_pipeline": _rewrite_json_text(
                    row.get("source_pipeline"),
                    context.media_path_map,
                ),
                "state_flags_json": json.dumps(
                    _strip_ma3_route_state(
                        _rewrite_media_values(
                            _json_loads(row.get("state_flags_json") or "{}", {}),
                            context.media_path_map,
                        )
                    )
                ),
                "provenance_json": json.dumps(
                    _rewrite_media_values(source_provenance, context.media_path_map)
                ),
                "created_at": now,
            }
        )
        _insert_layer(storage, new_row)

    for row in payload.get("timeline_objects", []):
        new_row = dict(row)
        new_row.update(
            {
                "id": context.id_map[str(row["id"])],
                "song_version_id": version_id,
                "main_content_id": context.id_map[str(row["main_content_id"])],
                "created_at": now,
            }
        )
        _insert_timeline_object(storage, new_row)

    for row in payload.get("object_contents", []):
        new_row = dict(row)
        new_row.update(
            {
                "id": context.id_map[str(row["id"])],
                "object_id": context.id_map[str(row["object_id"])],
                "payload_json": _rewrite_json_text(row.get("payload_json"), context.media_path_map),
                "source_ref_json": _rewrite_source_ref(row.get("source_ref_json"), context),
                "analysis_build_json": _rewrite_json_text(
                    row.get("analysis_build_json"),
                    context.media_path_map,
                ),
                "created_at": now,
            }
        )
        _insert_object_content(storage, new_row)

    for row in payload.get("object_candidates", []):
        new_row = dict(row)
        new_row.update(
            {
                "id": context.id_map[str(row["id"])],
                "object_id": context.id_map[str(row["object_id"])],
                "content_id": context.id_map[str(row["content_id"])],
                "created_at": now,
            }
        )
        _insert_object_candidate(storage, new_row)

    for row in payload.get("takes", []):
        new_row = dict(row)
        new_row.update(
            {
                "id": context.id_map[str(row["id"])],
                "layer_id": context.id_map[str(row["layer_id"])],
                "source_json": _rewrite_json_text(row.get("source_json"), context.media_path_map),
                "data_json": _rewrite_json_text(row.get("data_json"), context.media_path_map),
                "created_at": now,
            }
        )
        _insert_take(storage, new_row)

    if not _payload_has_video_clip(payload):
        _import_legacy_video_as_layer(
            storage,
            payload=payload,
            version_id=version_id,
            context=context,
            now=now,
        )


def _allocate_ids(context: _ImportContext, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        context.id_map.setdefault(str(row["id"]), uuid.uuid4().hex)


def _payload_has_video_clip(payload: dict[str, Any]) -> bool:
    for row in payload.get("object_contents", []):
        if str(row.get("content_kind") or "").strip().lower() == "video_clip":
            return True
    for row in payload.get("layers", []):
        state_flags = _json_loads(row.get("state_flags_json"), {})
        if (
            str(state_flags.get("reference_kind") or "").strip().lower() == "video"
            or bool(state_flags.get("package_video_layer"))
        ):
            return True
    return False


def _import_legacy_video_as_layer(
    storage: Any,
    *,
    payload: dict[str, Any],
    version_id: str,
    context: _ImportContext,
    now: str,
) -> None:
    attachment = payload.get("video_attachment")
    if attachment is None:
        return
    source_attachment = dict(attachment)
    video_file = _rewrite_media_path(str(source_attachment["video_file"]), context.media_path_map)
    extracted_audio_file = _rewrite_optional_media_path(
        source_attachment.get("extracted_audio_file"),
        context.media_path_map,
    )
    placement = dict(payload.get("video_placement") or {})
    layer_id = f"layer_video_{uuid.uuid4().hex}"
    take_id = f"take_video_{uuid.uuid4().hex}"
    object_id = f"object_{layer_id}"
    content_id = f"content_{take_id}"
    revision_hash = str(source_attachment.get("video_hash") or uuid.uuid4().hex)
    revision_id = f"revision_video_{revision_hash}"
    order = _next_layer_order(payload)
    provenance = _package_provenance(context, entity_id=str(source_attachment.get("id", "")))
    provenance["source_video_attachment_id"] = str(source_attachment.get("id", ""))
    _insert_layer(
        storage,
        {
            "id": layer_id,
            "song_version_id": version_id,
            "name": "Video Reference",
            "layer_type": "manual",
            "color": None,
            "order": order,
            "visible": True,
            "locked": False,
            "parent_layer_id": None,
            "source_pipeline": json.dumps({"package_video": True}),
            "state_flags_json": json.dumps(
                {
                    "manual_kind": "reference",
                    "reference_kind": "video",
                    "package_video_layer": True,
                }
            ),
            "provenance_json": json.dumps(provenance),
            "created_at": now,
        },
    )
    _insert_take(
        storage,
        {
            "id": take_id,
            "layer_id": layer_id,
            "label": "Video",
            "origin": "user",
            "is_main": 1,
            "is_archived": 0,
            "source_json": None,
            "data_json": json.dumps({"type": "EventData", "layers": []}),
            "created_at": now,
            "notes": "Imported .ezsong video reference.",
        },
    )
    _insert_timeline_object(
        storage,
        {
            "id": object_id,
            "song_version_id": version_id,
            "name": "Video Reference",
            "object_kind": "video_clip",
            "main_content_id": content_id,
            "created_at": now,
        },
    )
    _insert_object_content(
        storage,
        {
            "id": content_id,
            "object_id": object_id,
            "revision_id": revision_id,
            "content_kind": "video_clip",
            "payload_json": json.dumps(
                {
                    "video_file": video_file,
                    "video_hash": source_attachment.get("video_hash"),
                    "duration_seconds": source_attachment.get("duration_seconds"),
                    "extracted_audio_file": extracted_audio_file,
                    "extracted_audio_hash": source_attachment.get("extracted_audio_hash"),
                    "width": source_attachment.get("width"),
                    "height": source_attachment.get("height"),
                    "fps": source_attachment.get("fps"),
                    "video_start_seconds": placement.get("video_start_seconds", 0.0),
                    "video_trim_start_seconds": placement.get("video_trim_start_seconds", 0.0),
                    "video_visible_duration_seconds": placement.get(
                        "video_visible_duration_seconds"
                    ),
                    "video_loop_enabled": bool(placement.get("video_loop_enabled", False)),
                    "provenance": provenance,
                }
            ),
            "source_ref_json": None,
            "analysis_build_json": None,
            "created_at": now,
        },
    )


def _next_layer_order(payload: dict[str, Any]) -> int:
    orders: list[int] = []
    for row in payload.get("layers", []):
        try:
            orders.append(int(row.get("order", 0)))
        except (TypeError, ValueError):
            pass
    return (max(orders) + 1) if orders else 1


def _stage_package_media(
    *,
    archive: zipfile.ZipFile,
    manifest: SongPackageManifest,
    working_dir: Path,
    context: _ImportContext,
) -> _StagedPackageMedia:
    staging_dir = working_dir / ".package_import_tmp" / f"{manifest.package_id}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    try:
        for entry in manifest.media:
            requested_rel = _normalize_rel_path(entry.rel_path)
            destination_rel = _available_media_destination(
                working_dir=working_dir,
                requested_rel=requested_rel,
                expected_hash=entry.sha256,
                package_id=manifest.package_id,
            )
            destination_path = staging_dir / destination_rel
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry.package_path, "r") as source, open(
                destination_path,
                "wb",
            ) as dest:
                shutil.copyfileobj(source, dest)
            actual_hash = _hash_file(destination_path)
            if actual_hash != entry.sha256:
                raise PersistenceError(f"Media hash mismatch after import: {entry.rel_path}")
            context.media_path_map[requested_rel] = destination_rel
            copied.append(destination_rel)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return _StagedPackageMedia(staging_dir=staging_dir, destination_rel_paths=tuple(copied))


def _promote_staged_media(*, staged_media: _StagedPackageMedia, working_dir: Path) -> list[str]:
    promoted: list[str] = []
    for destination_rel in staged_media.destination_rel_paths:
        source_path = staged_media.staging_dir / destination_rel
        destination_path = working_dir / destination_rel
        if destination_path.exists():
            if _hash_file(destination_path) == _hash_file(source_path):
                continue
            raise PersistenceError(f"Media destination collision during import: {destination_rel}")
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(destination_path))
        except OSError as exc:
            raise PersistenceError(
                f"Could not promote imported media: {destination_rel}"
            ) from exc
        promoted.append(destination_rel)
    return promoted


def _cleanup_imported_media(working_dir: Path, rel_paths: list[str]) -> None:
    for rel_path in rel_paths:
        with suppress(Exception):
            path = working_dir / _normalize_rel_path(rel_path)
            if path.is_file():
                path.unlink()


def _available_media_destination(
    *,
    working_dir: Path,
    requested_rel: str,
    expected_hash: str,
    package_id: str,
) -> str:
    requested_path = working_dir / requested_rel
    if not requested_path.exists():
        return requested_rel
    if _hash_file(requested_path) == expected_hash:
        return requested_rel
    requested = Path(requested_rel)
    root = requested.parts[0] if requested.parts else "media"
    suffix = requested.suffix
    return f"{root}/song_packages/{package_id}/{expected_hash}{suffix}"


def _build_media_entries(working_dir: Path, media_paths: set[str]) -> list[SongPackageMediaEntry]:
    entries: list[SongPackageMediaEntry] = []
    for rel_path in sorted(media_paths):
        normalized = _normalize_rel_path(rel_path)
        _require_media_rel_path(normalized)
        source = working_dir / normalized
        if not source.is_file():
            raise PersistenceError(f"Cannot export missing media file: {normalized}")
        sha256 = _hash_file(source)
        entries.append(
            SongPackageMediaEntry(
                rel_path=normalized,
                package_path=f"media/{normalized}",
                sha256=sha256,
                size_bytes=source.stat().st_size,
            )
        )
    return entries


def _collect_media_paths(value: Any) -> set[str]:
    paths: set[str] = set()

    def visit(item: Any, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key)
            return
        if not isinstance(item, str):
            return
        if key is not None and key.endswith("_json"):
            try:
                visit(json.loads(item), None)
            except json.JSONDecodeError:
                pass
            return
        if key not in {"audio_file", "file_path", "video_file", "extracted_audio_file"}:
            return
        try:
            normalized = _normalize_rel_path(item)
        except PersistenceError:
            return
        if normalized.startswith(("audio/", "video/")):
            paths.add(normalized)

    visit(value)
    return paths


def _validate_package_members(archive: zipfile.ZipFile) -> None:
    for name in archive.namelist():
        _normalize_rel_path(name)


def _validate_manifest(manifest: SongPackageManifest) -> None:
    if manifest.format_version != SONG_PACKAGE_FORMAT_VERSION:
        if manifest.format_version > SONG_PACKAGE_FORMAT_VERSION:
            detail = "is newer than supported"
        else:
            detail = "is not supported"
        raise PersistenceError(
            "Song package format version "
            f"{manifest.format_version} {detail} "
            f"({SONG_PACKAGE_FORMAT_VERSION})."
        )
    if manifest.schema_version != SCHEMA_VERSION:
        raise PersistenceError(
            "Song package schema version "
            f"{manifest.schema_version} is not supported ({SCHEMA_VERSION})."
        )
    seen_rel_paths: set[str] = set()
    seen_package_paths: set[str] = set()
    for entry in manifest.media:
        rel_path = _normalize_rel_path(entry.rel_path)
        package_path = _normalize_rel_path(entry.package_path)
        _require_media_rel_path(rel_path)
        expected_package_path = f"media/{rel_path}"
        if package_path != expected_package_path:
            raise PersistenceError(f"Invalid media package path: {entry.package_path}")
        if rel_path in seen_rel_paths or package_path in seen_package_paths:
            raise PersistenceError(f"Duplicate media entry in song package: {rel_path}")
        seen_rel_paths.add(rel_path)
        seen_package_paths.add(package_path)


def _validate_package_member_allowlist(
    archive: zipfile.ZipFile,
    manifest: SongPackageManifest,
) -> None:
    allowed_members = set(_PACKAGE_ALLOWED_FIXED_MEMBERS)
    allowed_members.update(entry.package_path for entry in manifest.media)
    for name in archive.namelist():
        normalized = _normalize_rel_path(name)
        if normalized not in allowed_members:
            raise PersistenceError(f"Unsupported song package member: {normalized}")


def _validate_payload_media_index(
    payload: dict[str, Any],
    manifest: SongPackageManifest,
) -> None:
    media_paths = {_normalize_rel_path(path) for path in _collect_media_paths(payload)}
    manifest_paths = {_normalize_rel_path(entry.rel_path) for entry in manifest.media}
    missing = media_paths - manifest_paths
    extra = manifest_paths - media_paths
    if missing:
        raise PersistenceError(
            "Song package manifest is missing media entries: " + ", ".join(sorted(missing))
        )
    if extra:
        raise PersistenceError(
            "Song package manifest has unreferenced media entries: " + ", ".join(sorted(extra))
        )


def _validate_source_ref_closure(payload: dict[str, Any]) -> None:
    object_ids = {str(row.get("id")) for row in payload.get("timeline_objects", [])}
    content_by_id = {
        str(row.get("id")): row for row in payload.get("object_contents", [])
    }
    for row in payload.get("object_contents", []):
        source_ref = _json_loads(row.get("source_ref_json"), {})
        if not source_ref:
            continue
        object_id = str(source_ref.get("object_id") or "")
        content_id = str(source_ref.get("content_id") or "")
        if object_id not in object_ids or content_id not in content_by_id:
            raise PersistenceError(
                "Song package contains dangling source_ref: "
                f"object_id={object_id!r}, content_id={content_id!r}."
            )
        expected_object_id = str(content_by_id[content_id].get("object_id") or "")
        if expected_object_id != object_id:
            raise PersistenceError(
                "Song package contains inconsistent source_ref: "
                f"content_id={content_id!r} belongs to object_id={expected_object_id!r}."
            )
        revision_id = source_ref.get("revision_id")
        if revision_id is not None and str(revision_id) != str(
            content_by_id[content_id].get("revision_id")
        ):
            raise PersistenceError(
                "Song package contains stale source_ref revision: "
                f"content_id={content_id!r}."
            )


def _read_payload(archive: zipfile.ZipFile) -> dict[str, Any]:
    if "payload/song.json" not in archive.namelist():
        raise PersistenceError("Invalid song package: missing payload/song.json")
    try:
        return json.loads(archive.read("payload/song.json"))
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"Invalid song package payload: {exc}") from exc


def _verify_media_entries(archive: zipfile.ZipFile, manifest: SongPackageManifest) -> None:
    names = set(archive.namelist())
    for entry in manifest.media:
        if entry.package_path not in names:
            raise PersistenceError(f"Song package is missing media file: {entry.rel_path}")
        info = archive.getinfo(entry.package_path)
        if info.file_size != entry.size_bytes:
            raise PersistenceError(f"Song package media size mismatch: {entry.rel_path}")
        digest = hashlib.sha256()
        with archive.open(entry.package_path, "r") as media:
            while chunk := media.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != entry.sha256:
            raise PersistenceError(f"Song package media hash mismatch: {entry.rel_path}")


def _require_media_rel_path(value: str) -> None:
    if not value.startswith(_PACKAGE_MEDIA_ROOTS):
        raise PersistenceError(f"Media path must be under audio/ or video/: {value}")


def _rewrite_media_values(value: Any, media_path_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_media_values(child, media_path_map) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_media_values(child, media_path_map) for child in value]
    if isinstance(value, str):
        return _rewrite_media_path(value, media_path_map)
    return value


def _rewrite_media_path(value: str, media_path_map: dict[str, str]) -> str:
    try:
        normalized = _normalize_rel_path(value)
    except PersistenceError:
        return value
    return media_path_map.get(normalized, value)


def _rewrite_optional_media_path(value: Any, media_path_map: dict[str, str]) -> str | None:
    if value is None:
        return None
    return _rewrite_media_path(str(value), media_path_map)


def _rewrite_json_text(raw: Any, media_path_map: dict[str, str]) -> str | None:
    if raw is None:
        return None
    decoded = _json_loads(raw, {})
    return json.dumps(_rewrite_media_values(decoded, media_path_map))


def _rewrite_source_ref(raw: Any, context: _ImportContext) -> str | None:
    if raw is None:
        return None
    source_ref = _json_loads(raw, {})
    if not source_ref:
        return None
    object_id = str(source_ref.get("object_id") or "")
    content_id = str(source_ref.get("content_id") or "")
    if object_id in context.id_map:
        source_ref["object_id"] = context.id_map[object_id]
    if content_id in context.id_map:
        source_ref["content_id"] = context.id_map[content_id]
    return json.dumps(_rewrite_media_values(source_ref, context.media_path_map))


def _strip_ma3_route_state(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}
    stripped = dict(value)
    for key in tuple(stripped.keys()):
        normalized = str(key).strip().lower()
        if normalized in _LIVE_MA3_ROUTE_KEYS:
            stripped.pop(key, None)
    return stripped


def _json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return default


def _package_provenance(context: _ImportContext, *, entity_id: str) -> dict[str, str]:
    return {
        "package_id": context.package_id,
        "source_project_id": context.source_project_id,
        "source_song_id": context.source_song_id,
        "source_song_version_id": context.source_song_version_id,
        "source_entity_id": entity_id,
        "exported_at": context.created_at,
    }


def _resolve_import_ma3_pool(storage: Any, *, target_song_id: str) -> int | None:
    song = storage.songs.get(target_song_id)
    if song is not None and song.active_version_id is not None:
        version = storage.song_versions.get(song.active_version_id)
        if version is not None:
            return version.ma3_timecode_pool_no
    return storage._next_default_ma3_timecode_pool_no()


def _unique_version_label(storage: Any, *, song_id: str, requested_label: str) -> str:
    existing = {version.label for version in storage.song_versions.list_by_song(song_id)}
    if requested_label not in existing:
        return requested_label
    index = 2
    while f"{requested_label} ({index})" in existing:
        index += 1
    return f"{requested_label} ({index})"


def _mapped_optional(context: _ImportContext, value: Any) -> str | None:
    if value is None:
        return None
    return context.id_map.get(str(value))


def _normalize_rel_path(value: str) -> str:
    path = Path(str(value))
    if path.is_absolute():
        raise PersistenceError(f"Package path must be relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PersistenceError(f"Invalid package path: {value!r}")
    return path.as_posix()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_required_row(storage: Any, table: str, column: str, value: str) -> dict[str, Any]:
    row = storage.db.execute(f"SELECT * FROM {table} WHERE {column} = ?", (value,)).fetchone()
    if row is None:
        raise PersistenceError(f"Missing {table} row for {column}={value!r}")
    return dict(row)


def _fetch_optional_row(storage: Any, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    row = storage.db.execute(sql, params).fetchone()
    return None if row is None else dict(row)


def _fetch_rows(storage: Any, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in storage.db.execute(sql, params).fetchall()]


def _fetch_rows_for_ids(
    storage: Any,
    table: str,
    column: str,
    values: list[str],
    order_column: str,
) -> list[dict[str, Any]]:
    if not values:
        return []
    placeholders = ", ".join("?" for _ in values)
    return _fetch_rows(
        storage,
        f"SELECT * FROM {table} WHERE {column} IN ({placeholders}) ORDER BY {order_column}",
        tuple(values),
    )


def _insert_song_version(storage: Any, row: dict[str, Any]) -> None:
    storage.db.execute(
        "INSERT INTO song_versions "
        "(id, song_id, label, audio_file, duration_seconds, original_sample_rate, "
        "audio_hash, bpm, bpm_confidence, beat_anchor_seconds, ma3_timecode_pool_no, "
        "rebuild_plan_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["song_id"],
            row["label"],
            row["audio_file"],
            row["duration_seconds"],
            row["original_sample_rate"],
            row["audio_hash"],
            row.get("bpm"),
            row.get("bpm_confidence"),
            row.get("beat_anchor_seconds"),
            row.get("ma3_timecode_pool_no"),
            row.get("rebuild_plan_json") or "{}",
            row["created_at"],
        ),
    )


def _insert_song_default_config(storage: Any, row: dict[str, Any], *, now: str) -> None:
    storage.db.execute(
        "INSERT INTO song_default_pipeline_configs "
        "(id, song_id, template_id, name, graph_json, outputs_json, knob_values_json, "
        "block_overrides_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["song_id"],
            row["template_id"],
            row["name"],
            row["graph_json"],
            row.get("outputs_json") or "[]",
            row.get("knob_values_json") or "{}",
            row.get("block_overrides_json") or "{}",
            now,
            now,
        ),
    )


def _insert_pipeline_config(storage: Any, row: dict[str, Any], *, now: str) -> None:
    storage.db.execute(
        "INSERT INTO pipeline_configs "
        "(id, song_version_id, template_id, name, graph_json, outputs_json, knob_values_json, "
        "block_overrides_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["song_version_id"],
            row["template_id"],
            row["name"],
            row["graph_json"],
            row.get("outputs_json") or "[]",
            row.get("knob_values_json") or "{}",
            row.get("block_overrides_json") or "{}",
            now,
            now,
        ),
    )


def _insert_layer(storage: Any, row: dict[str, Any]) -> None:
    storage.db.execute(
        "INSERT INTO layers "
        '(id, song_version_id, name, layer_type, color, "order", visible, locked, '
        "parent_layer_id, source_pipeline, state_flags_json, provenance_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["song_version_id"],
            row["name"],
            row["layer_type"],
            row.get("color"),
            row["order"],
            row["visible"],
            row["locked"],
            row.get("parent_layer_id"),
            row.get("source_pipeline"),
            row.get("state_flags_json") or "{}",
            row.get("provenance_json") or "{}",
            row["created_at"],
        ),
    )


def _insert_take(storage: Any, row: dict[str, Any]) -> None:
    storage.db.execute(
        "INSERT INTO takes "
        "(id, layer_id, label, origin, is_main, is_archived, source_json, data_json, "
        "created_at, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["layer_id"],
            row["label"],
            row["origin"],
            row["is_main"],
            row["is_archived"],
            row.get("source_json"),
            row.get("data_json"),
            row["created_at"],
            row.get("notes") or "",
        ),
    )


def _insert_timeline_object(storage: Any, row: dict[str, Any]) -> None:
    storage.db.execute(
        "INSERT INTO timeline_objects "
        "(id, song_version_id, name, object_kind, main_content_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["song_version_id"],
            row["name"],
            row["object_kind"],
            row["main_content_id"],
            row["created_at"],
        ),
    )


def _insert_object_content(storage: Any, row: dict[str, Any]) -> None:
    storage.db.execute(
        "INSERT INTO object_contents "
        "(id, object_id, revision_id, content_kind, payload_json, source_ref_json, "
        "analysis_build_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["object_id"],
            row["revision_id"],
            row["content_kind"],
            row.get("payload_json") or "{}",
            row.get("source_ref_json"),
            row.get("analysis_build_json"),
            row["created_at"],
        ),
    )


def _insert_object_candidate(storage: Any, row: dict[str, Any]) -> None:
    storage.db.execute(
        "INSERT INTO object_candidates "
        "(id, object_id, content_id, label, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            row["id"],
            row["object_id"],
            row["content_id"],
            row.get("label") or "",
            row["created_at"],
        ),
    )
