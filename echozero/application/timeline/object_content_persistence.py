"""Persistence helpers for timeline object/content truth.
Exists because source refs must resolve to real object/content rows, never path-made IDs.
Connects ProjectStorage rows to runtime timeline projection, pipeline output, and storage sync.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from echozero.application.shared.ids import (
    ObjectContentId,
    ObjectRevisionId,
    TimelineObjectId,
)
from echozero.application.timeline.object_content import SourceRef
from echozero.domain.types import AudioData, EventData
from echozero.errors import PersistenceError, ValidationError
from echozero.persistence.entities import (
    ObjectCandidateRecord,
    ObjectContentRecord,
    TimelineObjectRecord,
)
from echozero.takes import Take as PersistedTake

if TYPE_CHECKING:
    from echozero.persistence.session import ProjectStorage


def object_id_for_layer(layer_id: object) -> str:
    """Return the persisted object id for a storage-backed layer."""

    return f"object_{layer_id}"


def content_id_for_take(take_id: object) -> str:
    """Return the persisted content id for a storage-backed take."""

    return f"content_{take_id}"


def revision_id_for_take(take_id: object) -> str:
    """Return the persisted revision id for a storage-backed take."""

    return f"revision_{take_id}"


def imported_song_object_id(version_id: object) -> str:
    """Return the imported-song object id for a song version."""

    return f"object_song_{version_id}"


def imported_song_content_id(version_id: object) -> str:
    """Return the imported-song content id for a song version."""

    return f"content_song_audio_{version_id}"


def imported_song_revision_id(audio_hash: object) -> str:
    """Return the imported-song revision id for an audio hash."""

    return f"revision_song_audio_{audio_hash}"


def require_imported_song_ref(
    session: ProjectStorage,
    *,
    song_version_id: str,
) -> SourceRef:
    """Return the persisted imported-song source ref, raising when rows are missing."""

    version = session.song_versions.get(song_version_id)
    if version is None:
        raise PersistenceError(f"Song version not found for object/content projection: {song_version_id}")
    return require_source_ref(
        session,
        object_id=imported_song_object_id(song_version_id),
        content_id=imported_song_content_id(song_version_id),
        revision_id=imported_song_revision_id(version.audio_hash),
        role="imported_song_audio",
        locator=str(resolve_project_audio_path(session, version.audio_file)),
    )


def require_source_ref(
    session: ProjectStorage,
    *,
    object_id: str,
    content_id: str,
    revision_id: str,
    role: str,
    locator: str | None = None,
) -> SourceRef:
    """Build a source ref only when the target object/content rows exist."""

    object_record = session.timeline_objects.get(object_id)
    content_record = session.object_contents.get(content_id)
    if object_record is None or content_record is None:
        raise PersistenceError(
            "Object/content source ref target is missing: "
            f"object_id={object_id!r}, content_id={content_id!r}."
        )
    if content_record.object_id != object_record.id:
        raise PersistenceError(
            "Object/content source ref target is inconsistent: "
            f"content {content_id!r} belongs to {content_record.object_id!r}, "
            f"not {object_id!r}."
        )
    if content_record.revision_id != revision_id:
        raise PersistenceError(
            "Object/content source ref revision is stale: "
            f"content_id={content_id!r}, expected={revision_id!r}, "
            f"actual={content_record.revision_id!r}."
        )
    return SourceRef(
        object_id=TimelineObjectId(object_id),
        content_id=ObjectContentId(content_id),
        revision_id=ObjectRevisionId(revision_id),
        role=role,
        locator=locator,
    )


def source_ref_from_record(
    record: ObjectContentRecord,
    *,
    role: str = "source",
    locator: str | None = None,
) -> SourceRef:
    """Build a typed source ref from an already loaded content row."""

    return SourceRef(
        object_id=TimelineObjectId(record.object_id),
        content_id=ObjectContentId(record.id),
        revision_id=ObjectRevisionId(record.revision_id),
        role=role,
        locator=locator,
    )


def resolve_audio_source_ref(
    session: ProjectStorage,
    *,
    locator: str,
    role: str = "audio_source",
) -> SourceRef:
    """Resolve an audio locator to an existing object content row."""

    normalized = str(locator or "").strip()
    if not normalized:
        raise ValidationError("Cannot persist object content with an empty audio source ref.")
    match = find_audio_content_by_locator(session, normalized)
    if match is None:
        raise ValidationError(
            "Cannot persist object content because the audio source does not resolve "
            f"to a persisted object/content row: {normalized}"
        )
    return source_ref_from_record(
        match,
        role=role,
        locator=str(resolve_project_audio_path(session, normalized)),
    )


def find_audio_content_by_locator(
    session: ProjectStorage,
    locator: str,
) -> ObjectContentRecord | None:
    """Find persisted audio content whose payload path matches a local locator."""

    requested = _path_candidates(session, locator)
    for object_record in _all_timeline_objects(session):
        for content in session.object_contents.list_by_object(object_record.id):
            audio_file = content.payload.get("audio_file")
            if audio_file is None:
                continue
            if requested & _path_candidates(session, str(audio_file)):
                return content
    return None


def persist_take_object_content(
    session: ProjectStorage,
    *,
    song_version_id: str,
    layer_record_id: str,
    layer_name: str,
    take: PersistedTake,
    content_kind: str,
    source_audio_path: str | None,
    analysis_build: dict[str, Any] | None,
    is_main: bool,
) -> None:
    """Persist object/content truth for one storage-backed take."""

    object_id = object_id_for_layer(layer_record_id)
    content_id = content_id_for_take(take.id)
    revision_id = revision_id_for_take(take.id)
    now = take.created_at
    source_ref = None
    if source_audio_path is not None and str(source_audio_path).strip():
        source_ref = resolve_audio_source_ref(
            session,
            locator=str(source_audio_path).strip(),
        ).to_dict()

    _upsert_timeline_object(
        session,
        TimelineObjectRecord(
            id=object_id,
            song_version_id=song_version_id,
            name=layer_name,
            object_kind=content_kind,
            main_content_id=content_id if is_main else _existing_main_content_id(session, object_id, content_id),
            created_at=now,
        ),
    )
    _upsert_object_content(
        session,
        ObjectContentRecord(
            id=content_id,
            object_id=object_id,
            revision_id=revision_id,
            content_kind=content_kind,
            payload=_payload_for_take(take),
            source_ref=source_ref,
            analysis_build=analysis_build,
            created_at=now,
        ),
    )
    if is_main:
        current = session.timeline_objects.get(object_id)
        if current is not None and current.main_content_id != content_id:
            session.timeline_objects.update(replace(current, main_content_id=content_id))


def persist_generated_audio_content(
    session: ProjectStorage,
    *,
    song_version_id: str,
    output_name: str,
    audio_file: str,
    analysis_build_id: str,
    source_audio_path: str | None,
    analysis_build: dict[str, Any],
    created_at: datetime,
) -> ObjectContentRecord:
    """Persist generated audio content even when it is not projected as a layer."""

    stable_name = _stable_id_part(output_name)
    stable_build = _stable_id_part(analysis_build_id)
    object_id = f"object_generated_audio_{stable_build}_{stable_name}"
    content_id = f"content_generated_audio_{stable_build}_{stable_name}"
    source_ref = None
    if source_audio_path is not None and str(source_audio_path).strip():
        source_ref = resolve_audio_source_ref(
            session,
            locator=str(source_audio_path).strip(),
        ).to_dict()
    _upsert_timeline_object(
        session,
        TimelineObjectRecord(
            id=object_id,
            song_version_id=song_version_id,
            name=output_name,
            object_kind="generated_audio",
            main_content_id=content_id,
            created_at=created_at,
        ),
    )
    content_record = ObjectContentRecord(
        id=content_id,
        object_id=object_id,
        revision_id=f"revision_{content_id}",
        content_kind="generated_audio",
        payload={"audio_file": audio_file},
        source_ref=source_ref,
        analysis_build=analysis_build,
        created_at=created_at,
    )
    _upsert_object_content(session, content_record)
    return content_record


def sync_layer_object_content(
    session: ProjectStorage,
    *,
    song_version_id: str,
    layer_id: str,
    layer_name: str,
    content_kind: str,
    takes: list[PersistedTake],
) -> None:
    """Make object/content rows match the current persisted takes for one layer."""

    if not takes:
        object_record = session.timeline_objects.get(object_id_for_layer(layer_id))
        if object_record is not None:
            session.timeline_objects.delete(object_record.id)
        return

    object_id = object_id_for_layer(layer_id)
    main_take = next((take for take in takes if take.is_main), takes[0])
    main_content_id = content_id_for_take(main_take.id)
    _upsert_timeline_object(
        session,
        TimelineObjectRecord(
            id=object_id,
            song_version_id=song_version_id,
            name=layer_name,
            object_kind=content_kind,
            main_content_id=main_content_id,
            created_at=main_take.created_at,
        ),
    )

    active_content_ids: set[str] = set()
    for take in takes:
        active_content_ids.add(content_id_for_take(take.id))
        source_ref = _source_ref_from_take(session, take)
        _upsert_object_content(
            session,
            ObjectContentRecord(
                id=content_id_for_take(take.id),
                object_id=object_id,
                revision_id=revision_id_for_take(take.id),
                content_kind=content_kind,
                payload=_payload_for_take(take),
                source_ref=source_ref.to_dict() if source_ref is not None else None,
                analysis_build=(
                    take.source.analysis_build.to_dict()
                    if take.source is not None and take.source.analysis_build is not None
                    else None
                ),
                created_at=take.created_at,
            ),
        )

    for content in session.object_contents.list_by_object(object_id):
        if content.id not in active_content_ids:
            session.object_contents.delete(content.id)

    session.object_candidates.delete_by_object(object_id)
    for take in takes:
        if take.id == main_take.id:
            continue
        session.object_candidates.create(
            ObjectCandidateRecord(
                id=f"candidate_{take.id}",
                object_id=object_id,
                content_id=content_id_for_take(take.id),
                label=take.label,
                created_at=take.created_at,
            )
        )


def load_layer_object_content(
    session: ProjectStorage,
    *,
    layer_record_id: str,
) -> tuple[TimelineObjectRecord, dict[str, ObjectContentRecord]]:
    """Load persisted object/content rows for a layer or fail loudly."""

    object_id = object_id_for_layer(layer_record_id)
    object_record = session.timeline_objects.get(object_id)
    if object_record is None:
        raise PersistenceError(f"Missing timeline object row for layer {layer_record_id!r}.")
    contents = {
        content.id: content for content in session.object_contents.list_by_object(object_id)
    }
    if object_record.main_content_id not in contents:
        raise PersistenceError(
            f"Timeline object {object_id!r} points at missing main content "
            f"{object_record.main_content_id!r}."
        )
    return object_record, contents


def source_ref_from_payload(payload: dict[str, Any] | None) -> SourceRef | None:
    """Deserialize a persisted source_ref payload."""

    if not payload:
        return None
    return SourceRef.from_dict(dict(payload))


def resolve_project_audio_path(session: ProjectStorage, audio_file: str) -> Path:
    """Resolve a stored or absolute audio path against a project working dir."""

    raw_path = Path(audio_file)
    if raw_path.is_absolute():
        return raw_path
    return (session.working_dir / raw_path).resolve()


def _source_ref_from_take(
    session: ProjectStorage,
    take: PersistedTake,
) -> SourceRef | None:
    if take.source is None:
        return None
    settings_snapshot = take.source.settings_snapshot or {}
    source_audio_path = settings_snapshot.get("source_audio_path")
    if source_audio_path in (None, ""):
        return None
    return resolve_audio_source_ref(session, locator=str(source_audio_path))


def _payload_for_take(take: PersistedTake) -> dict[str, Any]:
    payload: dict[str, Any] = {"take_id": take.id}
    if isinstance(take.data, AudioData):
        payload["audio_file"] = take.data.file_path
    elif isinstance(take.data, EventData):
        payload["event_layer_count"] = len(take.data.layers)
        payload["event_count"] = sum(len(layer.events) for layer in take.data.layers)
    return payload


def _upsert_timeline_object(
    session: ProjectStorage,
    record: TimelineObjectRecord,
) -> None:
    existing = session.timeline_objects.get(record.id)
    if existing is None:
        session.timeline_objects.create(record)
        return
    session.timeline_objects.update(
        replace(
            existing,
            name=record.name,
            object_kind=record.object_kind,
            main_content_id=record.main_content_id,
        )
    )


def _upsert_object_content(
    session: ProjectStorage,
    record: ObjectContentRecord,
) -> None:
    if session.object_contents.get(record.id) is None:
        session.object_contents.create(record)
    else:
        session.object_contents.update(record)


def _existing_main_content_id(
    session: ProjectStorage,
    object_id: str,
    fallback_content_id: str,
) -> str:
    existing = session.timeline_objects.get(object_id)
    return existing.main_content_id if existing is not None else fallback_content_id


def _all_timeline_objects(session: ProjectStorage) -> list[TimelineObjectRecord]:
    rows = session.db.execute(
        "SELECT DISTINCT song_version_id FROM timeline_objects ORDER BY song_version_id"
    ).fetchall()
    objects: list[TimelineObjectRecord] = []
    for row in rows:
        objects.extend(session.timeline_objects.list_by_version(str(row["song_version_id"])))
    return objects


def _path_candidates(session: ProjectStorage, value: str) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    raw_path = Path(text)
    resolved = raw_path if raw_path.is_absolute() else (session.working_dir / raw_path)
    candidates = {text, raw_path.as_posix(), str(raw_path), str(resolved.resolve())}
    try:
        candidates.add(resolved.resolve().relative_to(session.working_dir.resolve()).as_posix())
    except ValueError:
        pass
    return {candidate for candidate in candidates if candidate}


def _stable_id_part(value: str) -> str:
    return (
        str(value or "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .strip("_")
    ) or "unknown"


def utc_now() -> datetime:
    """Return current UTC time for object/content rows."""

    return datetime.now(timezone.utc)
