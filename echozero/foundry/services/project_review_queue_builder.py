"""ProjectReviewQueueBuilder builds Foundry review items from canonical EZ project data.
Exists because phone review must work from persisted project truth, not only manual imports.
Connects project archives or working dirs to deterministic review queues with durable provenance.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echozero.domain.types import Event as DomainEvent
from echozero.domain.types import EventData
from echozero.foundry.domain.review import ReviewItem, ReviewPolarity
from echozero.persistence.archive import unpack_ez
from echozero.persistence.entities import LayerRecord, ProjectRecord, SongRecord, SongVersionRecord
from echozero.persistence.repositories import (
    LayerRepository,
    ProjectRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
)

from .review_audio_clip_service import ReviewAudioClipService
from .review_queue_filters import select_review_items


@dataclass(slots=True)
class ProjectReviewQueue:
    """Deterministic review queue materialized from one EZ project snapshot."""

    project_name: str
    project_ref: str
    source_ref: str
    items: list[ReviewItem]
    metadata: dict[str, Any]


@dataclass(slots=True)
class _VersionScope:
    song: SongRecord
    version: SongVersionRecord


class ProjectReviewQueueBuilder:
    """Builds deterministic Foundry review queues from canonical EZ project state."""

    def __init__(
        self,
        root: Path,
        clip_service: ReviewAudioClipService | None = None,
    ) -> None:
        self._root = Path(root)
        self._clip_service = clip_service or ReviewAudioClipService()

    def build_queue(
        self,
        project_path: str | Path,
        *,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
    ) -> ProjectReviewQueue:
        """Build a persisted-review queue from one EZ project archive or working dir."""
        source_path = Path(project_path).expanduser().resolve()
        normalized_review_mode = _normalize_review_mode(
            review_mode,
            questionable_score_threshold=questionable_score_threshold,
        )
        cache_dir = self._project_cache_dir(source_path)
        working_dir = self._prepare_project_snapshot(source_path)
        connection = sqlite3.connect(
            f"file:{(working_dir / 'project.db').resolve()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            project_repo = ProjectRepository(connection)
            song_repo = SongRepository(connection)
            version_repo = SongVersionRepository(connection)
            layer_repo = LayerRepository(connection)
            take_repo = TakeRepository(connection)

            projects = project_repo.list()
            if not projects:
                raise ValueError(f"No project found in snapshot: {source_path}")
            project = projects[0]
            scopes = self._resolve_version_scopes(
                project=project,
                song_repo=song_repo,
                version_repo=version_repo,
                song_id=song_id,
                song_version_id=song_version_id,
            )
            items: list[ReviewItem] = []
            version_ids: list[str] = []
            song_ids: list[str] = []
            layer_ids: list[str] = []
            skipped_missing_audio: list[str] = []
            skipped_unmaterialized_clip_refs: list[str] = []
            for scope in scopes:
                song_ids.append(scope.song.id)
                version_ids.append(scope.version.id)
                layers = self._selected_layers(
                    layer_repo=layer_repo,
                    version_id=scope.version.id,
                    layer_id=layer_id,
                )
                for layer in layers:
                    layer_ids.append(layer.id)
                    main_take = take_repo.get_main(layer.id)
                    if main_take is None or not isinstance(main_take.data, EventData):
                        continue
                    audio_path = self._resolve_review_audio_path(
                        working_dir=working_dir,
                        version=scope.version,
                        take=main_take,
                    )
                    if audio_path is None or not audio_path.exists():
                        skipped_missing_audio.append(str(audio_path or layer.id))
                        continue
                    layer_items, layer_skipped_clip_refs = self._build_layer_items(
                        project=project,
                        song=scope.song,
                        version=scope.version,
                        layer=layer,
                        take=main_take,
                        source_audio_path=audio_path,
                        clip_cache_dir=cache_dir / "clips",
                        polarity=polarity,
                    )
                    items.extend(layer_items)
                    skipped_unmaterialized_clip_refs.extend(layer_skipped_clip_refs)
            total_item_count = len(items)
            items = select_review_items(
                items,
                review_mode=normalized_review_mode,
                questionable_score_threshold=questionable_score_threshold,
                item_limit=item_limit,
            )
            if not items:
                raise ValueError(
                    f"No reviewable project items found in scope for {source_path}"
                )
            deduped_song_ids = list(dict.fromkeys(song_ids))
            deduped_version_ids = list(dict.fromkeys(version_ids))
            deduped_layer_ids = list(dict.fromkeys(layer_ids))
            metadata = {
                "import_format": "project",
                "queue_source_kind": "ez_project",
                "project_ref": self._build_ref("project", project.id),
                "song_ids": deduped_song_ids,
                "version_ids": deduped_version_ids,
                "layer_ids": deduped_layer_ids,
                "polarity": polarity.value,
                "review_mode": normalized_review_mode,
                "total_item_count": total_item_count,
                "selected_item_count": len(items),
                "questionable_score_threshold": questionable_score_threshold,
                "item_limit": item_limit,
                "skipped_missing_audio_count": len(skipped_missing_audio),
                "skipped_missing_audio_refs": skipped_missing_audio,
                "skipped_unmaterialized_clip_count": len(skipped_unmaterialized_clip_refs),
                "skipped_unmaterialized_clip_refs": skipped_unmaterialized_clip_refs,
            }
            return ProjectReviewQueue(
                project_name=project.name,
                project_ref=self._build_ref("project", project.id),
                source_ref=str(source_path),
                items=items,
                metadata=metadata,
            )
        finally:
            connection.close()

    def _prepare_project_snapshot(self, source_path: Path) -> Path:
        if source_path.is_file() and source_path.name == "project.db":
            return source_path.parent
        if source_path.is_dir() and (source_path / "project.db").exists():
            return source_path
        if source_path.is_file() and source_path.suffix.lower() == ".ez":
            cache_dir = self._project_cache_dir(source_path)
            if not (cache_dir / "project.db").exists():
                unpack_ez(source_path, cache_dir)
            return cache_dir
        raise ValueError(
            f"Project source must be an .ez archive, project.db, or working dir: {source_path}"
        )

    def _project_cache_dir(self, source_path: Path) -> Path:
        cache_source = source_path
        if source_path.is_dir() and (source_path / "project.db").exists():
            cache_source = source_path / "project.db"
        elif source_path.is_file() and source_path.name == "project.db":
            cache_source = source_path
        stat = cache_source.stat()
        cache_key = hashlib.sha1(
            f"{cache_source.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = self._root / "foundry" / "cache" / "review_projects" / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _resolve_version_scopes(
        self,
        *,
        project: ProjectRecord,
        song_repo: SongRepository,
        version_repo: SongVersionRepository,
        song_id: str | None,
        song_version_id: str | None,
    ) -> list[_VersionScope]:
        if song_version_id is not None:
            version = version_repo.get(song_version_id)
            if version is None:
                raise ValueError(f"SongVersionRecord not found: {song_version_id}")
            song = song_repo.get(version.song_id)
            if song is None:
                raise ValueError(f"SongRecord not found for version: {song_version_id}")
            if song_id is not None and song.id != song_id:
                raise ValueError(
                    f"SongVersionRecord '{song_version_id}' does not belong to song '{song_id}'"
                )
            return [_VersionScope(song=song, version=version)]

        songs = song_repo.list_by_project(project.id)
        if song_id is not None:
            songs = [song for song in songs if song.id == song_id]
            if not songs:
                raise ValueError(f"SongRecord not found: {song_id}")
        scopes: list[_VersionScope] = []
        for song in sorted(songs, key=lambda candidate: (candidate.order, candidate.title, candidate.id)):
            if song.active_version_id is None:
                continue
            version = version_repo.get(song.active_version_id)
            if version is None:
                continue
            scopes.append(_VersionScope(song=song, version=version))
        return scopes

    def _selected_layers(
        self,
        *,
        layer_repo: LayerRepository,
        version_id: str,
        layer_id: str | None,
    ) -> list[LayerRecord]:
        layers = layer_repo.list_by_version(version_id)
        if layer_id is not None:
            filtered = [layer for layer in layers if layer.id == layer_id]
            return sorted(filtered, key=lambda layer: (layer.order, layer.name, layer.id))
        filtered = [layer for layer in layers if layer.layer_type == "analysis"]
        return sorted(filtered, key=lambda layer: (layer.order, layer.name, layer.id))

    def _build_layer_items(
        self,
        *,
        project: ProjectRecord,
        song: SongRecord,
        version: SongVersionRecord,
        layer: LayerRecord,
        take: Any,
        source_audio_path: Path,
        clip_cache_dir: Path,
        polarity: ReviewPolarity,
    ) -> tuple[list[ReviewItem], list[str]]:
        domain_layers = tuple(take.data.layers)
        projected_events: list[tuple[str, int, DomainEvent]] = []
        for domain_layer in domain_layers:
            for event_index, event in enumerate(domain_layer.events):
                projected_events.append((str(domain_layer.id), event_index, event))
        projected_events.sort(
            key=lambda item: (item[2].time, item[2].duration, str(item[2].id), item[0], item[1])
        )
        items: list[ReviewItem] = []
        skipped_clip_refs: list[str] = []
        for domain_layer_id, _event_index, event in projected_events:
            item_id = self._build_item_id(
                project_id=project.id,
                song_id=song.id,
                version_id=version.id,
                layer_id=layer.id,
                take_id=str(take.id),
                domain_layer_id=domain_layer_id,
                event_id=str(event.id),
            )
            clip_path = self._clip_service.materialize_event_clip(
                source_audio_path=source_audio_path,
                clip_cache_dir=clip_cache_dir,
                clip_stem=item_id,
                start_seconds=float(event.time),
                end_seconds=float(event.time + event.duration),
            )
            if clip_path is None:
                skipped_clip_refs.append(self._build_ref("event", event.id))
                continue
            items.append(
                ReviewItem(
                    item_id=item_id,
                    audio_path=str(clip_path),
                    predicted_label=self._predicted_label(event, fallback_label=layer.name),
                    target_class=layer.name,
                    polarity=polarity,
                    score=self._score(event),
                    source_provenance=self._build_source_provenance(
                        project=project,
                        song=song,
                        version=version,
                        layer=layer,
                        take=take,
                        audio_path=clip_path,
                        source_audio_path=source_audio_path,
                        domain_layer_id=domain_layer_id,
                        event=event,
                    ),
                )
            )
        return items, skipped_clip_refs

    def _build_source_provenance(
        self,
        *,
        project: ProjectRecord,
        song: SongRecord,
        version: SongVersionRecord,
        layer: LayerRecord,
        take: Any,
        audio_path: Path,
        source_audio_path: Path,
        domain_layer_id: str,
        event: DomainEvent,
    ) -> dict[str, Any]:
        event_ref = self._build_ref("event", event.id)
        source_event_ref = self._build_ref("event", event.source_event_id or event.id)
        return {
            "kind": "ez_project_review_queue",
            "project_ref": self._build_ref("project", project.id),
            "project_name": project.name,
            "song_ref": self._build_ref("song", song.id),
            "song_title": song.title,
            "version_ref": self._build_ref("version", version.id),
            "version_label": version.label,
            "layer_ref": self._build_ref("layer", layer.id),
            "layer_name": layer.name,
            "take_ref": self._build_ref("take", take.id),
            "domain_layer_ref": self._build_ref("domain_layer", domain_layer_id),
            "event_ref": event_ref,
            "source_event_ref": source_event_ref,
            "model_ref": self._model_ref(layer=layer, take=take, event=event),
            "audio_ref": str(audio_path),
            "source_audio_ref": str(source_audio_path),
            "original_start_ms": float(event.time) * 1000.0,
            "original_end_ms": float(event.time + event.duration) * 1000.0,
        }

    @staticmethod
    def _build_item_id(
        *,
        project_id: str,
        song_id: str,
        version_id: str,
        layer_id: str,
        take_id: str,
        domain_layer_id: str,
        event_id: str,
    ) -> str:
        digest = hashlib.sha1(
            "|".join(
                (
                    project_id,
                    song_id,
                    version_id,
                    layer_id,
                    take_id,
                    domain_layer_id,
                    event_id,
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"ri_{digest}"

    @staticmethod
    def _resolve_review_audio_path(
        *,
        working_dir: Path,
        version: SongVersionRecord,
        take: Any,
    ) -> Path | None:
        candidate = None
        source = getattr(take, "source", None)
        if source is not None:
            settings_snapshot = getattr(source, "settings_snapshot", {}) or {}
            source_audio_path = settings_snapshot.get("source_audio_path")
            if source_audio_path is not None:
                candidate = str(source_audio_path).strip() or None
        if candidate is None:
            candidate = str(version.audio_file).strip() or None
        if candidate is None:
            return None
        raw_path = Path(candidate)
        if raw_path.is_absolute():
            return raw_path
        return (working_dir / raw_path).resolve()

    @staticmethod
    def _predicted_label(event: DomainEvent, *, fallback_label: str) -> str:
        classifications = event.classifications if isinstance(event.classifications, dict) else {}
        for key in ("class", "label"):
            value = classifications.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        fallback = str(fallback_label).strip()
        return fallback or "event"

    @staticmethod
    def _score(event: DomainEvent) -> float | None:
        candidates = []
        if isinstance(event.classifications, dict):
            candidates.extend(
                event.classifications.get(key) for key in ("confidence", "score")
            )
        if isinstance(event.metadata, dict):
            candidates.extend(event.metadata.get(key) for key in ("confidence", "score"))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _model_ref(cls, *, layer: LayerRecord, take: Any, event: DomainEvent) -> str | None:
        event_payloads = []
        if isinstance(event.metadata, dict):
            event_payloads.append(event.metadata.get("model_artifact"))
        if isinstance(event.classifications, dict):
            event_payloads.append(event.classifications.get("model_artifact"))
        for payload in event_payloads:
            artifact_ref = cls._artifact_identity(payload)
            if artifact_ref is not None:
                return artifact_ref
        source = getattr(take, "source", None)
        analysis_build = getattr(source, "analysis_build", None) if source is not None else None
        for key in ("build_id", "execution_id", "pipeline_id"):
            value = getattr(analysis_build, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        provenance = dict(getattr(layer, "provenance", {}) or {})
        build_payload = provenance.get("analysis_build")
        if isinstance(build_payload, dict):
            for key in ("build_id", "execution_id", "pipeline_id"):
                value = build_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _artifact_identity(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        artifact_identity = payload.get("artifactIdentity")
        if isinstance(artifact_identity, dict):
            artifact_id = artifact_identity.get("artifactId")
            if isinstance(artifact_id, str) and artifact_id.strip():
                return artifact_id.strip()
        bundle_ref = payload.get("bundleRef")
        if isinstance(bundle_ref, str) and bundle_ref.strip():
            return bundle_ref.strip()
        return None

    @staticmethod
    def _build_ref(kind: str, raw_value: Any) -> str:
        text = str(raw_value).strip()
        if text.startswith(f"{kind}:"):
            return text
        return f"{kind}:{text}"


def _normalize_review_mode(
    review_mode: str | None,
    *,
    questionable_score_threshold: float | None,
) -> str:
    text = str(review_mode).strip() if review_mode is not None else ""
    if not text:
        return "questionables" if questionable_score_threshold is not None else "all_events"
    if text not in {"all_events", "questionables"}:
        raise ValueError("review_mode must be 'all_events' or 'questionables'")
    return text
