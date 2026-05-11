"""Object-action settings persistence helpers.
Exists to isolate scoped-config persistence and generated-layer provenance updates from the service root.
Connects project storage records and dirty tracking to the public execution service.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from echozero.application.session.models import Session
from echozero.application.timeline.object_action_scoped_config import (
    ObjectActionConfigRecord,
    ScopedConfigShell,
    load_scoped_action_config as _load_scoped_action_config,
)
from echozero.application.timeline.object_action_scoped_config import (
    persist_object_action_params as _persist_object_action_params,
)
from echozero.application.timeline.object_action_scoped_config import (
    require_object_action_config as _require_object_action_config,
)
from echozero.application.timeline.object_action_scoped_config import (
    store_scoped_action_config as _store_scoped_action_config,
)
from echozero.application.timeline.object_actions.descriptors import ActionDescriptor
from echozero.persistence.session import ProjectStorage
from echozero.services.orchestrator import AnalysisResult


class ObjectActionSettingsPersistenceShell(ScopedConfigShell, Protocol):
    @property
    def project_storage(self) -> ProjectStorage: ...

    @property
    def session(self) -> Session: ...

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]: ...


class ObjectActionSettingsPersistenceMixin:
    """Provides scoped-config storage and generated-layer provenance updates."""

    def _mark_scope_persist_dirty(
        self: ObjectActionSettingsPersistenceShell,
        *,
        scope: str,
        config: ObjectActionConfigRecord,
    ) -> None:
        if scope == "song_default":
            song_id = getattr(config, "song_id", None)
            if song_id:
                self.project_storage.dirty_tracker.mark_dirty(str(song_id))
            return
        song_version_id = getattr(config, "song_version_id", None)
        if song_version_id:
            self.project_storage.dirty_tracker.mark_dirty(str(song_version_id))

    def persist_generated_source_layer_id(
        self: ObjectActionSettingsPersistenceShell,
        *,
        analysis_result: AnalysisResult,
        source_layer_id: object | None,
    ) -> None:
        if source_layer_id is None:
            return
        persisted_source_layer_id = str(source_layer_id)
        persisted_parent_layer_id = (
            persisted_source_layer_id
            if self.project_storage.layers.get(persisted_source_layer_id) is not None
            else None
        )
        updated_version_ids: set[str] = set()
        with self.project_storage.transaction():
            for generated_layer_id in analysis_result.layer_ids:
                layer_record = self.project_storage.layers.get(generated_layer_id)
                if layer_record is None:
                    continue
                provenance = dict(layer_record.provenance)
                if (
                    provenance.get("source_layer_id") == persisted_source_layer_id
                    and layer_record.parent_layer_id == persisted_parent_layer_id
                ):
                    continue
                provenance["source_layer_id"] = persisted_source_layer_id
                self.project_storage.layers.update(
                    replace(
                        layer_record,
                        parent_layer_id=persisted_parent_layer_id,
                        provenance=provenance,
                    )
                )
                updated_version_ids.add(str(layer_record.song_version_id))
        for song_version_id in updated_version_ids:
            self.project_storage.dirty_tracker.mark_dirty(song_version_id)

    def _load_scoped_action_config(
        self: ObjectActionSettingsPersistenceShell,
        template_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ) -> ObjectActionConfigRecord:
        return _load_scoped_action_config(
            self,
            template_id,
            scope=scope,
            song_id=song_id,
            song_version_id=song_version_id,
        )

    def _require_object_action_config(
        self: ObjectActionSettingsPersistenceShell,
        template_id: str,
        *,
        scope: str = "version",
    ) -> ObjectActionConfigRecord:
        return _require_object_action_config(self, template_id, scope=scope)

    def _persist_object_action_params(
        self: ObjectActionSettingsPersistenceShell,
        config: ObjectActionConfigRecord,
        *,
        action_id: str,
        params: dict[str, object],
        scope: str = "version",
    ) -> ObjectActionConfigRecord:
        _workflow, pipeline_template_id = self._require_workflow(action_id)
        return _persist_object_action_params(
            self,
            config,
            action_id=action_id,
            pipeline_template_id=pipeline_template_id,
            params=params,
            scope=scope,
        )

    def _store_scoped_action_config(
        self: ObjectActionSettingsPersistenceShell,
        config: ObjectActionConfigRecord,
        *,
        scope: str,
    ) -> None:
        _store_scoped_action_config(self, config, scope=scope)

    def _require_active_song_id(
        self: ObjectActionSettingsPersistenceShell,
        action_name: str,
    ) -> str:
        if self.session.active_song_id is None:
            raise RuntimeError(f"{action_name} requires an active song.")
        return str(self.session.active_song_id)

    def _require_active_song_version_id(
        self: ObjectActionSettingsPersistenceShell,
        action_name: str,
    ) -> str:
        if self.session.active_song_version_id is None:
            raise RuntimeError(f"{action_name} requires an active song version.")
        return str(self.session.active_song_version_id)
