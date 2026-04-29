"""
App-shell specialized-model bridge for project-derived drum classifiers.
Exists to keep the EZ one-button specialized-model action on the runtime side while Foundry does the work.
Connects app-shell project storage to Foundry training, global promotion, and targeted config refresh for pending runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from echozero.foundry.services.project_specialized_model_service import (
    ProjectSpecializedModelResult,
    ProjectSpecializedModelService,
)
from echozero.models.runtime_bundle_selection import (
    resolve_installed_binary_drum_bundles,
)
from echozero.persistence.entities import PipelineConfigRecord, SongDefaultPipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry


class SpecializedModelShell(Protocol):
    """Protocol slice for app-shell specialized-model coordination."""

    project_storage: ProjectStorage


class AppShellSpecializedModelMixin:
    """Expose the bounded EZ-side specialized drum model creation flow."""

    def create_project_specialized_drum_models(
        self: SpecializedModelShell,
    ) -> ProjectSpecializedModelResult:
        """Train and promote project-derived kick/snare models into the active global runtime set."""
        return _create_project_specialized_drum_models(self, labels=("kick", "snare"))

    def create_project_specialized_snare_model(
        self: SpecializedModelShell,
    ) -> ProjectSpecializedModelResult:
        """Train and promote only the project-derived snare model into the active global runtime set."""
        return _create_project_specialized_drum_models(self, labels=("snare",))


def _create_project_specialized_drum_models(
    shell: SpecializedModelShell,
    *,
    labels: tuple[str, ...],
) -> ProjectSpecializedModelResult:
    """Run the specialized-model flow for the requested drum labels and refresh pending configs."""
    project_ref = _build_project_ref(shell.project_storage)
    previous_defaults = _resolve_binary_model_defaults(labels=labels)
    result = _build_service(shell).create_project_specialized_drum_models(
        project_ref=project_ref,
        labels=labels,
    )
    promoted_defaults = {
        f"{promotion.label}_model_path": str(promotion.manifest_path)
        for promotion in result.promotions
    }
    _refresh_project_binary_model_configs(
        shell.project_storage,
        previous_defaults=previous_defaults,
        promoted_defaults=promoted_defaults,
    )
    return result


def _build_service(shell: SpecializedModelShell) -> ProjectSpecializedModelService:
    return ProjectSpecializedModelService(shell.project_storage.working_dir)


def _build_project_ref(project_storage: ProjectStorage) -> str:
    return f"project:{project_storage.project.id}"


def _resolve_binary_model_defaults(*, labels: tuple[str, ...]) -> dict[str, str]:
    from echozero.application.timeline.object_action_settings_service import (
        ensure_installed_models_dir,
    )

    try:
        bundles = resolve_installed_binary_drum_bundles(
            labels=labels,
            models_dir=ensure_installed_models_dir(),
        )
    except FileNotFoundError:
        return {}
    return {
        f"{label}_model_path": str(bundle.manifest_path)
        for label, bundle in bundles.items()
    }


def _refresh_project_binary_model_configs(
    project_storage: ProjectStorage,
    *,
    previous_defaults: dict[str, str],
    promoted_defaults: dict[str, str],
) -> None:
    with project_storage.transaction():
        for song in project_storage.songs.list_by_project(project_storage.project.id):
            _update_configs(
                project_storage.song_default_pipeline_configs.list_by_song(song.id),
                project_storage=project_storage,
                previous_defaults=previous_defaults,
                promoted_defaults=promoted_defaults,
                scope="song_default",
            )
            for version in project_storage.song_versions.list_by_song(song.id):
                _update_configs(
                    project_storage.pipeline_configs.list_by_version(version.id),
                    project_storage=project_storage,
                    previous_defaults=previous_defaults,
                    promoted_defaults=promoted_defaults,
                    scope="version",
                )


def _update_configs(
    configs: Iterable[PipelineConfigRecord | SongDefaultPipelineConfigRecord],
    *,
    project_storage: ProjectStorage,
    previous_defaults: dict[str, str],
    promoted_defaults: dict[str, str],
    scope: str,
) -> None:
    template_ids = {"extract_classified_drums", "extract_song_drum_events"}
    for config in configs:
        if config.template_id not in template_ids:
            continue
        template = get_registry().get(config.template_id)
        if template is None:
            continue
        updates: dict[str, str] = {}
        for key, promoted_value in promoted_defaults.items():
            current_value = str(config.knob_values.get(key, "")).strip()
            if current_value == promoted_value:
                continue
            replaceable_values = {
                "",
                previous_defaults.get(key, ""),
            }
            if current_value not in replaceable_values:
                continue
            updates[key] = promoted_value
        if not updates:
            continue
        updated = config.with_knob_values(updates, knob_metadata=template.knobs)
        if scope == "song_default":
            project_storage.song_default_pipeline_configs.update(
                replace(updated, song_id=config.song_id)
            )
        else:
            project_storage.pipeline_configs.update(
                replace(updated, song_version_id=config.song_version_id)
            )
