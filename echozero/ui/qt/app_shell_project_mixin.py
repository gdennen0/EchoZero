"""Project lifecycle facade mixin for the Qt app shell runtime.
Exists to isolate song and project lifecycle methods from the orchestration root.
Connects AppShellRuntime to project persistence and baseline timeline reload helpers.
"""

from __future__ import annotations

from pathlib import Path

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.foundry.domain.review import ReviewPolarity
from echozero.ui.qt.app_shell_project_lifecycle import (
    ProjectLifecycleShell,
    add_song_from_path,
    add_song_version,
    delete_song,
    delete_song_version,
    list_song_version_transfer_layers,
    list_ma3_timecode_pools,
    move_song,
    new_project,
    open_project,
    refresh_from_storage,
    reorder_songs,
    save_project,
    save_project_as,
    select_song,
    set_song_version_ma3_timecode_pool,
    switch_song_version,
)
from echozero.ui.qt.app_shell_project_review import (
    ProjectReviewDatasetPaths,
    ProjectReviewLaunch,
    create_project_review_session,
    get_latest_project_review_dataset_version,
    latest_project_review_dataset_artifact_path,
    latest_project_review_dataset_folder,
    list_project_review_dataset_versions,
    open_project_review_session,
    reload_phone_review_status,
)


class AppShellProjectMixin:
    def new_project(self: ProjectLifecycleShell, name: str = "EchoZero Project") -> None:
        new_project(self, name=name)

    def save_project_as(self: ProjectLifecycleShell, path: str | Path) -> Path:
        return save_project_as(self, path)

    def save_project(self: ProjectLifecycleShell) -> Path:
        return save_project(self)

    def open_project(self: ProjectLifecycleShell, path: str | Path) -> None:
        open_project(self, path)

    def add_song_from_path(
        self: ProjectLifecycleShell,
        title: str,
        audio_path: str | Path,
        *,
        run_import_pipeline: bool | None = None,
        import_pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> TimelinePresentation:
        return add_song_from_path(
            self,
            title,
            audio_path,
            run_import_pipeline=run_import_pipeline,
            import_pipeline_action_ids=import_pipeline_action_ids,
        )

    def select_song(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
    ) -> TimelinePresentation:
        return select_song(self, song_id)

    def switch_song_version(
        self: ProjectLifecycleShell,
        song_version_id: str | SongVersionId,
    ) -> TimelinePresentation:
        return switch_song_version(self, song_version_id)

    def add_song_version(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
        audio_path: str | Path,
        *,
        label: str | None = None,
        activate: bool = True,
        transfer_layers: bool = False,
        transfer_layer_ids: list[str] | None = None,
        run_import_pipeline: bool | None = None,
        import_pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> TimelinePresentation:
        return add_song_version(
            self,
            song_id,
            audio_path,
            label=label,
            activate=activate,
            transfer_layers=transfer_layers,
            transfer_layer_ids=transfer_layer_ids,
            run_import_pipeline=run_import_pipeline,
            import_pipeline_action_ids=import_pipeline_action_ids,
        )

    def list_song_version_transfer_layers(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
    ) -> list[tuple[str, str]]:
        return list_song_version_transfer_layers(self, song_id)

    def reorder_songs(
        self: ProjectLifecycleShell,
        song_ids: list[str],
    ) -> TimelinePresentation:
        return reorder_songs(self, song_ids)

    def move_song(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
        *,
        steps: int,
    ) -> TimelinePresentation:
        return move_song(self, song_id, steps=steps)

    def delete_song(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
    ) -> TimelinePresentation:
        return delete_song(self, song_id)

    def delete_song_version(
        self: ProjectLifecycleShell,
        song_version_id: str | SongVersionId,
    ) -> TimelinePresentation:
        return delete_song_version(self, song_version_id)

    def list_ma3_timecode_pools(
        self: ProjectLifecycleShell,
    ) -> list[tuple[int, str | None]]:
        return list_ma3_timecode_pools(self)

    def set_song_version_ma3_timecode_pool(
        self: ProjectLifecycleShell,
        song_version_id: str | SongVersionId,
        timecode_pool_no: int | None,
    ) -> TimelinePresentation:
        return set_song_version_ma3_timecode_pool(
            self,
            song_version_id,
            timecode_pool_no,
        )

    def _refresh_from_storage(
        self: ProjectLifecycleShell,
        *,
        active_song_id: SongId | None = None,
        active_song_version_id: SongVersionId | None = None,
    ) -> None:
        refresh_from_storage(
            self,
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )

    def create_project_review_session(
        self: ProjectLifecycleShell,
        *,
        name: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
    ):
        return create_project_review_session(
            self,
            name=name,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            polarity=polarity,
            review_mode=review_mode,
            questionable_score_threshold=questionable_score_threshold,
            item_limit=item_limit,
        )

    def open_project_review_session(
        self: ProjectLifecycleShell,
        *,
        name: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
    ) -> ProjectReviewLaunch:
        return open_project_review_session(
            self,
            name=name,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            polarity=polarity,
            review_mode=review_mode,
            questionable_score_threshold=questionable_score_threshold,
            item_limit=item_limit,
        )

    def reload_phone_review_status(
        self: ProjectLifecycleShell,
    ) -> ProjectReviewLaunch:
        return reload_phone_review_status(self)

    def list_project_review_dataset_versions(
        self: ProjectLifecycleShell,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> list[ProjectReviewDatasetPaths]:
        return list_project_review_dataset_versions(
            self,
            queue_source_kind=queue_source_kind,
        )

    def get_latest_project_review_dataset_version(
        self: ProjectLifecycleShell,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> ProjectReviewDatasetPaths | None:
        return get_latest_project_review_dataset_version(
            self,
            queue_source_kind=queue_source_kind,
        )

    def latest_project_review_dataset_folder(
        self: ProjectLifecycleShell,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> Path:
        return latest_project_review_dataset_folder(
            self,
            queue_source_kind=queue_source_kind,
        )

    def latest_project_review_dataset_artifact_path(
        self: ProjectLifecycleShell,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> Path:
        return latest_project_review_dataset_artifact_path(
            self,
            queue_source_kind=queue_source_kind,
        )
