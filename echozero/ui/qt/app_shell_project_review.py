"""Project-review helpers for the Stage Zero app shell runtime.
Exists because open projects need one direct path into the Foundry review lane.
Connects live ProjectStorage roots to persisted review sessions and browser-ready URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from echozero.application.session.models import Session
from echozero.foundry.app import FoundryApp
from echozero.foundry.domain.review import ReviewPolarity, ReviewSession
from echozero.foundry.services.query_service import ProjectReviewDatasetVersionRef
from echozero.foundry.services.review_runtime_bridge import (
    clear_review_runtime_bridge,
    register_review_runtime_bridge,
)
from echozero.foundry.review_server_controller import (
    ReviewServerController,
)
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_timeline_review import apply_review_signal_to_runtime


class ProjectReviewShell(Protocol):
    """Minimal app-shell protocol needed by the review bridge."""

    project_storage: ProjectStorage
    session: Session
    _review_server_controller: ReviewServerController


@dataclass(slots=True)
class _AppShellReviewRuntimeBridge:
    shell: ProjectReviewShell

    def apply_review_signal(self, session, signal) -> dict[str, object] | None:
        return apply_review_signal_to_runtime(self.shell, signal)


@dataclass(slots=True, frozen=True)
class ProjectReviewLaunch:
    """One browser-ready review launch result for an EZ project batch."""

    session_id: str
    session_name: str
    item_count: int
    url: str
    desktop_url: str
    phone_url: str | None
    bind_host: str
    port: int


@dataclass(slots=True, frozen=True)
class ProjectReviewDatasetPaths:
    """Local paths for one persisted project review-dataset version."""

    dataset_id: str
    dataset_name: str
    version_id: str
    version_number: int
    sample_count: int
    folder_path: Path
    version_artifact_path: Path


def create_project_review_session(
    shell: ProjectReviewShell,
    *,
    name: str | None = None,
    song_id: str | None = None,
    song_version_id: str | None = None,
    layer_id: str | None = None,
    polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
    review_mode: str | None = None,
    questionable_score_threshold: float | None = None,
    item_limit: int | None = None,
) -> ReviewSession:
    """Create one persisted review session from the current project root."""

    root = _project_root(shell)
    register_review_runtime_bridge(root, _AppShellReviewRuntimeBridge(shell))
    publish_project_review_runtime_context(shell)
    return ReviewSessionService(root).create_project_session(
        root,
        name=name,
        song_id=song_id,
        song_version_id=song_version_id,
        layer_id=layer_id,
        polarity=polarity,
        review_mode=review_mode,
        questionable_score_threshold=questionable_score_threshold,
        item_limit=item_limit,
        application_session=_application_session_payload(shell),
    )


def open_project_review_session(
    shell: ProjectReviewShell,
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
    """Create one project-backed review session and return a browser URL for it."""

    publish_project_review_runtime_context(shell)
    session = create_project_review_session(
        shell,
        name=name,
        song_id=song_id,
        song_version_id=song_version_id,
        layer_id=layer_id,
        polarity=polarity,
        review_mode=review_mode,
        questionable_score_threshold=questionable_score_threshold,
        item_limit=item_limit,
    )
    server_launch = shell._review_server_controller.build_session_launch(
        _project_root(shell),
        session.id,
    )
    return ProjectReviewLaunch(
        session_id=session.id,
        session_name=session.name,
        item_count=len(session.items),
        url=server_launch.url,
        desktop_url=server_launch.desktop_url,
        phone_url=server_launch.phone_url,
        bind_host=server_launch.bind_host,
        port=server_launch.port,
    )


def reload_phone_review_status(shell: ProjectReviewShell) -> ProjectReviewLaunch:
    """Force the active phone-review session to publish a fresh status revision."""

    root = _project_root(shell)
    session_id = shell._review_server_controller.last_session_id
    if session_id is None or not session_id.strip():
        raise ValueError("Open a phone review session before reloading its status.")
    session = ReviewSessionService(root).get_session(session_id)
    if session is None:
        raise ValueError(f"Review session not found: {session_id}")
    server_launch = shell._review_server_controller.reload_status(root, session_id)
    return ProjectReviewLaunch(
        session_id=session.id,
        session_name=session.name,
        item_count=len(session.items),
        url=server_launch.url,
        desktop_url=server_launch.desktop_url,
        phone_url=server_launch.phone_url,
        bind_host=server_launch.bind_host,
        port=server_launch.port,
    )


def bind_phone_review_server_to_current_project(shell: ProjectReviewShell) -> None:
    """Point the reusable phone-review server at the currently open project root."""

    publish_project_review_runtime_context(shell, clear_active_session=True)


def list_project_review_dataset_versions(
    shell: ProjectReviewShell,
    *,
    queue_source_kind: str | None = "ez_project",
) -> list[ProjectReviewDatasetPaths]:
    """Return persisted review-dataset paths for the current EZ project."""

    app = FoundryApp(_project_root(shell))
    return [
        _dataset_paths_from_ref(item)
        for item in app.list_project_review_dataset_versions(
            project_ref=_project_ref(shell),
            queue_source_kind=queue_source_kind,
        )
    ]


def get_latest_project_review_dataset_version(
    shell: ProjectReviewShell,
    *,
    queue_source_kind: str | None = "ez_project",
) -> ProjectReviewDatasetPaths | None:
    """Return the newest persisted review-dataset paths for the current EZ project."""

    app = FoundryApp(_project_root(shell))
    match = app.get_latest_project_review_dataset_version(
        project_ref=_project_ref(shell),
        queue_source_kind=queue_source_kind,
    )
    if match is None:
        return None
    return _dataset_paths_from_ref(match)


def latest_project_review_dataset_folder(
    shell: ProjectReviewShell,
    *,
    queue_source_kind: str | None = "ez_project",
) -> Path:
    """Return the folder that currently anchors the latest project review dataset."""

    return _require_latest_project_review_dataset(
        shell,
        queue_source_kind=queue_source_kind,
    ).folder_path


def latest_project_review_dataset_artifact_path(
    shell: ProjectReviewShell,
    *,
    queue_source_kind: str | None = "ez_project",
) -> Path:
    """Return the persisted version artifact path for the latest project review dataset."""

    return _require_latest_project_review_dataset(
        shell,
        queue_source_kind=queue_source_kind,
    ).version_artifact_path


def _project_root(shell: ProjectReviewShell) -> Path:
    return Path(shell.project_storage.working_dir).resolve()


def clear_project_review_runtime_bridge(shell: ProjectReviewShell) -> None:
    clear_review_runtime_bridge(_project_root(shell))


def publish_project_review_runtime_context(
    shell: ProjectReviewShell,
    *,
    clear_active_session: bool = False,
) -> None:
    """Publish the current EZ runtime root and app context to phone review."""

    shell._review_server_controller.set_runtime_context(
        _project_root(shell),
        application_session=_application_session_payload(shell),
        clear_active_session=clear_active_session,
    )


def _project_ref(shell: ProjectReviewShell) -> str:
    project = shell.project_storage.project
    return f"project:{project.id}"


def _application_session_payload(shell: ProjectReviewShell) -> dict[str, object] | None:
    session = getattr(shell, "session", None)
    if session is None:
        return None
    active_song_id = str(session.active_song_id) if session.active_song_id is not None else None
    active_song_version_id = (
        str(session.active_song_version_id)
        if session.active_song_version_id is not None
        else None
    )
    active_song_title = None
    active_song_version_label = None
    if active_song_id is not None:
        song_record = shell.project_storage.songs.get(active_song_id)
        if song_record is not None:
            active_song_title = song_record.title
    if active_song_version_id is not None:
        version_record = shell.project_storage.song_versions.get(active_song_version_id)
        if version_record is not None:
            active_song_version_label = version_record.label
    return {
        "sessionId": str(session.id),
        "projectId": str(session.project_id),
        "projectName": shell.project_storage.project.name,
        "projectRef": _project_ref(shell),
        "activeSongId": active_song_id,
        "activeSongRef": f"song:{active_song_id}" if active_song_id is not None else None,
        "activeSongTitle": active_song_title,
        "activeSongVersionId": active_song_version_id,
        "activeSongVersionRef": (
            f"version:{active_song_version_id}"
            if active_song_version_id is not None
            else None
        ),
        "activeSongVersionLabel": active_song_version_label,
        "activeTimelineId": (
            str(session.active_timeline_id)
            if session.active_timeline_id is not None
            else None
        ),
        "activeSongVersionMa3TimecodePoolNo": session.active_song_version_ma3_timecode_pool_no,
        "uiPrefsRef": session.ui_prefs_ref,
    }


def _dataset_paths_from_ref(item: ProjectReviewDatasetVersionRef) -> ProjectReviewDatasetPaths:
    return ProjectReviewDatasetPaths(
        dataset_id=item.dataset_id,
        dataset_name=item.dataset_name,
        version_id=item.version_id,
        version_number=item.version_number,
        sample_count=item.sample_count,
        folder_path=item.dataset_folder_path,
        version_artifact_path=item.version_artifact_path,
    )


def _require_latest_project_review_dataset(
    shell: ProjectReviewShell,
    *,
    queue_source_kind: str | None,
) -> ProjectReviewDatasetPaths:
    match = get_latest_project_review_dataset_version(
        shell,
        queue_source_kind=queue_source_kind,
    )
    if match is None:
        raise ValueError("No extracted review dataset exists for the active project.")
    return match


__all__ = [
    "bind_phone_review_server_to_current_project",
    "ProjectReviewDatasetPaths",
    "ProjectReviewLaunch",
    "ProjectReviewShell",
    "publish_project_review_runtime_context",
    "create_project_review_session",
    "get_latest_project_review_dataset_version",
    "latest_project_review_dataset_artifact_path",
    "latest_project_review_dataset_folder",
    "list_project_review_dataset_versions",
    "open_project_review_session",
    "reload_phone_review_status",
]
