"""Project-review helpers for the Stage Zero app shell runtime.
Exists because open projects need one direct path into the Foundry review lane.
Connects live ProjectStorage roots to persisted review sessions and browser-ready URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from echozero.application.session.models import Session
from echozero.foundry.domain.review import ReviewPolarity, ReviewSession
from echozero.foundry.review_server_controller import (
    ReviewServerController,
)
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.persistence.session import ProjectStorage


class ProjectReviewShell(Protocol):
    """Minimal app-shell protocol needed by the review bridge."""

    project_storage: ProjectStorage
    session: Session
    _review_server_controller: ReviewServerController


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


def _project_root(shell: ProjectReviewShell) -> Path:
    return Path(shell.project_storage.working_dir).resolve()


def _application_session_payload(shell: ProjectReviewShell) -> dict[str, object] | None:
    session = getattr(shell, "session", None)
    if session is None:
        return None
    return {
        "sessionId": str(session.id),
        "projectId": str(session.project_id),
        "activeSongId": str(session.active_song_id) if session.active_song_id is not None else None,
        "activeSongVersionId": (
            str(session.active_song_version_id)
            if session.active_song_version_id is not None
            else None
        ),
        "activeTimelineId": (
            str(session.active_timeline_id)
            if session.active_timeline_id is not None
            else None
        ),
        "activeSongVersionMa3TimecodePoolNo": session.active_song_version_ma3_timecode_pool_no,
        "uiPrefsRef": session.ui_prefs_ref,
    }


__all__ = [
    "ProjectReviewLaunch",
    "ProjectReviewShell",
    "create_project_review_session",
    "open_project_review_session",
]
