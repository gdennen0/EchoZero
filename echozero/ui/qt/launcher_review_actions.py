"""Launcher review actions for the canonical EchoZero Qt shell.
Exists because project-backed review should be reachable from the real app surface.
Connects launcher menu actions to the app-shell review-session bridge and browser launch.
"""

from __future__ import annotations

from typing import Protocol

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox


class _ReviewActionHost(Protocol):
    runtime: object
    widget: object

    def _create_action(self, text: str, shortcut=None, handler=None) -> QAction: ...

    def _run_action(self, action_name: str, callback) -> bool: ...


def build_review_launcher_actions(host: _ReviewActionHost) -> dict[str, QAction]:
    """Return launcher actions for project-backed review when supported by the runtime."""

    if not callable(getattr(host.runtime, "open_project_review_session", None)):
        return {}
    actions = {
        "enable_phone_review_service": host._create_action(
            "Enable &Phone Review Service",
            None,
            lambda: host._run_action(
                "Enable Phone Review Service",
                lambda: _enable_phone_review_service(host),
            ),
        ),
        "disable_phone_review_service": host._create_action(
            "Disable Phone Review Service",
            None,
            lambda: host._run_action(
                "Disable Phone Review Service",
                lambda: _disable_phone_review_service(host),
            ),
        ),
    }
    actions["open_project_review"] = host._create_action(
        "Open &Questionable Review",
        None,
        lambda: host._run_action("Open Questionable Review", lambda: _open_questionable_review(host)),
    )
    actions["open_project_review_all"] = host._create_action(
        "Open &All-Events Review",
        None,
        lambda: host._run_action("Open All-Events Review", lambda: _open_all_events_review(host)),
    )
    return actions


def _enable_phone_review_service(host: _ReviewActionHost) -> None:
    runtime = host.runtime
    enable = getattr(runtime, "enable_phone_review_service", None)
    if not callable(enable):
        raise ValueError("Phone review service is not supported by this runtime.")
    enable()


def _disable_phone_review_service(host: _ReviewActionHost) -> None:
    runtime = host.runtime
    disable = getattr(runtime, "disable_phone_review_service", None)
    if not callable(disable):
        raise ValueError("Phone review service is not supported by this runtime.")
    disable()


def _open_questionable_review(host: _ReviewActionHost) -> None:
    session = getattr(host.runtime, "session", None)
    song_version_id = getattr(session, "active_song_version_id", None)
    if song_version_id is None:
        raise ValueError("Select a song version before opening a questionable review batch.")

    launch = host.runtime.open_project_review_session(
        song_version_id=str(song_version_id),
        review_mode="questionables",
        questionable_score_threshold=0.8,
        item_limit=25,
    )
    if not QDesktopServices.openUrl(QUrl(launch.url)):
        raise RuntimeError(f"Could not open review URL: {launch.url}")
    _show_phone_review_details(host, launch)


def _open_all_events_review(host: _ReviewActionHost) -> None:
    session = getattr(host.runtime, "session", None)
    song_version_id = getattr(session, "active_song_version_id", None)
    if song_version_id is None:
        raise ValueError("Select a song version before opening an all-events review batch.")

    launch = host.runtime.open_project_review_session(
        song_version_id=str(song_version_id),
        review_mode="all_events",
        item_limit=None,
    )
    if not QDesktopServices.openUrl(QUrl(launch.url)):
        raise RuntimeError(f"Could not open review URL: {launch.url}")
    _show_phone_review_details(host, launch)


def _show_phone_review_details(host: _ReviewActionHost, launch: object) -> None:
    phone_url = getattr(launch, "phone_url", None)
    desktop_url = getattr(launch, "desktop_url", getattr(launch, "url", None))
    if not isinstance(phone_url, str) or not phone_url.strip():
        return
    if isinstance(desktop_url, str) and desktop_url.strip() == phone_url.strip():
        return
    QMessageBox.information(
        getattr(host, "widget", None),
        "Phone Review Ready",
        "Desktop review opened.\n\n"
        f"Phone URL:\n{phone_url}",
    )


__all__ = ["build_review_launcher_actions"]
