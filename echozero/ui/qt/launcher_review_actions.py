"""Launcher review actions for the canonical EchoZero Qt shell.
Exists because project-backed review should be reachable from the real app surface.
Connects launcher menu actions to the app-shell review-session bridge and browser launch.
"""

from __future__ import annotations

from typing import Protocol

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMessageBox


class _ReviewActionHost(Protocol):
    runtime: object
    widget: object

    def _create_action(self, text: str, shortcut=None, handler=None) -> QAction: ...

    def _run_action(self, action_name: str, callback) -> bool: ...


def build_review_launcher_actions(host: _ReviewActionHost) -> dict[str, QAction]:
    """Return launcher actions for the minimal phone-review service entrypoint."""

    supports_enable_phone_review_service = callable(
        getattr(host.runtime, "enable_phone_review_service", None)
    )
    if not supports_enable_phone_review_service:
        return {}
    actions: dict[str, QAction] = {}
    actions["enable_phone_review_service"] = host._create_action(
        "Enable &Phone Review Service",
        None,
        lambda: host._run_action(
            "Enable Phone Review Service",
            lambda: _enable_phone_review_service(host),
        ),
    )
    return actions


def _enable_phone_review_service(host: _ReviewActionHost) -> None:
    runtime = host.runtime
    enable = getattr(runtime, "enable_phone_review_service", None)
    if not callable(enable):
        raise ValueError("Phone review service is not supported by this runtime.")
    launch = enable()
    open_live_review = getattr(runtime, "open_project_review_session", None)
    if callable(open_live_review):
        session = getattr(runtime, "session", None)
        active_song_version_id = getattr(session, "active_song_version_id", None)
        open_kwargs: dict[str, object] = {
            "review_mode": "all_events",
            "item_limit": None,
        }
        if active_song_version_id is not None:
            open_kwargs["song_version_id"] = str(active_song_version_id)
        try:
            launch = open_live_review(**open_kwargs)
        except Exception:
            # Keep the service enabled even when no review queue can be built yet.
            pass
    if launch is None:
        return
    _show_phone_review_details(
        host,
        launch,
        title="Phone Review Service Enabled",
        lead="Live link enabled for the current EZ session.",
    )


def _show_phone_review_details(
    host: _ReviewActionHost,
    launch: object,
    *,
    title: str = "Phone Review Ready",
    lead: str = "Desktop review opened.",
) -> None:
    phone_url = getattr(launch, "phone_url", None)
    desktop_url = getattr(launch, "desktop_url", getattr(launch, "url", None))
    if not isinstance(phone_url, str) or not phone_url.strip():
        return
    if isinstance(desktop_url, str) and desktop_url.strip() == phone_url.strip():
        return
    QMessageBox.information(
        getattr(host, "widget", None),
        title,
        f"{lead}\n\nPhone URL:\n{phone_url}",
    )


__all__ = ["build_review_launcher_actions"]
