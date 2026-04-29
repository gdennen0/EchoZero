"""Launcher review actions for the canonical EchoZero Qt shell.
Exists because project-backed review should be reachable from the real app surface.
Connects launcher menu actions to the app-shell review-session bridge and browser launch.
"""

from __future__ import annotations

from pathlib import Path
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

    supports_review_session = callable(getattr(host.runtime, "open_project_review_session", None))
    supports_review_reload = callable(getattr(host.runtime, "reload_phone_review_status", None))
    supports_dataset_access = (
        callable(getattr(host.runtime, "latest_project_review_dataset_folder", None))
        and callable(getattr(host.runtime, "latest_project_review_dataset_artifact_path", None))
    )
    supports_specialized_model = (
        callable(getattr(host.runtime, "create_project_specialized_drum_models", None))
        and supports_dataset_access
    )
    supports_specialized_snare_model = (
        callable(getattr(host.runtime, "create_project_specialized_snare_model", None))
        and supports_dataset_access
    )
    if (
        not supports_review_session
        and not supports_dataset_access
        and not supports_specialized_model
        and not supports_specialized_snare_model
    ):
        return {}
    actions: dict[str, QAction] = {}
    if supports_review_session:
        actions["enable_phone_review_service"] = host._create_action(
            "Enable &Phone Review Service",
            None,
            lambda: host._run_action(
                "Enable Phone Review Service",
                lambda: _enable_phone_review_service(host),
            ),
        )
        actions["disable_phone_review_service"] = host._create_action(
            "Disable Phone Review Service",
            None,
            lambda: host._run_action(
                "Disable Phone Review Service",
                lambda: _disable_phone_review_service(host),
            ),
        )
        if supports_review_reload:
            actions["reload_phone_review_status"] = host._create_action(
                "Reload Phone Review &Status",
                None,
                lambda: host._run_action(
                    "Reload Phone Review Status",
                    lambda: _reload_phone_review_status(host),
                ),
            )
        actions["open_project_review"] = host._create_action(
            "Open &Questionable Review",
            None,
            lambda: host._run_action(
                "Open Questionable Review",
                lambda: _open_questionable_review(host),
            ),
        )
        actions["open_project_review_all"] = host._create_action(
            "Open &All-Events Review",
            None,
            lambda: host._run_action(
                "Open All-Events Review",
                lambda: _open_all_events_review(host),
            ),
        )
    if supports_dataset_access:
        actions["open_project_review_dataset_folder"] = host._create_action(
            "Open Latest Review Dataset &Folder",
            None,
            lambda: host._run_action(
                "Open Latest Review Dataset Folder",
                lambda: _open_latest_project_review_dataset_folder(host),
            ),
        )
        actions["open_project_review_dataset_artifact"] = host._create_action(
            "Open Latest Review Dataset &Record",
            None,
            lambda: host._run_action(
                "Open Latest Review Dataset Record",
                lambda: _open_latest_project_review_dataset_artifact(host),
            ),
        )
    if supports_specialized_model:
        actions["create_project_specialized_model"] = host._create_action(
            "Create Project &Specialized Model",
            None,
            lambda: host._run_action(
                "Create Project Specialized Model",
                lambda: _create_project_specialized_model(host),
            ),
        )
    if supports_specialized_snare_model:
        actions["create_project_specialized_snare_model"] = host._create_action(
            "Create Project S&nare-Only Model",
            None,
            lambda: host._run_action(
                "Create Project Snare-Only Model",
                lambda: _create_project_specialized_snare_model(host),
            ),
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


def _open_latest_project_review_dataset_folder(host: _ReviewActionHost) -> None:
    folder_path = getattr(host.runtime, "latest_project_review_dataset_folder", None)
    if not callable(folder_path):
        raise ValueError("Project review dataset access is not supported by this runtime.")
    _open_local_path(Path(folder_path()), label="review dataset folder")


def _open_latest_project_review_dataset_artifact(host: _ReviewActionHost) -> None:
    artifact_path = getattr(host.runtime, "latest_project_review_dataset_artifact_path", None)
    if not callable(artifact_path):
        raise ValueError("Project review dataset access is not supported by this runtime.")
    _open_local_path(Path(artifact_path()), label="review dataset record")


def _reload_phone_review_status(host: _ReviewActionHost) -> None:
    runtime = host.runtime
    reload_status = getattr(runtime, "reload_phone_review_status", None)
    if not callable(reload_status):
        raise ValueError("Phone review status reload is not supported by this runtime.")
    launch = reload_status()
    _show_phone_review_details(
        host,
        launch,
        title="Phone Review Reloaded",
        lead="Phone review status reloaded.",
    )


def _create_project_specialized_model(host: _ReviewActionHost) -> None:
    runtime = host.runtime
    create = getattr(runtime, "create_project_specialized_drum_models", None)
    if not callable(create):
        raise ValueError("Project specialized model creation is not supported by this runtime.")
    result = create()
    _show_project_specialized_model_details(host, result)


def _create_project_specialized_snare_model(host: _ReviewActionHost) -> None:
    runtime = host.runtime
    create = getattr(runtime, "create_project_specialized_snare_model", None)
    if not callable(create):
        raise ValueError("Project snare-only specialized model creation is not supported by this runtime.")
    result = create()
    _show_project_specialized_model_details(host, result)


def _open_local_path(path: Path, *, label: str) -> None:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise ValueError(f"{label.capitalize()} not found: {resolved_path}")
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved_path))):
        raise RuntimeError(f"Could not open {label}: {resolved_path}")


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
        f"{lead}\n\n"
        f"Phone URL:\n{phone_url}",
    )


def _show_project_specialized_model_details(host: _ReviewActionHost, result: object) -> None:
    project_ref = str(getattr(result, "project_ref", "")).strip() or "unknown project"
    review_dataset_version_id = str(getattr(result, "review_dataset_version_id", "")).strip()
    review_dataset_id = str(getattr(result, "review_dataset_id", "")).strip()
    promotions = getattr(result, "promotions", ())
    promotion_items = promotions if isinstance(promotions, tuple) else tuple(promotions or ())
    lines = [
        f"Project: {project_ref}",
    ]
    if review_dataset_id or review_dataset_version_id:
        dataset_bits = [bit for bit in (review_dataset_id, review_dataset_version_id) if bit]
        lines.append(f"Review dataset: {' / '.join(dataset_bits)}")
    for promotion in promotion_items:
        label = str(getattr(promotion, "label", "")).strip() or "model"
        manifest_path = str(getattr(promotion, "manifest_path", "")).strip()
        if manifest_path:
            lines.append(f"Promoted {label}: {manifest_path}")
        else:
            lines.append(f"Promoted {label}")
    QMessageBox.information(
        getattr(host, "widget", None),
        "Project Specialized Model Ready",
        "Created Project-specialized model.\n\n" + "\n".join(lines),
    )


__all__ = ["build_review_launcher_actions"]
