"""Review-tab helpers for the Foundry desktop window.
Exists to keep one clean entrypoint into the live phone review controller.
Connects Foundry project roots to one active all-events review session and URL launch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry import FoundryApp
from echozero.foundry.domain.review import ReviewSession
from echozero.foundry.review_server_controller import ReviewServerController
from echozero.ui.style import SHELL_TOKENS


class FoundryWindowReviewMixin:
    """Adds a minimal live-review tab to the Foundry window shell."""

    _root: Path
    _app: FoundryApp
    _review_server_controller: ReviewServerController

    review_source_path: QLineEdit
    review_open_btn: QPushButton
    review_summary: QPlainTextEdit

    _set_status: Any
    _error: Any

    def _build_review_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        open_group = QGroupBox("Live Phone Review")
        open_layout = QGridLayout(open_group)
        open_layout.setHorizontalSpacing(SHELL_TOKENS.scales.layout_gap)
        open_layout.setVerticalSpacing(SHELL_TOKENS.scales.inline_gap)

        self.review_source_path = QLineEdit(str(self._root))
        self.review_open_btn = QPushButton("Open Active Live Review")
        self.review_open_btn.clicked.connect(self._open_active_live_review)

        open_layout.addWidget(QLabel("Project Source"), 0, 0)
        open_layout.addWidget(self.review_source_path, 0, 1)
        open_layout.addWidget(self.review_open_btn, 1, 1)

        self.review_summary = QPlainTextEdit()
        self.review_summary.setReadOnly(True)
        self.review_summary.setPlaceholderText(
            "Open active live review to bind this project to the phone controller."
        )

        layout.addWidget(open_group)
        layout.addWidget(QLabel("Live Review Status"))
        layout.addWidget(self.review_summary, stretch=1)
        return box

    def _refresh_review_sessions(self, *_args, select_session_id: str | None = None) -> None:
        del select_session_id
        session_id = self._review_server_controller.last_session_id
        if session_id is None or not session_id.strip():
            self.review_summary.setPlainText(
                "No active live review session yet.\n\n"
                "Use 'Open Active Live Review' to create one for this project."
            )
            return
        session = self._app.reviews.get_session(session_id)
        if session is None:
            self.review_summary.setPlainText(
                "Last live review session is not available anymore.\n\n"
                "Open a new live review session."
            )
            return
        self.review_summary.setPlainText(self._format_live_review_summary(session))

    def _open_active_live_review(self) -> None:
        try:
            source_root = self._current_review_source_path()
            self._review_server_controller.enable()
            session = self._app.reviews.create_project_session(
                source_root,
                review_mode="all_events",
                item_limit=None,
            )
            launch = self._review_server_controller.bind_root(
                source_root,
                default_session_id=session.id,
            )
            if launch is None:
                raise RuntimeError("Phone review service is not enabled.")
            if not QDesktopServices.openUrl(QUrl(launch.url)):
                raise RuntimeError(f"Could not open review URL: {launch.url}")
            self.review_summary.setPlainText(self._format_live_review_summary(session))
            self._set_status(f"Opened active live review: {session.id}")
        except Exception as exc:
            self._error(exc)

    def _current_review_source_path(self) -> Path:
        text = self.review_source_path.text().strip()
        candidate = Path(text) if text else self._root
        if not candidate.exists():
            raise ValueError(f"Review source not found: {candidate}")
        return candidate

    def _on_review_root_switched(self) -> None:
        self._review_server_controller.stop()
        self.review_source_path.setText(str(self._root))
        self._refresh_review_sessions()

    @staticmethod
    def _format_live_review_summary(session: ReviewSession) -> str:
        pending_count = sum(1 for item in session.items if item.review_outcome.value == "pending")
        reviewed_count = len(session.items) - pending_count
        lines = [
            "Mode: Active live controller",
            f"Session: {session.name}",
            f"Session ID: {session.id}",
            f"Source: {session.source_ref or '(not set)'}",
            f"Items: {len(session.items)}",
            f"Pending / Reviewed: {pending_count} / {reviewed_count}",
        ]
        return "\n".join(lines)
