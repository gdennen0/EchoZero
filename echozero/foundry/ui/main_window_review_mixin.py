"""Review-session tab helpers for the Foundry desktop window.
Exists to surface project-backed review batches without rebuilding the review product lane in Qt.
Connects Foundry review-session persistence to one desktop create/list/open workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry import FoundryApp
from echozero.foundry.domain.review import ReviewSession
from echozero.foundry.review_server_controller import ReviewServerController
from echozero.ui.style import SHELL_TOKENS


class FoundryWindowReviewMixin:
    """Adds one review-session tab to the Foundry window shell."""

    _root: Path
    _app: FoundryApp
    _review_server_controller: ReviewServerController

    review_source_path: QLineEdit
    review_session_name: QLineEdit
    review_phone_service_enabled: QCheckBox
    review_mode: QComboBox
    review_score_threshold: QDoubleSpinBox
    review_item_limit: QSpinBox
    review_session_list: QListWidget
    review_summary: QPlainTextEdit
    review_create_btn: QPushButton
    review_open_btn: QPushButton

    _set_status: Any
    _error: Any

    def _build_review_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        create_group = QGroupBox("Review Batch")
        create_layout = QGridLayout(create_group)
        create_layout.setHorizontalSpacing(SHELL_TOKENS.scales.layout_gap)
        create_layout.setVerticalSpacing(SHELL_TOKENS.scales.inline_gap)

        self.review_source_path = QLineEdit(str(self._root))
        self.review_session_name = QLineEdit("")
        self.review_phone_service_enabled = QCheckBox("Enable Phone Review Service")
        self.review_phone_service_enabled.setChecked(self._review_server_controller.is_enabled)
        self.review_phone_service_enabled.toggled.connect(self._review_server_controller.set_enabled)

        self.review_mode = QComboBox()
        self.review_mode.addItem("Questionables Only", "questionables")
        self.review_mode.addItem("All Detected Events", "all_events")
        self.review_mode.currentIndexChanged.connect(lambda _index: self._sync_review_mode_controls())

        self.review_score_threshold = QDoubleSpinBox()
        self.review_score_threshold.setRange(0.0, 1.0)
        self.review_score_threshold.setSingleStep(0.05)
        self.review_score_threshold.setDecimals(2)
        self.review_score_threshold.setValue(0.80)

        self.review_item_limit = QSpinBox()
        self.review_item_limit.setRange(0, 999)
        self.review_item_limit.setSpecialValueText("All")
        self.review_item_limit.setValue(25)

        self.review_create_btn = QPushButton("Create Review Batch")
        self.review_create_btn.clicked.connect(self._create_review_batch)

        create_layout.addWidget(QLabel("Project Source"), 0, 0)
        create_layout.addWidget(self.review_source_path, 0, 1, 1, 3)
        create_layout.addWidget(QLabel("Session Name"), 1, 0)
        create_layout.addWidget(self.review_session_name, 1, 1, 1, 3)
        create_layout.addWidget(self.review_phone_service_enabled, 2, 0, 1, 2)
        create_layout.addWidget(QLabel("Review Mode"), 2, 2)
        create_layout.addWidget(self.review_mode, 2, 3)
        create_layout.addWidget(QLabel("Questionable Max Score"), 3, 0)
        create_layout.addWidget(self.review_score_threshold, 3, 1)
        create_layout.addWidget(QLabel("Batch Limit"), 3, 2)
        create_layout.addWidget(self.review_item_limit, 3, 3)
        create_layout.addWidget(self.review_create_btn, 4, 3)

        sessions_group = QGroupBox("Review Sessions")
        sessions_layout = QVBoxLayout(sessions_group)
        self.review_session_list = QListWidget()
        self.review_session_list.currentItemChanged.connect(
            lambda current, _previous: self._select_review_session(current)
        )
        sessions_layout.addWidget(self.review_session_list)

        review_actions = QHBoxLayout()
        review_actions.setSpacing(SHELL_TOKENS.scales.inline_gap)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_review_sessions)
        self.review_open_btn = QPushButton("Open Click-Through Review")
        self.review_open_btn.clicked.connect(self._open_selected_review_session)
        review_actions.addWidget(refresh_btn)
        review_actions.addWidget(self.review_open_btn)
        sessions_layout.addLayout(review_actions)

        self.review_summary = QPlainTextEdit()
        self.review_summary.setReadOnly(True)
        self.review_summary.setPlaceholderText("Review session details will appear here.")

        self._sync_review_mode_controls()
        layout.addWidget(create_group)
        layout.addWidget(sessions_group)
        layout.addWidget(QLabel("Selected Review Session"))
        layout.addWidget(self.review_summary, stretch=1)
        return box

    def _refresh_review_sessions(self, *_args, select_session_id: str | None = None) -> None:
        sessions = sorted(self._app.reviews.list_sessions(), key=lambda session: session.created_at)
        selected_session_id = select_session_id
        if selected_session_id is None and self.review_session_list.currentItem() is not None:
            selected_session_id = self._selected_review_session_id()
        if selected_session_id is None and sessions:
            selected_session_id = sessions[-1].id

        self.review_session_list.blockSignals(True)
        self.review_session_list.clear()
        selected_row = -1
        for index, session in enumerate(sessions):
            pending_count = sum(1 for item in session.items if item.review_outcome.value == "pending")
            item = QListWidgetItem(f"{session.name} ({pending_count}/{len(session.items)} pending)")
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self.review_session_list.addItem(item)
            if session.id == selected_session_id:
                selected_row = index
        self.review_session_list.blockSignals(False)

        if selected_row >= 0:
            self.review_session_list.setCurrentRow(selected_row)
            current = self.review_session_list.item(selected_row)
            self._select_review_session(current)
        else:
            self.review_summary.setPlainText("No review sessions yet.")
            self.review_open_btn.setEnabled(False)

    def _select_review_session(self, item: QListWidgetItem | None) -> None:
        session_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        session = self._app.reviews.get_session(str(session_id)) if session_id else None
        if session is None:
            self.review_summary.setPlainText("No review session selected.")
            self.review_open_btn.setEnabled(False)
            return
        self.review_summary.setPlainText(self._format_review_summary(session))
        self.review_open_btn.setEnabled(True)

    def _create_review_batch(self) -> None:
        try:
            session = self._app.reviews.create_project_session(
                self._current_review_source_path(),
                name=self._optional_review_name(),
                review_mode=self._current_review_mode(),
                questionable_score_threshold=self._current_questionable_score_threshold(),
                item_limit=self._current_item_limit(),
            )
            self._refresh_review_sessions(select_session_id=session.id)
            self._set_status(
                f"Review batch ready: {session.id} ({len(session.items)} items)"
            )
        except Exception as exc:
            self._error(exc)

    def _open_selected_review_session(self) -> None:
        try:
            session_id = self._selected_review_session_id()
            if session_id is None:
                raise ValueError("Select a review session before opening the review page.")
            url = self._review_server_controller.build_session_url(self._root, session_id)
            if not QDesktopServices.openUrl(QUrl(url)):
                raise RuntimeError(f"Could not open review URL: {url}")
            self._set_status(f"Opened review session: {session_id}")
        except Exception as exc:
            self._error(exc)

    def _selected_review_session_id(self) -> str | None:
        item = self.review_session_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value is not None else None

    def _optional_review_name(self) -> str | None:
        text = self.review_session_name.text().strip()
        return text or None

    def _current_item_limit(self) -> int | None:
        value = int(self.review_item_limit.value())
        return value or None

    def _current_review_mode(self) -> str:
        value = self.review_mode.currentData()
        return str(value) if value is not None else "questionables"

    def _current_questionable_score_threshold(self) -> float | None:
        if self._current_review_mode() != "questionables":
            return None
        return float(self.review_score_threshold.value())

    def _sync_review_mode_controls(self) -> None:
        self.review_score_threshold.setEnabled(self._current_review_mode() == "questionables")

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
    def _format_review_summary(session: ReviewSession) -> str:
        pending_count = sum(1 for item in session.items if item.review_outcome.value == "pending")
        reviewed_count = len(session.items) - pending_count
        metadata = session.metadata or {}
        lines = [
            f"Session: {session.name}",
            f"Session ID: {session.id}",
            f"Source: {session.source_ref or '(not set)'}",
            f"Items: {len(session.items)}",
            f"Pending / Reviewed: {pending_count} / {reviewed_count}",
            f"Classes: {', '.join(session.class_map) or '(none)'}",
            f"Import Format: {metadata.get('import_format', 'unknown')}",
            f"Review Mode: {str(metadata.get('review_mode', 'all_events')).replace('_', ' ')}",
        ]
        if metadata.get("questionable_score_threshold") is not None:
            lines.append(
                f"Questionable Max Score: {float(metadata['questionable_score_threshold']):.2f}"
            )
        if metadata.get("item_limit") is not None:
            lines.append(f"Batch Limit: {metadata['item_limit']}")
        if metadata.get("total_item_count") is not None:
            lines.append(
                f"Selected / Total: {metadata.get('selected_item_count', len(session.items))} / "
                f"{metadata['total_item_count']}"
            )
        return "\n".join(lines)


__all__ = ["FoundryWindowReviewMixin"]
