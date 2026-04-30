"""Qt main window for the Foundry desktop surface.
Exists to compose dataset curation, training runs, and artifact review into one UI.
Connects Foundry application services and persistence to the operator workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDesktopServices
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QVBoxLayout, QWidget

from echozero.foundry import FoundryApp
from echozero.foundry.review_server_controller import ReviewServerController
from echozero.foundry.ui.main_window_dataset_mixin import FoundryWindowDatasetMixin
from echozero.foundry.ui.main_window_review_mixin import FoundryWindowReviewMixin
from echozero.foundry.ui.main_window_run_mixin import FoundryWindowRunMixin
from echozero.foundry.ui.main_window_worker import _RunWorker
from echozero.foundry.ui.main_window_workspace_mixin import FoundryWindowWorkspaceMixin
from echozero.ui.style import SHELL_TOKENS
from echozero.ui.style.qt import ensure_qt_theme_installed


class FoundryWindow(
    FoundryWindowDatasetMixin,
    FoundryWindowReviewMixin,
    FoundryWindowRunMixin,
    FoundryWindowWorkspaceMixin,
    QMainWindow,
):
    """Official EchoZero Foundry v1 window for local desktop workflows."""

    activity_item_received = pyqtSignal(str, str)
    _ACTIVE_RUN_STATUSES = {"queued", "preparing", "running", "evaluating", "exporting"}

    def __init__(self, root: Path):
        super().__init__()
        ensure_qt_theme_installed()
        self._root = Path(root)
        self._app = FoundryApp(self._root)
        self._review_server_controller = ReviewServerController()
        self._review_server_controller.enable()
        self._app.activity.set_listener(self._on_activity)
        self._show_error_dialogs = os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"

        self.setWindowTitle("EchoZero Foundry v1")
        self.resize(1280, 860)
        shell_scales = SHELL_TOKENS.scales

        self._dataset_id: str | None = None
        self._version_id: str | None = None
        self._run_id: str | None = None
        self._artifact_id: str | None = None
        self._selected_artifact_id: str | None = None
        self._run_thread: QThread | None = None
        self._run_worker: _RunWorker | None = None
        self._run_poll_timer = QTimer(self)
        self._run_poll_timer.setInterval(250)
        self._run_poll_timer.timeout.connect(self._poll_active_run)
        self._last_event_count_by_run: dict[str, int] = {}
        self._running_action_label: str | None = None
        self._run_cancel_event: Event | None = None

        container = QWidget()
        container.setObjectName("foundryRoot")
        self.setCentralWidget(container)
        root_layout = QVBoxLayout(container)
        root_layout.setSpacing(shell_scales.layout_gap)
        root_layout.setContentsMargins(
            shell_scales.layout_gap,
            shell_scales.layout_gap,
            shell_scales.layout_gap,
            shell_scales.layout_gap,
        )

        self.activity_item_received.connect(self._append_activity_item)

        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_workflow_tabs())
        splitter.addWidget(self._build_workspace_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        root_layout.addWidget(splitter, stretch=1)

        self._load_defaults()
        self._refresh_workspace_state()
        self._refresh_review_sessions()
        self._set_status(f"Workspace ready: {self._root}")

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._run_poll_timer.stop()
        if self._run_cancel_event is not None:
            self._run_cancel_event.set()
        self._review_server_controller.stop()
        self._app.activity.set_listener(None)
        self._app.activity.dispose()
        if self._run_thread is not None:
            self._run_thread.quit()
            self._run_thread.wait(1000)
        super().closeEvent(event)


def run_foundry_ui(root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = FoundryWindow(root or Path.cwd())
    window.show()
    return app.exec()


__all__ = ["FoundryWindow", "QDesktopServices", "run_foundry_ui"]
