from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from echozero.ui.qt.app_shell import build_app_shell
from echozero.ui.qt.timeline.widget import TimelineWidget


PROJECT_FILE_FILTER = "EchoZero Project (*.ez);;All Files (*)"


class LauncherController:
    def __init__(self, *, runtime, widget) -> None:
        self.runtime = runtime
        self.widget = widget
        self.actions: dict[str, QAction] = {}
        self._original_close_event = getattr(widget, "closeEvent", None)

    def install(self) -> None:
        self.actions = {
            "new_project": self._create_action("&New Project", QKeySequence.StandardKey.New, self.new_project),
            "open_project": self._create_action("&Open Project", QKeySequence.StandardKey.Open, self.open_project),
            "save_project": self._create_action("&Save Project", QKeySequence.StandardKey.Save, self.save_project),
            "save_project_as": self._create_action(
                "Save Project &As...",
                QKeySequence.StandardKey.SaveAs,
                self.save_project_as,
            ),
        }
        setattr(self.widget, "_launcher_actions", self.actions)
        self.widget.closeEvent = self.close_event

    def _create_action(self, text: str, shortcut, handler) -> QAction:
        action = QAction(text, self.widget)
        action.setShortcut(shortcut)
        action.triggered.connect(handler)
        self.widget.addAction(action)
        return action

    def _has_lifecycle(self, method_name: str) -> bool:
        return callable(getattr(self.runtime, method_name, None))

    def _refresh_presentation(self) -> None:
        if hasattr(self.widget, "set_presentation") and callable(getattr(self.runtime, "presentation", None)):
            self.widget.set_presentation(self.runtime.presentation())

    def _current_project_path(self) -> Path | None:
        path = getattr(self.runtime, "project_path", None)
        return Path(path) if path is not None else None

    def _choose_open_path(self) -> Path | None:
        current = self._current_project_path()
        selected, _ = QFileDialog.getOpenFileName(
            self.widget,
            "Open Project",
            str(current.parent if current is not None else Path.cwd()),
            PROJECT_FILE_FILTER,
        )
        return Path(selected) if selected else None

    def _choose_save_path(self) -> Path | None:
        current = self._current_project_path()
        selected, _ = QFileDialog.getSaveFileName(
            self.widget,
            "Save Project As",
            str(current if current is not None else Path.cwd() / "project.ez"),
            PROJECT_FILE_FILTER,
        )
        return Path(selected) if selected else None

    def new_project(self) -> bool:
        if not self._has_lifecycle("new_project"):
            return False
        self.runtime.new_project()
        self._refresh_presentation()
        return True

    def open_project(self) -> bool:
        if not self._has_lifecycle("open_project"):
            return False
        path = self._choose_open_path()
        if path is None:
            return False
        self.runtime.open_project(path)
        self._refresh_presentation()
        return True

    def save_project_as(self) -> bool:
        if not self._has_lifecycle("save_project_as"):
            return False
        path = self._choose_save_path()
        if path is None:
            return False
        self.runtime.save_project_as(path)
        self._refresh_presentation()
        return True

    def save_project(self) -> bool:
        current_path = self._current_project_path()
        if current_path is None:
            return self.save_project_as()
        if self._has_lifecycle("save_project"):
            self.runtime.save_project()
            self._refresh_presentation()
            return True
        if self._has_lifecycle("save_project_as"):
            self.runtime.save_project_as(current_path)
            self._refresh_presentation()
            return True
        return False

    def confirm_close(self) -> bool:
        if not bool(getattr(self.runtime, "is_dirty", False)):
            return True
        reply = QMessageBox.question(
            self.widget,
            "Unsaved Changes",
            "Save changes before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return self.save_project()

    def close_event(self, event) -> None:
        if not self.confirm_close():
            event.ignore()
            return
        if callable(self._original_close_event):
            self._original_close_event(event)
            return
        event.accept()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EchoZero Stage Zero shell.")
    parser.add_argument(
        "--smoke-exit-seconds",
        type=float,
        default=None,
        help="If set to a positive number, close the app after that many seconds.",
    )
    parser.add_argument(
        "--use-demo-fixture",
        action="store_true",
        help="Boot the legacy demo fixture path instead of the canonical app shell path.",
    )
    parser.add_argument(
        "--working-dir-root",
        type=Path,
        default=None,
        help="Override the project working-directory root used by the app shell runtime.",
    )
    parsed, qt_args = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    working_dir_root = parsed.working_dir_root
    if working_dir_root is None and parsed.smoke_exit_seconds is not None and parsed.smoke_exit_seconds > 0:
        working_dir_root = Path(tempfile.gettempdir()) / "EchoZero" / "smoke-working"
    demo = build_app_shell(
        use_demo_fixture=parsed.use_demo_fixture,
        working_dir_root=working_dir_root,
    )
    widget = TimelineWidget(
        demo.presentation(),
        on_intent=demo.dispatch,
        runtime_audio=demo.runtime_audio,
    )
    launcher = LauncherController(runtime=demo, widget=widget)
    launcher.install()
    widget.resize(1440, 720)
    widget.setWindowTitle("EchoZero")
    widget.show()

    smoke_exit_seconds = parsed.smoke_exit_seconds
    if smoke_exit_seconds is not None and smoke_exit_seconds > 0:
        def _smoke_shutdown() -> None:
            app.quit()

        QTimer.singleShot(int(smoke_exit_seconds * 1000), _smoke_shutdown)

    try:
        return app.exec()
    finally:
        if hasattr(demo, "shutdown"):
            demo.shutdown()
        elif demo.runtime_audio is not None:
            demo.runtime_audio.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
