"""
Launcher surface: Shared assembly for the canonical EchoZero Qt shell.
Exists so the real launcher and automation harness build the same app path.
Connects run_echozero, app-flow testing, and UI automation to one surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from echozero.application.settings import AppSettingsService, AudioOutputRuntimeConfig
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.preferences_dialog import PreferencesDialog
from echozero.ui.qt.timeline.widget import TimelineWidget


PROJECT_FILE_FILTER = "EchoZero Project (*.ez);;All Files (*)"


@dataclass(slots=True)
class LauncherSurface:
    """Canonical Stage Zero shell surface plus lifecycle wiring."""

    runtime: AppShellRuntime
    widget: TimelineWidget
    controller: "LauncherController"


class LauncherController:
    """Handles project-file actions and close confirmation for the shell widget."""

    def __init__(
        self,
        *,
        runtime,
        widget,
        app_settings_service: AppSettingsService | None = None,
    ) -> None:
        self.runtime = runtime
        self.widget = widget
        self._app_settings_service = (
            app_settings_service
            or getattr(runtime, "app_settings_service", None)
        )
        self.actions: dict[str, QAction] = {}
        self._original_close_event = getattr(widget, "closeEvent", None)

    def install(self) -> None:
        self.actions = {
            "undo": self._create_action("&Undo", QKeySequence.StandardKey.Undo, self.undo),
            "redo": self._create_action("&Redo", QKeySequence.StandardKey.Redo, self.redo),
            "new_project": self._create_action(
                "&New Project",
                QKeySequence.StandardKey.New,
                self.new_project,
            ),
            "open_project": self._create_action(
                "&Open Project",
                QKeySequence.StandardKey.Open,
                self.open_project,
            ),
            "save_project": self._create_action(
                "&Save Project",
                QKeySequence.StandardKey.Save,
                self.save_project,
            ),
            "save_project_as": self._create_action(
                "Save Project &As...",
                QKeySequence.StandardKey.SaveAs,
                self.save_project_as,
            ),
        }
        if self._app_settings_service is not None:
            self.actions["preferences"] = self._create_action(
                "&Preferences...",
                _preferences_shortcut(),
                self.preferences,
            )
        setattr(self.widget, "_launcher_actions", self.actions)
        configure_launcher_actions = getattr(self.widget, "configure_launcher_actions", None)
        if callable(configure_launcher_actions):
            configure_launcher_actions(self.actions)
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
        presentation = (
            self.runtime.presentation()
            if callable(getattr(self.runtime, "presentation", None))
            else None
        )
        if presentation is None:
            return
        apply_external_update = getattr(
            self.widget,
            "apply_external_presentation_update",
            None,
        )
        if callable(apply_external_update):
            apply_external_update(presentation)
            return
        if hasattr(self.widget, "set_presentation"):
            self.widget.set_presentation(presentation)

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
        if not selected:
            return None
        return self._normalize_project_path(Path(selected))

    @staticmethod
    def _normalize_project_path(path: Path) -> Path:
        return path.with_suffix(".ez") if path.suffix == "" else path

    def _run_action(self, action_name: str, callback) -> bool:
        try:
            callback()
        except Exception as exc:
            QMessageBox.critical(
                self.widget,
                f"{action_name} Failed",
                f"{action_name} failed.\n\n{exc}",
            )
            return False
        self._refresh_presentation()
        return True

    def _confirm_unsaved_changes(self, prompt: str) -> bool:
        if not bool(getattr(self.runtime, "is_dirty", False)):
            return True
        reply = QMessageBox.question(
            self.widget,
            "Unsaved Changes",
            prompt,
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

    def new_project(self) -> bool:
        if not self._has_lifecycle("new_project"):
            return False
        if not self._confirm_unsaved_changes(
            "Save changes before creating a new project?"
        ):
            return False
        return self._run_action("New Project", self.runtime.new_project)

    def open_project(self) -> bool:
        if not self._has_lifecycle("open_project"):
            return False
        path = self._choose_open_path()
        if path is None:
            return False
        if not self._confirm_unsaved_changes(
            "Save changes before opening another project?"
        ):
            return False
        return self._run_action("Open Project", lambda: self.runtime.open_project(path))

    def save_project_as(self) -> bool:
        if not self._has_lifecycle("save_project_as"):
            return False
        path = self._choose_save_path()
        if path is None:
            return False
        return self._run_action(
            "Save Project",
            lambda: self.runtime.save_project_as(path),
        )

    def save_project(self) -> bool:
        current_path = self._current_project_path()
        if current_path is None:
            return self.save_project_as()
        if self._has_lifecycle("save_project"):
            return self._run_action("Save Project", self.runtime.save_project)
        if self._has_lifecycle("save_project_as"):
            return self._run_action(
                "Save Project",
                lambda: self.runtime.save_project_as(current_path),
            )
        return False

    def undo(self) -> bool:
        if not self._has_lifecycle("undo"):
            return False
        return self._run_action("Undo", self.runtime.undo)

    def redo(self) -> bool:
        if not self._has_lifecycle("redo"):
            return False
        return self._run_action("Redo", self.runtime.redo)

    def preferences(self) -> bool:
        if self._app_settings_service is None:
            return False
        dialog = PreferencesDialog(
            self._app_settings_service,
            parent=self.widget,
        )
        return bool(dialog.exec())

    def confirm_close(self) -> bool:
        return self._confirm_unsaved_changes("Save changes before closing?")

    def close_event(self, event) -> None:
        if not self.confirm_close():
            event.ignore()
            return
        if callable(self._original_close_event):
            self._original_close_event(event)
            return
        event.accept()


def _preferences_shortcut():
    standard_key = getattr(QKeySequence.StandardKey, "Preferences", None)
    if standard_key is not None:
        return standard_key
    return QKeySequence("Ctrl+,")


def build_launcher_surface(
    *,
    working_dir_root: Path | None = None,
    initial_project_name: str = "EchoZero Project",
    analysis_service=None,
    sync_bridge=None,
    sync_service=None,
    app_settings_service: AppSettingsService | None = None,
    audio_output_config: AudioOutputRuntimeConfig | None = None,
) -> LauncherSurface:
    """Build the canonical shell widget/controller pair around AppShellRuntime."""

    runtime = build_app_shell(
        sync_bridge=sync_bridge,
        sync_service=sync_service,
        analysis_service=analysis_service,
        working_dir_root=working_dir_root,
        initial_project_name=initial_project_name,
        app_settings_service=app_settings_service,
        audio_output_config=audio_output_config,
    )
    if not isinstance(runtime, AppShellRuntime):
        raise TypeError("build_launcher_surface requires canonical AppShellRuntime")

    widget = TimelineWidget(
        runtime.presentation(),
        on_intent=runtime.dispatch,
        runtime_audio=runtime.runtime_audio,
        app_settings_service=(
            app_settings_service
            or getattr(runtime, "app_settings_service", None)
        ),
    )
    widget.setObjectName("echozero.shell")
    widget.resize(1440, 720)
    widget.setWindowTitle("EchoZero")

    controller = LauncherController(
        runtime=runtime,
        widget=widget,
        app_settings_service=app_settings_service,
    )
    controller.install()
    return LauncherSurface(runtime=runtime, widget=widget, controller=controller)
