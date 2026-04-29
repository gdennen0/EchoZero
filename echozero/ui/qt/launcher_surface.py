"""
Launcher surface: Shared assembly for the canonical EchoZero Qt shell.
Exists so the real launcher and automation harness build the same app path.
Connects run_echozero, app-flow testing, and UI automation to one surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from echozero.application.settings import AppSettingsService, AudioOutputRuntimeConfig
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.launcher_review_actions import build_review_launcher_actions
from echozero.ui.qt.osc_settings_dialog import OscSettingsDialog
from echozero.ui.qt.preferences_dialog import PreferencesDialog
from echozero.ui.qt.timeline.widget import TimelineWidget
from echozero.ui.qt.window_geometry import resolve_initial_window_size


PROJECT_FILE_FILTER = "EchoZero Project (*.ez);;All Files (*)"
MAX_RECENT_PROJECTS = 10


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
        self._recent_menu_actions: dict[str, QAction] = {}
        self._session_recent_paths: list[Path] = []
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
            self.actions["osc_settings"] = self._create_action(
                "OSC &Settings...",
                None,
                self.osc_settings,
            )
        self.actions.update(build_review_launcher_actions(self))
        setattr(self.widget, "_launcher_actions", self.actions)
        self._refresh_recent_project_menu()
        self.widget.closeEvent = self.close_event

    def _create_action(
        self,
        text: str,
        shortcut=None,
        handler: Callable[..., object] | None = None,
    ) -> QAction:
        action = QAction(text, self.widget)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if handler is not None:
            action.triggered.connect(handler)
        self.widget.addAction(action)
        return action

    def _configure_widget_menu(self) -> None:
        configure_launcher_actions = getattr(self.widget, "configure_launcher_actions", None)
        if not callable(configure_launcher_actions):
            return
        configure_launcher_actions({**self.actions, **self._recent_menu_actions})

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

    def _stage_runtime_presentation_for_save(self) -> None:
        stage = getattr(self.runtime, "stage_project_runtime_presentation", None)
        if not callable(stage):
            return
        try:
            stage(getattr(self.widget, "presentation", None))
        except Exception:
            return

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

    def _service_recent_project_paths(self) -> tuple[Path, ...]:
        if self._app_settings_service is None:
            return ()
        provider = getattr(self._app_settings_service, "recent_project_paths", None)
        if not callable(provider):
            return ()
        try:
            paths = provider()
        except Exception:
            return ()
        if not isinstance(paths, (list, tuple)):
            return ()
        return tuple(Path(path) for path in paths if str(path).strip())

    def _recent_project_path_key(self, path: str | Path) -> str:
        return str(path).strip().replace("\\", "/").lower()

    def _merge_recent_project_paths(self) -> tuple[Path, ...]:
        combined: list[Path] = []
        seen: set[str] = set()
        for source in (*self._session_recent_paths, *self._service_recent_project_paths()):
            key = self._recent_project_path_key(source)
            if not key or key in seen:
                continue
            seen.add(key)
            combined.append(Path(source))
            if len(combined) >= MAX_RECENT_PROJECTS:
                break
        return tuple(combined)

    def _record_recent_project_path_session(self, path: str | Path) -> None:
        candidate = Path(path)
        key = self._recent_project_path_key(candidate)
        if not key:
            return
        self._session_recent_paths = [
            entry
            for entry in self._session_recent_paths
            if self._recent_project_path_key(entry) != key
        ]
        self._session_recent_paths.insert(0, candidate)
        self._session_recent_paths = self._session_recent_paths[:MAX_RECENT_PROJECTS]

    def _forget_recent_project_path_session(self, path: str | Path) -> None:
        key = self._recent_project_path_key(path)
        self._session_recent_paths = [
            entry
            for entry in self._session_recent_paths
            if self._recent_project_path_key(entry) != key
        ]

    def _remember_recent_project_path(self, path: str | Path) -> None:
        self._record_recent_project_path_session(path)
        if self._app_settings_service is None:
            return
        remember = getattr(self._app_settings_service, "remember_recent_project_path", None)
        if callable(remember):
            try:
                remember(path, limit=MAX_RECENT_PROJECTS)
            except Exception:
                return

    def _forget_recent_project_path(self, path: str | Path) -> None:
        self._forget_recent_project_path_session(path)
        if self._app_settings_service is None:
            return
        forget = getattr(self._app_settings_service, "forget_recent_project_path", None)
        if callable(forget):
            try:
                forget(path)
            except Exception:
                return

    def _refresh_recent_project_menu(self) -> None:
        remove_action = getattr(self.widget, "removeAction", None)
        for action in self._recent_menu_actions.values():
            if callable(remove_action):
                remove_action(action)
        self._recent_menu_actions = self._build_recent_project_actions()
        self._configure_widget_menu()

    def _build_recent_project_actions(self) -> dict[str, QAction]:
        actions: dict[str, QAction] = {}
        recent_paths = self._merge_recent_project_paths()
        if not recent_paths:
            action_id = self._recent_action_id(0)
            placeholder_action = self._create_action(
                "No Recent Projects",
                None,
                lambda _checked=False: None,
            )
            set_enabled = getattr(placeholder_action, "setEnabled", None)
            if callable(set_enabled):
                set_enabled(False)
            actions[action_id] = placeholder_action
            return actions
        for index, path in enumerate(recent_paths):
            action_id = self._recent_action_id(index)
            label = self._recent_action_label(index, path)
            actions[action_id] = self._create_action(
                label,
                None,
                lambda _checked=False, target=Path(path): self.open_recent_project_path(target),
            )
        return actions

    @staticmethod
    def _recent_action_id(index: int) -> str:
        return f"open_recent_project::{index}"

    @staticmethod
    def _recent_action_label(index: int, path: Path) -> str:
        filename = path.name or str(path)
        return f"{index + 1}. {filename} ({path})"

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
        if not self._run_action("Open Project", lambda: self.runtime.open_project(path)):
            return False
        self._remember_recent_project_path(path)
        self._refresh_recent_project_menu()
        return True

    def open_recent_project_path(self, path: str | Path) -> bool:
        if not self._has_lifecycle("open_project"):
            return False
        target_path = Path(path)
        if not self._confirm_unsaved_changes(
            "Save changes before opening another project?"
        ):
            return False
        if not self._run_action(
            "Open Project",
            lambda: self.runtime.open_project(target_path),
        ):
            self._forget_recent_project_path(target_path)
            self._refresh_recent_project_menu()
            return False
        self._remember_recent_project_path(target_path)
        self._refresh_recent_project_menu()
        return True

    def save_project_as(self) -> bool:
        if not self._has_lifecycle("save_project_as"):
            return False
        path = self._choose_save_path()
        if path is None:
            return False
        self._stage_runtime_presentation_for_save()
        if not self._run_action(
            "Save Project",
            lambda: self.runtime.save_project_as(path),
        ):
            return False
        current_path = self._current_project_path() or path
        self._remember_recent_project_path(current_path)
        self._refresh_recent_project_menu()
        return True

    def save_project(self) -> bool:
        current_path = self._current_project_path()
        if current_path is None:
            return self.save_project_as()
        if self._has_lifecycle("save_project"):
            self._stage_runtime_presentation_for_save()
            if not self._run_action("Save Project", self.runtime.save_project):
                return False
            self._remember_recent_project_path(current_path)
            self._refresh_recent_project_menu()
            return True
        if self._has_lifecycle("save_project_as"):
            self._stage_runtime_presentation_for_save()
            if not self._run_action(
                "Save Project",
                lambda: self.runtime.save_project_as(current_path),
            ):
                return False
            self._remember_recent_project_path(current_path)
            self._refresh_recent_project_menu()
            return True
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

    def osc_settings(self) -> bool:
        if self._app_settings_service is None:
            return False
        dialog = OscSettingsDialog(
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

    resolved_app_settings_service = (
        app_settings_service
        or getattr(runtime, "app_settings_service", None)
    )
    widget_kwargs = {
        "on_intent": runtime.dispatch,
        "runtime_audio": runtime.runtime_audio,
    }
    if resolved_app_settings_service is not None:
        widget_kwargs["app_settings_service"] = resolved_app_settings_service
    try:
        widget = TimelineWidget(
            runtime.presentation(),
            **widget_kwargs,
        )
    except TypeError as exc:
        # Test fakes and legacy widget shims may not accept the settings-service kwarg yet.
        if "app_settings_service" not in str(exc):
            raise
        widget_kwargs.pop("app_settings_service", None)
        widget = TimelineWidget(
            runtime.presentation(),
            **widget_kwargs,
        )
    widget.setObjectName("echozero.shell")
    initial_width, initial_height = resolve_initial_window_size(widget)
    set_minimum_size = getattr(widget, "setMinimumSize", None)
    if callable(set_minimum_size):
        set_minimum_size(1, 1)
    widget.resize(initial_width, initial_height)
    widget.setWindowTitle("EchoZero")

    controller = LauncherController(
        runtime=runtime,
        widget=widget,
        app_settings_service=app_settings_service,
    )
    controller.install()
    return LauncherSurface(runtime=runtime, widget=widget, controller=controller)
