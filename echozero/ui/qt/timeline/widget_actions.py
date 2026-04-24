"""Timeline widget action routing and dialog orchestration.
Exists to keep TimelineWidget focused on rendering and input state.
Connects inspector and transfer actions to app intents and canonical object actions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.application.presentation.models import (
    ManualPullFlowPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.intents import (
    SetGain,
)
from echozero.application.timeline.object_actions import (
    ObjectActionSettingsSession,
    descriptor_for_action,
)
from echozero.models.paths import ensure_installed_models_dir
from echozero.ui.qt.timeline.widget_action_contract_mixin import (
    _AddSongRuntimeShell,
    _AddSongVersionRuntimeShell,
    _TimelineRuntimeShell,
    _coerce_event_id,
    _coerce_layer_id,
    _coerce_take_id,
    TimelineWidgetContractActionMixin,
)
from echozero.ui.qt.timeline.widget_action_transfer_mixin import (
    TimelineWidgetTransferActionMixin,
)
from echozero.ui.qt.settings_dialog import ActionSettingsDialog
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineSelectionResult,
)


class _ObjectActionSettingsRuntimeShell(_TimelineRuntimeShell, Protocol):
    def open_object_action_session(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: LayerId | None = None,
        object_type: str | None = None,
    ) -> ObjectActionSettingsSession: ...

    def dispatch_object_action_command(
        self, session_id: str, command: object
    ) -> ObjectActionSettingsSession: ...


class _RequestObjectActionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def request_object_action_run(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: LayerId,
        object_type: str,
    ) -> None: ...


class _RunObjectActionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def run_object_action(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: LayerId,
        object_type: str,
    ) -> TimelinePresentation: ...


class TimelineWidgetActionRouter(
    TimelineWidgetContractActionMixin,
    TimelineWidgetTransferActionMixin,
):
    """Routes inspector and transfer actions for the timeline widget."""

    def __init__(
        self,
        *,
        widget: QWidget,
        dispatch: Callable[[object], None],
        get_presentation: Callable[[], TimelinePresentation],
        set_presentation: Callable[[TimelinePresentation], None],
        resolve_runtime_shell: Callable[[], _TimelineRuntimeShell | None],
        selected_event_ids_for_selected_layers: Callable[[], list[EventId]],
        open_manual_pull_timeline_popup: (
            Callable[[ManualPullFlowPresentation], ManualPullTimelineSelectionResult | None] | None
        ) = None,
        input_dialog: type[QInputDialog] = QInputDialog,
        file_dialog: type[QFileDialog] = QFileDialog,
        message_box: type[QMessageBox] = QMessageBox,
        resolve_models_dir: Callable[[], Path] = ensure_installed_models_dir,
    ) -> None:
        self._widget = widget
        self._dispatch = dispatch
        self._get_presentation = get_presentation
        self._set_presentation = set_presentation
        self._resolve_runtime_shell = resolve_runtime_shell
        self._selected_event_ids_for_selected_layers = selected_event_ids_for_selected_layers
        self._open_manual_pull_timeline_popup: Callable[
            [ManualPullFlowPresentation], ManualPullTimelineSelectionResult | None
        ] = (
            open_manual_pull_timeline_popup
            if open_manual_pull_timeline_popup is not None
            else self._default_open_manual_pull_timeline_popup
        )
        self._input_dialog = input_dialog
        self._file_dialog = file_dialog
        self._message_box = message_box
        self._resolve_models_dir = resolve_models_dir

    def import_dropped_audio_path(
        self,
        audio_path: str,
        *,
        force_new_song: bool = False,
        target_song_id: str | None = None,
        target_song_title: str | None = None,
    ) -> bool:
        """Import one dropped audio path as a song or a new version for the active song."""

        runtime = self._resolve_runtime_shell()
        if runtime is None:
            self._message_box.warning(
                self._widget,
                "Add Song",
                "This runtime does not support adding songs from a drop.",
            )
            return False

        resolved_path = str(Path(audio_path))
        presentation = self._get_presentation()
        active_song_id = presentation.active_song_id.strip()
        resolved_target_song_id = target_song_id.strip() if target_song_id is not None else ""
        if resolved_target_song_id and not force_new_song:
            if self._prompt_to_add_drop_to_song(
                runtime=runtime,
                song_id=resolved_target_song_id,
                song_title=target_song_title,
                audio_path=resolved_path,
            ):
                return True
            force_new_song = True
        if active_song_id and not force_new_song:
            version_runtime = cast(_AddSongVersionRuntimeShell | None, runtime)
            if not callable(getattr(version_runtime, "add_song_version", None)):
                self._message_box.warning(
                    self._widget,
                    "Add Version",
                    "This runtime does not support creating a song version from a drop.",
                )
                return False
            active_title = presentation.active_song_title.strip() or "the active song"
            reply = self._message_box.question(
                self._widget,
                "Create New Version",
                (
                    "This timeline already has a source song loaded.\n\n"
                    f"Create a new version of \"{active_title}\" from "
                    f"\"{Path(resolved_path).name}\"?"
                ),
                self._message_box.StandardButton.Yes | self._message_box.StandardButton.No,
                self._message_box.StandardButton.No,
            )
            if reply != self._message_box.StandardButton.Yes:
                return True
            assert version_runtime is not None
            try:
                updated = version_runtime.add_song_version(active_song_id, resolved_path, label=None)
            except Exception as exc:
                self._message_box.warning(self._widget, "Add Version", str(exc))
                return False
            self._set_presentation(updated if updated is not None else runtime.presentation())
            return True

        song_runtime = cast(_AddSongRuntimeShell | None, runtime)
        if not callable(getattr(song_runtime, "add_song_from_path", None)):
            self._message_box.warning(
                self._widget,
                "Add Song",
                "This runtime does not support adding songs from a drop.",
            )
            return False
        title = Path(resolved_path).stem.strip() or "Imported Song"
        assert song_runtime is not None
        try:
            updated = song_runtime.add_song_from_path(title, resolved_path)
        except Exception as exc:
            self._message_box.warning(self._widget, "Add Song", str(exc))
            return False
        self._set_presentation(updated if updated is not None else runtime.presentation())
        return True

    def _prompt_to_add_drop_to_song(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        song_id: str,
        song_title: str | None,
        audio_path: str,
    ) -> bool:
        version_runtime = cast(_AddSongVersionRuntimeShell | None, runtime)
        if not callable(getattr(version_runtime, "add_song_version", None)):
            self._message_box.warning(
                self._widget,
                "Add Version",
                "This runtime does not support creating a song version from a drop.",
            )
            return True

        target_title = self._resolve_drop_target_song_title(song_id, song_title)
        reply = self._message_box.question(
            self._widget,
            "Add to Existing Song",
            (
                f'Add "{Path(audio_path).name}" to "{target_title}" as a new version?\n\n'
                "Choose No to import it as a new song instead."
            ),
            self._message_box.StandardButton.Yes
            | self._message_box.StandardButton.No
            | self._message_box.StandardButton.Cancel,
            self._message_box.StandardButton.Yes,
        )
        if reply == self._message_box.StandardButton.Cancel:
            return True
        if reply == self._message_box.StandardButton.No:
            return False
        assert version_runtime is not None
        try:
            updated = version_runtime.add_song_version(song_id, audio_path, label=None)
        except Exception as exc:
            self._message_box.warning(self._widget, "Add Version", str(exc))
            return True
        self._set_presentation(updated if updated is not None else runtime.presentation())
        return True

    def _resolve_drop_target_song_title(self, song_id: str, song_title: str | None) -> str:
        resolved_title = song_title.strip() if song_title is not None else ""
        if resolved_title:
            return resolved_title
        fallback_title = self._resolve_song_title(song_id)
        return fallback_title if fallback_title else "this song"

    def open_object_action_settings(self, action: InspectorAction) -> None:
        """Open the reusable settings dialog for one object action."""

        runtime = cast(_ObjectActionSettingsRuntimeShell | None, self._resolve_runtime_shell())
        if runtime is None:
            self._message_box.warning(
                self._widget,
                "Pipeline Settings",
                "This runtime does not support reusable pipeline settings.",
            )
            return
        if not callable(getattr(runtime, "open_object_action_session", None)) or not callable(
            getattr(runtime, "dispatch_object_action_command", None)
        ):
            self._message_box.warning(
                self._widget,
                "Pipeline Settings",
                "This runtime does not support reusable pipeline settings.",
            )
            return
        layer_id = _coerce_layer_id(action.params.get("layer_id"))
        try:
            session = runtime.open_object_action_session(
                action.action_id,
                action.params,
                object_id=layer_id,
                object_type="layer" if layer_id is not None else None,
            )
        except Exception as exc:
            self._message_box.warning(self._widget, "Pipeline Settings", str(exc))
            return

        dialog = ActionSettingsDialog(
            session,
            dispatch_command=runtime.dispatch_object_action_command,
            parent=self._widget,
        )
        dialog.exec()
        self._set_presentation(runtime.presentation())

    def _handle_runtime_pipeline_action(self, action_id: str, params: dict[str, object]) -> bool:
        runtime = self._resolve_runtime_shell()
        request_runtime = cast(_RequestObjectActionRuntimeShell | None, runtime)
        run_runtime = cast(_RunObjectActionRuntimeShell | None, runtime)
        request_object_action_run = (
            request_runtime.request_object_action_run
            if request_runtime is not None
            and callable(getattr(request_runtime, "request_object_action_run", None))
            else None
        )
        run_object_action = (
            run_runtime.run_object_action
            if run_runtime is not None and callable(getattr(run_runtime, "run_object_action", None))
            else None
        )
        if not callable(request_object_action_run) and not callable(run_object_action):
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"This runtime does not support '{action_id}'.",
            )
            return True
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"'{action_id}' requires a target layer.",
            )
            return True
        resolved_params = dict(params)
        descriptor = descriptor_for_action(action_id)
        if descriptor is None:
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"This runtime does not recognize '{action_id}'.",
            )
            return True
        if (
            descriptor.params_schema.get("model_path") == "dialog:file:model"
            and "model_path" not in resolved_params
            and "classify_model_path" not in resolved_params
        ):
            models_dir = self._resolve_models_dir()
            model_path, _ = self._file_dialog.getOpenFileName(
                self._widget,
                "Select Drum Classifier Model",
                str(models_dir),
                "Runtime Models (*.pth *.manifest.json);;PyTorch Models (*.pth);;Artifact Manifests (*.manifest.json);;All Files (*)",
            )
            if not model_path:
                return True
            resolved_params["model_path"] = model_path
        try:
            if callable(request_object_action_run):
                assert request_runtime is not None
                request_runtime.request_object_action_run(
                    action_id,
                    resolved_params,
                    object_id=layer_id,
                    object_type="layer",
                )
                updated = request_runtime.presentation()
            else:
                assert run_runtime is not None and callable(run_object_action)
                updated = run_runtime.run_object_action(
                    action_id,
                    resolved_params,
                    object_id=layer_id,
                    object_type="layer",
                )
        except NotImplementedError as exc:
            self._message_box.warning(self._widget, "Pipeline Action", str(exc))
            return True
        except Exception as exc:
            self._message_box.warning(self._widget, "Pipeline Action", str(exc))
            return True
        self._set_presentation(updated)
        return True
