"""Stage Zero timeline widget for the canonical Qt shell surface.
Exists to render timeline presentation and capture operator input in one UI shell.
Connects app-facing presentation and intents to reusable timeline blocks and dialogs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QAction, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMenuBar,
    QMessageBox,
    QScrollArea,
    QScrollBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import (
    CreateRegion,
    DeleteRegion,
    TimelineIntent,
    UpdateRegion,
)
from echozero.application.settings import AppSettingsService
from echozero.models.paths import ensure_installed_models_dir
from echozero.ui.qt.song_browser_drop import SongBrowserAudioDrop, dropped_audio_paths
from echozero.ui.qt.song_browser_panel import SongBrowserPanel
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineDialog,
    ManualPullTimelineSelectionResult,
)
from echozero.ui.qt.timeline.region_manager import RegionDraft, RegionManagerDialog
from echozero.ui.qt.timeline.object_info_panel import ObjectInfoPanel
from echozero.ui.qt.timeline.runtime_audio import (
    RuntimeAudioTimingSnapshot,
    TimelineRuntimeAudioController,
)
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, TimelineShellStyle
from echozero.ui.qt.timeline.time_grid import TimelineGridMode
from echozero.ui.qt.timeline.widget_actions import TimelineWidgetActionRouter
from echozero.ui.qt.timeline.widget_canvas import TimelineCanvas, badge_tooltip_labels
from echozero.ui.qt.timeline.widget_contract_mixin import TimelineWidgetContractMixin
from echozero.ui.qt.timeline.widget_controls import (
    TimelineEditorModeBar,
    TimelineRuler,
    TransportBar,
)
from echozero.ui.qt.timeline.widget_runtime_mixin import TimelineWidgetRuntimeMixin
from echozero.ui.qt.timeline.widget_viewport import (
    compute_follow_scroll_x,
    compute_scroll_bounds,
    estimate_timeline_span_seconds,
)
from echozero.ui.style.qt import ensure_qt_theme_installed


class TimelineWidget(TimelineWidgetRuntimeMixin, TimelineWidgetContractMixin, QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[TimelineIntent], TimelinePresentation | None] | None = None,
        *,
        runtime_audio: TimelineRuntimeAudioController | None = None,
        app_settings_service: AppSettingsService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        ensure_qt_theme_installed()
        self._style: TimelineShellStyle = TIMELINE_STYLE
        self.presentation = presentation
        self._on_intent = on_intent
        self._app_settings_service = app_settings_service
        self._runtime_audio = runtime_audio
        self._runtime_source_signature: tuple[tuple[str, str], ...] | None = None
        self._runtime_playhead_floor: float | None = None
        self._runtime_timing_snapshot: RuntimeAudioTimingSnapshot | None = None
        self._edit_mode = "select"
        self._snap_enabled = True
        self._grid_mode = TimelineGridMode.AUTO.value
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle(self._style.window_title)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._launcher_menu_bar = QMenuBar(self)
        self._launcher_menu_bar.setObjectName("timelineLauncherMenuBar")
        self._launcher_menu_bar.setNativeMenuBar(False)
        self._launcher_menu_bar.hide()
        root_layout.addWidget(self._launcher_menu_bar)

        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        root_layout.addWidget(content, 1)

        self._song_browser_panel = SongBrowserPanel(self.presentation, content)

        left_pane = QWidget(self)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._transport = TransportBar(self.presentation, on_intent=self._dispatch)
        left_layout.addWidget(self._transport)

        self._editor_bar = TimelineEditorModeBar(self)
        self._editor_bar.edit_mode_changed.connect(self._set_edit_mode)
        self._editor_bar.snap_toggled.connect(self._set_snap_enabled)
        self._editor_bar.grid_mode_changed.connect(self._set_grid_mode)
        self._editor_bar.settings_requested.connect(self._open_preferences_dialog)
        self._editor_bar.regions_requested.connect(self._open_region_manager_dialog)
        left_layout.addWidget(self._editor_bar)

        self._pipeline_status = QFrame(self)
        self._pipeline_status.setObjectName("timelinePipelineStatus")
        self._pipeline_status.setVisible(False)
        pipeline_status_layout = QHBoxLayout(self._pipeline_status)
        pipeline_status_layout.setContentsMargins(12, 8, 12, 8)
        pipeline_status_layout.setSpacing(8)
        self._pipeline_status_label = QLabel(self._pipeline_status)
        self._pipeline_status_label.setObjectName("timelinePipelineStatusLabel")
        self._pipeline_status_label.setWordWrap(True)
        self._pipeline_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        pipeline_status_layout.addWidget(self._pipeline_status_label, 1)
        left_layout.addWidget(self._pipeline_status)

        self._canvas = TimelineCanvas(self.presentation)
        self._ruler = TimelineRuler(self.presentation, header_width=self._canvas._header_width)
        left_layout.addWidget(self._ruler)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("timelineCanvasScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.select_adjacent_layer_requested.connect(self._select_adjacent_layer)
        self._canvas.active_clicked.connect(self._set_active_playback_target)
        self._canvas.pipeline_actions_clicked.connect(self._open_layer_pipeline_actions)
        self._canvas.push_clicked.connect(self._open_push_from_layer_action)
        self._canvas.pull_clicked.connect(self._open_pull_from_layer_action)
        self._canvas.take_toggle_clicked.connect(self._toggle_take_selector)
        self._canvas.take_selected.connect(self._select_take)
        self._canvas.event_selected.connect(self._select_event)
        self._canvas.select_adjacent_event_requested.connect(
            self._select_adjacent_event_in_selected_layer
        )
        self._canvas.move_selected_events_requested.connect(self._move_selected_events)
        self._canvas.move_selected_events_to_adjacent_layer_requested.connect(
            self._move_selected_events_to_adjacent_layer
        )
        self._canvas.take_action_selected.connect(self._trigger_take_action)
        self._canvas.contract_action_selected.connect(self._handle_contract_action)
        self._canvas.playhead_drag_requested.connect(self._seek)
        self._canvas.horizontal_scroll_requested.connect(self._scroll_horizontally_by_steps)
        self._canvas.zoom_requested.connect(self._zoom_from_input)
        self._canvas.clear_selection_requested.connect(self._clear_selection)
        self._canvas.select_all_requested.connect(self._select_all_events)
        self._canvas.set_selected_events_requested.connect(self._set_selected_events)
        self._canvas.create_event_requested.connect(self._create_event)
        self._canvas.delete_events_requested.connect(self._delete_events)
        self._canvas.nudge_requested.connect(self._nudge_selected_events)
        self._canvas.duplicate_requested.connect(self._duplicate_selected_events)
        self._canvas.edit_mode_requested.connect(self._set_edit_mode)
        self._canvas.snap_toggle_requested.connect(self._toggle_snap_enabled)
        self._canvas.grid_mode_cycle_requested.connect(self._cycle_grid_mode)
        self._canvas.preview_transfer_plan_requested.connect(self._preview_active_transfer_plan)
        self._canvas.apply_transfer_plan_requested.connect(self._apply_active_transfer_plan)
        self._canvas.cancel_transfer_plan_requested.connect(self._cancel_active_transfer_plan)
        self._ruler.seek_requested.connect(self._seek)
        self._ruler.region_span_requested.connect(self._create_region_from_ruler_span)
        self._scroll.setWidget(self._canvas)
        self.setFocusProxy(self._canvas)
        left_layout.addWidget(self._scroll)

        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.setSingleStep(24)
        self._hscroll.setPageStep(200)
        self._hscroll.valueChanged.connect(self._on_horizontal_scroll_changed)
        left_layout.addWidget(self._hscroll)

        self._object_info = ObjectInfoPanel(self)
        self._object_info.action_requested.connect(self._handle_contract_action)
        self._object_info.settings_requested.connect(self._open_action_settings_dialog)
        self._object_info_panel = self._object_info
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.setObjectName("timelineMainSplitter")
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(left_pane)
        self._main_splitter.addWidget(self._object_info)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.setSizes([1080, 320])
        self._shell_splitter = QSplitter(Qt.Orientation.Horizontal, content)
        self._shell_splitter.setObjectName("timelineShellSplitter")
        self._shell_splitter.setChildrenCollapsible(False)
        self._shell_splitter.addWidget(self._song_browser_panel)
        self._shell_splitter.addWidget(self._main_splitter)
        self._shell_splitter.setStretchFactor(0, 0)
        self._shell_splitter.setStretchFactor(1, 1)
        self._shell_splitter.setSizes([self._song_browser_panel.target_width(), 1400])
        self._shell_splitter.splitterMoved.connect(self._sync_song_browser_splitter_width)
        content_layout.addWidget(self._shell_splitter, 1)
        self._song_drop_targets: tuple[QWidget, ...] = ()

        self._runtime_timer = QTimer(self)
        self._runtime_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._runtime_timer.setInterval(8)
        self._runtime_timer.timeout.connect(self._on_runtime_tick)
        self._runtime_timer.start()
        self._action_router = TimelineWidgetActionRouter(
            widget=self,
            dispatch=cast(Callable[[object], None], self._dispatch),
            get_presentation=lambda: self.presentation,
            set_presentation=self.apply_external_presentation_update,
            resolve_runtime_shell=self._resolve_runtime_shell,
            selected_event_ids_for_selected_layers=self._selected_event_ids_for_selected_layers,
            open_manual_pull_timeline_popup=lambda flow: self._open_manual_pull_timeline_popup(
                flow
            ),
            input_dialog=QInputDialog,
            file_dialog=QFileDialog,
            message_box=QMessageBox,
            resolve_models_dir=ensure_installed_models_dir,
        )
        self._song_browser_panel.song_selected.connect(self._action_router.select_song)
        self._song_browser_panel.song_version_selected.connect(
            self._action_router.switch_song_version
        )
        self._song_browser_panel.add_song_requested.connect(
            self._action_router.add_song_from_dialog
        )
        self._song_browser_panel.add_song_version_requested.connect(
            self._action_router.add_song_version
        )
        self._song_browser_panel.delete_song_requested.connect(self._action_router.delete_song)
        self._song_browser_panel.delete_song_version_requested.connect(
            self._action_router.delete_song_version
        )
        self._song_browser_panel.audio_paths_dropped.connect(self._handle_song_browser_drop)
        self._song_browser_panel.collapsed_changed.connect(
            self._sync_song_browser_collapsed_state
        )
        self._install_song_drop_targets(
            left_pane,
            self._transport,
            self._editor_bar,
            self._ruler,
            self._scroll.viewport(),
            self._canvas,
        )

        self.set_presentation(self.presentation)

    def _open_preferences_dialog(self) -> None:
        actions = getattr(self, "_launcher_actions", {})
        action = actions.get("preferences") if isinstance(actions, dict) else None
        if action is not None:
            action.trigger()

    def _open_region_manager_dialog(self) -> None:
        dialog = RegionManagerDialog(self.presentation, parent=self)
        if dialog.exec() != RegionManagerDialog.DialogCode.Accepted:
            return
        self._apply_region_manager_changes(dialog.region_drafts())

    def _create_region_from_ruler_span(self, start_seconds: float, end_seconds: float) -> None:
        start = max(0.0, min(float(start_seconds), float(end_seconds)))
        end = max(start + 0.01, max(float(start_seconds), float(end_seconds)))
        label = f"Region {len(self.presentation.regions) + 1}"
        self._dispatch(
            CreateRegion(
                time_range=TimeRange(start=start, end=end),
                label=label,
            )
        )

    def _apply_region_manager_changes(self, drafts: list[RegionDraft]) -> None:
        existing_regions = list(self.presentation.regions)
        keep_ids = {draft.region_id for draft in drafts if draft.region_id is not None}

        for region in existing_regions:
            if region.region_id not in keep_ids:
                self._dispatch(DeleteRegion(region_id=region.region_id))

        existing_by_id = {region.region_id: region for region in self.presentation.regions}
        for draft in drafts:
            if draft.region_id is None:
                self._dispatch(
                    CreateRegion(
                        time_range=TimeRange(start=float(draft.start), end=float(draft.end)),
                        label=draft.label,
                        color=draft.color,
                        kind=draft.kind,
                    )
                )
                continue

            current = existing_by_id.get(draft.region_id)
            if current is None:
                self._dispatch(
                    CreateRegion(
                        time_range=TimeRange(start=float(draft.start), end=float(draft.end)),
                        label=draft.label,
                        color=draft.color,
                        kind=draft.kind,
                    )
                )
                continue

            if (
                abs(float(current.start) - float(draft.start)) <= 1e-6
                and abs(float(current.end) - float(draft.end)) <= 1e-6
                and current.label == draft.label
                and current.color == draft.color
                and current.kind == draft.kind
            ):
                continue

            self._dispatch(
                UpdateRegion(
                    region_id=draft.region_id,
                    time_range=TimeRange(start=float(draft.start), end=float(draft.end)),
                    label=draft.label,
                    color=draft.color,
                    kind=draft.kind,
                )
            )

    def configure_launcher_actions(self, actions: Mapping[str, QAction]) -> None:
        """Attach the canonical launcher actions to the widget menu bar."""

        self._launcher_menu_bar.clear()
        file_menu = self._launcher_menu_bar.addMenu("&File")
        if file_menu is None:
            self._launcher_menu_bar.setVisible(False)
            return
        self._add_menu_action(file_menu, actions, "new_project")
        self._add_menu_action(file_menu, actions, "open_project")
        file_menu.addSeparator()
        self._add_menu_action(file_menu, actions, "save_project")
        self._add_menu_action(file_menu, actions, "save_project_as")

        edit_menu = self._launcher_menu_bar.addMenu("&Edit")
        if edit_menu is None:
            self._launcher_menu_bar.setVisible(bool(actions))
            return
        self._add_menu_action(edit_menu, actions, "undo")
        self._add_menu_action(edit_menu, actions, "redo")
        if "preferences" in actions:
            edit_menu.addSeparator()
            self._add_menu_action(edit_menu, actions, "preferences")

        self._launcher_menu_bar.setVisible(bool(actions))

    @staticmethod
    def _add_menu_action(menu: QMenu, actions: Mapping[str, QAction], action_id: str) -> None:
        action = actions.get(action_id)
        if action is not None:
            menu.addAction(action)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is not None and self._accept_song_drag(event):
            return
        QWidget.dragEnterEvent(self, event)

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        if event is not None and self._accept_song_drag(event):
            return
        QWidget.dragMoveEvent(self, event)

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is not None and self._handle_song_drop_event(event):
            return
        QWidget.dropEvent(self, event)

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if watched in self._song_drop_targets and event is not None:
            if event.type() == QEvent.Type.DragEnter:
                return self._accept_song_drag(cast(QDragEnterEvent, event))
            if event.type() == QEvent.Type.DragMove:
                return self._accept_song_drag(cast(QDragMoveEvent, event))
            if event.type() == QEvent.Type.Drop:
                return self._handle_song_drop_event(cast(QDropEvent, event))
        return super().eventFilter(watched, event)

    def _install_song_drop_targets(self, *targets: QWidget | None) -> None:
        resolved: list[QWidget] = []
        for target in targets:
            if target is None:
                continue
            target.setAcceptDrops(True)
            if target is not self:
                target.installEventFilter(self)
            resolved.append(target)
        self._song_drop_targets = tuple(resolved)

    def _accept_song_drag(
        self,
        event: QDragEnterEvent | QDragMoveEvent,
    ) -> bool:
        if self._dropped_audio_paths(event):
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _handle_song_drop_event(self, event: QDropEvent) -> bool:
        audio_paths = self._dropped_audio_paths(event)
        if not audio_paths:
            event.ignore()
            return False
        self._handle_song_drop(audio_paths)
        event.acceptProposedAction()
        return True

    def _handle_song_drop(
        self,
        audio_paths: tuple[str, ...],
        *,
        force_new_song: bool = False,
        target_song_id: str | None = None,
        target_song_title: str | None = None,
    ) -> bool:
        if not audio_paths:
            return False
        if len(audio_paths) > 1:
            QMessageBox.warning(
                self,
                "Add Song",
                "Drop one audio file at a time.",
            )
            return True
        return self._action_router.import_dropped_audio_path(
            audio_paths[0],
            force_new_song=force_new_song,
            target_song_id=target_song_id,
            target_song_title=target_song_title,
        )

    def _sync_song_browser_splitter_width(self, *_args: object) -> None:
        song_browser = getattr(self, "_song_browser_panel", None)
        shell_splitter = getattr(self, "_shell_splitter", None)
        if song_browser is None or shell_splitter is None or song_browser.is_collapsed:
            return
        sizes = shell_splitter.sizes()
        if sizes:
            song_browser.remember_expanded_width(sizes[0])

    def _sync_song_browser_collapsed_state(self, collapsed: bool) -> None:
        shell_splitter = getattr(self, "_shell_splitter", None)
        song_browser = getattr(self, "_song_browser_panel", None)
        if shell_splitter is None or song_browser is None:
            return
        sizes = shell_splitter.sizes()
        trailing_width = sizes[1] if len(sizes) > 1 else 1400
        browser_width = song_browser.target_width()
        shell_splitter.setSizes([browser_width, max(1, trailing_width)])

    def _handle_song_browser_drop(self, audio_paths: object) -> None:
        target_song_id: str | None = None
        target_song_title: str | None = None
        force_new_song = True
        if isinstance(audio_paths, SongBrowserAudioDrop):
            resolved_paths = tuple(str(path) for path in audio_paths.audio_paths)
            target_song_id = audio_paths.target_song_id
            target_song_title = audio_paths.target_song_title
            force_new_song = target_song_id is None
        else:
            resolved_paths = tuple(audio_paths) if isinstance(audio_paths, (list, tuple)) else ()
        self._handle_song_drop(
            tuple(str(path) for path in resolved_paths),
            force_new_song=force_new_song,
            target_song_id=target_song_id,
            target_song_title=target_song_title,
        )

    @staticmethod
    def _dropped_audio_paths(
        event: QDragEnterEvent | QDragMoveEvent | QDropEvent,
    ) -> tuple[str, ...]:
        return dropped_audio_paths(event)


__all__ = [
    "ManualPullTimelineDialog",
    "ManualPullTimelineSelectionResult",
    "ObjectInfoPanel",
    "TimelineCanvas",
    "TimelineEditorModeBar",
    "TimelineRuler",
    "TimelineWidget",
    "TransportBar",
    "badge_tooltip_labels",
    "compute_follow_scroll_x",
    "compute_scroll_bounds",
    "estimate_timeline_span_seconds",
]
