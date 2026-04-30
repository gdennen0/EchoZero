"""Stage Zero timeline widget for the canonical Qt shell surface.
Exists to render timeline presentation and capture operator input in one UI shell.
Connects app-facing presentation and intents to reusable timeline blocks and dialogs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, SectionCueId
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import (
    CommitMissedEventsReview,
    CommitMissedEventReview,
    CreateEvent,
    CreateRegion,
    DeleteRegion,
    Play,
    ReplaceSectionCues,
    SectionCueEdit,
    SelectRegion,
    Stop,
    TimelineIntent,
    UpdateRegion,
)
from echozero.application.settings import AppSettingsService
from echozero.models.paths import ensure_installed_models_dir
from echozero.ui.FEEL import (
    TIMELINE_RUNTIME_TICK_ACTIVE_MS,
    TIMELINE_TRANSPORT_TOP_GAP_PX,
)
from echozero.ui.qt.song_browser_drop import (
    SongBrowserAudioDrop,
    dropped_audio_paths,
    has_droppable_audio,
)
from echozero.ui.qt.song_browser_panel import SongBrowserPanel
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineDialog,
    ManualPullTimelineSelectionResult,
)
from echozero.ui.qt.timeline.region_manager import (
    RegionDraft,
    RegionManagerDialog,
    RegionPropertiesDialog,
)
from echozero.ui.qt.timeline.section_manager import SectionCueDraft, SectionManagerDialog
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
        initial_header_width: int | None = None,
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
        self._fix_action = "select"
        self._fix_nav_include_demoted = False
        self._snap_enabled = True
        self._grid_mode = TimelineGridMode.AUTO.value
        self._pipeline_status_visible_key: str | None = None
        self._pipeline_status_dismissed_key: str | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle(self._style.window_title)
        self._space_play_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_play_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._space_play_shortcut.activated.connect(self._play_transport_from_spacebar)
        self._space_play_shortcut.setEnabled(True)
        self._shift_space_preview_shortcut = QShortcut(QKeySequence("Shift+Space"), self)
        self._shift_space_preview_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._shift_space_preview_shortcut.activated.connect(
            self._preview_selected_event_from_shift_space
        )

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

        self._editor_bar = TimelineEditorModeBar(self)
        self._editor_bar.edit_mode_changed.connect(self._set_edit_mode)
        self._editor_bar.fix_action_changed.connect(self._set_fix_action)
        self._editor_bar.fix_nav_include_demoted_toggled.connect(self._set_fix_nav_include_demoted)
        self._editor_bar.snap_toggled.connect(self._set_snap_enabled)
        self._editor_bar.grid_mode_changed.connect(self._set_grid_mode)
        self._editor_bar.add_event_at_playhead_requested.connect(
            self._create_event_at_playhead
        )
        self._editor_bar.zoom_fit_requested.connect(self._zoom_to_fit_all)
        self._editor_bar.settings_requested.connect(self._open_settings_dialog)
        self._editor_bar.osc_settings_requested.connect(self._open_osc_settings_dialog)
        self._editor_bar.pipeline_settings_requested.connect(
            self._open_pipeline_settings_browser
        )
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
        self._pipeline_status_close_button = QPushButton("X", self._pipeline_status)
        self._pipeline_status_close_button.setObjectName("timelinePipelineStatusCloseButton")
        self._pipeline_status_close_button.setToolTip("Dismiss")
        self._pipeline_status_close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pipeline_status_close_button.setFixedSize(22, 22)
        self._pipeline_status_close_button.clicked.connect(self._dismiss_pipeline_status_banner)
        pipeline_status_layout.addWidget(self._pipeline_status_close_button, 0)
        left_layout.addWidget(self._pipeline_status)
        self._pipeline_status_auto_dismiss_timer = QTimer(self)
        self._pipeline_status_auto_dismiss_timer.setSingleShot(True)
        self._pipeline_status_auto_dismiss_timer.timeout.connect(
            self._on_pipeline_status_auto_dismiss_timeout
        )

        self._canvas = TimelineCanvas(self.presentation)
        if initial_header_width is not None:
            self._canvas.set_header_width(initial_header_width)
        self._ruler = TimelineRuler(self.presentation, header_width=self._canvas._header_width)
        left_layout.addWidget(self._ruler)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("timelineCanvasScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.layer_reorder_requested.connect(self._reorder_layer)
        self._canvas.select_adjacent_layer_requested.connect(self._select_adjacent_layer)
        self._canvas.mute_clicked.connect(self._toggle_layer_mute_from_header)
        self._canvas.solo_clicked.connect(self._toggle_layer_solo_from_header)
        self._canvas.pipeline_actions_clicked.connect(self._open_layer_pipeline_actions)
        self._canvas.push_clicked.connect(self._open_push_from_layer_action)
        self._canvas.pull_clicked.connect(self._open_pull_from_layer_action)
        self._canvas.section_manager_clicked.connect(self._open_section_manager_from_header)
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
        self._canvas.section_label_double_clicked.connect(self._rename_section_cue_from_canvas)
        self._canvas.section_boundary_double_clicked.connect(self._edit_section_cue_from_canvas)
        self._canvas.delete_events_requested.connect(self._delete_events)
        self._canvas.nudge_requested.connect(self._nudge_selected_events)
        self._canvas.duplicate_requested.connect(self._duplicate_selected_events)
        self._canvas.edit_mode_requested.connect(self._set_edit_mode)
        self._canvas.fix_action_requested.connect(self._set_fix_action)
        self._canvas.fix_nav_include_demoted_toggle_requested.connect(
            self._toggle_fix_nav_include_demoted
        )
        self._canvas.fix_promote_requested.connect(self._promote_fix_onset_event)
        self._canvas.fix_promote_batch_requested.connect(self._promote_fix_onset_events)
        self._canvas.fix_demote_selected_requested.connect(self._demote_fix_selected_events)
        self._canvas.fix_promote_selected_requested.connect(self._promote_fix_selected_events)
        self._canvas.snap_toggle_requested.connect(self._toggle_snap_enabled)
        self._canvas.grid_mode_cycle_requested.connect(self._cycle_grid_mode)
        self._canvas.add_event_at_playhead_requested.connect(self._create_event_at_playhead)
        self._canvas.preview_transfer_plan_requested.connect(self._preview_active_transfer_plan)
        self._canvas.apply_transfer_plan_requested.connect(self._apply_active_transfer_plan)
        self._canvas.cancel_transfer_plan_requested.connect(self._cancel_active_transfer_plan)
        self._canvas.preview_selected_event_clip_requested.connect(self._preview_selected_event_clip)
        self._canvas.header_width_changed.connect(self._on_canvas_header_width_changed)
        self._ruler.seek_requested.connect(self._seek)
        self._ruler.region_span_requested.connect(self._create_region_from_ruler_span)
        self._ruler.region_selected.connect(self._select_region_from_ruler)
        self._ruler.region_edit_requested.connect(self._edit_region_from_ruler)
        self._scroll.setWidget(self._canvas)
        self.setFocusProxy(self._canvas)
        left_layout.addWidget(self._scroll)

        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.setSingleStep(24)
        self._hscroll.setPageStep(200)
        self._hscroll.valueChanged.connect(self._on_horizontal_scroll_changed)
        left_layout.addWidget(self._hscroll)
        left_layout.addSpacing(TIMELINE_TRANSPORT_TOP_GAP_PX)
        left_layout.addWidget(self._transport)

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
        self._main_splitter.setSizes([1080, self._object_info.target_width()])
        self._main_splitter.splitterMoved.connect(self._sync_object_info_splitter_width)
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
        self._runtime_timer.setInterval(TIMELINE_RUNTIME_TICK_ACTIVE_MS)
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
        self._song_browser_panel.move_song_up_requested.connect(
            self._action_router.move_song_up
        )
        self._song_browser_panel.move_song_down_requested.connect(
            self._action_router.move_song_down
        )
        self._song_browser_panel.delete_song_requested.connect(self._action_router.delete_song)
        self._song_browser_panel.delete_song_version_requested.connect(
            self._action_router.delete_song_version
        )
        self._song_browser_panel.batch_move_songs_to_top_requested.connect(
            self._action_router.move_songs_to_top
        )
        self._song_browser_panel.batch_move_songs_to_bottom_requested.connect(
            self._action_router.move_songs_to_bottom
        )
        self._song_browser_panel.batch_delete_songs_requested.connect(
            self._action_router.delete_songs
        )
        self._song_browser_panel.songs_reordered_requested.connect(
            self._action_router.reorder_songs
        )
        self._song_browser_panel.audio_paths_dropped.connect(self._handle_song_browser_drop)
        self._song_browser_panel.collapsed_changed.connect(
            self._sync_song_browser_collapsed_state
        )
        self._object_info.collapsed_changed.connect(self._sync_object_info_collapsed_state)
        self._install_song_drop_targets(
            left_pane,
            self._transport,
            self._editor_bar,
            self._ruler,
            self._scroll.viewport(),
            self._canvas,
        )

        self.set_presentation(self.presentation)

    def layer_header_width_px(self) -> int:
        return int(self._canvas._header_width)

    def set_layer_header_width(self, width: int) -> None:
        self._canvas.set_header_width(int(width))
        self._on_canvas_header_width_changed(int(self._canvas._header_width))

    def _on_canvas_header_width_changed(self, width: int) -> None:
        self._ruler.set_header_width(float(width))
        self._update_horizontal_scroll_bounds(sync_bar_value=False)
        self._ruler.update()
        self._canvas.update()

    def _play_transport_from_spacebar(self) -> None:
        self._dispatch(Stop() if self.presentation.is_playing else Play())

    def _preview_selected_event_from_shift_space(self) -> None:
        self._preview_selected_event_clip()

    def _open_settings_dialog(self) -> None:
        actions = getattr(self, "_launcher_actions", {})
        action = actions.get("preferences") if isinstance(actions, dict) else None
        if action is None and isinstance(actions, dict):
            action = actions.get("project_settings")
        if action is not None:
            action.trigger()

    def _open_osc_settings_dialog(self) -> None:
        actions = getattr(self, "_launcher_actions", {})
        action = actions.get("osc_settings") if isinstance(actions, dict) else None
        if action is not None:
            action.trigger()

    def _open_pipeline_settings_browser(self) -> None:
        self._action_router.open_pipeline_settings_browser()

    def _open_region_manager_dialog(self) -> None:
        dialog = RegionManagerDialog(self.presentation, parent=self)
        if dialog.exec() != RegionManagerDialog.DialogCode.Accepted:
            return
        self._apply_region_manager_changes(dialog.region_drafts())

    def _section_manager_target_layer(
        self,
        *,
        preferred_layer_id: LayerId | None = None,
    ) -> LayerPresentation | None:
        if preferred_layer_id is not None:
            explicit_layer = next(
                (
                    layer
                    for layer in self.presentation.layers
                    if layer.layer_id == preferred_layer_id and layer.kind is LayerKind.SECTION
                ),
                None,
            )
            if explicit_layer is not None:
                return explicit_layer
        selected_layer_id = self.presentation.selected_layer_id
        if selected_layer_id is not None:
            selected_layer = next(
                (
                    layer
                    for layer in self.presentation.layers
                    if layer.layer_id == selected_layer_id and layer.kind is LayerKind.SECTION
                ),
                None,
            )
            if selected_layer is not None:
                return selected_layer
        section_layers = [
            layer for layer in self.presentation.layers if layer.kind is LayerKind.SECTION
        ]
        if len(section_layers) == 1:
            return section_layers[0]
        return None

    def _open_section_manager_from_header(self, layer_id: LayerId) -> None:
        self._focus_layer_for_header_action(layer_id)
        self._open_section_manager_dialog(target_layer_id=layer_id)

    @staticmethod
    def _section_layer_drafts(layer: LayerPresentation) -> list[SectionCueDraft]:
        ordered_events = sorted(
            layer.events,
            key=lambda event: (float(event.start), float(event.end), str(event.event_id)),
        )
        drafts: list[SectionCueDraft] = []
        for event in ordered_events:
            cue_ref = str(event.cue_ref or "").strip()
            if not cue_ref:
                continue
            drafts.append(
                SectionCueDraft(
                    cue_id=SectionCueId(str(event.event_id)),
                    start=float(event.start),
                    cue_ref=cue_ref,
                    name=str(event.label or "").strip() or cue_ref,
                    cue_number=event.cue_number,
                    color=event.color,
                    notes=event.notes,
                    payload_ref=event.payload_ref,
                )
            )
        return drafts

    def _open_section_manager_dialog(
        self,
        *,
        selected_cue_id: SectionCueId | None = None,
        target_layer_id: LayerId | None = None,
    ) -> None:
        target_layer = self._section_manager_target_layer(preferred_layer_id=target_layer_id)
        if target_layer is None:
            QMessageBox.information(
                self,
                "Select Section Layer",
                (
                    "Section manager is per section layer.\n\n"
                    "Select a section layer first (or create one) to open its cue stack."
                ),
            )
            return
        dialog = SectionManagerDialog(
            self.presentation,
            parent=self,
            cues=self._section_layer_drafts(target_layer),
            worksheet_title=f"{target_layer.title} Cue Stack",
            selected_cue_id=selected_cue_id,
        )
        if dialog.exec() != SectionManagerDialog.DialogCode.Accepted:
            return
        self._apply_section_manager_changes(
            dialog.section_cue_drafts(),
            target_layer_id=target_layer.layer_id,
        )

    def _rename_section_cue_from_canvas(self, cue_id: object) -> None:
        section_cue_id = SectionCueId(str(cue_id or "").strip())
        target_layer = self._section_manager_target_layer()
        if target_layer is None:
            return
        layer_drafts = self._section_layer_drafts(target_layer)
        current_draft = next((cue for cue in layer_drafts if cue.cue_id == section_cue_id), None)
        if current_draft is None:
            return
        current_name = str(current_draft.name or "").strip() or str(current_draft.cue_ref or "").strip()
        renamed_value, accepted = QInputDialog.getText(
            self,
            "Rename Section",
            "Section name",
            text=current_name,
        )
        if not accepted:
            return
        next_name = str(renamed_value or "").strip()
        if not next_name or next_name == current_name:
            return
        drafts: list[SectionCueDraft] = [
            SectionCueDraft(
                cue_id=cue.cue_id,
                start=float(cue.start),
                cue_ref=cue.cue_ref,
                name=next_name if cue.cue_id == section_cue_id else cue.name,
                cue_number=cue.cue_number,
                color=cue.color,
                notes=cue.notes,
                payload_ref=cue.payload_ref,
            )
            for cue in layer_drafts
        ]
        self._apply_section_manager_changes(
            drafts,
            target_layer_id=target_layer.layer_id,
        )

    def _edit_section_cue_from_canvas(self, cue_id: object) -> None:
        section_cue_id = SectionCueId(str(cue_id or "").strip())
        if not str(section_cue_id).strip():
            return
        self._open_section_manager_dialog(selected_cue_id=section_cue_id)

    def _promote_fix_onset_event(
        self,
        layer_id: object,
        take_id: object,
        start_seconds: float,
        end_seconds: float,
        source_event_id: str,
    ) -> None:
        self._dispatch(
            self._build_missed_fix_review_intent(
                layer_id=layer_id,
                take_id=take_id,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                source_event_id=source_event_id,
            )
        )

    def _promote_fix_onset_events(self, payload: object) -> None:
        entries = list(payload or [])
        if not entries:
            return
        intents: list[CommitMissedEventReview] = []
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 5:
                continue
            layer_id, take_id, start_seconds, end_seconds, source_event_id = entry
            intents.append(
                self._build_missed_fix_review_intent(
                    layer_id=layer_id,
                    take_id=take_id,
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                    source_event_id=str(source_event_id),
                )
            )
        if not intents:
            return
        if len(intents) == 1:
            self._dispatch(intents[0])
            return
        self._dispatch(CommitMissedEventsReview(intents=intents))

    def _build_missed_fix_review_intent(
        self,
        *,
        layer_id: object,
        take_id: object,
        start_seconds: float,
        end_seconds: float,
        source_event_id: str,
    ) -> CommitMissedEventReview:
        start = max(0.0, min(float(start_seconds), float(end_seconds)))
        end = max(start + 0.01, max(float(start_seconds), float(end_seconds)))
        label = self._default_fix_event_label(layer_id)
        source_id = str(source_event_id).strip()
        return CommitMissedEventReview(
            layer_id=layer_id,
            take_id=take_id,
            time_range=TimeRange(start=start, end=end),
            label=label,
            source_event_id=source_id or None,
            payload_ref=source_id or None,
        )

    def _default_fix_event_label(self, layer_id: object) -> str:
        layer = next(
            (candidate for candidate in self.presentation.layers if candidate.layer_id == layer_id),
            None,
        )
        if layer is None:
            return "Event"
        label = str(layer.title or "").strip()
        lower = label.lower()
        for suffix in (" events", " event", " layer", " lane"):
            if lower.endswith(suffix):
                label = label[: -len(suffix)].strip()
                break
        return label or "Event"

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

    def _select_region_from_ruler(self, region_id: object) -> None:
        target_region_id = next(
            (
                region.region_id
                for region in self.presentation.regions
                if region.region_id == region_id
            ),
            None,
        )
        if target_region_id is None:
            return
        self._dispatch(SelectRegion(region_id=target_region_id))

    def _edit_region_from_ruler(self, region_id: object) -> None:
        target = next(
            (
                region
                for region in self.presentation.regions
                if region.region_id == region_id
            ),
            None,
        )
        if target is None:
            return
        dialog = RegionPropertiesDialog(
            RegionDraft(
                region_id=target.region_id,
                start=float(target.start),
                end=float(target.end),
                label=target.label,
                color=target.color,
                kind=target.kind,
            ),
            parent=self,
        )
        if dialog.exec() != RegionPropertiesDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self._dispatch(
            UpdateRegion(
                region_id=target.region_id,
                time_range=TimeRange(start=float(values.start), end=float(values.end)),
                label=values.label,
                color=values.color,
                kind=target.kind,
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

    def _apply_section_manager_changes(
        self,
        drafts: list[SectionCueDraft],
        *,
        target_layer_id: LayerId | None = None,
    ) -> None:
        edits = [
            SectionCueEdit(
                cue_id=draft.cue_id,
                start=float(draft.start),
                cue_ref=draft.cue_ref,
                name=draft.name,
                cue_number=draft.cue_number,
                color=draft.color,
                notes=draft.notes,
                payload_ref=draft.payload_ref,
            )
            for draft in drafts
        ]
        self._dispatch(
            ReplaceSectionCues(
                cues=edits,
                target_layer_id=target_layer_id,
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
        self._add_recent_project_menu(file_menu, actions)
        file_menu.addSeparator()
        self._add_menu_action(file_menu, actions, "save_project")
        self._add_menu_action(file_menu, actions, "save_project_as")
        self._add_menu_action(file_menu, actions, "enable_phone_review_service")

        edit_menu = self._launcher_menu_bar.addMenu("&Edit")
        if edit_menu is None:
            self._launcher_menu_bar.setVisible(bool(actions))
            return
        self._add_menu_action(edit_menu, actions, "undo")
        self._add_menu_action(edit_menu, actions, "redo")
        if any(
            action_id in actions
            for action_id in ("project_settings", "preferences", "osc_settings")
        ):
            edit_menu.addSeparator()
            self._add_menu_action(edit_menu, actions, "project_settings")
            self._add_menu_action(edit_menu, actions, "preferences")
            self._add_menu_action(edit_menu, actions, "osc_settings")

        self._launcher_menu_bar.setVisible(bool(actions))

    @staticmethod
    def _add_menu_action(menu: QMenu, actions: Mapping[str, QAction], action_id: str) -> None:
        action = actions.get(action_id)
        if action is not None:
            menu.addAction(action)

    @staticmethod
    def _add_recent_project_menu(menu: QMenu, actions: Mapping[str, QAction]) -> None:
        recent_action_ids = [
            action_id
            for action_id in actions
            if action_id.startswith("open_recent_project::")
        ]
        if not recent_action_ids:
            return
        recent_menu = menu.addMenu("Open &Recent Project")
        if recent_menu is None:
            return
        for action_id in recent_action_ids:
            action = actions.get(action_id)
            if action is not None:
                recent_menu.addAction(action)

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
        if has_droppable_audio(event):
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _handle_song_drop_event(self, event: QDropEvent) -> bool:
        audio_paths = self._dropped_audio_paths(event, include_directory_audio=True)
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
        return self._action_router.import_dropped_audio_paths(
            audio_paths,
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

    def _sync_object_info_splitter_width(self, *_args: object) -> None:
        object_info = getattr(self, "_object_info", None)
        main_splitter = getattr(self, "_main_splitter", None)
        if object_info is None or main_splitter is None or object_info.is_collapsed:
            return
        sizes = main_splitter.sizes()
        if len(sizes) > 1:
            object_info.remember_expanded_width(sizes[1])

    def _sync_song_browser_collapsed_state(self, collapsed: bool) -> None:
        shell_splitter = getattr(self, "_shell_splitter", None)
        song_browser = getattr(self, "_song_browser_panel", None)
        if shell_splitter is None or song_browser is None:
            return
        sizes = shell_splitter.sizes()
        trailing_width = sizes[1] if len(sizes) > 1 else 1400
        browser_width = song_browser.target_width()
        shell_splitter.setSizes([browser_width, max(1, trailing_width)])

    def _sync_object_info_collapsed_state(self, collapsed: bool) -> None:
        del collapsed
        main_splitter = getattr(self, "_main_splitter", None)
        object_info = getattr(self, "_object_info", None)
        if main_splitter is None or object_info is None:
            return
        sizes = main_splitter.sizes()
        leading_width = sizes[0] if sizes else 1080
        object_info_width = object_info.target_width()
        main_splitter.setSizes([max(1, leading_width), object_info_width])

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
        *,
        include_directory_audio: bool = False,
    ) -> tuple[str, ...]:
        return dropped_audio_paths(
            event,
            include_directory_audio=include_directory_audio,
        )


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
