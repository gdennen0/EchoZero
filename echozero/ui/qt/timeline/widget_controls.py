"""Reusable timeline chrome widgets.
Exists to keep ruler, transport, and editor-mode widgets out of the main timeline shell.
Connects timeline presentation and basic transport intents to compact Qt controls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QResizeEvent,
    QShowEvent,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from echozero.application.presentation.models import RegionPresentation, TimelinePresentation
from echozero.application.shared.enums import FollowMode
from echozero.application.timeline.intents import (
    Pause,
    Play,
    SetFollowCursorEnabled,
    Stop,
    TimelineIntent,
)
from echozero.ui.FEEL import (
    LAYER_HEADER_WIDTH_PX,
    RULER_HEIGHT_PX,
    TIMELINE_EDITOR_BAR_PADDING_X_PX,
    TIMELINE_EDITOR_BAR_PADDING_Y_PX,
    TIMELINE_EDITOR_GROUP_PADDING_X_PX,
    TIMELINE_EDITOR_GROUP_PADDING_Y_PX,
    TIMELINE_EDITOR_GROUP_SPACING_PX,
    TIMELINE_TRANSPORT_HEIGHT_PX,
)
from echozero.ui.qt.timeline.blocks.ruler import (
    RulerBlock,
    RulerLayout,
    seek_time_for_x,
    timeline_x_for_time,
)
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.time_grid import TimelineGridMode


class TimelineEditorModeBar(QWidget):
    """Compact editor-mode strip for timeline tools and shell actions."""

    edit_mode_changed = pyqtSignal(str)
    fix_action_changed = pyqtSignal(str)
    fix_nav_include_demoted_toggled = pyqtSignal(bool)
    add_event_at_playhead_requested = pyqtSignal()
    snap_toggled = pyqtSignal(bool)
    grid_mode_changed = pyqtSignal(str)
    zoom_fit_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    osc_settings_requested = pyqtSignal()
    pipeline_settings_requested = pyqtSignal()
    regions_requested = pyqtSignal()
    _COMPACT_WIDTH_THRESHOLD_PX = 1400

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelineEditorModeBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._compact_mode = False
        self._pending_standalone_width: int | None = None
        self.setProperty("compact", False)
        self._toolbar_labels: list[QLabel] = []
        self._mode_labels_full: dict[str, str] = {
            "select": "↖ Select",
            "move": "↔ Move",
            "draw": "+ Draw",
            "fix": "🩹 Fix",
            "region": "R Region",
            "erase": "- Erase",
        }
        self._mode_labels_compact: dict[str, str] = {
            "select": "↖",
            "move": "↔",
            "draw": "+",
            "fix": "🩹",
            "region": "R",
            "erase": "-",
        }
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            TIMELINE_EDITOR_BAR_PADDING_X_PX,
            TIMELINE_EDITOR_BAR_PADDING_Y_PX,
            TIMELINE_EDITOR_BAR_PADDING_X_PX,
            TIMELINE_EDITOR_BAR_PADDING_Y_PX,
        )
        layout.setSpacing(TIMELINE_EDITOR_GROUP_SPACING_PX)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        self._fix_action_group = QButtonGroup(self)
        self._fix_action_group.setExclusive(True)
        self._fix_action_buttons: dict[str, QPushButton] = {}
        mode_group = QWidget(self)
        mode_group.setObjectName("timelineEditorModeGroup")
        mode_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        mode_layout = self._create_group_layout(mode_group)
        mode_label = QLabel("Edit", mode_group)
        mode_label.setProperty("timelineToolbarLabel", True)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._toolbar_labels.append(mode_label)
        mode_layout.addWidget(mode_label, 0, Qt.AlignmentFlag.AlignVCenter)
        mode_tooltips = {
            "select": "Select mode",
            "move": "Move selected events",
            "draw": "Draw new events",
            "fix": "Fix assistant mode",
            "region": "Edit timeline regions",
            "erase": "Erase selected events",
        }
        for mode in ("select", "move", "draw", "erase", "fix", "region"):
            button = QPushButton(self._mode_labels_full[mode], mode_group)
            button.setProperty("timelineModeButton", True)
            button.setCheckable(True)
            button.setToolTip(mode_tooltips[mode])
            button.clicked.connect(
                lambda _checked=False, mode_name=mode: self.edit_mode_changed.emit(mode_name)
            )
            mode_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            self._button_group.addButton(button)
            self._mode_buttons[mode] = button
        layout.addWidget(mode_group, 0, Qt.AlignmentFlag.AlignVCenter)

        assist_group = QWidget(self)
        assist_group.setObjectName("timelineEditorAssistGroup")
        assist_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        assist_layout = self._create_group_layout(assist_group)
        assist_label = QLabel("Guides", assist_group)
        assist_label.setProperty("timelineToolbarLabel", True)
        assist_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._toolbar_labels.append(assist_label)
        assist_layout.addWidget(assist_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._snap_button = QPushButton("⌁ Snap", assist_group)
        self._snap_button.setObjectName("timelineEditorSnapButton")
        self._snap_button.setCheckable(True)
        self._snap_button.setToolTip("Toggle snap to timeline grid")
        self._snap_button.clicked.connect(lambda checked: self.snap_toggled.emit(bool(checked)))
        assist_layout.addWidget(self._snap_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._grid_button = QPushButton("▦ Grid: Auto", assist_group)
        self._grid_button.setObjectName("timelineEditorGridButton")
        self._grid_button.setToolTip("Cycle grid mode")
        self._grid_button.clicked.connect(self._cycle_grid_mode)
        assist_layout.addWidget(self._grid_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._fit_button = QPushButton("Fit All", assist_group)
        self._fit_button.setObjectName("timelineEditorFitAllButton")
        self._fit_button.setToolTip("Fit full timeline into view")
        self._fit_button.clicked.connect(self.zoom_fit_requested.emit)
        assist_layout.addWidget(self._fit_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._add_event_at_playhead_button = QPushButton("+ Playhead", assist_group)
        self._add_event_at_playhead_button.setObjectName("timelineEditorAddAtPlayheadButton")
        self._add_event_at_playhead_button.setToolTip(
            "Draw mode: add a 0.5s event at the current playhead (shortcut: A)"
        )
        self._add_event_at_playhead_button.clicked.connect(
            self.add_event_at_playhead_requested.emit
        )
        assist_layout.addWidget(self._add_event_at_playhead_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._fix_remove_button = QPushButton("−", assist_group)
        self._fix_remove_button.setObjectName("timelineEditorFixRemoveButton")
        self._fix_remove_button.setCheckable(True)
        self._fix_remove_button.setToolTip("Fix mode tool: demote false event (shortcut: Z)")
        self._fix_remove_button.clicked.connect(
            lambda _checked=False: self.fix_action_changed.emit("remove")
        )
        assist_layout.addWidget(self._fix_remove_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._fix_action_group.addButton(self._fix_remove_button)
        self._fix_action_buttons["remove"] = self._fix_remove_button

        self._fix_select_button = QPushButton("◉", assist_group)
        self._fix_select_button.setObjectName("timelineEditorFixSelectButton")
        self._fix_select_button.setCheckable(True)
        self._fix_select_button.setToolTip(
            "Fix mode tool: normal click/select + preview (shortcut: Shift+X)"
        )
        self._fix_select_button.clicked.connect(
            lambda _checked=False: self.fix_action_changed.emit("select")
        )
        assist_layout.addWidget(self._fix_select_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._fix_action_group.addButton(self._fix_select_button)
        self._fix_action_buttons["select"] = self._fix_select_button

        self._fix_promote_button = QPushButton("+", assist_group)
        self._fix_promote_button.setObjectName("timelineEditorFixPromoteButton")
        self._fix_promote_button.setCheckable(True)
        self._fix_promote_button.setToolTip(
            "Fix mode tool: promote missing event (shortcut: X)"
        )
        self._fix_promote_button.clicked.connect(
            lambda _checked=False: self.fix_action_changed.emit("promote")
        )
        assist_layout.addWidget(self._fix_promote_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._fix_action_group.addButton(self._fix_promote_button)
        self._fix_action_buttons["promote"] = self._fix_promote_button

        self._fix_include_demoted_button = QPushButton("D Demoted", assist_group)
        self._fix_include_demoted_button.setObjectName("timelineEditorFixDemotedNavButton")
        self._fix_include_demoted_button.setCheckable(True)
        self._fix_include_demoted_button.setToolTip(
            "Fix mode navigation: include demoted events in arrow key selection (shortcut: D)"
        )
        self._fix_include_demoted_button.clicked.connect(
            lambda checked: self.fix_nav_include_demoted_toggled.emit(bool(checked))
        )
        assist_layout.addWidget(self._fix_include_demoted_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._fix_promote_button.setEnabled(False)
        self._fix_remove_button.setEnabled(False)
        self._fix_select_button.setEnabled(False)
        self._fix_include_demoted_button.setEnabled(False)
        self._add_event_at_playhead_button.setEnabled(False)
        self._fix_select_button.setChecked(True)
        self._fix_promote_button.setVisible(False)
        self._fix_remove_button.setVisible(False)
        self._fix_select_button.setVisible(False)
        self._fix_include_demoted_button.setVisible(False)
        self._add_event_at_playhead_button.setVisible(False)

        layout.addWidget(assist_group, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

        shell_group = QWidget(self)
        shell_group.setObjectName("timelineEditorShellGroup")
        shell_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        shell_layout = self._create_group_layout(shell_group)
        shell_label = QLabel("Shell", shell_group)
        shell_label.setProperty("timelineToolbarLabel", True)
        shell_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._toolbar_labels.append(shell_label)
        shell_layout.addWidget(shell_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self._settings_button = QPushButton("⚙ Settings", shell_group)
        self._settings_button.setObjectName("timelineEditorSettingsButton")
        self._settings_button.setToolTip("Open application preferences")
        self._settings_button.clicked.connect(self.settings_requested.emit)
        shell_layout.addWidget(self._settings_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._osc_settings_button = QPushButton("OSC", shell_group)
        self._osc_settings_button.setObjectName("timelineEditorOscSettingsButton")
        self._osc_settings_button.setToolTip("Open OSC settings")
        self._osc_settings_button.clicked.connect(self.osc_settings_requested.emit)
        shell_layout.addWidget(self._osc_settings_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._pipeline_settings_button = QPushButton("Pipeline", shell_group)
        self._pipeline_settings_button.setObjectName("timelineEditorPipelineSettingsButton")
        self._pipeline_settings_button.setToolTip("Open reusable pipeline stage settings")
        self._pipeline_settings_button.clicked.connect(self.pipeline_settings_requested.emit)
        shell_layout.addWidget(
            self._pipeline_settings_button,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self._regions_button = QPushButton("▤ Regions", shell_group)
        self._regions_button.setObjectName("timelineEditorRegionsButton")
        self._regions_button.setToolTip("Open timeline regions manager")
        self._regions_button.clicked.connect(self.regions_requested.emit)
        shell_layout.addWidget(self._regions_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(shell_group, 0, Qt.AlignmentFlag.AlignVCenter)

        self._grid_modes: tuple[TimelineGridMode, ...] = (TimelineGridMode.AUTO,)
        self._grid_mode = TimelineGridMode.AUTO
        self._set_shell_button_labels(compact=False)
        self._set_mode_button_labels(compact=False)
        self._sync_fix_include_demoted_button_label(enabled=False)
        self._apply_button_width_hints(compact=False)
        if parent is not None:
            self._set_compact_mode(self._should_compact())

    def set_state(
        self,
        *,
        edit_mode: str,
        fix_action: str = "select",
        fix_nav_include_demoted: bool = False,
        snap_enabled: bool,
        grid_mode: str,
        beat_available: bool,
    ) -> None:
        self._grid_modes = (
            (TimelineGridMode.AUTO, TimelineGridMode.BEAT, TimelineGridMode.OFF)
            if beat_available
            else (TimelineGridMode.AUTO, TimelineGridMode.OFF)
        )
        self._grid_mode = TimelineGridMode(str(grid_mode))
        if self._grid_mode not in self._grid_modes:
            self._grid_mode = self._grid_modes[0]

        for mode_name, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(mode_name == edit_mode)
            button.blockSignals(False)
        resolved_fix_action = self._normalize_fix_action(fix_action)
        fix_action_enabled = edit_mode == "fix"
        for action_name, button in self._fix_action_buttons.items():
            button.blockSignals(True)
            button.setEnabled(fix_action_enabled)
            button.setChecked(action_name == resolved_fix_action)
            button.setVisible(fix_action_enabled)
            button.blockSignals(False)
        add_action_enabled = edit_mode == "draw"
        self._add_event_at_playhead_button.setEnabled(add_action_enabled)
        self._add_event_at_playhead_button.setVisible(add_action_enabled)
        self._fix_include_demoted_button.blockSignals(True)
        self._fix_include_demoted_button.setEnabled(fix_action_enabled)
        self._fix_include_demoted_button.setChecked(bool(fix_nav_include_demoted))
        self._fix_include_demoted_button.setVisible(fix_action_enabled)
        self._fix_include_demoted_button.blockSignals(False)
        self._sync_fix_include_demoted_button_label(enabled=bool(fix_nav_include_demoted))
        self._snap_button.blockSignals(True)
        self._snap_button.setChecked(bool(snap_enabled))
        self._snap_button.blockSignals(False)
        self._sync_grid_button_text()
        self._sync_grid_button_tooltip()

    def _cycle_grid_mode(self) -> None:
        if not self._grid_modes:
            return
        try:
            current_index = self._grid_modes.index(self._grid_mode)
        except ValueError:
            current_index = 0
        next_mode = self._grid_modes[(current_index + 1) % len(self._grid_modes)]
        self._grid_mode = next_mode
        self._sync_grid_button_text()
        self._sync_grid_button_tooltip()
        self.grid_mode_changed.emit(next_mode.value)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        should_compact = self._should_compact()
        self._set_compact_mode(should_compact)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        self._set_compact_mode(self._should_compact())
        self._pending_standalone_width = None

    def resize(self, *args: object) -> None:
        if self.parentWidget() is None and not self.isVisible():
            requested_width = self._extract_requested_width(args)
            if requested_width is not None and requested_width > 0:
                self._pending_standalone_width = requested_width
        if len(args) == 1 and isinstance(args[0], QSize):
            super().resize(args[0])
            return
        if len(args) == 2 and all(isinstance(arg, (int, float)) for arg in args):
            super().resize(int(args[0]), int(args[1]))
            return
        raise TypeError("TimelineEditorModeBar.resize expects QSize or width/height")

    def _set_compact_mode(self, enabled: bool) -> None:
        if self._compact_mode == enabled:
            return
        self._compact_mode = enabled
        self.setProperty("compact", enabled)
        for label in self._toolbar_labels:
            label.setVisible(not enabled)
        self._set_mode_button_labels(compact=enabled)
        self._set_shell_button_labels(compact=enabled)
        self._sync_fix_include_demoted_button_label(
            enabled=bool(self._fix_include_demoted_button.isChecked())
        )
        self._sync_add_at_playhead_button_label()
        self._apply_button_width_hints(compact=enabled)
        self._sync_grid_button_text()
        self._sync_grid_button_tooltip()
        self._repolish_toolbar_widgets()
        self.updateGeometry()
        self.update()

    def _should_compact(self) -> bool:
        host_width = self._host_width()
        return host_width < self._COMPACT_WIDTH_THRESHOLD_PX

    def _host_width(self) -> int:
        parent = self.parentWidget()
        if parent is None and self._pending_standalone_width is not None:
            return self._pending_standalone_width
        if parent is not None and parent.width() > 0:
            return parent.width()
        if self.width() > 0:
            return self.width()
        return self.sizeHint().width()

    @staticmethod
    def _extract_requested_width(args: tuple[object, ...]) -> int | None:
        if len(args) == 1 and isinstance(args[0], QSize):
            return args[0].width()
        if len(args) == 2 and all(isinstance(arg, (int, float)) for arg in args):
            return int(args[0])
        return None

    def _set_mode_button_labels(self, *, compact: bool) -> None:
        labels = self._mode_labels_compact if compact else self._mode_labels_full
        for mode, button in self._mode_buttons.items():
            button.setText(labels.get(mode, self._mode_labels_full.get(mode, mode.title())))

    def _set_shell_button_labels(self, *, compact: bool) -> None:
        if compact:
            self._settings_button.setText("⚙")
            self._osc_settings_button.setText("O")
            self._pipeline_settings_button.setText("P")
            self._regions_button.setText("▤")
            return
        self._settings_button.setText("⚙ Settings")
        self._osc_settings_button.setText("OSC")
        self._pipeline_settings_button.setText("Pipeline")
        self._regions_button.setText("▤ Regions")

    def _sync_fix_include_demoted_button_label(self, *, enabled: bool) -> None:
        if self._compact_mode:
            self._fix_include_demoted_button.setText(f"D {'On' if enabled else 'Off'}")
            return
        self._fix_include_demoted_button.setText(
            f"D Demoted {'On' if enabled else 'Off'}"
        )

    def _sync_add_at_playhead_button_label(self) -> None:
        self._add_event_at_playhead_button.setText("+@" if self._compact_mode else "+ Playhead")

    def _apply_button_width_hints(self, *, compact: bool) -> None:
        mode_width = 32 if compact else 58
        snap_width = 34 if compact else 64
        grid_width = 44 if compact else 84
        shell_width = 34 if compact else 88
        fix_small_width = 22 if compact else 30
        fix_select_width = 28 if compact else 36
        fix_toggle_width = 48 if compact else 90
        add_playhead_width = 36 if compact else 92

        for button in self._mode_buttons.values():
            button.setMinimumWidth(mode_width)
        self._snap_button.setMinimumWidth(snap_width)
        self._grid_button.setMinimumWidth(grid_width)
        self._settings_button.setMinimumWidth(shell_width)
        self._osc_settings_button.setMinimumWidth(shell_width)
        self._pipeline_settings_button.setMinimumWidth(shell_width)
        self._regions_button.setMinimumWidth(shell_width)
        self._fix_remove_button.setMinimumWidth(fix_small_width)
        self._fix_promote_button.setMinimumWidth(fix_small_width)
        self._fix_select_button.setMinimumWidth(fix_select_width)
        self._fix_include_demoted_button.setMinimumWidth(fix_toggle_width)
        self._add_event_at_playhead_button.setMinimumWidth(add_playhead_width)

    def _repolish_toolbar_widgets(self) -> None:
        widgets: tuple[QWidget, ...] = (
            self,
            self._snap_button,
            self._grid_button,
            self._settings_button,
            self._osc_settings_button,
            self._pipeline_settings_button,
            self._regions_button,
            self._fix_remove_button,
            self._fix_select_button,
            self._fix_promote_button,
            self._fix_include_demoted_button,
            self._add_event_at_playhead_button,
            *tuple(self._mode_buttons.values()),
        )
        for widget in widgets:
            style = widget.style()
            if style is None:
                continue
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def _sync_grid_button_text(self) -> None:
        mode_label = self._grid_mode.value.title()
        if self._compact_mode:
            compact_label = {"Auto": "A", "Beat": "B", "Off": "O"}.get(mode_label, mode_label)
            self._grid_button.setText(f"▦{compact_label}")
            return
        self._grid_button.setText(f"▦ Grid: {mode_label}")

    def _sync_grid_button_tooltip(self) -> None:
        self._grid_button.setToolTip(
            f"Cycle grid mode (current: {self._grid_mode.value.title()})"
        )

    def _create_group_layout(self, parent: QWidget) -> QHBoxLayout:
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(
            TIMELINE_EDITOR_GROUP_PADDING_X_PX,
            TIMELINE_EDITOR_GROUP_PADDING_Y_PX,
            TIMELINE_EDITOR_GROUP_PADDING_X_PX,
            TIMELINE_EDITOR_GROUP_PADDING_Y_PX,
        )
        layout.setSpacing(TIMELINE_EDITOR_GROUP_SPACING_PX)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        return layout

    @staticmethod
    def _normalize_fix_action(action: str) -> str:
        normalized = str(action or "select").strip().lower()
        if normalized in {"remove", "promote"}:
            return normalized
        return "select"


class TransportBar(QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[TimelineIntent], TimelinePresentation | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self._block = TransportBarBlock()
        self._control_rects: dict[str, QRectF] = {}
        self.setMinimumHeight(TIMELINE_TRANSPORT_HEIGHT_PX)
        self.setMaximumHeight(TIMELINE_TRANSPORT_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._control_rects = cast(
            dict[str, QRectF],
            self._block.paint(
                painter,
                TransportLayout.create(width=self.width(), height=self.height()),
                self.presentation,
            ),
        )

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        pos = event.position()
        if (play_rect := self._control_rects.get("play")) is not None and play_rect.contains(pos):
            self._dispatch(Pause() if self.presentation.is_playing else Play())
            return
        if (stop_rect := self._control_rects.get("stop")) is not None and stop_rect.contains(pos):
            self._dispatch(Stop())
            return
        if (follow_rect := self._control_rects.get("follow")) is not None and follow_rect.contains(
            pos
        ):
            self._dispatch(
                SetFollowCursorEnabled(
                    enabled=self.presentation.follow_mode == FollowMode.OFF,
                )
            )
            return
        super().mousePressEvent(event)

    def _dispatch(self, intent: TimelineIntent) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            self.set_presentation(updated)


class TimelineRuler(QWidget):
    seek_requested = pyqtSignal(float)
    region_span_requested = pyqtSignal(float, float)
    region_selected = pyqtSignal(object)
    region_edit_requested = pyqtSignal(object)

    def __init__(
        self,
        presentation: TimelinePresentation,
        *,
        header_width: float = float(LAYER_HEADER_WIDTH_PX),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.presentation = presentation
        self._header_width = header_width
        self._block = RulerBlock()
        self._dragging = False
        self._edit_mode = "select"
        self._drag_anchor_time: float | None = None
        self._drag_current_time: float | None = None
        self.setMinimumHeight(RULER_HEIGHT_PX)
        self.setMaximumHeight(RULER_HEIGHT_PX)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def set_header_width(self, width: float) -> None:
        next_width = max(1.0, float(width))
        if abs(next_width - self._header_width) <= 0.1:
            return
        self._header_width = next_width
        self.update()

    def set_editor_mode(self, mode: str) -> None:
        previous_mode = self._edit_mode
        normalized = (mode or "select").strip().lower()
        self._edit_mode = normalized
        if previous_mode != normalized and (
            previous_mode == "region" or normalized == "region"
        ):
            self._dragging = False
            self._drag_anchor_time = None
            self._drag_current_time = None
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._block.paint(
            painter,
            RulerLayout(QRectF(0, 0, self.width(), self.height()), self._header_width),
            self.presentation,
        )
        self._paint_region_drag_preview(painter)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().x() >= self._header_width
        ):
            if self._edit_mode == "region":
                hit_region = self._region_at_x(event.position().x())
                if hit_region is not None:
                    self._dragging = False
                    self._drag_anchor_time = None
                    self._drag_current_time = None
                    self.region_selected.emit(hit_region.region_id)
                    self.update()
                    event.accept()
                    return
                self._dragging = True
                current_time = self._seek_time_at_x(event.position().x())
                self._drag_anchor_time = current_time
                self._drag_current_time = current_time
                self.update()
            else:
                self._dragging = True
                current_time = self._seek_time_at_x(event.position().x())
                self.seek_requested.emit(current_time)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if (
            event is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self._edit_mode == "region"
            and event.position().x() >= self._header_width
        ):
            hit_region = self._region_at_x(event.position().x())
            if hit_region is not None:
                self.region_selected.emit(hit_region.region_id)
                self.region_edit_requested.emit(hit_region.region_id)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            current_time = self._seek_time_at_x(event.position().x())
            if self._edit_mode == "region":
                self._drag_current_time = current_time
                self.update()
            else:
                self.seek_requested.emit(current_time)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging and self._edit_mode == "region":
                anchor = self._drag_anchor_time
                if anchor is not None:
                    current_time = self._seek_time_at_x(event.position().x())
                    self._drag_current_time = current_time
                    start_seconds = min(anchor, current_time)
                    end_seconds = max(anchor, current_time)
                    if (end_seconds - start_seconds) >= 0.02:
                        self.region_span_requested.emit(start_seconds, end_seconds)
            self._dragging = False
            self._drag_anchor_time = None
            self._drag_current_time = None
            self.update()
        super().mouseReleaseEvent(event)

    def _seek_time_at_x(self, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )

    def _paint_region_drag_preview(self, painter: QPainter) -> None:
        if (
            self._edit_mode != "region"
            or not self._dragging
            or self._drag_anchor_time is None
            or self._drag_current_time is None
        ):
            return

        start_seconds = min(self._drag_anchor_time, self._drag_current_time)
        end_seconds = max(self._drag_anchor_time, self._drag_current_time)
        if (end_seconds - start_seconds) < 1e-6:
            return

        start_x = timeline_x_for_time(
            start_seconds,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        end_x = timeline_x_for_time(
            end_seconds,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        left = max(float(self._header_width), min(start_x, end_x))
        right = min(float(self.width()), max(start_x, end_x))
        width = max(0.0, right - left)
        if width <= 0.0:
            return

        preview_rect = QRectF(left, 1.0, width, max(1.0, float(self.height()) - 2.0))
        fill = QColor(self._block.playhead_color_hex)
        fill.setAlpha(46)
        border = QColor(self._block.playhead_color_hex)
        border.setAlpha(140)
        painter.fillRect(preview_rect, fill)
        painter.setPen(QPen(border, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(preview_rect.adjusted(0.5, 0.5, -0.5, -0.5))

    def _region_at_x(self, x: float) -> RegionPresentation | None:
        if x < float(self._header_width):
            return None
        for region in reversed(self.presentation.regions):
            start_x = timeline_x_for_time(
                float(region.start),
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=self._header_width,
            )
            end_x = timeline_x_for_time(
                float(region.end),
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=self._header_width,
            )
            left = min(start_x, end_x)
            right = max(start_x, end_x)
            if right - left <= 0.0:
                continue
            if left <= x <= right:
                return region
        return None


__all__ = ["TimelineEditorModeBar", "TimelineRuler", "TransportBar"]
